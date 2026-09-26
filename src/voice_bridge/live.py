"""Reach the Claude Code sessions that are already running on this machine.

Every session -- the CLI, the VS Code extension, this bridge -- registers itself
as ``~/.claude/sessions/<pid>.json`` and listens on the unix socket named there
in ``messagingSocketPath``. One JSON line written to that socket is delivered
into the live process as a user turn: it is the same channel ``claude`` uses to
message a peer session. That is what lets a phone join a session already open in
the editor instead of resuming a second copy of it, which Claude Code cannot do
(two processes would each load the .jsonl at their own start and then overwrite
each other's tail).

Nothing comes back over the socket. The session appends its own output to its
``.jsonl`` transcript, so :func:`read_new` tails that file instead.

The socket wire format is internal to Claude Code and carries no version
negotiation, so :func:`list_sessions` and the transcript tail -- the halves
built on documented files -- keep working even if a future release changes the
envelope and :func:`send` starts failing.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import uuid
from dataclasses import dataclass
from pathlib import Path

from .claude_history import EMPTY, _alive, _blocks_text, title

SESSIONS_DIR = Path.home() / ".claude" / "sessions"
SEND_TIMEOUT = 5.0

# Envelope constants, read out of the 2.1.217 bundle (`sendToUdsSocket`).
_TAG = "cross-session-message"
_PROTOCOL = 1
_NAME_RE = re.compile(r"^[A-Za-z0-9_-]{1,80}$")

# Appended to every message we deliver. Claude Code frames an incoming peer
# message with "reply via SendMessage to the from= address", which is wrong
# here twice over: this sender is a human on a phone, and the bridge has no
# socket of its own to be replied to, so the session wastes a turn failing to
# find the peer and then answers as if talking about the request instead of to
# the person. It reads the session's own transcript, so the answer belongs in
# the session. The picker note is here rather than sent once at attach time
# because a session decides how to ask on the turn it is asked, and a message
# sent while a picker is already open only queues behind it.
PEER_NOTE = (
    "[bridge] This is not another Claude: it is your user, typing from Telegram, "
    "and they are reading this session's transcript live. Answer here as you "
    "normally would -- do not reply with SendMessage, and do not treat this as a "
    "peer request. Ask follow-up questions as plain text, not AskUserQuestion: "
    "its picker can only be answered in the editor, which they are away from."
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class LiveSession:
    """One Claude Code process that is up right now and can be messaged."""

    pid: int
    session_id: str
    cwd: str
    socket_path: str
    name: str = ""
    entrypoint: str = ""
    status: str = ""
    started_at: int = 0
    updated_at: int = 0

    @property
    def last_active(self) -> int:
        """When it last did something (ms); start time for older builds."""
        return self.updated_at or self.started_at

    @property
    def surface(self) -> str:
        """Where it is being driven from, for the picker."""
        entrypoint = self.entrypoint.lower()
        if "vscode" in entrypoint:
            return "VSCode"
        if "voice-bridge" in entrypoint:
            return "bridge"
        return "CLI"

    @property
    def label(self) -> str:
        return self.name or self.session_id[:8]


def list_sessions(
    sessions_dir: Path | None = None,
    is_alive=None,
    skip_pid: int | None = None,
) -> list[LiveSession]:
    """Messageable sessions, newest first.

    ``~/.claude/sessions`` keeps the files of dead processes too, hence the pid
    check; entries without a socket are older builds and cannot be joined.
    """
    directory = SESSIONS_DIR if sessions_dir is None else sessions_dir
    alive = is_alive or _alive
    out: list[LiveSession] = []
    if not directory.is_dir():
        return out
    for path in directory.glob("*.json"):
        try:
            data = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        pid, sock = data.get("pid"), data.get("messagingSocketPath")
        sid = data.get("sessionId")
        if not isinstance(pid, int) or not sock or not sid:
            continue
        if pid == skip_pid or not alive(pid):
            continue
        out.append(
            LiveSession(
                pid=pid,
                session_id=str(sid),
                cwd=str(data.get("cwd") or "?"),
                socket_path=str(sock),
                name=str(data.get("name") or ""),
                entrypoint=str(data.get("entrypoint") or ""),
                status=str(data.get("status") or ""),
                started_at=int(data.get("startedAt") or 0),
                updated_at=int(data.get("updatedAt") or 0),
            )
        )
    return sorted(out, key=lambda s: s.started_at, reverse=True)


def find(
    pid: int, sessions_dir: Path | None = None, is_alive=None
) -> LiveSession | None:
    """The live session with this pid, or None once it is gone."""
    for session in list_sessions(sessions_dir, is_alive):
        if session.pid == pid:
            return session
    return None


def envelope(text: str, from_name: str = "telegram") -> dict:
    """The JSON object one line of which is a user turn for a peer session."""
    name = from_name if _NAME_RE.match(from_name) else "telegram"
    body = f'<{_TAG} from-name="{name}">\n{_escape(text)}\n</{_TAG}>\n{PEER_NOTE}'
    return {
        "msgV": _PROTOCOL,
        "msg_id": str(uuid.uuid4()),
        "type": "user",
        "message": {"role": "user", "content": body},
        # "next" queues the turn ahead of anything the session is still
        # holding, which is what a person waiting on a phone expects.
        "priority": "next",
    }


def _escape(text: str) -> str:
    """Neutralise a closing tag inside the body so it cannot end the wrapper."""
    return re.sub(rf"</(?={_TAG}(?:[>\s/]|$))", r"<\\/", text, flags=re.IGNORECASE)


async def send(socket_path: str, text: str, from_name: str = "telegram") -> None:
    """Deliver ``text`` into the live session listening on ``socket_path``.

    Raises ``OSError``/``asyncio.TimeoutError`` if the session is gone or wedged;
    callers report that to the user rather than dropping the turn silently.
    """
    payload = json.dumps(envelope(text, from_name)).encode() + b"\n"

    async def _write() -> None:
        _reader, writer = await asyncio.open_unix_connection(socket_path)
        try:
            writer.write(payload)
            await writer.drain()
        finally:
            writer.close()
            try:
                await writer.wait_closed()
            except (OSError, ConnectionError):
                pass

    await asyncio.wait_for(_write(), SEND_TIMEOUT)


def transcript_of(projects_root: Path, session_id: str) -> Path | None:
    """The .jsonl a live session writes, found by id across all projects.

    The directory name is an unreversible encoding of the cwd, so the file is
    located by globbing rather than by rebuilding the name.
    """
    try:
        return next(projects_root.glob(f"*/{session_id}.jsonl"))
    except (StopIteration, OSError):
        return None


def title_of(projects_root: Path, session_id: str) -> str:
    """The session's own name, the one ``/resume`` and the CLI picker show.

    ``name`` in the registry is derived from the directory ("root-47"), which
    says nothing about what the session is doing; the transcript carries the
    rename, Claude's auto title, or failing both the first user message.
    """
    if path := transcript_of(projects_root, session_id):
        found = title(path)
        # A session that has not been talked to yet summarises as "(empty)",
        # which is worse than the derived name the caller falls back to.
        return "" if found == EMPTY else found
    return ""


def end_of(path: Path | None) -> int:
    """Current size, so a tail starts at "from now on" instead of replaying."""
    if path is None:
        return 0
    try:
        return path.stat().st_size
    except OSError:
        return 0


def read_new(path: Path, offset: int) -> tuple[list[str], int]:
    """Rendered assistant output appended after ``offset``, and the new offset.

    A shrunken file means the session was compacted or rotated; re-reading it
    whole would replay the entire history into Telegram, so we skip to its end.
    """
    try:
        size = path.stat().st_size
    except OSError:
        return [], offset
    if size < offset:
        return [], size
    if size == offset:
        return [], offset
    lines: list[str] = []
    try:
        with path.open(errors="replace") as fh:
            fh.seek(offset)
            for line in fh:
                if not line.endswith("\n"):
                    # Partial write: leave it for the next poll.
                    break
                offset += len(line.encode("utf-8", "replace"))
                stripped = line.strip()
                if not stripped:
                    continue
                try:
                    entry = json.loads(stripped)
                except json.JSONDecodeError:
                    continue
                rendered = render(entry)
                if rendered:
                    lines.append(rendered)
    except OSError:
        return lines, offset
    return lines, offset


def typed_here(path: Path, start: int, end: int) -> bool:
    """Whether a turn typed at the keyboard landed between ``start`` and ``end``.

    Claude Code stamps each user entry with where it came from:
    ``origin.kind == "human"`` is typed in that session's own editor or
    terminal; a Telegram message arrives as ``"peer"``, a finished background
    task as ``"task-notification"``. Only "human" means the person is sitting
    at the screen. Never raises -- a failed read reads as "no"."""
    if end <= start:
        return False
    try:
        with path.open("rb") as fh:
            fh.seek(start)
            chunk = fh.read(end - start)
    except OSError:
        return False
    for raw in chunk.decode("utf-8", "replace").splitlines():
        try:
            entry = json.loads(raw)
        except json.JSONDecodeError:
            continue
        if entry.get("type") != "user":
            continue
        origin = entry.get("origin")
        if isinstance(origin, dict) and origin.get("kind") == "human":
            return True
    return False


def render(entry: dict) -> str:
    """One transcript entry as a Telegram line, or "" if it is not worth one.

    Only assistant entries are surfaced: the user side of that session is either
    typed at the PC (where it is already visible) or sent from here.
    """
    if entry.get("type") != "assistant":
        return ""
    content = entry.get("message", {}).get("content")
    if not isinstance(content, list):
        text = _blocks_text(content).strip()
        return text

    parts = []
    for block in content:
        if not isinstance(block, dict):
            continue
        kind = block.get("type")
        if kind == "text" and block.get("text", "").strip():
            parts.append(block["text"].strip())
        elif kind == "thinking":
            reasoning = " ".join(str(block.get("thinking") or "").split())
            if reasoning:
                if len(reasoning) > THINKING_LEN:
                    reasoning = reasoning[: THINKING_LEN - 1] + "…"
                parts.append(f"{THINKING_MARK} *{reasoning}*")
        elif kind == "tool_use":
            if block.get("name") == "AskUserQuestion":
                parts.append(_question_lines(block.get("input")))
            else:
                parts.append(f"{TOOL_MARK} {_tool_line(block)}")
    return "\n".join(parts)


def spoken_of(body: str) -> str:
    """The part of a rendered ``/live`` body worth reading aloud.

    Drops the thinking and tool-activity lines :func:`render` prefixes with
    their markers: a busy session emits a tool line every second or two, and
    hearing "Bash: ls" read out is noise. The assistant's own text -- and a
    pending question, which is exactly what you want to hear when away from
    the keyboard -- is kept.
    """
    kept = [
        line
        for line in body.splitlines()
        if line.strip()
        and not line.lstrip().startswith((THINKING_MARK, TOOL_MARK))
    ]
    return "\n".join(kept).strip()


def _question_lines(inputs) -> str:
    """A pending AskUserQuestion, spelled out with its options.

    The choice itself cannot be made from here: the tool is answered by the
    picker in the editor that raised it, and a message sent meanwhile is only
    queued for after. Showing the options still beats a bare "AskUserQuestion",
    because it says what that tab is stuck on.
    """
    questions = (inputs or {}).get("questions") if isinstance(inputs, dict) else None
    if not isinstance(questions, list) or not questions:
        return "\U0001F914 Waiting for an answer in the editor."
    out = []
    for question in questions:
        if not isinstance(question, dict):
            continue
        out.append(f"\U0001F914 **{question.get('question') or 'Question'}**")
        for index, option in enumerate(question.get("options") or [], 1):
            if not isinstance(option, dict):
                continue
            note = option.get("description") or ""
            out.append(f"   {index}. {option.get('label') or '?'}"
                       + (f" — {note}" if note else ""))
    out.append("   *answer it in the editor; this tab is blocked until then*")
    return "\n".join(out)


# What each tool is actually doing lives under a different input key per tool,
# and a bare "Bash" told the reader nothing.
_TOOL_DETAIL = {
    "Bash": ("description", "command"),
    "BashOutput": ("bash_id",),
    "Read": ("file_path",),
    "Write": ("file_path",),
    "Edit": ("file_path",),
    "NotebookEdit": ("notebook_path",),
    "Glob": ("pattern",),
    "Grep": ("pattern",),
    "Skill": ("skill",),
    "Task": ("description",),
    "Agent": ("description",),
    "WebFetch": ("url",),
    "WebSearch": ("query",),
    "TodoWrite": (),
}
_DETAIL_LEN = 90
# Reasoning is shown, but a long block would push the answer off the screen.
THINKING_LEN = 400
# Line markers render() stamps on the two kinds of output that are worth
# SEEING but not HEARING; spoken_of strips them back out.
THINKING_MARK = "\U0001F4AD"
TOOL_MARK = "\U0001F527"


def _tool_line(block: dict) -> str:
    name = str(block.get("name") or "?")
    inputs = block.get("input")
    if not isinstance(inputs, dict):
        return name
    keys = _TOOL_DETAIL.get(name)
    if keys is None:
        # Unknown tool: the first short string in its input is the best guess
        # at what identifies the call.
        keys = tuple(k for k, v in inputs.items() if isinstance(v, str))[:1]
    for key in keys:
        value = inputs.get(key)
        if not isinstance(value, str) or not value.strip():
            continue
        detail = " ".join(value.split())
        if key.endswith("path"):
            detail = detail.rsplit("/", 1)[-1] or detail
        if len(detail) > _DETAIL_LEN:
            detail = detail[: _DETAIL_LEN - 1] + "…"
        return f"{name} {detail}"
    return name


# A numbered option in a PLAIN-TEXT question: "1) Ship it", "2. Wait", "3 - Stop".
# Two digits max, so a list item can never be confused with a year or a count.
_OPTION_RE = re.compile(r"^\s*(\d{1,2})\s*[).\]:-]\s+(\S.*?)\s*$")
# A question answerable this way needs at least this many options; one "1)" is
# far more likely to be a list item in ordinary prose than a choice.
_MIN_OPTIONS = 2
# Longer than this and it is prose that happens to start with a digit.
_OPTION_MAX = 120


def parse_options(text: str) -> list[str]:
    """The numbered options offered in a plain-text question, in order, or [].

    ``AskUserQuestion``'s own picker can only be answered in the editor that
    raised it, so the bridge asks live sessions to put their questions in plain
    text instead (see :data:`PEER_NOTE`). Models write those as a numbered list,
    which is enough to offer real buttons: this pulls the labels out so the
    caller can send the chosen NUMBER back down the socket, exactly as the user
    would have typed it.

    Deliberately strict — a miss just means no buttons and the user answers by
    typing, while a false positive puts buttons on something that is not a
    question. Requires at least :data:`_MIN_OPTIONS` options numbered 1..n with
    no gaps, each short enough to be a label. Never raises.
    """
    if not isinstance(text, str) or not text.strip():
        return []
    found: list[tuple[int, str]] = []
    for line in text.splitlines():
        match = _OPTION_RE.match(line)
        if match is None:
            continue
        label = match.group(2)
        if len(label) > _OPTION_MAX:
            return []  # prose, not a choice list
        found.append((int(match.group(1)), label))
    if len(found) < _MIN_OPTIONS:
        return []
    # Must be 1..n in order: a stray "2)" mid-paragraph is not a menu.
    if [n for n, _ in found] != list(range(1, len(found) + 1)):
        return []
    return [label for _, label in found]
