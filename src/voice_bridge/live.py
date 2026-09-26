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
import os
import signal
import time
from datetime import datetime
import logging
import re
import uuid
from dataclasses import dataclass
from pathlib import Path

from .i18n import t

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
    "its picker can only be answered in the editor, which they are away from. "
    "They get buttons for a numbered list of options, and for a yes/no question "
    "that ends with ({cue})."
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
        # SDK sessions -- the bridge's own background ones among them -- have
        # had a socket since Claude Code 2.1.2xx, but nobody sees them in an
        # editor: routing a message there would hide it, so they never count.
        if str(data.get("entrypoint") or "").startswith("sdk"):
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
    note = PEER_NOTE.format(cue=t("answer.cue"))
    body = f'<{_TAG} from-name="{name}">\n{_escape(text)}\n</{_TAG}>\n{note}'
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
        return t("live.waiting_in_editor")
    out = []
    for question in questions:
        if not isinstance(question, dict):
            continue
        out.append(f"\U0001F914 **{question.get('question') or t('live.question')}**")
        for index, option in enumerate(question.get("options") or [], 1):
            if not isinstance(option, dict):
                continue
            note = option.get("description") or ""
            out.append(f"   {index}. {option.get('label') or '?'}"
                       + (f" — {note}" if note else ""))
    out.append("   " + t("live.answer_in_editor"))
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
    # Only a list that ENDS the message and follows a question or a colon is
    # a menu: a numbered explanation mid-message got buttons, and a tap sent
    # "1" to a session that had asked nothing.
    lines = [ln for ln in text.strip().splitlines()]
    end = len(lines)
    start = end
    while start > 0 and _OPTION_RE.match(lines[start - 1]):
        start -= 1
    lead = next((ln.strip() for ln in reversed(lines[:start]) if ln.strip()), "")
    if start == end or not lead.rstrip("*_ ").endswith(("?", ":")):
        return []
    found: list[tuple[int, str]] = []
    for line in lines[start:end]:
        match = _OPTION_RE.match(line)
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


# An explicit yes/no cue anywhere in the closing paragraph: „taip“ arba „ne“,
# taip/ne, (yes/no), yes or no.
_YES_NO_CUE_RE = re.compile(
    r"\btaip\b\W{0,3}\s*(?:/|arba|ar)\s*\W{0,3}\bne\b|\byes\b\W{0,3}\s*(?:/|or)\s*\W{0,3}\bno\b",
    re.IGNORECASE,
)
# A closing question that can only be answered yes or no.
_YES_NO_START_RE = re.compile(
    r"^(?:ar|should i|shall i|do you want|want me to|can i|may i|is it ok|ok to)\b",
    re.IGNORECASE,
)


def is_yes_no_question(text: str) -> bool:
    """Does the message end by asking for a yes or a no?

    Only the closing paragraph counts -- a question buried mid-message is not
    what the session is waiting on. Either an explicit cue ("taip arba ne",
    "yes/no") or a last sentence like "Ar daryti?" / "Should I ...?". Strict
    for the same reason as :func:`parse_options`: a miss means typing the
    answer, a false positive puts buttons under something that asked nothing.
    Never raises.
    """
    if not isinstance(text, str) or not text.strip():
        return False
    closing = text.strip().split("\n\n")[-1][-400:]
    if _YES_NO_CUE_RE.search(closing):
        return True
    sentences = re.split(r"(?<=[.!?])\s+", closing.strip())
    last = sentences[-1].strip(" *_") if sentences else ""
    return last.endswith("?") and bool(_YES_NO_START_RE.match(last))


def last_assistant_text(path: Path | None, tail_bytes: int = 256 * 1024) -> str:
    """Text of the session's final assistant message, or "" if it ended otherwise.

    "" when the last assistant entry is a tool call (a permission prompt, an
    editor picker): only a turn that ended in words can be answered from the
    phone. Reads the tail of the transcript only. Never raises.
    """
    if path is None:
        return ""
    try:
        with path.open("rb") as fh:
            fh.seek(0, 2)
            fh.seek(max(0, fh.tell() - tail_bytes))
            lines = fh.read().decode(errors="replace").splitlines()
    except OSError:
        return ""
    for line in reversed(lines):
        try:
            entry = json.loads(line)
        except json.JSONDecodeError:
            continue
        if entry.get("type") != "assistant":
            continue
        content = (entry.get("message") or {}).get("content")
        if isinstance(content, str):
            return content
        if not isinstance(content, list):
            return ""
        if any(isinstance(b, dict) and b.get("type") == "tool_use" for b in content):
            return ""
        texts = [b.get("text", "") for b in content if isinstance(b, dict) and b.get("type") == "text"]
        if any(t.strip() for t in texts):
            return "\n".join(texts)
        # A thinking-only entry: the text is in an earlier line of this reply.
    return ""


def current_activity(path: Path | None, tail_bytes: int = 512 * 1024) -> tuple[str, float] | None:
    """What a busy session is doing right now, and since when (unix time).

    The newest tool call with no result yet is still running -- a test run
    stuck for hours shows up exactly like that. Otherwise the time of the last
    assistant entry, as "thinking". None when nothing can be read.
    Never raises.
    """
    if path is None:
        return None
    try:
        with path.open("rb") as fh:
            fh.seek(0, 2)
            fh.seek(max(0, fh.tell() - tail_bytes))
            lines = fh.read().decode(errors="replace").splitlines()
    except OSError:
        return None
    finished: set[str] = set()
    for line in reversed(lines):
        try:
            entry = json.loads(line)
        except json.JSONDecodeError:
            continue
        content = (entry.get("message") or {}).get("content")
        blocks = content if isinstance(content, list) else []
        try:
            ts = datetime.fromisoformat(str(entry.get("timestamp")).replace("Z", "+00:00")).timestamp()
        except ValueError:
            continue
        if entry.get("type") == "user":
            finished.update(b.get("tool_use_id") for b in blocks
                            if isinstance(b, dict) and b.get("type") == "tool_result")
        elif entry.get("type") == "assistant":
            for block in reversed(blocks):
                if isinstance(block, dict) and block.get("type") == "tool_use":
                    if block.get("id") not in finished:
                        return "🔧 " + _tool_line(block), ts
                    return "🤔", ts
            return "🤔", ts
    return None


def _children_from_proc(pid: int) -> list[int]:
    """Child pids of *pid*, read from /proc (Linux)."""
    out: list[int] = []
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        try:
            stat = (entry / "stat").read_text()
        except OSError:
            continue
        # "pid (comm) state ppid ..." -- comm may contain spaces/parens.
        fields = stat.rsplit(")", 1)[-1].split()
        if len(fields) > 1 and fields[1] == str(pid):
            out.append(int(entry.name))
    return out


def _cmdline(pid: int) -> str:
    try:
        return Path(f"/proc/{pid}/cmdline").read_bytes().replace(b"\0", b" ").decode(errors="replace")
    except OSError:
        return ""


# The Bash tool runs each command as `bash -c source <shell snapshot> ...`;
# MCP servers and the session itself look nothing like it.
_BASH_TOOL_MARK = "/.claude/shell-snapshots/"


def running_commands(session_pid: int, children=_children_from_proc, cmdline=_cmdline) -> list[int]:
    """Pids of the Bash-tool commands a session is running right now."""
    return [c for c in children(session_pid) if _BASH_TOOL_MARK in cmdline(c)]


def stop_commands(session_pid: int, children=_children_from_proc, cmdline=_cmdline,
                  kill=os.kill, sleep=time.sleep) -> int:
    """Stop the session's running Bash-tool commands -- nothing else.

    The session gets "command terminated" back as the tool's result and
    carries on, which is what a step stuck for hours needs. Every descendant
    of each command gets SIGTERM, survivors SIGKILL two seconds later.
    Returns how many commands were stopped. Never raises.
    """
    def tree(pid: int) -> list[int]:
        found = []
        for c in children(pid):
            found += tree(c)
        return found + [pid]

    commands = running_commands(session_pid, children, cmdline)
    targets = []
    for command in commands:
        targets += tree(command)
    for signal_number in (signal.SIGTERM, signal.SIGKILL):
        for pid in targets:
            try:
                kill(pid, signal_number)
            except (ProcessLookupError, PermissionError):
                pass
        if signal_number == signal.SIGTERM and targets:
            sleep(2)
    return len(commands)
