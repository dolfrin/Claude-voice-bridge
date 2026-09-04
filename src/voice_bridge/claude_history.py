"""Read Claude Code's own on-disk session history from ``~/.claude/projects``.

Pure file reading: no Telegram, no SDK, no side effects. The bridge runs as the
same user as the interactive CLI and the VS Code extension, so it sees exactly
the same sessions they do.

``claude --resume <uuid>`` (and ``ClaudeAgentOptions(resume=...)``) only works
from the session's original working directory, so ``cwd`` is read out of the
``.jsonl`` itself rather than decoded from the directory name: the encoding
``/home/x/my-app`` -> ``-home-x-my-app`` is not reversible, a dash may be either
a path separator or part of a name.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

SUMMARY_LEN = 40
EMPTY = "(empty)"

# Claude writes its own machinery into ``user`` entries too: slash commands,
# hook output, caveats. None of that identifies a session in a list, so we look
# for the first real user message instead.
_META_PREFIXES = (
    "<local-command-caveat>",
    "<local-command-stdout>",
    "<command-name>",
    "<command-message>",
    "<command-args>",
    "<user-prompt-submit-hook>",
    "<system-reminder>",
)


@dataclass(frozen=True)
class Session:
    uuid: str
    cwd: str
    mtime: float
    summary: str
    title: str = ""
    last_prompt: str = ""
    live: str = ""  # "" = not open anywhere; else "VSCode" / "CLI"


@dataclass(frozen=True)
class Project:
    dir: Path
    cwd: str
    count: int
    mtime: float

    @property
    def label(self) -> str:
        return Path(self.cwd).name or self.cwd


def _entries(path: Path):
    """JSON records from a .jsonl; broken lines are skipped, not fatal."""
    try:
        with path.open(errors="replace") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    yield json.loads(line)
                except json.JSONDecodeError:
                    continue
    except OSError:
        return


def _blocks_text(content) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return " ".join(
            b.get("text", "")
            for b in content
            if isinstance(b, dict) and b.get("type") == "text"
        )
    return ""


def _user_text(entry: dict) -> str:
    """The user's own text from an entry, or "" if it is not a real message."""
    if entry.get("type") != "user" or entry.get("isMeta"):
        return ""
    text = " ".join(_blocks_text(entry.get("message", {}).get("content")).split())
    if text.startswith(_META_PREFIXES):
        return ""
    return text


def summary(path: Path) -> str:
    """First real user message, clipped — enough to recognise a session."""
    for entry in _entries(path):
        text = _user_text(entry)
        if text:
            return text if len(text) <= SUMMARY_LEN else text[:SUMMARY_LEN] + "…"
    return EMPTY


def title(path: Path) -> str:
    """The session name as Claude itself knows it.

    Precedence: the user's own rename (``custom-title``, set with Ctrl+R in the
    native picker) -> Claude's auto title (``ai-title``) -> first user message.
    Renaming is repeatable, so the LAST record of each kind wins.
    """
    custom = ai = ""
    for entry in _entries(path):
        kind = entry.get("type")
        if kind == "custom-title" and entry.get("customTitle"):
            custom = entry["customTitle"]
        elif kind == "ai-title" and entry.get("aiTitle"):
            ai = entry["aiTitle"]
    return custom or ai or summary(path)


def last_prompt(path: Path) -> str:
    """The last thing the user said — "where we left off"."""
    last = ""
    for entry in _entries(path):
        if entry.get("type") == "last-prompt" and entry.get("lastPrompt"):
            last = entry["lastPrompt"]
    return " ".join(last.split())


def turns(path: Path, limit: int | None = 12) -> list[tuple[str, str]]:
    """The last ``limit`` (role, text) turns of a session, oldest first.

    ``limit=None`` returns the whole conversation, which is what the downloadable
    transcript needs — anything shown in a Telegram message is necessarily a
    fragment.

    Reads the real Claude transcript, so it also contains turns that happened in
    VS Code or the CLI — unlike the bridge's own Markdown mirror, which only
    sees what came through Telegram.
    """
    out: list[tuple[str, str]] = []
    for entry in _entries(path):
        kind = entry.get("type")
        if kind == "user":
            text = _user_text(entry)
            if text:
                out.append(("user", text))
        elif kind == "assistant":
            text = " ".join(
                _blocks_text(entry.get("message", {}).get("content")).split()
            )
            if text:
                out.append(("assistant", text))
    return out if limit is None else out[-limit:]


def count_sessions(project_dir: Path) -> int:
    """How many sessions a project has, so a capped list can say what it hid."""
    try:
        return sum(1 for _ in project_dir.glob("*.jsonl"))
    except OSError:
        return 0


def _alive(pid: int) -> bool:
    """Is the process alive AND actually claude? pids get reused."""
    try:
        cmdline = Path(f"/proc/{pid}/cmdline").read_bytes()
    except OSError:
        return False
    return b"claude" in cmdline


def live_sessions(sessions_dir: Path, is_alive=None) -> dict[str, str]:
    """``{session_uuid: "VSCode"|"CLI"}`` for sessions open in another process.

    Claude Code cannot run one session in two processes: each loads the .jsonl
    state at its own start and then writes without seeing the other, so the last
    messages of one of them are lost. Such a session must be opened as a fork.
    ``~/.claude/sessions/`` is full of dead processes' leftovers, hence the pid
    check.
    """
    alive = is_alive or _alive
    out: dict[str, str] = {}
    if not sessions_dir.is_dir():
        return out
    for path in sessions_dir.glob("*.json"):
        try:
            data = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        sid, pid = data.get("sessionId"), data.get("pid")
        if not sid or not isinstance(pid, int) or not alive(pid):
            continue
        entrypoint = str(data.get("entrypoint", "")).lower()
        out[sid] = "VSCode" if "vscode" in entrypoint else "CLI"
    return out


def _cwd(path: Path) -> str | None:
    for entry in _entries(path):
        cwd = entry.get("cwd")
        if cwd:
            return cwd
    return None


def recent(
    project_dir: Path, limit: int = 8, live: dict[str, str] | None = None
) -> list[Session]:
    """The ``limit`` newest sessions of one project directory, newest first."""
    try:
        files = sorted(
            project_dir.glob("*.jsonl"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
    except OSError:
        return []
    live = live or {}
    out: list[Session] = []
    for path in files[:limit]:
        cwd = _cwd(path)
        if cwd is None:
            continue
        out.append(
            Session(
                uuid=path.stem,
                cwd=cwd,
                mtime=path.stat().st_mtime,
                summary=summary(path),
                title=title(path),
                last_prompt=last_prompt(path),
                live=live.get(path.stem, ""),
            )
        )
    return out


def projects(root: Path) -> list[Project]:
    """Every project directory that has sessions, freshest first."""
    if not root.is_dir():
        return []
    out: list[Project] = []
    for directory in root.iterdir():
        if not directory.is_dir():
            continue
        files = list(directory.glob("*.jsonl"))
        if not files:
            continue
        newest = max(files, key=lambda p: p.stat().st_mtime)
        cwd = _cwd(newest)
        if cwd is None:
            continue
        out.append(Project(directory, cwd, len(files), newest.stat().st_mtime))
    return sorted(out, key=lambda p: p.mtime, reverse=True)


def project_dir_for_cwd(root: Path, cwd: str) -> Path | None:
    """The history directory holding the sessions of a working directory.

    The bridge knows a project by its ``cwd`` and needs the directory, which is
    the opposite direction from :func:`projects`. Encoding the name is unsafe
    (see the module docstring), so we match on the ``cwd`` the files report.
    """
    try:
        wanted = str(Path(cwd).resolve())
    except OSError:
        wanted = cwd
    for project in projects(root):
        try:
            found = str(Path(project.cwd).resolve())
        except OSError:
            found = project.cwd
        if found == wanted:
            return project.dir
    return None


def session_file(root: Path, cwd: str, uuid: str) -> Path | None:
    """The .jsonl of one session, or None if it is not on disk."""
    directory = project_dir_for_cwd(root, cwd)
    if directory is None:
        return None
    path = directory / f"{uuid}.jsonl"
    return path if path.is_file() else None
