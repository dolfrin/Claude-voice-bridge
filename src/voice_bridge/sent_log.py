"""Which Telegram message came from which Claude Code session.

Written by the bridge for what it sends and by the editor's notification hooks
(``~/.claude/notify-*.sh``), which post straight to Telegram with curl and
would otherwise be anonymous. It is what lets a reply -- or a plain message
right after one -- go back to the conversation that spoke, instead of to
whatever project happened to be active last.

One JSON object per line: ``{"m": message_id, "s": session_id or null,
"c": cwd, "t": unix_time}``. Only the tail is ever read.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

_TAIL_BYTES = 256 * 1024
_KEEP_LINES = 2000


def _path(path: Path | None) -> Path:
    return path or Path.home() / ".claude" / ".voice-bridge-sent.jsonl"


def record(message_id: int, session_id: str | None, cwd: str, path: Path | None = None) -> None:
    """Remember that *message_id* belongs to *session_id* (or just *cwd*)."""
    line = json.dumps({"m": message_id, "s": session_id or None, "c": cwd or "", "t": time.time()})
    target = _path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("a") as fh:
        fh.write(line + "\n")


def _tail(path: Path | None) -> list[dict]:
    target = _path(path)
    try:
        with target.open("rb") as fh:
            fh.seek(0, os.SEEK_END)
            size = fh.tell()
            fh.seek(max(0, size - _TAIL_BYTES))
            data = fh.read().decode(errors="replace")
    except OSError:
        return []
    out = []
    for line in data.splitlines():
        try:
            entry = json.loads(line)
        except json.JSONDecodeError:
            continue  # includes a line cut in half by the seek
        if isinstance(entry, dict) and "m" in entry:
            out.append(entry)
    return out


def lookup(message_id: int, path: Path | None = None) -> dict | None:
    """The entry for *message_id*, or None if it was not recorded."""
    for entry in reversed(_tail(path)):
        if entry.get("m") == message_id:
            return entry
    return None


def last(path: Path | None = None) -> dict | None:
    """The most recently sent recorded message."""
    entries = _tail(path)
    return max(entries, key=lambda e: e.get("t") or 0) if entries else None


def prune(path: Path | None = None) -> None:
    """Keep the file small; called at bridge start. Never raises."""
    target = _path(path)
    try:
        lines = target.read_text().splitlines()
        if len(lines) <= _KEEP_LINES:
            return
        tmp = target.with_suffix(".tmp")
        tmp.write_text("\n".join(lines[-_KEEP_LINES:]) + "\n")
        os.replace(tmp, target)
    except OSError:
        pass
