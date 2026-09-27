"""A spare Claude account for the bridge's own sessions.

Someone with a second, cheaper subscription wants its 5-hour allowance used
rather than left to expire. The PC's login stays on the main account -- VS Code
and every terminal are never touched. Only the sessions the bridge starts run
on the spare one, through ``CLAUDE_CODE_OAUTH_TOKEN``: Claude Code's documented
per-process login, a one-year token minted by ``claude setup-token``. When
Claude Code reports the spare account's limit hit, the bridge moves its session
to the main login and back once the limit resets.

Files, next to the bridge's database:

* ``claude-spare-token`` (0600) -- the token. Its presence switches this on.
* ``claude-spare.json`` -- the last limit reading, until when the account is
  used up, and which turns ran on it, so /usage does not count them against
  the main account.

Set up once at the PC: ``python -m voice_bridge.spare_account``.
"""

from __future__ import annotations

import json
import os
import re
import sys
import time
from pathlib import Path

TOKEN_FILE = "claude-spare-token"
STATE_FILE = "claude-spare.json"
_TOKEN_RE = re.compile(r"sk-ant-oat[0-9A-Za-z_-]+")
_PIECE_RE = re.compile(r"[0-9A-Za-z_-]+")
# Turns kept for the /usage exclusion: the longest limit window is a week.
_KEEP = 8 * 86400


def state_path(db_dir: Path) -> Path:
    return db_dir / STATE_FILE


def _read(path: Path) -> dict:
    try:
        data = json.loads(path.read_text())
    except (OSError, ValueError):
        return {}
    return data if isinstance(data, dict) else {}


def spans(db_dir: Path) -> list[tuple[str, float, float]]:
    """``(session_id, start, end)`` of every turn that ran on the spare account."""
    return [tuple(s) for s in _read(state_path(db_dir)).get("turns", []) if len(s) == 3]


def status(db_dir: Path) -> dict | None:
    """What /usage shows: ``{until, utilization, resets_at, kind}``, or None
    when no spare account is set up."""
    if not (db_dir / TOKEN_FILE).exists():
        return None
    state = _read(state_path(db_dir))
    return {k: state.get(k) for k in ("until", "utilization", "resets_at", "kind", "windows")}


class SpareAccount:
    """The spare login as the session manager sees it."""

    def __init__(self, db_dir: Path, clock=time.time) -> None:
        self._dir = db_dir
        self._clock = clock

    def token(self) -> str | None:
        try:
            value = (self._dir / TOKEN_FILE).read_text().strip()
        except OSError:
            return None
        return value or None

    def usable(self) -> bool:
        """Set up, and not used up right now."""
        until = _read(state_path(self._dir)).get("until") or 0
        return self.token() is not None and self._clock() >= until

    def record_limit(self, info) -> bool:
        """Store a rate-limit reading from a session on the spare account.
        True when it says the account is used up.

        Extra paid usage ("overage") counts as used up: the point is the
        allowance already paid for, not a new bill."""
        state = _read(state_path(self._dir))
        # Every window's reading ("five_hour", "seven_day") when Claude Code
        # sends them; the top-level utilization is often missing.
        windows = (info.raw or {}).get("unifiedWindows") or {}
        state.update(utilization=info.utilization, resets_at=info.resets_at,
                     kind=info.rate_limit_type, windows=windows, seen=self._clock())
        spent = info.status == "rejected" or info.rate_limit_type == "overage"
        if spent:
            state["until"] = info.resets_at or self._clock() + 3600
        self._write(state)
        return spent

    def mark_spent(self) -> None:
        """Used up, as a failed turn says, when no reading gave the reset
        time: try again in an hour."""
        state = _read(state_path(self._dir))
        if (state.get("until") or 0) <= self._clock():
            state["until"] = self._clock() + 3600
            self._write(state)

    def record_turn(self, session_id: str, start: float, end: float) -> None:
        state = _read(state_path(self._dir))
        keep = [s for s in state.get("turns", []) if len(s) == 3 and s[2] > end - _KEEP]
        state["turns"] = keep + [[session_id, start, end]]
        self._write(state)

    def _write(self, state: dict) -> None:
        path = state_path(self._dir)
        tmp = path.with_name(path.name + ".tmp")
        tmp.write_text(json.dumps(state))
        os.replace(tmp, path)


def capture_token(output: bytes) -> str | None:
    """The token ``claude setup-token`` printed, from its terminal output.

    The terminal wraps the ~100-character token, so the lines after it that
    are nothing but token characters are its continuation."""
    text = re.sub(r"\x1b\[[0-9;?]*[A-Za-z]", "", output.decode(errors="replace"))
    lines = [x.strip() for x in text.replace("\r\n", "\n").replace("\r", "\n").split("\n")]
    token = None
    for i, line in enumerate(lines):
        found = _TOKEN_RE.search(line)
        if not found:
            continue
        token = found.group()
        if found.end() == len(line):
            for rest in lines[i + 1:]:
                if not _PIECE_RE.fullmatch(rest):
                    break
                token += rest
    return token if token and len(token) > 40 else None


def save_token(db_dir: Path, token: str) -> Path:
    db_dir.mkdir(parents=True, exist_ok=True)
    path = db_dir / TOKEN_FILE
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w") as fh:
        fh.write(token)
    os.chmod(path, 0o600)
    return path


def _db_dir() -> Path:
    """Where the running bridge keeps its database: ``DB_PATH`` from the
    environment or the repository's ``.env`` (the service reads the same)."""
    repo = Path(__file__).resolve().parents[2]
    value = os.environ.get("DB_PATH")
    if not value:
        try:
            lines = (repo / ".env").read_text().splitlines()
        except OSError:
            lines = []
        value = next((x.split("=", 1)[1].strip().strip("\"'") for x in lines
                      if x.startswith("DB_PATH=")), "voice-bridge.db")
    path = Path(value).expanduser()
    return (path if path.is_absolute() else repo / path).parent


def main() -> int:  # pragma: no cover - interactive, needs a browser
    import pty

    from claude_agent_sdk import __file__ as sdk_file

    db_dir = _db_dir()
    claude = Path(sdk_file).parent / "_bundled" / "claude"
    print("Sign in with the SPARE account in the browser that opens.\n")
    seen = bytearray()

    def read(fd):
        data = os.read(fd, 4096)
        seen.extend(data)
        return data

    pty.spawn([str(claude), "setup-token"], read)
    token = capture_token(bytes(seen))
    if token is None:
        print("\nNo token found in the output; nothing was saved.")
        return 1
    print(f"\nSaved to {save_token(db_dir, token)} (0600). Restart the bridge.")
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
