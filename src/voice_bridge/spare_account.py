"""Spare Claude accounts: their allowance is used before it expires.

Someone holding several subscriptions wants each one's allowance used rather
than left to reset unused. The PC's login stays on the main account -- VS Code
and every terminal are never touched. Work runs on a spare account only in a
process that gets its token through ``CLAUDE_CODE_OAUTH_TOKEN``: Claude Code's
documented per-process login, a one-year token minted by ``claude setup-token``.
Two such kinds of process:

* the bridge's own sessions (``SessionManager``), and
* a *worker*: a Claude Code session in a project, run in the background by the
  session you are talking to (``python -m voice_bridge.spare_account run``), so
  a big job is done on a spare account while the conversation stays where it is.

Which account: the usable one whose allowance expires first -- an allowance
that resets tomorrow is lost tomorrow, one that resets next week is not. When
Claude Code reports an account's limit reached (paid extra usage counts as
reached: the point is what is already paid for), the work moves to the next
spare account, and to the main login when none is left.

Files, next to the bridge's database:

* ``claude-spare/<name>.token`` (0600) -- one per account.
* ``claude-spare.json`` -- each account's last limit reading and until when it
  is used up, and which turns ran on a spare account, so /usage does not count
  them against the main one.

Add an account once at the PC: ``python -m voice_bridge.spare_account add NAME``.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

TOKEN_DIR = "claude-spare"
STATE_FILE = "claude-spare.json"
_LEGACY_TOKEN = "claude-spare-token"  # the single-account layout of d46b368
_TOKEN_RE = re.compile(r"sk-ant-oat[0-9A-Za-z_-]+")
_PIECE_RE = re.compile(r"[0-9A-Za-z_-]+")
_NAME_RE = re.compile(r"[A-Za-z0-9._-]{1,40}")
# Turns kept for the /usage exclusion: the longest limit window is a week.
_KEEP = 8 * 86400
NO_ACCOUNT = 3  # worker exit status: no spare account has allowance now


def state_path(db_dir: Path) -> Path:
    return db_dir / STATE_FILE


def _read(path: Path) -> dict:
    try:
        data = json.loads(path.read_text())
    except (OSError, ValueError):
        return {}
    return data if isinstance(data, dict) else {}


def spans(db_dir: Path) -> list[tuple[str, float, float]]:
    """``(session_id, start, end)`` of every turn that ran on a spare account."""
    return [tuple(s) for s in _read(state_path(db_dir)).get("turns", []) if len(s) == 3]


def _expires(state: dict) -> float:
    """When this account's next allowance would be lost unused: the soonest
    reset among its windows that still have room. Unknown sorts last."""
    windows = state.get("windows") or {}
    resets = [
        float(w["resetsAt"]) for w in windows.values()
        if isinstance(w, dict) and w.get("resetsAt") and float(w.get("utilization") or 0) < 1
    ]
    return min(resets) if resets else float("inf")


class Spares:
    """Every spare account, as the bridge and the worker see them."""

    def __init__(self, db_dir: Path, clock=time.time) -> None:
        self._dir = db_dir
        self._clock = clock
        self._migrate()

    def _migrate(self) -> None:
        """The first version kept one token and flat state; it becomes the
        account named "spare"."""
        legacy = self._dir / _LEGACY_TOKEN
        if not legacy.exists():
            return
        try:
            save_token(self._dir, "spare", legacy.read_text().strip())
            legacy.unlink()
        except OSError:
            return
        state = _read(state_path(self._dir))
        flat = {k: state.pop(k) for k in list(state) if k not in ("accounts", "turns")}
        if flat:
            state.setdefault("accounts", {})["spare"] = flat
            self._write(state)

    def names(self) -> list[str]:
        return sorted(p.stem for p in (self._dir / TOKEN_DIR).glob("*.token"))

    def token(self, name: str) -> str | None:
        try:
            value = (self._dir / TOKEN_DIR / f"{name}.token").read_text().strip()
        except OSError:
            return None
        return value or None

    def account(self, name: str) -> dict:
        return (_read(state_path(self._dir)).get("accounts") or {}).get(name) or {}

    def usable(self, name: str) -> bool:
        """Set up, and not used up right now."""
        return self.token(name) is not None and self._clock() >= (self.account(name).get("until") or 0)

    def pick(self) -> str | None:
        """The usable account whose allowance expires first, or None."""
        usable = [n for n in self.names() if self.usable(n)]
        return min(usable, key=lambda n: (_expires(self.account(n)), n), default=None)

    def record_limit(self, name: str, info) -> bool:
        """Store a rate-limit reading from work on *name*. True when it says
        the account is used up."""
        # Claude Code sends every window ("five_hour", "seven_day") under
        # unifiedWindows; the top-level utilization is often missing.
        windows = (info.raw or {}).get("unifiedWindows") or {}
        spent = info.status == "rejected" or info.rate_limit_type == "overage"

        def change(acc: dict) -> None:
            acc.update(utilization=info.utilization, resets_at=info.resets_at,
                       kind=info.rate_limit_type, windows=windows, seen=self._clock())
            if spent:
                acc["until"] = info.resets_at or self._clock() + 3600

        self._change(name, change)
        return spent

    def mark_spent(self, name: str) -> None:
        """Used up, as a failed turn says, with no reset time given: try again
        in an hour."""
        def change(acc: dict) -> None:
            if (acc.get("until") or 0) <= self._clock():
                acc["until"] = self._clock() + 3600

        self._change(name, change)

    def record_turn(self, session_id: str, start: float, end: float) -> None:
        state = _read(state_path(self._dir))
        keep = [s for s in state.get("turns", []) if len(s) == 3 and s[2] > end - _KEEP]
        state["turns"] = keep + [[session_id, start, end]]
        self._write(state)

    def _change(self, name: str, change) -> None:
        state = _read(state_path(self._dir))
        change(state.setdefault("accounts", {}).setdefault(name, {}))
        self._write(state)

    def _write(self, state: dict) -> None:
        path = state_path(self._dir)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(path.name + ".tmp")
        tmp.write_text(json.dumps(state))
        os.replace(tmp, path)


def status(db_dir: Path) -> list[tuple[str, dict]]:
    """``(name, reading)`` of every spare account, for /usage."""
    spares = Spares(db_dir)
    return [(n, spares.account(n)) for n in spares.names()]


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


def save_token(db_dir: Path, name: str, token: str) -> Path:
    if not _NAME_RE.fullmatch(name):
        raise ValueError(f"account name {name!r}: letters, digits, . _ - only")
    folder = db_dir / TOKEN_DIR
    folder.mkdir(parents=True, exist_ok=True)
    os.chmod(folder, 0o700)
    path = folder / f"{name}.token"
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w") as fh:
        fh.write(token)
    os.chmod(path, 0o600)
    return path


def _claude() -> str:
    from claude_agent_sdk import __file__ as sdk_file

    return str(Path(sdk_file).parent / "_bundled" / "claude")


def worker_env(token: str, base: dict | None = None) -> dict:
    """The environment of a worker: the caller's, without the variables that
    tie a process to the Claude Code session it runs in (its id, messaging
    socket, entrypoint), plus the spare account's token. The "sdk" entrypoint
    keeps the bridge from listing it as a conversation to write to."""
    env = {k: v for k, v in (os.environ if base is None else base).items()
           if not (k.startswith("CLAUDE") or k.startswith("ANTHROPIC")) or k == "CLAUDE_CONFIG_DIR"}
    env.update(CLAUDE_CODE_OAUTH_TOKEN=token, CLAUDE_CODE_ENTRYPOINT="sdk-cli")
    return env


def run(task: str, cwd: str, db_dir: Path, resume: str | None = None, model: str | None = None,
        spawn=subprocess.Popen, clock=time.time, out=sys.stdout) -> int:
    """Do *task* in *cwd* as a Claude Code session on a spare account.

    Prints the session's final answer, then one ``[worker]`` line naming the
    account and the session id (give it back as *resume* to continue). When an
    account runs out mid-task, the same session continues on the next one.
    Exit status 0 done, 1 the session ended in an error, 3 no spare account has
    allowance now (the line says until when)."""
    from claude_agent_sdk.types import RateLimitInfo

    spares = Spares(db_dir, clock=clock)
    prompt = task
    while True:
        name = spares.pick()
        if name is None:
            left = [f"{n} until {time.strftime('%m-%d %H:%M', time.localtime(spares.account(n).get('until') or 0))}"
                    for n in spares.names()]
            print(f"[worker] no spare account has allowance now ({', '.join(left) or 'none set up'});"
                  " do it on the main account", file=out)
            return NO_ACCOUNT
        cmd = [_claude(), "-p", prompt, "--output-format", "stream-json", "--verbose",
               "--permission-mode", "acceptEdits", "--settings", json.dumps({"disableAllHooks": True})]
        if resume:
            cmd += ["--resume", resume]
        if model:
            cmd += ["--model", model]
        started = clock()
        spent, answer, failed, session = False, "", False, resume
        proc = spawn(cmd, cwd=cwd, env=worker_env(spares.token(name)),
                     stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True)
        for line in proc.stdout:
            try:
                msg = json.loads(line)
            except ValueError:
                continue
            kind = msg.get("type")
            if kind == "rate_limit_event":
                raw = msg.get("rate_limit_info") or {}
                info = RateLimitInfo(status=raw.get("status"), resets_at=raw.get("resetsAt"),
                                     rate_limit_type=raw.get("rateLimitType"),
                                     utilization=raw.get("utilization"), raw=raw)
                spent = spares.record_limit(name, info) or spent
            elif kind == "assistant" and msg.get("error") == "rate_limit":
                spent = True
            elif kind == "result":
                session = msg.get("session_id") or session
                answer = str(msg.get("result") or "")
                failed = bool(msg.get("is_error"))
        proc.wait()
        if session:
            spares.record_turn(session, started, clock())
        if spent and failed:
            spares.mark_spent(name)
            resume, prompt = session, "Continue the task from where you stopped."
            continue
        print(answer, file=out)
        print(f"[worker] account={name} session={session}", file=out)
        return 1 if failed else 0


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


def _add(name: str) -> int:  # pragma: no cover - interactive, needs a browser
    import pty

    print(f"Sign in with the account to add as {name!r} in the browser that opens.\n")
    seen = bytearray()

    def read(fd):
        data = os.read(fd, 4096)
        seen.extend(data)
        return data

    pty.spawn([_claude(), "setup-token"], read)
    token = capture_token(bytes(seen))
    if token is None:
        print("\nNo token found in the output; nothing was saved.")
        return 1
    print(f"\nSaved to {save_token(_db_dir(), name, token)} (0600).")
    return 0


def _status() -> int:
    spares = Spares(_db_dir())
    if not spares.names():
        print("No spare account. Add one: python -m voice_bridge.spare_account add NAME")
    for name in spares.names():
        acc = spares.account(name)
        state = "usable" if spares.usable(name) else \
            f"used up until {time.strftime('%m-%d %H:%M', time.localtime(acc.get('until') or 0))}"
        windows = ", ".join(
            f"{key} {float(w.get('utilization') or 0) * 100:.0f}% resets "
            f"{time.strftime('%m-%d %H:%M', time.localtime(w['resetsAt']))}"
            for key, w in (acc.get("windows") or {}).items() if isinstance(w, dict) and w.get("resetsAt")
        )
        star = " <- next" if name == spares.pick() else ""
        print(f"{name}: {state}{star}" + (f" ({windows})" if windows else " (no reading yet)"))
    return 0


def main(argv: list[str]) -> int:  # pragma: no cover - thin CLI over the functions above
    import argparse

    parser = argparse.ArgumentParser(prog="python -m voice_bridge.spare_account")
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("add", help="add an account (runs claude setup-token)").add_argument("name")
    sub.add_parser("status", help="each account's limits and which one is next")
    work = sub.add_parser("run", help="do a task in a project on a spare account")
    work.add_argument("task")
    work.add_argument("--cwd", default=os.getcwd())
    work.add_argument("--resume")
    work.add_argument("--model")
    args = parser.parse_args(argv)
    if args.cmd == "add":
        return _add(args.name)
    if args.cmd == "status":
        return _status()
    return run(args.task, args.cwd, _db_dir(), resume=args.resume, model=args.model)


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main(sys.argv[1:]))
