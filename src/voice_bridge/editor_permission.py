"""Answer a VS Code session's permission prompt from Telegram.

Run by the ``PermissionRequest`` hook (``hooks/editor-permission.sh``). The
bridge side is :meth:`TelegramIO._watch_permissions`; the two talk through
files in ``~/.claude/.voice-bridge-perm`` (see README).

Why waiting here is safe: in the VS Code extension (Claude Code 2.1.283) this
hook runs *alongside* the editor's own dialog, not before it. Whichever answers
first wins, and a hook answer makes Claude Code withdraw the dialog. So the
editor stays usable while the phone decides, and a tap from Telegram closes
the dialog there.

Only VS Code sessions are engaged. A terminal session may run the hook before
drawing its dialog, and a long wait there would hold the prompt back; those
keep the plain notification. So do AskUserQuestion and ExitPlanMode, whose
answer is more than yes/no.

The session registry (``~/.claude/sessions/<pid>.json``) shows ``waiting`` while
a prompt is open. Once it has been seen and then leaves that state, the prompt
was answered in the editor: the request is withdrawn with reason ``editor`` so
the Telegram message says so instead of offering buttons nothing reads.

Exit status: 0 -- answered (decision on stdout) or gave up (nothing printed, the
editor decides); 3 -- not engaged, the caller sends the plain notification.
"""

from __future__ import annotations

import json
import os
import re
import sys
import time
import uuid
from pathlib import Path

NOT_ENGAGED = 3
# The bridge rewrites its heartbeat every second; older than this = it is down.
_ALIVE_FOR = 15
_POLL = 0.5
# Stay under the hook timeout registered in settings.json (3600 s), so the hook
# always withdraws its own request instead of being killed mid-wait.
_DEADLINE = 3540
_KEEP_SPOOL = 86400
_SKIP_TOOLS = {"AskUserQuestion", "ExitPlanMode"}
_SAFE_ID_RE = re.compile(r"[A-Za-z0-9._-]{1,80}")


def _session_entry(home: Path, session_id: str) -> dict | None:
    """The registry entry Claude Code keeps for *session_id*, or None."""
    for path in (home / ".claude" / "sessions").glob("*.json"):
        try:
            entry = json.loads(path.read_text())
        except (OSError, ValueError):
            continue
        if isinstance(entry, dict) and entry.get("sessionId") == session_id:
            return entry
    return None


def _bridge_alive(home: Path, now: float) -> bool:
    try:
        beat = int((home / ".claude" / ".voice-bridge-alive").read_text().strip())
    except (OSError, ValueError):
        return False
    return now - beat <= _ALIVE_FOR


def _project(cwd: str) -> str:
    """Name of the git repository holding *cwd* (its folder name otherwise)."""
    path = os.path.abspath(cwd) if cwd else ""
    probe = path
    while probe and probe != "/":
        if os.path.isdir(os.path.join(probe, ".git")):
            return os.path.basename(probe)
        probe = os.path.dirname(probe)
    return os.path.basename(path) or "IDE"


def _detail(tool_input: dict) -> str:
    """What the tool is about to do, as the editor dialog would show it."""
    parts = []
    for key in ("description", "command", "file_path", "url", "prompt"):
        value = tool_input.get(key)
        if isinstance(value, str) and value.strip():
            parts.append(value.strip())
    return "\n".join(parts[:2])


def engaged(hook: dict, home: Path, now: float) -> bool:
    """Should this prompt go to Telegram with buttons?"""
    if hook.get("hook_event_name") != "PermissionRequest":
        return False
    if hook.get("tool_name") in _SKIP_TOOLS:
        return False
    # The Telegram channel is lent to Codex while this marker exists.
    if (home / ".claude" / ".telegram-bridge-disabled").exists():
        return False
    entry = _session_entry(home, str(hook.get("session_id") or ""))
    if entry is None or entry.get("entrypoint") != "claude-vscode":
        return False
    return _bridge_alive(home, now)


def _sweep(spool: Path, now: float) -> None:
    """Drop leftovers of hooks that were killed before they could clean up."""
    for path in spool.glob("*"):
        try:
            if now - path.stat().st_mtime > _KEEP_SPOOL:
                path.unlink()
        except OSError:
            pass


def _decision(answer: str) -> dict:
    if answer == "allow":
        decision = {"behavior": "allow"}
    else:
        decision = {"behavior": "deny", "message": "The user denied this from Telegram."}
    return {"hookSpecificOutput": {"hookEventName": "PermissionRequest", "decision": decision}}


def ask(hook: dict, home: Path, *, clock=time.time, sleep=time.sleep,
        ident: str | None = None) -> dict | None:
    """Put the prompt on the phone and wait. The decision, or None to let the
    editor decide. Always removes its request file before returning."""
    spool = home / ".claude" / ".voice-bridge-perm"
    spool.mkdir(parents=True, exist_ok=True)
    _sweep(spool, clock())
    ident = ident or uuid.uuid4().hex
    assert _SAFE_ID_RE.fullmatch(ident)
    request, answer, gone = (spool / f"{ident}.req.json", spool / f"{ident}.ans",
                             spool / f"{ident}.gone")
    session_id = str(hook.get("session_id") or "")
    cwd = str(hook.get("cwd") or "")
    tool_input = hook.get("tool_input") if isinstance(hook.get("tool_input"), dict) else {}
    body = {
        "project": _project(cwd),
        "tool": str(hook.get("tool_name") or "?"),
        "detail": _detail(tool_input),
        "cwd": cwd,
        "session_id": session_id,
        # notify-notification.sh greps for this exact spelling to stay quiet
        # while the buttons are out.
        "session": session_id,
    }
    staged = spool / f"{ident}.req.json.tmp"
    staged.write_text(json.dumps(body, ensure_ascii=False))
    os.replace(staged, request)

    reason = "timeout"
    seen_waiting = False
    deadline = clock() + _DEADLINE
    try:
        while clock() < deadline:
            if answer.exists():
                return _decision(answer.read_text().strip())
            if not _bridge_alive(home, clock()):
                reason = "bridge"
                break
            entry = _session_entry(home, session_id)
            if entry is None:
                reason = "closed"
                break
            # ponytail: one status for the whole session -- with two prompts
            # open at once, answering one in the editor is noticed only when
            # both are settled. Match on the tool_use_id in the transcript if
            # parallel prompts become common.
            if entry.get("status") == "waiting":
                seen_waiting = True
            elif seen_waiting:
                reason = "editor"
                break
            sleep(_POLL)
        # A tap that landed while the loop was deciding to give up still counts.
        if answer.exists():
            return _decision(answer.read_text().strip())
        return None
    finally:
        answered = answer.exists()
        if not answered:
            try:
                gone.write_text(reason)
            except OSError:
                pass
        for path in (request, answer):
            try:
                path.unlink()
            except OSError:
                pass


def main() -> int:
    try:
        hook = json.load(sys.stdin)
    except ValueError:
        return NOT_ENGAGED
    home = Path.home()
    if not isinstance(hook, dict) or not engaged(hook, home, time.time()):
        return NOT_ENGAGED
    decision = ask(hook, home)
    if decision is not None:
        print(json.dumps(decision))
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
