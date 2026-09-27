"""The PermissionRequest hook that puts VS Code prompts on Telegram.

Everything runs against a private fake home: a session registry, the bridge
heartbeat and the spool directory, never the real ~/.claude.
"""

import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from voice_bridge import editor_permission as ep
from voice_bridge.telegram_io import TelegramIO, permission_hook_registered

from test_telegram_io import FakeControls, make_cfg

REPO = Path(__file__).resolve().parents[1]
SID = "11111111-2222-3333-4444-555555555555"


def make_home(tmp_path, *, entrypoint="claude-vscode", status="busy", alive=True):
    home = tmp_path / "home"
    (home / ".claude" / "sessions").mkdir(parents=True)
    set_status(home, status, entrypoint)
    if alive:
        beat(home)
    return home


def set_status(home, status, entrypoint="claude-vscode"):
    (home / ".claude" / "sessions" / "4242.json").write_text(json.dumps(
        {"pid": 4242, "sessionId": SID, "entrypoint": entrypoint, "status": status}))


def beat(home, when=None):
    (home / ".claude" / ".voice-bridge-alive").write_text(str(int(when or time.time())))


def hook(tool="Bash", event="PermissionRequest"):
    return {"hook_event_name": event, "session_id": SID, "cwd": str(REPO),
            "tool_name": tool, "tool_input": {"command": "ls -la", "description": "List"}}


def spool(home):
    return home / ".claude" / ".voice-bridge-perm"


# --- when to engage ----------------------------------------------------------

def test_engages_a_vscode_session_while_the_bridge_runs(tmp_path):
    assert ep.engaged(hook(), make_home(tmp_path), time.time())


@pytest.mark.parametrize("change", ["terminal", "bridge_down", "codex", "question", "plan", "event"])
def test_everything_else_keeps_the_plain_notification(change, tmp_path):
    home = make_home(tmp_path, entrypoint="cli" if change == "terminal" else "claude-vscode",
                     alive=change != "bridge_down")
    if change == "codex":
        (home / ".claude" / ".telegram-bridge-disabled").write_text("")
    data = hook(tool={"question": "AskUserQuestion", "plan": "ExitPlanMode"}.get(change, "Bash"),
                event="PreToolUse" if change == "event" else "PermissionRequest")
    assert not ep.engaged(data, home, time.time())


def test_a_stale_heartbeat_means_the_bridge_is_down(tmp_path):
    home = make_home(tmp_path)
    beat(home, time.time() - 60)
    assert not ep.engaged(hook(), home, time.time())


# --- waiting for the answer ----------------------------------------------------

def run_ask(home, on_poll):
    """ask() with a fake clock; *on_poll(n)* runs at each poll."""
    clock = {"t": time.time(), "n": 0}

    def sleep(_):
        clock["n"] += 1
        clock["t"] += ep._POLL
        beat(home, clock["t"])
        on_poll(clock["n"])

    return ep.ask(hook(), home, clock=lambda: clock["t"], sleep=sleep, ident="abc")


def test_a_tap_on_allow_becomes_the_decision(tmp_path):
    home = make_home(tmp_path, status="waiting")
    seen = {}

    def on_poll(n):
        seen["req"] = json.loads((spool(home) / "abc.req.json").read_text())
        (spool(home) / "abc.ans").write_text("allow")

    decision = run_ask(home, on_poll)

    assert decision == {"hookSpecificOutput": {"hookEventName": "PermissionRequest",
                                               "decision": {"behavior": "allow"}}}
    assert seen["req"]["project"] == "claude-voice-bridge"
    assert seen["req"]["detail"] == "List\nls -la"
    assert list(spool(home).iterdir()) == []            # nothing left behind


def test_a_tap_on_deny_becomes_a_denial(tmp_path):
    home = make_home(tmp_path, status="waiting")
    decision = run_ask(home, lambda n: (spool(home) / "abc.ans").write_text("deny"))
    assert decision["hookSpecificOutput"]["decision"]["behavior"] == "deny"


def test_the_request_names_the_session_as_the_notification_hook_greps_it(tmp_path):
    # notify-notification.sh stays quiet while buttons are out by grepping for
    # this exact spelling; json.dumps with other separators would break it.
    home = make_home(tmp_path, status="waiting")
    seen = {}

    def on_poll(n):
        seen["raw"] = (spool(home) / "abc.req.json").read_text()
        (spool(home) / "abc.ans").write_text("allow")

    run_ask(home, on_poll)
    assert f'"session": "{SID}"' in seen["raw"]


def test_an_answer_in_the_editor_withdraws_the_request(tmp_path):
    # The registry shows "waiting" while the dialog is open; leaving it means
    # the user answered in VS Code, so the phone must stop offering buttons.
    home = make_home(tmp_path, status="waiting")

    decision = run_ask(home, lambda n: set_status(home, "busy") if n == 3 else None)

    assert decision is None
    assert not (spool(home) / "abc.req.json").exists()
    assert (spool(home) / "abc.gone").read_text() == "editor"


def test_busy_before_the_dialog_opens_is_not_an_answer(tmp_path):
    # The hook can start before the registry turns "waiting".
    home = make_home(tmp_path, status="busy")

    def on_poll(n):
        if n == 2:
            set_status(home, "waiting")
        if n == 4:
            (spool(home) / "abc.ans").write_text("allow")

    assert run_ask(home, on_poll)["hookSpecificOutput"]["decision"]["behavior"] == "allow"


def test_the_bridge_going_down_hands_the_prompt_back(tmp_path):
    home = make_home(tmp_path, status="waiting")
    clock = {"t": time.time()}

    def sleep(_):
        clock["t"] += 30                                    # heartbeat goes stale

    assert ep.ask(hook(), home, clock=lambda: clock["t"], sleep=sleep, ident="abc") is None
    assert (spool(home) / "abc.gone").read_text() == "bridge"
    assert not (spool(home) / "abc.req.json").exists()


def test_the_hook_gives_up_before_claude_code_kills_it(tmp_path):
    home = make_home(tmp_path, status="waiting")
    clock = {"t": time.time()}

    def sleep(_):
        clock["t"] += 600
        beat(home, clock["t"])

    assert ep.ask(hook(), home, clock=lambda: clock["t"], sleep=sleep, ident="abc") is None
    assert (spool(home) / "abc.gone").read_text() == "timeout"
    assert clock["t"] - time.time() < 3600


# --- the real script, end to end -----------------------------------------------

def run_script(home, data, on_request=None):
    env = {**os.environ, "HOME": str(home)}
    notify = home / ".claude" / "notify-question.sh"
    notify.write_text('#!/bin/bash\ncat > "$HOME/notified.json"\n')
    notify.chmod(0o755)
    proc = subprocess.Popen(["bash", str(REPO / "hooks" / "editor-permission.sh")], env=env,
                            stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True)
    if on_request:
        def answer():
            for _ in range(100):
                requests = list(spool(home).glob("*.req.json")) if spool(home).exists() else []
                if requests:
                    on_request(requests[0])
                    return
                time.sleep(0.1)
        threading.Thread(target=answer, daemon=True).start()
    out, _ = proc.communicate(json.dumps(data), timeout=30)
    return proc.returncode, out


def test_script_answers_from_telegram(tmp_path):
    home = make_home(tmp_path, status="waiting")
    code, out = run_script(home, hook(),
                           lambda req: req.with_name(req.name.replace(".req.json", ".ans"))
                           .write_text("allow"))
    assert code == 0
    assert json.loads(out)["hookSpecificOutput"]["decision"] == {"behavior": "allow"}
    assert not (home / "notified.json").exists()        # no second, button-less message


def test_script_falls_back_to_the_plain_notification(tmp_path):
    home = make_home(tmp_path, entrypoint="cli")
    code, out = run_script(home, hook())
    assert code == 0 and out == ""                      # the editor decides
    assert json.loads((home / "notified.json").read_text())["tool_name"] == "Bash"


# --- bridge side ------------------------------------------------------------------

@pytest.mark.asyncio
async def test_buttons_say_when_the_editor_answered_first(tmp_path):
    io = TelegramIO(make_cfg(), AsyncMock(), FakeControls())
    directory = io.perm_dir()
    directory.mkdir(parents=True)
    (directory / "abc.gone").write_text("editor")
    message = AsyncMock()
    io._perm_pending["abc"] = message

    await io._expire_permissions(directory)

    assert "Jau atsakyta editoriuje" in message.edit_text.await_args.args[0]
    assert not (directory / "abc.gone").exists()


def test_the_bridge_notices_a_missing_hook(tmp_path):
    settings = tmp_path / "settings.json"
    assert not permission_hook_registered(settings)
    settings.write_text(json.dumps({"hooks": {"PermissionRequest": [{"hooks": [
        {"type": "command", "command": "/home/x/.claude/notify-question.sh"}]}]}}))
    assert not permission_hook_registered(settings)
    settings.write_text(json.dumps({"hooks": {"PermissionRequest": [{"hooks": [
        {"type": "command", "command": str(REPO / "hooks" / "editor-permission.sh")}]}]}}))
    assert permission_hook_registered(settings)
