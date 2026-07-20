"""TDD tests for the unified IDE-notification hook ``hooks/voice-bridge-notify.py``.

The hook runs in a SEPARATE Claude Code process (not the bridge), so it never
imports the package: it only reads the hook JSON from stdin, classifies the
event, formats a plain-text message, and ATOMICALLY writes ONE spool file the
bridge later drains. The contract these tests pin down:

* one spool file per spoolable event, with the right ``kind``/``urgent``/``text``;
* ``urgent`` is True only for question/permission (they get TTS), False for
  stop/notification (text-only, to avoid noise);
* the content ``hash`` is stable for identical events (cross-event dedup);
* malformed / empty stdin never crashes — ``main`` returns 0 and writes nothing;
* a generic (non-Ask) PreToolUse writes nothing (would otherwise spool on every
  tool call);
* the write is atomic — no ``.tmp`` file is left behind, the drainer only ever
  sees a complete ``.json``.

The module filename has hyphens, so it is loaded by path via importlib.
"""
from __future__ import annotations

import importlib.util
import io
import json
from pathlib import Path

import pytest

_HOOK_PATH = Path(__file__).resolve().parent.parent / "hooks" / "voice-bridge-notify.py"


def _load_hook():
    spec = importlib.util.spec_from_file_location("vb_notify_hook", _HOOK_PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture
def hook():
    return _load_hook()


def _spool_files(directory: Path) -> list[Path]:
    return sorted(directory.glob("*.json"))


def _only_event(directory: Path) -> dict:
    files = _spool_files(directory)
    assert len(files) == 1, f"expected exactly one spool file, got {files}"
    return json.loads(files[0].read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# classification / build_event
# ---------------------------------------------------------------------------

def test_stop_event(hook):
    ev = hook.build_event({"hook_event_name": "Stop", "cwd": "/home/x/Projects/qwing"})
    assert ev["kind"] == "stop"
    assert ev["urgent"] is False
    assert ev["project"] == "qwing"
    assert ev["cwd"] == "/home/x/Projects/qwing"
    assert "baig" in ev["text"].lower()  # "✅ baigė"


def test_notification_event(hook):
    ev = hook.build_event({
        "hook_event_name": "Notification",
        "message": "Claude is waiting for your input",
        "cwd": "/home/x/Projects/qwing",
    })
    assert ev["kind"] == "notification"
    assert ev["urgent"] is False
    assert "waiting" in ev["text"]


def test_question_event_via_pretooluse(hook):
    ev = hook.build_event({
        "hook_event_name": "PreToolUse",
        "tool_name": "AskUserQuestion",
        "cwd": "/home/x/Projects/qwing",
        "tool_input": {
            "questions": [
                {
                    "question": "Deploy now?",
                    "options": [
                        {"label": "Yes", "description": "ship it"},
                        {"label": "No"},
                    ],
                }
            ]
        },
    })
    assert ev["kind"] == "question"
    assert ev["urgent"] is True
    assert "Deploy now?" in ev["text"]
    assert "1) Yes" in ev["text"] and "ship it" in ev["text"]
    assert "2) No" in ev["text"]


def test_permission_request_event(hook):
    ev = hook.build_event({
        "hook_event_name": "PermissionRequest",
        "tool_name": "Bash",
        "cwd": "/home/x/Projects/qwing",
        "tool_input": {"command": "git push origin main"},
    })
    assert ev["kind"] == "permission"
    assert ev["urgent"] is True
    assert "Bash" in ev["text"]
    assert "git push origin main" in ev["text"]


def test_permission_request_carrying_ask_is_a_question(hook):
    # A PermissionRequest whose tool IS AskUserQuestion must format as the full
    # question (urgent), not as a bare "prašo leidimo" line.
    ev = hook.build_event({
        "hook_event_name": "PermissionRequest",
        "tool_name": "AskUserQuestion",
        "cwd": "/home/x/Projects/qwing",
        "tool_input": {"questions": [{"question": "Pick one", "options": [{"label": "A"}]}]},
    })
    assert ev["kind"] == "question"
    assert ev["urgent"] is True
    assert "Pick one" in ev["text"] and "1) A" in ev["text"]


def test_generic_pretooluse_is_not_spooled(hook):
    # A normal PreToolUse (not AskUserQuestion) must produce nothing, else every
    # tool call in the IDE would spool a notification.
    assert hook.build_event({
        "hook_event_name": "PreToolUse",
        "tool_name": "Bash",
        "cwd": "/home/x/Projects/qwing",
        "tool_input": {"command": "ls"},
    }) is None


def test_unknown_event_is_not_spooled(hook):
    assert hook.build_event({"hook_event_name": "SessionStart", "cwd": "/x"}) is None


def test_project_is_basename_of_cwd(hook):
    ev = hook.build_event({"hook_event_name": "Stop", "cwd": "/a/b/c/myrepo"})
    assert ev["project"] == "myrepo"


def test_event_has_all_required_fields(hook):
    ev = hook.build_event({"hook_event_name": "Stop", "cwd": "/home/x/Projects/qwing"})
    for field in ("kind", "project", "cwd", "text", "urgent", "hash", "ts"):
        assert field in ev, field
    assert isinstance(ev["ts"], (int, float))
    assert isinstance(ev["hash"], str) and ev["hash"]


# ---------------------------------------------------------------------------
# dedup hash stability
# ---------------------------------------------------------------------------

def test_hash_stable_for_identical_events(hook):
    payload = {
        "hook_event_name": "PermissionRequest",
        "tool_name": "Bash",
        "cwd": "/home/x/Projects/qwing",
        "tool_input": {"command": "rm -rf build"},
    }
    a = hook.build_event(dict(payload))
    b = hook.build_event(dict(payload))
    assert a["hash"] == b["hash"]


def test_hash_differs_for_different_content(hook):
    a = hook.build_event({"hook_event_name": "Notification", "message": "one", "cwd": "/x/q"})
    b = hook.build_event({"hook_event_name": "Notification", "message": "two", "cwd": "/x/q"})
    assert a["hash"] != b["hash"]


def test_hash_differs_across_projects(hook):
    a = hook.build_event({"hook_event_name": "Stop", "cwd": "/x/alpha"})
    b = hook.build_event({"hook_event_name": "Stop", "cwd": "/x/beta"})
    assert a["hash"] != b["hash"]


# ---------------------------------------------------------------------------
# write_event — atomicity
# ---------------------------------------------------------------------------

def test_write_event_writes_one_json_no_tmp(hook, tmp_path):
    ev = hook.build_event({"hook_event_name": "Stop", "cwd": "/x/qwing"})
    hook.write_event(ev, tmp_path)
    assert len(_spool_files(tmp_path)) == 1
    assert list(tmp_path.glob("*.tmp")) == []  # atomic: no half-written temp left
    assert _only_event(tmp_path)["kind"] == "stop"


def test_write_event_creates_missing_dir(hook, tmp_path):
    target = tmp_path / "does" / "not" / "exist"
    ev = hook.build_event({"hook_event_name": "Stop", "cwd": "/x/qwing"})
    hook.write_event(ev, target)
    assert len(_spool_files(target)) == 1


# ---------------------------------------------------------------------------
# main() — stdin -> spool, never crashes
# ---------------------------------------------------------------------------

def _run_main(hook, monkeypatch, tmp_path, stdin_text):
    monkeypatch.setenv("VOICE_BRIDGE_INBOX_DIR", str(tmp_path))
    monkeypatch.setattr(hook.sys, "stdin", io.StringIO(stdin_text))
    return hook.main()


def test_main_writes_exactly_one_file(hook, monkeypatch, tmp_path):
    payload = json.dumps({"hook_event_name": "Stop", "cwd": "/home/x/Projects/qwing"})
    rc = _run_main(hook, monkeypatch, tmp_path, payload)
    assert rc == 0
    ev = _only_event(tmp_path)
    assert ev["kind"] == "stop" and ev["project"] == "qwing"


def test_main_malformed_stdin_no_crash_exit_0(hook, monkeypatch, tmp_path):
    rc = _run_main(hook, monkeypatch, tmp_path, "this is not json {{{")
    assert rc == 0
    assert _spool_files(tmp_path) == []  # nothing spooled, no crash


def test_main_empty_stdin_exit_0(hook, monkeypatch, tmp_path):
    rc = _run_main(hook, monkeypatch, tmp_path, "")
    assert rc == 0
    assert _spool_files(tmp_path) == []


def test_main_generic_pretooluse_writes_nothing(hook, monkeypatch, tmp_path):
    payload = json.dumps({
        "hook_event_name": "PreToolUse",
        "tool_name": "Bash",
        "cwd": "/x/q",
        "tool_input": {"command": "ls"},
    })
    rc = _run_main(hook, monkeypatch, tmp_path, payload)
    assert rc == 0
    assert _spool_files(tmp_path) == []


def test_main_question_is_urgent(hook, monkeypatch, tmp_path):
    payload = json.dumps({
        "hook_event_name": "PreToolUse",
        "tool_name": "AskUserQuestion",
        "cwd": "/x/q",
        "tool_input": {"questions": [{"question": "Go?", "options": [{"label": "Yes"}]}]},
    })
    _run_main(hook, monkeypatch, tmp_path, payload)
    assert _only_event(tmp_path)["urgent"] is True
