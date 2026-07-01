"""Tests for SessionManager.

The Claude Agent SDK is mocked: we patch ``voice_bridge.sessions.ClaudeSDKClient``
with a fake that records construction + drives a scripted response stream per
turn. We import the *real* AssistantMessage / ResultMessage / TextBlock from the
SDK so the loop's ``isinstance`` checks pass, building minimal real instances.
"""
from __future__ import annotations

import asyncio

import pytest

from claude_agent_sdk import AssistantMessage, ResultMessage, TextBlock

import voice_bridge.sessions as sessions_mod
from voice_bridge.sessions import SessionManager
from voice_bridge.config import Config, ProjectConfig
from voice_bridge.types import Outbound


# --------------------------------------------------------------------------- #
# Helpers to build real SDK message instances
# --------------------------------------------------------------------------- #

def assistant(*texts: str) -> AssistantMessage:
    return AssistantMessage(
        content=[TextBlock(text=t) for t in texts],
        model="claude-test",
    )


def result(session_id: str = "sess-123") -> ResultMessage:
    return ResultMessage(
        subtype="success",
        duration_ms=1,
        duration_api_ms=1,
        is_error=False,
        num_turns=1,
        session_id=session_id,
    )


# --------------------------------------------------------------------------- #
# Fake SDK client
# --------------------------------------------------------------------------- #

class FakeClaudeSDKClient:
    """Records construction + drives a scripted response stream per turn."""

    instances: list["FakeClaudeSDKClient"] = []

    def __init__(self, options=None):
        self.options = options
        self.connected = False
        self.disconnected = False
        self.queries: list[str] = []
        # scripted_turns: list of (list-of-messages OR Exception), one per turn.
        self.scripted_turns: list = []
        self._turn_index = 0
        FakeClaudeSDKClient.instances.append(self)

    async def connect(self):
        self.connected = True

    async def disconnect(self):
        self.disconnected = True

    async def query(self, prompt, session_id="default"):
        self.queries.append(prompt)

    async def receive_response(self):
        if self._turn_index < len(self.scripted_turns):
            turn = self.scripted_turns[self._turn_index]
        else:
            turn = [assistant("ack"), result()]
        self._turn_index += 1
        if isinstance(turn, Exception):
            raise turn
        for msg in turn:
            await asyncio.sleep(0)
            if isinstance(msg, Exception):
                raise msg
            yield msg


# --------------------------------------------------------------------------- #
# Collaborator fakes
# --------------------------------------------------------------------------- #

class FakeStore:
    def __init__(self, enabled=None, session_ids=None):
        self._enabled = dict(enabled or {})
        self._session_ids = dict(session_ids or {})
        self.set_session_calls: list[tuple[str, str]] = []

    async def is_enabled(self, project):
        return self._enabled.get(project, True)

    async def set_enabled(self, project, enabled):
        self._enabled[project] = enabled

    async def get_session_id(self, project):
        return self._session_ids.get(project)

    async def set_session_id(self, project, session_id):
        self._session_ids[project] = session_id
        self.set_session_calls.append((project, session_id))


class FakeApprovals:
    pass


def make_cfg(**over):
    base = dict(
        telegram_bot_token="t",
        telegram_allowed_user_id=1,
        anthropic_api_key="a",
        openai_api_key="o",
        together_api_key="t",
        together_tts_model="cartesia/sonic",
        together_tts_language="lt",
        tts_backend="openai",
        tts_voice="alloy",
        piper_voice_path="/x.onnx",
        whisper_model="large-v3",
        autonomy_mode="safe",
        approval_timeout=300,
        db_path=":memory:",
        open_vscode_on_enable=False,
        close_vscode_on_disable=False,
    )
    base.update(over)
    return Config(**base)


def make_project(name="qwing", **over):
    base = dict(name=name, cwd="/tmp/qwing", enabled=True)
    base.update(over)
    return ProjectConfig(**base)


def make_sm(projects, store, on_outbound, cfg=None):
    return SessionManager(
        projects,
        cfg or make_cfg(),
        store,
        on_outbound,
        FakeApprovals(),
    )


@pytest.fixture(autouse=True)
def _patch_sdk(monkeypatch):
    FakeClaudeSDKClient.instances = []
    monkeypatch.setattr(sessions_mod, "ClaudeSDKClient", FakeClaudeSDKClient)
    yield
    FakeClaudeSDKClient.instances = []


async def _wait_for(predicate, tries=300, delay=0.005):
    for _ in range(tries):
        if predicate():
            return True
        await asyncio.sleep(delay)
    return False


# --------------------------------------------------------------------------- #
# start_all
# --------------------------------------------------------------------------- #

async def test_start_all_starts_only_enabled_projects():
    projects = [make_project("qwing"), make_project("other", enabled=False)]
    store = FakeStore(enabled={"qwing": True, "other": False})
    outbound: list[Outbound] = []

    async def on_outbound(o):
        outbound.append(o)

    sm = make_sm(projects, store, on_outbound)
    await sm.start_all()

    assert sm.is_running("qwing") is True
    assert sm.is_running("other") is False
    assert len(FakeClaudeSDKClient.instances) == 1
    assert FakeClaudeSDKClient.instances[0].connected is True

    await sm.stop_all()


def test_names_returns_configured_project_names_in_order():
    projects = [make_project("qwing"), make_project("bridge")]
    store = FakeStore()

    async def on_outbound(o):
        pass

    sm = make_sm(projects, store, on_outbound)

    assert sm.names() == ["qwing", "bridge"]


# --------------------------------------------------------------------------- #
# deliver / streaming loop
# --------------------------------------------------------------------------- #

async def test_deliver_drains_turn_captures_session_id_and_emits_outbound():
    project = make_project("qwing")
    store = FakeStore(enabled={"qwing": True})
    outbound: list[Outbound] = []

    async def on_outbound(o):
        outbound.append(o)

    sm = make_sm([project], store, on_outbound)
    await sm.start_all()

    client = FakeClaudeSDKClient.instances[0]
    client.scripted_turns = [[
        assistant("Done.\n---\ndiff here"),
        result("sess-123"),
    ]]

    await sm.deliver("qwing", "build the thing")

    assert await _wait_for(lambda: len(outbound) >= 2)

    assert client.queries == ["build the thing"]
    assert store._session_ids["qwing"] == "sess-123"
    assert outbound[0].text == "Vykdau."
    assert outbound[0].spoken == " "
    final = outbound[-1]
    assert isinstance(final, Outbound)
    assert final.project == "qwing"
    assert "Done." in final.text
    assert "diff here" in final.text
    assert final.spoken == ""

    await sm.stop_all()


async def test_deliver_reports_queue_position_when_busy():
    project = make_project("qwing")
    store = FakeStore(enabled={"qwing": True})
    outbound: list[Outbound] = []

    async def on_outbound(o):
        outbound.append(o)

    sm = make_sm([project], store, on_outbound)
    await sm.start_all()

    await sm.deliver("qwing", "first")
    await sm.deliver("qwing", "second")

    assert await _wait_for(lambda: any(o.text.startswith("Eilėje:") for o in outbound))
    queued = [o for o in outbound if o.text.startswith("Eilėje:")][0]
    assert queued.text == "Eilėje: 2."
    assert queued.spoken == " "

    await sm.stop_all()


async def test_deliver_mirrors_turns_to_project_transcript(tmp_path):
    project = make_project("qwing", cwd=str(tmp_path))
    store = FakeStore(enabled={"qwing": True})
    outbound: list[Outbound] = []

    async def on_outbound(o):
        outbound.append(o)

    sm = make_sm([project], store, on_outbound)
    await sm.start_all()
    client = FakeClaudeSDKClient.instances[0]
    client.scripted_turns = [[assistant("Padaryta."), result("sess-1")]]

    await sm.deliver("qwing", "sutvarkyk sita vieta")

    transcript = tmp_path / ".claude" / "voice-bridge-chat.md"
    assert await _wait_for(lambda: transcript.exists())
    text = transcript.read_text(encoding="utf-8")
    assert "Telegram" in text
    assert "sutvarkyk sita vieta" in text
    assert "Claude" in text
    assert "Padaryta." in text

    await sm.stop_all()


async def test_deliver_to_unstarted_project_is_noop():
    project = make_project("qwing", enabled=False)
    store = FakeStore(enabled={"qwing": False})

    async def on_outbound(o):
        pass

    sm = make_sm([project], store, on_outbound)
    await sm.start_all()
    await sm.deliver("qwing", "hello")  # must not raise
    assert FakeClaudeSDKClient.instances == []
    await sm.stop_all()


async def test_deliver_to_unknown_project_is_noop():
    project = make_project("qwing")
    store = FakeStore(enabled={"qwing": True})

    async def on_outbound(o):
        pass

    sm = make_sm([project], store, on_outbound)
    await sm.start_all()
    await sm.deliver("nope", "hello")  # unknown project, must not raise
    await sm.stop_all()


# --------------------------------------------------------------------------- #
# on/off lifecycle
# --------------------------------------------------------------------------- #

async def test_set_enabled_false_stops_and_persists_then_true_restarts():
    project = make_project("qwing")
    store = FakeStore(enabled={"qwing": True})

    async def on_outbound(o):
        pass

    sm = make_sm([project], store, on_outbound)
    await sm.start_all()
    assert sm.is_running("qwing") is True
    first_client = FakeClaudeSDKClient.instances[0]

    await sm.set_enabled("qwing", False)
    assert sm.is_running("qwing") is False
    assert first_client.disconnected is True
    assert store._enabled["qwing"] is False

    await sm.set_enabled("qwing", True)
    assert sm.is_running("qwing") is True
    assert store._enabled["qwing"] is True
    assert len(FakeClaudeSDKClient.instances) == 2

    await sm.stop_all()


async def test_set_enabled_true_opens_vscode_when_configured(monkeypatch):
    project = make_project("qwing", cwd="/tmp/qwing")
    store = FakeStore(enabled={"qwing": False})
    calls = []

    class FakeProc:
        returncode = 0

        async def wait(self):
            return 0

    async def fake_exec(*args, **kwargs):
        calls.append((args, kwargs))
        return FakeProc()

    monkeypatch.setattr(sessions_mod.shutil, "which", lambda name: "/usr/bin/code")
    monkeypatch.setattr(sessions_mod.asyncio, "create_subprocess_exec", fake_exec)

    async def on_outbound(o):
        pass

    sm = make_sm(
        [project],
        store,
        on_outbound,
        cfg=make_cfg(open_vscode_on_enable=True),
    )

    await sm.set_enabled("qwing", True)

    assert calls
    assert calls[0][0][:2] == ("/usr/bin/code", "/tmp/qwing")
    assert sm.is_running("qwing") is True

    await sm.stop_all()


async def test_set_enabled_false_closes_matching_vscode_window(monkeypatch):
    project = make_project("qwing", cwd="/tmp/WhisperX")
    store = FakeStore(enabled={"qwing": True})
    calls = []

    class FakeProc:
        def __init__(self, stdout=b"", returncode=0):
            self.stdout = stdout
            self.returncode = returncode

        async def communicate(self):
            return self.stdout, b""

        async def wait(self):
            return self.returncode

    async def fake_exec(*args, **kwargs):
        calls.append((args, kwargs))
        if args[:2] == ("/usr/bin/wmctrl", "-l"):
            return FakeProc(
                b"0x1 0 home Something - Other - Visual Studio Code\n"
                b"0x2 0 home Port Android app - WhisperX - Visual Studio Code\n"
            )
        return FakeProc()

    monkeypatch.setattr(sessions_mod.shutil, "which", lambda name: "/usr/bin/wmctrl")
    monkeypatch.setattr(sessions_mod.asyncio, "create_subprocess_exec", fake_exec)

    async def on_outbound(o):
        pass

    sm = make_sm(
        [project],
        store,
        on_outbound,
        cfg=make_cfg(close_vscode_on_disable=True),
    )
    await sm.start_all()

    await sm.set_enabled("qwing", False)

    assert any(call[0] == ("/usr/bin/wmctrl", "-ic", "0x2") for call in calls)
    assert not any(call[0] == ("/usr/bin/wmctrl", "-ic", "0x1") for call in calls)


# --------------------------------------------------------------------------- #
# permission wiring / resume / mode
# --------------------------------------------------------------------------- #

async def test_full_mode_uses_bypass_permissions():
    project = make_project("qwing", autonomy="full")
    store = FakeStore(enabled={"qwing": True})

    async def on_outbound(o):
        pass

    sm = make_sm([project], store, on_outbound)
    await sm.start_all()

    opts = FakeClaudeSDKClient.instances[0].options
    assert opts.permission_mode == "bypassPermissions"
    assert opts.can_use_tool is None

    await sm.stop_all()


async def test_safe_mode_uses_can_use_tool_not_bypass():
    project = make_project("qwing", autonomy="safe")
    store = FakeStore(enabled={"qwing": True})

    async def on_outbound(o):
        pass

    sm = make_sm([project], store, on_outbound)
    await sm.start_all()

    opts = FakeClaudeSDKClient.instances[0].options
    assert opts.permission_mode == "default"
    assert callable(opts.can_use_tool)

    await sm.stop_all()


async def test_ask_mode_uses_can_use_tool_not_bypass():
    project = make_project("qwing", autonomy="ask")
    store = FakeStore(enabled={"qwing": True})

    async def on_outbound(o):
        pass

    sm = make_sm([project], store, on_outbound)
    await sm.start_all()

    opts = FakeClaudeSDKClient.instances[0].options
    assert opts.permission_mode == "default"
    assert callable(opts.can_use_tool)

    await sm.stop_all()


async def test_resume_session_id_passed_to_options():
    project = make_project("qwing", autonomy="safe")
    store = FakeStore(enabled={"qwing": True},
                      session_ids={"qwing": "prev-sess-9"})

    async def on_outbound(o):
        pass

    sm = make_sm([project], store, on_outbound)
    await sm.start_all()

    opts = FakeClaudeSDKClient.instances[0].options
    assert opts.resume == "prev-sess-9"

    await sm.stop_all()


async def test_no_resume_when_no_session_id():
    project = make_project("qwing", autonomy="safe")
    store = FakeStore(enabled={"qwing": True})

    async def on_outbound(o):
        pass

    sm = make_sm([project], store, on_outbound)
    await sm.start_all()

    opts = FakeClaudeSDKClient.instances[0].options
    assert opts.resume is None

    await sm.stop_all()


async def test_system_prompt_includes_extra_and_split_instruction():
    project = make_project("qwing", system_prompt_extra="Be terse.")
    store = FakeStore(enabled={"qwing": True})

    async def on_outbound(o):
        pass

    sm = make_sm([project], store, on_outbound)
    await sm.start_all()

    sp = FakeClaudeSDKClient.instances[0].options.system_prompt
    assert isinstance(sp, dict)
    assert sp["type"] == "preset"
    assert sp["preset"] == "claude_code"
    append = sp["append"]
    assert "Be terse." in append
    assert "---" in append

    await sm.stop_all()


async def test_set_mode_rebuilds_session_with_new_permission_wiring():
    project = make_project("qwing", autonomy="safe")
    store = FakeStore(enabled={"qwing": True})

    async def on_outbound(o):
        pass

    sm = make_sm([project], store, on_outbound)
    await sm.start_all()
    assert callable(FakeClaudeSDKClient.instances[0].options.can_use_tool)
    first_client = FakeClaudeSDKClient.instances[0]

    await sm.set_mode("qwing", "full")

    assert first_client.disconnected is True
    assert len(FakeClaudeSDKClient.instances) == 2
    new_opts = FakeClaudeSDKClient.instances[1].options
    assert new_opts.permission_mode == "bypassPermissions"
    assert new_opts.can_use_tool is None
    assert sm.project("qwing").autonomy == "full"

    await sm.stop_all()


async def test_set_mode_when_not_running_only_updates_config():
    project = make_project("qwing", autonomy="safe", enabled=False)
    store = FakeStore(enabled={"qwing": False})

    async def on_outbound(o):
        pass

    sm = make_sm([project], store, on_outbound)
    await sm.start_all()
    assert sm.is_running("qwing") is False

    await sm.set_mode("qwing", "full")
    assert sm.project("qwing").autonomy == "full"
    assert FakeClaudeSDKClient.instances == []  # never started

    await sm.stop_all()


async def test_set_mode_invalid_mode_is_ignored():
    project = make_project("qwing", autonomy="safe")
    store = FakeStore(enabled={"qwing": True})

    async def on_outbound(o):
        pass

    sm = make_sm([project], store, on_outbound)
    await sm.start_all()
    assert sm.is_running("qwing") is True
    first_client = FakeClaudeSDKClient.instances[0]

    await sm.set_mode("qwing", "turbo")  # invalid mode

    # Mode unchanged, session not restarted
    assert sm.project("qwing").autonomy == "safe"
    assert first_client.disconnected is False
    assert len(FakeClaudeSDKClient.instances) == 1

    await sm.stop_all()


# --------------------------------------------------------------------------- #
# notify wiring (C6)
# --------------------------------------------------------------------------- #

async def test_notify_callback_emits_outbound_with_detail_and_summary(monkeypatch):
    project = make_project("qwing")
    store = FakeStore(enabled={"qwing": True})
    outbound: list[Outbound] = []

    async def on_outbound(o):
        outbound.append(o)

    captured = {}

    def fake_make_notify_server(on_notify, on_send_file=None, on_ask_user=None):
        captured["on_notify"] = on_notify
        captured["on_send_file"] = on_send_file
        captured["on_ask_user"] = on_ask_user
        return object()

    monkeypatch.setattr(sessions_mod, "make_notify_server", fake_make_notify_server)

    sm = make_sm([project], store, on_outbound)
    await sm.start_all()

    on_notify = captured["on_notify"]
    await on_notify("Need a decision", "Should I push to main?")

    assert len(outbound) == 1
    assert outbound[0].project == "qwing"
    assert outbound[0].text == "Should I push to main?"
    assert outbound[0].spoken == "Need a decision"

    # detail empty -> text falls back to summary
    await on_notify("Just a heads up", "")
    assert outbound[1].text == "Just a heads up"
    assert outbound[1].spoken == "Just a heads up"

    await sm.stop_all()


async def test_send_file_callback_emits_project_file_outbound(tmp_path, monkeypatch):
    project = make_project("qwing", cwd=str(tmp_path))
    target = tmp_path / "dist" / "out.txt"
    target.parent.mkdir()
    target.write_text("ok")
    store = FakeStore(enabled={"qwing": True})
    outbound: list[Outbound] = []

    async def on_outbound(o):
        outbound.append(o)

    captured = {}

    def fake_make_notify_server(on_notify, on_send_file=None, on_ask_user=None):
        captured["on_send_file"] = on_send_file
        return object()

    monkeypatch.setattr(sessions_mod, "make_notify_server", fake_make_notify_server)

    sm = make_sm([project], store, on_outbound)
    await sm.start_all()

    result = await captured["on_send_file"]("dist/out.txt", "rezultatas")

    assert result == "delivered"
    assert len(outbound) == 1
    assert outbound[0].project == "qwing"
    assert outbound[0].text == "rezultatas"
    assert outbound[0].spoken == ""
    assert outbound[0].file_path == str(target.resolve())

    await sm.stop_all()


async def test_send_file_callback_denies_path_outside_project(tmp_path, monkeypatch):
    project = make_project("qwing", cwd=str(tmp_path))
    outside = tmp_path.parent / "secret.txt"
    outside.write_text("secret")
    store = FakeStore(enabled={"qwing": True})
    outbound: list[Outbound] = []

    async def on_outbound(o):
        outbound.append(o)

    captured = {}

    def fake_make_notify_server(on_notify, on_send_file=None, on_ask_user=None):
        captured["on_send_file"] = on_send_file
        return object()

    monkeypatch.setattr(sessions_mod, "make_notify_server", fake_make_notify_server)

    sm = make_sm([project], store, on_outbound)
    await sm.start_all()

    result = await captured["on_send_file"](str(outside), "no")

    assert result.startswith("denied:")
    assert outbound == []

    await sm.stop_all()


async def test_ask_user_callback_routes_to_injected_callback(monkeypatch):
    project = make_project("qwing")
    store = FakeStore(enabled={"qwing": True})
    calls = []

    async def on_outbound(o):
        pass

    async def ask_user(project_name, question, choices):
        calls.append((project_name, question, choices))
        return "B"

    captured = {}

    def fake_make_notify_server(on_notify, on_send_file=None, on_ask_user=None):
        captured["on_ask_user"] = on_ask_user
        return object()

    monkeypatch.setattr(sessions_mod, "make_notify_server", fake_make_notify_server)

    sm = SessionManager(
        [project],
        make_cfg(),
        store,
        on_outbound,
        FakeApprovals(),
        ask_user,
    )
    await sm.start_all()

    result = await captured["on_ask_user"]("Rinktis?", ["A", "B"])

    assert result == "B"
    assert calls == [("qwing", "Rinktis?", ["A", "B"])]

    await sm.stop_all()


async def test_each_project_gets_its_own_notify_server(monkeypatch):
    projects = [make_project("qwing"), make_project("beta", cwd="/tmp/beta")]
    store = FakeStore(enabled={"qwing": True, "beta": True})
    outbound: list[Outbound] = []

    async def on_outbound(o):
        outbound.append(o)

    notifies = {}

    def fake_make_notify_server(on_notify, on_send_file=None, on_ask_user=None):
        # bind by identity; collect them in order
        notifies[len(notifies)] = on_notify
        return object()

    monkeypatch.setattr(sessions_mod, "make_notify_server", fake_make_notify_server)

    sm = make_sm(projects, store, on_outbound)
    await sm.start_all()

    assert len(notifies) == 2
    await notifies[0]("s0", "d0")
    await notifies[1]("s1", "d1")
    projects_seen = {o.project for o in outbound}
    assert projects_seen == {"qwing", "beta"}

    await sm.stop_all()


# --------------------------------------------------------------------------- #
# C8 resilience
# --------------------------------------------------------------------------- #

async def test_receive_response_error_emits_error_outbound_and_keeps_other_sessions():
    projects = [make_project("qwing"), make_project("beta", cwd="/tmp/beta")]
    store = FakeStore(enabled={"qwing": True, "beta": True})
    outbound: list[Outbound] = []

    async def on_outbound(o):
        outbound.append(o)

    sm = make_sm(projects, store, on_outbound)
    await sm.start_all()

    by_cwd = {c.options.cwd: c for c in FakeClaudeSDKClient.instances}
    qwing_client = by_cwd["/tmp/qwing"]
    beta_client = by_cwd["/tmp/beta"]

    qwing_client.scripted_turns = [RuntimeError("boom")]
    beta_client.scripted_turns = [[assistant("hi from beta"), result("beta-1")]]

    await sm.deliver("qwing", "do crash")
    await sm.deliver("beta", "do work")

    assert await _wait_for(lambda: len(outbound) >= 2)

    texts = {o.project: o for o in outbound}
    assert "Sesija krito" in texts["qwing"].text
    assert texts["qwing"].spoken == "Sesija krito, žiūrėk tekstą."
    assert "beta" in texts and "hi from beta" in texts["beta"].text

    # The crashed session is marked stopped; beta keeps running.
    assert sm.is_running("qwing") is False
    assert sm.is_running("beta") is True

    await sm.stop_all()


# --------------------------------------------------------------------------- #
# project lookup / stop_all
# --------------------------------------------------------------------------- #

async def test_project_lookup_returns_config_or_none():
    project = make_project("qwing")
    store = FakeStore(enabled={"qwing": True})

    async def on_outbound(o):
        pass

    sm = make_sm([project], store, on_outbound)
    assert sm.project("qwing") is project
    assert sm.project("nope") is None


async def test_stop_all_disconnects_every_running_client():
    projects = [make_project("qwing"), make_project("beta", cwd="/tmp/beta")]
    store = FakeStore(enabled={"qwing": True, "beta": True})

    async def on_outbound(o):
        pass

    sm = make_sm(projects, store, on_outbound)
    await sm.start_all()
    assert len(FakeClaudeSDKClient.instances) == 2

    await sm.stop_all()

    assert all(c.disconnected for c in FakeClaudeSDKClient.instances)
    assert sm.is_running("qwing") is False
    assert sm.is_running("beta") is False
