"""Tests for SessionManager.

The Claude Agent SDK is mocked: we patch ``voice_bridge.sessions.ClaudeSDKClient``
with a fake that records construction + drives a scripted response stream per
turn. We import the *real* AssistantMessage / ResultMessage / TextBlock from the
SDK so the loop's ``isinstance`` checks pass, building minimal real instances.
"""
from __future__ import annotations

import asyncio

import pytest

from claude_agent_sdk import (
    AssistantMessage,
    ResultMessage,
    TextBlock,
    ToolUseBlock,
)

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


def tool(name: str, _id: str = "t", **inp) -> ToolUseBlock:
    return ToolUseBlock(id=_id, name=name, input=inp)


def tool_msg(*blocks) -> AssistantMessage:
    return AssistantMessage(content=list(blocks), model="claude-test")


def result(
    session_id: str = "sess-123",
    *,
    usage: dict | None = None,
    total_cost_usd: float | None = None,
) -> ResultMessage:
    return ResultMessage(
        subtype="success",
        duration_ms=1,
        duration_api_ms=1,
        is_error=False,
        num_turns=1,
        session_id=session_id,
        usage=usage,
        total_cost_usd=total_cost_usd,
    )


# --------------------------------------------------------------------------- #
# Fake SDK client
# --------------------------------------------------------------------------- #

class FakeClaudeSDKClient:
    """Records construction + drives a scripted response stream per turn."""

    instances: list["FakeClaudeSDKClient"] = []
    # When True, every new client's connect() raises (simulates a broken
    # project). Tests flip this to exercise restart/connect-failure paths.
    fail_connect: bool = False

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
        if type(self).fail_connect:
            raise RuntimeError("connect boom")
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
        self.usage_calls: list[dict] = []

    async def is_enabled(self, project):
        return self._enabled.get(project, True)

    async def set_enabled(self, project, enabled):
        self._enabled[project] = enabled

    async def get_session_id(self, project):
        return self._session_ids.get(project)

    async def set_session_id(self, project, session_id):
        self._session_ids[project] = session_id
        self.set_session_calls.append((project, session_id))

    async def add_usage(
        self,
        project,
        *,
        cost_usd,
        input_tokens,
        output_tokens,
        cache_read_tokens=0,
        cache_creation_tokens=0,
    ):
        self.usage_calls.append({
            "project": project,
            "cost_usd": cost_usd,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "cache_read_tokens": cache_read_tokens,
            "cache_creation_tokens": cache_creation_tokens,
        })


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
    FakeClaudeSDKClient.fail_connect = False
    monkeypatch.setattr(sessions_mod, "ClaudeSDKClient", FakeClaudeSDKClient)
    yield
    FakeClaudeSDKClient.instances = []
    FakeClaudeSDKClient.fail_connect = False


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
    assert outbound[0].text == "Working."
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

    assert await _wait_for(lambda: any(o.text.startswith("Queued:") for o in outbound))
    queued = [o for o in outbound if o.text.startswith("Queued:")][0]
    assert queued.text == "Queued: 2."
    assert queued.spoken == " "

    await sm.stop_all()


async def test_interrupt_restarts_enabled_session_and_emits_status():
    project = make_project("qwing")
    store = FakeStore(enabled={"qwing": True})
    outbound: list[Outbound] = []

    async def on_outbound(o):
        outbound.append(o)

    sm = make_sm([project], store, on_outbound)
    await sm.start_all()
    first = FakeClaudeSDKClient.instances[0]

    stopped = await sm.interrupt("qwing")

    assert stopped is True
    assert first.disconnected is True
    assert len(FakeClaudeSDKClient.instances) == 2
    assert outbound[-1].text == "Interrupted."
    assert outbound[-1].spoken == " "

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
    assert texts["qwing"].spoken == "The session crashed. Check the text."
    # crash notices are ALERT-class so TTS can use the distinct alert voice
    assert texts["qwing"].alert is True
    assert "beta" in texts and "hi from beta" in texts["beta"].text
    # a normal assistant turn stays on the routine (non-alert) voice
    assert texts["beta"].alert is False

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


# --------------------------------------------------------------------------- #
# start_all per-project isolation (problem 1)
# --------------------------------------------------------------------------- #

async def test_start_all_isolates_a_failing_project():
    projects = [make_project("qwing"), make_project("beta", cwd="/tmp/beta")]
    store = FakeStore(enabled={"qwing": True, "beta": True})
    outbound: list[Outbound] = []

    async def on_outbound(o):
        outbound.append(o)

    sm = make_sm(projects, store, on_outbound)

    orig_start = sm._start

    async def flaky_start(name):
        if name == "qwing":
            raise RuntimeError("connect boom")
        await orig_start(name)

    sm._start = flaky_start

    # Must NOT raise even though qwing's start blows up.
    await sm.start_all()

    assert sm.is_running("qwing") is False
    assert sm.is_running("beta") is True
    fails = [o for o in outbound if o.project == "qwing"]
    assert fails, "expected a failure Outbound for qwing"
    assert "nepavyko paleisti" in fails[0].text
    assert fails[0].spoken == "nepavyko paleisti"

    await sm.stop_all()


# --------------------------------------------------------------------------- #
# supervised auto-restart (problem 2a)
# --------------------------------------------------------------------------- #

def _fast_backoff(sm):
    sm._restart_backoff_base = 0
    sm._restart_backoff_cap = 0


async def test_crash_schedules_supervised_restart_and_resumes():
    project = make_project("qwing")
    store = FakeStore(enabled={"qwing": True}, session_ids={"qwing": "prev-9"})
    outbound: list[Outbound] = []

    async def on_outbound(o):
        outbound.append(o)

    sm = make_sm([project], store, on_outbound)
    _fast_backoff(sm)
    await sm.start_all()

    client0 = FakeClaudeSDKClient.instances[0]
    client0.scripted_turns = [RuntimeError("boom")]

    await sm.deliver("qwing", "go")

    # The crash schedules a supervised restart; a new client resumes.
    assert await _wait_for(lambda: len(FakeClaudeSDKClient.instances) >= 2)
    assert await _wait_for(lambda: sm.is_running("qwing"))
    client1 = FakeClaudeSDKClient.instances[1]
    assert client1.options.resume == "prev-9"
    assert client1.connected is True

    await sm.stop_all()


async def test_attempt_counter_resets_after_successful_turn():
    project = make_project("qwing")
    store = FakeStore(enabled={"qwing": True})
    outbound: list[Outbound] = []

    async def on_outbound(o):
        outbound.append(o)

    sm = make_sm([project], store, on_outbound)
    _fast_backoff(sm)
    await sm.start_all()

    client0 = FakeClaudeSDKClient.instances[0]
    client0.scripted_turns = [RuntimeError("boom")]

    await sm.deliver("qwing", "crash")

    assert await _wait_for(
        lambda: sm.is_running("qwing") and len(FakeClaudeSDKClient.instances) >= 2
    )
    # After a restart but before any successful turn the counter is non-zero.
    assert sm._attempts.get("qwing") == 1

    # A turn that completes cleanly resets the counter.
    await sm.deliver("qwing", "ok now")
    assert await _wait_for(lambda: sm._attempts.get("qwing", 0) == 0)

    await sm.stop_all()


async def test_restart_gives_up_after_max_failures_and_disables():
    project = make_project("qwing")
    store = FakeStore(enabled={"qwing": True})
    outbound: list[Outbound] = []

    async def on_outbound(o):
        outbound.append(o)

    sm = make_sm([project], store, on_outbound)
    _fast_backoff(sm)
    sm._max_restart_attempts = 3
    await sm.start_all()

    client0 = FakeClaudeSDKClient.instances[0]
    client0.scripted_turns = [RuntimeError("boom")]
    # Every restart's connect fails from now on.
    FakeClaudeSDKClient.fail_connect = True

    await sm.deliver("qwing", "crash")

    assert await _wait_for(lambda: store._enabled.get("qwing") is False)
    giveup = [o for o in outbound if "išjungiau" in o.text]
    assert giveup, "expected a give-up Outbound"
    assert "/on qwing" in giveup[0].text
    assert sm.is_running("qwing") is False
    # No lingering supervisor task.
    assert await _wait_for(lambda: "qwing" not in sm._restart_tasks)

    FakeClaudeSDKClient.fail_connect = False
    await sm.stop_all()


async def test_intentional_disable_does_not_auto_restart():
    project = make_project("qwing")
    store = FakeStore(enabled={"qwing": True})
    outbound: list[Outbound] = []

    async def on_outbound(o):
        outbound.append(o)

    sm = make_sm([project], store, on_outbound)
    _fast_backoff(sm)
    await sm.start_all()
    assert sm.is_running("qwing") is True

    await sm.set_enabled("qwing", False)
    assert sm.is_running("qwing") is False

    # Give the loop ample time to (wrongly) resurrect the session.
    await asyncio.sleep(0.02)
    assert sm.is_running("qwing") is False
    assert sm._restart_tasks == {}
    assert len(FakeClaudeSDKClient.instances) == 1

    await sm.stop_all()


async def test_stop_all_cancels_a_pending_restart():
    project = make_project("qwing")
    store = FakeStore(enabled={"qwing": True})
    outbound: list[Outbound] = []

    async def on_outbound(o):
        outbound.append(o)

    sm = make_sm([project], store, on_outbound)
    # Long backoff so the supervisor is still sleeping when we stop.
    sm._restart_backoff_base = 100
    sm._restart_backoff_cap = 100
    await sm.start_all()

    client0 = FakeClaudeSDKClient.instances[0]
    client0.scripted_turns = [RuntimeError("boom")]
    await sm.deliver("qwing", "crash")

    assert await _wait_for(lambda: "qwing" in sm._restart_tasks)

    await sm.stop_all()

    assert sm._restart_tasks == {}
    assert sm.is_running("qwing") is False
    # The pending supervisor never resurrected the session.
    assert len(FakeClaudeSDKClient.instances) == 1


# --------------------------------------------------------------------------- #
# deliver recovery (problem 2b)
# --------------------------------------------------------------------------- #

async def test_deliver_recovery_starts_enabled_session_and_delivers():
    project = make_project("qwing")
    store = FakeStore(enabled={"qwing": True}, session_ids={"qwing": "s-42"})
    outbound: list[Outbound] = []

    async def on_outbound(o):
        outbound.append(o)

    sm = make_sm([project], store, on_outbound)
    # No start_all: the project is enabled but has no live session.
    assert sm.is_running("qwing") is False

    await sm.deliver("qwing", "wake up")

    assert await _wait_for(lambda: sm.is_running("qwing"))
    client = FakeClaudeSDKClient.instances[0]
    assert client.options.resume == "s-42"
    assert await _wait_for(lambda: client.queries == ["wake up"])

    await sm.stop_all()


async def test_deliver_recovery_no_double_start_when_called_twice():
    project = make_project("qwing")
    store = FakeStore(enabled={"qwing": True})
    outbound: list[Outbound] = []

    async def on_outbound(o):
        outbound.append(o)

    sm = make_sm([project], store, on_outbound)
    assert sm.is_running("qwing") is False

    await asyncio.gather(
        sm.deliver("qwing", "one"),
        sm.deliver("qwing", "two"),
    )

    assert await _wait_for(lambda: sm.is_running("qwing"))
    # Exactly one client despite two racing delivers.
    assert len(FakeClaudeSDKClient.instances) == 1
    client = FakeClaudeSDKClient.instances[0]
    assert await _wait_for(lambda: set(client.queries) == {"one", "two"})

    await sm.stop_all()


# --------------------------------------------------------------------------- #
# error ResultMessage with no assistant text (problem 3)
# --------------------------------------------------------------------------- #

async def test_error_result_without_text_emits_outbound():
    project = make_project("qwing")
    store = FakeStore(enabled={"qwing": True})
    outbound: list[Outbound] = []

    async def on_outbound(o):
        outbound.append(o)

    sm = make_sm([project], store, on_outbound)
    await sm.start_all()

    client = FakeClaudeSDKClient.instances[0]
    err_result = ResultMessage(
        subtype="error_max_turns",
        duration_ms=1,
        duration_api_ms=1,
        is_error=True,
        num_turns=1,
        session_id="s-err",
    )
    err_result_with_text = ResultMessage(
        subtype="error_during_execution",
        duration_ms=1,
        duration_api_ms=1,
        is_error=True,
        num_turns=1,
        session_id="s-err2",
        result="explicit error detail",
    )
    client.scripted_turns = [[err_result], [err_result_with_text]]

    await sm.deliver("qwing", "do too much")
    assert await _wait_for(
        lambda: any(
            o.project == "qwing" and "klaida" in o.text for o in outbound
        )
    )
    first = [o for o in outbound if o.project == "qwing" and "klaida" in o.text][-1]
    assert "error_max_turns" in first.text  # falls back to subtype
    assert first.spoken.strip()  # not silent

    await sm.deliver("qwing", "again")
    assert await _wait_for(
        lambda: any(o.text == "explicit error detail" for o in outbound)
    )

    await sm.stop_all()


# --------------------------------------------------------------------------- #
# per-turn token & cost usage capture (B3c)
# --------------------------------------------------------------------------- #

async def test_result_message_usage_and_cost_captured_in_store():
    project = make_project("qwing")
    store = FakeStore(enabled={"qwing": True})
    outbound: list[Outbound] = []

    async def on_outbound(o):
        outbound.append(o)

    sm = make_sm([project], store, on_outbound)
    await sm.start_all()

    client = FakeClaudeSDKClient.instances[0]
    client.scripted_turns = [[
        assistant("Done."),
        result(
            "sess-123",
            usage={
                "input_tokens": 120,
                "output_tokens": 45,
                "cache_read_input_tokens": 10,
                "cache_creation_input_tokens": 3,
            },
            total_cost_usd=0.0456,
        ),
    ]]

    await sm.deliver("qwing", "build the thing")
    assert await _wait_for(lambda: len(store.usage_calls) >= 1)

    call = store.usage_calls[0]
    assert call["project"] == "qwing"
    assert call["cost_usd"] == 0.0456
    assert call["input_tokens"] == 120
    assert call["output_tokens"] == 45
    assert call["cache_read_tokens"] == 10
    assert call["cache_creation_tokens"] == 3

    await sm.stop_all()


async def test_result_message_missing_usage_and_none_cost_defaults_to_zero():
    # Claude Code subscription auth: total_cost_usd is None and usage may be
    # absent entirely. Missing keys/usage must default to 0 tokens; the None
    # cost is passed through as-is (Store.add_usage treats None as 0).
    project = make_project("qwing")
    store = FakeStore(enabled={"qwing": True})
    outbound: list[Outbound] = []

    async def on_outbound(o):
        outbound.append(o)

    sm = make_sm([project], store, on_outbound)
    await sm.start_all()

    client = FakeClaudeSDKClient.instances[0]
    client.scripted_turns = [[
        assistant("Done."),
        result("sess-123", usage=None, total_cost_usd=None),
    ]]

    await sm.deliver("qwing", "build the thing")
    assert await _wait_for(lambda: len(store.usage_calls) >= 1)

    call = store.usage_calls[0]
    assert call["cost_usd"] is None
    assert call["input_tokens"] == 0
    assert call["output_tokens"] == 0
    assert call["cache_read_tokens"] == 0
    assert call["cache_creation_tokens"] == 0

    await sm.stop_all()


async def test_malformed_usage_payload_does_not_crash_turn():
    # A non-numeric usage value must be swallowed (logged, not raised) so a
    # bad usage payload can never be mistaken for a turn crash (C8 corollary).
    project = make_project("qwing")
    store = FakeStore(enabled={"qwing": True})
    outbound: list[Outbound] = []

    async def on_outbound(o):
        outbound.append(o)

    sm = make_sm([project], store, on_outbound)
    await sm.start_all()

    client = FakeClaudeSDKClient.instances[0]
    client.scripted_turns = [[
        assistant("Done."),
        result("sess-123", usage={"input_tokens": "not-a-number"}),
    ]]

    await sm.deliver("qwing", "build the thing")
    assert await _wait_for(lambda: any(o.text == "Done." for o in outbound))

    # the turn completed normally and the session is still healthy
    assert sm.is_running("qwing")
    assert not any("krito" in o.text.lower() for o in outbound)
    # session_id was still persisted despite the bad usage payload
    assert store._session_ids["qwing"] == "sess-123"

    await sm.stop_all()


# --------------------------------------------------------------------------- #
# heartbeat watchdog during long turns (item 1)
# --------------------------------------------------------------------------- #

async def test_heartbeat_emits_during_long_silent_turn():
    project = make_project("qwing")
    store = FakeStore(enabled={"qwing": True})
    outbound: list[Outbound] = []

    async def on_outbound(o):
        outbound.append(o)

    sm = make_sm([project], store, on_outbound)
    sm.heartbeat_interval = 0.01
    await sm.start_all()

    client = FakeClaudeSDKClient.instances[0]

    async def slow_response():
        # Genuine silence: the SDK is busy running tools and emits no text
        # for far longer than the heartbeat interval before finishing.
        await asyncio.sleep(0.08)
        yield assistant("done")
        yield result("s-1")

    client.receive_response = slow_response

    await sm.deliver("qwing", "long task")

    assert await _wait_for(
        lambda: any("dirbu" in o.text.lower() for o in outbound)
    ), "expected at least one 'still working' heartbeat during the silent turn"

    await sm.stop_all()


async def test_heartbeat_outbound_failure_does_not_poison_turn():
    # A failure while emitting a heartbeat (e.g. a transient store hiccup) must
    # be swallowed: the turn's real output still lands and the session survives.
    project = make_project("qwing")
    store = FakeStore(enabled={"qwing": True})
    outbound: list[Outbound] = []

    async def on_outbound(o):
        if "dirbu" in o.text.lower():
            raise RuntimeError("store hiccup during heartbeat")
        outbound.append(o)

    sm = make_sm([project], store, on_outbound)
    sm.heartbeat_interval = 0.01
    await sm.start_all()

    client = FakeClaudeSDKClient.instances[0]

    async def slow_response():
        await asyncio.sleep(0.08)  # silence long enough for a heartbeat to fire
        yield assistant("done")
        yield result("s-1")

    client.receive_response = slow_response

    await sm.deliver("qwing", "long task")

    assert await _wait_for(lambda: any(o.text == "done" for o in outbound)), (
        "the turn's real output must still be delivered despite a failing heartbeat"
    )
    assert not any("krito" in o.text.lower() for o in outbound), (
        "a failing heartbeat must not crash/restart the session"
    )
    assert sm.is_running("qwing")

    await sm.stop_all()


async def test_no_heartbeat_on_fast_turn():
    project = make_project("qwing")
    store = FakeStore(enabled={"qwing": True})
    outbound: list[Outbound] = []

    async def on_outbound(o):
        outbound.append(o)

    sm = make_sm([project], store, on_outbound)
    # Interval far longer than a fast turn: no heartbeat should ever fire.
    sm.heartbeat_interval = 5.0
    await sm.start_all()

    client = FakeClaudeSDKClient.instances[0]
    client.scripted_turns = [[assistant("quick"), result("s-1")]]

    await sm.deliver("qwing", "quick task")
    assert await _wait_for(lambda: any(o.text == "quick" for o in outbound))

    assert not any("dirbu" in o.text.lower() for o in outbound)

    await sm.stop_all()


async def test_heartbeat_does_not_fire_after_intentional_cancel():
    project = make_project("qwing")
    store = FakeStore(enabled={"qwing": True})
    outbound: list[Outbound] = []

    async def on_outbound(o):
        outbound.append(o)

    sm = make_sm([project], store, on_outbound)
    sm.heartbeat_interval = 0.02
    await sm.start_all()

    client = FakeClaudeSDKClient.instances[0]
    started = asyncio.Event()

    async def hanging_response():
        started.set()
        await asyncio.sleep(3600)
        yield result("s-1")

    client.receive_response = hanging_response

    await sm.deliver("qwing", "hang")
    await asyncio.wait_for(started.wait(), timeout=1)

    await sm.stop_all()

    count_after_stop = sum(1 for o in outbound if "dirbu" in o.text.lower())
    # Give any leaked watchdog several intervals to (wrongly) fire.
    await asyncio.sleep(0.08)
    count_later = sum(1 for o in outbound if "dirbu" in o.text.lower())
    assert count_later == count_after_stop, (
        "watchdog fired after the turn was intentionally cancelled"
    )


# --------------------------------------------------------------------------- #
# connect-orphan leak / double-start window in _start (item 2)
# --------------------------------------------------------------------------- #

async def test_start_cancelled_during_connect_disconnects_and_no_orphan(monkeypatch):
    project = make_project("qwing")
    store = FakeStore(enabled={"qwing": True})

    async def on_outbound(o):
        pass

    sm = make_sm([project], store, on_outbound)

    connecting = asyncio.Event()

    class BlockingClient(FakeClaudeSDKClient):
        async def connect(self):
            connecting.set()
            await asyncio.sleep(3600)  # hang until cancelled

    monkeypatch.setattr(sessions_mod, "ClaudeSDKClient", BlockingClient)

    task = asyncio.create_task(sm._start("qwing"))
    await asyncio.wait_for(connecting.wait(), timeout=1)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    # The connected-but-unregistered client must be cleaned up, not orphaned.
    assert sm.is_running("qwing") is False
    assert len(FakeClaudeSDKClient.instances) == 1
    assert FakeClaudeSDKClient.instances[0].disconnected is True


async def test_concurrent_start_builds_single_client(monkeypatch):
    project = make_project("qwing")
    store = FakeStore(enabled={"qwing": True})

    async def on_outbound(o):
        pass

    sm = make_sm([project], store, on_outbound)

    class SlowConnectClient(FakeClaudeSDKClient):
        async def connect(self):
            # Yield control so a racing _start can also reach the build/connect
            # stage before this one registers into _sessions. Without the
            # per-project start lock this window builds two CLI subprocesses.
            await asyncio.sleep(0.02)
            self.connected = True

    monkeypatch.setattr(sessions_mod, "ClaudeSDKClient", SlowConnectClient)

    await asyncio.gather(sm._start("qwing"), sm._start("qwing"))

    assert sm.is_running("qwing") is True
    assert len(FakeClaudeSDKClient.instances) == 1

    await sm.stop_all()


# --------------------------------------------------------------------------- #
# verbose tool-activity streaming (B3d)
# --------------------------------------------------------------------------- #

def _activity(outbound):
    """Tool-activity Outbounds are the text-only ones whose text starts 🔧."""
    return [o for o in outbound if o.text.startswith("🔧")]


async def test_set_verbose_toggles_flag_per_project_and_all():
    projects = [make_project("qwing"), make_project("beta", cwd="/tmp/beta")]
    store = FakeStore(enabled={"qwing": True, "beta": True})

    async def on_outbound(o):
        pass

    sm = make_sm(projects, store, on_outbound)

    await sm.set_verbose("qwing", True)
    assert sm.project("qwing").verbose is True
    assert sm.project("beta").verbose is False

    await sm.set_verbose(None, True)
    assert sm.project("qwing").verbose is True
    assert sm.project("beta").verbose is True

    await sm.set_verbose(None, False)
    assert sm.project("qwing").verbose is False
    assert sm.project("beta").verbose is False

    # unknown project is a no-op (must not raise)
    await sm.set_verbose("nope", True)


async def test_verbose_streams_coalesced_tool_activity_then_final_text():
    project = make_project("qwing", verbose=True)
    store = FakeStore(enabled={"qwing": True})
    outbound: list[Outbound] = []

    async def on_outbound(o):
        outbound.append(o)

    sm = make_sm([project], store, on_outbound)
    sm.verbose_batch_size = 2  # deterministic size-based flush
    await sm.start_all()

    client = FakeClaudeSDKClient.instances[0]
    client.scripted_turns = [[
        tool_msg(
            tool("Bash", command="npm run build && echo done"),
            tool("Read", file_path="/home/x/src/app.py"),
            tool("Write", file_path="/home/x/out.txt",
                 content="SECRETLARGECONTENT" * 100),
        ),
        assistant("All done."),
        result("s-1"),
    ]]

    await sm.deliver("qwing", "go")

    assert await _wait_for(lambda: any(o.text == "All done." for o in outbound))

    acts = _activity(outbound)
    # batch_size=2 with 3 tool blocks -> flush of 2, then remaining 1 before text
    assert len(acts) == 2
    # Activity must be TEXT-ONLY. It uses the SILENT sentinel (a single space),
    # NOT spoken="" (which is the turn-end sentinel that make_outbound would
    # turn back into a spoken line and speak). Assert both the sentinel and the
    # downstream no-TTS property (make_outbound skips TTS when to_spoken is empty).
    from voice_bridge.sanitizer import to_spoken
    assert all(a.spoken == sessions_mod._SILENT_SPOKEN for a in acts)
    assert all(a.spoken and to_spoken(a.spoken) == "" for a in acts), (
        "silent sentinel must yield no TTS downstream"
    )

    first_lines = acts[0].text.split("\n")
    assert first_lines == [
        "🔧 Bash: npm run build && echo done",
        "🔧 Read: app.py",
    ]
    assert acts[1].text == "🔧 Write: out.txt"

    # NEVER include large content
    assert "SECRETLARGECONTENT" not in "\n".join(a.text for a in acts)

    # the final assistant text still lands (make_outbound derives its TTS)
    final = outbound[-1]
    assert final.text == "All done."
    assert final.spoken == ""

    # activity is emitted BEFORE the final text
    assert outbound.index(acts[-1]) < outbound.index(final)

    await sm.stop_all()


async def test_verbose_off_emits_no_tool_activity_regression():
    project = make_project("qwing")  # verbose defaults OFF
    store = FakeStore(enabled={"qwing": True})
    outbound: list[Outbound] = []

    async def on_outbound(o):
        outbound.append(o)

    sm = make_sm([project], store, on_outbound)
    sm.verbose_batch_size = 2
    await sm.start_all()

    client = FakeClaudeSDKClient.instances[0]
    client.scripted_turns = [[
        tool_msg(
            tool("Bash", command="ls -la"),
            tool("Read", file_path="/home/x/src/app.py"),
        ),
        assistant("All done."),
        result("s-1"),
    ]]

    await sm.deliver("qwing", "go")
    assert await _wait_for(lambda: any(o.text == "All done." for o in outbound))

    assert _activity(outbound) == [], "verbose OFF must emit no tool activity"

    await sm.stop_all()


async def test_verbose_malformed_tool_block_does_not_crash_turn():
    project = make_project("qwing", verbose=True)
    store = FakeStore(enabled={"qwing": True})
    outbound: list[Outbound] = []

    async def on_outbound(o):
        outbound.append(o)

    sm = make_sm([project], store, on_outbound)
    sm.verbose_batch_size = 4
    await sm.start_all()

    client = FakeClaudeSDKClient.instances[0]
    # input=None is malformed; a missing-key input must also be tolerated.
    bad = ToolUseBlock(id="b", name="Bash", input=None)
    empty = ToolUseBlock(id="e", name="Read", input={})
    client.scripted_turns = [[
        tool_msg(bad, empty),
        assistant("Done."),
        result("s-1"),
    ]]

    await sm.deliver("qwing", "go")
    assert await _wait_for(lambda: any(o.text == "Done." for o in outbound))

    # never crashed / restarted; session still healthy
    assert sm.is_running("qwing")
    assert not any("krito" in o.text.lower() for o in outbound)

    await sm.stop_all()


async def test_verbose_session_id_and_usage_still_captured():
    project = make_project("qwing", verbose=True)
    store = FakeStore(enabled={"qwing": True})
    outbound: list[Outbound] = []

    async def on_outbound(o):
        outbound.append(o)

    sm = make_sm([project], store, on_outbound)
    sm.verbose_batch_size = 2
    await sm.start_all()

    client = FakeClaudeSDKClient.instances[0]
    client.scripted_turns = [[
        tool_msg(tool("Bash", command="make"), tool("Read", file_path="a.py")),
        assistant("Done."),
        result(
            "sess-verbose",
            usage={"input_tokens": 10, "output_tokens": 5},
            total_cost_usd=0.01,
        ),
    ]]

    await sm.deliver("qwing", "go")
    assert await _wait_for(lambda: len(store.usage_calls) >= 1)

    assert store._session_ids["qwing"] == "sess-verbose"
    call = store.usage_calls[0]
    assert call["input_tokens"] == 10
    assert call["output_tokens"] == 5

    await sm.stop_all()
