"""Conversations: one Claude session per forum sub-topic, plus live progress.

The project's own topic is a control desk — turns happen in ``<project> #N``.
"""

import asyncio

import pytest
from claude_agent_sdk import (
    AssistantMessage,
    ResultMessage,
    TextBlock,
    ToolResultBlock,
    ToolUseBlock,
    UserMessage,
)
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import voice_bridge.sessions as sessions_mod
from voice_bridge.sessions import SessionManager, format_tool
from voice_bridge.telegram_io import TelegramIO
from voice_bridge.types import Outbound

from test_sessions import (
    FakeApprovals,
    FakeClaudeSDKClient,
    FakeStore,
    make_cfg,
    make_project,
    _wait_for,
    assistant,
    result,
)
from test_topics import make_cfg as make_tg_cfg


@pytest.fixture(autouse=True)
def _patch_sdk(monkeypatch):
    FakeClaudeSDKClient.instances = []
    monkeypatch.setattr(sessions_mod, "ClaudeSDKClient", FakeClaudeSDKClient)
    yield
    FakeClaudeSDKClient.instances = []


def make_sm(projects, store, on_outbound=None, cfg=None, **hooks):
    async def sink(o):
        pass

    return SessionManager(
        projects,
        cfg or make_cfg(),
        store,
        on_outbound or sink,
        FakeApprovals(),
        **hooks,
    )


# --------------------------------------------------------------------------
# opening / closing
# --------------------------------------------------------------------------
async def test_start_all_gives_an_enabled_project_its_first_conversation():
    store = FakeStore(enabled={"qwing": True})
    sm = make_sm([make_project("qwing")], store)

    await sm.start_all()

    assert sm.running_keys() == ["qwing#1"]
    await sm.stop_all()


async def test_open_allocates_the_next_number_and_creates_its_topic():
    store = FakeStore(enabled={"qwing": True})
    opened = []

    async def on_open(key, project, ordinal):
        # The real callback records the thread it created, which is what makes
        # a second ensure-topic pass a no-op.
        opened.append((key, project, ordinal))
        await store.set_conversation_topic(key, -100, 40 + ordinal)

    sm = make_sm([make_project("qwing")], store, on_open=on_open)
    await sm.start_all()

    second = await sm.open("qwing")

    assert second == "qwing#2"
    assert opened == [("qwing#1", "qwing", 1), ("qwing#2", "qwing", 2)]
    assert sm.running_keys() == ["qwing#1", "qwing#2"]
    await sm.stop_all()


async def test_two_conversations_of_one_project_are_independent():
    store = FakeStore(enabled={"qwing": True})
    outbound: list[Outbound] = []

    async def on_outbound(o):
        outbound.append(o)

    sm = make_sm([make_project("qwing")], store, on_outbound)
    await sm.start_all()
    await sm.open("qwing")

    first, second = FakeClaudeSDKClient.instances
    await sm.deliver("qwing#1", "one")
    await sm.deliver("qwing#2", "two")
    assert await _wait_for(lambda: first.queries and second.queries)

    assert first.queries == ["one"]
    assert second.queries == ["two"]
    await sm.stop_all()


async def test_closing_a_conversation_stops_it_and_removes_its_topic():
    store = FakeStore(enabled={"qwing": True})
    closed = []

    async def on_close(key):
        closed.append(key)

    sm = make_sm([make_project("qwing")], store, on_close=on_close)
    await sm.start_all()

    assert await sm.close("qwing#1") is True

    assert closed == ["qwing#1"]
    assert sm.running_keys() == []
    assert await store.conversations("qwing") == []
    await sm.stop_all()


async def test_closing_an_unknown_conversation_is_a_no_op():
    store = FakeStore(enabled={"qwing": True})
    sm = make_sm([make_project("qwing")], store)

    assert await sm.close("qwing#9") is False


async def test_a_closed_number_is_never_handed_out_again():
    # #2 reappearing would put a new conversation under the old topic's title.
    store = FakeStore(enabled={"qwing": True})
    sm = make_sm([make_project("qwing")], store)
    await sm.start_all()
    await sm.open("qwing")
    await sm.close("qwing#2")

    assert await sm.open("qwing") == "qwing#3"
    await sm.stop_all()


async def test_a_conversation_without_a_topic_gets_one_on_start():
    # Conversations carried over from a pre-topics database have no thread, and
    # would post into General forever if only a fresh open created topics.
    store = FakeStore(enabled={"qwing": True}, conversations={"qwing#1": "sess-a"})
    opened = []

    async def on_open(key, project, ordinal):
        opened.append((key, project, ordinal))
        await store.set_conversation_topic(key, -100, 42)

    sm = make_sm([make_project("qwing")], store, on_open=on_open)

    await sm.start_all()

    assert opened == [("qwing#1", "qwing", 1)]
    await sm.stop_all()


async def test_an_existing_topic_is_not_created_twice():
    store = FakeStore(enabled={"qwing": True}, conversations={"qwing#1": "sess-a"})
    await store.set_conversation_topic("qwing#1", -100, 42)
    opened = []

    async def on_open(key, project, ordinal):
        opened.append(key)

    sm = make_sm([make_project("qwing")], store, on_open=on_open)

    await sm.start_all()

    assert opened == []
    await sm.stop_all()


async def test_restart_resumes_every_open_conversation():
    store = FakeStore(
        enabled={"qwing": True},
        conversations={"qwing#1": "sess-a", "qwing#2": "sess-b"},
    )
    sm = make_sm([make_project("qwing")], store)

    await sm.start_all()

    assert sorted(sm.running_keys()) == ["qwing#1", "qwing#2"]
    assert sorted(c.options.resume for c in FakeClaudeSDKClient.instances) == [
        "sess-a", "sess-b",
    ]
    await sm.stop_all()


async def test_attaching_a_session_open_elsewhere_forks_it():
    # Two processes writing one .jsonl silently lose each other's last messages.
    store = FakeStore(enabled={"qwing": True})
    sm = make_sm([make_project("qwing")], store)

    await sm.open("qwing", resume="uuid-1", fork=True)

    options = FakeClaudeSDKClient.instances[0].options
    assert options.resume == "uuid-1"
    assert options.fork_session is True
    await sm.stop_all()


async def test_attaching_a_dead_session_does_not_fork():
    store = FakeStore(enabled={"qwing": True})
    sm = make_sm([make_project("qwing")], store)

    await sm.open("qwing", resume="uuid-1")

    assert FakeClaudeSDKClient.instances[0].options.fork_session is False
    await sm.stop_all()


async def test_disabling_a_project_stops_all_its_conversations():
    store = FakeStore(enabled={"qwing": True, "beta": True})
    sm = make_sm(
        [make_project("qwing"), make_project("beta", cwd="/tmp/beta")], store
    )
    await sm.start_all()
    await sm.open("qwing")

    await sm.set_enabled("qwing", False)

    assert sm.running_keys() == ["beta#1"]
    # The rows survive, so enabling again restores the same conversations.
    await sm.set_enabled("qwing", True)
    assert sorted(sm.running_keys()) == ["beta#1", "qwing#1", "qwing#2"]
    await sm.stop_all()


# --------------------------------------------------------------------------
# live progress
# --------------------------------------------------------------------------
def test_format_tool_names_the_tool_and_its_argument():
    assert format_tool("Bash", {"command": "pytest -q"}) == "\U0001F527 Bash: pytest -q"
    assert format_tool("Read", {"file_path": "src/a.py"}) == "\U0001F4D6 Read: src/a.py"


def test_format_tool_falls_back_for_unknown_tools():
    # MCP tools we know nothing about still need a readable line.
    line = format_tool("mcp__bridge__notify_user", {"summary": "done"})

    assert line == "⚙️ notify_user: done"


def test_format_tool_clips_a_long_argument():
    line = format_tool("Bash", {"command": "x" * 200})

    assert len(line) <= 80
    assert line.endswith("…")


def test_format_tool_without_a_usable_argument():
    assert format_tool("TodoWrite", {"todos": []}) == "\U0001F4CB TodoWrite"


async def test_progress_streams_tools_then_ends_with_a_summary():
    store = FakeStore(enabled={"qwing": True})
    updates: list[tuple[str, str, bool]] = []

    async def on_progress(key, text, final):
        updates.append((key, text, final))

    sm = make_sm(
        [make_project("qwing")],
        store,
        cfg=make_cfg(stream_progress=True, stream_interval=0.0),
        on_progress=on_progress,
    )
    await sm.start_all()
    client = FakeClaudeSDKClient.instances[0]
    client.scripted_turns = [[
        AssistantMessage(
            content=[ToolUseBlock(id="t1", name="Bash", input={"command": "ls"})],
            model="m",
        ),
        UserMessage(content=[ToolResultBlock(tool_use_id="t1", content="ok")]),
        assistant("all done"),
        result("sess-1"),
    ]]

    await sm.deliver("qwing#1", "go")
    assert await _wait_for(lambda: any(final for _k, _t, final in updates))

    assert all(key == "qwing#1" for key, _t, _f in updates)
    tool_lines = [t for _k, t, final in updates if not final and "Bash: ls" in t]
    assert tool_lines, updates
    assert "✓" in tool_lines[-1]  # the result marked the line
    final_text = [t for _k, t, final in updates if final][-1]
    assert final_text.startswith("✅ done")
    assert "1 tools" in final_text

    await sm.stop_all()


async def test_a_failed_tool_is_marked_as_failed():
    store = FakeStore(enabled={"qwing": True})
    updates = []

    async def on_progress(key, text, final):
        updates.append(text)

    sm = make_sm(
        [make_project("qwing")],
        store,
        cfg=make_cfg(stream_progress=True, stream_interval=0.0),
        on_progress=on_progress,
    )
    await sm.start_all()
    FakeClaudeSDKClient.instances[0].scripted_turns = [[
        AssistantMessage(
            content=[ToolUseBlock(id="t1", name="Bash", input={"command": "ls"})],
            model="m",
        ),
        UserMessage(
            content=[ToolResultBlock(tool_use_id="t1", content="no", is_error=True)]
        ),
        result("s"),
    ]]

    await sm.deliver("qwing#1", "go")
    assert await _wait_for(lambda: any(t.startswith("✅") for t in updates))

    assert any("✗" in t for t in updates)
    await sm.stop_all()


async def test_progress_off_falls_back_to_the_plain_working_status():
    store = FakeStore(enabled={"qwing": True})
    outbound: list[Outbound] = []
    progress = []

    async def on_outbound(o):
        outbound.append(o)

    async def on_progress(key, text, final):
        progress.append(text)

    sm = make_sm(
        [make_project("qwing")],
        store,
        on_outbound,
        cfg=make_cfg(stream_progress=False),
        on_progress=on_progress,
    )
    await sm.start_all()

    await sm.deliver("qwing#1", "go")
    assert await _wait_for(lambda: len(outbound) >= 2)

    assert progress == []
    assert outbound[0].text == "Working."
    await sm.stop_all()


async def test_progress_is_rate_limited_between_updates():
    # Telegram will not take an edit per tool call on a busy turn.
    store = FakeStore(enabled={"qwing": True})
    updates = []

    async def on_progress(key, text, final):
        updates.append((text, final))

    sm = make_sm(
        [make_project("qwing")],
        store,
        cfg=make_cfg(stream_progress=True, stream_interval=60.0),
        on_progress=on_progress,
    )
    await sm.start_all()
    FakeClaudeSDKClient.instances[0].scripted_turns = [[
        AssistantMessage(
            content=[ToolUseBlock(id=f"t{i}", name="Bash", input={"command": f"c{i}"})],
            model="m",
        )
        for i in range(5)
    ] + [result("s")]]

    await sm.deliver("qwing#1", "go")
    assert await _wait_for(lambda: any(final for _t, final in updates))

    # The forced first update plus the final summary; the five tool calls all
    # land inside one interval.
    assert len(updates) == 2
    await sm.stop_all()


async def test_a_progress_failure_never_breaks_the_turn():
    store = FakeStore(enabled={"qwing": True})
    outbound: list[Outbound] = []

    async def on_outbound(o):
        outbound.append(o)

    async def on_progress(key, text, final):
        raise RuntimeError("telegram down")

    sm = make_sm(
        [make_project("qwing")],
        store,
        on_outbound,
        cfg=make_cfg(stream_progress=True, stream_interval=0.0),
        on_progress=on_progress,
    )
    await sm.start_all()
    FakeClaudeSDKClient.instances[0].scripted_turns = [[
        assistant("done anyway"), result("s"),
    ]]

    await sm.deliver("qwing#1", "go")
    assert await _wait_for(lambda: any("done anyway" in o.text for o in outbound))

    assert sm.is_running("qwing#1") is True
    await sm.stop_all()


async def test_is_busy_is_true_only_while_a_turn_runs():
    store = FakeStore(enabled={"qwing": True})

    async def on_progress(key, text, final):
        pass

    sm = make_sm(
        [make_project("qwing")],
        store,
        cfg=make_cfg(stream_progress=True, stream_interval=0.0),
        on_progress=on_progress,
    )
    await sm.start_all()
    assert sm.is_busy("qwing#1") is False

    FakeClaudeSDKClient.instances[0].scripted_turns = [[assistant("x"), result("s")]]
    await sm.deliver("qwing#1", "go")
    assert await _wait_for(lambda: sm.is_busy("qwing#1") is False)

    await sm.stop_all()


# --------------------------------------------------------------------------
# Telegram side: hub topic vs session sub-topic
# --------------------------------------------------------------------------
def make_io(chat_id=-100123, controls=None):
    io = TelegramIO(
        make_tg_cfg(chat_id), AsyncMock(), controls or MagicMock(), store=None
    )
    io.app = MagicMock()
    return io


def message(thread_id=None, text="hello"):
    msg = MagicMock()
    msg.message_id = 5
    msg.from_user.id = 42
    msg.message_thread_id = thread_id
    msg.text = text
    msg.reply_to_message = None
    msg.reply_text = AsyncMock()
    return msg


async def test_a_session_sub_topic_routes_to_its_conversation():
    io = make_io()
    io._conv_topics = {"paprika#2": 9}
    update = MagicMock()
    update.message = message(thread_id=9)

    await io._handle_text(update, MagicMock())

    io.on_user_message.assert_awaited_once()
    assert io.on_user_message.await_args.args[0]["project"] == "paprika#2"


async def test_the_project_topic_answers_instead_of_delivering():
    # The hub is a control desk: a stray note there must not become a turn.
    controls = MagicMock()
    controls.snapshot.return_value = [
        {"project": "paprika", "display_name": "Paprika ASR", "enabled": True,
         "mode": "safe", "voice": "alloy", "engine": "openai",
         "last_active": True, "cwd": "/w"}
    ]
    io = make_io(controls=controls)
    io._topics = {"paprika": 5}
    update = MagicMock()
    update.message = message(thread_id=5)

    await io._handle_text(update, MagicMock())

    io.on_user_message.assert_not_awaited()
    update.message.reply_text.assert_awaited_once()
    assert "Paprika ASR" in update.message.reply_text.await_args.args[0]


async def test_a_conversation_topic_wins_over_the_hub_lookup():
    io = make_io()
    io._topics = {"paprika": 5}
    io._conv_topics = {"paprika#1": 5}  # same id: the conversation must win
    update = MagicMock()
    update.message = message(thread_id=5)

    await io._handle_text(update, MagicMock())

    io.on_user_message.assert_awaited_once()


async def test_sends_go_to_the_conversation_topic_not_the_project_one():
    io = make_io()
    io._topics = {"paprika": 5}
    io._conv_topics = {"paprika#1": 9}

    assert io._dest("paprika#1") == {"chat_id": -100123, "message_thread_id": 9}
    # A conversation with no topic yet still reaches its project's.
    assert io._dest("paprika#2") == {"chat_id": -100123, "message_thread_id": 5}


async def test_sub_topics_are_named_after_the_project_and_number():
    from voice_bridge.config import ProjectConfig

    store = MagicMock()
    store.set_conversation_topic = AsyncMock()
    io = TelegramIO(
        make_tg_cfg(-100123),
        AsyncMock(),
        MagicMock(),
        store=store,
        projects=[ProjectConfig(name="paprika", cwd="/w", display_name="Paprika ASR")],
    )
    io.app = MagicMock()
    io.app.bot.create_forum_topic = AsyncMock(
        return_value=SimpleNamespace(message_thread_id=77)
    )

    thread_id = await io.open_conversation_topic("paprika#3", "paprika", 3)

    assert thread_id == 77
    io.app.bot.create_forum_topic.assert_awaited_once_with(
        chat_id=-100123, name="Paprika ASR #3"
    )
    assert io._conv_topics["paprika#3"] == 77
    store.set_conversation_topic.assert_awaited_once_with("paprika#3", -100123, 77)


async def test_a_failed_sub_topic_does_not_stop_the_open():
    io = make_io()
    io.app.bot.create_forum_topic = AsyncMock(side_effect=RuntimeError("no rights"))

    assert await io.open_conversation_topic("paprika#1", "paprika", 1) is None
    assert io._conv_topics == {}


# --------------------------------------------------------------------------
# Telegram side: the live progress message
# --------------------------------------------------------------------------
async def test_progress_sends_once_then_edits_the_same_message():
    io = make_io()
    io._conv_topics = {"paprika#1": 9}
    io.app.bot.send_message = AsyncMock(return_value=SimpleNamespace(message_id=500))
    io.app.bot.edit_message_text = AsyncMock()

    await io.send_progress("paprika#1", "⏳ 1s", False)
    await io.send_progress("paprika#1", "⏳ 2s", False)

    io.app.bot.send_message.assert_awaited_once()
    io.app.bot.edit_message_text.assert_awaited_once()
    assert io.app.bot.edit_message_text.await_args.kwargs["message_id"] == 500


async def test_the_final_update_edits_and_releases_the_message():
    io = make_io()
    io._conv_topics = {"paprika#1": 9}
    io.app.bot.send_message = AsyncMock(return_value=SimpleNamespace(message_id=500))
    io.app.bot.edit_message_text = AsyncMock()

    await io.send_progress("paprika#1", "⏳", False)
    await io.send_progress("paprika#1", "✅ done", True)

    assert "paprika#1" not in io._progress_msgs
    # No Stop button once the turn is over.
    assert io.app.bot.edit_message_text.await_args.kwargs["reply_markup"] is None


async def test_a_final_update_with_nothing_to_edit_sends_nothing():
    io = make_io()
    io.app.bot.send_message = AsyncMock()

    await io.send_progress("paprika#1", "✅ done", True)

    io.app.bot.send_message.assert_not_awaited()


async def test_the_progress_message_carries_a_stop_button():
    io = make_io()
    io.app.bot.send_message = AsyncMock(return_value=SimpleNamespace(message_id=1))

    await io.send_progress("paprika#1", "⏳", False)

    markup = io.app.bot.send_message.await_args.kwargs["reply_markup"]
    assert markup.inline_keyboard[0][0].callback_data == "pstop:paprika#1"


# --------------------------------------------------------------------------
# Controls: the session browser and transcript
# --------------------------------------------------------------------------
def make_controls(tmp_path, cwd):
    from voice_bridge.bridge import _Controls
    from test_bridge import FakeCfg, FakeProject, FakeSessions
    from test_bridge import FakeStore as BridgeStore

    cfg = FakeCfg()
    cfg.claude_projects_dir = str(tmp_path)
    cfg.claude_sessions_dir = str(tmp_path / "sessions")
    cfg.resume_limit = 8
    sessions = FakeSessions([FakeProject("qwing", cwd=cwd)])
    store = BridgeStore(enabled={"qwing": True})
    return _Controls(sessions, store, cfg, {"backend": None}), store


async def test_resume_options_lists_the_projects_own_claude_sessions(tmp_path):
    from test_claude_history import write_session, user

    write_session(tmp_path / "-w-qwing", "uuid-a", [
        {"cwd": "/w/qwing"},
        user("first thing"),
        {"type": "last-prompt", "lastPrompt": "and then this"},
    ])
    write_session(tmp_path / "-w-other", "uuid-b", [{"cwd": "/w/other"}, user("nope")])
    controls, _store = make_controls(tmp_path, "/w/qwing")
    await controls.seed()

    options = controls.resume_options("qwing")

    assert [s.uuid for s in options] == ["uuid-a"]
    assert options[0].title == "first thing"
    assert options[0].last_prompt == "and then this"


async def test_resume_options_is_empty_for_a_project_without_history(tmp_path):
    controls, _store = make_controls(tmp_path, "/w/qwing")
    await controls.seed()

    assert controls.resume_options("qwing") == []


async def test_history_reads_the_real_claude_transcript(tmp_path):
    # Not the bridge's Markdown mirror: this also has the VS Code turns.
    from test_claude_history import assistant as a_rec, write_session, user

    write_session(tmp_path / "-w-qwing", "uuid-a", [
        {"cwd": "/w/qwing"}, user("do it"), a_rec("done"),
    ])
    controls, store = make_controls(tmp_path, "/w/qwing")
    await store.add_conversation("qwing#1", "qwing", 1, session_id="uuid-a")
    await controls.seed()

    text = controls.history_text("qwing#1")

    assert "do it" in text
    assert "done" in text


async def test_project_sessions_reports_what_the_cap_left_out(tmp_path):
    from test_claude_history import write_session, user
    import os

    for i in range(5):
        path = write_session(tmp_path / "-w-qwing", f"uuid-{i}", [
            {"cwd": "/w/qwing"}, user(f"session {i}"),
        ])
        os.utime(path, (1000 + i, 1000 + i))
    controls, _store = make_controls(tmp_path, "/w/qwing")
    await controls.seed()

    sessions, total = controls.project_sessions("qwing", limit=2)

    assert total == 5
    assert [s.uuid for s in sessions] == ["uuid-4", "uuid-3"]


async def test_any_session_transcript_is_readable_not_just_open_ones(tmp_path):
    # The point of /history: sessions the bridge never opened.
    from test_claude_history import assistant as a_rec, write_session, user

    write_session(tmp_path / "-w-qwing", "uuid-old", [
        {"cwd": "/w/qwing"}, user("old question"), a_rec("old answer"),
    ])
    controls, _store = make_controls(tmp_path, "/w/qwing")
    await controls.seed()

    text = controls.session_history_text("qwing", "uuid-old")

    assert "old question" in text
    assert "old answer" in text


async def test_a_missing_session_transcript_says_so(tmp_path):
    controls, _store = make_controls(tmp_path, "/w/qwing")
    await controls.seed()

    assert "no transcript on disk" in controls.session_history_text("qwing", "nope")


async def test_the_full_transcript_file_clips_nothing(tmp_path):
    # The message view clips each turn to 600 chars; the file must not.
    from test_claude_history import assistant as a_rec, write_session, user

    long_answer = "x" * 5000
    write_session(tmp_path / "-w-qwing", "uuid-old", [
        {"cwd": "/w/qwing"},
        user("question"),
        a_rec(long_answer),
        {"type": "ai-title", "aiTitle": "Long one"},
    ])
    controls, _store = make_controls(tmp_path, "/w/qwing")
    await controls.seed()

    name, data = controls.session_transcript_file("qwing", "uuid-old")
    text = data.decode()

    assert name == "uuid-old.md"
    assert "# Long one" in text
    assert long_answer in text
    assert "## You" in text and "## Claude" in text


async def test_no_transcript_file_for_a_session_that_is_not_there(tmp_path):
    controls, _store = make_controls(tmp_path, "/w/qwing")
    await controls.seed()

    assert controls.session_transcript_file("qwing", "nope") is None


async def test_history_says_so_when_there_is_no_transcript_yet(tmp_path):
    controls, _store = make_controls(tmp_path, "/w/qwing")
    await controls.seed()

    assert "no transcript" in controls.history_text("qwing#1")


async def test_history_of_an_unknown_conversation(tmp_path):
    controls, store = make_controls(tmp_path, "/w/qwing")
    store._conversations.clear()
    await controls.seed()

    assert "unknown session" in controls.history_text("qwing#9")


async def test_conversation_rows_show_the_display_name_and_number(tmp_path):
    controls, _store = make_controls(tmp_path, "/w/qwing")
    await controls.seed()

    rows = controls.conversation_rows()

    assert [r["title"] for r in rows] == ["qwing #1"]
    assert rows[0]["running"] is True


async def test_conversation_rows_carry_claudes_own_session_name(tmp_path):
    # "qwing #2" alone says nothing about which of five sessions it is.
    from test_claude_history import write_session, user

    write_session(tmp_path / "-w-qwing", "uuid-a", [
        {"cwd": "/w/qwing"},
        user("first message"),
        {"type": "ai-title", "aiTitle": "Fix topic routing bug"},
    ])
    controls, store = make_controls(tmp_path, "/w/qwing")
    await store.add_conversation("qwing#1", "qwing", 1, session_id="uuid-a")
    await controls.seed()

    rows = controls.conversation_rows()

    assert rows[0]["session_title"] == "Fix topic routing bug"


async def test_a_session_with_no_history_yet_has_no_name(tmp_path):
    controls, _store = make_controls(tmp_path, "/w/qwing")
    await controls.seed()

    assert controls.conversation_rows()[0]["session_title"] == ""


async def test_empty_reservations_are_left_out_of_the_list(tmp_path):
    # Upgrading an old database reserves a #1 per project, disabled ones too.
    # Twelve placeholders would bury the handful of real sessions.
    controls, store = make_controls(tmp_path, "/w/qwing")
    controls._sessions.running = {"qwing#1"}
    await store.add_conversation("idle#1", "idle", 1)
    await store.add_conversation("used#1", "used", 1, session_id="uuid-old")
    await controls.seed()

    keys = [row["key"] for row in controls.conversation_rows()]

    assert "qwing#1" in keys       # running
    assert "used#1" in keys        # stopped, but it has a session to go back to
    assert "idle#1" not in keys    # never used anything


def test_the_sessions_list_shows_both_names():
    from voice_bridge.telegram_io import format_conversations

    text = format_conversations([
        {"key": "qwing#1", "title": "Qwing #1", "session_title": "Fix routing",
         "running": True, "busy": True},
        {"key": "qwing#2", "title": "Qwing #2", "session_title": "",
         "running": True, "busy": False},
        {"key": "beta#1", "title": "Beta #1", "session_title": "Old work",
         "running": False, "busy": False},
    ])

    assert "⚡ <b>Qwing #1</b>" in text
    assert "Fix routing" in text
    assert "\U0001F7E2 <b>Qwing #2</b>" in text
    assert "no name yet" in text
    assert "⚪ <b>Beta #1</b>" in text


def test_a_button_label_pairs_the_topic_with_the_session_name():
    from voice_bridge.telegram_io import conversation_label

    row = {"key": "qwing#2", "title": "Qwing #2", "session_title": "Fix topic routing"}

    assert conversation_label(row) == "Qwing #2 · Fix topic routing"


def test_a_long_topic_name_drops_the_session_name_rather_than_truncating_both():
    from voice_bridge.telegram_io import conversation_label

    row = {"key": "x#2", "title": "A very long project display name #2",
           "session_title": "Fix topic routing"}

    assert conversation_label(row) == "A very long project display name #2"


def test_a_button_label_stays_within_its_budget():
    from voice_bridge.telegram_io import conversation_label

    row = {"key": "q#1", "title": "Qwing #1", "session_title": "x" * 200}

    assert len(conversation_label(row)) <= 42


# --------------------------------------------------------------------------
# /menu wiring for the session actions
# --------------------------------------------------------------------------
def menu_controls():
    controls = MagicMock()
    controls.snapshot.return_value = [
        {"project": "qwing", "display_name": "Qwing", "enabled": True, "mode": "safe",
         "voice": "alloy", "engine": "openai", "last_active": True, "cwd": "/w"},
        {"project": "beta", "display_name": "Beta", "enabled": False, "mode": "safe",
         "voice": "alloy", "engine": "openai", "last_active": False, "cwd": "/b"},
    ]
    controls.reload_conversations = AsyncMock()
    controls.conversation_rows.return_value = [
        {"key": "qwing#1", "title": "Qwing #1", "project": "qwing", "ordinal": 1,
         "thread_id": 9, "session_id": "u1", "running": True, "busy": False},
    ]
    controls.history_text.return_value = "📜 qwing#1"
    controls.open_conversation = AsyncMock(return_value="qwing#2")
    return controls


def menu_query():
    query = MagicMock()
    query.edit_message_text = AsyncMock()
    return query


async def test_menu_new_asks_which_project_including_disabled_ones():
    # /menu is opened from General, where no topic names a project.
    io = make_io(controls=menu_controls())
    query = menu_query()

    await io._handle_menu_callback(query, "new")

    markup = query.edit_message_text.await_args.kwargs["reply_markup"]
    picks = [b.callback_data for row in markup.inline_keyboard for b in row]
    assert picks == ["cnew:0", "cnew:1", "menu:home"]


async def test_menu_resume_asks_which_project():
    io = make_io(controls=menu_controls())
    query = menu_query()

    await io._handle_menu_callback(query, "resume")

    markup = query.edit_message_text.await_args.kwargs["reply_markup"]
    assert [b.callback_data for row in markup.inline_keyboard for b in row] == [
        "cres:0", "cres:1", "menu:home",
    ]


async def test_picking_a_project_from_the_menu_opens_a_session():
    controls = menu_controls()
    io = make_io(controls=controls)
    query = menu_query()

    await io._handle_conversation_callback(query, "cnew", "0")

    controls.open_conversation.assert_awaited_once_with("qwing")
    assert "qwing#2" in query.edit_message_text.await_args.kwargs["text"]


def history_controls(sessions=None, total=None):
    controls = menu_controls()
    sessions = (
        [
            SimpleNamespace(uuid="uuid-a", title="Fix routing", live="",
                            cwd="/w", last_prompt="", mtime=0.0),
            SimpleNamespace(uuid="uuid-b", title="Old work", live="VSCode",
                            cwd="/w", last_prompt="", mtime=0.0),
        ]
        if sessions is None
        else sessions
    )
    controls.project_sessions.return_value = (
        sessions, len(sessions) if total is None else total
    )
    controls.session_history_text.return_value = "\U0001F4DC uuid-a\n\n👤 hi"
    controls.session_transcript_file.return_value = ("uuid-a.md", b"# Fix routing")
    return controls


async def test_menu_history_asks_which_project_first():
    # A project can have 75 sessions the bridge never opened; listing only its
    # own conversations hid every one of them.
    controls = history_controls()
    io = make_io(controls=controls)
    query = menu_query()

    await io._handle_menu_callback(query, "history")

    markup = query.edit_message_text.await_args.kwargs["reply_markup"]
    assert [b.callback_data for row in markup.inline_keyboard for b in row] == [
        "chp:0", "chp:1", "menu:home",
    ]


async def test_history_lists_the_projects_sessions_from_disk():
    controls = history_controls()
    io = make_io(controls=controls)
    query = menu_query()

    await io._handle_history_callback(query, "chp", "0")

    controls.project_sessions.assert_called_once_with("qwing", 20)
    markup = query.edit_message_text.await_args.kwargs["reply_markup"]
    assert [b.callback_data for row in markup.inline_keyboard for b in row] == [
        "chs:0:uuid-a", "chs:0:uuid-b", "menu:home",
    ]


async def test_a_capped_list_says_how_many_it_hid():
    # Silently showing 20 of 75 reads as "that is all of them".
    controls = history_controls(total=75)
    io = make_io(controls=controls)
    query = menu_query()

    await io._handle_history_callback(query, "chp", "0")

    assert "showing 2 of 75" in query.edit_message_text.await_args.kwargs["text"]


async def test_a_complete_list_does_not_pretend_to_be_capped():
    controls = history_controls()
    io = make_io(controls=controls)
    query = menu_query()

    await io._handle_history_callback(query, "chp", "0")

    assert "2 sessions" in query.edit_message_text.await_args.kwargs["text"]


async def test_a_project_with_no_history_says_so():
    controls = history_controls(sessions=[])
    io = make_io(controls=controls)
    query = menu_query()

    await io._handle_history_callback(query, "chp", "0")

    assert "no Claude sessions" in query.edit_message_text.await_args.kwargs["text"]


async def test_picking_a_session_shows_its_transcript():
    controls = history_controls()
    io = make_io(controls=controls)
    query = menu_query()

    await io._handle_history_callback(query, "chs", "0:uuid-a")

    controls.session_history_text.assert_called_once_with("qwing", "uuid-a")
    assert "hi" in query.edit_message_text.await_args.kwargs["text"]
    markup = query.edit_message_text.await_args.kwargs["reply_markup"]
    assert [b.callback_data for row in markup.inline_keyboard for b in row] == [
        "cfull:0:uuid-a", "chp:0",
    ]


async def test_the_full_transcript_comes_as_a_file():
    # A Telegram message can only ever hold a fragment of a long session.
    controls = history_controls()
    io = make_io(controls=controls)
    io.app.bot.send_document = AsyncMock()
    query = menu_query()

    await io._handle_history_callback(query, "cfull", "0:uuid-a")

    sent = io.app.bot.send_document.await_args.kwargs
    assert sent["filename"] == "uuid-a.md"
    assert sent["document"] == b"# Fix routing"


async def test_the_transcript_file_lands_where_the_button_was_tapped():
    # Browsing from a project's control topic must not dump the file in General.
    controls = history_controls()
    io = make_io(controls=controls)
    io.app.bot.send_document = AsyncMock()
    query = menu_query()
    query.message = MagicMock(message_thread_id=7)

    await io._handle_history_callback(query, "cfull", "0:uuid-a")

    assert io.app.bot.send_document.await_args.kwargs["message_thread_id"] == 7


async def test_a_full_transcript_that_is_gone_does_not_send_an_empty_file():
    controls = history_controls()
    controls.session_transcript_file.return_value = None
    io = make_io(controls=controls)
    io.app.bot.send_document = AsyncMock()
    query = menu_query()
    query.answer = AsyncMock()

    await io._handle_history_callback(query, "cfull", "0:uuid-a")

    io.app.bot.send_document.assert_not_awaited()
    query.answer.assert_awaited()


async def test_menu_home_goes_back():
    io = make_io(controls=menu_controls())
    query = menu_query()

    await io._handle_menu_callback(query, "home")

    markup = query.edit_message_text.await_args.kwargs["reply_markup"]
    assert markup.inline_keyboard[0][0].callback_data == "menu:new"


async def test_history_refreshes_before_reading(tmp_path):
    # The session id only lands in the store when the first turn answers; a
    # stale mirror would report "no transcript yet" for a session that has one.
    from test_claude_history import assistant as a_rec, write_session, user

    write_session(tmp_path / "-w-qwing", "uuid-a", [
        {"cwd": "/w/qwing"}, user("do it"), a_rec("done"),
    ])
    controls, store = make_controls(tmp_path, "/w/qwing")
    await controls.seed()
    # ...the session id appears only now, after the mirror was seeded.
    await store.add_conversation("qwing#1", "qwing", 1, session_id="uuid-a")

    io = make_io(controls=controls)
    io._conv_topics = {"qwing#1": 9}
    update = MagicMock()
    update.message = message(thread_id=9, text="/history")
    context = MagicMock()
    context.args = []

    await io._cmd_history(update, context)

    assert "do it" in update.message.reply_text.await_args.args[0]


# --------------------------------------------------------------------------
# Attaching an old session: card in its own topic, confirmed there
# --------------------------------------------------------------------------
def attach_controls(live="", transcript="\U0001F4DC old talk"):
    controls = menu_controls()
    controls.resume_options.return_value = [
        SimpleNamespace(uuid="uuid-a", title="Fix routing", cwd="/w/qwing",
                        live=live, last_prompt="add a test", mtime=0.0),
    ]
    controls.stage_conversation = AsyncMock(return_value="qwing#2")
    controls.attach_conversation = AsyncMock(return_value=True)
    controls.history_text.return_value = transcript
    return controls


async def test_picking_a_session_puts_its_card_in_the_new_topic():
    # The picker often sits in General; nothing about the session should be
    # left behind in a shared thread.
    controls = attach_controls()
    io = make_io(controls=controls)
    io._conv_topics = {"qwing#2": 9}
    io.app.bot.send_message = AsyncMock()
    query = menu_query()

    await io._handle_conversation_callback(query, "rs", "0:uuid-a")

    controls.stage_conversation.assert_awaited_once_with("qwing", "uuid-a")
    # nothing is started yet
    controls.attach_conversation.assert_not_awaited()
    card = io.app.bot.send_message.await_args.kwargs
    assert card["message_thread_id"] == 9
    assert "Fix routing" in card["text"]
    assert "add a test" in card["text"]
    assert "old talk" in card["text"]
    assert [b.callback_data for b in card["reply_markup"].inline_keyboard[0]] == [
        "cok:qwing#2", "cdel:qwing#2",
    ]


async def test_the_picker_message_becomes_a_pointer_to_the_topic():
    controls = attach_controls()
    io = make_io(controls=controls)
    io._conv_topics = {"qwing#2": 9}
    io.app.bot.send_message = AsyncMock()
    query = menu_query()

    await io._handle_conversation_callback(query, "rs", "0:uuid-a")

    text = query.edit_message_text.await_args.kwargs["text"]
    assert "qwing#2" in text
    markup = query.edit_message_text.await_args.kwargs["reply_markup"]
    assert markup.inline_keyboard[0][0].url.endswith("/9")


async def test_a_session_open_elsewhere_is_flagged_on_the_card():
    controls = attach_controls(live="VSCode")
    io = make_io(controls=controls)
    io._conv_topics = {"qwing#2": 9}
    io.app.bot.send_message = AsyncMock()

    await io._handle_conversation_callback(menu_query(), "rs", "0:uuid-a")

    assert "fork" in io.app.bot.send_message.await_args.kwargs["text"]


async def test_a_missing_transcript_is_left_off_the_card():
    controls = attach_controls(transcript="qwing#2: no transcript yet.")
    io = make_io(controls=controls)
    io._conv_topics = {"qwing#2": 9}
    io.app.bot.send_message = AsyncMock()

    await io._handle_conversation_callback(menu_query(), "rs", "0:uuid-a")

    assert "no transcript" not in io.app.bot.send_message.await_args.kwargs["text"]


async def test_a_session_that_vanished_stages_nothing():
    controls = attach_controls()
    controls.resume_options.return_value = []
    io = make_io(controls=controls)
    io.app.bot.send_message = AsyncMock()
    query = menu_query()

    await io._handle_conversation_callback(query, "rs", "0:uuid-a")

    controls.stage_conversation.assert_not_awaited()
    io.app.bot.send_message.assert_not_awaited()
    assert "no longer on disk" in query.edit_message_text.await_args.kwargs["text"]


async def test_confirming_in_the_topic_starts_the_session():
    controls = attach_controls()
    io = make_io(controls=controls)
    query = menu_query()

    await io._handle_conversation_callback(query, "cok", "qwing#2")

    controls.attach_conversation.assert_awaited_once_with("qwing#2")
    text = query.edit_message_text.await_args.kwargs["text"]
    assert "Attached" in text
    # No buttons left once it is live.
    assert query.edit_message_text.await_args.kwargs["reply_markup"] is None


async def test_a_refused_attach_says_so():
    controls = attach_controls()
    controls.attach_conversation = AsyncMock(return_value=False)
    io = make_io(controls=controls)
    query = menu_query()

    await io._handle_conversation_callback(query, "cok", "qwing#2")

    assert "Could not attach" in query.edit_message_text.await_args.kwargs["text"]


async def test_cancelling_closes_the_staged_conversation():
    controls = attach_controls()
    controls.close_conversation = AsyncMock(return_value=True)
    io = make_io(controls=controls)
    query = menu_query()

    await io._handle_conversation_callback(query, "cdel", "qwing#2")

    controls.close_conversation.assert_awaited_once_with("qwing#2")


# --------------------------------------------------------------------------
# Staging at the SessionManager level
# --------------------------------------------------------------------------
async def test_stage_reserves_a_topic_without_starting_anything():
    store = FakeStore(enabled={"qwing": True})
    opened = []

    async def on_open(key, project, ordinal):
        opened.append(key)
        await store.set_conversation_topic(key, -100, 42)

    sm = make_sm([make_project("qwing")], store, on_open=on_open)

    key = await sm.stage("qwing", resume="uuid-a")

    assert key == "qwing#1"
    assert opened == ["qwing#1"]
    assert FakeClaudeSDKClient.instances == []  # nothing connected yet
    assert sm.is_running("qwing#1") is False


async def test_attach_starts_a_staged_conversation_with_its_session():
    store = FakeStore(enabled={"qwing": True})
    sm = make_sm([make_project("qwing")], store)
    key = await sm.stage("qwing", resume="uuid-a")

    assert await sm.attach(key, fork=True) is True

    options = FakeClaudeSDKClient.instances[0].options
    assert options.resume == "uuid-a"
    assert options.fork_session is True
    await sm.stop_all()


async def test_attaching_twice_does_not_start_a_second_client():
    store = FakeStore(enabled={"qwing": True})
    sm = make_sm([make_project("qwing")], store)
    key = await sm.stage("qwing", resume="uuid-a")
    await sm.attach(key)

    assert await sm.attach(key) is False
    assert len(FakeClaudeSDKClient.instances) == 1
    await sm.stop_all()


async def test_attaching_a_cancelled_conversation_is_refused():
    store = FakeStore(enabled={"qwing": True})
    sm = make_sm([make_project("qwing")], store)
    key = await sm.stage("qwing", resume="uuid-a")
    await sm.close(key)

    assert await sm.attach(key) is False
    assert FakeClaudeSDKClient.instances == []


async def test_a_restored_session_forks_if_it_is_open_elsewhere(tmp_path, monkeypatch):
    # A restart resumes conversations too, and by then the session may have been
    # opened in VS Code — the fork check has to cover that path, not just the
    # one where the user picks a session from the list.
    import json

    registry = tmp_path / "sessions"
    registry.mkdir()
    (registry / "1.json").write_text(
        json.dumps({"sessionId": "uuid-a", "pid": 4242, "entrypoint": "vscode"})
    )
    monkeypatch.setattr(sessions_mod.claude_history, "_alive", lambda pid: True)

    store = FakeStore(
        enabled={"qwing": True}, conversations={"qwing#1": "uuid-a"}
    )
    sm = make_sm(
        [make_project("qwing")],
        store,
        cfg=make_cfg(claude_sessions_dir=str(registry)),
    )

    await sm.start_all()

    assert FakeClaudeSDKClient.instances[0].options.fork_session is True
    await sm.stop_all()


async def test_a_restored_session_that_is_not_open_elsewhere_is_not_forked(tmp_path):
    store = FakeStore(
        enabled={"qwing": True}, conversations={"qwing#1": "uuid-a"}
    )
    sm = make_sm(
        [make_project("qwing")],
        store,
        cfg=make_cfg(claude_sessions_dir=str(tmp_path / "empty")),
    )

    await sm.start_all()

    assert FakeClaudeSDKClient.instances[0].options.fork_session is False
    await sm.stop_all()
