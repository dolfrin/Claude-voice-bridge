"""TDD tests for voice_bridge.telegram_io — whitelist, inbound voice+text,
outbound send_update/send_question, /panel control board, slash commands,
and run()/stop() lifecycle. All telegram network I/O is mocked."""

from unittest.mock import AsyncMock, MagicMock

import pytest
from telegram.error import BadRequest

from voice_bridge.config import Config
from voice_bridge.telegram_io import (
    TelegramIO,
    build_mode_markup,
    build_panel_markup,
    build_projects_list_markup,
    build_voice_markup,
    format_projects,
    parse_callback,
)


def make_cfg(allowed_id=42):
    return Config(
        telegram_bot_token="TESTTOKEN",
        telegram_allowed_user_id=allowed_id,
        anthropic_api_key="ak",
        openai_api_key="ok",
        together_api_key="tk",
        together_tts_model="cartesia/sonic",
        together_tts_language="lt",
        tts_backend="openai",
        tts_voice="alloy",
        piper_voice_path="/opt/piper/x.onnx",
        whisper_model="large-v3",
        autonomy_mode="safe",
        approval_timeout=300,
        db_path=":memory:",
        open_vscode_on_enable=False,
        close_vscode_on_disable=False,
    )


class FakeControls:
    def __init__(self):
        self.calls = []
        self._snapshot = [
            {"project": "qwing", "enabled": True, "mode": "safe",
             "voice": "alloy", "engine": "openai", "last_active": True,
             "cwd": "/home/home/Projects/WhisperX"},
            {"project": "othersapp", "enabled": False, "mode": "full",
             "voice": "echo", "engine": "openai", "last_active": False,
             "cwd": "/home/home/Projects/othersapp"},
        ]

    async def toggle(self, project, on):
        self.calls.append(("toggle", project, on))
        for row in self._snapshot:
            if project is None or row["project"] == project:
                row["enabled"] = on

    async def select(self, project):
        self.calls.append(("select", project))
        for row in self._snapshot:
            if row["project"] == project:
                row["last_active"] = True
            else:
                row["last_active"] = False

    async def set_mode(self, project, mode):
        self.calls.append(("set_mode", project, mode))

    async def set_voice(self, project, voice):
        self.calls.append(("set_voice", project, voice))

    async def set_engine(self, name):
        self.calls.append(("set_engine", name))

    def snapshot(self):
        return self._snapshot


def make_message(*, message_id=10, user_id=42, text=None, voice=None,
                 reply_to=None):
    msg = MagicMock()
    msg.message_id = message_id
    msg.from_user = MagicMock()
    msg.from_user.id = user_id
    msg.text = text
    msg.voice = voice
    msg.reply_to_message = (
        MagicMock(message_id=reply_to) if reply_to is not None else None
    )
    return msg


# --------------------------------------------------------------------------
# inbound: whitelist + message routing
# --------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_text_message_from_allowed_user_routes_to_callback():
    received = []

    async def on_user_message(d):
        received.append(d)

    io = TelegramIO(make_cfg(), on_user_message, FakeControls())
    update = MagicMock()
    update.message = make_message(message_id=11, user_id=42,
                                  text="kaip sekasi", reply_to=7)
    update.callback_query = None

    await io._handle_text(update, MagicMock())

    assert received == [{
        "message_id": 11,
        "reply_to": 7,
        "text": "kaip sekasi",
        "is_voice": False,
        "audio": None,
    }]


@pytest.mark.asyncio
async def test_voice_message_downloads_bytes_and_marks_is_voice():
    received = []

    async def on_user_message(d):
        received.append(d)

    voice_obj = MagicMock()
    tg_file = AsyncMock()
    tg_file.download_as_bytearray = AsyncMock(return_value=bytearray(b"OGGDATA"))
    voice_obj.get_file = AsyncMock(return_value=tg_file)

    io = TelegramIO(make_cfg(), on_user_message, FakeControls())
    update = MagicMock()
    update.message = make_message(message_id=12, user_id=42, voice=voice_obj,
                                  reply_to=None)
    update.callback_query = None

    await io._handle_voice(update, MagicMock())

    assert received == [{
        "message_id": 12,
        "reply_to": None,
        "text": "",
        "is_voice": True,
        "audio": b"OGGDATA",
    }]


@pytest.mark.asyncio
async def test_non_whitelisted_user_is_ignored():
    received = []

    async def on_user_message(d):
        received.append(d)

    io = TelegramIO(make_cfg(allowed_id=42), on_user_message, FakeControls())
    update = MagicMock()
    update.message = make_message(message_id=13, user_id=999, text="hello")
    update.callback_query = None

    await io._handle_text(update, MagicMock())

    assert received == []


@pytest.mark.asyncio
async def test_non_whitelisted_voice_is_ignored_no_download():
    received = []

    async def on_user_message(d):
        received.append(d)

    voice_obj = MagicMock()
    voice_obj.get_file = AsyncMock()
    io = TelegramIO(make_cfg(allowed_id=42), on_user_message, FakeControls())
    update = MagicMock()
    update.message = make_message(message_id=14, user_id=999, voice=voice_obj)
    update.callback_query = None

    await io._handle_voice(update, MagicMock())

    assert received == []
    voice_obj.get_file.assert_not_awaited()


# --------------------------------------------------------------------------
# outbound: send_update + send_question
# --------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_send_update_sends_text_then_voice_and_returns_ids():
    io = TelegramIO(make_cfg(), AsyncMock(), FakeControls())
    bot = MagicMock()
    bot.send_message = AsyncMock(return_value=MagicMock(message_id=100))
    bot.send_voice = AsyncMock(return_value=MagicMock(message_id=101))
    io.app = MagicMock()
    io.app.bot = bot

    ids = await io.send_update(
        project="qwing", voice_label="alloy",
        text="Pushintas kodas\n---\n```py\nx=1\n```",
        voice_bytes=b"OGGVOICE",
    )

    assert ids == [100, 101]
    bot.send_message.assert_awaited_once()
    sent_text = bot.send_message.await_args.kwargs["text"]
    assert "qwing" in sent_text
    assert "Pushintas kodas" in sent_text
    bot.send_voice.assert_awaited_once()
    assert bot.send_voice.await_args.kwargs["voice"] == b"OGGVOICE"


@pytest.mark.asyncio
async def test_send_update_text_only_when_no_voice_bytes():
    io = TelegramIO(make_cfg(), AsyncMock(), FakeControls())
    bot = MagicMock()
    bot.send_message = AsyncMock(return_value=MagicMock(message_id=200))
    bot.send_voice = AsyncMock()
    io.app = MagicMock()
    io.app.bot = bot

    ids = await io.send_update(
        project="qwing", voice_label="alloy",
        text="tik tekstas", voice_bytes=None,
    )

    assert ids == [200]
    bot.send_voice.assert_not_awaited()


@pytest.mark.asyncio
async def test_send_question_returns_message_id():
    io = TelegramIO(make_cfg(), AsyncMock(), FakeControls())
    bot = MagicMock()
    bot.send_message = AsyncMock(return_value=MagicMock(message_id=300))
    io.app = MagicMock()
    io.app.bot = bot

    mid = await io.send_question("qwing", "Allow git push?")

    assert mid == 300
    sent = bot.send_message.await_args.kwargs["text"]
    assert "qwing" in sent and "Allow git push?" in sent


# --------------------------------------------------------------------------
# /panel render + callback dispatch
# --------------------------------------------------------------------------
def test_build_panel_markup_has_per_project_and_global_rows():
    snap = FakeControls().snapshot()
    markup = build_panel_markup(snap)
    kb = markup.inline_keyboard

    # two project rows + one global row
    assert len(kb) == 3
    # per-project toggle buttons use index-based callback_data
    toggle_btns = [b for row in kb for b in row
                   if b.callback_data.startswith("tog:")]
    assert {b.callback_data for b in toggle_btns} == {"tog:0", "tog:1"}
    # global row carries all-on/all-off/engine (no colon suffix)
    last = kb[-1]
    assert [b.callback_data for b in last] == ["allon", "alloff", "engine"]


def test_build_panel_markup_reflects_enabled_mode_voice_engine():
    snap = FakeControls().snapshot()
    markup = build_panel_markup(snap)
    texts = [b.text for row in markup.inline_keyboard for b in row]
    joined = " ".join(texts)
    # ON for qwing (enabled), OFF for othersapp (disabled)
    on_labels = [t for t in texts if t in ("ON", "OFF")]
    assert "ON" in on_labels and "OFF" in on_labels
    # modes / voices / engine surfaced
    assert any("safe" in t for t in texts)
    assert any("full" in t for t in texts)
    assert any("alloy" in t for t in texts)
    assert any("echo" in t for t in texts)
    assert "openai" in joined


def test_build_panel_markup_empty_snapshot():
    markup = build_panel_markup([])
    # only the global row
    assert len(markup.inline_keyboard) == 1
    assert [b.callback_data for b in markup.inline_keyboard[0]] == [
        "allon", "alloff", "engine"]


def test_format_projects_uses_status_path_and_last_active_first():
    snap = [
        {"project": "off", "enabled": False, "mode": "safe",
         "voice": "marin", "engine": "openai", "last_active": False,
         "cwd": "/home/home/Projects/off"},
        {"project": "active", "enabled": True, "mode": "ask",
         "voice": "echo", "engine": "openai", "last_active": True,
         "cwd": "/home/home/Projects/active"},
    ]

    text = format_projects(snap, show_all=True)

    assert text.index("<b>active</b>") < text.index("<b>off</b>")
    assert "\U0001F7E2 <b>active</b> \u2B50" in text
    assert "\u26AA <b>off</b>" in text
    assert "\U0001F4C1 ~/Projects/active · ask · echo · openai" in text


def test_format_projects_default_hides_inactive_projects():
    text = format_projects(FakeControls().snapshot())

    assert "<b>qwing</b>" in text
    assert "<b>othersapp</b>" not in text


def test_format_projects_all_includes_inactive_projects():
    text = format_projects(FakeControls().snapshot(), show_all=True)

    assert "<b>qwing</b>" in text
    assert "<b>othersapp</b>" in text


def test_build_projects_list_markup_uses_select_and_toggle_buttons():
    snap = FakeControls().snapshot() + [
        {"project": "third", "enabled": True, "mode": "safe",
         "voice": "alloy", "engine": "openai", "last_active": False,
         "cwd": "/home/home/Projects/third"},
    ]
    markup = build_projects_list_markup(snap, show_all=True)
    buttons = [button for row in markup.inline_keyboard for button in row]

    assert [button.callback_data for button in buttons] == [
        "sel:0", "ptgl:0", "sel:1", "ptgl:1", "sel:2", "ptgl:2",
    ]
    assert buttons[0].text == "\u270D \U0001F7E2 qwing \u2B50"
    assert buttons[2].text == "\u270D \u26AA othersapp"
    assert buttons[1].text == "ON"
    assert buttons[3].text == "OFF"
    assert [len(row) for row in markup.inline_keyboard] == [2, 2, 2]


def test_build_mode_markup_lists_explicit_modes():
    snap = FakeControls().snapshot()
    markup = build_mode_markup(snap, 0)
    texts = [b.text for row in markup.inline_keyboard for b in row]
    data = [b.callback_data for row in markup.inline_keyboard for b in row]

    assert "qwing mode" in texts
    assert "✓ safe" in texts
    assert "full" in texts
    assert "ask" in texts
    assert {"mset:0:safe", "mset:0:full", "mset:0:ask"} <= set(data)
    assert "back" in data


def test_build_voice_markup_lists_explicit_voices():
    snap = FakeControls().snapshot()
    markup = build_voice_markup(snap, 0)
    texts = [b.text for row in markup.inline_keyboard for b in row]
    data = [b.callback_data for row in markup.inline_keyboard for b in row]

    assert "qwing voice" in texts
    assert "✓ alloy" in texts
    assert "ash" in texts
    assert "echo" in texts
    assert {"vset:0:alloy", "vset:0:ash", "vset:0:echo"} <= set(data)
    assert "back" in data


def test_parse_callback_splits_action_and_index():
    assert parse_callback("tog:0") == ("tog", "0")
    assert parse_callback("mode:1") == ("mode", "1")
    assert parse_callback("mset:1:ask") == ("mset", "1:ask")
    assert parse_callback("allon") == ("allon", "")
    assert parse_callback("engine") == ("engine", "")


@pytest.mark.asyncio
async def test_callback_toggle_off_project_calls_controls():
    controls = FakeControls()  # qwing is index 0, currently enabled=True
    io = TelegramIO(make_cfg(), AsyncMock(), controls)
    query = AsyncMock()
    query.data = "tog:0"
    query.from_user = MagicMock(id=42)
    query.answer = AsyncMock()
    query.edit_message_reply_markup = AsyncMock()
    update = MagicMock()
    update.callback_query = query

    await io._handle_callback(update, MagicMock())

    assert ("toggle", "qwing", False) in controls.calls
    query.answer.assert_awaited()
    query.edit_message_reply_markup.assert_awaited_once()


@pytest.mark.asyncio
async def test_callback_ignores_not_modified_markup_error():
    controls = FakeControls()
    io = TelegramIO(make_cfg(), AsyncMock(), controls)
    query = AsyncMock()
    query.data = "tog:0"
    query.from_user = MagicMock(id=42)
    query.answer = AsyncMock()
    query.edit_message_reply_markup = AsyncMock(
        side_effect=BadRequest("Message is not modified")
    )
    update = MagicMock()
    update.callback_query = query

    await io._handle_callback(update, MagicMock())

    assert ("toggle", "qwing", False) in controls.calls
    query.answer.assert_awaited_once()


@pytest.mark.asyncio
async def test_callback_ignores_stale_callback_query():
    controls = FakeControls()
    io = TelegramIO(make_cfg(), AsyncMock(), controls)
    query = AsyncMock()
    query.data = "tog:0"
    query.from_user = MagicMock(id=42)
    query.answer = AsyncMock(side_effect=BadRequest("Query is too old"))
    query.edit_message_reply_markup = AsyncMock()
    update = MagicMock()
    update.callback_query = query

    await io._handle_callback(update, MagicMock())

    assert controls.calls == []
    query.edit_message_reply_markup.assert_not_awaited()


@pytest.mark.asyncio
async def test_callback_mode_opens_mode_picker():
    controls = FakeControls()  # qwing is index 0, mode == "safe"
    io = TelegramIO(make_cfg(), AsyncMock(), controls)
    query = AsyncMock()
    query.data = "mode:0"
    query.from_user = MagicMock(id=42)
    query.answer = AsyncMock()
    query.edit_message_reply_markup = AsyncMock()
    update = MagicMock()
    update.callback_query = query

    await io._handle_callback(update, MagicMock())

    assert controls.calls == []
    markup = query.edit_message_reply_markup.await_args.kwargs["reply_markup"]
    texts = [b.text for row in markup.inline_keyboard for b in row]
    assert "✓ safe" in texts
    assert "full" in texts


@pytest.mark.asyncio
async def test_callback_mode_choice_sets_mode():
    controls = FakeControls()
    io = TelegramIO(make_cfg(), AsyncMock(), controls)
    query = AsyncMock()
    query.data = "mset:0:ask"
    query.from_user = MagicMock(id=42)
    query.answer = AsyncMock()
    query.edit_message_reply_markup = AsyncMock()
    update = MagicMock()
    update.callback_query = query

    await io._handle_callback(update, MagicMock())

    assert ("set_mode", "qwing", "ask") in controls.calls


@pytest.mark.asyncio
async def test_callback_voice_opens_voice_picker():
    controls = FakeControls()  # qwing is index 0, voice == "alloy"
    io = TelegramIO(make_cfg(), AsyncMock(), controls)
    query = AsyncMock()
    query.data = "voice:0"
    query.from_user = MagicMock(id=42)
    query.answer = AsyncMock()
    query.edit_message_reply_markup = AsyncMock()
    update = MagicMock()
    update.callback_query = query

    await io._handle_callback(update, MagicMock())

    assert controls.calls == []
    markup = query.edit_message_reply_markup.await_args.kwargs["reply_markup"]
    texts = [b.text for row in markup.inline_keyboard for b in row]
    assert "✓ alloy" in texts
    assert "ash" in texts


@pytest.mark.asyncio
async def test_callback_voice_choice_sets_voice():
    controls = FakeControls()
    io = TelegramIO(make_cfg(), AsyncMock(), controls)
    query = AsyncMock()
    query.data = "vset:0:ash"
    query.from_user = MagicMock(id=42)
    query.answer = AsyncMock()
    query.edit_message_reply_markup = AsyncMock()
    update = MagicMock()
    update.callback_query = query

    await io._handle_callback(update, MagicMock())

    assert ("set_voice", "qwing", "ash") in controls.calls


@pytest.mark.asyncio
async def test_callback_back_returns_to_main_panel():
    controls = FakeControls()
    io = TelegramIO(make_cfg(), AsyncMock(), controls)
    query = AsyncMock()
    query.data = "back"
    query.from_user = MagicMock(id=42)
    query.answer = AsyncMock()
    query.edit_message_reply_markup = AsyncMock()
    update = MagicMock()
    update.callback_query = query

    await io._handle_callback(update, MagicMock())

    assert controls.calls == []
    markup = query.edit_message_reply_markup.await_args.kwargs["reply_markup"]
    data = [b.callback_data for row in markup.inline_keyboard for b in row]
    assert "tog:0" in data


@pytest.mark.asyncio
async def test_callback_all_on_and_engine_toggle():
    controls = FakeControls()
    io = TelegramIO(make_cfg(), AsyncMock(), controls)

    async def run_cb(data):
        q = AsyncMock()
        q.data = data
        q.from_user = MagicMock(id=42)
        q.answer = AsyncMock()
        q.edit_message_reply_markup = AsyncMock()
        upd = MagicMock()
        upd.callback_query = q
        await io._handle_callback(upd, MagicMock())

    await run_cb("allon")
    await run_cb("alloff")
    await run_cb("engine")  # current engine openai -> piper

    assert ("toggle", None, True) in controls.calls
    assert ("toggle", None, False) in controls.calls
    assert ("set_engine", "piper") in controls.calls


@pytest.mark.asyncio
async def test_callback_projects_picker_selects_and_redraws_list():
    controls = FakeControls()
    io = TelegramIO(make_cfg(), AsyncMock(), controls)
    query = AsyncMock()
    query.data = "sel:1"
    query.from_user = MagicMock(id=42)
    query.answer = AsyncMock()
    query.edit_message_text = AsyncMock()
    update = MagicMock()
    update.callback_query = query

    await io._handle_callback(update, MagicMock())

    assert ("select", "othersapp") in controls.calls
    kwargs = query.edit_message_text.await_args.kwargs
    assert kwargs["parse_mode"] == "HTML"
    assert kwargs["reply_markup"].inline_keyboard[0][0].callback_data == "sel:1"


@pytest.mark.asyncio
async def test_callback_projects_picker_toggle_redraws_list():
    controls = FakeControls()
    io = TelegramIO(make_cfg(), AsyncMock(), controls)
    query = AsyncMock()
    query.data = "ptgl:1"
    query.from_user = MagicMock(id=42)
    query.answer = AsyncMock()
    query.edit_message_text = AsyncMock()
    update = MagicMock()
    update.callback_query = query

    await io._handle_callback(update, MagicMock())

    assert ("toggle", "othersapp", True) in controls.calls
    kwargs = query.edit_message_text.await_args.kwargs
    assert kwargs["parse_mode"] == "HTML"
    assert kwargs["reply_markup"].inline_keyboard[1][1].callback_data == "ptgl:1"


@pytest.mark.asyncio
async def test_callback_from_non_whitelisted_user_is_ignored():
    controls = FakeControls()
    io = TelegramIO(make_cfg(allowed_id=42), AsyncMock(), controls)
    query = AsyncMock()
    query.data = "tog:0"
    query.from_user = MagicMock(id=999)
    query.answer = AsyncMock()
    update = MagicMock()
    update.callback_query = query

    await io._handle_callback(update, MagicMock())

    assert controls.calls == []


# --------------------------------------------------------------------------
# regression: project names containing ":" round-trip correctly
# --------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_colon_project_name_toggle_resolves_correctly():
    """A project named with colons (e.g. 'a:b:c') must round-trip through
    build_panel_markup -> callback_data -> _handle_callback and resolve to
    the correct controls.toggle('a:b:c', ...) call."""

    class ColonControls:
        def __init__(self):
            self.calls = []
            self._snapshot = [
                {"project": "a:b:c", "enabled": True, "mode": "safe",
                 "voice": "alloy", "engine": "openai", "last_active": False},
                {"project": "normal", "enabled": False, "mode": "full",
                 "voice": "echo", "engine": "openai", "last_active": False},
            ]

        async def toggle(self, project, on):
            self.calls.append(("toggle", project, on))

        async def set_mode(self, project, mode):
            self.calls.append(("set_mode", project, mode))

        async def set_voice(self, project, voice):
            self.calls.append(("set_voice", project, voice))

        async def set_engine(self, name):
            self.calls.append(("set_engine", name))

        def snapshot(self):
            return self._snapshot

    controls = ColonControls()
    snap = controls.snapshot()

    # Verify the panel encodes project 'a:b:c' (index 0) as "tog:0"
    markup = build_panel_markup(snap)
    kb = markup.inline_keyboard
    tog_btn = next(b for b in kb[0] if b.callback_data.startswith("tog:"))
    assert tog_btn.callback_data == "tog:0", (
        f"Expected 'tog:0', got {tog_btn.callback_data!r}")

    # Simulate tapping that button: _handle_callback must call toggle('a:b:c', False)
    io = TelegramIO(make_cfg(), AsyncMock(), controls)
    query = AsyncMock()
    query.data = tog_btn.callback_data  # "tog:0"
    query.from_user = MagicMock(id=42)
    query.answer = AsyncMock()
    query.edit_message_reply_markup = AsyncMock()
    update = MagicMock()
    update.callback_query = query

    await io._handle_callback(update, MagicMock())

    assert ("toggle", "a:b:c", False) in controls.calls, (
        f"Expected toggle('a:b:c', False), got {controls.calls}")


# --------------------------------------------------------------------------
# /panel command renders markup
# --------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_cmd_panel_replies_with_markup():
    controls = FakeControls()
    io = TelegramIO(make_cfg(), AsyncMock(), controls)
    msg = MagicMock()
    msg.from_user = MagicMock(id=42)
    msg.reply_text = AsyncMock()
    upd = MagicMock()
    upd.message = msg
    upd.callback_query = None

    await io._cmd_panel(upd, MagicMock())

    msg.reply_text.assert_awaited_once()
    markup = msg.reply_text.await_args.kwargs["reply_markup"]
    assert len(markup.inline_keyboard) == 3


# --------------------------------------------------------------------------
# text slash commands
# --------------------------------------------------------------------------
def make_cmd_update(text, user_id=42, message_id=50):
    msg = MagicMock()
    msg.message_id = message_id
    msg.from_user = MagicMock(id=user_id)
    msg.text = text
    msg.reply_text = AsyncMock()
    upd = MagicMock()
    upd.message = msg
    upd.callback_query = None
    return upd


def make_ctx(args):
    ctx = MagicMock()
    ctx.args = args
    return ctx


@pytest.mark.asyncio
async def test_cmd_projects_lists_snapshot():
    controls = FakeControls()
    io = TelegramIO(make_cfg(), AsyncMock(), controls)
    upd = make_cmd_update("/projects")

    await io._cmd_projects(upd, make_ctx([]))

    sent = upd.message.reply_text.await_args.args[0]
    kwargs = upd.message.reply_text.await_args.kwargs
    assert "<b>qwing</b>" in sent
    assert "<b>othersapp</b>" not in sent
    assert "safe" in sent and "alloy" in sent
    assert "~/Projects/WhisperX" in sent
    assert kwargs["parse_mode"] == "HTML"
    assert kwargs["reply_markup"].inline_keyboard[0][0].callback_data == "sel:0"
    assert kwargs["reply_markup"].inline_keyboard[0][1].callback_data == "ptgl:0"


@pytest.mark.asyncio
async def test_cmd_projects_all_lists_inactive_projects():
    controls = FakeControls()
    io = TelegramIO(make_cfg(), AsyncMock(), controls)
    upd = make_cmd_update("/projects all")

    await io._cmd_projects(upd, make_ctx(["all"]))

    sent = upd.message.reply_text.await_args.args[0]
    kwargs = upd.message.reply_text.await_args.kwargs
    assert "<b>qwing</b>" in sent and "<b>othersapp</b>" in sent
    assert kwargs["reply_markup"].inline_keyboard[1][0].callback_data == "sel:1"
    assert kwargs["reply_markup"].inline_keyboard[1][1].callback_data == "ptgl:1"


@pytest.mark.asyncio
async def test_cmd_projects_all_alias_lists_inactive_projects():
    controls = FakeControls()
    io = TelegramIO(make_cfg(), AsyncMock(), controls)
    upd = make_cmd_update("/projects_all")

    await io._cmd_projects_all(upd, make_ctx([]))

    sent = upd.message.reply_text.await_args.args[0]
    kwargs = upd.message.reply_text.await_args.kwargs
    assert "<b>qwing</b>" in sent and "<b>othersapp</b>" in sent
    assert kwargs["reply_markup"].inline_keyboard[1][0].callback_data == "sel:1"


@pytest.mark.asyncio
async def test_cmd_on_with_project_calls_toggle_true():
    controls = FakeControls()
    io = TelegramIO(make_cfg(), AsyncMock(), controls)
    upd = make_cmd_update("/on qwing")

    await io._cmd_on(upd, make_ctx(["qwing"]))

    assert ("toggle", "qwing", True) in controls.calls


@pytest.mark.asyncio
async def test_cmd_off_no_arg_is_global():
    controls = FakeControls()
    io = TelegramIO(make_cfg(), AsyncMock(), controls)
    upd = make_cmd_update("/off")

    await io._cmd_off(upd, make_ctx([]))

    assert ("toggle", None, False) in controls.calls


@pytest.mark.asyncio
async def test_cmd_mode_sets_per_project():
    controls = FakeControls()
    io = TelegramIO(make_cfg(), AsyncMock(), controls)
    upd = make_cmd_update("/mode full qwing")

    await io._cmd_mode(upd, make_ctx(["full", "qwing"]))

    assert ("set_mode", "qwing", "full") in controls.calls


@pytest.mark.asyncio
async def test_cmd_mode_rejects_invalid_mode():
    controls = FakeControls()
    io = TelegramIO(make_cfg(), AsyncMock(), controls)
    upd = make_cmd_update("/mode bogus")

    await io._cmd_mode(upd, make_ctx(["bogus"]))

    assert controls.calls == []
    upd.message.reply_text.assert_awaited()


@pytest.mark.asyncio
async def test_cmd_voice_list_replies_voices():
    controls = FakeControls()
    io = TelegramIO(make_cfg(), AsyncMock(), controls)
    upd = make_cmd_update("/voice list")

    await io._cmd_voice(upd, make_ctx(["list"]))

    sent = upd.message.reply_text.await_args.args[0]
    assert "alloy" in sent and "echo" in sent
    assert controls.calls == []  # listing must not mutate state


@pytest.mark.asyncio
async def test_cmd_voice_set_for_project():
    controls = FakeControls()
    io = TelegramIO(make_cfg(), AsyncMock(), controls)
    upd = make_cmd_update("/voice shimmer for qwing")

    await io._cmd_voice(upd, make_ctx(["shimmer", "for", "qwing"]))

    assert ("set_voice", "qwing", "shimmer") in controls.calls


@pytest.mark.asyncio
async def test_cmd_engine_switches_backend():
    controls = FakeControls()
    io = TelegramIO(make_cfg(), AsyncMock(), controls)
    upd = make_cmd_update("/engine piper")

    await io._cmd_engine(upd, make_ctx(["piper"]))

    assert ("set_engine", "piper") in controls.calls


@pytest.mark.asyncio
async def test_cmd_engine_rejects_invalid():
    controls = FakeControls()
    io = TelegramIO(make_cfg(), AsyncMock(), controls)
    upd = make_cmd_update("/engine bogus")

    await io._cmd_engine(upd, make_ctx(["bogus"]))

    assert controls.calls == []
    upd.message.reply_text.assert_awaited()


@pytest.mark.asyncio
async def test_cmd_status_routes_into_on_user_message():
    received = []

    async def on_user_message(d):
        received.append(d)

    io = TelegramIO(make_cfg(), on_user_message, FakeControls())
    upd = make_cmd_update("/status qwing", message_id=77)

    await io._cmd_status(upd, make_ctx(["qwing"]))

    assert len(received) == 1
    assert received[0]["message_id"] == 77
    assert received[0]["is_voice"] is False
    assert "qwing" in received[0]["text"]


@pytest.mark.asyncio
async def test_cmd_rejects_non_whitelisted():
    controls = FakeControls()
    io = TelegramIO(make_cfg(allowed_id=42), AsyncMock(), controls)
    upd = make_cmd_update("/on qwing", user_id=999)

    await io._cmd_on(upd, make_ctx(["qwing"]))

    assert controls.calls == []


# --------------------------------------------------------------------------
# run() / stop() lifecycle
# --------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_run_builds_application_and_registers_handlers(monkeypatch):
    import voice_bridge.telegram_io as mod

    added = []
    fake_app = MagicMock()
    fake_app.add_handler = MagicMock(side_effect=lambda h: added.append(h))
    fake_app.initialize = AsyncMock()
    fake_app.start = AsyncMock()
    fake_app.bot = MagicMock()
    fake_app.bot.set_my_commands = AsyncMock()
    fake_app.updater = MagicMock()
    fake_app.updater.start_polling = AsyncMock()

    fake_builder = MagicMock()
    fake_builder.token.return_value = fake_builder
    fake_builder.build.return_value = fake_app

    monkeypatch.setattr(
        mod.Application, "builder",
        classmethod(lambda cls: fake_builder),
    )

    io = TelegramIO(make_cfg(), AsyncMock(), FakeControls())
    await io.run()

    fake_builder.token.assert_called_once_with("TESTTOKEN")
    assert io.app is fake_app
    fake_app.initialize.assert_awaited_once()
    fake_app.bot.set_my_commands.assert_awaited_once()
    fake_app.start.assert_awaited_once()
    fake_app.updater.start_polling.assert_awaited_once()
    # at least: panel, projects, projects_all, on, off, mode, voice, engine,
    # status, callback, text msg, voice msg == 12 handlers
    assert len(added) >= 12

    cmd_names = set()
    for h in added:
        cmds = getattr(h, "commands", None)
        if cmds:
            cmd_names |= set(cmds)
    assert {"panel", "projects", "projects_all", "on", "off",
            "mode", "voice", "engine", "status"} <= cmd_names

    registered = fake_app.bot.set_my_commands.await_args.args[0]
    registered_names = {cmd.command for cmd in registered}
    assert {"panel", "projects", "projects_all", "status", "on", "off",
            "mode", "voice", "engine"} == registered_names


@pytest.mark.asyncio
async def test_run_returns_without_blocking(monkeypatch):
    """run() must return so bridge main() owns the run-forever wait (C3)."""
    import voice_bridge.telegram_io as mod

    fake_app = MagicMock()
    fake_app.add_handler = MagicMock()
    fake_app.initialize = AsyncMock()
    fake_app.start = AsyncMock()
    fake_app.bot = MagicMock()
    fake_app.bot.set_my_commands = AsyncMock()
    fake_app.updater = MagicMock()
    fake_app.updater.start_polling = AsyncMock()

    fake_builder = MagicMock()
    fake_builder.token.return_value = fake_builder
    fake_builder.build.return_value = fake_app
    monkeypatch.setattr(
        mod.Application, "builder",
        classmethod(lambda cls: fake_builder),
    )

    io = TelegramIO(make_cfg(), AsyncMock(), FakeControls())
    # Should complete promptly (does not block forever).
    await io.run()


@pytest.mark.asyncio
async def test_stop_shuts_down_application():
    io = TelegramIO(make_cfg(), AsyncMock(), FakeControls())
    fake_app = MagicMock()
    fake_app.updater = MagicMock()
    fake_app.updater.running = True
    fake_app.updater.stop = AsyncMock()
    fake_app.running = True
    fake_app.stop = AsyncMock()
    fake_app.shutdown = AsyncMock()
    io.app = fake_app

    await io.stop()

    fake_app.updater.stop.assert_awaited_once()
    fake_app.stop.assert_awaited_once()
    fake_app.shutdown.assert_awaited_once()


@pytest.mark.asyncio
async def test_stop_is_noop_when_never_run():
    io = TelegramIO(make_cfg(), AsyncMock(), FakeControls())
    # app is None; stop must not raise.
    await io.stop()
