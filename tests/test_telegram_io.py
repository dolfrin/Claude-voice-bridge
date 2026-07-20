"""TDD tests for voice_bridge.telegram_io — whitelist, inbound voice+text,
outbound send_update/send_question, /panel control board, slash commands,
and run()/stop() lifecycle. All telegram network I/O is mocked."""

import asyncio
import datetime
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest
from telegram.error import BadRequest, Forbidden, NetworkError, RetryAfter, TimedOut

from voice_bridge.config import AUTONOMY_MODES, Config, EFFORT_LEVELS, TTS_BACKENDS
from voice_bridge.telegram_io import (
    build_menu_markup,
    TelegramIO,
    _CAPTION_LIMIT,
    _chunk_text,
    _ENGINES,
    _friendly_path,
    _MODES,
    _send_with_retry,
    build_mode_markup,
    build_panel_markup,
    build_projects_list_markup,
    build_voice_markup,
    format_projects,
    parse_callback,
)
from voice_bridge.transcript import transcript_path


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
        self.recap_text = "Nieko naujo."
        self.cost_text = "qwing: 3 turai, 1000+400 tok, $0.0567\nTOTAL: 3 turai, 1000+400 tok, $0.0567"
        self.info_text = (
            "qwing: model=default (real: claude-x) · effort=high · "
            "mode=safe · voice=alloy · verbose=on\nengine: openai"
        )
        self._snapshot = [
            {"project": "qwing", "enabled": True, "mode": "safe",
             "voice": "alloy", "engine": "openai", "last_active": True,
             "cwd": "/home/home/Projects/WhisperX", "verbose": True},
            {"project": "othersapp", "enabled": False, "mode": "full",
             "voice": "echo", "engine": "openai", "last_active": False,
             "cwd": "/home/home/Projects/othersapp", "verbose": False},
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

    async def enable_and_deliver(self, project, text):
        self.calls.append(("enable_and_deliver", project, text))
        for row in self._snapshot:
            if row["project"] == project:
                row["enabled"] = True
                row["last_active"] = True
            else:
                row["last_active"] = False

    async def refresh_projects(self):
        self.calls.append(("refresh_projects",))
        self._snapshot.append(
            {"project": "fresh", "enabled": False, "mode": "safe",
             "voice": "alloy", "engine": "openai", "last_active": False,
             "cwd": "/home/home/Projects/Fresh"}
        )
        return 1

    async def create_project(self, name):
        self.calls.append(("create_project", name))
        return f"Sukurtas projektas {name} (/home/home/Projects/{name}). Siųsk užduotį — dirbsiu jame."

    async def set_mode(self, project, mode):
        self.calls.append(("set_mode", project, mode))

    async def set_effort(self, project, level):
        self.calls.append(("set_effort", project, level))

    def info(self):
        self.calls.append(("info",))
        return self.info_text

    async def set_verbose(self, project, on):
        self.calls.append(("set_verbose", project, on))

    async def set_voice(self, project, voice):
        self.calls.append(("set_voice", project, voice))

    async def set_engine(self, name):
        self.calls.append(("set_engine", name))

    async def interrupt(self, project):
        self.calls.append(("interrupt", project))
        return f"{project or 'active'}: nutraukta."

    def recap(self):
        self.calls.append(("recap",))
        return self.recap_text

    async def cost_summary(self):
        self.calls.append(("cost_summary",))
        return self.cost_text

    def snapshot(self):
        return self._snapshot


def make_message(*, message_id=10, user_id=42, text=None, voice=None,
                 reply_to=None):
    msg = MagicMock()
    msg.message_id = message_id
    msg.from_user = MagicMock()
    msg.from_user.id = user_id
    msg.text = text
    msg.caption = None
    msg.voice = voice
    msg.photo = []
    msg.document = None
    msg.audio = None
    msg.video = None
    msg.video_note = None
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
async def test_document_message_downloads_attachment_with_caption():
    received = []

    async def on_user_message(d):
        received.append(d)

    doc = MagicMock()
    doc.file_name = "report.pdf"
    doc.mime_type = "application/pdf"
    tg_file = AsyncMock()
    tg_file.download_as_bytearray = AsyncMock(return_value=bytearray(b"PDFDATA"))
    doc.get_file = AsyncMock(return_value=tg_file)

    io = TelegramIO(make_cfg(), on_user_message, FakeControls())
    update = MagicMock()
    msg = make_message(message_id=15, user_id=42, reply_to=7)
    msg.document = doc
    msg.caption = "peržiūrėk"
    update.message = msg
    update.callback_query = None

    await io._handle_attachment(update, MagicMock())

    assert received == [{
        "message_id": 15,
        "reply_to": 7,
        "text": "peržiūrėk",
        "is_voice": False,
        "audio": None,
        "attachments": [{
            "kind": "document",
            "file_name": "report.pdf",
            "mime_type": "application/pdf",
            "data": b"PDFDATA",
        }],
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
# Bug 3: a download over Telegram's ~20 MB getFile cap raises BadRequest;
# this must reply to the user instead of silently dropping the attachment
# and must not crash the handler.
# --------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_handle_voice_file_too_big_replies_and_does_not_crash():
    received = []

    async def on_user_message(d):
        received.append(d)

    voice_obj = MagicMock()
    voice_obj.get_file = AsyncMock(side_effect=BadRequest("File is too big"))

    io = TelegramIO(make_cfg(), on_user_message, FakeControls())
    msg = make_message(message_id=20, user_id=42, voice=voice_obj)
    msg.reply_text = AsyncMock()
    update = MagicMock()
    update.message = msg
    update.callback_query = None

    await io._handle_voice(update, MagicMock())

    assert received == []
    msg.reply_text.assert_awaited_once()
    reply = msg.reply_text.await_args.args[0]
    assert "per didelis" in reply


@pytest.mark.asyncio
async def test_handle_attachment_file_too_big_replies_and_does_not_crash():
    received = []

    async def on_user_message(d):
        received.append(d)

    doc = MagicMock()
    doc.file_name = "huge.bin"
    doc.mime_type = "application/octet-stream"
    doc.get_file = AsyncMock(side_effect=BadRequest("File is too big"))

    io = TelegramIO(make_cfg(), on_user_message, FakeControls())
    msg = make_message(message_id=21, user_id=42)
    msg.document = doc
    msg.reply_text = AsyncMock()
    update = MagicMock()
    update.message = msg
    update.callback_query = None

    await io._handle_attachment(update, MagicMock())

    assert received == []
    msg.reply_text.assert_awaited_once()
    reply = msg.reply_text.await_args.args[0]
    assert "per didelis" in reply


# --------------------------------------------------------------------------
# Review fix 2: only a "too big"/"too large" BadRequest is the oversize case.
# An unrelated BadRequest (e.g. an expired/invalid file_id) must NOT be
# mislabeled as "per didelis" -- it must re-raise instead.
# --------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_handle_voice_unrelated_bad_request_is_not_reported_as_too_big():
    received = []

    async def on_user_message(d):
        received.append(d)

    voice_obj = MagicMock()
    voice_obj.get_file = AsyncMock(side_effect=BadRequest("wrong file_id"))

    io = TelegramIO(make_cfg(), on_user_message, FakeControls())
    msg = make_message(message_id=22, user_id=42, voice=voice_obj)
    msg.reply_text = AsyncMock()
    update = MagicMock()
    update.message = msg
    update.callback_query = None

    with pytest.raises(BadRequest):
        await io._handle_voice(update, MagicMock())

    assert received == []
    for call in msg.reply_text.await_args_list:
        assert "per didelis" not in call.args[0]


@pytest.mark.asyncio
async def test_handle_attachment_unrelated_bad_request_is_not_reported_as_too_big():
    received = []

    async def on_user_message(d):
        received.append(d)

    doc = MagicMock()
    doc.file_name = "report.pdf"
    doc.mime_type = "application/pdf"
    doc.get_file = AsyncMock(side_effect=BadRequest("wrong file_id"))

    io = TelegramIO(make_cfg(), on_user_message, FakeControls())
    msg = make_message(message_id=23, user_id=42)
    msg.document = doc
    msg.reply_text = AsyncMock()
    update = MagicMock()
    update.message = msg
    update.callback_query = None

    with pytest.raises(BadRequest):
        await io._handle_attachment(update, MagicMock())

    assert received == []
    for call in msg.reply_text.await_args_list:
        assert "per didelis" not in call.args[0]


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
async def test_send_file_sends_document_for_unknown_suffix(tmp_path):
    path = tmp_path / "report.txt"
    path.write_text("hello")
    io = TelegramIO(make_cfg(), AsyncMock(), FakeControls())
    bot = MagicMock()
    bot.send_document = AsyncMock(return_value=MagicMock(message_id=210))
    bot.send_voice = AsyncMock()
    io.app = MagicMock()
    io.app.bot = bot

    ids = await io.send_file(
        project="qwing",
        voice_label="alloy",
        text="čia failas",
        voice_bytes=None,
        file_path=str(path),
    )

    assert ids == [210]
    bot.send_document.assert_awaited_once()
    kwargs = bot.send_document.await_args.kwargs
    assert kwargs["chat_id"] == 42
    assert kwargs["caption"] == "[qwing] čia failas"
    assert kwargs["filename"] == "report.txt"
    bot.send_voice.assert_not_awaited()


@pytest.mark.asyncio
async def test_send_file_uses_photo_method_for_images(tmp_path):
    path = tmp_path / "shot.png"
    path.write_bytes(b"PNG")
    io = TelegramIO(make_cfg(), AsyncMock(), FakeControls())
    bot = MagicMock()
    bot.send_photo = AsyncMock(return_value=MagicMock(message_id=211))
    bot.send_voice = AsyncMock(return_value=MagicMock(message_id=212))
    io.app = MagicMock()
    io.app.bot = bot

    ids = await io.send_file(
        project="qwing",
        voice_label="alloy",
        text="screenshot",
        voice_bytes=b"VOICE",
        file_path=str(path),
    )

    assert ids == [211, 212]
    bot.send_photo.assert_awaited_once()
    assert bot.send_photo.await_args.kwargs["caption"] == "[qwing] screenshot"
    bot.send_voice.assert_awaited_once()


@pytest.mark.asyncio
async def test_send_file_caption_over_limit_sends_short_caption_and_overflow_text(tmp_path):
    path = tmp_path / "report.txt"
    path.write_text("hello")
    io = TelegramIO(make_cfg(), AsyncMock(), FakeControls())
    bot = MagicMock()
    bot.send_document = AsyncMock(return_value=MagicMock(message_id=210))
    next_id = iter([300, 301, 302, 303])
    bot.send_message = AsyncMock(
        side_effect=lambda **kw: MagicMock(message_id=next(next_id))
    )
    bot.send_voice = AsyncMock()
    io.app = MagicMock()
    io.app.bot = bot

    long_text = "x" * 1500  # full caption "[qwing] " + text > 1024 caption limit

    ids = await io.send_file(
        project="qwing",
        voice_label="alloy",
        text=long_text,
        voice_bytes=None,
        file_path=str(path),
    )

    # File goes out with a SHORT caption (well under the 1024 caption cap).
    bot.send_document.assert_awaited_once()
    caption = bot.send_document.await_args.kwargs["caption"]
    assert caption == "[qwing]"
    assert len(caption) <= _CAPTION_LIMIT

    # The full text follows as a separate chunked message with the project prefix.
    assert bot.send_message.await_count >= 1
    sent_texts = [c.kwargs["text"] for c in bot.send_message.await_args_list]
    assert sent_texts[0].startswith("[qwing] ")
    assert "".join(sent_texts).count("x") == 1500

    # All message_ids returned: the document plus the overflow chunk(s).
    assert ids[0] == 210
    assert len(ids) == 1 + bot.send_message.await_count
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


@pytest.mark.asyncio
async def test_send_question_attaches_approval_buttons():
    io = TelegramIO(make_cfg(), AsyncMock(), FakeControls())
    bot = MagicMock()
    bot.send_message = AsyncMock(return_value=MagicMock(message_id=610))
    bot.send_voice = AsyncMock()
    io.app = MagicMock()
    io.app.bot = bot

    mid = await io.send_question("qwing", "run: git push", approval_token=7)

    assert mid == 610
    markup = bot.send_message.await_args.kwargs["reply_markup"]
    buttons = [b for row in markup.inline_keyboard for b in row]
    assert [b.callback_data for b in buttons] == ["apv:7:1", "apv:7:0"]
    assert "Leisti" in buttons[0].text
    assert "Neleisti" in buttons[1].text
    # no token -> no voice send in this case
    bot.send_voice.assert_not_awaited()


@pytest.mark.asyncio
async def test_send_question_sends_alert_voice_when_bytes_provided():
    io = TelegramIO(make_cfg(), AsyncMock(), FakeControls())
    bot = MagicMock()
    bot.send_message = AsyncMock(return_value=MagicMock(message_id=611))
    bot.send_voice = AsyncMock(return_value=MagicMock(message_id=612))
    io.app = MagicMock()
    io.app.bot = bot

    mid = await io.send_question(
        "qwing", "run: git push",
        approval_token=8, voice_label="shimmer", voice_bytes=b"ALERT",
    )

    # returns the TEXT message id (the id the approval future keys on)
    assert mid == 611
    bot.send_voice.assert_awaited_once()
    assert bot.send_voice.await_args.kwargs["voice"] == b"ALERT"


# --------------------------------------------------------------------------
# Bug 1: send_question truncates an oversized approval preview instead of
# silently exceeding Telegram's 4096-char hard limit (buttons must stay on
# the one message the user taps, so this truncates rather than chunks).
# --------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_send_question_truncates_huge_preview_keeps_buttons_attached():
    io = TelegramIO(make_cfg(), AsyncMock(), FakeControls())
    bot = MagicMock()
    bot.send_message = AsyncMock(return_value=MagicMock(message_id=700))
    io.app = MagicMock()
    io.app.bot = bot

    huge_preview = "qwing — approval reikalingas:\n\n" + ("x" * 6000)

    mid = await io.send_question("qwing", huge_preview, approval_token=9)

    assert mid == 700
    bot.send_message.assert_awaited_once()
    sent_text = bot.send_message.await_args.kwargs["text"]
    assert len(sent_text) <= 3500
    assert "[truncated]" in sent_text

    markup = bot.send_message.await_args.kwargs["reply_markup"]
    buttons = [b for row in markup.inline_keyboard for b in row]
    assert [b.callback_data for b in buttons] == ["apv:9:1", "apv:9:0"]


@pytest.mark.asyncio
async def test_send_question_short_preview_is_unchanged():
    io = TelegramIO(make_cfg(), AsyncMock(), FakeControls())
    bot = MagicMock()
    bot.send_message = AsyncMock(return_value=MagicMock(message_id=701))
    io.app = MagicMock()
    io.app.bot = bot

    mid = await io.send_question("qwing", "run: git push", approval_token=11)

    assert mid == 701
    sent_text = bot.send_message.await_args.kwargs["text"]
    assert sent_text == "[qwing] run: git push"
    assert "[truncated]" not in sent_text


@pytest.mark.asyncio
async def test_send_question_retries_transient_error_via_send_with_retry(monkeypatch):
    async def fake_sleep(seconds):
        return None

    monkeypatch.setattr(asyncio, "sleep", fake_sleep)

    io = TelegramIO(make_cfg(), AsyncMock(), FakeControls())
    bot = MagicMock()
    calls: list[dict] = []

    async def flaky(**kwargs):
        calls.append(kwargs)
        if len(calls) == 1:
            raise RetryAfter(retry_after=0)
        return MagicMock(message_id=702)

    bot.send_message = AsyncMock(side_effect=flaky)
    io.app = MagicMock()
    io.app.bot = bot

    mid = await io.send_question("qwing", "run: git push", approval_token=12)

    assert mid == 702
    assert len(calls) == 2


@pytest.mark.asyncio
async def test_approval_callback_resolves_and_edits_message():
    resolved: list[tuple[int, bool]] = []

    def on_approval(token, approved):
        resolved.append((token, approved))
        return True

    io = TelegramIO(make_cfg(), AsyncMock(), FakeControls(), on_approval=on_approval)
    query = AsyncMock()
    query.data = "apv:7:1"
    query.from_user = MagicMock(id=42)
    query.answer = AsyncMock()
    query.edit_message_text = AsyncMock()
    update = MagicMock()
    update.callback_query = query

    await io._handle_callback(update, MagicMock())

    assert resolved == [(7, True)]
    query.edit_message_text.assert_awaited_once()
    edited = query.edit_message_text.await_args.args[0]
    assert "Leista" in edited


@pytest.mark.asyncio
async def test_approval_callback_deny_edits_message():
    resolved: list[tuple[int, bool]] = []

    def on_approval(token, approved):
        resolved.append((token, approved))
        return True

    io = TelegramIO(make_cfg(), AsyncMock(), FakeControls(), on_approval=on_approval)
    query = AsyncMock()
    query.data = "apv:7:0"
    query.from_user = MagicMock(id=42)
    query.answer = AsyncMock()
    query.edit_message_text = AsyncMock()
    update = MagicMock()
    update.callback_query = query

    await io._handle_callback(update, MagicMock())

    assert resolved == [(7, False)]
    edited = query.edit_message_text.await_args.args[0]
    assert "Neleista" in edited


@pytest.mark.asyncio
async def test_approval_callback_stale_token_shows_toast_no_edit():
    def on_approval(token, approved):
        return False  # no live pending (already answered / timed out)

    io = TelegramIO(make_cfg(), AsyncMock(), FakeControls(), on_approval=on_approval)
    query = AsyncMock()
    query.data = "apv:99:1"
    query.from_user = MagicMock(id=42)
    query.answer = AsyncMock()
    query.edit_message_text = AsyncMock()
    update = MagicMock()
    update.callback_query = query

    await io._handle_callback(update, MagicMock())

    # a stale tap answers with a toast and does NOT edit the message
    query.answer.assert_awaited_once()
    assert query.answer.await_args.args[0]  # non-empty toast text
    query.edit_message_text.assert_not_awaited()


@pytest.mark.asyncio
async def test_approval_callback_from_non_whitelisted_user_is_ignored():
    resolved: list[tuple[int, bool]] = []

    def on_approval(token, approved):
        resolved.append((token, approved))
        return True

    io = TelegramIO(make_cfg(allowed_id=42), AsyncMock(), FakeControls(),
                    on_approval=on_approval)
    query = AsyncMock()
    query.data = "apv:7:1"
    query.from_user = MagicMock(id=999)  # not the owner
    query.answer = AsyncMock()
    query.edit_message_text = AsyncMock()
    update = MagicMock()
    update.callback_query = query

    await io._handle_callback(update, MagicMock())

    # the whitelist gates the approval callback: a non-owner tap is ignored
    assert resolved == []
    query.edit_message_text.assert_not_awaited()


@pytest.mark.asyncio
async def test_ask_user_sends_buttons_and_returns_selected_choice():
    io = TelegramIO(make_cfg(), AsyncMock(), FakeControls())
    bot = MagicMock()
    bot.send_message = AsyncMock(return_value=MagicMock(message_id=320))
    io.app = MagicMock()
    io.app.bot = bot

    task = asyncio.create_task(io.ask_user("qwing", "Rinktis?", ["A", "B"]))
    await asyncio.sleep(0)

    kwargs = bot.send_message.await_args.kwargs
    assert kwargs["text"] == "[qwing] Rinktis?"
    markup = kwargs["reply_markup"]
    assert markup.inline_keyboard[0][0].text == "A"
    assert markup.inline_keyboard[0][0].callback_data == "ask:1:0"
    assert markup.inline_keyboard[1][0].callback_data == "ask:1:1"

    query = MagicMock()
    query.from_user = MagicMock(id=42)
    query.data = "ask:1:1"
    query.answer = AsyncMock()
    query.edit_message_text = AsyncMock()
    update = MagicMock()
    update.callback_query = query

    await io._handle_callback(update, MagicMock())

    assert await task == "B"
    query.edit_message_text.assert_awaited_once_with("Selected: B")


# --------------------------------------------------------------------------
# Bug 2: ask_user / send_disabled_project_prompt must send via
# _send_with_retry and register their pending entry only AFTER a successful
# send -- registering it first (the old bug) leaks a phantom pending entry
# that can never resolve when the send itself failed.
# --------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_ask_user_retries_then_registers_pending_after_success(monkeypatch):
    # NOTE: a genuine (unpatched) sleep reference is kept for the polling
    # loop below -- _send_with_retry's own internal sleep is replaced with a
    # no-delay stub so the retry itself doesn't add real wall-clock time, but
    # the poll needs a REAL suspension point to let the created task actually
    # get scheduled and run between checks.
    real_sleep = asyncio.sleep

    async def fake_sleep(seconds):
        return None

    monkeypatch.setattr(asyncio, "sleep", fake_sleep)

    io = TelegramIO(make_cfg(), AsyncMock(), FakeControls())
    bot = MagicMock()
    attempts: list[dict] = []

    async def flaky(**kwargs):
        attempts.append(kwargs)
        if len(attempts) == 1:
            raise RetryAfter(retry_after=0)
        return MagicMock(message_id=800)

    bot.send_message = AsyncMock(side_effect=flaky)
    io.app = MagicMock()
    io.app.bot = bot

    task = asyncio.create_task(io.ask_user("qwing", "Rinktis?", ["A", "B"]))
    for _ in range(50):
        if len(attempts) >= 2 and io._pending_asks:
            break
        await real_sleep(0)

    assert len(attempts) == 2  # first attempt raised, retry succeeded
    assert len(io._pending_asks) == 1
    token = next(iter(io._pending_asks))
    future, choices = io._pending_asks[token]
    assert choices == ["A", "B"]
    future.set_result("B")

    assert await task == "B"
    assert io._pending_asks == {}


@pytest.mark.asyncio
async def test_ask_user_persistent_send_failure_leaks_no_pending_and_does_not_crash(monkeypatch):
    async def fake_sleep(seconds):
        return None

    monkeypatch.setattr(asyncio, "sleep", fake_sleep)

    io = TelegramIO(make_cfg(), AsyncMock(), FakeControls())
    bot = MagicMock()
    bot.send_message = AsyncMock(side_effect=NetworkError("down"))
    io.app = MagicMock()
    io.app.bot = bot

    result = await io.ask_user("qwing", "Rinktis?", ["A", "B"])

    assert result == ""
    assert io._pending_asks == {}


@pytest.mark.asyncio
async def test_ask_user_forbidden_send_failure_returns_cleanly_and_does_not_crash(monkeypatch):
    """Review fix 1: Forbidden ("bot was blocked by the user") is a sibling of
    NetworkError under TelegramError, not caught by the old narrow
    (BadRequest, NetworkError, RetryAfter, TimedOut) tuple -- it used to raise
    straight out of ask_user, contradicting the docstring's "never raises"
    claim. Must return the same "" sentinel as any other persistent send
    failure, with no pending entry leaked."""
    async def fake_sleep(seconds):
        return None

    monkeypatch.setattr(asyncio, "sleep", fake_sleep)

    io = TelegramIO(make_cfg(), AsyncMock(), FakeControls())
    bot = MagicMock()
    bot.send_message = AsyncMock(side_effect=Forbidden("bot was blocked by the user"))
    io.app = MagicMock()
    io.app.bot = bot

    result = await io.ask_user("qwing", "Rinktis?", ["A", "B"])

    assert result == ""
    assert io._pending_asks == {}


@pytest.mark.asyncio
async def test_send_disabled_project_prompt_forbidden_send_failure_returns_cleanly_and_does_not_crash(
    monkeypatch,
):
    """Review fix 1: same Forbidden gap as ask_user, for
    send_disabled_project_prompt's except clause."""
    async def fake_sleep(seconds):
        return None

    monkeypatch.setattr(asyncio, "sleep", fake_sleep)

    io = TelegramIO(make_cfg(), AsyncMock(), FakeControls())
    bot = MagicMock()
    bot.send_message = AsyncMock(side_effect=Forbidden("bot was blocked by the user"))
    io.app = MagicMock()
    io.app.bot = bot

    result = await io.send_disabled_project_prompt("othersapp", "go")

    assert result is None
    assert io._pending_off_sends == {}


@pytest.mark.asyncio
async def test_send_disabled_project_prompt_retries_then_registers_pending_after_success(monkeypatch):
    async def fake_sleep(seconds):
        return None

    monkeypatch.setattr(asyncio, "sleep", fake_sleep)

    io = TelegramIO(make_cfg(), AsyncMock(), FakeControls())
    bot = MagicMock()
    attempts: list[dict] = []

    async def flaky(**kwargs):
        attempts.append(kwargs)
        if len(attempts) == 1:
            raise RetryAfter(retry_after=0)
        return MagicMock(message_id=900)

    bot.send_message = AsyncMock(side_effect=flaky)
    io.app = MagicMock()
    io.app.bot = bot

    mid = await io.send_disabled_project_prompt("othersapp", "go")

    assert mid == 900
    assert len(attempts) == 2
    assert len(io._pending_off_sends) == 1
    token = next(iter(io._pending_off_sends))
    assert io._pending_off_sends[token] == ("othersapp", "go")


@pytest.mark.asyncio
async def test_send_disabled_project_prompt_persistent_failure_leaks_no_pending_and_does_not_crash(monkeypatch):
    async def fake_sleep(seconds):
        return None

    monkeypatch.setattr(asyncio, "sleep", fake_sleep)

    io = TelegramIO(make_cfg(), AsyncMock(), FakeControls())
    bot = MagicMock()
    bot.send_message = AsyncMock(side_effect=NetworkError("down"))
    io.app = MagicMock()
    io.app.bot = bot

    result = await io.send_disabled_project_prompt("othersapp", "go")

    assert result is None
    assert io._pending_off_sends == {}


# --------------------------------------------------------------------------
# _MODES / _ENGINES sourced from config.py (single source of truth)
# --------------------------------------------------------------------------
def test_modes_and_engines_mirror_config_canonical_ordered_tuples():
    assert _MODES == list(AUTONOMY_MODES)
    assert _ENGINES == list(TTS_BACKENDS)
    # Panel cycle order is preserved exactly.
    assert _MODES == ["safe", "full", "ask"]
    assert _ENGINES == ["auto", "openai", "piper", "together"]


# --------------------------------------------------------------------------
# _friendly_path (portable home-dir shortening)
# --------------------------------------------------------------------------
def test_friendly_path_shortens_real_home_prefix():
    home = str(Path.home())
    assert _friendly_path(f"{home}/Projects/x") == "~/Projects/x"


def test_friendly_path_leaves_non_home_path_unchanged():
    assert _friendly_path("/var/other/proj") == "/var/other/proj"


def test_friendly_path_uses_path_home_not_a_hardcoded_host_path(monkeypatch):
    fake_home = "/srv/someuser"
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: Path(fake_home)))
    assert _friendly_path(f"{fake_home}/proj") == "~/proj"
    assert _friendly_path("/home/home/Projects/x") == "/home/home/Projects/x"


# --------------------------------------------------------------------------
# _chunk_text
# --------------------------------------------------------------------------
def test_chunk_text_short_text_returns_single_chunk():
    text = "hello world"
    assert _chunk_text(text) == [text]


def test_chunk_text_splits_long_text_and_preserves_content():
    text = "a" * 10000
    chunks = _chunk_text(text)
    assert len(chunks) > 1
    assert all(len(c) <= 4096 for c in chunks)
    assert "".join(chunks) == text


def test_chunk_text_hard_splits_a_single_line_longer_than_limit():
    text = "b" * 5000  # single "line", no newlines to break on
    chunks = _chunk_text(text, limit=4096)
    assert len(chunks) == 2
    assert len(chunks[0]) == 4096
    assert all(len(c) <= 4096 for c in chunks)
    assert "".join(chunks) == text


def test_chunk_text_prefers_breaking_on_last_newline_before_limit():
    first = "x" * 4000
    second = "y" * 4000
    text = first + "\n" + second
    chunks = _chunk_text(text, limit=4096)
    assert all(len(c) <= 4096 for c in chunks)
    assert chunks[0] == first + "\n"
    assert "".join(chunks) == text


# --------------------------------------------------------------------------
# _send_with_retry
# --------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_send_with_retry_retries_once_after_retry_after_then_succeeds(monkeypatch):
    sleeps = []

    async def fake_sleep(seconds):
        sleeps.append(seconds)

    monkeypatch.setattr(asyncio, "sleep", fake_sleep)

    calls = []

    async def factory():
        calls.append(1)
        if len(calls) == 1:
            raise RetryAfter(retry_after=0)
        return "ok"

    result = await _send_with_retry(factory)

    assert result == "ok"
    assert len(calls) == 2
    assert len(sleeps) == 1


@pytest.mark.asyncio
async def test_send_with_retry_backs_off_exponentially_on_timed_out(monkeypatch):
    sleeps = []

    async def fake_sleep(seconds):
        sleeps.append(seconds)

    monkeypatch.setattr(asyncio, "sleep", fake_sleep)

    calls = []

    async def factory():
        calls.append(1)
        if len(calls) < 3:
            raise TimedOut()
        return "ok"

    result = await _send_with_retry(factory)

    assert result == "ok"
    assert len(calls) == 3
    assert sleeps == [0.5, 1.0]


@pytest.mark.asyncio
async def test_send_with_retry_raises_after_exhausting_attempts_on_network_error(monkeypatch):
    async def fake_sleep(seconds):
        return None

    monkeypatch.setattr(asyncio, "sleep", fake_sleep)

    calls = []

    async def factory():
        calls.append(1)
        raise NetworkError("boom")

    with pytest.raises(NetworkError):
        await _send_with_retry(factory, attempts=3)

    assert len(calls) == 3


@pytest.mark.asyncio
async def test_send_with_retry_bad_request_raises_immediately_no_retry(monkeypatch):
    async def fake_sleep(seconds):
        raise AssertionError("must not sleep on a non-transient BadRequest")

    monkeypatch.setattr(asyncio, "sleep", fake_sleep)

    calls = []

    async def factory():
        calls.append(1)
        raise BadRequest("bad entities")

    with pytest.raises(BadRequest):
        await _send_with_retry(factory)

    assert len(calls) == 1


@pytest.mark.asyncio
async def test_send_with_retry_normalizes_timedelta_retry_after(monkeypatch):
    # PTB (opt-in today, future default) returns RetryAfter.retry_after as a
    # datetime.timedelta; _send_with_retry must convert it to float seconds
    # instead of raising TypeError on timedelta + float.
    monkeypatch.setenv("PTB_TIMEDELTA", "1")
    exc = RetryAfter(retry_after=5)
    assert isinstance(exc.retry_after, datetime.timedelta)  # sanity: PTB gives timedelta

    sleeps = []

    async def fake_sleep(seconds):
        sleeps.append(seconds)

    monkeypatch.setattr(asyncio, "sleep", fake_sleep)

    calls = []

    async def factory():
        calls.append(1)
        if len(calls) == 1:
            raise exc
        return "ok"

    result = await _send_with_retry(factory)

    assert result == "ok"
    assert len(sleeps) == 1
    # 5 seconds + jitter in [0.05, 0.25], as a plain float (no TypeError).
    assert isinstance(sleeps[0], float)
    assert 5.0 < sleeps[0] < 5.3


# --------------------------------------------------------------------------
# send_update: chunking of long text
# --------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_send_update_chunks_text_over_4096_chars_into_multiple_messages():
    io = TelegramIO(make_cfg(), AsyncMock(), FakeControls())
    bot = MagicMock()
    next_id = iter([100, 101, 102, 103])
    bot.send_message = AsyncMock(
        side_effect=lambda **kw: MagicMock(message_id=next(next_id))
    )
    io.app = MagicMock()
    io.app.bot = bot

    long_text = "line\n" * 2000  # well over 4096 chars

    ids = await io.send_update(
        project="qwing", voice_label="alloy", text=long_text, voice_bytes=None,
    )

    assert bot.send_message.await_count > 1
    assert len(ids) == bot.send_message.await_count
    first_chunk = bot.send_message.await_args_list[0].kwargs["text"]
    assert first_chunk.startswith("[qwing] ")
    for call in bot.send_message.await_args_list:
        assert len(call.kwargs["text"]) <= 4096


# --------------------------------------------------------------------------
# /panel render + callback dispatch
# --------------------------------------------------------------------------
def test_build_panel_markup_has_per_project_and_global_rows():
    snap = FakeControls().snapshot()
    markup = build_panel_markup(snap)
    kb = markup.inline_keyboard

    # two project rows + all-on/off/engine row + cost/recap row
    assert len(kb) == 4
    # per-project toggle buttons use index-based callback_data
    toggle_btns = [b for row in kb for b in row
                   if b.callback_data.startswith("tog:")]
    assert {b.callback_data for b in toggle_btns} == {"tog:0", "tog:1"}
    # all-on/all-off/engine row (no colon suffix)
    engine_row = kb[-2]
    assert [b.callback_data for b in engine_row] == ["allon", "alloff", "engine"]
    # new cost/recap row
    cost_recap_row = kb[-1]
    assert [b.callback_data for b in cost_recap_row] == ["cost", "recap"]


def test_build_panel_markup_has_verbose_toggle_buttons_with_state_label():
    snap = FakeControls().snapshot()  # qwing verbose=True, othersapp verbose=False
    markup = build_panel_markup(snap)
    kb = markup.inline_keyboard

    verb_btns = {b.callback_data: b.text for row in kb for b in row
                 if b.callback_data.startswith("verb:")}
    assert verb_btns == {"verb:0": "\U0001F527✓", "verb:1": "\U0001F527·"}
    # verb button is the last button on its project row.
    assert kb[0][-1].callback_data == "verb:0"
    assert kb[1][-1].callback_data == "verb:1"


def test_build_menu_markup_has_primary_actions():
    markup = build_menu_markup()
    callbacks = [b.callback_data for row in markup.inline_keyboard for b in row]
    assert callbacks == [
        "menu:projects",
        "menu:projects_all",
        "menu:panel",
        "menu:handoff",
        "menu:stop",
        "menu:refresh",
    ]


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
    # no project rows, but the two global rows are always present.
    kb = markup.inline_keyboard
    assert len(kb) == 2
    assert [b.callback_data for b in kb[0]] == ["allon", "alloff", "engine"]
    assert [b.callback_data for b in kb[1]] == ["cost", "recap"]


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
async def test_callback_verbose_toggle_calls_controls_and_redraws():
    controls = FakeControls()  # qwing is index 0, currently verbose=True
    io = TelegramIO(make_cfg(), AsyncMock(), controls)
    query = AsyncMock()
    query.data = "verb:0"
    query.from_user = MagicMock(id=42)
    query.answer = AsyncMock()
    query.edit_message_reply_markup = AsyncMock()
    update = MagicMock()
    update.callback_query = query

    await io._handle_callback(update, MagicMock())

    assert ("set_verbose", "qwing", False) in controls.calls
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
async def test_callback_disabled_project_prompt_enables_and_sends():
    controls = FakeControls()
    io = TelegramIO(make_cfg(), AsyncMock(), controls)
    io._pending_off_sends["7"] = ("othersapp", "go")
    query = AsyncMock()
    query.data = "offsend:7"
    query.from_user = MagicMock(id=42)
    query.answer = AsyncMock()
    query.edit_message_text = AsyncMock()
    update = MagicMock()
    update.callback_query = query

    await io._handle_callback(update, MagicMock())

    assert ("enable_and_deliver", "othersapp", "go") in controls.calls
    query.edit_message_text.assert_awaited_once_with(
        "Enabled and sent to othersapp."
    )
    assert io._pending_off_sends == {}


@pytest.mark.asyncio
async def test_callback_disabled_project_prompt_can_cancel():
    controls = FakeControls()
    io = TelegramIO(make_cfg(), AsyncMock(), controls)
    io._pending_off_sends["8"] = ("othersapp", "go")
    query = AsyncMock()
    query.data = "offcancel:8"
    query.from_user = MagicMock(id=42)
    query.answer = AsyncMock()
    query.edit_message_text = AsyncMock()
    update = MagicMock()
    update.callback_query = query

    await io._handle_callback(update, MagicMock())

    assert controls.calls == []
    query.edit_message_text.assert_awaited_once_with("Cancelled: othersapp")
    assert io._pending_off_sends == {}


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


@pytest.mark.asyncio
async def test_callback_cost_replies_with_cost_summary_and_does_not_rerender():
    controls = FakeControls()
    io = TelegramIO(make_cfg(), AsyncMock(), controls)
    query = AsyncMock()
    query.data = "cost"
    query.from_user = MagicMock(id=42)
    query.answer = AsyncMock()
    query.message = AsyncMock()
    query.edit_message_reply_markup = AsyncMock()
    update = MagicMock()
    update.callback_query = query

    await io._handle_callback(update, MagicMock())

    assert ("cost_summary",) in controls.calls
    query.message.reply_text.assert_awaited_once_with(controls.cost_text)
    query.edit_message_reply_markup.assert_not_awaited()


@pytest.mark.asyncio
async def test_callback_recap_replies_with_recap_and_does_not_rerender():
    controls = FakeControls()
    io = TelegramIO(make_cfg(), AsyncMock(), controls)
    query = AsyncMock()
    query.data = "recap"
    query.from_user = MagicMock(id=42)
    query.answer = AsyncMock()
    query.message = AsyncMock()
    query.edit_message_reply_markup = AsyncMock()
    update = MagicMock()
    update.callback_query = query

    await io._handle_callback(update, MagicMock())

    assert ("recap",) in controls.calls
    query.message.reply_text.assert_awaited_once_with(controls.recap_text)
    query.edit_message_reply_markup.assert_not_awaited()


@pytest.mark.asyncio
async def test_callback_cost_from_non_whitelisted_user_is_ignored():
    controls = FakeControls()
    io = TelegramIO(make_cfg(allowed_id=42), AsyncMock(), controls)
    query = AsyncMock()
    query.data = "cost"
    query.from_user = MagicMock(id=999)
    query.answer = AsyncMock()
    query.message = AsyncMock()
    update = MagicMock()
    update.callback_query = query

    await io._handle_callback(update, MagicMock())

    assert controls.calls == []
    query.message.reply_text.assert_not_awaited()


@pytest.mark.asyncio
async def test_callback_verb_from_non_whitelisted_user_is_ignored():
    controls = FakeControls()
    io = TelegramIO(make_cfg(allowed_id=42), AsyncMock(), controls)
    query = AsyncMock()
    query.data = "verb:0"
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
    assert len(markup.inline_keyboard) == 4


@pytest.mark.asyncio
async def test_cmd_menu_replies_with_main_menu():
    controls = FakeControls()
    io = TelegramIO(make_cfg(), AsyncMock(), controls)
    upd = make_cmd_update("/menu")

    await io._cmd_menu(upd, MagicMock())

    sent = upd.message.reply_text.await_args.args[0]
    markup = upd.message.reply_text.await_args.kwargs["reply_markup"]
    assert "Alex for Claude" in sent
    assert markup.inline_keyboard[0][0].callback_data == "menu:projects"


@pytest.mark.asyncio
async def test_menu_projects_callback_edits_to_projects_list():
    controls = FakeControls()
    io = TelegramIO(make_cfg(), AsyncMock(), controls)
    query = MagicMock()
    query.from_user = MagicMock(id=42)
    query.data = "menu:projects"
    query.answer = AsyncMock()
    query.edit_message_text = AsyncMock()
    update = MagicMock()
    update.callback_query = query

    await io._handle_callback(update, MagicMock())

    kwargs = query.edit_message_text.await_args.kwargs
    assert "<b>qwing</b>" in kwargs["text"]
    assert "<b>othersapp</b>" not in kwargs["text"]
    assert kwargs["reply_markup"].inline_keyboard[0][0].callback_data == "sel:0"


@pytest.mark.asyncio
async def test_menu_stop_callback_interrupts_active_project():
    controls = FakeControls()
    io = TelegramIO(make_cfg(), AsyncMock(), controls)
    query = MagicMock()
    query.from_user = MagicMock(id=42)
    query.data = "menu:stop"
    query.answer = AsyncMock()
    query.edit_message_text = AsyncMock()
    update = MagicMock()
    update.callback_query = query

    await io._handle_callback(update, MagicMock())

    assert ("interrupt", None) in controls.calls
    kwargs = query.edit_message_text.await_args.kwargs
    assert "active: nutraukta" in kwargs["text"]
    assert kwargs["reply_markup"].inline_keyboard[2][0].callback_data == "menu:stop"


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
async def test_cmd_projects_refresh_scans_and_lists_all_projects():
    controls = FakeControls()
    io = TelegramIO(make_cfg(), AsyncMock(), controls)
    upd = make_cmd_update("/projects_refresh")

    await io._cmd_projects_refresh(upd, make_ctx([]))

    assert ("refresh_projects",) in controls.calls
    sent = upd.message.reply_text.await_args.args[0]
    kwargs = upd.message.reply_text.await_args.kwargs
    assert "New projects added: 1" in sent
    assert "<b>fresh</b>" in sent
    assert kwargs["reply_markup"].inline_keyboard[2][0].callback_data == "sel:2"


# --------------------------------------------------------------------------
# /newproject
# --------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_cmd_newproject_calls_create_project_and_replies():
    controls = FakeControls()
    io = TelegramIO(make_cfg(), AsyncMock(), controls)
    upd = make_cmd_update("/newproject foo")

    await io._cmd_newproject(upd, make_ctx(["foo"]))

    assert ("create_project", "foo") in controls.calls
    sent = upd.message.reply_text.await_args.args[0]
    assert "foo" in sent


@pytest.mark.asyncio
async def test_cmd_newproject_no_arg_shows_usage():
    controls = FakeControls()
    io = TelegramIO(make_cfg(), AsyncMock(), controls)
    upd = make_cmd_update("/newproject")

    await io._cmd_newproject(upd, make_ctx([]))

    assert controls.calls == []
    sent = upd.message.reply_text.await_args.args[0]
    assert "/newproject" in sent


@pytest.mark.asyncio
async def test_cmd_newproject_non_owner_ignored():
    controls = FakeControls()
    io = TelegramIO(make_cfg(allowed_id=42), AsyncMock(), controls)
    upd = make_cmd_update("/newproject foo", user_id=999)

    await io._cmd_newproject(upd, make_ctx(["foo"]))

    assert controls.calls == []
    upd.message.reply_text.assert_not_awaited()


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
async def test_cmd_stop_interrupts_project():
    controls = FakeControls()
    io = TelegramIO(make_cfg(), AsyncMock(), controls)
    upd = make_cmd_update("/stop qwing")

    await io._cmd_stop(upd, make_ctx(["qwing"]))

    assert ("interrupt", "qwing") in controls.calls
    sent = upd.message.reply_text.await_args.args[0]
    assert "qwing: nutraukta" in sent


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
async def test_cmd_effort_sets_per_project():
    controls = FakeControls()
    io = TelegramIO(make_cfg(), AsyncMock(), controls)
    upd = make_cmd_update("/effort high qwing")

    await io._cmd_effort(upd, make_ctx(["high", "qwing"]))

    assert ("set_effort", "qwing", "high") in controls.calls
    upd.message.reply_text.assert_awaited()


@pytest.mark.asyncio
async def test_cmd_effort_no_project_targets_all():
    controls = FakeControls()
    io = TelegramIO(make_cfg(), AsyncMock(), controls)
    upd = make_cmd_update("/effort max")

    await io._cmd_effort(upd, make_ctx(["max"]))

    assert ("set_effort", None, "max") in controls.calls


@pytest.mark.asyncio
async def test_cmd_effort_rejects_invalid_level():
    controls = FakeControls()
    io = TelegramIO(make_cfg(), AsyncMock(), controls)
    upd = make_cmd_update("/effort turbo")

    await io._cmd_effort(upd, make_ctx(["turbo"]))

    assert controls.calls == []
    upd.message.reply_text.assert_awaited()


@pytest.mark.asyncio
async def test_cmd_effort_accepts_every_level():
    for level in EFFORT_LEVELS:
        controls = FakeControls()
        io = TelegramIO(make_cfg(), AsyncMock(), controls)
        upd = make_cmd_update(f"/effort {level}")
        await io._cmd_effort(upd, make_ctx([level]))
        assert ("set_effort", None, level) in controls.calls


@pytest.mark.asyncio
async def test_cmd_effort_non_owner_ignored():
    controls = FakeControls()
    io = TelegramIO(make_cfg(allowed_id=42), AsyncMock(), controls)
    upd = make_cmd_update("/effort high qwing", user_id=999)

    await io._cmd_effort(upd, make_ctx(["high", "qwing"]))

    assert controls.calls == []
    upd.message.reply_text.assert_not_awaited()


@pytest.mark.asyncio
async def test_cmd_info_replies_with_controls_info():
    controls = FakeControls()
    io = TelegramIO(make_cfg(), AsyncMock(), controls)
    upd = make_cmd_update("/info")

    await io._cmd_info(upd, make_ctx([]))

    assert ("info",) in controls.calls
    sent = upd.message.reply_text.await_args.args[0]
    assert "model=default" in sent
    assert "real: claude-x" in sent
    assert "effort=high" in sent
    assert "verbose=on" in sent
    assert "engine: openai" in sent


@pytest.mark.asyncio
async def test_cmd_info_non_owner_ignored():
    controls = FakeControls()
    io = TelegramIO(make_cfg(allowed_id=42), AsyncMock(), controls)
    upd = make_cmd_update("/info", user_id=999)

    await io._cmd_info(upd, make_ctx([]))

    assert controls.calls == []
    upd.message.reply_text.assert_not_awaited()


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
async def test_cmd_verbose_on_sets_project():
    controls = FakeControls()
    io = TelegramIO(make_cfg(), AsyncMock(), controls)
    upd = make_cmd_update("/verbose on qwing")

    await io._cmd_verbose(upd, make_ctx(["on", "qwing"]))

    assert ("set_verbose", "qwing", True) in controls.calls
    upd.message.reply_text.assert_awaited()


@pytest.mark.asyncio
async def test_cmd_verbose_off_all_projects():
    controls = FakeControls()
    io = TelegramIO(make_cfg(), AsyncMock(), controls)
    upd = make_cmd_update("/verbose off")

    await io._cmd_verbose(upd, make_ctx(["off"]))

    assert ("set_verbose", None, False) in controls.calls


@pytest.mark.asyncio
async def test_cmd_verbose_defaults_to_on_when_no_state_arg():
    controls = FakeControls()
    io = TelegramIO(make_cfg(), AsyncMock(), controls)
    upd = make_cmd_update("/verbose")

    await io._cmd_verbose(upd, make_ctx([]))

    assert ("set_verbose", None, True) in controls.calls


@pytest.mark.asyncio
async def test_cmd_verbose_non_owner_ignored():
    controls = FakeControls()
    io = TelegramIO(make_cfg(allowed_id=42), AsyncMock(), controls)
    upd = make_cmd_update("/verbose on qwing", user_id=999)

    await io._cmd_verbose(upd, make_ctx(["on", "qwing"]))

    assert controls.calls == []
    upd.message.reply_text.assert_not_awaited()


@pytest.mark.asyncio
async def test_cmd_engine_switches_backend():
    controls = FakeControls()
    io = TelegramIO(make_cfg(), AsyncMock(), controls)
    upd = make_cmd_update("/engine auto")

    await io._cmd_engine(upd, make_ctx(["auto"]))

    assert ("set_engine", "auto") in controls.calls


@pytest.mark.asyncio
async def test_cmd_engine_rejects_invalid():
    controls = FakeControls()
    io = TelegramIO(make_cfg(), AsyncMock(), controls)
    upd = make_cmd_update("/engine bogus")

    await io._cmd_engine(upd, make_ctx(["bogus"]))

    assert controls.calls == []
    upd.message.reply_text.assert_awaited()


# --------------------------------------------------------------------------
# unknown-project guard: /on /off /stop /mode /effort /verbose /voice must
# reject a project name that isn't in controls.snapshot() instead of
# silently "succeeding" (audit finding #1).
# --------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_cmd_on_unknown_project_replies_and_skips_toggle():
    controls = FakeControls()
    io = TelegramIO(make_cfg(), AsyncMock(), controls)
    upd = make_cmd_update("/on nope")

    await io._cmd_on(upd, make_ctx(["nope"]))

    assert controls.calls == []
    sent = upd.message.reply_text.await_args.args[0]
    assert "nope" in sent
    assert "qwing" in sent and "othersapp" in sent


@pytest.mark.asyncio
async def test_cmd_off_unknown_project_replies_and_skips_toggle():
    controls = FakeControls()
    io = TelegramIO(make_cfg(), AsyncMock(), controls)
    upd = make_cmd_update("/off nope")

    await io._cmd_off(upd, make_ctx(["nope"]))

    assert controls.calls == []
    sent = upd.message.reply_text.await_args.args[0]
    assert "nope" in sent
    assert "qwing" in sent and "othersapp" in sent


@pytest.mark.asyncio
async def test_cmd_stop_unknown_project_replies_and_skips_interrupt():
    controls = FakeControls()
    io = TelegramIO(make_cfg(), AsyncMock(), controls)
    upd = make_cmd_update("/stop nope")

    await io._cmd_stop(upd, make_ctx(["nope"]))

    assert controls.calls == []
    sent = upd.message.reply_text.await_args.args[0]
    assert "nope" in sent


@pytest.mark.asyncio
async def test_cmd_mode_unknown_project_replies_and_skips_set_mode():
    controls = FakeControls()
    io = TelegramIO(make_cfg(), AsyncMock(), controls)
    upd = make_cmd_update("/mode full nope")

    await io._cmd_mode(upd, make_ctx(["full", "nope"]))

    assert controls.calls == []
    sent = upd.message.reply_text.await_args.args[0]
    assert "nope" in sent


@pytest.mark.asyncio
async def test_cmd_effort_unknown_project_replies_and_skips_set_effort():
    controls = FakeControls()
    io = TelegramIO(make_cfg(), AsyncMock(), controls)
    upd = make_cmd_update("/effort high nope")

    await io._cmd_effort(upd, make_ctx(["high", "nope"]))

    assert controls.calls == []
    sent = upd.message.reply_text.await_args.args[0]
    assert "nope" in sent


@pytest.mark.asyncio
async def test_cmd_verbose_unknown_project_replies_and_skips_set_verbose():
    controls = FakeControls()
    io = TelegramIO(make_cfg(), AsyncMock(), controls)
    upd = make_cmd_update("/verbose on nope")

    await io._cmd_verbose(upd, make_ctx(["on", "nope"]))

    assert controls.calls == []
    sent = upd.message.reply_text.await_args.args[0]
    assert "nope" in sent


@pytest.mark.asyncio
async def test_cmd_voice_unknown_project_replies_and_skips_set_voice():
    controls = FakeControls()
    io = TelegramIO(make_cfg(), AsyncMock(), controls)
    upd = make_cmd_update("/voice shimmer for nope")

    await io._cmd_voice(upd, make_ctx(["shimmer", "for", "nope"]))

    assert controls.calls == []
    sent = upd.message.reply_text.await_args.args[0]
    assert "nope" in sent


@pytest.mark.asyncio
async def test_cmd_on_valid_project_still_works():
    controls = FakeControls()
    io = TelegramIO(make_cfg(), AsyncMock(), controls)
    upd = make_cmd_update("/on qwing")

    await io._cmd_on(upd, make_ctx(["qwing"]))

    assert ("toggle", "qwing", True) in controls.calls
    sent = upd.message.reply_text.await_args.args[0]
    assert "Nežinomas" not in sent


# --------------------------------------------------------------------------
# /off must acknowledge that pending/queued work for the disabled project
# was dropped, not just silently reply "x off" (audit finding #2).
# --------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_cmd_off_valid_project_notes_dropped_turns():
    controls = FakeControls()
    io = TelegramIO(make_cfg(), AsyncMock(), controls)
    upd = make_cmd_update("/off qwing")

    await io._cmd_off(upd, make_ctx(["qwing"]))

    assert ("toggle", "qwing", False) in controls.calls
    sent = upd.message.reply_text.await_args.args[0]
    assert "qwing" in sent
    assert "atmestos" in sent or "dropped" in sent.lower()


@pytest.mark.asyncio
async def test_cmd_off_no_arg_also_notes_dropped_turns():
    controls = FakeControls()
    io = TelegramIO(make_cfg(), AsyncMock(), controls)
    upd = make_cmd_update("/off")

    await io._cmd_off(upd, make_ctx([]))

    assert ("toggle", None, False) in controls.calls
    sent = upd.message.reply_text.await_args.args[0]
    assert "atmestos" in sent or "dropped" in sent.lower()


@pytest.mark.asyncio
async def test_callback_projects_picker_toggle_off_notes_dropped_turns():
    # ptgl:0 flips qwing (index 0, currently enabled=True) OFF.
    controls = FakeControls()
    io = TelegramIO(make_cfg(), AsyncMock(), controls)
    query = AsyncMock()
    query.data = "ptgl:0"
    query.from_user = MagicMock(id=42)
    query.answer = AsyncMock()
    query.edit_message_text = AsyncMock()
    update = MagicMock()
    update.callback_query = query

    await io._handle_callback(update, MagicMock())

    assert ("toggle", "qwing", False) in controls.calls
    kwargs = query.edit_message_text.await_args.kwargs
    text = query.edit_message_text.await_args.args[0] if query.edit_message_text.await_args.args else kwargs.get("text", "")
    assert "atmestos" in text or "dropped" in text.lower()


# --------------------------------------------------------------------------
# /voice must validate the name against the engine's known voices instead
# of silently accepting anything (audit finding #3).
# --------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_cmd_voice_invalid_name_rejects_and_lists_valid_voices():
    controls = FakeControls()
    io = TelegramIO(make_cfg(), AsyncMock(), controls)
    upd = make_cmd_update("/voice totallybogusvoice")

    await io._cmd_voice(upd, make_ctx(["totallybogusvoice"]))

    assert controls.calls == []
    sent = upd.message.reply_text.await_args.args[0]
    assert "alloy" in sent  # lists valid openai voices


@pytest.mark.asyncio
async def test_cmd_voice_invalid_name_for_project_rejects():
    controls = FakeControls()
    io = TelegramIO(make_cfg(), AsyncMock(), controls)
    upd = make_cmd_update("/voice totallybogusvoice for qwing")

    await io._cmd_voice(upd, make_ctx(["totallybogusvoice", "for", "qwing"]))

    assert controls.calls == []
    sent = upd.message.reply_text.await_args.args[0]
    assert "qwing" in sent or "alloy" in sent


@pytest.mark.asyncio
async def test_cmd_voice_valid_name_still_sets_voice():
    controls = FakeControls()
    io = TelegramIO(make_cfg(), AsyncMock(), controls)
    upd = make_cmd_update("/voice shimmer")

    await io._cmd_voice(upd, make_ctx(["shimmer"]))

    assert ("set_voice", None, "shimmer") in controls.calls


@pytest.mark.asyncio
async def test_cmd_voice_valid_name_for_project_still_works():
    controls = FakeControls()
    io = TelegramIO(make_cfg(), AsyncMock(), controls)
    upd = make_cmd_update("/voice shimmer for qwing")

    await io._cmd_voice(upd, make_ctx(["shimmer", "for", "qwing"]))

    assert ("set_voice", "qwing", "shimmer") in controls.calls


@pytest.mark.asyncio
async def test_cmd_voice_list_still_unchanged():
    controls = FakeControls()
    io = TelegramIO(make_cfg(), AsyncMock(), controls)
    upd = make_cmd_update("/voice list")

    await io._cmd_voice(upd, make_ctx(["list"]))

    sent = upd.message.reply_text.await_args.args[0]
    assert "alloy" in sent and "echo" in sent
    assert controls.calls == []


@pytest.mark.asyncio
async def test_cmd_voice_accepts_piper_voice_when_engine_is_auto():
    # "auto" only advertises the OpenAI voice list via available_voices(),
    # but AutoTTS can fall back to piper/together at runtime, so validation
    # for an "auto" project must accept the union of concrete backends.
    controls = FakeControls()
    controls._snapshot[0]["engine"] = "auto"
    io = TelegramIO(make_cfg(), AsyncMock(), controls)
    upd = make_cmd_update("/voice default for qwing")

    await io._cmd_voice(upd, make_ctx(["default", "for", "qwing"]))

    assert ("set_voice", "qwing", "default") in controls.calls


@pytest.mark.asyncio
async def test_cmd_handoff_replies_with_active_project_transcript(tmp_path):
    controls = FakeControls()
    controls._snapshot[0]["cwd"] = str(tmp_path)
    path = transcript_path(str(tmp_path))
    path.parent.mkdir(parents=True)
    path.write_text("## user\nlabas\n\n## assistant\npadariau", encoding="utf-8")
    io = TelegramIO(make_cfg(), AsyncMock(), controls)
    upd = make_cmd_update("/handoff")

    await io._cmd_handoff(upd, make_ctx([]))

    sent = upd.message.reply_text.await_args.args[0]
    assert "qwing handoff" in sent
    assert "voice-bridge-chat.md" in sent
    assert "labas" in sent
    assert "padariau" in sent


@pytest.mark.asyncio
async def test_cmd_handoff_unknown_project_replies_help():
    controls = FakeControls()
    io = TelegramIO(make_cfg(), AsyncMock(), controls)
    upd = make_cmd_update("/handoff nope")

    await io._cmd_handoff(upd, make_ctx(["nope"]))

    sent = upd.message.reply_text.await_args.args[0]
    assert "Project not found" in sent


@pytest.mark.asyncio
async def test_format_handoff_escapes_transcript_html(tmp_path):
    # Transcripts routinely contain <, >, & (code). The handoff /panel button
    # edits with parse_mode=HTML, so raw markup would raise BadRequest and the
    # button would silently do nothing. The dynamic content must be escaped.
    controls = FakeControls()
    controls._snapshot[0]["cwd"] = str(tmp_path)
    path = transcript_path(str(tmp_path))
    path.parent.mkdir(parents=True)
    path.write_text('## user\n<div> & "x"\n', encoding="utf-8")
    io = TelegramIO(make_cfg(), AsyncMock(), controls)

    text = io._format_handoff_text("")

    # No raw markup metacharacters survive from the file content.
    assert "<div>" not in text
    assert "<" not in text
    assert ">" not in text
    # The content is HTML-escaped so a parse_mode=HTML edit would not raise.
    assert "&lt;div&gt;" in text
    assert "&amp;" in text


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


@pytest.mark.asyncio
async def test_cmd_recap_replies_with_controls_recap():
    controls = FakeControls()
    controls.recap_text = "qwing — 2 atnaujinimai per 1 min: Done."
    io = TelegramIO(make_cfg(), AsyncMock(), controls)
    upd = make_cmd_update("/recap")

    await io._cmd_recap(upd, make_ctx([]))

    assert ("recap",) in controls.calls
    sent = upd.message.reply_text.await_args.args[0]
    assert sent == "qwing — 2 atnaujinimai per 1 min: Done."


@pytest.mark.asyncio
async def test_cmd_recap_rejects_non_whitelisted():
    controls = FakeControls()
    io = TelegramIO(make_cfg(allowed_id=42), AsyncMock(), controls)
    upd = make_cmd_update("/recap", user_id=999)

    await io._cmd_recap(upd, make_ctx([]))

    assert controls.calls == []
    upd.message.reply_text.assert_not_awaited()


@pytest.mark.asyncio
async def test_cmd_cost_replies_with_controls_cost_summary():
    controls = FakeControls()
    controls.cost_text = "qwing: 3 turai, 1000+400 tok, $0.0567"
    io = TelegramIO(make_cfg(), AsyncMock(), controls)
    upd = make_cmd_update("/cost")

    await io._cmd_cost(upd, make_ctx([]))

    assert ("cost_summary",) in controls.calls
    sent = upd.message.reply_text.await_args.args[0]
    assert sent == "qwing: 3 turai, 1000+400 tok, $0.0567"


@pytest.mark.asyncio
async def test_cmd_cost_shows_tokens_and_cost_unavailable_note():
    controls = FakeControls()
    controls.cost_text = (
        "qwing: 2 turai, 500+200 tok, $0.0000\n"
        "TOTAL: 2 turai, 500+200 tok (cost n/a — subscription auth?)"
    )
    io = TelegramIO(make_cfg(), AsyncMock(), controls)
    upd = make_cmd_update("/cost")

    await io._cmd_cost(upd, make_ctx([]))

    sent = upd.message.reply_text.await_args.args[0]
    assert "500+200 tok" in sent
    assert "n/a" in sent.lower()


@pytest.mark.asyncio
async def test_cmd_cost_rejects_non_whitelisted():
    controls = FakeControls()
    io = TelegramIO(make_cfg(allowed_id=42), AsyncMock(), controls)
    upd = make_cmd_update("/cost", user_id=999)

    await io._cmd_cost(upd, make_ctx([]))

    assert controls.calls == []
    upd.message.reply_text.assert_not_awaited()


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
    # at least: menu, panel, projects, projects_all, projects_refresh,
    # newproject, handoff, on, off, stop, mode, effort, voice, verbose,
    # engine, status, recap, cost, info, callback, text, voice, attachments.
    assert len(added) >= 22

    cmd_names = set()
    for h in added:
        cmds = getattr(h, "commands", None)
        if cmds:
            cmd_names |= set(cmds)
    assert {"menu", "panel", "projects", "projects_all", "projects_refresh", "newproject", "handoff", "on", "off", "stop",
            "mode", "effort", "voice", "engine", "status", "recap", "cost", "info"} <= cmd_names

    registered = fake_app.bot.set_my_commands.await_args.args[0]
    registered_names = {cmd.command for cmd in registered}
    assert {"menu", "panel", "projects", "projects_all", "projects_refresh", "newproject", "handoff", "status", "on", "off", "stop",
            "mode", "effort", "voice", "verbose", "engine", "recap", "cost", "info"} == registered_names


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
