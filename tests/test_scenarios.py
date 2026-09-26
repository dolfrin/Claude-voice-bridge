"""End-to-end routing scenarios, the way they went wrong on 2026-09-26.

Two editor sessions open in ONE project, a bridge restart, "finished"
notices from the hooks, replies and plain messages -- through the real
TelegramIO and the real inbound routing, against a private fake home (no
real ~/.claude file is touched, no real Telegram message is sent).
"""

import json
import os
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

import voice_bridge.live as live_mod
from voice_bridge import sent_log
from voice_bridge.bridge import make_inbound
from voice_bridge.config import Config
from voice_bridge.telegram_io import TelegramIO

A = "aaaaaaaa-0000-0000-0000-000000000001"  # the conversation the user talks in
B = "bbbbbbbb-0000-0000-0000-000000000002"  # a second, untitled one in the same project


@pytest.fixture
def world(tmp_path, monkeypatch):
    home = tmp_path / "home"
    project = home / "Projects" / "claude-voice-bridge"
    project.mkdir(parents=True)
    sessions = home / ".claude" / "sessions"
    sessions.mkdir(parents=True)
    transcripts = home / ".claude" / "projects" / "-p"
    transcripts.mkdir(parents=True)
    for pid, sid, title in ((101, A, "Valdyti Claude balsui"), (102, B, "")):
        (sessions / f"{pid}.json").write_text(json.dumps({
            "pid": pid, "sessionId": sid, "cwd": str(project), "entrypoint": "claude-vscode",
            "messagingSocketPath": f"/sock/{pid}", "status": "idle", "startedAt": 1,
        }))
        lines = [{"type": "ai-title", "aiTitle": title}] if title else []
        (transcripts / f"{sid}.jsonl").write_text("".join(json.dumps(x) + "\n" for x in lines))
    monkeypatch.setattr(os.path, "expanduser", lambda p: p.replace("~", str(home)))
    monkeypatch.setattr("pathlib.Path.home", classmethod(lambda cls: home))
    monkeypatch.setattr(live_mod, "SESSIONS_DIR", sessions)
    monkeypatch.setattr(live_mod, "_alive", lambda pid: True)
    delivered = []

    async def fake_send(socket_path, text, from_name="telegram"):
        delivered.append((socket_path, text))

    monkeypatch.setattr(live_mod, "send", fake_send)
    return SimpleNamespace(home=home, project=project, delivered=delivered)


def _bridge(world, tmp_path):
    cfg = Config(
        telegram_bot_token="t", telegram_allowed_user_id=42, anthropic_api_key="",
        openai_api_key="", together_api_key="", together_tts_model="", together_tts_language="",
        tts_backend="openai", tts_voice="alloy", piper_voice_path="", whisper_model="x",
        autonomy_mode="safe", approval_timeout=60, db_path=str(tmp_path / "state.db"),
    )
    row = {"project": "bridge", "display_name": "Valdyti Claude balsui", "cwd": str(world.project),
           "enabled": True, "last_active": True, "mode": "safe", "voice": "alloy", "engine": "openai"}
    controls = MagicMock()
    controls.snapshot = lambda: [row]
    controls.select = AsyncMock()
    io = TelegramIO(cfg, AsyncMock(), controls)
    io.app = MagicMock()
    io.app.bot = AsyncMock()
    io.app.bot.send_message.return_value = MagicMock(message_id=900)
    io._tail_live = AsyncMock()  # no transcript tailing in a test
    store = MagicMock()
    store.project_for_message = AsyncMock(return_value=None)
    store.is_enabled = AsyncMock(return_value=True)
    store.get_last_active = AsyncMock(return_value="bridge")
    store.set_last_active = AsyncMock()
    sessions = MagicMock()
    sessions.names = lambda: ["bridge"]
    sessions.project = lambda name: SimpleNamespace(cwd=str(world.project), display_name="Valdyti Claude balsui")
    sessions.deliver = AsyncMock()
    inbound = make_inbound(MagicMock(), store, MagicMock(has_pending=lambda rid: False), sessions, io, controls)
    return io, inbound, sessions


def _msg(text, reply_to=None):
    return {"message_id": 1, "reply_to": reply_to, "text": text, "is_voice": False, "audio": None}


@pytest.mark.asyncio
async def test_two_sessions_one_project_restart_notices_replies(world, tmp_path, monkeypatch):
    import voice_bridge.telegram_io as tio

    monkeypatch.setattr(tio.asyncio, "sleep", AsyncMock())
    io, inbound, sessions = _bridge(world, tmp_path)
    io.live_marker().write_text(A)  # the user was talking to A before the restart

    # 1. Restart: the bridge re-joins A, even if the registry is caught mid-write once.
    real_list = live_mod.list_sessions
    torn = [True]  # the first read catches the registry mid-write

    def listing(*a, **k):
        if torn.pop() if torn else False:
            return []
        return real_list(*a, **k)

    monkeypatch.setattr(live_mod, "list_sessions", listing)
    await io._restore_live()
    assert io.live_target().session_id == A
    assert io.live_marker().read_text() == A

    # 2. The two sessions never look alike.
    a_label = io.session_label(io.live_target())
    b_label = io.session_label(next(s for s in real_list() if s.session_id == B))
    assert a_label.startswith("„Valdyti Claude balsui“") and "naujas pokalbis" in b_label
    assert a_label != b_label

    # 3. "Finished" notices: 🎯 only under the OTHER session's notice.
    sent_log.record(10, A, str(world.project))
    sent_log.record(11, B, str(world.project))
    await io._add_hook_buttons(0.0, world.home / ".claude" / "projects")
    edited = [c.kwargs["message_id"] for c in io.app.bot.edit_message_reply_markup.await_args_list]
    assert edited == [11]

    # 4. A plain message goes to A (current) -- not to B, which wrote last.
    await inbound(_msg("labas"))
    assert world.delivered[-1] == ("/sock/101", "labas")

    # 5. A reply to B's notice goes to B, and B becomes current and pinned.
    await inbound(_msg("tau", reply_to=11))
    assert world.delivered[-1] == ("/sock/102", "tau")
    assert io.live_target().session_id == B
    pinned = io.app.bot.edit_message_text.await_args_list or io.app.bot.send_message.await_args_list
    assert "naujas pokalbis" in str(pinned[-1])

    # 6. Nothing went to a hidden bridge session along the way.
    sessions.deliver.assert_not_awaited()


def test_tests_never_point_at_the_real_marker(tmp_path):
    """Detaching in a test once deleted the real ~/.claude/.voice-bridge-live
    and broke the running bridge's re-join after its next restart."""
    marker = TelegramIO.live_marker()
    assert str(marker).startswith(str(tmp_path.parent.parent))
    assert ".claude" not in marker.parts
