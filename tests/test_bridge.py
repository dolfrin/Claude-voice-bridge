"""Tests for the bridge wiring (Task 10).

External boundaries (telegram, sessions, store, transcriber, tts, approvals)
are stubbed with in-memory fakes. No network, no real SDK, no signals.
"""
from __future__ import annotations

import asyncio

import pytest

from voice_bridge.bridge import (
    _Controls,
    build,
    make_inbound,
    make_outbound,
    resolve_target,
    run_until_stopped,
)
from voice_bridge.types import Outbound


# --------------------------------------------------------------------------- #
# Fakes
# --------------------------------------------------------------------------- #


class FakeStore:
    """In-memory stand-in for routing.Store covering the methods bridge uses."""

    def __init__(self, by_message=None, last_active=None, enabled=None):
        self._by_message = dict(by_message or {})
        self._last_active = last_active
        self._enabled = dict(enabled or {})
        self.mapped: list[tuple[int, str]] = []
        self.last_active_calls: list[str] = []
        self.inited = 0
        self.seeded: list[list] = []

    async def init(self):
        self.inited += 1

    async def seed(self, projects):
        self.seeded.append(list(projects))

    async def project_for_message(self, message_id):
        return self._by_message.get(message_id)

    async def map_message(self, message_id, project):
        self._by_message[message_id] = project
        self.mapped.append((message_id, project))

    async def set_last_active(self, project):
        self._last_active = project
        self.last_active_calls.append(project)

    async def get_last_active(self):
        return self._last_active

    async def is_enabled(self, project):
        return self._enabled.get(project, True)

    async def set_enabled(self, project, enabled):
        self._enabled[project] = enabled

    async def enabled_map(self):
        return dict(self._enabled)


class FakeTTS:
    def __init__(self, out=b"OGGDATA", boom=False):
        self.out = out
        self.boom = boom
        self.calls: list[tuple[str, str]] = []

    async def synthesize(self, text, voice):
        self.calls.append((text, voice))
        if self.boom:
            raise RuntimeError("tts down")
        return self.out


class FakeTelegram:
    def __init__(self, ids=None):
        self.ids = ids or [101]
        self.updates: list[tuple] = []
        self.files: list[tuple] = []
        self.questions: list[tuple[str, str]] = []
        self.disabled_prompts: list[tuple[str, str]] = []
        self.ran = 0
        self.stopped = 0

    async def send_update(self, project, voice_label, text, voice_bytes):
        self.updates.append((project, voice_label, text, voice_bytes))
        return list(self.ids)

    async def send_file(self, project, voice_label, text, voice_bytes, file_path):
        self.files.append((project, voice_label, text, voice_bytes, file_path))
        return list(self.ids)

    async def send_question(self, project, text):
        self.questions.append((project, text))
        return 999

    async def send_disabled_project_prompt(self, project, text):
        self.disabled_prompts.append((project, text))
        return 1000

    async def run(self):
        self.ran += 1

    async def stop(self):
        self.stopped += 1


class FakeProject:
    def __init__(self, name, voice=None, autonomy=None, cwd=None, enabled=True, display_name=None):
        self.name = name
        self.voice = voice
        self.autonomy = autonomy
        self.cwd = cwd or f"/tmp/{name}"
        self.enabled = enabled
        self.display_name = display_name


class FakeSessions:
    def __init__(self, projects=None):
        self._projects = {p.name: p for p in (projects or [])}
        self.delivered: list[tuple[str, str]] = []
        self.enabled_calls: list[tuple[str, bool]] = []
        self.mode_calls: list[tuple[str, str]] = []
        self.started = 0
        self.stopped = 0

    def project(self, name):
        return self._projects.get(name)

    def names(self):
        return list(self._projects)

    def add_projects(self, projects):
        added = 0
        for project in projects:
            if project.name in self._projects:
                continue
            self._projects[project.name] = project
            added += 1
        return added

    async def deliver(self, project, text):
        self.delivered.append((project, text))

    async def set_enabled(self, project, enabled):
        self.enabled_calls.append((project, enabled))

    async def set_mode(self, project, mode):
        self.mode_calls.append((project, mode))

    async def start_all(self):
        self.started += 1

    async def stop_all(self):
        self.stopped += 1


class FakeApprovals:
    def __init__(self, pending=None):
        self._pending = set(pending or [])
        self.resolved: list[tuple[int, bool]] = []

    def has_pending(self, message_id):
        return message_id in self._pending

    def resolve(self, message_id, approved):
        self.resolved.append((message_id, approved))
        return True


class FakeTranscriber:
    def __init__(self, text="transcribed"):
        self.text = text
        self.calls: list[bytes] = []

    async def transcribe(self, audio):
        self.calls.append(audio)
        return self.text


class FakeCfg:
    tts_voice = "alloy"
    tts_backend = "openai"
    autonomy_mode = "safe"
    db_path = "/tmp/ignored.db"
    whisper_model = "large-v3"
    approval_timeout = 300
    auto_discover_projects = False
    auto_discover_limit = 12
    open_vscode_on_enable = False
    close_vscode_on_disable = False


def _msg(message_id=7, reply_to=None, text="", is_voice=False, audio=None):
    return {
        "message_id": message_id,
        "reply_to": reply_to,
        "text": text,
        "is_voice": is_voice,
        "audio": audio,
    }


# --------------------------------------------------------------------------- #
# resolve_target
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_resolve_target_reply_to_maps_to_project():
    store = FakeStore(by_message={42: "qwing"}, last_active="othersapp",
                      enabled={"qwing": True})
    project, reason = await resolve_target(_msg(reply_to=42, text="go on"), store)
    assert (project, reason) == ("qwing", "ok")


@pytest.mark.asyncio
async def test_resolve_target_no_reply_falls_back_to_last_active():
    store = FakeStore(last_active="othersapp", enabled={"othersapp": True})
    project, reason = await resolve_target(_msg(reply_to=None, text="go on"), store)
    assert (project, reason) == ("othersapp", "ok")


@pytest.mark.asyncio
async def test_resolve_target_reply_to_unknown_falls_back_to_last_active():
    store = FakeStore(by_message={}, last_active="othersapp",
                      enabled={"othersapp": True})
    project, reason = await resolve_target(_msg(reply_to=999), store)
    assert (project, reason) == ("othersapp", "ok")


@pytest.mark.asyncio
async def test_resolve_target_none_when_nothing():
    store = FakeStore(by_message={}, last_active=None)
    project, reason = await resolve_target(_msg(reply_to=None), store)
    assert (project, reason) == (None, "none")


@pytest.mark.asyncio
async def test_resolve_target_off_when_disabled():
    store = FakeStore(by_message={42: "qwing"}, enabled={"qwing": False})
    project, reason = await resolve_target(_msg(reply_to=42), store)
    assert (project, reason) == ("qwing", "off")


# --------------------------------------------------------------------------- #
# make_outbound
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_make_outbound_assistant_split_synth_send_map_active():
    store = FakeStore()
    tts_holder = {"backend": FakeTTS(out=b"VOICE")}
    telegram = FakeTelegram(ids=[101, 102])
    sessions = FakeSessions([FakeProject("qwing", voice="echo")])
    controls = _Controls(sessions, store, FakeCfg(), tts_holder)
    controls._mirror["qwing"] = {
        "enabled": True, "mode": "safe", "voice": "echo",
        "engine": "openai", "last_active": False,
    }

    outbound = make_outbound(tts_holder, telegram, store, FakeCfg(), sessions, controls)
    # assistant turn-end: spoken empty -> prepare_outbound split on '---'
    o = Outbound(project="qwing", text="Done.\n---\n`code here`", spoken="")
    await outbound(o)

    assert tts_holder["backend"].calls == [("Done.", "echo")]
    assert telegram.updates == [("qwing", "echo", "Done.\n---\n`code here`", b"VOICE")]
    assert store.mapped == [(101, "qwing"), (102, "qwing")]
    assert store.last_active_calls == ["qwing"]
    # mirror last_active flipped on
    assert controls._mirror["qwing"]["last_active"] is True


@pytest.mark.asyncio
async def test_make_outbound_notify_path_uses_text_and_spoken_hint():
    store = FakeStore()
    tts_holder = {"backend": FakeTTS(out=b"VOICE")}
    telegram = FakeTelegram(ids=[201])
    sessions = FakeSessions([FakeProject("qwing", voice="echo")])
    controls = _Controls(sessions, store, FakeCfg(), tts_holder)

    outbound = make_outbound(tts_holder, telegram, store, FakeCfg(), sessions, controls)
    # notify path: spoken set -> full=o.text, voice=to_spoken(o.spoken)
    o = Outbound(project="qwing", text="Build finished. path/to/x.py changed.",
                 spoken="Build finished.")
    await outbound(o)

    assert tts_holder["backend"].calls == [("Build finished.", "echo")]
    assert telegram.updates == [
        ("qwing", "echo", "Build finished. path/to/x.py changed.", b"VOICE")
    ]


@pytest.mark.asyncio
async def test_make_outbound_empty_spoken_text_only():
    store = FakeStore()
    tts = FakeTTS(out=b"VOICE")
    tts_holder = {"backend": tts}
    telegram = FakeTelegram(ids=[301])
    sessions = FakeSessions([FakeProject("qwing", voice="echo")])
    controls = _Controls(sessions, store, FakeCfg(), tts_holder)

    outbound = make_outbound(tts_holder, telegram, store, FakeCfg(), sessions, controls)
    # text whose spoken part collapses to empty -> voice_bytes None, no synth
    o = Outbound(project="qwing", text="`only code`\n---\nmore", spoken="")
    await outbound(o)

    assert tts.calls == []  # nothing to speak
    assert telegram.updates == [("qwing", "echo", "`only code`\n---\nmore", None)]


@pytest.mark.asyncio
async def test_make_outbound_unknown_project_uses_default_voice():
    store = FakeStore()
    tts_holder = {"backend": FakeTTS(out=b"V")}
    telegram = FakeTelegram(ids=[401])
    sessions = FakeSessions([])  # no projects registered
    controls = _Controls(sessions, store, FakeCfg(), tts_holder)

    outbound = make_outbound(tts_holder, telegram, store, FakeCfg(), sessions, controls)
    await outbound(Outbound(project="ghost", text="Hi there", spoken="Hi there"))

    assert tts_holder["backend"].calls == [("Hi there", "alloy")]  # cfg.tts_voice


@pytest.mark.asyncio
async def test_make_outbound_file_sends_file_and_maps_message(tmp_path):
    store = FakeStore()
    tts_holder = {"backend": FakeTTS(out=b"V")}
    telegram = FakeTelegram(ids=[501])
    sessions = FakeSessions([FakeProject("qwing", voice="echo")])
    controls = _Controls(sessions, store, FakeCfg(), tts_holder)
    path = tmp_path / "result.txt"
    path.write_text("ok")

    outbound = make_outbound(tts_holder, telegram, store, FakeCfg(), sessions, controls)
    await outbound(
        Outbound(
            project="qwing",
            text="Rezultatas prisegtas",
            spoken="",
            file_path=str(path),
        )
    )

    assert telegram.updates == []
    assert telegram.files == [
        ("qwing", "echo", "Rezultatas prisegtas", b"V", str(path))
    ]
    assert tts_holder["backend"].calls == [("Rezultatas prisegtas", "echo")]
    assert store.mapped == [(501, "qwing")]


# --------------------------------------------------------------------------- #
# make_inbound
# --------------------------------------------------------------------------- #


def _inbound(transcriber, store, approvals, sessions, telegram):
    return make_inbound(transcriber, store, approvals, sessions, telegram)


@pytest.mark.asyncio
async def test_make_inbound_text_reply_routes_to_replied_project():
    store = FakeStore(by_message={42: "qwing"}, enabled={"qwing": True})
    approvals = FakeApprovals()
    transcriber = FakeTranscriber()
    sessions = FakeSessions()
    telegram = FakeTelegram()

    inbound = _inbound(transcriber, store, approvals, sessions, telegram)
    await inbound(_msg(reply_to=42, text="continue"))

    assert transcriber.calls == []
    assert sessions.delivered == [("qwing", "continue")]
    assert approvals.resolved == []


@pytest.mark.asyncio
async def test_make_inbound_voice_transcribed_then_delivered():
    store = FakeStore(last_active="qwing", enabled={"qwing": True})
    approvals = FakeApprovals()
    transcriber = FakeTranscriber(text="tęsk darbą")
    sessions = FakeSessions()
    telegram = FakeTelegram()

    inbound = _inbound(transcriber, store, approvals, sessions, telegram)
    await inbound(_msg(reply_to=None, is_voice=True, audio=b"OGG"))

    assert transcriber.calls == [b"OGG"]
    assert sessions.delivered == [("qwing", "tęsk darbą")]


@pytest.mark.asyncio
async def test_make_inbound_empty_transcript_asks_repeat_no_deliver():
    store = FakeStore(last_active="qwing", enabled={"qwing": True})
    approvals = FakeApprovals()
    transcriber = FakeTranscriber(text="   ")
    sessions = FakeSessions()
    telegram = FakeTelegram()

    inbound = _inbound(transcriber, store, approvals, sessions, telegram)
    await inbound(_msg(reply_to=None, is_voice=True, audio=b"OGG"))

    assert sessions.delivered == []
    assert len(telegram.questions) == 1
    assert "Nesupratau" in telegram.questions[0][1]


@pytest.mark.asyncio
async def test_make_inbound_voice_without_audio_asks_repeat_no_transcribe():
    store = FakeStore(last_active="qwing", enabled={"qwing": True})
    approvals = FakeApprovals()
    transcriber = FakeTranscriber(text="should not run")
    sessions = FakeSessions()
    telegram = FakeTelegram()

    inbound = _inbound(transcriber, store, approvals, sessions, telegram)
    await inbound(_msg(reply_to=None, is_voice=True, audio=None))

    assert transcriber.calls == []
    assert sessions.delivered == []
    assert len(telegram.questions) == 1
    assert "Nesupratau" in telegram.questions[0][1]


@pytest.mark.asyncio
async def test_make_inbound_pending_approval_resolves_yes_no_deliver():
    store = FakeStore()
    approvals = FakeApprovals(pending=[55])
    transcriber = FakeTranscriber()
    sessions = FakeSessions()
    telegram = FakeTelegram()

    inbound = _inbound(transcriber, store, approvals, sessions, telegram)
    await inbound(_msg(reply_to=55, text="taip"))

    assert approvals.resolved == [(55, True)]
    assert sessions.delivered == []


@pytest.mark.asyncio
async def test_make_inbound_pending_approval_resolves_no():
    store = FakeStore()
    approvals = FakeApprovals(pending=[55])
    transcriber = FakeTranscriber()
    sessions = FakeSessions()
    telegram = FakeTelegram()

    inbound = _inbound(transcriber, store, approvals, sessions, telegram)
    await inbound(_msg(reply_to=55, text="ne"))

    assert approvals.resolved == [(55, False)]
    assert sessions.delivered == []


@pytest.mark.asyncio
async def test_make_inbound_pending_approval_unparseable_asks_again():
    store = FakeStore()
    approvals = FakeApprovals(pending=[55])
    transcriber = FakeTranscriber()
    sessions = FakeSessions()
    telegram = FakeTelegram()

    inbound = _inbound(transcriber, store, approvals, sessions, telegram)
    await inbound(_msg(reply_to=55, text="maybe later"))

    assert approvals.resolved == []
    assert sessions.delivered == []
    assert len(telegram.questions) == 1
    assert "taip" in telegram.questions[0][1].lower()


@pytest.mark.asyncio
async def test_make_inbound_no_target_sends_which_project():
    store = FakeStore(by_message={}, last_active=None)
    approvals = FakeApprovals()
    transcriber = FakeTranscriber()
    sessions = FakeSessions([FakeProject("qwing"), FakeProject("othersapp")])
    telegram = FakeTelegram()

    inbound = _inbound(transcriber, store, approvals, sessions, telegram)
    await inbound(_msg(reply_to=None, text="hi"))

    assert sessions.delivered == []
    assert len(telegram.questions) == 1
    body = telegram.questions[0][1]
    assert "qwing" in body and "othersapp" in body


@pytest.mark.asyncio
async def test_make_inbound_disabled_target_asks_to_enable_and_send():
    store = FakeStore(by_message={42: "qwing"}, enabled={"qwing": False})
    approvals = FakeApprovals()
    transcriber = FakeTranscriber()
    sessions = FakeSessions([FakeProject("qwing")])
    telegram = FakeTelegram()

    inbound = _inbound(transcriber, store, approvals, sessions, telegram)
    await inbound(_msg(reply_to=42, text="go"))

    assert sessions.delivered == []
    assert telegram.questions == []
    assert telegram.disabled_prompts == [("qwing", "go")]


# --------------------------------------------------------------------------- #
# Controls
# --------------------------------------------------------------------------- #


def _make_controls():
    cfg = FakeCfg()
    store = FakeStore(enabled={"qwing": True, "othersapp": False})
    sessions = FakeSessions([
        FakeProject("qwing", voice="echo", autonomy="full"),
        FakeProject("othersapp", voice=None, autonomy=None),
    ])
    tts_holder = {"backend": FakeTTS()}
    controls = _Controls(sessions, store, cfg, tts_holder)
    return controls, sessions, store, cfg, tts_holder


@pytest.mark.asyncio
async def test_make_inbound_attachment_is_saved_and_added_to_prompt(tmp_path):
    store = FakeStore(last_active="qwing", enabled={"qwing": True})
    approvals = FakeApprovals()
    transcriber = FakeTranscriber()
    sessions = FakeSessions([FakeProject("qwing", cwd=str(tmp_path))])
    telegram = FakeTelegram()

    inbound = _inbound(transcriber, store, approvals, sessions, telegram)
    await inbound({
        "message_id": 77,
        "reply_to": None,
        "text": "peržiūrėk",
        "is_voice": False,
        "audio": None,
        "attachments": [{
            "kind": "document",
            "file_name": "log.txt",
            "mime_type": "text/plain",
            "data": b"hello",
        }],
    })

    assert len(sessions.delivered) == 1
    project, prompt = sessions.delivered[0]
    assert project == "qwing"
    assert "peržiūrėk" in prompt
    assert ".claude/voice-bridge-inbox/" in prompt
    assert "log.txt" in prompt
    saved_files = list((tmp_path / ".claude" / "voice-bridge-inbox").glob("*log.txt"))
    assert len(saved_files) == 1
    assert saved_files[0].read_bytes() == b"hello"


@pytest.mark.asyncio
async def test_make_inbound_audio_attachment_is_transcribed_and_saved(tmp_path):
    store = FakeStore(last_active="qwing", enabled={"qwing": True})
    approvals = FakeApprovals()
    transcriber = FakeTranscriber(text="čia garso tekstas")
    sessions = FakeSessions([FakeProject("qwing", cwd=str(tmp_path))])
    telegram = FakeTelegram()

    inbound = _inbound(transcriber, store, approvals, sessions, telegram)
    await inbound({
        "message_id": 78,
        "reply_to": None,
        "text": "",
        "is_voice": False,
        "audio": None,
        "attachments": [{
            "kind": "audio",
            "file_name": "note.mp3",
            "mime_type": "audio/mpeg",
            "data": b"MP3",
        }],
    })

    assert transcriber.calls == [b"MP3"]
    project, prompt = sessions.delivered[0]
    assert project == "qwing"
    assert "Audio transkripcija" in prompt
    assert "čia garso tekstas" in prompt
    assert "note.mp3" in prompt
    assert ".claude/voice-bridge-inbox/" in prompt


@pytest.mark.asyncio
async def test_controls_snapshot_sync_exact_keys():
    controls, *_ = _make_controls()
    await controls.seed()
    snap = controls.snapshot()  # SYNC, no await
    assert isinstance(snap, list)
    for row in snap:
        assert set(row.keys()) == {
            "project", "enabled", "mode", "voice", "engine", "last_active",
            "cwd", "display_name",
        }
    by_name = {r["project"]: r for r in snap}
    assert by_name["qwing"]["enabled"] is True
    assert by_name["qwing"]["display_name"] == "qwing"
    assert by_name["qwing"]["mode"] == "full"
    assert by_name["qwing"]["voice"] == "echo"
    assert by_name["othersapp"]["enabled"] is False
    assert by_name["othersapp"]["mode"] == "safe"   # falls back to cfg
    assert by_name["othersapp"]["voice"] == "alloy"  # falls back to cfg
    assert all(r["engine"] == "openai" for r in snap)


@pytest.mark.asyncio
async def test_controls_toggle_all_off_disables_every_project():
    controls, sessions, *_ = _make_controls()
    await controls.seed()
    await controls.toggle(None, False)
    assert sorted(sessions.enabled_calls) == [("othersapp", False), ("qwing", False)]
    assert all(r["enabled"] is False for r in controls.snapshot())


@pytest.mark.asyncio
async def test_controls_select_marks_last_active_without_enabling_project():
    controls, sessions, store, *_ = _make_controls()
    await controls.seed()

    await controls.select("othersapp")

    assert sessions.enabled_calls == []
    assert store.last_active_calls == ["othersapp"]
    snap = {row["project"]: row for row in controls.snapshot()}
    assert snap["othersapp"]["enabled"] is False
    assert snap["othersapp"]["last_active"] is True
    assert snap["qwing"]["last_active"] is False


@pytest.mark.asyncio
async def test_controls_enable_and_deliver_starts_project_then_sends_text():
    controls, sessions, store, *_ = _make_controls()
    await controls.seed()

    await controls.enable_and_deliver("othersapp", "go")

    assert sessions.enabled_calls == [("othersapp", True)]
    assert sessions.delivered == [("othersapp", "go")]
    assert store.last_active_calls == ["othersapp"]
    snap = {row["project"]: row for row in controls.snapshot()}
    assert snap["othersapp"]["enabled"] is True
    assert snap["othersapp"]["last_active"] is True


@pytest.mark.asyncio
async def test_controls_refresh_projects_discovers_new_disabled_project(monkeypatch):
    import voice_bridge.bridge as bridge_mod

    controls, sessions, store, *_ = _make_controls()
    await controls.seed()
    controls._cfg.auto_discover_projects = True
    monkeypatch.setattr(bridge_mod, "load_projects", lambda: [
        FakeProject("qwing", voice="echo", autonomy="full"),
        FakeProject("othersapp"),
    ])
    monkeypatch.setattr(bridge_mod, "discover_projects", lambda limit, explicit_cwds=None: [
        FakeProject("fresh", cwd="/home/home/Projects/Fresh", enabled=False),
    ])

    added = await controls.refresh_projects()

    assert added == 1
    assert "fresh" in sessions.names()
    snap = {row["project"]: row for row in controls.snapshot()}
    assert snap["fresh"]["enabled"] is False
    assert snap["fresh"]["cwd"] == "/home/home/Projects/Fresh"


@pytest.mark.asyncio
async def test_controls_set_voice_updates_mirror_and_project():
    controls, sessions, *_ = _make_controls()
    await controls.seed()
    await controls.set_voice("qwing", "sage")
    assert sessions.project("qwing").voice == "sage"
    snap = {r["project"]: r for r in controls.snapshot()}
    assert snap["qwing"]["voice"] == "sage"


@pytest.mark.asyncio
async def test_controls_set_mode_sends_notice(monkeypatch):
    controls, sessions, store, cfg, tts_holder = _make_controls()
    await controls.seed()
    telegram = FakeTelegram()
    controls.attach_telegram(telegram)
    await controls.set_mode("qwing", "ask")
    assert sessions.mode_calls == [("qwing", "ask")]
    assert len(telegram.questions) == 1
    assert "ask" in telegram.questions[0][1]


@pytest.mark.asyncio
async def test_controls_set_engine_rebuilds_backend_and_outbound_uses_it():
    import voice_bridge.bridge as bridge_mod

    controls, sessions, store, cfg, tts_holder = _make_controls()
    await controls.seed()

    class PiperLike:
        async def synthesize(self, text, voice):
            return b"PIPER"

    # monkeypatch get_tts used inside set_engine
    import voice_bridge.bridge as bm
    orig = bm.get_tts
    try:
        bm.get_tts = lambda c: PiperLike()
        before = type(tts_holder["backend"])
        await controls.set_engine("piper")
        assert cfg.tts_backend == "piper"
        assert type(tts_holder["backend"]) is not before
        assert isinstance(tts_holder["backend"], PiperLike)
        assert all(r["engine"] == "piper" for r in controls.snapshot())

        # C4: a subsequent outbound uses the NEW backend (read at send time)
        telegram = FakeTelegram(ids=[1])
        outbound = make_outbound(tts_holder, telegram, store, cfg, sessions, controls)
        await outbound(Outbound(project="qwing", text="Hi", spoken="Hi"))
        assert telegram.updates[0][3] == b"PIPER"
    finally:
        bm.get_tts = orig


# --------------------------------------------------------------------------- #
# build() + run loop
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_build_wires_and_run_loop(monkeypatch):
    import voice_bridge.bridge as bridge_mod

    cfg = FakeCfg()
    projects = [FakeProject("qwing", voice="echo")]

    store = FakeStore(enabled={"qwing": True})
    telegram = FakeTelegram()
    sessions = FakeSessions(projects)

    monkeypatch.setattr(bridge_mod, "load_config", lambda env=None: cfg)
    monkeypatch.setattr(bridge_mod, "load_projects", lambda path="projects.yaml": projects)
    monkeypatch.setattr(bridge_mod, "Store", lambda db_path: store)
    monkeypatch.setattr(bridge_mod, "Transcriber",
                        lambda model_name, language="lt": FakeTranscriber())
    monkeypatch.setattr(bridge_mod, "get_tts", lambda c: FakeTTS())
    monkeypatch.setattr(bridge_mod, "ApprovalManager",
                        lambda send_question, timeout: FakeApprovals())
    monkeypatch.setattr(bridge_mod, "SessionManager",
                        lambda *a, **k: sessions)
    monkeypatch.setattr(bridge_mod, "TelegramIO",
                        lambda cfg, on_user_message, controls: telegram)

    wired = await build()
    assert store.inited == 1
    assert len(store.seeded) == 1
    assert wired.telegram is telegram
    assert wired.sessions is sessions

    # the run loop: start_all, run, then stop on the event being set
    stop = asyncio.Event()
    stop.set()  # already stopped -> returns immediately after startup
    await run_until_stopped(wired, stop)

    assert sessions.started == 1
    assert telegram.ran == 1
    assert telegram.stopped == 1
    assert sessions.stopped == 1


@pytest.mark.asyncio
async def test_build_inbound_forwards_disabled_project_prompt(monkeypatch):
    import voice_bridge.bridge as bridge_mod

    cfg = FakeCfg()
    projects = [FakeProject("qwing", voice="echo")]

    store = FakeStore(last_active="qwing", enabled={"qwing": False})
    telegram = FakeTelegram()
    sessions = FakeSessions(projects)

    monkeypatch.setattr(bridge_mod, "load_config", lambda env=None: cfg)
    monkeypatch.setattr(bridge_mod, "load_projects", lambda path="projects.yaml": projects)
    monkeypatch.setattr(bridge_mod, "Store", lambda db_path: store)
    monkeypatch.setattr(
        bridge_mod, "Transcriber", lambda model_name, language="lt": FakeTranscriber()
    )
    monkeypatch.setattr(bridge_mod, "get_tts", lambda c: FakeTTS())
    monkeypatch.setattr(
        bridge_mod, "ApprovalManager", lambda send_question, timeout: FakeApprovals()
    )
    monkeypatch.setattr(bridge_mod, "SessionManager", lambda *a, **k: sessions)
    monkeypatch.setattr(
        bridge_mod, "TelegramIO", lambda cfg, on_user_message, controls: telegram
    )

    wired = await build()
    await wired.inbound(_msg(text="go"))

    assert telegram.disabled_prompts == [("qwing", "go")]
    assert sessions.delivered == []
