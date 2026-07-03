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
    parse_name_prefix,
    resolve_target,
    run_until_stopped,
)
from voice_bridge.types import Outbound


# --------------------------------------------------------------------------- #
# Fakes
# --------------------------------------------------------------------------- #


class FakeStore:
    """In-memory stand-in for routing.Store covering the methods bridge uses."""

    def __init__(self, by_message=None, last_active=None, enabled=None, usage=None):
        self._by_message = dict(by_message or {})
        self._last_active = last_active
        self._enabled = dict(enabled or {})
        self._usage = dict(usage or {})
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

    async def all_usage(self):
        return {k: dict(v) for k, v in self._usage.items()}


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
        self.question_kwargs: list[dict] = []
        self.disabled_prompts: list[tuple[str, str]] = []
        self.ran = 0
        self.stopped = 0

    async def send_update(self, project, voice_label, text, voice_bytes):
        self.updates.append((project, voice_label, text, voice_bytes))
        return list(self.ids)

    async def send_file(self, project, voice_label, text, voice_bytes, file_path):
        self.files.append((project, voice_label, text, voice_bytes, file_path))
        return list(self.ids)

    async def send_question(self, project, text, **kwargs):
        self.questions.append((project, text))
        self.question_kwargs.append(kwargs)
        return 999

    async def send_disabled_project_prompt(self, project, text):
        self.disabled_prompts.append((project, text))
        return 1000

    async def run(self):
        self.ran += 1

    async def stop(self):
        self.stopped += 1


class FakeProject:
    def __init__(self, name, voice=None, autonomy=None, cwd=None, enabled=True,
                 display_name=None, model=None, effort=None):
        self.name = name
        self.voice = voice
        self.autonomy = autonomy
        self.cwd = cwd or f"/tmp/{name}"
        self.enabled = enabled
        self.display_name = display_name
        self.model = model
        self.effort = effort


class FakeSessions:
    def __init__(self, projects=None):
        self._projects = {p.name: p for p in (projects or [])}
        self.delivered: list[tuple[str, str]] = []
        self.enabled_calls: list[tuple[str, bool]] = []
        self.mode_calls: list[tuple[str, str]] = []
        self.effort_calls: list[tuple[str, str]] = []
        self.verbose_calls: list[tuple[str, bool]] = []
        self.interrupt_calls: list[str] = []
        self._last_model: dict[str, str] = {}
        self.started = 0
        self.stopped = 0

    def project(self, name):
        return self._projects.get(name)

    def names(self):
        return list(self._projects)

    def last_model(self, name):
        return self._last_model.get(name)

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

    async def set_effort(self, project, level):
        self.effort_calls.append((project, level))
        proj = self._projects.get(project)
        if proj is not None:
            proj.effort = level

    async def set_verbose(self, project, on):
        self.verbose_calls.append((project, on))

    async def interrupt(self, project):
        self.interrupt_calls.append(project)
        return True

    async def start_all(self):
        self.started += 1

    async def stop_all(self):
        self.stopped += 1


class FakeApprovals:
    def __init__(self, pending=None):
        self._pending = set(pending or [])
        self.resolved: list[tuple[int, bool]] = []
        self.token_resolved: list[tuple[int, bool]] = []

    def has_pending(self, message_id):
        return message_id in self._pending

    def resolve(self, message_id, approved):
        self.resolved.append((message_id, approved))
        return True

    def resolve_token(self, token, approved):
        self.token_resolved.append((token, approved))
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
    tts_alert_voice = ""
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
async def test_make_outbound_alert_uses_alert_voice_when_set():
    store = FakeStore()
    tts = FakeTTS(out=b"V")
    tts_holder = {"backend": tts}
    telegram = FakeTelegram(ids=[601])
    sessions = FakeSessions([FakeProject("qwing", voice="echo")])

    class AlertCfg(FakeCfg):
        tts_alert_voice = "shimmer"

    cfg = AlertCfg()
    controls = _Controls(sessions, store, cfg, tts_holder)
    outbound = make_outbound(tts_holder, telegram, store, cfg, sessions, controls)

    # an ALERT Outbound synthesizes with the distinct alert voice ...
    await outbound(Outbound(project="qwing", text="Crashed.", spoken="Crashed.", alert=True))
    # ... while a normal Outbound stays on the per-project voice.
    await outbound(Outbound(project="qwing", text="Done.", spoken="Done."))

    assert tts.calls == [("Crashed.", "shimmer"), ("Done.", "echo")]


@pytest.mark.asyncio
async def test_make_outbound_alert_falls_back_to_project_voice_when_unset():
    store = FakeStore()
    tts = FakeTTS(out=b"V")
    tts_holder = {"backend": tts}
    telegram = FakeTelegram(ids=[602])
    sessions = FakeSessions([FakeProject("qwing", voice="echo")])
    cfg = FakeCfg()  # tts_alert_voice == ""
    controls = _Controls(sessions, store, cfg, tts_holder)
    outbound = make_outbound(tts_holder, telegram, store, cfg, sessions, controls)

    await outbound(Outbound(project="qwing", text="Crashed.", spoken="Crashed.", alert=True))

    assert tts.calls == [("Crashed.", "echo")]  # no alert voice -> project voice


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


@pytest.mark.asyncio
async def test_make_outbound_survives_send_update_failure_and_sends_fallback():
    """C8 corollary: a bad Telegram send must never kill the session turn
    loop. make_outbound must swallow the error, log it, and best-effort
    notify the user via send_question instead of propagating."""
    store = FakeStore()
    tts_holder = {"backend": FakeTTS(out=b"VOICE")}
    telegram = FakeTelegram(ids=[101])
    sessions = FakeSessions([FakeProject("qwing", voice="echo")])
    controls = _Controls(sessions, store, FakeCfg(), tts_holder)

    async def boom(*a, **k):
        raise RuntimeError("telegram down")

    telegram.send_update = boom

    outbound = make_outbound(tts_holder, telegram, store, FakeCfg(), sessions, controls)

    # Must not raise.
    await outbound(Outbound(project="qwing", text="Done.\n---\n`code`", spoken=""))

    assert store.mapped == []
    assert store.last_active_calls == []
    assert len(telegram.questions) == 1
    assert telegram.questions[0][0] == "qwing"


@pytest.mark.asyncio
async def test_make_outbound_survives_even_if_fallback_notify_also_fails():
    store = FakeStore()
    tts_holder = {"backend": FakeTTS(out=b"VOICE")}
    telegram = FakeTelegram(ids=[101])
    sessions = FakeSessions([FakeProject("qwing", voice="echo")])
    controls = _Controls(sessions, store, FakeCfg(), tts_holder)

    async def boom(*a, **k):
        raise RuntimeError("telegram down")

    telegram.send_update = boom
    telegram.send_question = boom

    outbound = make_outbound(tts_holder, telegram, store, FakeCfg(), sessions, controls)

    # Must not raise even though BOTH the send and the fallback notice fail.
    await outbound(Outbound(project="qwing", text="Done.", spoken=""))


@pytest.mark.asyncio
async def test_make_outbound_survives_send_file_failure(tmp_path):
    store = FakeStore()
    tts_holder = {"backend": FakeTTS(out=b"VOICE")}
    telegram = FakeTelegram(ids=[101])
    sessions = FakeSessions([FakeProject("qwing", voice="echo")])
    controls = _Controls(sessions, store, FakeCfg(), tts_holder)
    path = tmp_path / "result.txt"
    path.write_text("ok")

    async def boom(*a, **k):
        raise RuntimeError("telegram down")

    telegram.send_file = boom

    outbound = make_outbound(tts_holder, telegram, store, FakeCfg(), sessions, controls)

    await outbound(
        Outbound(project="qwing", text="result", spoken="", file_path=str(path))
    )

    assert store.mapped == []
    assert len(telegram.questions) == 1


@pytest.mark.asyncio
async def test_make_outbound_recovers_after_a_previous_send_failure():
    """A single bad send must not wedge subsequent, healthy turns."""
    store = FakeStore()
    tts_holder = {"backend": FakeTTS(out=b"VOICE")}
    telegram = FakeTelegram(ids=[101])
    sessions = FakeSessions([FakeProject("qwing", voice="echo")])
    controls = _Controls(sessions, store, FakeCfg(), tts_holder)

    real_send_update = telegram.send_update
    call_count = {"n": 0}

    async def flaky_once(*a, **k):
        call_count["n"] += 1
        if call_count["n"] == 1:
            raise RuntimeError("telegram down")
        return await real_send_update(*a, **k)

    telegram.send_update = flaky_once

    outbound = make_outbound(tts_holder, telegram, store, FakeCfg(), sessions, controls)

    await outbound(Outbound(project="qwing", text="first", spoken=""))
    await outbound(Outbound(project="qwing", text="second", spoken=""))

    assert telegram.updates == [("qwing", "echo", "second", b"VOICE")]
    assert store.mapped == [(101, "qwing")]
    assert store.last_active_calls == ["qwing"]


# --------------------------------------------------------------------------- #
# make_inbound
# --------------------------------------------------------------------------- #


def _inbound(transcriber, store, approvals, sessions, telegram, controls=None):
    if controls is None:
        controls = _Controls(sessions, store, FakeCfg(), {"backend": FakeTTS()})
    return make_inbound(transcriber, store, approvals, sessions, telegram, controls)


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
async def test_make_inbound_bang_prefix_interrupts_then_delivers_without_prefix():
    store = FakeStore(last_active="qwing", enabled={"qwing": True})
    approvals = FakeApprovals()
    transcriber = FakeTranscriber()
    sessions = FakeSessions([FakeProject("qwing")])
    telegram = FakeTelegram()

    inbound = _inbound(transcriber, store, approvals, sessions, telegram)
    await inbound(_msg(text="! stop and do this"))

    assert sessions.interrupt_calls == ["qwing"]
    assert sessions.delivered == [("qwing", "stop and do this")]


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
    assert "did not understand" in telegram.questions[0][1]


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
    assert "did not understand" in telegram.questions[0][1]


@pytest.mark.asyncio
async def test_make_inbound_pending_approval_resolves_yes_no_deliver():
    store = FakeStore()
    approvals = FakeApprovals(pending=[55])
    transcriber = FakeTranscriber()
    sessions = FakeSessions()
    telegram = FakeTelegram()

    inbound = _inbound(transcriber, store, approvals, sessions, telegram)
    await inbound(_msg(reply_to=55, text="yes"))

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
    await inbound(_msg(reply_to=55, text="no"))

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
    assert "yes" in telegram.questions[0][1].lower()


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
# parse_name_prefix (pure)
# --------------------------------------------------------------------------- #


def test_parse_name_prefix_colon_separator():
    assert parse_name_prefix("qwing: run tests", ["qwing", "othersapp"]) == (
        "qwing",
        "run tests",
    )


def test_parse_name_prefix_whitespace_only_separator():
    assert parse_name_prefix("qwing run tests", ["qwing", "othersapp"]) == (
        "qwing",
        "run tests",
    )


def test_parse_name_prefix_case_insensitive_returns_canonical_name():
    assert parse_name_prefix("Qwing: x", ["qwing", "othersapp"]) == ("qwing", "x")


def test_parse_name_prefix_unknown_name_returns_none_unchanged():
    text = "unknown: x"
    assert parse_name_prefix(text, ["qwing", "othersapp"]) == (None, text)


def test_parse_name_prefix_colon_with_no_matching_name_returns_none_unchanged():
    text = "just a colon: here"
    assert parse_name_prefix(text, ["qwing", "othersapp"]) == (None, text)


def test_parse_name_prefix_plain_message_returns_none_unchanged():
    text = "hello there, how are you"
    assert parse_name_prefix(text, ["qwing", "othersapp"]) == (None, text)


def test_parse_name_prefix_prefers_longer_name_over_shorter_prefix_match():
    assert parse_name_prefix("qwingtest: hi", ["qwing", "qwingtest"]) == (
        "qwingtest",
        "hi",
    )


def test_parse_name_prefix_no_names_returns_none_unchanged():
    text = "qwing: x"
    assert parse_name_prefix(text, []) == (None, text)


def test_parse_name_prefix_colon_with_no_trailing_space():
    # B3b Minor: users type "qwing:build" with no space after the colon.
    assert parse_name_prefix("qwing:build", ["qwing", "othersapp"]) == (
        "qwing",
        "build",
    )


def test_parse_name_prefix_dash_with_no_trailing_space():
    assert parse_name_prefix("qwing-build", ["qwing", "othersapp"]) == (
        "qwing",
        "build",
    )


def test_parse_name_prefix_no_separator_at_all_does_not_match():
    # Exact-token guarantee preserved: no ":"/"-"/whitespace right after the
    # name means it is not a routing prefix at all.
    text = "qwingbuild"
    assert parse_name_prefix(text, ["qwing", "othersapp"]) == (None, text)


def test_parse_name_prefix_extra_letters_before_colon_does_not_match():
    # "qwinger" is not the token "qwing" even though it starts with it.
    text = "qwinger: x"
    assert parse_name_prefix(text, ["qwing", "othersapp"]) == (None, text)


def test_parse_name_prefix_none_text_returns_none_and_empty_string():
    assert parse_name_prefix(None, ["qwing", "othersapp"]) == (None, "")


# --------------------------------------------------------------------------- #
# make_inbound: name-prefix routing precedence
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_make_inbound_name_prefix_routes_to_named_project_no_reply():
    store = FakeStore(last_active="othersapp", enabled={"qwing": True, "othersapp": True})
    approvals = FakeApprovals()
    transcriber = FakeTranscriber()
    sessions = FakeSessions([FakeProject("qwing"), FakeProject("othersapp")])
    telegram = FakeTelegram()

    inbound = _inbound(transcriber, store, approvals, sessions, telegram)
    await inbound(_msg(reply_to=None, text="qwing: build"))

    assert sessions.delivered == [("qwing", "build")]


@pytest.mark.asyncio
async def test_make_inbound_name_prefix_routes_voice_after_transcription():
    store = FakeStore(last_active="othersapp", enabled={"qwing": True, "othersapp": True})
    approvals = FakeApprovals()
    transcriber = FakeTranscriber(text="qwing: paleisk testus")
    sessions = FakeSessions([FakeProject("qwing"), FakeProject("othersapp")])
    telegram = FakeTelegram()

    inbound = _inbound(transcriber, store, approvals, sessions, telegram)
    await inbound(_msg(reply_to=None, is_voice=True, audio=b"OGG"))

    assert sessions.delivered == [("qwing", "paleisk testus")]


@pytest.mark.asyncio
async def test_make_inbound_reply_to_wins_over_name_prefix():
    store = FakeStore(by_message={42: "othersapp"}, enabled={"qwing": True, "othersapp": True})
    approvals = FakeApprovals()
    transcriber = FakeTranscriber()
    sessions = FakeSessions([FakeProject("qwing"), FakeProject("othersapp")])
    telegram = FakeTelegram()

    inbound = _inbound(transcriber, store, approvals, sessions, telegram)
    await inbound(_msg(reply_to=42, text="qwing: build"))

    # the quote-reply target (othersapp) wins outright; the leading
    # "qwing:" is not treated as a routing prefix at all.
    assert sessions.delivered == [("othersapp", "qwing: build")]


@pytest.mark.asyncio
async def test_make_inbound_name_prefix_to_disabled_project_sends_off_notice():
    store = FakeStore(last_active="othersapp", enabled={"qwing": False, "othersapp": True})
    approvals = FakeApprovals()
    transcriber = FakeTranscriber()
    sessions = FakeSessions([FakeProject("qwing"), FakeProject("othersapp")])
    telegram = FakeTelegram()

    inbound = _inbound(transcriber, store, approvals, sessions, telegram)
    await inbound(_msg(reply_to=None, text="qwing: build"))

    assert sessions.delivered == []
    assert telegram.disabled_prompts == [("qwing", "build")]


# --------------------------------------------------------------------------- #
# Recap
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_recap_lists_project_with_update_count_and_latest_line():
    controls, sessions, store, cfg, tts_holder = _make_controls()
    telegram = FakeTelegram()
    outbound = make_outbound(tts_holder, telegram, store, cfg, sessions, controls)

    controls.mark_recap_boundary()
    await outbound(Outbound(project="qwing", text="Started.", spoken="Started."))
    await outbound(Outbound(project="qwing", text="Done.\n---\ncode", spoken=""))

    text = controls.recap()
    assert "qwing" in text
    assert "2 atnaujinimai" in text
    assert "Done." in text  # latest line, spoken empty -> falls back to first line


@pytest.mark.asyncio
async def test_recap_ignores_transient_status_and_heartbeat_noise():
    # B3b Minor: a turn that emits the "Working." status, a "still working"
    # heartbeat, and one real reply must show 1 update in /recap, not 3.
    controls, sessions, store, cfg, tts_holder = _make_controls()
    telegram = FakeTelegram()
    outbound = make_outbound(tts_holder, telegram, store, cfg, sessions, controls)

    controls.mark_recap_boundary()
    await outbound(
        Outbound(project="qwing", text="Working.", spoken=" ", transient=True)
    )
    await outbound(
        Outbound(
            project="qwing", text="Vis dar dirbu…", spoken="Vis dar dirbu…",
            transient=True,
        )
    )
    await outbound(Outbound(project="qwing", text="Done.", spoken="Done."))

    text = controls.recap()
    assert "qwing" in text
    assert "1 atnaujinimai" in text
    assert "Done." in text


@pytest.mark.asyncio
async def test_make_outbound_transient_skips_record_recap():
    controls, sessions, store, cfg, tts_holder = _make_controls()
    telegram = FakeTelegram()
    outbound = make_outbound(tts_holder, telegram, store, cfg, sessions, controls)

    controls.mark_recap_boundary()
    await outbound(
        Outbound(project="qwing", text="Working.", spoken=" ", transient=True)
    )

    assert controls._recap_lines.get("qwing", []) == []
    assert controls.recap() == "Nieko naujo."


@pytest.mark.asyncio
async def test_recap_omits_projects_with_no_activity():
    controls, sessions, store, cfg, tts_holder = _make_controls()
    telegram = FakeTelegram()
    outbound = make_outbound(tts_holder, telegram, store, cfg, sessions, controls)

    controls.mark_recap_boundary()
    await outbound(Outbound(project="qwing", text="Done.", spoken="Done."))

    text = controls.recap()
    assert "qwing" in text
    assert "othersapp" not in text


def test_recap_nothing_new_message_when_no_activity():
    controls, sessions, store, cfg, tts_holder = _make_controls()
    controls.mark_recap_boundary()
    assert controls.recap() == "Nieko naujo."


@pytest.mark.asyncio
async def test_recap_new_inbound_resets_buffers():
    controls, sessions, store, cfg, tts_holder = _make_controls()
    telegram = FakeTelegram()
    outbound = make_outbound(tts_holder, telegram, store, cfg, sessions, controls)
    approvals = FakeApprovals()
    transcriber = FakeTranscriber()
    store._last_active = "qwing"
    inbound = make_inbound(transcriber, store, approvals, sessions, telegram, controls)

    await outbound(Outbound(project="qwing", text="Done.", spoken="Done."))
    assert controls.recap() != "Nieko naujo."

    await inbound(_msg(text="hi"))
    assert controls.recap() == "Nieko naujo."


@pytest.mark.asyncio
async def test_make_outbound_recap_append_never_raises_when_controls_broken():
    # B1b: recap tracking must never break the never-raises outbound guard.
    store = FakeStore()
    tts_holder = {"backend": FakeTTS(out=b"V")}
    telegram = FakeTelegram(ids=[701])
    sessions = FakeSessions([FakeProject("qwing", voice="echo")])

    class BoomControls(_Controls):
        def record_recap(self, project, line):
            raise RuntimeError("boom")

    controls = BoomControls(sessions, store, FakeCfg(), tts_holder)
    outbound = make_outbound(tts_holder, telegram, store, FakeCfg(), sessions, controls)

    # Must not raise despite record_recap blowing up.
    await outbound(Outbound(project="qwing", text="Done.", spoken="Done."))
    assert telegram.updates  # the send itself still happened


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
            "cwd", "display_name", "verbose", "model", "effort",
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
async def test_controls_interrupt_defaults_to_last_active_project():
    controls, sessions, store, *_ = _make_controls()
    store._last_active = "qwing"
    await controls.seed()

    result = await controls.interrupt(None)

    assert result == "qwing: interrupted."
    assert sessions.interrupt_calls == ["qwing"]
    assert store.last_active_calls == ["qwing"]


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
async def test_controls_set_verbose_updates_mirror_and_sessions():
    controls, sessions, *_ = _make_controls()
    await controls.seed()
    # default OFF everywhere
    assert all(r["verbose"] is False for r in controls.snapshot())

    await controls.set_verbose("qwing", True)

    assert sessions.verbose_calls == [("qwing", True)]
    snap = {r["project"]: r for r in controls.snapshot()}
    assert snap["qwing"]["verbose"] is True
    assert snap["othersapp"]["verbose"] is False


@pytest.mark.asyncio
async def test_controls_set_verbose_all_projects():
    controls, sessions, *_ = _make_controls()
    await controls.seed()

    await controls.set_verbose(None, True)

    assert sorted(sessions.verbose_calls) == [("othersapp", True), ("qwing", True)]
    assert all(r["verbose"] is True for r in controls.snapshot())


@pytest.mark.asyncio
async def test_controls_snapshot_includes_model_and_effort():
    cfg = FakeCfg()
    store = FakeStore(enabled={"qwing": True})
    sessions = FakeSessions([
        FakeProject("qwing", model="claude-opus-4-8", effort="high"),
    ])
    controls = _Controls(sessions, store, cfg, {"backend": FakeTTS()})
    await controls.seed()
    snap = {r["project"]: r for r in controls.snapshot()}
    assert snap["qwing"]["model"] == "claude-opus-4-8"
    assert snap["qwing"]["effort"] == "high"


@pytest.mark.asyncio
async def test_controls_set_effort_updates_mirror_and_sessions():
    controls, sessions, *_ = _make_controls()
    await controls.seed()
    assert all(r["effort"] is None for r in controls.snapshot())

    await controls.set_effort("qwing", "high")

    assert sessions.effort_calls == [("qwing", "high")]
    snap = {r["project"]: r for r in controls.snapshot()}
    assert snap["qwing"]["effort"] == "high"
    assert snap["othersapp"]["effort"] is None


@pytest.mark.asyncio
async def test_controls_set_effort_all_projects():
    controls, sessions, *_ = _make_controls()
    await controls.seed()

    await controls.set_effort(None, "max")

    assert sorted(sessions.effort_calls) == [("othersapp", "max"), ("qwing", "max")]
    assert all(r["effort"] == "max" for r in controls.snapshot())


@pytest.mark.asyncio
async def test_controls_set_effort_invalid_level_is_ignored():
    controls, sessions, *_ = _make_controls()
    await controls.seed()

    await controls.set_effort("qwing", "turbo")

    assert sessions.effort_calls == []
    assert all(r["effort"] is None for r in controls.snapshot())


@pytest.mark.asyncio
async def test_controls_info_renders_model_effort_mode_voice_verbose():
    controls, sessions, *_ = _make_controls()
    # qwing: explicit config model, and a captured REAL model from a turn
    sessions.project("qwing").model = "claude-opus-4-8"
    sessions._last_model["qwing"] = "claude-opus-4-8-20990101"
    await controls.seed()
    await controls.set_effort("qwing", "high")

    text = controls.info()

    lines = {ln.split(":")[0]: ln for ln in text.splitlines()}
    qwing_line = next(ln for ln in text.splitlines() if ln.startswith("qwing"))
    assert "model=claude-opus-4-8" in qwing_line
    assert "real: claude-opus-4-8-20990101" in qwing_line
    assert "effort=high" in qwing_line
    assert "mode=full" in qwing_line  # qwing autonomy="full" in _make_controls
    assert "voice=echo" in qwing_line
    assert "verbose=off" in qwing_line
    # othersapp: no config model, no real model captured, no effort -> defaults
    others_line = next(ln for ln in text.splitlines() if ln.startswith("othersapp"))
    assert "model=default" in others_line
    assert "real: —" in others_line
    assert "effort=default" in others_line
    # the global engine is shown too
    assert "openai" in text


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
# cost_summary (B3c: per-project token & cost tracking)
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_cost_summary_reports_per_project_and_total():
    cfg = FakeCfg()
    store = FakeStore(
        enabled={"qwing": True, "othersapp": False},
        usage={
            "qwing": {
                "turns": 3, "input_tokens": 1000, "output_tokens": 400,
                "cache_read_tokens": 50, "cache_creation_tokens": 20,
                "cost_usd": 0.0567,
            },
            "othersapp": {
                "turns": 1, "input_tokens": 100, "output_tokens": 40,
                "cache_read_tokens": 0, "cache_creation_tokens": 0,
                "cost_usd": 0.0033,
            },
        },
    )
    sessions = FakeSessions([FakeProject("qwing"), FakeProject("othersapp")])
    tts_holder = {"backend": FakeTTS()}
    controls = _Controls(sessions, store, cfg, tts_holder)
    await controls.seed()

    text = await controls.cost_summary()

    assert "qwing: 3 turai, 1000+400 tok, $0.0567" in text
    assert "othersapp: 1 turai, 100+40 tok, $0.0033" in text
    assert "TOTAL: 4 turai, 1100+440 tok, $0.0600" in text


@pytest.mark.asyncio
async def test_cost_summary_shows_tokens_and_notes_cost_unavailable_when_zero():
    # Claude Code subscription auth: total_cost_usd is always None -> every
    # accumulated cost_usd stays 0. The summary must not lie with "$0.0000"
    # for every project; it should show tokens and flag cost as unavailable.
    cfg = FakeCfg()
    store = FakeStore(
        enabled={"qwing": True},
        usage={
            "qwing": {
                "turns": 2, "input_tokens": 500, "output_tokens": 200,
                "cache_read_tokens": 0, "cache_creation_tokens": 0,
                "cost_usd": 0.0,
            },
        },
    )
    sessions = FakeSessions([FakeProject("qwing")])
    tts_holder = {"backend": FakeTTS()}
    controls = _Controls(sessions, store, cfg, tts_holder)
    await controls.seed()

    text = await controls.cost_summary()

    assert "500+200 tok" in text
    assert "n/a" in text.lower() or "unavailable" in text.lower()


@pytest.mark.asyncio
async def test_cost_summary_no_usage_recorded_yet():
    cfg = FakeCfg()
    store = FakeStore(enabled={"qwing": True})
    sessions = FakeSessions([FakeProject("qwing")])
    tts_holder = {"backend": FakeTTS()}
    controls = _Controls(sessions, store, cfg, tts_holder)
    await controls.seed()

    text = await controls.cost_summary()
    assert text  # non-empty, does not raise on empty usage


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
                        lambda cfg, on_user_message, controls, on_approval=None: telegram)

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
        bridge_mod,
        "TelegramIO",
        lambda cfg, on_user_message, controls, on_approval=None: telegram,
    )

    wired = await build()
    await wired.inbound(_msg(text="go"))

    assert telegram.disabled_prompts == [("qwing", "go")]
    assert sessions.delivered == []


@pytest.mark.asyncio
async def test_build_approval_send_uses_alert_voice_and_token(monkeypatch):
    """The send_question closure wired into ApprovalManager synthesizes the
    spoken approval line with the ALERT voice and forwards the token so the
    inline buttons can be attached. It also wires on_approval to resolve_token.
    """
    import voice_bridge.bridge as bridge_mod

    class AlertCfg(FakeCfg):
        tts_alert_voice = "shimmer"

    cfg = AlertCfg()
    projects = [FakeProject("qwing", voice="echo")]
    store = FakeStore(enabled={"qwing": True})
    telegram = FakeTelegram()
    sessions = FakeSessions(projects)
    tts = FakeTTS(out=b"ALERT")
    approvals = FakeApprovals()

    captured: dict = {}

    def capture_am(send_question, timeout):
        captured["send_question"] = send_question
        return approvals

    def capture_tg(cfg_, on_user_message, controls, on_approval=None):
        captured["on_approval"] = on_approval
        return telegram

    monkeypatch.setattr(bridge_mod, "load_config", lambda env=None: cfg)
    monkeypatch.setattr(bridge_mod, "load_projects", lambda path="projects.yaml": projects)
    monkeypatch.setattr(bridge_mod, "Store", lambda db_path: store)
    monkeypatch.setattr(
        bridge_mod, "Transcriber", lambda model_name, language="lt": FakeTranscriber()
    )
    monkeypatch.setattr(bridge_mod, "get_tts", lambda c: tts)
    monkeypatch.setattr(bridge_mod, "ApprovalManager", capture_am)
    monkeypatch.setattr(bridge_mod, "SessionManager", lambda *a, **k: sessions)
    monkeypatch.setattr(bridge_mod, "TelegramIO", capture_tg)

    await build()

    # on_approval is wired to the approvals' token resolver
    assert captured["on_approval"] == approvals.resolve_token

    send_question = captured["send_question"]
    mid = await send_question(
        "qwing", "qwing — approval reikalingas:\n\n```\ngit push\n```",
        "qwing nori paleisti komandą — leidžiu?", 3,
    )

    assert mid == 999
    # the spoken line was synthesized with the alert voice
    assert tts.calls and tts.calls[-1][1] == "shimmer"
    # the buttoned text + token + voice reached telegram.send_question
    kwargs = telegram.question_kwargs[-1]
    assert kwargs["approval_token"] == 3
    assert kwargs["voice_label"] == "shimmer"
    assert kwargs["voice_bytes"] == b"ALERT"
