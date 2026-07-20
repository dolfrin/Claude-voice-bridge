"""Tests for the bridge wiring (Task 10).

External boundaries (telegram, sessions, store, transcriber, tts, approvals)
are stubbed with in-memory fakes. No network, no real SDK, no signals.
"""
from __future__ import annotations

import asyncio

import pytest

from voice_bridge.bridge import (
    _Controls,
    _sanitize_project_name,
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

    def __init__(self, by_message=None, last_active=None, enabled=None, usage=None,
                 overrides=None, created=None):
        self._by_message = dict(by_message or {})
        self._last_active = last_active
        self._enabled = dict(enabled or {})
        self._usage = dict(usage or {})
        self._overrides = {k: dict(v) for k, v in (overrides or {}).items()}
        self._created = list(created or [])
        self.mapped: list[tuple[int, str]] = []
        self.last_active_calls: list[str] = []
        self.override_calls: list[tuple[str, str, object]] = []
        self.created_calls: list[tuple[str, str, object]] = []
        self._policies: set[tuple[str, str]] = set()
        self.policy_calls: list[tuple] = []
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

    async def set_override(self, project, field, value):
        self.override_calls.append((project, field, value))
        self._overrides.setdefault(project, {})[field] = value

    async def overrides(self):
        return {k: dict(v) for k, v in self._overrides.items()}

    async def add_created_project(self, name, cwd, display_name):
        self.created_calls.append((name, cwd, display_name))
        self._created.append(
            {"name": name, "cwd": cwd, "display_name": display_name}
        )

    async def created_projects(self):
        return [dict(row) for row in self._created]

    # always-allow policies
    async def add_policy(self, project, signature):
        self.policy_calls.append(("add", project, signature))
        self._policies.add((project, signature))

    async def has_policy(self, project, signature):
        return (project, signature) in self._policies

    async def list_policies(self):
        return sorted(self._policies)

    async def clear_policy(self, project=None, signature=None):
        self.policy_calls.append(("clear", project, signature))
        if project is None:
            self._policies.clear()
        elif signature is None:
            self._policies = {p for p in self._policies if p[0] != project}
        else:
            self._policies.discard((project, signature))


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
    def __init__(self, ids=None, ask_by_message=None, single_ask=None,
                 resolve_ask_result=True):
        self.ids = ids or [101]
        self.updates: list[tuple] = []
        self.files: list[tuple] = []
        self.questions: list[tuple[str, str]] = []
        self.question_kwargs: list[dict] = []
        self.disabled_prompts: list[tuple[str, str]] = []
        self.ran = 0
        self.stopped = 0
        # I2 ask_user interception surface. ``ask_by_message`` maps a question
        # message_id -> token (quote-reply path); ``single_ask`` is the sole
        # outstanding token (single-pending fallback path).
        self._ask_by_message = dict(ask_by_message or {})
        self._single_ask = single_ask
        self._resolve_ask_result = resolve_ask_result
        self.resolved_asks: list[tuple[str, str]] = []

    def pending_ask_token_for_message(self, message_id):
        return self._ask_by_message.get(message_id)

    def single_pending_ask_token(self):
        return self._single_ask

    def resolve_ask(self, token, answer_text):
        self.resolved_asks.append((token, answer_text))
        return self._resolve_ask_result

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


def test_parse_name_prefix_comma_separator():
    # "Qwing, daryk x" — a comma right after the name must route, same as
    # ":"/"-"/whitespace.
    assert parse_name_prefix("Qwing, daryk x", ["qwing", "othersapp"]) == (
        "qwing",
        "daryk x",
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


@pytest.mark.asyncio
async def test_make_inbound_urgent_bang_before_name_prefix_targets_named_project():
    # "!qwing: fix" — the urgent '!' must be consumed BEFORE name-prefix
    # routing, so the message still routes to "qwing" (not last-active) and
    # interrupts qwing, delivering the prefix-stripped text.
    store = FakeStore(last_active="othersapp", enabled={"qwing": True, "othersapp": True})
    approvals = FakeApprovals()
    transcriber = FakeTranscriber()
    sessions = FakeSessions([FakeProject("qwing"), FakeProject("othersapp")])
    telegram = FakeTelegram()

    inbound = _inbound(transcriber, store, approvals, sessions, telegram)
    await inbound(_msg(text="!qwing: fix"))

    assert sessions.interrupt_calls == ["qwing"]
    assert sessions.delivered == [("qwing", "fix")]


@pytest.mark.asyncio
async def test_make_inbound_urgent_without_name_prefix_still_targets_last_active():
    # "!fix" — no name prefix present, urgent still falls back to last-active
    # (unchanged behavior).
    store = FakeStore(last_active="qwing", enabled={"qwing": True})
    approvals = FakeApprovals()
    transcriber = FakeTranscriber()
    sessions = FakeSessions([FakeProject("qwing")])
    telegram = FakeTelegram()

    inbound = _inbound(transcriber, store, approvals, sessions, telegram)
    await inbound(_msg(text="!fix"))

    assert sessions.interrupt_calls == ["qwing"]
    assert sessions.delivered == [("qwing", "fix")]


@pytest.mark.asyncio
async def test_make_inbound_name_prefix_without_bang_is_not_urgent():
    # "qwing: fix it" — non-urgent name-prefixed message must NOT interrupt.
    store = FakeStore(last_active="othersapp", enabled={"qwing": True, "othersapp": True})
    approvals = FakeApprovals()
    transcriber = FakeTranscriber()
    sessions = FakeSessions([FakeProject("qwing"), FakeProject("othersapp")])
    telegram = FakeTelegram()

    inbound = _inbound(transcriber, store, approvals, sessions, telegram)
    await inbound(_msg(text="qwing: fix it"))

    assert sessions.interrupt_calls == []
    assert sessions.delivered == [("qwing", "fix it")]


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
async def test_make_outbound_transient_does_not_steal_last_active():
    # A transient send (heartbeat / "Working." status / verbose flush) must
    # not hijack routing away from whatever the user is actively talking to.
    store = FakeStore()
    tts_holder = {"backend": FakeTTS(out=b"V")}
    telegram = FakeTelegram(ids=[701])
    sessions = FakeSessions([FakeProject("qwing", voice="echo")])
    controls = _Controls(sessions, store, FakeCfg(), tts_holder)
    controls._mirror["qwing"] = {
        "enabled": True, "mode": "safe", "voice": "echo",
        "engine": "openai", "last_active": False,
    }

    outbound = make_outbound(tts_holder, telegram, store, FakeCfg(), sessions, controls)
    await outbound(
        Outbound(project="qwing", text="Working.", spoken=" ", transient=True)
    )

    # map_message still happens (a reply to this send must still resolve) ...
    assert store.mapped == [(701, "qwing")]
    # ... but last-active tracking is untouched by a transient send.
    assert store.last_active_calls == []
    assert controls._mirror["qwing"]["last_active"] is False


@pytest.mark.asyncio
async def test_make_outbound_non_transient_still_sets_last_active():
    store = FakeStore()
    tts_holder = {"backend": FakeTTS(out=b"V")}
    telegram = FakeTelegram(ids=[702])
    sessions = FakeSessions([FakeProject("qwing", voice="echo")])
    controls = _Controls(sessions, store, FakeCfg(), tts_holder)
    controls._mirror["qwing"] = {
        "enabled": True, "mode": "safe", "voice": "echo",
        "engine": "openai", "last_active": False,
    }

    outbound = make_outbound(tts_holder, telegram, store, FakeCfg(), sessions, controls)
    await outbound(Outbound(project="qwing", text="Done.", spoken="Done."))

    assert store.last_active_calls == ["qwing"]
    assert controls._mirror["qwing"]["last_active"] is True


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
async def test_controls_list_policies_delegates_to_store():
    controls, sessions, store, *_ = _make_controls()
    await store.add_policy("qwing", "git push")
    await store.add_policy("qwing", "rm")

    assert await controls.list_policies() == [("qwing", "git push"), ("qwing", "rm")]


@pytest.mark.asyncio
async def test_controls_clear_policies_all_and_by_project():
    controls, sessions, store, *_ = _make_controls()
    await store.add_policy("qwing", "git push")
    await store.add_policy("other", "rm")

    await controls.clear_policies("qwing")
    assert ("clear", "qwing", None) in store.policy_calls
    assert await store.list_policies() == [("other", "rm")]

    await controls.clear_policies(None)
    assert await store.list_policies() == []


@pytest.mark.asyncio
async def test_make_on_always_allow_persists_pending_policy():
    from voice_bridge.bridge import make_on_always_allow

    store = FakeStore()

    class _Approvals:
        def policy_for_token(self, token):
            return ("qwing", "git push") if token == 7 else None

    fn = make_on_always_allow(_Approvals(), store)
    assert await fn(7) is True  # persisted -> honest "Visada" label

    assert ("add", "qwing", "git push") in store.policy_calls
    assert await store.has_policy("qwing", "git push") is True


@pytest.mark.asyncio
async def test_make_on_always_allow_noop_for_unknown_token():
    from voice_bridge.bridge import make_on_always_allow

    store = FakeStore()

    class _Approvals:
        def policy_for_token(self, token):
            return None

    fn = make_on_always_allow(_Approvals(), store)
    assert await fn(123) is False  # unknown token -> nothing persisted, no raise

    assert store.policy_calls == []


@pytest.mark.asyncio
async def test_make_on_always_allow_skips_none_signature():
    # A NOT-policy-eligible call carries signature None; the tap degrades to
    # allow-once (nothing persisted) and the hook returns False.
    from voice_bridge.bridge import make_on_always_allow

    store = FakeStore()

    class _Approvals:
        def policy_for_token(self, token):
            return ("qwing", None)

    fn = make_on_always_allow(_Approvals(), store)
    assert await fn(7) is False
    assert store.policy_calls == []


@pytest.mark.asyncio
async def test_make_on_always_allow_swallows_store_failure():
    from voice_bridge.bridge import make_on_always_allow

    class _BoomStore:
        async def add_policy(self, project, signature):
            raise RuntimeError("db down")

    class _Approvals:
        def policy_for_token(self, token):
            return ("qwing", "git push")

    fn = make_on_always_allow(_Approvals(), _BoomStore())
    # a persist failure must never raise; it degrades to allow-once (False)
    assert await fn(7) is False


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


# --------------------------------------------------------------------------- #
# _sanitize_project_name (SECURITY, /newproject)
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("name", [
    "myproject",
    "My-Project_2.0",
    "a",
    "a" * 64,
    "under_score",
    "dots.and-dashes_9",
])
def test_sanitize_project_name_accepts_safe_names(name):
    assert _sanitize_project_name(name) == name


@pytest.mark.parametrize("name", [
    "",
    ".",
    "..",
    "../evil",
    "/abs",
    "a b",
    ".hidden",
    "-flag",
    "a" * 65,
    "foo/bar",
    "foo\nbar",
])
def test_sanitize_project_name_rejects_unsafe_names(name):
    assert _sanitize_project_name(name) == ""


# --------------------------------------------------------------------------- #
# Controls.create_project (/newproject)
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_controls_create_project_invalid_name_creates_nothing(tmp_path, monkeypatch):
    import voice_bridge.bridge as bridge_mod

    monkeypatch.setattr(bridge_mod.Path, "home", lambda: tmp_path)
    controls, sessions, store, *_ = _make_controls()
    await controls.seed()

    result = await controls.create_project("../evil")

    assert "Netinkamas" in result
    assert not (tmp_path / "Projects").exists()
    assert sessions.names() == ["qwing", "othersapp"]


@pytest.mark.asyncio
async def test_controls_create_project_fresh_creates_dir_git_init_and_selects(tmp_path, monkeypatch):
    import voice_bridge.bridge as bridge_mod

    monkeypatch.setattr(bridge_mod.Path, "home", lambda: tmp_path)
    monkeypatch.setattr(bridge_mod.shutil, "which", lambda name: "/usr/bin/git")
    git_calls = []

    class FakeProc:
        returncode = 0

        async def wait(self):
            return 0

    async def fake_exec(*args, **kwargs):
        git_calls.append((args, kwargs))
        return FakeProc()

    monkeypatch.setattr(bridge_mod.asyncio, "create_subprocess_exec", fake_exec)

    controls, sessions, store, *_ = _make_controls()
    await controls.seed()

    result = await controls.create_project("newapp")

    target = tmp_path / "Projects" / "newapp"
    assert target.is_dir()
    assert git_calls and git_calls[0][0][:2] == ("/usr/bin/git", "init")
    assert git_calls[0][1]["cwd"] == str(target)
    assert "newapp" in sessions.names()
    assert store.seeded and store.seeded[-1][0].name == "newapp"
    assert ("newapp", True) in sessions.enabled_calls
    snap = {row["project"]: row for row in controls.snapshot()}
    assert snap["newapp"]["last_active"] is True
    assert snap["newapp"]["cwd"] == str(target)
    assert "newapp" in result
    assert str(target) in result


@pytest.mark.asyncio
async def test_controls_create_project_missing_git_binary_is_non_fatal(tmp_path, monkeypatch):
    import voice_bridge.bridge as bridge_mod

    monkeypatch.setattr(bridge_mod.Path, "home", lambda: tmp_path)
    monkeypatch.setattr(bridge_mod.shutil, "which", lambda name: None)

    controls, sessions, store, *_ = _make_controls()
    await controls.seed()

    result = await controls.create_project("nogit")

    target = tmp_path / "Projects" / "nogit"
    assert target.is_dir()
    assert "nogit" in sessions.names()
    assert "nogit" in result


@pytest.mark.asyncio
async def test_controls_create_project_existing_registered_selects_without_mkdir(tmp_path, monkeypatch):
    import voice_bridge.bridge as bridge_mod

    monkeypatch.setattr(bridge_mod.Path, "home", lambda: tmp_path)
    controls, sessions, store, *_ = _make_controls()
    await controls.seed()
    # "qwing" is already a registered project in the mirror (from _make_controls)
    project_dir = tmp_path / "Projects" / "qwing"
    project_dir.mkdir(parents=True)

    result = await controls.create_project("qwing")

    assert "jau užregistruotas" in result
    snap = {row["project"]: row for row in controls.snapshot()}
    assert snap["qwing"]["last_active"] is True


@pytest.mark.asyncio
async def test_controls_create_project_registered_with_mismatched_cwd_not_recreated(
    tmp_path, monkeypatch
):
    """A registered project whose cwd basename differs from its name (e.g.
    qwing -> .../WhisperX) must take the "already registered" branch even
    though ~/Projects/<name> does NOT exist on disk — no stray dir, and the
    mirror's cwd/mode/voice must stay untouched (regression: the check used
    to be nested inside target.exists())."""
    import voice_bridge.bridge as bridge_mod

    monkeypatch.setattr(bridge_mod.Path, "home", lambda: tmp_path)
    controls, sessions, store, *_ = _make_controls()
    await controls.seed()
    # NOTE: ~/Projects/qwing deliberately NOT created — the registered
    # project's real cwd (from _make_controls) points elsewhere.
    before = {row["project"]: dict(row) for row in controls.snapshot()}["qwing"]

    result = await controls.create_project("qwing")

    assert "jau užregistruotas" in result
    assert not (tmp_path / "Projects" / "qwing").exists(), (
        "no stray directory may be created for an already-registered project"
    )
    after = {row["project"]: dict(row) for row in controls.snapshot()}["qwing"]
    assert after["cwd"] == before["cwd"]
    assert after["mode"] == before["mode"]
    assert after["voice"] == before["voice"]
    assert after["last_active"] is True


@pytest.mark.asyncio
async def test_controls_create_project_existing_on_disk_unregistered(tmp_path, monkeypatch):
    import voice_bridge.bridge as bridge_mod

    monkeypatch.setattr(bridge_mod.Path, "home", lambda: tmp_path)
    controls, sessions, store, *_ = _make_controls()
    await controls.seed()

    on_disk = tmp_path / "Projects" / "orphan"
    on_disk.mkdir(parents=True)

    result = await controls.create_project("orphan")

    assert "rastas diske" in result
    assert "orphan" in sessions.names()
    snap = {row["project"]: row for row in controls.snapshot()}
    assert snap["orphan"]["cwd"] == str(on_disk)
    assert snap["orphan"]["last_active"] is True


@pytest.mark.asyncio
async def test_controls_create_project_unexpected_error_returns_message_not_raise(tmp_path, monkeypatch):
    import voice_bridge.bridge as bridge_mod

    monkeypatch.setattr(bridge_mod.Path, "home", lambda: tmp_path)

    def boom_mkdir(self, *a, **k):
        raise OSError("disk full")

    monkeypatch.setattr(bridge_mod.Path, "mkdir", boom_mkdir)

    controls, sessions, store, *_ = _make_controls()
    await controls.seed()

    result = await controls.create_project("boom")

    assert "Nepavyko sukurti projekto boom" in result
    assert "boom" not in sessions.names()


@pytest.mark.asyncio
async def test_controls_create_project_fresh_persists_created(tmp_path, monkeypatch):
    # Task A: a freshly created project must be persisted (created=1) so it is
    # reloaded across restarts instead of vanishing.
    import voice_bridge.bridge as bridge_mod

    monkeypatch.setattr(bridge_mod.Path, "home", lambda: tmp_path)
    monkeypatch.setattr(bridge_mod.shutil, "which", lambda name: None)

    controls, sessions, store, *_ = _make_controls()
    await controls.seed()

    await controls.create_project("newapp")

    target = tmp_path / "Projects" / "newapp"
    assert (("newapp", str(target), None)) in store.created_calls


@pytest.mark.asyncio
async def test_controls_create_project_existing_on_disk_persists_created(tmp_path, monkeypatch):
    import voice_bridge.bridge as bridge_mod

    monkeypatch.setattr(bridge_mod.Path, "home", lambda: tmp_path)
    controls, sessions, store, *_ = _make_controls()
    await controls.seed()
    on_disk = tmp_path / "Projects" / "orphan"
    on_disk.mkdir(parents=True)

    await controls.create_project("orphan")

    assert (("orphan", str(on_disk), None)) in store.created_calls


@pytest.mark.asyncio
async def test_controls_create_project_already_registered_does_not_persist_created(tmp_path, monkeypatch):
    # An already-registered project (yaml or previously created) is neither
    # created nor newly registered -> no add_created_project.
    import voice_bridge.bridge as bridge_mod

    monkeypatch.setattr(bridge_mod.Path, "home", lambda: tmp_path)
    controls, sessions, store, *_ = _make_controls()
    await controls.seed()

    await controls.create_project("qwing")  # already in the mirror

    assert store.created_calls == []


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
async def test_controls_set_mode_warns_when_autonomy_persist_fails():
    # SECURITY: if the autonomy override can't be persisted, a demotion would
    # silently re-escalate on restart — the user must be told.
    controls, sessions, store, cfg, tts_holder = _make_controls()
    await controls.seed()
    telegram = FakeTelegram()
    controls.attach_telegram(telegram)

    async def boom(project, field, value):
        raise RuntimeError("db locked")

    store.set_override = boom
    await controls.set_mode("qwing", "safe")  # a demotion from yaml "full"

    # in-memory change still applied (never blocked)
    assert sessions.mode_calls == [("qwing", "safe")]
    # the user got BOTH the normal "mode changed" notice AND the persist warning
    texts = [t for _, t in telegram.questions]
    assert any("nepavyko išsaugoti" in t for t in texts)


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


# --------------------------------------------------------------------------- #
# Task A: persist runtime state on change (set_mode/effort/voice/verbose)
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_controls_set_mode_persists_autonomy_override():
    controls, sessions, store, cfg, tts_holder = _make_controls()
    await controls.seed()
    await controls.set_mode("qwing", "safe")
    assert ("qwing", "autonomy", "safe") in store.override_calls


@pytest.mark.asyncio
async def test_controls_set_effort_persists_override():
    controls, sessions, store, cfg, tts_holder = _make_controls()
    await controls.seed()
    await controls.set_effort("qwing", "high")
    assert ("qwing", "effort", "high") in store.override_calls


@pytest.mark.asyncio
async def test_controls_set_effort_invalid_level_does_not_persist():
    controls, sessions, store, cfg, tts_holder = _make_controls()
    await controls.seed()
    await controls.set_effort("qwing", "turbo")
    assert store.override_calls == []


@pytest.mark.asyncio
async def test_controls_set_voice_persists_override():
    controls, sessions, store, cfg, tts_holder = _make_controls()
    await controls.seed()
    await controls.set_voice("qwing", "sage")
    assert ("qwing", "voice", "sage") in store.override_calls


@pytest.mark.asyncio
async def test_controls_set_verbose_persists_override():
    controls, sessions, store, cfg, tts_holder = _make_controls()
    await controls.seed()
    await controls.set_verbose("qwing", True)
    assert ("qwing", "verbose", True) in store.override_calls


@pytest.mark.asyncio
async def test_controls_set_effort_all_projects_persists_each():
    controls, sessions, store, cfg, tts_holder = _make_controls()
    await controls.seed()
    await controls.set_effort(None, "max")
    persisted = {(p, v) for (p, f, v) in store.override_calls if f == "effort"}
    assert persisted == {("qwing", "max"), ("othersapp", "max")}


@pytest.mark.asyncio
async def test_controls_persist_failure_never_crashes_command(monkeypatch):
    # Persist-on-change is best-effort: a store write failure must not crash
    # the command nor block the in-memory change.
    controls, sessions, store, cfg, tts_holder = _make_controls()
    await controls.seed()

    async def boom(project, field, value):
        raise RuntimeError("db down")

    store.set_override = boom
    # Must not raise despite the failing store.
    await controls.set_voice("qwing", "sage")
    # in-memory change still applied
    assert sessions.project("qwing").voice == "sage"


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
                        lambda cfg, on_user_message, controls, on_approval=None, on_always_allow=None: telegram)

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
        lambda cfg, on_user_message, controls, on_approval=None, on_always_allow=None: telegram,
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

    def capture_tg(cfg_, on_user_message, controls, on_approval=None,
                   on_always_allow=None):
        captured["on_approval"] = on_approval
        captured["on_always_allow"] = on_always_allow
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
    # the always-allow persist hook is wired too
    assert callable(captured["on_always_allow"])

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


# --------------------------------------------------------------------------- #
# Task A: build() reloads created projects + applies persisted overrides
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_build_reloads_created_projects_and_applies_overrides(tmp_path, monkeypatch):
    import voice_bridge.bridge as bridge_mod
    from voice_bridge.config import ProjectConfig, effective_autonomy

    cfg = FakeCfg()
    # yaml boots qwing at the DANGEROUS default autonomy "full".
    yaml_projects = [ProjectConfig(name="qwing", cwd=str(tmp_path), autonomy="full")]
    created_dir = tmp_path / "newapp"
    created_dir.mkdir()
    store = FakeStore(
        enabled={"qwing": True, "newapp": True},
        overrides={"qwing": {"autonomy": "safe"}},
        created=[{"name": "newapp", "cwd": str(created_dir), "display_name": "New App"}],
    )

    captured: dict = {}

    def capture_sm(projects, *a, **k):
        captured["projects"] = list(projects)
        return FakeSessions(projects)

    monkeypatch.setattr(bridge_mod, "load_config", lambda env=None: cfg)
    monkeypatch.setattr(bridge_mod, "load_projects", lambda path="projects.yaml": yaml_projects)
    monkeypatch.setattr(bridge_mod, "Store", lambda db_path: store)
    monkeypatch.setattr(bridge_mod, "Transcriber", lambda m, language="lt": FakeTranscriber())
    monkeypatch.setattr(bridge_mod, "get_tts", lambda c: FakeTTS())
    monkeypatch.setattr(bridge_mod, "ApprovalManager", lambda sq, t: FakeApprovals())
    monkeypatch.setattr(bridge_mod, "SessionManager", capture_sm)
    monkeypatch.setattr(bridge_mod, "TelegramIO",
                        lambda c, oi, controls, on_approval=None, on_always_allow=None: FakeTelegram())

    await build()

    by_name = {p.name: p for p in captured["projects"]}
    # the created project was reloaded and merged as a ProjectConfig
    assert "newapp" in by_name
    assert by_name["newapp"].cwd == str(created_dir)
    # persisted override wins over yaml: qwing is demoted to safe BEFORE start
    assert effective_autonomy(by_name["qwing"], cfg) == "safe"


@pytest.mark.asyncio
async def test_build_skips_created_project_missing_on_disk(tmp_path, monkeypatch):
    import voice_bridge.bridge as bridge_mod
    from voice_bridge.config import ProjectConfig

    cfg = FakeCfg()
    yaml_projects = [ProjectConfig(name="qwing", cwd=str(tmp_path))]
    store = FakeStore(
        enabled={"qwing": True},
        created=[{"name": "gone", "cwd": str(tmp_path / "gone"), "display_name": None}],
    )

    captured: dict = {}

    def capture_sm(projects, *a, **k):
        captured["projects"] = list(projects)
        return FakeSessions(projects)

    monkeypatch.setattr(bridge_mod, "load_config", lambda env=None: cfg)
    monkeypatch.setattr(bridge_mod, "load_projects", lambda path="projects.yaml": yaml_projects)
    monkeypatch.setattr(bridge_mod, "Store", lambda db_path: store)
    monkeypatch.setattr(bridge_mod, "Transcriber", lambda m, language="lt": FakeTranscriber())
    monkeypatch.setattr(bridge_mod, "get_tts", lambda c: FakeTTS())
    monkeypatch.setattr(bridge_mod, "ApprovalManager", lambda sq, t: FakeApprovals())
    monkeypatch.setattr(bridge_mod, "SessionManager", capture_sm)
    monkeypatch.setattr(bridge_mod, "TelegramIO",
                        lambda c, oi, controls, on_approval=None, on_always_allow=None: FakeTelegram())

    await build()

    names = {p.name for p in captured["projects"]}
    assert "gone" not in names  # missing on disk -> skipped gracefully


@pytest.mark.asyncio
async def test_persisted_mode_override_survives_rebuild(tmp_path, monkeypatch):
    """SECURITY: /mode qwing safe then a restart -> qwing's effective autonomy
    is 'safe', NOT the yaml 'full'. A restart must never silently RE-ESCALATE a
    project the user demoted. Uses the REAL Store on a tmp db (the actual
    persist path), stubbing only the heavy runtime components."""
    import voice_bridge.bridge as bridge_mod
    from voice_bridge.config import ProjectConfig, effective_autonomy

    db = str(tmp_path / "state.db")

    class Cfg(FakeCfg):
        db_path = db

    cfg = Cfg()
    proj_dir = tmp_path / "qwing"
    proj_dir.mkdir()

    def fresh_projects(path="projects.yaml"):
        # yaml keeps qwing at the dangerous default "full" every boot.
        return [ProjectConfig(name="qwing", cwd=str(proj_dir), autonomy="full")]

    captured: dict = {}

    def capture_sm(projects, *a, **k):
        captured["projects"] = list(projects)
        return FakeSessions(projects)

    monkeypatch.setattr(bridge_mod, "load_config", lambda env=None: cfg)
    monkeypatch.setattr(bridge_mod, "load_projects", fresh_projects)
    # NOTE: bridge_mod.Store is left as the REAL Store (tmp db) on purpose.
    monkeypatch.setattr(bridge_mod, "Transcriber", lambda m, language="lt": FakeTranscriber())
    monkeypatch.setattr(bridge_mod, "get_tts", lambda c: FakeTTS())
    monkeypatch.setattr(bridge_mod, "ApprovalManager", lambda sq, t: FakeApprovals())
    monkeypatch.setattr(bridge_mod, "SessionManager", capture_sm)
    monkeypatch.setattr(bridge_mod, "TelegramIO",
                        lambda c, oi, controls, on_approval=None, on_always_allow=None: FakeTelegram())

    # First boot: qwing runs at the yaml autonomy "full".
    w1 = await build()
    assert effective_autonomy(captured["projects"][0], cfg) == "full"

    # User demotes qwing to safe at runtime (/mode qwing safe).
    await w1.controls.set_mode("qwing", "safe")

    # Restart: rebuild against the SAME db.
    await build()
    proj = captured["projects"][0]
    assert proj.name == "qwing"
    assert effective_autonomy(proj, cfg) == "safe"  # NOT re-escalated to "full"


# --------------------------------------------------------------------------- #
# make_inbound: I2 pending ask_user interception (answer by text/voice reply)
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_make_inbound_quote_reply_to_ask_resolves_it_no_deliver():
    store = FakeStore(by_message={}, last_active="qwing", enabled={"qwing": True})
    approvals = FakeApprovals()
    transcriber = FakeTranscriber()
    sessions = FakeSessions([FakeProject("qwing")])
    # A quote-reply to ask message 900 -> its token "7".
    telegram = FakeTelegram(ask_by_message={900: "7"})

    inbound = _inbound(transcriber, store, approvals, sessions, telegram)
    await inbound(_msg(reply_to=900, text="rollback"))

    assert telegram.resolved_asks == [("7", "rollback")]
    assert sessions.delivered == []


@pytest.mark.asyncio
async def test_make_inbound_quote_reply_to_ask_wins_even_with_multiple_pending():
    # The quote-reply path does NOT require the single-ask condition: a reply
    # to a specific ask message resolves it even if several are outstanding
    # (single_ask None here == "not exactly one pending").
    store = FakeStore(by_message={}, last_active="qwing", enabled={"qwing": True})
    approvals = FakeApprovals()
    transcriber = FakeTranscriber()
    sessions = FakeSessions([FakeProject("qwing")])
    telegram = FakeTelegram(ask_by_message={900: "7"}, single_ask=None)

    inbound = _inbound(transcriber, store, approvals, sessions, telegram)
    await inbound(_msg(reply_to=900, text="2"))

    assert telegram.resolved_asks == [("7", "2")]
    assert sessions.delivered == []


@pytest.mark.asyncio
async def test_make_inbound_single_pending_ask_plain_text_resolves_it():
    store = FakeStore(by_message={}, last_active="qwing", enabled={"qwing": True})
    approvals = FakeApprovals()
    transcriber = FakeTranscriber()
    sessions = FakeSessions([FakeProject("qwing")])
    telegram = FakeTelegram(single_ask="3")

    inbound = _inbound(transcriber, store, approvals, sessions, telegram)
    await inbound(_msg(reply_to=None, text="first one please"))

    assert telegram.resolved_asks == [("3", "first one please")]
    assert sessions.delivered == []


@pytest.mark.asyncio
async def test_make_inbound_single_pending_ask_voice_reply_resolves_it():
    store = FakeStore(by_message={}, last_active="qwing", enabled={"qwing": True})
    approvals = FakeApprovals()
    transcriber = FakeTranscriber(text="rollback it")
    sessions = FakeSessions([FakeProject("qwing")])
    telegram = FakeTelegram(single_ask="3")

    inbound = _inbound(transcriber, store, approvals, sessions, telegram)
    await inbound(_msg(reply_to=None, is_voice=True, audio=b"OGG"))

    assert transcriber.calls == [b"OGG"]
    assert telegram.resolved_asks == [("3", "rollback it")]
    assert sessions.delivered == []


@pytest.mark.asyncio
async def test_make_inbound_single_pending_ask_with_name_prefix_delivers_turn():
    # A leading "<project>:" is explicit routing intent, so even with one ask
    # outstanding the message is delivered as a NEW turn (not the answer).
    store = FakeStore(last_active="othersapp",
                      enabled={"qwing": True, "othersapp": True})
    approvals = FakeApprovals()
    transcriber = FakeTranscriber()
    sessions = FakeSessions([FakeProject("qwing"), FakeProject("othersapp")])
    telegram = FakeTelegram(single_ask="3")

    inbound = _inbound(transcriber, store, approvals, sessions, telegram)
    await inbound(_msg(reply_to=None, text="qwing: build"))

    assert telegram.resolved_asks == []
    assert sessions.delivered == [("qwing", "build")]


@pytest.mark.asyncio
async def test_make_inbound_urgent_name_prefix_not_hijacked_by_single_ask():
    # "!qwing: build" is an URGENT turn for qwing. A leading '!' must not hide
    # the name-prefix and let the single-pending-ask fallback swallow it — the
    # name-prefix guard checks the urgent-stripped text.
    store = FakeStore(last_active="othersapp",
                      enabled={"qwing": True, "othersapp": True})
    approvals = FakeApprovals()
    transcriber = FakeTranscriber()
    sessions = FakeSessions([FakeProject("qwing"), FakeProject("othersapp")])
    telegram = FakeTelegram(single_ask="3")

    inbound = _inbound(transcriber, store, approvals, sessions, telegram)
    await inbound(_msg(reply_to=None, text="!qwing: build"))

    assert telegram.resolved_asks == []
    assert sessions.delivered == [("qwing", "build")]


@pytest.mark.asyncio
async def test_make_inbound_two_pending_asks_plain_text_routes_normally():
    # Two asks outstanding -> single_pending_ask_token() is None, and there is
    # no quote-reply, so the message must NOT be hijacked: it routes as a turn.
    store = FakeStore(last_active="qwing", enabled={"qwing": True})
    approvals = FakeApprovals()
    transcriber = FakeTranscriber()
    sessions = FakeSessions([FakeProject("qwing")])
    telegram = FakeTelegram(single_ask=None)  # None == not exactly one pending

    inbound = _inbound(transcriber, store, approvals, sessions, telegram)
    await inbound(_msg(reply_to=None, text="continue"))

    assert telegram.resolved_asks == []
    assert sessions.delivered == [("qwing", "continue")]


@pytest.mark.asyncio
async def test_make_inbound_reply_to_project_while_ask_pending_routes_turn():
    # A quote-reply to a PROJECT message (not the ask) while a single ask is
    # pending elsewhere still routes as a turn -- the reply target is explicit.
    store = FakeStore(by_message={42: "qwing"}, enabled={"qwing": True})
    approvals = FakeApprovals()
    transcriber = FakeTranscriber()
    sessions = FakeSessions([FakeProject("qwing")])
    # 42 is NOT an ask message; single ask "3" is pending elsewhere.
    telegram = FakeTelegram(ask_by_message={}, single_ask="3")

    inbound = _inbound(transcriber, store, approvals, sessions, telegram)
    await inbound(_msg(reply_to=42, text="continue"))

    assert telegram.resolved_asks == []
    assert sessions.delivered == [("qwing", "continue")]


@pytest.mark.asyncio
async def test_make_inbound_resolve_ask_false_falls_through_to_routing():
    # resolve_ask returning False (empty/stale/already-answered) must fall
    # through to normal routing rather than swallowing the turn.
    store = FakeStore(by_message={}, last_active="qwing", enabled={"qwing": True})
    approvals = FakeApprovals()
    transcriber = FakeTranscriber()
    sessions = FakeSessions([FakeProject("qwing")])
    telegram = FakeTelegram(single_ask="3", resolve_ask_result=False)

    inbound = _inbound(transcriber, store, approvals, sessions, telegram)
    await inbound(_msg(reply_to=None, text="hello"))

    assert telegram.resolved_asks == [("3", "hello")]
    assert sessions.delivered == [("qwing", "hello")]


@pytest.mark.asyncio
async def test_make_inbound_pending_approval_takes_precedence_over_ask():
    # The approval interception runs FIRST: a reply to a pending approval is a
    # yes/no, never routed to an ask, even if an ask is also outstanding.
    store = FakeStore()
    approvals = FakeApprovals(pending=[55])
    transcriber = FakeTranscriber()
    sessions = FakeSessions()
    telegram = FakeTelegram(single_ask="3")

    inbound = _inbound(transcriber, store, approvals, sessions, telegram)
    await inbound(_msg(reply_to=55, text="yes"))

    assert approvals.resolved == [(55, True)]
    assert telegram.resolved_asks == []
    assert sessions.delivered == []
