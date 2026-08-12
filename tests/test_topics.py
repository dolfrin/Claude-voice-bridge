"""Group mode: one forum topic per project, routing by topic."""

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

from voice_bridge.bridge import resolve_target
from voice_bridge.config import Config, ProjectConfig
from voice_bridge.routing import Conversation, Store
from voice_bridge.telegram_io import TelegramIO


def make_cfg(chat_id=None):
    return Config(
        telegram_bot_token="TESTTOKEN",
        telegram_allowed_user_id=42,
        anthropic_api_key="ak",
        openai_api_key="ok",
        together_api_key="tk",
        together_tts_model="cartesia/sonic",
        together_tts_language="lt",
        tts_backend="together",
        tts_voice="alloy",
        piper_voice_path="",
        whisper_model="large-v3",
        autonomy_mode="safe",
        approval_timeout=300,
        db_path=":memory:",
        telegram_chat_id=chat_id,
    )


class FakeStore:
    def __init__(self, existing=None):
        self.saved: list[tuple[str, int, int]] = []
        self._existing = dict(existing or {})

    async def topics_for_chat(self, chat_id):
        return dict(self._existing)

    async def set_topic(self, project, chat_id, thread_id):
        self.saved.append((project, chat_id, thread_id))


def make_io(chat_id=None, store=None, projects=(), topics=None):
    io = TelegramIO(make_cfg(chat_id), AsyncMock(), MagicMock(),
                    store=store, projects=list(projects))
    if topics:
        io._topics = dict(topics)
    return io


# --------------------------------------------------------------------------
# routing
# --------------------------------------------------------------------------
class RoutingStore:
    def __init__(self, by_message=None, last_active=None):
        self._by_message = dict(by_message or {})
        self._last_active = last_active

    async def project_for_message(self, mid):
        return self._by_message.get(mid)

    async def get_last_active(self):
        return self._last_active

    async def is_enabled(self, project):
        return True

    async def conversations(self, project=None, include_closed=False):
        # Every project in these tests has exactly its #1 open.
        return [Conversation(f"{project}#1", project, 1, None, None, None, False, 0.0)]


async def test_transcript_choice_goes_back_to_its_own_topic():
    # Regression: the choice was sent with a hardcoded "stt" label, which is not
    # a project, so _dest found no topic and every prompt landed in General.
    from voice_bridge.bridge import _pick_transcript

    class Recorder:
        def __init__(self):
            self.project = None

        async def ask_per_message(self, project, options, button="Accept this"):
            self.project = project
            return "base-turbo"

    telegram = Recorder()
    results = [
        {"model": "paprika-lt", "text": "a"},
        {"model": "base-turbo", "text": "b"},
    ]

    assert await _pick_transcript(results, telegram, "paprika") == "b"
    assert telegram.project == "paprika"


async def test_topic_beats_reply_and_last_active():
    # The exact failure this feature exists to prevent: work meant for one
    # project landing in whichever was active last.
    store = RoutingStore(by_message={7: "rektbot"}, last_active="rektbot")
    msg = {"project": "paprika#1", "reply_to": 7, "text": "go"}

    assert await resolve_target(msg, store) == ("paprika#1", "ok")


async def test_without_a_topic_the_old_routing_still_applies():
    store = RoutingStore(by_message={7: "rektbot"}, last_active="mayhem")

    assert await resolve_target({"project": None, "reply_to": 7}, store) == (
        "rektbot#1", "ok",
    )
    assert await resolve_target({"project": None, "reply_to": None}, store) == (
        "mayhem#1", "ok",
    )


# --------------------------------------------------------------------------
# destination + reverse lookup
# --------------------------------------------------------------------------
async def test_private_mode_sends_to_the_user_and_names_no_topic():
    io = make_io(chat_id=None)

    assert io._dest("paprika") == {"chat_id": 42}


async def test_group_mode_sends_into_the_project_topic():
    io = make_io(chat_id=-100123, topics={"paprika": 5})

    assert io._dest("paprika") == {"chat_id": -100123, "message_thread_id": 5}


async def test_project_without_a_topic_falls_back_to_general():
    io = make_io(chat_id=-100123, topics={"paprika": 5})

    # No message_thread_id at all rather than a failed send.
    assert io._dest("unknown") == {"chat_id": -100123}
    assert io._dest(None) == {"chat_id": -100123}


async def test_incoming_thread_resolves_to_its_project():
    io = make_io(chat_id=-100123, topics={"paprika": 5, "rektbot": 9})

    assert io.project_for_thread(9) == "rektbot"
    assert io.project_for_thread(404) is None
    assert io.project_for_thread(None) is None


# --------------------------------------------------------------------------
# topic creation
# --------------------------------------------------------------------------
async def test_missing_topics_are_created_and_remembered():
    store = FakeStore(existing={"paprika": 5})
    projects = [
        ProjectConfig(name="paprika", cwd="/tmp/a", display_name="Paprika ASR"),
        ProjectConfig(name="rektbot", cwd="/tmp/b", display_name="REKT Bot"),
    ]
    io = make_io(chat_id=-100123, store=store, projects=projects)
    io.app = MagicMock()
    io.app.bot.create_forum_topic = AsyncMock(
        return_value=SimpleNamespace(message_thread_id=9)
    )

    await io.ensure_topics()

    # paprika already had one, so only rektbot is created.
    io.app.bot.create_forum_topic.assert_awaited_once_with(
        chat_id=-100123, name="REKT Bot"
    )
    assert store.saved == [("rektbot", -100123, 9)]
    assert io._topics == {"paprika": 5, "rektbot": 9}


async def test_a_failed_topic_does_not_stop_startup():
    store = FakeStore()
    projects = [ProjectConfig(name="paprika", cwd="/tmp/a")]
    io = make_io(chat_id=-100123, store=store, projects=projects)
    io.app = MagicMock()
    io.app.bot.create_forum_topic = AsyncMock(side_effect=RuntimeError("not admin"))

    await io.ensure_topics()

    assert store.saved == []
    assert io._topics == {}


async def test_private_mode_creates_nothing():
    store = FakeStore()
    io = make_io(chat_id=None, store=store,
                 projects=[ProjectConfig(name="paprika", cwd="/tmp/a")])
    io.app = MagicMock()
    io.app.bot.create_forum_topic = AsyncMock()

    await io.ensure_topics()

    io.app.bot.create_forum_topic.assert_not_awaited()
    assert store.saved == []


# --------------------------------------------------------------------------
# persistence
# --------------------------------------------------------------------------
async def test_topics_are_scoped_to_their_chat(tmp_path):
    store = Store(str(tmp_path / "state.db"))
    await store.init()
    await store.set_topic("paprika", -100123, 5)
    await store.set_topic("rektbot", -100123, 9)
    await store.set_topic("paprika", -100999, 77)  # same project, other group

    assert await store.topics_for_chat(-100123) == {"rektbot": 9}
    assert await store.topics_for_chat(-100999) == {"paprika": 77}
