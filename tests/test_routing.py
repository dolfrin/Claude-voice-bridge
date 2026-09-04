"""Tests for voice_bridge.routing.Store (SQLite-backed routing/state)."""
import pytest
import aiosqlite

from voice_bridge.config import ProjectConfig
from voice_bridge.routing import Store


@pytest.fixture
def tmp_db(tmp_path):
    return str(tmp_path / "state.db")


def _proj(name, enabled=True):
    return ProjectConfig(name=name, cwd=f"/p/{name}", enabled=enabled)


# ---------------------------------------------------------------------------
# Step 1: init creates tables
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_init_creates_tables(tmp_db):
    store = Store(tmp_db)
    await store.init()

    async with aiosqlite.connect(tmp_db) as db:
        db.row_factory = aiosqlite.Row
        cur = await db.execute(
            "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
        )
        rows = [r["name"] for r in await cur.fetchall()]

    assert "messages" in rows
    assert "projects" in rows
    assert "meta" in rows


# ---------------------------------------------------------------------------
# Step 2: seed with enabled defaults + idempotency
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_seed_uses_enabled_defaults(tmp_db):
    store = Store(tmp_db)
    await store.init()
    await store.seed([_proj("qwing", enabled=True), _proj("othersapp", enabled=False)])

    assert await store.enabled_map() == {"qwing": True, "othersapp": False}


@pytest.mark.asyncio
async def test_seed_is_idempotent_and_preserves_state(tmp_db):
    store = Store(tmp_db)
    await store.init()
    await store.seed([_proj("qwing", enabled=True)])
    # user disabled it at runtime
    await store.set_enabled("qwing", False)
    # re-seed (e.g. restart) must NOT flip it back to the yaml default
    await store.seed([_proj("qwing", enabled=True)])

    assert await store.is_enabled("qwing") is False


@pytest.mark.asyncio
async def test_is_enabled_false_for_unseeded(tmp_db):
    store = Store(tmp_db)
    await store.init()

    assert await store.is_enabled("ghost") is False


# ---------------------------------------------------------------------------
# Step 3: message->project map
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_map_message_round_trip(tmp_db):
    store = Store(tmp_db)
    await store.init()
    await store.map_message(1001, "qwing")

    assert await store.project_for_message(1001) == "qwing"


@pytest.mark.asyncio
async def test_project_for_unknown_message_is_none(tmp_db):
    store = Store(tmp_db)
    await store.init()

    assert await store.project_for_message(999) is None


@pytest.mark.asyncio
async def test_map_message_upserts_existing_id(tmp_db):
    store = Store(tmp_db)
    await store.init()
    await store.map_message(1001, "qwing")
    await store.map_message(1001, "othersapp")

    assert await store.project_for_message(1001) == "othersapp"


# ---------------------------------------------------------------------------
# Step 4: last_active round-trip (meta table)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_last_active_unset_is_none(tmp_db):
    store = Store(tmp_db)
    await store.init()

    assert await store.get_last_active() is None


@pytest.mark.asyncio
async def test_last_active_round_trip_and_overwrite(tmp_db):
    store = Store(tmp_db)
    await store.init()
    await store.set_last_active("qwing")
    assert await store.get_last_active() == "qwing"

    await store.set_last_active("othersapp")
    assert await store.get_last_active() == "othersapp"


@pytest.mark.asyncio
async def test_last_active_stored_in_meta(tmp_db):
    store = Store(tmp_db)
    await store.init()
    await store.set_last_active("qwing")

    async with aiosqlite.connect(tmp_db) as db:
        cur = await db.execute("SELECT value FROM meta WHERE key = 'last_active'")
        row = await cur.fetchone()

    assert row is not None and row[0] == "qwing"


# ---------------------------------------------------------------------------
# Step 5: conversations — one Claude session each
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_ordinals_start_at_one_and_never_repeat(tmp_db):
    # A reused number would put a new conversation under an old topic title.
    store = Store(tmp_db)
    await store.init()

    assert await store.next_ordinal("qwing") == 1
    await store.add_conversation("qwing#1", "qwing", 1)
    assert await store.next_ordinal("qwing") == 2

    await store.add_conversation("qwing#2", "qwing", 2)
    await store.close_conversation("qwing#2")

    assert await store.next_ordinal("qwing") == 3


@pytest.mark.asyncio
async def test_an_old_database_keeps_its_running_sessions(tmp_db):
    # Sessions used to live on the project row. Dropping them on upgrade would
    # silently reset the context of every project that was mid-task.
    async with aiosqlite.connect(tmp_db) as db:
        await db.executescript(
            "CREATE TABLE projects (name TEXT PRIMARY KEY, enabled INTEGER "
            "NOT NULL DEFAULT 1, session_id TEXT);"
        )
        await db.execute(
            "INSERT INTO projects VALUES ('paprika', 1, 'uuid-live'), "
            "('idle', 0, NULL)"
        )
        await db.commit()

    store = Store(tmp_db)
    await store.init()

    assert (await store.conversation("paprika#1")).session_id == "uuid-live"
    assert (await store.conversation("idle#1")).session_id is None


@pytest.mark.asyncio
async def test_the_migration_runs_only_once(tmp_db):
    # A second pass must not resurrect conversations the user has closed.
    async with aiosqlite.connect(tmp_db) as db:
        await db.executescript(
            "CREATE TABLE projects (name TEXT PRIMARY KEY, enabled INTEGER "
            "NOT NULL DEFAULT 1, session_id TEXT);"
        )
        await db.execute("INSERT INTO projects VALUES ('paprika', 1, 'uuid-live')")
        await db.commit()

    await Store(tmp_db).init()
    store = Store(tmp_db)
    await store.close_conversation("paprika#1")
    await store.init()

    assert await store.conversations("paprika") == []
    assert (await store.conversation("paprika#1")).closed is True


@pytest.mark.asyncio
async def test_a_fresh_database_migrates_nothing(tmp_db):
    store = Store(tmp_db)
    await store.init()

    assert await store.conversations(include_closed=True) == []


@pytest.mark.asyncio
async def test_conversation_round_trip(tmp_db):
    store = Store(tmp_db)
    await store.init()
    await store.add_conversation("qwing#1", "qwing", 1)
    await store.set_conversation_topic("qwing#1", -100123, 77)
    await store.set_conversation_session("qwing#1", "sess-abc")

    conv = await store.conversation("qwing#1")

    assert conv.project == "qwing"
    assert conv.ordinal == 1
    assert (conv.chat_id, conv.thread_id) == (-100123, 77)
    assert conv.session_id == "sess-abc"
    assert conv.closed is False


@pytest.mark.asyncio
async def test_closed_conversations_are_hidden_by_default(tmp_db):
    store = Store(tmp_db)
    await store.init()
    await store.add_conversation("qwing#1", "qwing", 1)
    await store.add_conversation("qwing#2", "qwing", 2)
    await store.add_conversation("other#1", "other", 1)
    await store.close_conversation("qwing#2")

    assert [c.key for c in await store.conversations("qwing")] == ["qwing#1"]
    assert [c.key for c in await store.conversations("qwing", include_closed=True)] == [
        "qwing#1", "qwing#2",
    ]
    assert {c.key for c in await store.conversations()} == {"qwing#1", "other#1"}


@pytest.mark.asyncio
async def test_a_claude_session_is_found_by_its_uuid(tmp_db):
    # Resuming a session that is already open must land in its topic, not spawn
    # a second writer for the same .jsonl.
    store = Store(tmp_db)
    await store.init()
    await store.add_conversation("qwing#1", "qwing", 1, session_id="uuid-1")

    found = await store.conversation_for_session("uuid-1")

    assert found is not None and found.key == "qwing#1"
    assert await store.conversation_for_session("uuid-nope") is None


# ---------------------------------------------------------------------------
# Step 6: state survives new Store instance (restart survival)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_state_survives_new_store_instance(tmp_db):
    s1 = Store(tmp_db)
    await s1.init()
    await s1.seed([_proj("qwing", enabled=True)])
    await s1.set_enabled("qwing", False)
    await s1.add_conversation("qwing#1", "qwing", 1, thread_id=9)
    await s1.set_conversation_session("qwing#1", "sess-xyz")
    await s1.map_message(42, "qwing#1")
    await s1.set_last_active("qwing#1")

    # simulate restart: fresh object, same file, init() must be non-destructive
    s2 = Store(tmp_db)
    await s2.init()
    await s2.seed([_proj("qwing", enabled=True)])

    assert await s2.is_enabled("qwing") is False
    conv = await s2.conversation("qwing#1")
    assert conv.session_id == "sess-xyz"
    assert conv.thread_id == 9
    assert await s2.project_for_message(42) == "qwing#1"
    assert await s2.get_last_active() == "qwing#1"
