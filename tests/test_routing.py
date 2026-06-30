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
# Step 5: session_id round-trip
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_session_id_unset_is_none(tmp_db):
    store = Store(tmp_db)
    await store.init()
    await store.seed([_proj("qwing", enabled=True)])

    assert await store.get_session_id("qwing") is None


@pytest.mark.asyncio
async def test_session_id_unknown_project_is_none(tmp_db):
    store = Store(tmp_db)
    await store.init()

    assert await store.get_session_id("ghost") is None


@pytest.mark.asyncio
async def test_session_id_round_trip_and_overwrite(tmp_db):
    store = Store(tmp_db)
    await store.init()
    await store.seed([_proj("qwing", enabled=True)])
    await store.set_session_id("qwing", "sess-abc")
    assert await store.get_session_id("qwing") == "sess-abc"

    await store.set_session_id("qwing", "sess-def")
    assert await store.get_session_id("qwing") == "sess-def"


@pytest.mark.asyncio
async def test_set_session_id_creates_row_preserving_enabled(tmp_db):
    store = Store(tmp_db)
    await store.init()  # no seeded projects
    await store.set_session_id("lazyproj", "sess-1")

    assert await store.get_session_id("lazyproj") == "sess-1"
    # row created via DEFAULT enabled=1
    assert await store.is_enabled("lazyproj") is True


# ---------------------------------------------------------------------------
# Step 6: state survives new Store instance (restart survival)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_state_survives_new_store_instance(tmp_db):
    s1 = Store(tmp_db)
    await s1.init()
    await s1.seed([_proj("qwing", enabled=True)])
    await s1.set_enabled("qwing", False)
    await s1.set_session_id("qwing", "sess-xyz")
    await s1.map_message(42, "qwing")
    await s1.set_last_active("qwing")

    # simulate restart: fresh object, same file, init() must be non-destructive
    s2 = Store(tmp_db)
    await s2.init()
    await s2.seed([_proj("qwing", enabled=True)])

    assert await s2.is_enabled("qwing") is False
    assert await s2.get_session_id("qwing") == "sess-xyz"
    assert await s2.project_for_message(42) == "qwing"
    assert await s2.get_last_active() == "qwing"
