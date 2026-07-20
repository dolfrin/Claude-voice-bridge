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


# ---------------------------------------------------------------------------
# Step 7: per-project token & cost usage (B3c)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_get_usage_zeros_for_unknown_project(tmp_db):
    store = Store(tmp_db)
    await store.init()

    assert await store.get_usage("ghost") == {
        "turns": 0,
        "input_tokens": 0,
        "output_tokens": 0,
        "cache_read_tokens": 0,
        "cache_creation_tokens": 0,
        "cost_usd": 0.0,
    }


@pytest.mark.asyncio
async def test_add_usage_creates_row_on_first_call(tmp_db):
    store = Store(tmp_db)
    await store.init()

    await store.add_usage(
        "qwing",
        cost_usd=0.0123,
        input_tokens=100,
        output_tokens=50,
        cache_read_tokens=10,
        cache_creation_tokens=5,
    )

    assert await store.get_usage("qwing") == {
        "turns": 1,
        "input_tokens": 100,
        "output_tokens": 50,
        "cache_read_tokens": 10,
        "cache_creation_tokens": 5,
        "cost_usd": pytest.approx(0.0123),
    }


@pytest.mark.asyncio
async def test_add_usage_accumulates_across_calls(tmp_db):
    store = Store(tmp_db)
    await store.init()

    await store.add_usage(
        "qwing", cost_usd=0.01, input_tokens=100, output_tokens=50,
        cache_read_tokens=10, cache_creation_tokens=5,
    )
    await store.add_usage(
        "qwing", cost_usd=0.02, input_tokens=200, output_tokens=75,
        cache_read_tokens=20, cache_creation_tokens=15,
    )

    usage = await store.get_usage("qwing")
    assert usage["turns"] == 2
    assert usage["input_tokens"] == 300
    assert usage["output_tokens"] == 125
    assert usage["cache_read_tokens"] == 30
    assert usage["cache_creation_tokens"] == 20
    assert usage["cost_usd"] == pytest.approx(0.03)


@pytest.mark.asyncio
async def test_add_usage_none_cost_adds_zero(tmp_db):
    store = Store(tmp_db)
    await store.init()

    await store.add_usage(
        "qwing", cost_usd=None, input_tokens=10, output_tokens=5,
    )
    await store.add_usage(
        "qwing", cost_usd=None, input_tokens=10, output_tokens=5,
    )

    usage = await store.get_usage("qwing")
    assert usage["cost_usd"] == 0.0
    assert usage["turns"] == 2
    assert usage["input_tokens"] == 20


@pytest.mark.asyncio
async def test_add_usage_defaults_cache_tokens_to_zero(tmp_db):
    store = Store(tmp_db)
    await store.init()

    await store.add_usage("qwing", cost_usd=0.1, input_tokens=1, output_tokens=1)

    usage = await store.get_usage("qwing")
    assert usage["cache_read_tokens"] == 0
    assert usage["cache_creation_tokens"] == 0


@pytest.mark.asyncio
async def test_all_usage_returns_every_project(tmp_db):
    store = Store(tmp_db)
    await store.init()

    await store.add_usage("qwing", cost_usd=0.1, input_tokens=1, output_tokens=1)
    await store.add_usage("othersapp", cost_usd=0.2, input_tokens=2, output_tokens=2)

    all_usage = await store.all_usage()
    assert set(all_usage) == {"qwing", "othersapp"}
    assert all_usage["qwing"]["input_tokens"] == 1
    assert all_usage["othersapp"]["input_tokens"] == 2


@pytest.mark.asyncio
async def test_all_usage_empty_when_none_recorded(tmp_db):
    store = Store(tmp_db)
    await store.init()

    assert await store.all_usage() == {}


@pytest.mark.asyncio
async def test_usage_survives_new_store_instance(tmp_db):
    s1 = Store(tmp_db)
    await s1.init()
    await s1.add_usage(
        "qwing", cost_usd=0.05, input_tokens=42, output_tokens=7,
        cache_read_tokens=3, cache_creation_tokens=1,
    )

    s2 = Store(tmp_db)
    await s2.init()

    usage = await s2.get_usage("qwing")
    assert usage["turns"] == 1
    assert usage["input_tokens"] == 42
    assert usage["cost_usd"] == pytest.approx(0.05)


# ---------------------------------------------------------------------------
# Step 8: per-project runtime overrides (autonomy/voice/verbose/effort) (Task A)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_overrides_empty_when_none_set(tmp_db):
    store = Store(tmp_db)
    await store.init()
    await store.seed([_proj("qwing", enabled=True)])

    assert await store.overrides() == {}


@pytest.mark.asyncio
async def test_set_override_round_trip_only_non_null_fields(tmp_db):
    store = Store(tmp_db)
    await store.init()
    await store.seed([_proj("qwing", enabled=True)])

    await store.set_override("qwing", "autonomy", "safe")
    await store.set_override("qwing", "effort", "high")

    assert await store.overrides() == {"qwing": {"autonomy": "safe", "effort": "high"}}


@pytest.mark.asyncio
async def test_set_override_verbose_stored_as_bool(tmp_db):
    store = Store(tmp_db)
    await store.init()
    await store.set_override("qwing", "verbose", True)

    assert await store.overrides() == {"qwing": {"verbose": True}}

    await store.set_override("qwing", "verbose", False)
    assert await store.overrides() == {"qwing": {"verbose": False}}


@pytest.mark.asyncio
async def test_set_override_creates_row_preserving_enabled_default(tmp_db):
    store = Store(tmp_db)
    await store.init()  # no seed
    await store.set_override("lazy", "voice", "sage")

    assert await store.overrides() == {"lazy": {"voice": "sage"}}
    assert await store.is_enabled("lazy") is True  # row created enabled=1


@pytest.mark.asyncio
async def test_set_override_rejects_unknown_field(tmp_db):
    store = Store(tmp_db)
    await store.init()
    with pytest.raises(ValueError):
        await store.set_override("qwing", "model", "claude-opus-4-8")


@pytest.mark.asyncio
async def test_override_survives_new_store_instance(tmp_db):
    s1 = Store(tmp_db)
    await s1.init()
    await s1.seed([_proj("qwing", enabled=True)])
    await s1.set_override("qwing", "autonomy", "safe")

    s2 = Store(tmp_db)
    await s2.init()
    assert await s2.overrides() == {"qwing": {"autonomy": "safe"}}


# ---------------------------------------------------------------------------
# Step 9: dynamically-created projects (/newproject persistence) (Task A)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_created_projects_empty_by_default(tmp_db):
    store = Store(tmp_db)
    await store.init()
    await store.seed([_proj("qwing", enabled=True)])  # a plain seeded project

    assert await store.created_projects() == []


@pytest.mark.asyncio
async def test_add_created_project_round_trip(tmp_db):
    store = Store(tmp_db)
    await store.init()
    await store.add_created_project("newapp", "/home/me/Projects/newapp", "New App")

    created = await store.created_projects()
    assert created == [
        {"name": "newapp", "cwd": "/home/me/Projects/newapp", "display_name": "New App"}
    ]
    # a created project is enabled by default so it boots on the next restart
    assert await store.is_enabled("newapp") is True


@pytest.mark.asyncio
async def test_add_created_project_preserves_enabled_toggle(tmp_db):
    store = Store(tmp_db)
    await store.init()
    await store.add_created_project("newapp", "/p/newapp", None)
    await store.set_enabled("newapp", False)
    # re-registering the same created project must not flip a user toggle back on
    await store.add_created_project("newapp", "/p/newapp", None)

    assert await store.is_enabled("newapp") is False


@pytest.mark.asyncio
async def test_created_projects_survive_new_store_instance(tmp_db):
    s1 = Store(tmp_db)
    await s1.init()
    await s1.add_created_project("newapp", "/p/newapp", None)

    s2 = Store(tmp_db)
    await s2.init()
    assert await s2.created_projects() == [
        {"name": "newapp", "cwd": "/p/newapp", "display_name": None}
    ]


# ---------------------------------------------------------------------------
# Step 10: schema migration must not destroy an existing (old-schema) db
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_init_migrates_old_projects_table_without_data_loss(tmp_db):
    # Simulate a pre-existing db created by the OLD schema (no override /
    # created columns), carrying real user state.
    async with aiosqlite.connect(tmp_db) as db:
        await db.executescript(
            """
            CREATE TABLE projects (
                name       TEXT PRIMARY KEY,
                enabled    INTEGER NOT NULL DEFAULT 1,
                session_id TEXT
            );
            """
        )
        await db.execute(
            "INSERT INTO projects (name, enabled, session_id) VALUES (?, ?, ?)",
            ("qwing", 0, "sess-old"),
        )
        await db.commit()

    # New code migrates additively: existing rows and their state survive.
    store = Store(tmp_db)
    await store.init()

    assert await store.is_enabled("qwing") is False
    assert await store.get_session_id("qwing") == "sess-old"
    # the new columns exist and read as "no override"
    assert await store.overrides() == {}
    await store.set_override("qwing", "autonomy", "safe")
    assert await store.overrides() == {"qwing": {"autonomy": "safe"}}


# ---------------------------------------------------------------------------
# Step 11: always-allow approval policies (project, signature)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_policies_empty_by_default(tmp_db):
    store = Store(tmp_db)
    await store.init()
    assert await store.list_policies() == []
    assert await store.has_policy("qwing", "git push") is False


@pytest.mark.asyncio
async def test_add_has_list_policy_round_trip(tmp_db):
    store = Store(tmp_db)
    await store.init()
    await store.add_policy("qwing", "git push")
    await store.add_policy("qwing", "npm install")

    assert await store.has_policy("qwing", "git push") is True
    assert await store.has_policy("qwing", "npm install") is True
    # a different signature / project is NOT covered
    assert await store.has_policy("qwing", "rm") is False
    assert await store.has_policy("other", "git push") is False

    assert await store.list_policies() == [
        ("qwing", "git push"),
        ("qwing", "npm install"),
    ]


@pytest.mark.asyncio
async def test_add_policy_is_idempotent(tmp_db):
    store = Store(tmp_db)
    await store.init()
    await store.add_policy("qwing", "git push")
    await store.add_policy("qwing", "git push")  # duplicate must not raise/duplicate

    assert await store.list_policies() == [("qwing", "git push")]


@pytest.mark.asyncio
async def test_clear_policy_all(tmp_db):
    store = Store(tmp_db)
    await store.init()
    await store.add_policy("qwing", "git push")
    await store.add_policy("other", "rm")

    await store.clear_policy()  # no args -> clear everything

    assert await store.list_policies() == []


@pytest.mark.asyncio
async def test_clear_policy_by_project(tmp_db):
    store = Store(tmp_db)
    await store.init()
    await store.add_policy("qwing", "git push")
    await store.add_policy("qwing", "rm")
    await store.add_policy("other", "rm")

    await store.clear_policy("qwing")

    assert await store.list_policies() == [("other", "rm")]


@pytest.mark.asyncio
async def test_clear_policy_by_project_and_signature(tmp_db):
    store = Store(tmp_db)
    await store.init()
    await store.add_policy("qwing", "git push")
    await store.add_policy("qwing", "rm")

    await store.clear_policy("qwing", "git push")

    assert await store.list_policies() == [("qwing", "rm")]


@pytest.mark.asyncio
async def test_policies_survive_new_store_instance(tmp_db):
    s1 = Store(tmp_db)
    await s1.init()
    await s1.add_policy("qwing", "git push")

    s2 = Store(tmp_db)
    await s2.init()
    assert await s2.has_policy("qwing", "git push") is True
    assert await s2.list_policies() == [("qwing", "git push")]


@pytest.mark.asyncio
async def test_init_adds_policy_table_to_old_db(tmp_db):
    # An old db with no approval_policy table must gain it on init (no data loss).
    async with aiosqlite.connect(tmp_db) as db:
        await db.executescript(
            """
            CREATE TABLE projects (
                name TEXT PRIMARY KEY,
                enabled INTEGER NOT NULL DEFAULT 1
            );
            """
        )
        await db.commit()

    store = Store(tmp_db)
    await store.init()
    # table now exists and is usable
    await store.add_policy("qwing", "git push")
    assert await store.has_policy("qwing", "git push") is True


# ---------------------------------------------------------------------------
# Step 12: scheduled/recurring turns (I4) — schedules table + methods
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_schedules_empty_by_default(tmp_db):
    store = Store(tmp_db)
    await store.init()
    assert await store.list_schedules() == []


@pytest.mark.asyncio
async def test_add_list_remove_enable_round_trip(tmp_db):
    store = Store(tmp_db)
    await store.init()
    sid = await store.add_schedule("qwing", "07:30", "check overnight CI")
    assert isinstance(sid, int) and sid > 0

    rows = await store.list_schedules()
    assert len(rows) == 1
    row = rows[0]
    assert row["id"] == sid
    assert row["project"] == "qwing"
    assert row["hhmm"] == "07:30"
    assert row["prompt"] == "check overnight CI"
    assert row["enabled"] is True
    assert row["last_run"] is None

    # a second one, listed ordered by project then hhmm
    sid2 = await store.add_schedule("qwing", "06:00", "morning ping")
    rows = await store.list_schedules()
    assert [r["hhmm"] for r in rows] == ["06:00", "07:30"]

    # filter by project
    await store.add_schedule("other", "10:00", "standup")
    assert [r["project"] for r in await store.list_schedules("qwing")] == [
        "qwing", "qwing"
    ]

    # disable/enable round-trip
    assert await store.set_schedule_enabled(sid, False) is True
    row = [r for r in await store.list_schedules() if r["id"] == sid][0]
    assert row["enabled"] is False
    assert await store.set_schedule_enabled(sid, True) is True
    row = [r for r in await store.list_schedules() if r["id"] == sid][0]
    assert row["enabled"] is True

    # remove
    assert await store.remove_schedule(sid2) is True
    assert sid2 not in {r["id"] for r in await store.list_schedules()}
    # removing an unknown id returns False
    assert await store.remove_schedule(9999) is False
    # toggling an unknown id returns False
    assert await store.set_schedule_enabled(9999, False) is False


@pytest.mark.asyncio
async def test_due_schedules_boundaries(tmp_db):
    store = Store(tmp_db)
    await store.init()
    at_now = await store.add_schedule("qwing", "07:30", "at now")      # hhmm == now
    before = await store.add_schedule("qwing", "07:00", "before now")  # hhmm < now
    after = await store.add_schedule("qwing", "09:00", "after now")    # hhmm > now
    disabled = await store.add_schedule("qwing", "06:00", "disabled")
    await store.set_schedule_enabled(disabled, False)

    due = await store.due_schedules("2026-07-21", "07:30")
    due_ids = {r["id"] for r in due}
    # hhmm == now and hhmm < now are due; hhmm > now and disabled are not.
    assert at_now in due_ids
    assert before in due_ids
    assert after not in due_ids
    assert disabled not in due_ids


@pytest.mark.asyncio
async def test_due_schedules_last_run_dedup(tmp_db):
    store = Store(tmp_db)
    await store.init()
    sid = await store.add_schedule("qwing", "07:30", "check CI")

    # not run yet -> due
    assert {r["id"] for r in await store.due_schedules("2026-07-21", "07:30")} == {sid}

    # marked ran TODAY -> not due again today
    await store.mark_schedule_ran(sid, "2026-07-21")
    assert await store.due_schedules("2026-07-21", "07:30") == []

    # a NEW day -> due again (last_run != today)
    assert {r["id"] for r in await store.due_schedules("2026-07-22", "07:30")} == {sid}


@pytest.mark.asyncio
async def test_mark_schedule_ran_persists(tmp_db):
    store = Store(tmp_db)
    await store.init()
    sid = await store.add_schedule("qwing", "07:30", "check CI")
    await store.mark_schedule_ran(sid, "2026-07-21")

    row = [r for r in await store.list_schedules() if r["id"] == sid][0]
    assert row["last_run"] == "2026-07-21"


@pytest.mark.asyncio
async def test_init_adds_schedules_table_to_old_db(tmp_db):
    # An old db with no schedules table must gain it on init (no data loss).
    async with aiosqlite.connect(tmp_db) as db:
        await db.executescript(
            """
            CREATE TABLE projects (
                name TEXT PRIMARY KEY,
                enabled INTEGER NOT NULL DEFAULT 1
            );
            """
        )
        await db.commit()

    store = Store(tmp_db)
    await store.init()
    # table now exists and is usable
    sid = await store.add_schedule("qwing", "07:30", "check CI")
    assert [r["id"] for r in await store.list_schedules()] == [sid]


@pytest.mark.asyncio
async def test_schedules_survive_new_store_instance(tmp_db):
    s1 = Store(tmp_db)
    await s1.init()
    sid = await s1.add_schedule("qwing", "07:30", "check CI")

    s2 = Store(tmp_db)
    await s2.init()
    rows = await s2.list_schedules()
    assert [r["id"] for r in rows] == [sid]
    assert rows[0]["prompt"] == "check CI"
