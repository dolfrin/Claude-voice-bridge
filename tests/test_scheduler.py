"""TDD tests for voice_bridge.scheduler (I4: scheduled/recurring turns).

The core (``run_scheduler``) takes an INJECTED clock (``now_fn``) and sleep
(``sleep_fn``) so the loop is fully deterministic: no wall-clock reads, no real
sleeping. ``parse_hhmm`` is a pure normalizer/validator. A real
:class:`~voice_bridge.routing.Store` backs the loop tests so the per-day dedup
(``last_run``) and "fire again next day" behaviour are exercised end to end.
"""
from __future__ import annotations

import asyncio

import pytest

from voice_bridge.routing import Store
from voice_bridge.scheduler import parse_hhmm, run_scheduler


@pytest.fixture
def tmp_db(tmp_path):
    return str(tmp_path / "state.db")


# ---------------------------------------------------------------------------
# parse_hhmm
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "raw,expected",
    [
        ("07:30", "07:30"),
        ("7:5", "07:05"),
        ("0:0", "00:00"),
        ("00:00", "00:00"),
        ("23:59", "23:59"),
        ("9:00", "09:00"),
        ("  7:30  ", "07:30"),
    ],
)
def test_parse_hhmm_valid(raw, expected):
    assert parse_hhmm(raw) == expected


@pytest.mark.parametrize(
    "raw",
    [
        "",
        "abc",
        "24:00",
        "12:60",
        "25:61",
        "7",
        "730",
        "7:5:5",
        ":30",
        "30:",
        "-1:00",
        "07:5a",
        "7.30",
        None,
        1230,
    ],
)
def test_parse_hhmm_invalid(raw):
    assert parse_hhmm(raw) is None


# ---------------------------------------------------------------------------
# run_scheduler — helpers
# ---------------------------------------------------------------------------


def _collectors():
    delivered: list[tuple[str, str]] = []
    notified: list[tuple[str, str]] = []

    async def deliver(project, prompt):
        delivered.append((project, prompt))
        return True  # delivered (not skipped) -> notify fires

    async def notify(project, prompt):
        notified.append((project, prompt))

    return delivered, notified, deliver, notify


async def _run_once(store, deliver, notify, *nows):
    """Run exactly ``len(nows)`` loop iterations against a fake clock.

    ``sleep_fn`` sets the stop event after the last scheduled ``now`` value has
    been consumed, so the loop terminates deterministically with no real sleep.
    """
    stop = asyncio.Event()
    seq = list(nows)
    state = {"i": 0}

    def now_fn():
        idx = min(state["i"], len(seq) - 1)
        return seq[idx]

    async def sleep_fn(_interval):
        state["i"] += 1
        if state["i"] >= len(seq):
            stop.set()

    await run_scheduler(
        store, deliver, notify, stop,
        now_fn=now_fn, sleep_fn=sleep_fn, interval=0,
    )


# ---------------------------------------------------------------------------
# run_scheduler — behaviour
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_fires_exactly_the_due_schedules(tmp_db):
    store = Store(tmp_db)
    await store.init()
    due_id = await store.add_schedule("qwing", "07:30", "check CI")
    await store.add_schedule("qwing", "09:00", "later")  # not yet due at 07:30

    delivered, notified, deliver, notify = _collectors()
    await _run_once(store, deliver, notify, ("2026-07-21", "07:30"))

    assert delivered == [("qwing", "check CI")]
    assert notified == [("qwing", "check CI")]
    # the fired schedule is marked ran today; the other stays unrun.
    by_id = {s["id"]: s["last_run"] for s in await store.list_schedules()}
    assert by_id[due_id] == "2026-07-21"
    other = [s for s in await store.list_schedules() if s["id"] != due_id][0]
    assert other["last_run"] is None


@pytest.mark.asyncio
async def test_hhmm_after_now_does_not_fire(tmp_db):
    store = Store(tmp_db)
    await store.init()
    await store.add_schedule("qwing", "09:00", "later")

    delivered, _notified, deliver, notify = _collectors()
    await _run_once(store, deliver, notify, ("2026-07-21", "07:30"))

    assert delivered == []


@pytest.mark.asyncio
async def test_disabled_schedule_excluded(tmp_db):
    store = Store(tmp_db)
    await store.init()
    sid = await store.add_schedule("qwing", "06:00", "off one")
    await store.set_schedule_enabled(sid, False)

    delivered, _notified, deliver, notify = _collectors()
    await _run_once(store, deliver, notify, ("2026-07-21", "08:00"))

    assert delivered == []


@pytest.mark.asyncio
async def test_already_ran_today_does_not_refire(tmp_db):
    store = Store(tmp_db)
    await store.init()
    await store.add_schedule("qwing", "07:30", "check CI")

    delivered, _notified, deliver, notify = _collectors()
    # Two iterations SAME day, both past 07:30.
    await _run_once(
        store, deliver, notify,
        ("2026-07-21", "07:30"), ("2026-07-21", "07:45"),
    )

    assert delivered == [("qwing", "check CI")]  # fired once, not twice


@pytest.mark.asyncio
async def test_fires_again_next_day(tmp_db):
    store = Store(tmp_db)
    await store.init()
    await store.add_schedule("qwing", "07:30", "check CI")

    delivered, _notified, deliver, notify = _collectors()
    await _run_once(
        store, deliver, notify,
        ("2026-07-21", "07:30"), ("2026-07-22", "07:30"),
    )

    assert delivered == [("qwing", "check CI"), ("qwing", "check CI")]


@pytest.mark.asyncio
async def test_deliver_exception_does_not_stop_loop_or_refire(tmp_db):
    store = Store(tmp_db)
    await store.init()
    bad_id = await store.add_schedule("bad", "07:00", "boom")
    await store.add_schedule("good", "07:00", "ok")

    notified: list[tuple[str, str]] = []

    async def deliver(project, prompt):
        if project == "bad":
            raise RuntimeError("deliver down")
        return True

    async def notify(project, prompt):
        notified.append((project, prompt))

    # Two iterations same day: the failing schedule must be marked ran the
    # first time and NOT re-fire the second time; the good one still fires.
    await _run_once(
        store, deliver, notify,
        ("2026-07-21", "07:00"), ("2026-07-21", "07:05"),
    )

    # The good schedule delivered+notified exactly once; the bad one never
    # notified (deliver raised before notify).
    assert notified == [("good", "ok")]
    # The bad schedule was marked ran FIRST, so it will not re-fire despite the
    # deliver failure (no crash loop).
    by_id = {s["id"]: s["last_run"] for s in await store.list_schedules()}
    assert by_id[bad_id] == "2026-07-21"


@pytest.mark.asyncio
async def test_skipped_deliver_does_not_notify(tmp_db):
    # A deliver that returns False (project disabled/removed) means nothing ran,
    # so the "task launched" notice must NOT be posted — otherwise a schedule
    # outliving its project would ping a false notice every day.
    store = Store(tmp_db)
    await store.init()
    await store.add_schedule("gone", "07:00", "do it")
    notified: list[tuple[str, str]] = []

    async def deliver(project, prompt):
        return False  # skipped (disabled/unknown project)

    async def notify(project, prompt):
        notified.append((project, prompt))

    await _run_once(store, deliver, notify, ("2026-07-21", "07:01"))

    assert notified == []  # no false "launched" notice
    # It WAS marked ran (won't spin re-firing), just silently skipped.
    assert (await store.list_schedules())[0]["last_run"] == "2026-07-21"


@pytest.mark.asyncio
async def test_loop_exits_when_stop_already_set(tmp_db):
    store = Store(tmp_db)
    await store.init()
    await store.add_schedule("qwing", "07:30", "check CI")

    delivered, _notified, deliver, notify = _collectors()
    stop = asyncio.Event()
    stop.set()

    async def now_fn():  # pragma: no cover - must never be called
        raise AssertionError("now_fn called after stop set")

    async def sleep_fn(_):  # pragma: no cover - must never be called
        raise AssertionError("sleep_fn called after stop set")

    # A pre-set stop must exit before doing any work (no now_fn/sleep_fn call).
    await run_scheduler(
        store, deliver, notify, stop,
        now_fn=(lambda: ("2026-07-21", "07:30")), sleep_fn=sleep_fn,
    )
    assert delivered == []
