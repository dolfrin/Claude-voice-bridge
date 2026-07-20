"""Scheduled / recurring turns (I4): deliver a prompt to a project once a day.

Two pieces, both intentionally free of any real clock so the core is
deterministically testable:

* :func:`parse_hhmm` — a pure normalizer/validator for user-typed "HH:MM"
  (``"7:5"`` -> ``"07:05"``), rejecting junk and out-of-range values.
* :func:`run_scheduler` — a thin async loop that, on each tick, asks the store
  which schedules are due (via the INJECTED ``now_fn`` clock), fires each one
  through ``deliver`` and posts a short ``notify`` line, then waits via the
  INJECTED ``sleep_fn``. Injecting ``now_fn``/``sleep_fn`` is what lets a test
  drive the loop with a fake clock and no real sleeping — the bridge passes the
  real local clock and ``asyncio.sleep``.

Robustness contract: the loop must NEVER die on one bad schedule or a failing
deliver. Each schedule is marked ran BEFORE its deliver so a deliver failure
can never cause a re-fire loop (the per-day dedup already stamped it), and the
deliver/notify calls are individually guarded so one exception is logged and
the loop moves on.
"""
from __future__ import annotations

import asyncio
import logging
import re
from typing import Awaitable, Callable

logger = logging.getLogger(__name__)

# A user-typed daily time: 1-2 digit hour, ':', 1-2 digit minute, surrounding
# whitespace tolerated. Range is validated separately so "24:00" / "12:60" are
# rejected rather than silently normalized.
_HHMM_RE = re.compile(r"^\s*(\d{1,2}):(\d{1,2})\s*$")


def parse_hhmm(text: object) -> str | None:
    """Normalize a user "HH:MM" string to zero-padded 24h form, or None.

    ``"7:5"`` -> ``"07:05"``, ``"9:00"`` -> ``"09:00"``; surrounding whitespace
    is stripped. Returns None (never raises) for anything that is not a valid
    00:00-23:59 time: non-strings, missing/extra colons, non-digits, or an
    out-of-range hour/minute. The zero-padded output is what makes the store's
    string ``hhmm <= now_hhmm`` comparison correct.
    """
    if not isinstance(text, str):
        return None
    m = _HHMM_RE.match(text)
    if m is None:
        return None
    hh, mm = int(m.group(1)), int(m.group(2))
    if not (0 <= hh <= 23 and 0 <= mm <= 59):
        return None
    return f"{hh:02d}:{mm:02d}"


async def run_scheduler(
    store,
    deliver: Callable[[str, str], Awaitable[None]],
    notify: Callable[[str, str], Awaitable[None]],
    stop_event: asyncio.Event,
    *,
    now_fn: Callable[[], tuple[str, str]],
    sleep_fn: Callable[[float], Awaitable[None]],
    interval: float = 30,
) -> None:
    """Fire due schedules once per local day, until *stop_event* is set.

    Each iteration: read ``(now_date, now_hhmm)`` from the injected ``now_fn``,
    ask the store for ``due_schedules``, and for each due schedule:

    1. ``mark_schedule_ran`` FIRST — so even if the deliver below fails, the
       per-day dedup is already stamped and the schedule cannot re-fire in a
       tight loop.
    2. ``await deliver(project, prompt)`` inside try/except — one bad schedule
       or a failing delivery must never kill the loop.
    3. ``await notify(project, prompt)`` (also guarded) to post a short "task
       fired" line back to the user.

    Then ``await sleep_fn(interval)`` before the next tick. ``now_fn`` and
    ``sleep_fn`` are injected so the whole loop is testable with a fake clock;
    the core never reads the wall clock or sleeps for real on its own.
    """
    while not stop_event.is_set():
        try:
            now_date, now_hhmm = now_fn()
            due = await store.due_schedules(now_date, now_hhmm)
        except Exception:  # noqa: BLE001 - a store hiccup must not kill the loop
            logger.exception("scheduler: due_schedules failed; skipping this tick")
            due = []

        for sched in due:
            sid = sched.get("id")
            project = sched.get("project")
            prompt = sched.get("prompt")
            # Mark ran FIRST (dedup before side effects): a deliver failure can
            # then never spin the loop re-firing the same schedule.
            try:
                await store.mark_schedule_ran(sid, now_date)
            except Exception:  # noqa: BLE001 - can't dedup? skip rather than risk a re-fire storm
                logger.exception(
                    "scheduler: mark_schedule_ran failed for %s; skipping", sid
                )
                continue
            try:
                await deliver(project, prompt)
            except Exception:  # noqa: BLE001 - one bad deliver must not kill the loop
                logger.exception(
                    "scheduler: deliver failed for schedule %s (%s)", sid, project
                )
                continue
            try:
                await notify(project, prompt)
            except Exception:  # noqa: BLE001 - a failed notice must not kill the loop
                logger.exception(
                    "scheduler: notify failed for schedule %s (%s)", sid, project
                )

        await sleep_fn(interval)
