"""Unified-inbox spool drainer (I3): drain what the IDE notify hook writes.

The IDE notify hook (``hooks/voice-bridge-notify.py``) runs in a separate
Claude Code process and cannot speak — the bridge owns TTS. So it drops each
notification into a spool directory as one atomically-written JSON file, and
this module (running INSIDE the bridge, as a long-lived background task next to
the scheduler) drains that directory and hands each event to an ``emit``
closure that formats + speaks + sends it through Telegram.

Two pieces, mirroring ``scheduler.py`` so the core is deterministically
testable without a real clock or filesystem races:

* :func:`read_spool` — a pure-ish core: list, parse, and DELETE each ``*.json``
  file, returning the parsed events sorted by ``ts``. It NEVER raises: a missing
  directory yields ``[]``; a malformed or non-dict file is deleted (so a single
  bad file can never wedge the queue) and skipped.
* :func:`run_inbox` — a thin async loop that, each tick, drains the spool and
  ``await emit(event)`` for every event whose content ``hash`` is not already in
  a BOUNDED dedup set (the hook can emit two files for one logical event, e.g.
  PermissionRequest + PreToolUse), then waits via the INJECTED ``sleep_fn``.

Robustness contract: the loop must NEVER die or block on one bad event. A
failing ``emit`` is logged and the loop moves on; the dedup set is capped so it
cannot grow without bound over a long-running process.
"""
from __future__ import annotations

import asyncio
import json
import logging
from collections import OrderedDict
from pathlib import Path
from typing import Awaitable, Callable

logger = logging.getLogger(__name__)

# Cap on the dedup set so a long-lived bridge process cannot accumulate hashes
# forever. Oldest hashes are evicted first (FIFO) — a duplicate only ever
# arrives right after its twin, so a modest window is plenty.
_SEEN_MAX = 512


def read_spool(spool_dir) -> list[dict]:
    """List, parse and DELETE every ``*.json`` spool file; return events by ``ts``.

    Never raises. A missing/unreadable directory yields ``[]``. Each file is
    deleted whether or not it parsed — a malformed or non-dict file is removed
    and skipped so it can NEVER wedge the queue by being re-read every tick.
    Only ``*.json`` is globbed, so a half-written ``.tmp`` (pre-rename) is never
    touched. The returned list is sorted by ``ts`` so events are emitted in the
    order they occurred even if the filesystem lists them otherwise.
    """
    directory = Path(spool_dir)
    try:
        paths = sorted(directory.glob("*.json"))
    except OSError:
        return []

    events: list[dict] = []
    for path in paths:
        try:
            raw = path.read_text(encoding="utf-8")
        except OSError:
            # Could not even read it — leave it for a later tick rather than
            # deleting blind; a transient read error should not lose the event.
            continue
        # Delete BEFORE parsing: a repeatedly-unparseable file must not be
        # re-read (and re-fail) on every drain.
        try:
            path.unlink()
        except OSError:
            pass
        try:
            data = json.loads(raw)
        except (ValueError, TypeError):
            continue
        if isinstance(data, dict):
            events.append(data)

    events.sort(key=lambda e: e.get("ts") or 0)
    return events


async def run_inbox(
    spool_dir,
    emit: Callable[[dict], Awaitable[None]],
    stop_event: asyncio.Event,
    *,
    sleep_fn: Callable[[float], Awaitable[None]],
    interval: float = 3,
    seen: "OrderedDict | None" = None,
) -> None:
    """Drain the spool and emit unseen events until *stop_event* is set.

    Each iteration:

    1. :func:`read_spool` (parse + delete) — guarded so even a surprise error
       there just skips this tick rather than killing the loop.
    2. For each event whose ``hash`` is not already in ``seen``, ``await
       emit(event)`` inside try/except — a failing emit (TTS/Telegram down) is
       logged and the loop continues with the next event.
    3. Record the hash in the BOUNDED ``seen`` set (oldest evicted past
       ``_SEEN_MAX``) — AFTER the emit attempt, whether it succeeded or not: the
       spool file is already gone, so re-emitting a duplicate file for the same
       logical event would only spam. A genuinely new event carries a new hash.
    4. ``await sleep_fn(interval)``.

    ``seen`` is an ``OrderedDict`` used as an insertion-ordered bounded set
    (hash -> None); the caller may pass one in to share/inspect it, else a fresh
    one is created. ``sleep_fn`` is injected so tests drive the loop with no real
    sleeping (the bridge passes ``asyncio.sleep``).
    """
    if seen is None:
        seen = OrderedDict()

    while not stop_event.is_set():
        try:
            events = read_spool(spool_dir)
        except Exception:  # noqa: BLE001 - a drain hiccup must not kill the loop
            logger.exception("inbox: read_spool failed; skipping this tick")
            events = []

        for event in events:
            h = event.get("hash")
            if h is not None and h in seen:
                continue
            try:
                await emit(event)
            except Exception:  # noqa: BLE001 - one bad emit must not kill the loop
                logger.exception("inbox: emit failed for event %r", h)
            if h is not None:
                seen[h] = None
                while len(seen) > _SEEN_MAX:
                    seen.popitem(last=False)

        await sleep_fn(interval)
