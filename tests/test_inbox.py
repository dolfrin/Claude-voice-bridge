"""TDD tests for voice_bridge.inbox (I3: unified-inbox spool drainer).

Mirrors the scheduler's design: a pure core (:func:`read_spool`) plus a thin
async loop (:func:`run_inbox`) that takes an INJECTED ``sleep_fn`` so the loop
is fully deterministic — no real sleeping. ``read_spool`` must be crash-proof:
a malformed or non-dict file is deleted (so it can't wedge the queue) and
skipped, and a missing directory yields ``[]``. ``run_inbox`` dedups by content
hash across drains (the hook can emit two files for one logical event) with a
BOUNDED ``seen`` set, and a single failing ``emit`` must never stop the loop.
"""
from __future__ import annotations

import asyncio
import json
from collections import OrderedDict

import pytest

import voice_bridge.inbox as inbox
from voice_bridge.inbox import read_spool, run_inbox


def _write_event(directory, *, hash="h", ts=1.0, kind="stop", text="x",
                 project="qwing", urgent=False, name=None):
    directory.mkdir(parents=True, exist_ok=True)
    name = name or f"{ts}-{hash}"
    (directory / f"{name}.json").write_text(json.dumps({
        "kind": kind, "project": project, "cwd": "/x/" + project,
        "text": text, "urgent": urgent, "hash": hash, "ts": ts,
    }), encoding="utf-8")


# ---------------------------------------------------------------------------
# read_spool
# ---------------------------------------------------------------------------

def test_read_spool_parses_and_deletes(tmp_path):
    _write_event(tmp_path, hash="a", ts=1)
    events = read_spool(tmp_path)
    assert [e["hash"] for e in events] == ["a"]
    # the file is consumed (deleted) so a second drain sees nothing.
    assert read_spool(tmp_path) == []


def test_read_spool_sorts_by_ts(tmp_path):
    _write_event(tmp_path, hash="late", ts=30, name="late")
    _write_event(tmp_path, hash="early", ts=10, name="early")
    _write_event(tmp_path, hash="mid", ts=20, name="mid")
    events = read_spool(tmp_path)
    assert [e["hash"] for e in events] == ["early", "mid", "late"]


def test_read_spool_skips_and_deletes_malformed(tmp_path):
    tmp_path.mkdir(parents=True, exist_ok=True)
    (tmp_path / "bad.json").write_text("{ not json", encoding="utf-8")
    _write_event(tmp_path, hash="good", ts=5, name="good")
    events = read_spool(tmp_path)
    assert [e["hash"] for e in events] == ["good"]
    # the malformed file is deleted too, so it can never wedge the queue.
    assert list(tmp_path.glob("*.json")) == []


def test_read_spool_skips_non_dict_json(tmp_path):
    tmp_path.mkdir(parents=True, exist_ok=True)
    (tmp_path / "list.json").write_text("[1, 2, 3]", encoding="utf-8")
    (tmp_path / "str.json").write_text('"hello"', encoding="utf-8")
    assert read_spool(tmp_path) == []
    assert list(tmp_path.glob("*.json")) == []  # both deleted


def test_read_spool_ignores_tmp_files(tmp_path):
    tmp_path.mkdir(parents=True, exist_ok=True)
    # A half-written temp file (pre-rename) must never be read or deleted.
    (tmp_path / "1-a.tmp").write_text("{incomplete", encoding="utf-8")
    _write_event(tmp_path, hash="a", ts=1, name="done")
    events = read_spool(tmp_path)
    assert [e["hash"] for e in events] == ["a"]
    assert (tmp_path / "1-a.tmp").exists()  # temp left untouched


def test_read_spool_missing_dir_returns_empty(tmp_path):
    assert read_spool(tmp_path / "nope") == []


# ---------------------------------------------------------------------------
# run_inbox — helpers
# ---------------------------------------------------------------------------

async def _drive(spool_dir, emit, *, iterations=1, seen=None, on_sleep=None):
    """Run exactly ``iterations`` loop iterations against an injected sleep_fn.

    ``on_sleep(i)`` (optional) lets a test mutate the spool between drains to
    exercise cross-drain dedup. The stop event is set once ``iterations`` sleeps
    have happened, so the loop terminates with no real sleeping.
    """
    stop = asyncio.Event()
    state = {"i": 0}

    async def sleep_fn(_interval):
        state["i"] += 1
        if on_sleep is not None:
            on_sleep(state["i"])
        if state["i"] >= iterations:
            stop.set()

    await run_inbox(spool_dir, emit, stop, sleep_fn=sleep_fn, interval=0, seen=seen)


# ---------------------------------------------------------------------------
# run_inbox — behaviour
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_run_inbox_emits_unseen_events(tmp_path):
    _write_event(tmp_path, hash="a", ts=1, name="a")
    _write_event(tmp_path, hash="b", ts=2, name="b")
    got = []

    async def emit(ev):
        got.append(ev["hash"])

    await _drive(tmp_path, emit, iterations=1)
    assert got == ["a", "b"]


@pytest.mark.asyncio
async def test_run_inbox_passes_full_event_to_emit(tmp_path):
    _write_event(tmp_path, hash="q", ts=1, kind="question", urgent=True,
                 text="Deploy?", name="q")
    got = []

    async def emit(ev):
        got.append(ev)

    await _drive(tmp_path, emit, iterations=1)
    assert got[0]["urgent"] is True
    assert got[0]["kind"] == "question"
    assert got[0]["text"] == "Deploy?"


@pytest.mark.asyncio
async def test_run_inbox_dedups_same_hash_within_one_drain(tmp_path):
    _write_event(tmp_path, hash="same", ts=1, name="one")
    _write_event(tmp_path, hash="same", ts=2, name="two")
    got = []

    async def emit(ev):
        got.append(ev["hash"])

    await _drive(tmp_path, emit, iterations=1)
    assert got == ["same"]  # two files, one logical event -> emitted once


@pytest.mark.asyncio
async def test_run_inbox_dedups_same_hash_across_drains(tmp_path):
    _write_event(tmp_path, hash="H", ts=1, name="first")
    got = []

    async def emit(ev):
        got.append(ev["hash"])

    def on_sleep(i):
        if i == 1:  # after the first drain, the hook re-emits the same event
            _write_event(tmp_path, hash="H", ts=2, name="second")

    await _drive(tmp_path, emit, iterations=2, on_sleep=on_sleep)
    assert got == ["H"]  # deduped across drains


@pytest.mark.asyncio
async def test_run_inbox_bad_emit_does_not_stop_loop(tmp_path):
    _write_event(tmp_path, hash="boom", ts=1, name="boom")
    _write_event(tmp_path, hash="ok", ts=2, name="ok")
    got = []

    async def emit(ev):
        if ev["hash"] == "boom":
            raise RuntimeError("emit down")
        got.append(ev["hash"])

    await _drive(tmp_path, emit, iterations=1)
    assert got == ["ok"]  # the good event still emits despite the bad one


@pytest.mark.asyncio
async def test_run_inbox_seen_set_is_bounded(tmp_path, monkeypatch):
    monkeypatch.setattr(inbox, "_SEEN_MAX", 3)
    for i in range(6):
        _write_event(tmp_path, hash=f"h{i}", ts=i, name=f"e{i}")
    seen = OrderedDict()
    got = []

    async def emit(ev):
        got.append(ev["hash"])

    await _drive(tmp_path, emit, iterations=1, seen=seen)
    assert len(got) == 6  # all emitted
    assert len(seen) <= 3  # but the dedup set stays bounded


@pytest.mark.asyncio
async def test_run_inbox_exits_when_stop_already_set(tmp_path):
    _write_event(tmp_path, hash="a", ts=1, name="a")
    got = []

    async def emit(ev):  # pragma: no cover - must never be called
        got.append(ev)

    stop = asyncio.Event()
    stop.set()

    async def sleep_fn(_):  # pragma: no cover - must never be called
        raise AssertionError("sleep_fn called after stop set")

    await run_inbox(tmp_path, emit, stop, sleep_fn=sleep_fn, interval=0)
    assert got == []
