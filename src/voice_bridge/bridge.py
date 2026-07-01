"""Bridge: wire every module together and run the main async loop.

This is the integration capstone. It constructs config/Store/Transcriber/TTS/
ApprovalManager/SessionManager/TelegramIO, builds the outbound and inbound
closures, implements the :class:`Controls` panel surface, and runs until a
SIGINT/SIGTERM stop event fires.

Design for testability (system-prompt C3): ``build()`` constructs and wires
every component and returns a :class:`Wiring`; ``run_until_stopped(wiring, stop)``
runs the startup/run/shutdown lifecycle against an injected ``asyncio.Event``;
``main()`` is the thin top: it calls ``build``, installs signal handlers, and
delegates to ``run_until_stopped``. No real signal handling is exercised in
tests — the run loop takes the stop event as a parameter.

Live TTS engine switch (C4): the outbound closure reads
``tts_holder["backend"]`` AT SEND TIME, never a captured instance. Switching
engines rebuilds the holder's backend so subsequent sends use it immediately.
"""

from __future__ import annotations

import asyncio
import logging
import signal
from dataclasses import dataclass
from typing import Awaitable, Callable

from .approvals import ApprovalManager, parse_yes_no
from .config import (
    Config,
    ProjectConfig,
    effective_autonomy,
    effective_voice,
    load_config,
    load_projects,
)
from .discovery import discover_projects, merge_projects
from .routing import Store
from .sanitizer import prepare_outbound, to_spoken
from .sessions import SessionManager
from .stt import Transcriber
from .telegram_io import TelegramIO
from .tts import get_tts
from .types import Outbound

logger = logging.getLogger(__name__)

# User-facing Lithuanian micro-copy.
_MSG_NOT_UNDERSTOOD = "Nesupratau, pakartok."
_MSG_YES_OR_NO = "Atsakyk taip arba ne."


# --------------------------------------------------------------------------- #
# Routing helper (pure-ish)
# --------------------------------------------------------------------------- #


async def resolve_target(msg: dict, store: Store) -> tuple[str | None, str]:
    """Resolve which project a (non-approval) inbound message goes to.

    Returns ``(project_or_None, reason)`` where ``reason`` is one of:

    * ``"ok"``   — deliverable to ``project``.
    * ``"off"``  — ``project`` is known but disabled (do not deliver).
    * ``"none"`` — no project could be resolved at all.

    A ``reply_to`` that maps to a project wins; otherwise we fall back to the
    last-active project. (An unknown ``reply_to`` also falls back rather than
    failing, so stray replies still route to the active conversation.)
    """
    rid = msg.get("reply_to")
    project = await store.project_for_message(rid) if rid is not None else None
    if project is None:
        project = await store.get_last_active()
    if project is None:
        return None, "none"
    if not await store.is_enabled(project):
        return project, "off"
    return project, "ok"


# --------------------------------------------------------------------------- #
# Outbound closure
# --------------------------------------------------------------------------- #


def make_outbound(
    tts_holder: dict,
    telegram: TelegramIO,
    store: Store,
    cfg: Config,
    sessions: SessionManager,
    controls: "_Controls",
) -> Callable[[Outbound], Awaitable[None]]:
    """Build the outbound closure.

    Two shapes of :class:`Outbound`:

    * ``spoken`` set (notify_user path) — ``full_text`` is the detail
      (``o.text``) and the spoken line is ``to_spoken(o.spoken)``.
    * ``spoken`` empty (assistant turn-end) — split ``o.text`` on the ``---``
      separator via :func:`prepare_outbound`.

    Synthesizes with the project's effective voice; reads the live TTS backend
    from ``tts_holder`` AT SEND TIME (C4). Empty/whitespace spoken text ->
    text-only (no voice). Maps every returned message id to the project and
    marks it last-active (also updating the Controls mirror).
    """

    async def outbound(o: Outbound) -> None:
        if o.spoken:
            full_text, spoken = o.text, to_spoken(o.spoken)
        else:
            full_text, spoken = prepare_outbound(o.text)

        proj = sessions.project(o.project)
        voice = effective_voice(proj, cfg) if proj is not None else cfg.tts_voice

        voice_bytes: bytes | None = None
        if spoken.strip():
            try:
                voice_bytes = await tts_holder["backend"].synthesize(spoken, voice)
            except Exception:  # noqa: BLE001 - never let TTS failure drop the text
                logger.exception("TTS synthesize failed for %s; sending text-only", o.project)
                voice_bytes = None

        ids = await telegram.send_update(o.project, voice, full_text, voice_bytes)
        for mid in ids:
            await store.map_message(mid, o.project)
        await store.set_last_active(o.project)
        controls.mark_last_active(o.project)

    return outbound


# --------------------------------------------------------------------------- #
# Inbound closure
# --------------------------------------------------------------------------- #


def make_inbound(
    transcriber: Transcriber,
    store: Store,
    approvals: ApprovalManager,
    sessions: SessionManager,
    telegram: TelegramIO,
) -> Callable[[dict], Awaitable[None]]:
    """Build the inbound closure.

    Flow (system-prompt C7):

    1. Voice -> transcribe; empty transcript -> ask to repeat and stop.
    2. If ``reply_to`` has a pending approval -> parse yes/no; unparseable ->
       ask again; otherwise resolve. Never delivered as a turn.
    3. Otherwise route via :func:`resolve_target`; ``none`` -> ask which
       project; ``off`` -> tell the user it is disabled; ``ok`` -> deliver.
    """

    async def inbound(msg: dict) -> None:
        if msg.get("is_voice"):
            audio = msg.get("audio")
            if audio is None:
                await telegram.send_question("bridge", _MSG_NOT_UNDERSTOOD)
                return
            text = await transcriber.transcribe(audio)
            if not text.strip():
                await telegram.send_question("bridge", _MSG_NOT_UNDERSTOOD)
                return
        else:
            text = msg.get("text") or ""

        rid = msg.get("reply_to")
        if rid is not None and approvals.has_pending(rid):
            ans = parse_yes_no(text)
            if ans is None:
                await telegram.send_question("bridge", _MSG_YES_OR_NO)
                return
            approvals.resolve(rid, ans)
            return

        project, reason = await resolve_target(msg, store)
        if reason == "none":
            names = ", ".join(sessions.names()) if hasattr(sessions, "names") else ""
            await telegram.send_question("bridge", f"Į kurį projektą? {names}".strip())
            return
        if reason == "off":
            await telegram.send_question(
                "bridge", f"{project} išjungtas (/on {project})"
            )
            return
        await sessions.deliver(project, text)

    return inbound


# --------------------------------------------------------------------------- #
# Controls (panel surface) — C2
# --------------------------------------------------------------------------- #


class _Controls:
    """In-memory Controls implementation backing /panel and slash commands.

    Keeps a mirror ``dict[str, dict]`` seeded from ``store.enabled_map()`` +
    each project's effective mode/voice + ``cfg.tts_backend``. ``snapshot()`` is
    SYNC and reads the mirror so the panel never awaits.
    """

    def __init__(
        self,
        sessions: SessionManager,
        store: Store,
        cfg: Config,
        tts_holder: dict,
    ) -> None:
        self._sessions = sessions
        self._store = store
        self._cfg = cfg
        self._tts_holder = tts_holder
        self._telegram: TelegramIO | None = None
        self._mirror: dict[str, dict] = {}

    def attach_telegram(self, telegram: TelegramIO) -> None:
        """Wire the telegram instance used for set_mode user notices."""
        self._telegram = telegram

    def mark_last_active(self, project: str) -> None:
        """Flip last_active on for *project* and off for every other in the mirror."""
        for name, row in self._mirror.items():
            row["last_active"] = name == project

    async def seed(self) -> None:
        """Populate the mirror from the store + each project's effective state."""
        enabled = await self._store.enabled_map()
        last_active = await self._store.get_last_active()
        names = self._sessions.names() if hasattr(self._sessions, "names") else list(enabled)
        for name in names:
            proj = self._sessions.project(name)
            mode = effective_autonomy(proj, self._cfg) if proj is not None else self._cfg.autonomy_mode
            voice = effective_voice(proj, self._cfg) if proj is not None else self._cfg.tts_voice
            self._mirror[name] = {
                "display_name": (
                    getattr(proj, "display_name", None) or name
                ) if proj is not None else name,
                "enabled": enabled.get(name, True),
                "mode": mode,
                "voice": voice,
                "engine": self._cfg.tts_backend,
                "last_active": last_active == name,
                "cwd": getattr(proj, "cwd", "") if proj is not None else "",
            }

    def snapshot(self) -> list[dict]:
        """SYNC: list of dicts keyed exactly project/enabled/mode/voice/engine/last_active."""
        return [
            {
                "project": name,
                "display_name": row["display_name"],
                "enabled": row["enabled"],
                "mode": row["mode"],
                "voice": row["voice"],
                "engine": row["engine"],
                "last_active": row["last_active"],
                "cwd": row["cwd"],
            }
            for name, row in self._mirror.items()
        ]

    async def toggle(self, project: str | None, on: bool) -> None:
        targets = [project] if project is not None else list(self._mirror)
        for name in targets:
            if name in self._mirror:
                self._mirror[name]["enabled"] = on
            await self._sessions.set_enabled(name, on)

    async def select(self, project: str) -> None:
        if project not in self._mirror:
            return
        await self._store.set_last_active(project)
        self.mark_last_active(project)

    async def set_mode(self, project: str | None, mode: str) -> None:
        targets = [project] if project is not None else list(self._mirror)
        for name in targets:
            if name in self._mirror:
                self._mirror[name]["mode"] = mode
            await self._sessions.set_mode(name, mode)
        # Forwarded from Task 8: a live set_mode restarts the session and drops
        # any in-flight turn silently. Tell the user so they can re-issue it.
        if self._telegram is not None:
            label = project or "visi"
            await self._telegram.send_question(
                "bridge",
                f"Režimas pakeistas į {mode} ({label}). "
                "Jei buvo vykdoma užduotis, pakartok.",
            )

    async def set_voice(self, project: str | None, voice: str) -> None:
        targets = [project] if project is not None else list(self._mirror)
        for name in targets:
            if name in self._mirror:
                self._mirror[name]["voice"] = voice
            proj = self._sessions.project(name)
            if proj is not None:
                proj.voice = voice  # so effective_voice picks it up

    async def set_engine(self, name: str) -> None:
        # C4: rebuild the live TTS backend so subsequent sends use it.
        self._cfg.tts_backend = name
        self._tts_holder["backend"] = get_tts(self._cfg)
        for row in self._mirror.values():
            row["engine"] = name


# --------------------------------------------------------------------------- #
# Wiring + lifecycle
# --------------------------------------------------------------------------- #


@dataclass
class Wiring:
    """The fully-wired set of components produced by :func:`build`."""

    cfg: Config
    store: Store
    sessions: SessionManager
    telegram: TelegramIO
    controls: _Controls
    outbound: Callable[[Outbound], Awaitable[None]]
    inbound: Callable[[dict], Awaitable[None]]


async def build() -> Wiring:
    """Construct and wire every component. No polling, no run loop.

    Split from :func:`main` so the wiring is testable without signal handling.
    """
    cfg = load_config()
    projects = load_projects()
    if cfg.auto_discover_projects:
        explicit_cwds = {p.cwd for p in projects}
        projects = merge_projects(
            projects,
            discover_projects(cfg.auto_discover_limit, explicit_cwds=explicit_cwds),
        )

    store = Store(cfg.db_path)
    await store.init()
    await store.seed(projects)

    tts_holder = {"backend": get_tts(cfg)}
    transcriber = Transcriber(cfg.whisper_model, language="lt")

    # Telegram is constructed last (it needs the controls + inbound closure),
    # but ApprovalManager.send_question and the controls notices need it. Use a
    # one-slot holder resolved at call time to break the cycle.
    telegram_ref: dict = {}

    async def send_question(project: str, text: str) -> int:
        return await telegram_ref["io"].send_question(project, text)

    approvals = ApprovalManager(send_question, cfg.approval_timeout)

    class _LazyTelegram:
        async def send_update(self, project, voice_label, text, voice_bytes):
            return await telegram_ref["io"].send_update(
                project, voice_label, text, voice_bytes
            )

        async def send_question(self, project, text):
            return await telegram_ref["io"].send_question(project, text)

    lazy_telegram = _LazyTelegram()

    sessions_ref: dict = {}

    class _LazySessions:
        def project(self, name):
            sm = sessions_ref.get("sm")
            return sm.project(name) if sm is not None else None

        def names(self):
            sm = sessions_ref.get("sm")
            return sm.names() if sm is not None and hasattr(sm, "names") else []

        async def deliver(self, project, text):
            await sessions_ref["sm"].deliver(project, text)

        async def set_enabled(self, project, enabled):
            await sessions_ref["sm"].set_enabled(project, enabled)

        async def set_mode(self, project, mode):
            await sessions_ref["sm"].set_mode(project, mode)

    lazy_sessions = _LazySessions()

    controls = _Controls(lazy_sessions, store, cfg, tts_holder)

    outbound = make_outbound(
        tts_holder, lazy_telegram, store, cfg, lazy_sessions, controls
    )

    sessions = SessionManager(projects, cfg, store, outbound, approvals)
    sessions_ref["sm"] = sessions

    inbound = make_inbound(transcriber, store, approvals, lazy_sessions, lazy_telegram)

    telegram = TelegramIO(cfg, inbound, controls)
    telegram_ref["io"] = telegram
    controls.attach_telegram(telegram)

    await controls.seed()

    return Wiring(
        cfg=cfg,
        store=store,
        sessions=sessions,
        telegram=telegram,
        controls=controls,
        outbound=outbound,
        inbound=inbound,
    )


async def run_until_stopped(wiring: Wiring, stop: asyncio.Event) -> None:
    """Start sessions, start telegram polling (returns), wait for *stop*, shut down.

    ``telegram.run()`` does NOT block (C3); this function owns the run-forever
    wait via ``stop``. Shutdown is symmetric and runs in ``finally``.
    """
    await wiring.sessions.start_all()
    await wiring.telegram.run()
    try:
        await stop.wait()
    finally:
        await wiring.telegram.stop()
        await wiring.sessions.stop_all()


async def main() -> None:
    """Top-level entry: build, install signal handlers, run until stopped."""
    wiring = await build()

    stop = asyncio.Event()
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, stop.set)
        except (NotImplementedError, ValueError):  # pragma: no cover
            # Some platforms / non-main threads cannot install handlers.
            pass

    await run_until_stopped(wiring, stop)


if __name__ == "__main__":  # pragma: no cover
    asyncio.run(main())
