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

from .attachments import format_attachment_prompt, save_attachments
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

# User-facing micro-copy.
_MSG_NOT_UNDERSTOOD = "I did not understand. Please repeat."
_MSG_YES_OR_NO = "Answer yes or no."


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
# TTS helpers (shared by the outbound closure and the approval question path)
# --------------------------------------------------------------------------- #


def _resolve_voice(cfg: Config, proj, alert: bool) -> str:
    """Pick the TTS voice: the ALERT voice for alert-class sends (when
    configured), otherwise the project's effective voice / global default."""
    base = effective_voice(proj, cfg) if proj is not None else cfg.tts_voice
    if alert and getattr(cfg, "tts_alert_voice", ""):
        return cfg.tts_alert_voice
    return base


async def _synthesize(tts_holder: dict, spoken: str, voice: str, project: str) -> bytes | None:
    """Synthesize ``spoken`` via the live TTS backend; never raise.

    Empty/whitespace spoken -> no audio. A backend failure logs and returns
    None so the caller still sends the text (mirrors make_outbound's guard and
    keeps the never-raises invariant intact)."""
    if not spoken.strip():
        return None
    try:
        return await tts_holder["backend"].synthesize(spoken, voice)
    except Exception:  # noqa: BLE001 - never let TTS failure drop the text
        logger.exception("TTS synthesize failed for %s; sending text-only", project)
        return None


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
        voice = _resolve_voice(cfg, proj, o.alert)

        voice_bytes = await _synthesize(tts_holder, spoken, voice, o.project)

        try:
            if o.file_path:
                ids = await telegram.send_file(
                    o.project, voice, full_text, voice_bytes, o.file_path
                )
            else:
                ids = await telegram.send_update(
                    o.project, voice, full_text, voice_bytes
                )
        except Exception:  # noqa: BLE001 - C8 corollary: a bad send must
            # never kill the session turn loop (sessions._run_loop's except
            # would otherwise treat this as a turn crash and permanently
            # stop the project's session). Mirrors the TTS guard above:
            # log, best-effort notify, and return without re-raising.
            logger.exception(
                "telegram send failed for %s; session turn loop continues",
                o.project,
            )
            try:
                await telegram.send_question(
                    o.project, "(pranešimo išsiųsti nepavyko — žr. logus)"
                )
            except Exception:  # noqa: BLE001 - best effort only, swallow
                logger.exception(
                    "fallback send_question also failed for %s", o.project
                )
            return

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
            await telegram.send_question("bridge", f"Which project? {names}".strip())
            return
        text = await _append_attachment_transcripts(text, msg, transcriber)
        text = await _attach_files_to_prompt(project, text, msg, sessions)
        if reason == "off":
            await telegram.send_disabled_project_prompt(project, text)
            return
        urgent, text = _consume_urgent_prefix(text)
        if urgent and hasattr(sessions, "interrupt"):
            await sessions.interrupt(project)
        await sessions.deliver(project, text)

    return inbound


async def _attach_files_to_prompt(
    project: str,
    text: str,
    msg: dict,
    sessions: SessionManager,
) -> str:
    attachments = msg.get("attachments") or []
    if not attachments:
        return text
    proj = sessions.project(project) if hasattr(sessions, "project") else None
    if proj is None:
        return text
    saved = await save_attachments(proj.cwd, attachments)
    return format_attachment_prompt(text, saved)


async def _append_attachment_transcripts(
    text: str,
    msg: dict,
    transcriber: Transcriber,
) -> str:
    lines = [text.strip()] if text.strip() else []
    transcripts: list[str] = []
    for item in msg.get("attachments") or []:
        if item.get("kind") != "audio":
            continue
        data = item.get("data")
        if not data:
            continue
        transcript = await transcriber.transcribe(bytes(data))
        if transcript.strip():
            name = item.get("file_name") or "audio"
            transcripts.append(f"- {name}: {transcript.strip()}")
    if transcripts:
        lines.append("Audio transkripcija:")
        lines.extend(transcripts)
    return "\n".join(lines).strip()


def _consume_urgent_prefix(text: str) -> tuple[bool, str]:
    stripped = text.lstrip()
    if not stripped.startswith("!"):
        return False, text
    return True, stripped[1:].lstrip()


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

    async def enable_and_deliver(self, project: str, text: str) -> None:
        await self.toggle(project, True)
        await self._sessions.deliver(project, text)
        await self._store.set_last_active(project)
        self.mark_last_active(project)

    async def interrupt(self, project: str | None) -> str:
        target = project
        if target is None:
            for name, row in self._mirror.items():
                if row.get("last_active"):
                    target = name
                    break
        if target is None or target not in self._mirror:
            return "No active project found."
        stopped = await self._sessions.interrupt(target)
        await self._store.set_last_active(target)
        self.mark_last_active(target)
        return f"{target}: interrupted." if stopped else f"{target}: restarted."

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
                f"Mode changed to {mode} ({label}). "
                "If a task was running, send it again.",
            )

    async def set_voice(self, project: str | None, voice: str) -> None:
        targets = [project] if project is not None else list(self._mirror)
        for name in targets:
            if name in self._mirror:
                self._mirror[name]["voice"] = voice
            proj = self._sessions.project(name)
            if proj is not None:
                proj.voice = voice  # so effective_voice picks it up

    async def refresh_projects(self) -> int:
        explicit = load_projects()
        discovered: list[ProjectConfig] = []
        if self._cfg.auto_discover_projects:
            explicit_cwds = {p.cwd for p in explicit}
            discovered = discover_projects(
                self._cfg.auto_discover_limit,
                explicit_cwds=explicit_cwds,
            )
        candidates = merge_projects(explicit, discovered)
        existing = set(self._mirror)
        new_projects = [project for project in candidates if project.name not in existing]
        if not new_projects:
            return 0

        if hasattr(self._sessions, "add_projects"):
            self._sessions.add_projects(new_projects)
        await self._store.seed(new_projects)
        enabled = await self._store.enabled_map()
        last_active = await self._store.get_last_active()
        for project in new_projects:
            self._mirror[project.name] = {
                "display_name": getattr(project, "display_name", None) or project.name,
                "enabled": enabled.get(project.name, project.enabled),
                "mode": effective_autonomy(project, self._cfg),
                "voice": effective_voice(project, self._cfg),
                "engine": self._cfg.tts_backend,
                "last_active": last_active == project.name,
                "cwd": project.cwd,
            }
        return len(new_projects)

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
    transcriber = Transcriber(cfg.whisper_model)

    # Telegram is constructed last (it needs the controls + inbound closure),
    # but ApprovalManager.send_question and the controls notices need it. Use a
    # one-slot holder resolved at call time to break the cycle.
    telegram_ref: dict = {}
    sessions_ref: dict = {}

    async def send_question(
        project: str, text: str, spoken: str = "", token: int | None = None
    ) -> int:
        """ApprovalManager's send hook: attach the token-encoded inline buttons
        and speak the (code-free) approval line with the ALERT voice."""
        voice_bytes: bytes | None = None
        voice_label: str | None = None
        if spoken and spoken.strip():
            sm = sessions_ref.get("sm")
            proj = sm.project(project) if sm is not None else None
            voice_label = _resolve_voice(cfg, proj, alert=True)
            voice_bytes = await _synthesize(
                tts_holder, to_spoken(spoken), voice_label, project
            )
        return await telegram_ref["io"].send_question(
            project,
            text,
            voice_label=voice_label,
            voice_bytes=voice_bytes,
            approval_token=token,
        )

    approvals = ApprovalManager(send_question, cfg.approval_timeout)

    class _LazyTelegram:
        async def send_update(self, project, voice_label, text, voice_bytes):
            return await telegram_ref["io"].send_update(
                project, voice_label, text, voice_bytes
            )

        async def send_file(self, project, voice_label, text, voice_bytes, file_path):
            return await telegram_ref["io"].send_file(
                project, voice_label, text, voice_bytes, file_path
            )

        async def send_question(self, project, text):
            return await telegram_ref["io"].send_question(project, text)

        async def send_disabled_project_prompt(self, project, text):
            return await telegram_ref["io"].send_disabled_project_prompt(project, text)

        async def ask_user(self, project, question, choices):
            return await telegram_ref["io"].ask_user(project, question, choices)

    lazy_telegram = _LazyTelegram()

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

        async def interrupt(self, project):
            return await sessions_ref["sm"].interrupt(project)

        def add_projects(self, projects):
            return sessions_ref["sm"].add_projects(projects)

    lazy_sessions = _LazySessions()

    controls = _Controls(lazy_sessions, store, cfg, tts_holder)

    outbound = make_outbound(
        tts_holder, lazy_telegram, store, cfg, lazy_sessions, controls
    )

    sessions = SessionManager(
        projects, cfg, store, outbound, approvals, lazy_telegram.ask_user
    )
    sessions_ref["sm"] = sessions

    inbound = make_inbound(transcriber, store, approvals, lazy_sessions, lazy_telegram)

    telegram = TelegramIO(
        cfg, inbound, controls, on_approval=approvals.resolve_token
    )
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
