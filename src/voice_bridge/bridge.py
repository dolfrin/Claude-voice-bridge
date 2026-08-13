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
import html
import logging
import signal
from dataclasses import dataclass
from typing import Awaitable, Callable

from . import claude_history
from .attachments import format_attachment_prompt, save_attachments
from .approvals import ApprovalManager, parse_yes_no
from .config import (
    Config,
    ProjectConfig,
    claude_projects_path,
    claude_sessions_path,
    effective_autonomy,
    effective_voice,
    load_config,
    load_projects,
)
from .discovery import discover_projects, merge_projects
from .routing import Store, project_of
from .sanitizer import prepare_outbound, to_spoken
from .sessions import SessionManager
from .stt import Transcriber
from .telegram_io import TelegramIO, tail_for_telegram
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
    """Resolve which conversation a (non-approval) inbound message goes to.

    Returns ``(conversation_key_or_None, reason)`` where ``reason`` is one of:

    * ``"ok"``   — deliverable to that conversation.
    * ``"off"``  — its project is known but disabled (do not deliver).
    * ``"none"`` — nothing could be resolved at all.

    A session sub-topic names its conversation outright and wins over
    everything else. Next a ``reply_to`` that maps to one; otherwise we fall
    back to the last-active conversation. (An unknown ``reply_to`` also falls
    back rather than failing, so stray replies still route somewhere sensible.)
    A bare project name from any of those resolves to its newest open
    conversation, so older state and ``/status`` still land in a real topic.
    """
    target = msg.get("project")
    if target is None:
        rid = msg.get("reply_to")
        target = await store.project_for_message(rid) if rid is not None else None
    if target is None:
        target = await store.get_last_active()
    if target is None:
        return None, "none"
    if "#" not in target:
        conversations = await store.conversations(target)
        if not conversations:
            return None, "none"
        target = conversations[-1].key
    if not await store.is_enabled(project_of(target)):
        return target, "off"
    return target, "ok"


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

        if o.file_path:
            ids = await telegram.send_file(
                o.project, voice, full_text, voice_bytes, o.file_path
            )
        else:
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
        # Anything the bridge says back belongs in the forum topic the message
        # arrived in; outside group mode there is no topic and it stays
        # "bridge", which is also the label the user sees.
        here = msg.get("project") or "bridge"

        if msg.get("is_voice"):
            audio = msg.get("audio")
            if audio is None:
                await telegram.send_question(here, _MSG_NOT_UNDERSTOOD)
                return
            text = await _pick_transcript(
                await transcriber.transcribe_all(audio), telegram, here
            )
            if text is None:
                # Edit tapped: the user resends corrected text themselves.
                return
            if not text.strip():
                await telegram.send_question(here, _MSG_NOT_UNDERSTOOD)
                return
        else:
            text = msg.get("text") or ""

        rid = msg.get("reply_to")
        if rid is not None and approvals.has_pending(rid):
            ans = parse_yes_no(text)
            if ans is None:
                await telegram.send_question(here, _MSG_YES_OR_NO)
                return
            approvals.resolve(rid, ans)
            return

        live_pid = msg.get("live_pid")
        if live_pid is not None:
            # This topic is attached to a Claude Code process running outside
            # the bridge (VS Code, CLI). Its turns go down that session's socket
            # instead of a bridge-owned session; telegram reports any failure
            # into the same topic, so there is nothing to say here.
            await telegram.deliver_live(live_pid, text)
            return

        project, reason = await resolve_target(msg, store)
        if reason == "none":
            names = ", ".join(sessions.names()) if hasattr(sessions, "names") else ""
            await telegram.send_question(here, f"Which project? {names}".strip())
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


async def _pick_transcript(
    results: list[dict], telegram, project: str = "stt"
) -> str | None:
    """Return the transcript to send to Claude, asking the user if there is a choice.

    The ASR service returns one entry per model it has loaded. Each transcript
    is sent as its own message with an accept button under it, so a long one
    gets a full message to itself. With a single entry there is nothing to
    choose and no message is sent; that is also what happens once one model
    wins and the server stops running both. On timeout the primary (first)
    transcript is used.

    Only the display is capped at Telegram's message limit — Claude always
    receives the untruncated transcript.

    Returns ``None`` when the user tapped Edit — the voice turn must end
    silently; the user resends corrected text as a normal message.
    """
    usable = [r for r in results if r.get("text", "").strip()]
    if len(usable) < 2:
        return usable[0]["text"] if usable else ""

    picked = await telegram.ask_per_message(
        project, [(r["model"], r["text"]) for r in usable]
    )
    if picked is None:
        return None
    for r in usable:
        if r["model"] == picked:
            return r["text"]
    return usable[0]["text"]


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


def _render_turns(label: str, turns: list[tuple[str, str]], max_chars: int) -> str:
    """Turns as a Telegram message: clipped per turn AND overall. A full read
    needs the downloadable file, not this."""
    if not turns:
        return f"{label}: transcript is empty."
    out = [f"\U0001F4DC <b>{html.escape(label)}</b>\n"]
    for role, text in turns:
        who = "\U0001F464" if role == "user" else "\U0001F916"
        out.append(f"{who} {html.escape(text[:600])}")
    return tail_for_telegram("\n\n".join(out), max_chars)


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
        self._mirror: dict[str, dict] = {}
        # Conversation rows mirrored the same way, so the Telegram side can read
        # them without awaiting: key -> {key, project, ordinal, session_id}.
        self._conversations: dict[str, dict] = {}

    def mark_last_active(self, target: str) -> None:
        """Flip last_active on for a target's project and off for every other."""
        project = project_of(target)
        for name, row in self._mirror.items():
            row["last_active"] = name == project

    async def seed(self) -> None:
        """Populate the mirror from the store + each project's effective state."""
        enabled = await self._store.enabled_map()
        last_active = project_of(await self._store.get_last_active() or "")
        await self.reload_conversations()
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
        # Accepts a conversation key too: enabling is a project-level switch.
        targets = [project_of(project)] if project is not None else list(self._mirror)
        for name in targets:
            if name in self._mirror:
                self._mirror[name]["enabled"] = on
            await self._sessions.set_enabled(name, on)

    async def select(self, project: str) -> None:
        if project not in self._mirror:
            return
        await self._store.set_last_active(project)
        self.mark_last_active(project)

    async def enable_and_deliver(self, target: str, text: str) -> None:
        await self.toggle(project_of(target), True)
        await self.reload_conversations()
        if "#" not in target:
            keys = [k for k, row in self._conversations.items() if row["project"] == target]
            if not keys:
                return
            target = keys[-1]
        await self._sessions.deliver(target, text)
        await self._store.set_last_active(target)
        self.mark_last_active(target)

    async def interrupt(self, target: str | None) -> str:
        """Stop work: one conversation, every conversation of a project, or the
        last active one when nothing is named."""
        keys = self._interrupt_targets(target)
        if not keys:
            return "No active session found."
        for key in keys:
            await self._sessions.interrupt(key)
        await self._store.set_last_active(keys[0])
        self.mark_last_active(keys[0])
        return ", ".join(keys) + ": interrupted."

    def _interrupt_targets(self, target: str | None) -> list[str]:
        if target and "#" in target:
            return [target] if target in self._conversations else []
        if target:
            return [k for k, row in self._conversations.items() if row["project"] == target]
        running = [k for k in self._conversations if self._sessions.is_running(k)]
        active = next(
            (name for name, row in self._mirror.items() if row.get("last_active")), None
        )
        preferred = [k for k in running if project_of(k) == active]
        return preferred or running[:1]

    # -- conversations ------------------------------------------------------

    async def reload_conversations(self) -> None:
        self._conversations = {
            conv.key: {
                "key": conv.key,
                "project": conv.project,
                "ordinal": conv.ordinal,
                "thread_id": conv.thread_id,
                "session_id": conv.session_id,
            }
            for conv in await self._store.conversations()
        }

    async def open_conversation(
        self, project: str, resume: str | None = None, fork: bool = False
    ) -> str | None:
        key = await self._sessions.open(project, resume=resume, fork=fork)
        await self.reload_conversations()
        if key is not None:
            await self._store.set_last_active(key)
            self.mark_last_active(key)
        return key

    async def stage_conversation(self, project: str, resume: str) -> str | None:
        """Reserve a conversation and its topic for a session, not started yet."""
        key = await self._sessions.stage(project, resume=resume)
        await self.reload_conversations()
        return key

    async def attach_conversation(self, key: str) -> bool:
        """Start a staged conversation. SessionManager decides about forking —
        it re-checks liveness at start time, which may differ from when the card
        was posted."""
        started = await self._sessions.attach(key)
        if started:
            await self._store.set_last_active(key)
            self.mark_last_active(key)
        return started

    async def close_conversation(self, key: str) -> bool:
        closed = await self._sessions.close(key)
        await self.reload_conversations()
        return closed

    def conversation_rows(self) -> list[dict]:
        """SYNC snapshot of open conversations for the /sessions view.

        Carries Claude's own name for each session as well as the topic name:
        ``Paprika ASR #2`` says where it lives, ``Fix topic routing bug`` says
        what it is, and only the pair is enough to pick one out of five.
        """
        # One directory lookup per project, not per conversation: resolving it
        # scans every history directory.
        dirs: dict[str, object] = {}
        rows = []
        for key, row in self._conversations.items():
            # A row that is neither running nor has ever held a session is an
            # empty reservation — a disabled project's placeholder. Listing one
            # per project would bury the sessions that actually exist.
            if not self._sessions.is_running(key) and not row.get("session_id"):
                continue
            project = self._mirror.get(row["project"], {})
            label = project.get("display_name") or row["project"]
            rows.append({
                **row,
                "title": f"{label} #{row['ordinal']}",
                "session_title": self._session_title(row, project, dirs),
                "running": self._sessions.is_running(key),
                "busy": self._sessions.is_busy(key),
            })
        return rows

    def _session_title(self, row: dict, project: dict, dirs: dict) -> str:
        """What Claude calls this conversation's session, or "" if it has none."""
        session_id = row.get("session_id")
        cwd = project.get("cwd") or ""
        if not session_id or not cwd:
            return ""
        name = row["project"]
        if name not in dirs:
            dirs[name] = claude_history.project_dir_for_cwd(
                claude_projects_path(self._cfg), cwd
            )
        directory = dirs[name]
        if directory is None:
            return ""
        path = directory / f"{session_id}.jsonl"
        return claude_history.title(path) if path.is_file() else ""

    def _project_dir(self, project: str):
        cwd = self._mirror.get(project, {}).get("cwd") or ""
        if not cwd:
            return None
        return claude_history.project_dir_for_cwd(
            claude_projects_path(self._cfg), cwd
        )

    def resume_options(self, project: str, limit: int | None = None) -> list:
        """Claude sessions on disk for a project, newest first."""
        return self.project_sessions(project, limit)[0]

    def project_sessions(
        self, project: str, limit: int | None = None
    ) -> tuple[list, int]:
        """``(sessions, total_on_disk)`` — the tail plus what the cap hid.

        A project can have 75 sessions; a Telegram keyboard cannot. Returning
        the total lets the list say so instead of quietly looking complete.
        """
        directory = self._project_dir(project)
        if directory is None:
            return [], 0
        live = claude_history.live_sessions(claude_sessions_path(self._cfg))
        capped = self._cfg.resume_limit if limit is None else limit
        return (
            claude_history.recent(directory, capped, live),
            claude_history.count_sessions(directory),
        )

    def session_history_text(
        self, project: str, uuid: str, limit: int = 12, max_chars: int = 3500
    ) -> str:
        """The tail of any session's transcript, whether the bridge opened it or not."""
        directory = self._project_dir(project)
        path = directory / f"{uuid}.jsonl" if directory is not None else None
        if path is None or not path.is_file():
            return f"{uuid[:8]}: no transcript on disk."
        return _render_turns(
            uuid[:8], claude_history.turns(path, limit), max_chars
        )

    def session_transcript_file(self, project: str, uuid: str) -> tuple[str, bytes] | None:
        """The WHOLE conversation as a Markdown file, nothing clipped.

        Anything rendered into a Telegram message is a fragment by construction;
        this is the version you can actually read end to end.
        """
        directory = self._project_dir(project)
        path = directory / f"{uuid}.jsonl" if directory is not None else None
        if path is None or not path.is_file():
            return None
        parts = [f"# {claude_history.title(path)}\n", f"session: {uuid}\n"]
        for role, text in claude_history.turns(path, None):
            parts.append(f"\n## {'You' if role == 'user' else 'Claude'}\n\n{text}\n")
        return f"{uuid[:8]}.md", "".join(parts).encode("utf-8")

    def history_text(self, key: str, limit: int = 12, max_chars: int = 3500) -> str:
        """The tail of a conversation's real Claude transcript.

        Reads Claude's own ``.jsonl``, so it also shows turns that happened in
        VS Code or the CLI — the bridge's Markdown mirror only sees Telegram.
        """
        row = self._conversations.get(key)
        if row is None:
            return f"{key}: unknown session."
        session_id = row.get("session_id")
        cwd = self._mirror.get(row["project"], {}).get("cwd") or ""
        path = (
            claude_history.session_file(claude_projects_path(self._cfg), cwd, session_id)
            if session_id and cwd
            else None
        )
        if path is None:
            return f"{key}: no transcript yet."
        return _render_turns(key, claude_history.turns(path, limit), max_chars)

    async def set_mode(self, project: str | None, mode: str) -> None:
        targets = [project] if project is not None else list(self._mirror)
        for name in targets:
            if name in self._mirror:
                self._mirror[name]["mode"] = mode
            await self._sessions.set_mode(name, mode)
        # No notice needed any more: the mode is read per tool call, so a live
        # switch cannot drop an in-flight turn.

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

    async def send_question(project: str, text: str) -> int:
        return await telegram_ref["io"].send_question(project, text)

    approvals = ApprovalManager(send_question, cfg.approval_timeout)

    class _LazyTelegram:
        """Forward every call to the TelegramIO that build() creates later.

        Explicit one-line wrappers per method used to be listed here, and a
        method added to TelegramIO without a wrapper failed at runtime with
        AttributeError on the FIRST user who hit that path -- never in tests,
        which inject their own double. Forwarding by name cannot go stale.
        """

        def __getattr__(self, name):
            async def call(*args, **kwargs):
                return await getattr(telegram_ref["io"], name)(*args, **kwargs)

            return call

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

        async def interrupt(self, project):
            return await sessions_ref["sm"].interrupt(project)

        def add_projects(self, projects):
            return sessions_ref["sm"].add_projects(projects)

        async def open(self, project, resume=None, fork=False):
            return await sessions_ref["sm"].open(project, resume=resume, fork=fork)

        async def stage(self, project, resume=None):
            return await sessions_ref["sm"].stage(project, resume=resume)

        async def attach(self, key, fork=False):
            return await sessions_ref["sm"].attach(key, fork=fork)

        async def close(self, key):
            return await sessions_ref["sm"].close(key)

        def is_running(self, key):
            sm = sessions_ref.get("sm")
            return sm.is_running(key) if sm is not None else False

        def is_busy(self, key):
            sm = sessions_ref.get("sm")
            return sm.is_busy(key) if sm is not None else False

    lazy_sessions = _LazySessions()

    controls = _Controls(lazy_sessions, store, cfg, tts_holder)

    outbound = make_outbound(
        tts_holder, lazy_telegram, store, cfg, lazy_sessions, controls
    )

    sessions = SessionManager(
        projects,
        cfg,
        store,
        outbound,
        approvals,
        lazy_telegram.ask_user,
        on_open=lazy_telegram.open_conversation_topic,
        on_close=lazy_telegram.close_conversation_topic,
        on_progress=lazy_telegram.send_progress,
    )
    sessions_ref["sm"] = sessions

    inbound = make_inbound(transcriber, store, approvals, lazy_sessions, lazy_telegram)

    telegram = TelegramIO(cfg, inbound, controls, store=store, projects=projects)
    telegram_ref["io"] = telegram

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

    Telegram starts FIRST: restoring conversations needs their sub-topics, and
    opening a project's first one has to create a topic, which needs a live
    Application. Polling early only means the first messages queue.
    """
    await wiring.telegram.run()
    await wiring.sessions.start_all()
    await wiring.controls.reload_conversations()
    # Topics attached with /live point at editor sessions that outlive this
    # process; re-attach the ones still running instead of making the user
    # pick them again after every restart.
    await wiring.telegram.restore_live()
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
