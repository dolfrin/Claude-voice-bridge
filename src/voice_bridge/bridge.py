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
import re
import shutil
import signal
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Awaitable, Callable

from .attachments import format_attachment_prompt, save_attachments
from .approvals import ApprovalManager, parse_yes_no
from .config import (
    Config,
    EFFORT_LEVELS,
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

# Recap (B3b): bounded per-project buffer of outbound status lines kept in
# _Controls, rendered synchronously by /recap with no LLM call.
_RECAP_MAX_LINES = 20

# Separator between a name-prefix token ("qwing" / "Qwing") and the rest of
# the message: EITHER a ":", "-" or "," followed by zero or more whitespace
# chars (so "qwing:build", with no trailing space, matches), OR one or more
# whitespace chars with no separator character at all. Either way the char
# immediately after the name can never be a bare word char, so "qwingbuild"
# (no separator) and "qwinger: x" (extra letters before the colon) still do
# NOT match. Matched against the remainder of the text AFTER the literal
# name is stripped off (see parse_name_prefix), so it never needs to know
# what a "name" character is.
_NAME_PREFIX_SEP_RE = re.compile(r"^\s*(?:[:\-,]\s*|\s+)(.*)$", re.DOTALL)


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


def parse_name_prefix(text: str | None, names: list[str]) -> tuple[str | None, str]:
    """Parse a leading ``"<project>: ..."`` / ``"<project> ..."`` token.

    Case-insensitive EXACT match against a project in *names*: the first
    token of *text* must equal a known project name in full — not merely be
    prefixed by one — and must be followed by a ``:``/``-`` separator (with
    or without trailing whitespace, e.g. ``"qwing:build"`` needs no space)
    OR at least one whitespace char with no separator before the remainder.
    Longer names are tried first so a name that is itself a prefix of
    another known name (``"qwing"`` vs. ``"qwingtest"``) cannot shadow the
    longer, more specific match.

    Returns ``(name, remainder)`` on a match (``name`` is the CANONICAL
    entry from *names*, not the casing typed by the user), otherwise
    ``(None, text)`` UNCHANGED — including when *text* merely contains a
    colon later on (``"just a colon: here"`` does not match unless "just"
    is itself a known project name). ``text=None`` is treated as ``""`` (so
    the call cannot raise) and returns ``(None, "")``.
    """
    text = text or ""
    stripped = text.lstrip()
    lower = stripped.lower()
    for name in sorted(names, key=len, reverse=True):
        if not lower.startswith(name.lower()):
            continue
        rest = stripped[len(name):]
        m = _NAME_PREFIX_SEP_RE.match(rest)
        if m:
            return name, m.group(1)
    return None, text


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


def _recap_summary_line(spoken: str, full_text: str) -> str:
    """One-line recap summary for an Outbound: the spoken line, trimmed, or
    (when spoken is empty) the first non-blank line of the full text."""
    spoken = (spoken or "").strip()
    if spoken:
        return spoken
    for line in (full_text or "").splitlines():
        line = line.strip()
        if line:
            return line
    return ""


def _format_elapsed(seconds: float) -> str:
    """Render an elapsed duration as a short human string (s / min / val)."""
    total = max(0, int(seconds))
    if total < 60:
        return f"{total} s"
    minutes, _ = divmod(total, 60)
    if minutes < 60:
        return f"{minutes} min"
    hours, rem_minutes = divmod(minutes, 60)
    if rem_minutes:
        return f"{hours} val {rem_minutes} min"
    return f"{hours} val"


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
    text-only (no voice). Maps every returned message id to the project
    UNCONDITIONALLY (a reply to any sent message, even a transient one, must
    still resolve to its project). Marking last-active (both the store and
    the Controls mirror) and appending a one-line recap summary to the
    project's recap buffer (B3b's ``/recap``) are both skipped when
    ``o.transient`` is True: NOISE like the per-turn "Working." status, the
    heartbeat, and verbose tool-activity flushes must not inflate the
    recap's update count, nor hijack routing away from whatever project the
    user is actively conversing with while a background project's transient
    send fires — guarded so a recap-tracking failure can never break the
    never-raises send path.
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
        if not o.transient:
            await store.set_last_active(o.project)
            controls.mark_last_active(o.project)
        try:
            if not o.transient:
                controls.record_recap(o.project, _recap_summary_line(spoken, full_text))
        except Exception:  # noqa: BLE001 - recap tracking must never break a send
            logger.exception("recap record failed for %s", o.project)

    return outbound


# --------------------------------------------------------------------------- #
# Always-allow persistence hook
# --------------------------------------------------------------------------- #


def make_on_always_allow(
    approvals: ApprovalManager, store: Store
) -> Callable[[int], Awaitable[bool]]:
    """Build the "✅♾ Visada leisti" persist hook wired into TelegramIO.

    Given a just-resolved approval *token*, look up the (project, signature)
    the ApprovalManager stashed for it and persist an always-allow policy so
    future matching calls auto-approve. Reads ``policy_for_token`` SYNCHRONOUSLY
    (before any await) since the resolved request cleans the mapping up on the
    next loop turn.

    Returns True only when a policy was actually persisted, so the caller can
    show an honest label. Returns False when the call is NOT policy-eligible
    (signature is None) or the store write fails — both degrade to allow-once
    (the approval is ALREADY resolved as allow); a store failure is logged,
    never raised.
    """

    async def on_always_allow(token: int) -> bool:
        info = approvals.policy_for_token(token)
        if info is None:
            return False
        project, signature = info
        if signature is None:
            # Not eligible for a persisted policy (compound/exfil/egress/
            # out-of-cwd/interpreter). Allow-once only.
            return False
        try:
            await store.add_policy(project, signature)
            return True
        except Exception:  # noqa: BLE001 - persist is best-effort (allow-once)
            logger.exception(
                "add_policy failed for %s/%s; grant is allow-once only",
                project,
                signature,
            )
            return False

    return on_always_allow


# --------------------------------------------------------------------------- #
# Inbound closure
# --------------------------------------------------------------------------- #


def make_inbound(
    transcriber: Transcriber,
    store: Store,
    approvals: ApprovalManager,
    sessions: SessionManager,
    telegram: TelegramIO,
    controls: "_Controls",
) -> Callable[[dict], Awaitable[None]]:
    """Build the inbound closure.

    Flow (system-prompt C7, extended by B3b for recap + name-prefix routing):

    0. EVERY call marks the recap boundary (``controls.mark_recap_boundary``)
       — any inbound user turn means subsequent outbounds are "new since
       last seen", regardless of how this turn itself resolves.
    1. Voice -> transcribe; empty transcript -> ask to repeat and stop.
    2. If ``reply_to`` has a pending approval -> parse yes/no; unparseable ->
       ask again; otherwise resolve. Never delivered as a turn.
    3. The urgent ``"!"`` prefix (:func:`_consume_urgent_prefix`) is consumed
       NEXT, BEFORE any routing — a leading "!" would otherwise defeat the
       name-prefix match below (``"!qwing: fix"`` doesn't start with a known
       name) and silently fall back to last-active, interrupting the wrong
       project. Stripping it first means "!qwing: fix" is treated exactly
       like "qwing: fix" for routing purposes, just urgent.
    4. Routing precedence:
       * an explicit ``reply_to`` that resolves to a KNOWN project wins
         outright (a quote-reply is unambiguous) — routed via
         :func:`resolve_target`, unchanged from before;
       * otherwise a leading ``"<project>: ..."`` / ``"<project> ..."``
         name-prefix (case-insensitive exact match against
         ``sessions.names()``, see :func:`parse_name_prefix`) wins, with the
         prefix stripped from the delivered text;
       * otherwise fall back to :func:`resolve_target` (last_active/off/none).
       ``none`` -> ask which project; ``off`` -> tell the user it is
       disabled; ``ok`` -> deliver (interrupting first if urgent).
    """

    async def inbound(msg: dict) -> None:
        controls.mark_recap_boundary()

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

        # Urgent '!' is consumed BEFORE name-prefix routing: otherwise
        # "!qwing: fix it" fails parse_name_prefix (text starts with '!', not
        # a known name) and falls back to last-active, interrupting the
        # WRONG project and delivering the literal "qwing: fix it". Stripping
        # '!' first turns it into "qwing: fix it" so name-prefix routing sees
        # the name normally, and "!fix it" (no name) still falls back to
        # last-active exactly as before.
        urgent, text = _consume_urgent_prefix(text)

        reply_project = await store.project_for_message(rid) if rid is not None else None
        if reply_project is not None:
            # A quote-reply that resolves to a known project is unambiguous
            # and wins outright, exactly like before name-prefix routing
            # existed.
            project, reason = await resolve_target(msg, store)
        else:
            names = sessions.names() if hasattr(sessions, "names") else []
            prefix_project, prefix_text = parse_name_prefix(text, names)
            if prefix_project is not None:
                project, text = prefix_project, prefix_text
                reason = "ok" if await store.is_enabled(project) else "off"
            else:
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

# /newproject folder-name sanitizer (SECURITY). Only a bare, safe folder
# name may ever reach the filesystem: no '/', no spaces, no other
# punctuation, no path traversal ('..'), and no dotfile/flag-like name.
_PROJECT_NAME_RE = re.compile(r"^[A-Za-z0-9._-]+$")
_PROJECT_NAME_MAX_LEN = 64


def _sanitize_project_name(name: str) -> str:
    """Return *name* if it is a safe bare project folder name, else ``""``.

    Allowed characters are exactly ``[A-Za-z0-9._-]``; the whole string must
    match (so a stray ``/`` or space anywhere rejects it outright — never
    silently rewritten). Also rejects empty, ``.``, ``..``, a name starting
    with ``.`` or ``-``, and anything longer than
    :data:`_PROJECT_NAME_MAX_LEN` chars. Never raises: a non-string input
    degrades to rejection.
    """
    if not isinstance(name, str):
        return ""
    if not name or len(name) > _PROJECT_NAME_MAX_LEN:
        return ""
    if name in {".", ".."}:
        return ""
    if name[0] in {".", "-"}:
        return ""
    if not _PROJECT_NAME_RE.match(name):
        return ""
    return name


class _Controls:
    """In-memory Controls implementation backing /panel and slash commands.

    Keeps a mirror ``dict[str, dict]`` seeded from ``store.enabled_map()`` +
    each project's effective mode/voice + ``cfg.tts_backend``. ``snapshot()`` is
    SYNC and reads the mirror so the panel never awaits.

    Also owns the B3b recap state: a per-project bounded buffer of recent
    outbound one-liners plus a single "since" boundary shared by every
    project, both in-memory only. ``make_outbound`` appends to the buffer on
    every successful send; ``make_inbound`` resets the boundary on every
    inbound user turn; ``/recap`` renders it synchronously with no LLM call.
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
        self._recap_lines: dict[str, list[str]] = {}
        self._recap_since: float = time.monotonic()

    def attach_telegram(self, telegram: TelegramIO) -> None:
        """Wire the telegram instance used for set_mode user notices."""
        self._telegram = telegram

    # -- Recap (B3b) -------------------------------------------------------

    def record_recap(self, project: str, line: str) -> None:
        """Append a one-line summary to *project*'s recap buffer.

        Blank lines are skipped. Bounded to the last ``_RECAP_MAX_LINES``
        entries per project (oldest dropped first)."""
        line = (line or "").strip()
        if not line:
            return
        buf = self._recap_lines.setdefault(project, [])
        buf.append(line)
        if len(buf) > _RECAP_MAX_LINES:
            del buf[: len(buf) - _RECAP_MAX_LINES]

    def mark_recap_boundary(self) -> None:
        """Reset every project's recap buffer and stamp a fresh "since" time.

        Called on every inbound user turn: subsequent outbounds become "new
        since last seen" for the NEXT ``/recap``."""
        self._recap_lines = {}
        self._recap_since = time.monotonic()

    def recap(self) -> str:
        """SYNC, no LLM call: render what happened since the recap boundary.

        One line per project with activity: "{display} — {N} atnaujinimai
        per {elapsed}: {latest line}". Projects with no activity since the
        boundary are omitted entirely. ``"Nieko naujo."`` when nothing at
        all happened."""
        elapsed = _format_elapsed(time.monotonic() - self._recap_since)
        parts: list[str] = []
        for name, lines in self._recap_lines.items():
            if not lines:
                continue
            display = self._mirror.get(name, {}).get("display_name", name)
            parts.append(
                f"{display} — {len(lines)} atnaujinimai per {elapsed}: {lines[-1]}"
            )
        return "\n".join(parts) if parts else "Nieko naujo."

    # -- Cost / token usage (B3c) -------------------------------------------

    # -- Always-allow policies (visibility / revocation) --------------------

    async def list_policies(self) -> list[tuple[str, str]]:
        """Every always-allow (project, signature) grant, for /policies."""
        return await self._store.list_policies()

    async def clear_policies(self, project: str | None = None) -> None:
        """Revoke always-allow grants: all, or just one project's."""
        await self._store.clear_policy(project)

    async def cost_summary(self) -> str:
        """Per-project + TOTAL token/cost summary, read fresh from the store.

        One line per project with recorded usage:
        ``"{display}: {turns} turai, {in}+{out} tok, ${cost:.4f}"`` plus a
        TOTAL line. Under Claude Code subscription auth the SDK never reports
        ``total_cost_usd`` so every accumulated cost stays 0 — in that case
        (no project shows a nonzero cost) the TOTAL line notes the cost is
        unavailable instead of a misleading ``$0.0000``.
        """
        all_usage = await self._store.all_usage()
        if not all_usage:
            return "No usage recorded yet."

        lines: list[str] = []
        total_turns = total_in = total_out = 0
        total_cost = 0.0
        for name, row in all_usage.items():
            display = self._mirror.get(name, {}).get("display_name", name)
            turns = row.get("turns", 0)
            tin = row.get("input_tokens", 0)
            tout = row.get("output_tokens", 0)
            cost = row.get("cost_usd", 0.0) or 0.0
            total_turns += turns
            total_in += tin
            total_out += tout
            total_cost += cost
            lines.append(f"{display}: {turns} turai, {tin}+{tout} tok, ${cost:.4f}")

        if total_cost > 0:
            lines.append(
                f"TOTAL: {total_turns} turai, {total_in}+{total_out} tok, "
                f"${total_cost:.4f}"
            )
        else:
            lines.append(
                f"TOTAL: {total_turns} turai, {total_in}+{total_out} tok "
                "(cost n/a — subscription auth?)"
            )
        return "\n".join(lines)

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
                "verbose": bool(getattr(proj, "verbose", False)) if proj is not None else False,
                "model": getattr(proj, "model", None) if proj is not None else None,
                "effort": getattr(proj, "effort", None) if proj is not None else None,
            }

    def snapshot(self) -> list[dict]:
        """SYNC: per-project dicts for the panel/status views, keyed
        project/display_name/enabled/mode/voice/engine/last_active/cwd/verbose/
        model/effort."""
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
                "verbose": row.get("verbose", False),
                "model": row.get("model"),
                "effort": row.get("effort"),
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

    async def _persist_override(self, project: str, field: str, value) -> bool:
        """Best-effort persist of a runtime override (Task A). Returns whether
        the store write SUCCEEDED.

        A store write failure must NEVER crash the command or block the
        in-memory change: it is logged and swallowed. Persisting the change
        (esp. a demoted autonomy) is what makes it survive a restart instead of
        silently reverting to — or re-escalating from — the yaml default. The
        caller uses the return value to surface a failed AUTONOMY persist (a
        security setting) to the user; voice/verbose/effort stay silent."""
        try:
            await self._store.set_override(project, field, value)
            return True
        except Exception:  # noqa: BLE001 - persist is best-effort
            logger.exception(
                "persist override %s=%r for %s failed", field, value, project
            )
            return False

    async def _persist_created(
        self, name: str, cwd: str, display_name: str | None = None
    ) -> None:
        """Best-effort persist of a runtime-created project (Task A).

        Guarded like :meth:`_persist_override` so a store failure never turns a
        successful /newproject into a reported error."""
        try:
            await self._store.add_created_project(name, cwd, display_name)
        except Exception:  # noqa: BLE001 - persist is best-effort
            logger.exception("persist created project %s failed", name)

    async def set_mode(self, project: str | None, mode: str) -> None:
        targets = [project] if project is not None else list(self._mirror)
        persist_failed = False
        for name in targets:
            if name in self._mirror:
                self._mirror[name]["mode"] = mode
            await self._sessions.set_mode(name, mode)
            if not await self._persist_override(name, "autonomy", mode):
                persist_failed = True
        # Forwarded from Task 8: a live set_mode restarts the session and drops
        # any in-flight turn silently. Tell the user so they can re-issue it.
        if self._telegram is not None:
            label = project or "visi"
            await self._telegram.send_question(
                "bridge",
                f"Mode changed to {mode} ({label}). "
                "If a task was running, send it again.",
            )
            # SECURITY: autonomy is the one override whose failed persist is
            # dangerous — a demotion that isn't saved silently re-escalates to
            # the yaml default on the next restart. Surface it so the user knows
            # the setting is not durable (voice/verbose/effort stay silent).
            if persist_failed:
                await self._telegram.send_question(
                    "bridge",
                    "⚠️ Režimo nepavyko išsaugoti — po perkrovimo grįš prie "
                    "projects.yaml numatytojo. Patikrink diską / DB.",
                )

    async def set_voice(self, project: str | None, voice: str) -> None:
        targets = [project] if project is not None else list(self._mirror)
        for name in targets:
            if name in self._mirror:
                self._mirror[name]["voice"] = voice
            proj = self._sessions.project(name)
            if proj is not None:
                proj.voice = voice  # so effective_voice picks it up
            await self._persist_override(name, "voice", voice)

    async def set_verbose(self, project: str | None, on: bool) -> None:
        """Toggle live tool-activity streaming and mirror it for the snapshot.

        Flips the flag on the SessionManager's ProjectConfig — where the live
        turn loop reads it — updates the mirror so a later /panel can surface
        it, and PERSISTS the override (Task A) so a restart restores it instead
        of reverting to the yaml default."""
        targets = [project] if project is not None else list(self._mirror)
        for name in targets:
            if name in self._mirror:
                self._mirror[name]["verbose"] = on
            await self._sessions.set_verbose(name, on)
            await self._persist_override(name, "verbose", on)

    async def set_effort(self, project: str | None, level: str) -> None:
        """Set per-project reasoning effort and mirror it for the snapshot.

        An invalid level is ignored/rejected (neither the mirror nor the live
        session changes). Otherwise the mirror is updated and the change is
        forwarded to the SessionManager, which RESTARTS the running session so
        the new effort applies (mirrors :meth:`set_mode`). ``project=None``
        targets every project."""
        if level not in EFFORT_LEVELS:
            return
        targets = [project] if project is not None else list(self._mirror)
        for name in targets:
            if name in self._mirror:
                self._mirror[name]["effort"] = level
            await self._sessions.set_effort(name, level)
            await self._persist_override(name, "effort", level)

    def info(self) -> str:
        """SYNC: one line per project with model (config + REAL), effort, mode,
        voice, verbose, plus the global TTS engine.

        The real model is the last ACTUAL model that answered a turn, read from
        the SessionManager's in-memory last-model map (None -> em dash)."""
        lines: list[str] = []
        for name, row in self._mirror.items():
            display = row.get("display_name", name)
            model = row.get("model") or "default"
            real = None
            if hasattr(self._sessions, "last_model"):
                real = self._sessions.last_model(name)
            effort = row.get("effort") or "default"
            verbose = "on" if row.get("verbose") else "off"
            lines.append(
                f"{display}: model={model} (real: {real or '—'}) · "
                f"effort={effort} · mode={row['mode']} · voice={row['voice']} · "
                f"verbose={verbose}"
            )
        lines.append(f"engine: {self._cfg.tts_backend}")
        return "\n".join(lines)

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

        await self._register_projects(new_projects)
        return len(new_projects)

    async def _register_projects(self, projects: list[ProjectConfig]) -> None:
        """Register *projects* the same way :meth:`refresh_projects` does:
        runtime SessionManager registration, store seeding, and a mirror
        entry per project (all reflecting current store/effective state).
        Shared by :meth:`refresh_projects` and :meth:`create_project` so the
        two paths can never drift apart.
        """
        if not projects:
            return
        if hasattr(self._sessions, "add_projects"):
            self._sessions.add_projects(projects)
        await self._store.seed(projects)
        enabled = await self._store.enabled_map()
        last_active = await self._store.get_last_active()
        for project in projects:
            self._mirror[project.name] = {
                "display_name": getattr(project, "display_name", None) or project.name,
                "enabled": enabled.get(project.name, project.enabled),
                "mode": effective_autonomy(project, self._cfg),
                "voice": effective_voice(project, self._cfg),
                "engine": self._cfg.tts_backend,
                "last_active": last_active == project.name,
                "cwd": project.cwd,
                "verbose": bool(getattr(project, "verbose", False)),
                "model": getattr(project, "model", None),
                "effort": getattr(project, "effort", None),
            }

    async def _git_init(self, path: Path) -> None:
        """Best-effort ``git init`` in a freshly created project directory.

        Guarded like :meth:`sessions.SessionManager._open_vscode`: a missing
        ``git`` binary or a failing exec is logged and swallowed — it must
        never fail project creation."""
        git = shutil.which("git")
        if git is None:
            logger.warning(
                "create_project: 'git' not on PATH, skipping git init for %s", path
            )
            return
        try:
            proc = await asyncio.create_subprocess_exec(
                git,
                "init",
                cwd=str(path),
                stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.DEVNULL,
            )
            await proc.wait()
            if proc.returncode != 0:
                logger.warning("git init exited with %s for %s", proc.returncode, path)
        except OSError:
            logger.exception("git init failed for %s", path)

    async def create_project(self, name: str) -> str:
        """/newproject: create a brand-new project folder and switch to it.

        SECURITY: *name* is sanitized to a bare ``[A-Za-z0-9._-]`` folder
        name (see :func:`_sanitize_project_name`) before it ever touches the
        filesystem — no path traversal, no absolute paths, nothing shell-
        adjacent. An invalid name creates nothing and returns an error
        string.

        If ``~/Projects/<name>`` already exists: an already-registered
        project is just enabled + selected; one that exists on disk but is
        unregistered is registered (like :meth:`refresh_projects` would)
        then enabled + selected. Otherwise the directory is created,
        ``git init`` is attempted (non-fatal), the project is registered,
        then enabled + selected so the user's NEXT message routes straight
        to it. Never raises: any unexpected failure is caught and returned
        as an error string.
        """
        try:
            safe = _sanitize_project_name(name)
            if not safe:
                return (
                    f"Netinkamas projekto pavadinimas: {name!r}. Leidžiama tik "
                    "raidės, skaičiai, taškas, brūkšnys ir apatinis brūkšnys "
                    "(be tarpų ir kelio simbolių)."
                )

            target = Path.home() / "Projects" / safe

            # Already-registered check FIRST and unconditionally: a registered
            # project's cwd basename may differ from its name (e.g. qwing ->
            # .../WhisperX), so gating this behind target.exists() would fall
            # through to "fresh create" — corrupting the mirror (cwd/mode/voice
            # reset in the panel) and littering ~/Projects with a stray repo.
            if safe in self._mirror:
                await self.toggle(safe, True)
                await self.select(safe)
                return f"Projektas {safe} jau užregistruotas — perjungiau į jį."

            if target.exists():
                project = ProjectConfig(name=safe, cwd=str(target), enabled=True)
                await self._register_projects([project])
                # Persist so this runtime-registered project is reloaded across
                # restarts instead of vanishing (Task A).
                await self._persist_created(safe, str(target))
                await self.toggle(safe, True)
                await self.select(safe)
                return f"Projektas {safe} rastas diske ({target}) — užregistravau ir perjungiau į jį."

            target.mkdir(parents=True)
            await self._git_init(target)

            project = ProjectConfig(name=safe, cwd=str(target), enabled=True)
            await self._register_projects([project])
            # Persist so this freshly created project survives a restart (Task A).
            await self._persist_created(safe, str(target))
            await self.toggle(safe, True)
            await self.select(safe)
            return f"Sukurtas projektas {safe} ({target}). Siųsk užduotį — dirbsiu jame."
        except Exception as err:  # noqa: BLE001 - /newproject must never crash the bot
            logger.exception("create_project failed for %r", name)
            return f"Nepavyko sukurti projekto {name}: {err}"

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

    # Merge dynamically-created projects persisted from a previous run (Task A):
    # /newproject projects are not in projects.yaml, so without this they vanish
    # on restart. One whose directory no longer exists on disk is skipped (log).
    existing_names = {p.name for p in projects}
    for row in await store.created_projects():
        name = row.get("name")
        cwd = row.get("cwd")
        if not name or name in existing_names:
            continue
        if not cwd or not Path(cwd).is_dir():
            logger.warning(
                "build: created project %r cwd missing (%s), skipping", name, cwd
            )
            continue
        projects.append(
            ProjectConfig(
                name=name,
                cwd=cwd,
                display_name=row.get("display_name"),
                enabled=True,
            )
        )
        existing_names.add(name)

    await store.seed(projects)

    # Apply persisted runtime overrides onto the ProjectConfig objects BEFORE
    # sessions start (Task A). Precedence: persisted override > yaml — so a
    # project the user demoted (e.g. autonomy full -> safe) is restored demoted,
    # never silently RE-ESCALATED back to the yaml default on restart.
    by_name = {p.name: p for p in projects}
    for name, override in (await store.overrides()).items():
        proj = by_name.get(name)
        if proj is None:
            continue
        if "autonomy" in override:
            proj.autonomy = override["autonomy"]
        if "voice" in override:
            proj.voice = override["voice"]
        if "verbose" in override:
            proj.verbose = override["verbose"]
        if "effort" in override:
            proj.effort = override["effort"]

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

        def last_model(self, name):
            sm = sessions_ref.get("sm")
            return sm.last_model(name) if sm is not None and hasattr(sm, "last_model") else None

        async def deliver(self, project, text):
            await sessions_ref["sm"].deliver(project, text)

        async def set_enabled(self, project, enabled):
            await sessions_ref["sm"].set_enabled(project, enabled)

        async def set_mode(self, project, mode):
            await sessions_ref["sm"].set_mode(project, mode)

        async def set_effort(self, project, level):
            await sessions_ref["sm"].set_effort(project, level)

        async def set_verbose(self, project, on):
            await sessions_ref["sm"].set_verbose(project, on)

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

    inbound = make_inbound(
        transcriber, store, approvals, lazy_sessions, lazy_telegram, controls
    )

    telegram = TelegramIO(
        cfg,
        inbound,
        controls,
        on_approval=approvals.resolve_token,
        on_always_allow=make_on_always_allow(approvals, store),
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
