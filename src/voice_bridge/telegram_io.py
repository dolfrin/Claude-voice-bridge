"""python-telegram-bot Application front end: whitelists the owner, receives
inbound voice+text, sends outbound text+voice, handles slash commands, and
renders the /panel inline control board.

The LOGIC here is structured to be unit-testable with a mocked Bot:

* ``build_panel_markup`` / ``parse_callback`` are pure helpers.
* every handler is a method that reads only ``update`` / ``context`` and the
  injected ``Controls`` object, so a test can call it with MagicMock updates.
* ``send_update`` / ``send_question`` take the bot from ``self.app.bot`` so a
  test can inject an ``AsyncMock`` bot.

C2: ``controls.snapshot()`` is SYNCHRONOUS and each dict is keyed exactly
``{"project", "display_name", "enabled", "mode", "voice", "engine",
"last_active", "cwd", "verbose"}`` (see ``bridge._Controls.snapshot()``).
C3: ``run()`` starts polling and RETURNS; ``bridge.main()`` owns the
run-forever wait. ``stop()`` shuts the Application down.
"""

from __future__ import annotations

import asyncio
import datetime
import html
import json
import logging
import os
import re
import signal
import time
import random
from pathlib import Path
from typing import Awaitable, Callable, Protocol, TypeVar

from telegram import (
    InlineKeyboardButton,
    InlineKeyboardMarkup,
    Update,
)
from telegram.error import (
    BadRequest,
    NetworkError,
    RetryAfter,
    TelegramError,
    TimedOut,
)
from telegram.ext import (
    Application,
    CallbackQueryHandler,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    filters,
)

from . import live, sent_log, usage
from .approvals import _TOKEN_RE, _fold
from .config import AGENT_BACKENDS, Config, set_env_value
from .i18n import t
from .scheduler import parse_hhmm
from .transcript import transcript_path
from .tts import available_voices

# Pure view/format/parse helpers were extracted into telegram_views for
# maintainability. Re-export them here so existing references
# (telegram_io.build_panel_markup, .parse_callback, ...) and the test import
# surface stay identical after the split. telegram_views must NOT import this
# module (would create a cycle).
from .telegram_views import (
    bot_commands,
    _EFFORTS,
    _ENGINES,
    _MODES,
    _clean_choices,
    _find_project_row,
    _format_help,
    _format_policies,
    _format_schedules,
    _friendly_path,
    _project_list_rows,
    _tail_for_telegram,
    build_menu_markup,
    build_mode_markup,
    build_panel_markup,
    build_projects_list_markup,
    build_voice_markup,
    format_projects,
    parse_callback,
)

logger = logging.getLogger(__name__)

_T = TypeVar("_T")


# /pc actions, as systemctl verbs; logind lets the logged-in user run them.
_PC_ACTIONS = ("suspend", "poweroff", "reboot")


def _answer_markup(text: str) -> InlineKeyboardMarkup | None:
    """Answer buttons for a message that asks something, or None.

    Numbered options get a button each (the number is sent back, as typing
    it would); a yes/no question gets ✅ Taip / ❌ Ne."""
    options = live.parse_options(text)
    if options:
        return InlineKeyboardMarkup([
            [InlineKeyboardButton(f"{i}. {label[:40]}", callback_data=f"ans:{i}")]
            for i, label in enumerate(options, 1)
        ])
    if live.is_yes_no_question(text):
        return InlineKeyboardMarkup([[
            InlineKeyboardButton(t("answer.yes_button"), callback_data=f"ans:{t('answer.yes')}"),
            InlineKeyboardButton(t("answer.no_button"), callback_data=f"ans:{t('answer.no')}"),
        ]])
    return None


class Controls(Protocol):
    """State surface implemented by bridge.py (Task 10).

    ``snapshot`` is synchronous; the mutators are coroutines. ``project=None``
    means "all projects".
    """

    def snapshot(self) -> list[dict]:
        # each dict keyed EXACTLY:
        # {"project": str, "display_name": str, "enabled": bool,
        #  "mode": str, "voice": str, "engine": str, "last_active": bool,
        #  "cwd": str, "verbose": bool, "model": str | None,
        #  "effort": str | None}
        ...

    async def toggle(self, project: str | None, on: bool) -> None: ...
    async def select(self, project: str) -> None: ...
    async def enable_and_deliver(self, project: str, text: str) -> None: ...
    async def refresh_projects(self) -> int: ...
    async def create_project(self, name: str) -> str: ...
    async def set_mode(self, project: str | None, mode: str) -> None: ...
    async def set_effort(self, project: str | None, level: str) -> None: ...
    async def set_verbose(self, project: str | None, on: bool) -> None: ...
    async def set_voice(self, project: str | None, voice: str) -> None: ...
    async def set_engine(self, name: str) -> None: ...
    async def interrupt(self, project: str | None) -> str: ...
    def recap(self) -> str: ...
    def info(self) -> str: ...
    async def list_policies(self) -> list[tuple[str, str]]: ...
    async def clear_policies(self, project: str | None) -> None: ...
    async def list_schedules(self, project: str | None = None) -> list[dict]: ...
    async def add_schedule(
        self, project: str, hhmm: str, prompt: str, last_run: str | None = None
    ) -> int: ...
    async def remove_schedule(self, schedule_id: int) -> bool: ...
    async def set_schedule_enabled(self, schedule_id: int, enabled: bool) -> bool: ...


_PHOTO_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}
_AUDIO_SUFFIXES = {".mp3", ".m4a", ".ogg", ".opus", ".wav", ".flac"}
_VIDEO_SUFFIXES = {".mp4", ".mov", ".mkv", ".webm"}

# Telegram hard limits: a text message tops out at 4096 chars, a media
# caption at 1024. Reliability fix (audit-confirmed HIGH impact): outbound
# sends used to be unguarded and unbounded, so a long assistant reply or a
# transient API error would raise out of make_outbound and permanently kill
# the session's turn loop. _chunk_text keeps every send under the hard cap;
# _send_with_retry absorbs transient errors; the call site in bridge.py's
# make_outbound never lets a send failure propagate.
_MESSAGE_LIMIT = 4096
_CAPTION_LIMIT = 1024
_RETRY_BACKOFF = (0.5, 1.0, 2.0)

# Bug fix (audit-confirmed): send_question sends exactly ONE message so its
# inline Allow/Deny buttons stay attached to the text being approved -- it
# must NOT chunk like _send_text_chunks does. Without a cap, an approval
# preview built from a huge Bash command can exceed Telegram's 4096-char hard
# limit, the send raises, and the approval is silently lost (times out to
# DENY, and the user never even sees the prompt). Truncating instead of
# chunking keeps the buttons on the one message the user taps.
_APPROVAL_PREVIEW_LIMIT = 1500
_APPROVAL_TRUNCATED_MARKER = "…[truncated]"

# Bug fix: Telegram's bot API refuses to hand back a file over ~20 MB via
# getFile (raises BadRequest), which used to have no handler -- the
# attachment just vanished with no feedback to the user.
# Permission-request ids we generate: hex + dashes only, so a callback
# payload can never walk out of the spool directory.
_SAFE_ID_RE = re.compile(r"[A-Za-z0-9._-]{1,80}")




def _parse_schedule_id(raw: str | None) -> int | None:
    """Parse a /schedule id argument to a positive int, or None on junk.

    Kept tiny and total (never raises) so the command handler can reply with a
    usage line instead of crashing on ``/schedule remove abc``."""
    if raw is None:
        return None
    try:
        sid = int(raw)
    except (TypeError, ValueError):
        return None
    return sid if sid > 0 else None


def _truncate_approval_preview(text: str, limit: int = _APPROVAL_PREVIEW_LIMIT) -> str:
    """Cap an approval preview to `limit` chars plus a truncation marker.

    Keeps the total ``send_question`` message (project prefix + this text)
    comfortably under Telegram's 4096-char hard cap, even for a giant Bash
    command preview.
    """
    if len(text) <= limit:
        return text
    return text[:limit] + _APPROVAL_TRUNCATED_MARKER


def _chunk_text(text: str, limit: int = _MESSAGE_LIMIT) -> list[str]:
    """Split ``text`` into pieces no longer than ``limit`` characters.

    Prefers to break on the last newline within the current window so a
    Telegram message never cuts a line in half; falls back to a hard cut at
    ``limit`` when a single line is itself longer than the limit. The
    newline (when used as the break point) stays with the earlier chunk, so
    ``"".join(_chunk_text(text)) == text`` always holds.
    """
    if len(text) <= limit:
        return [text]
    chunks: list[str] = []
    remaining = text
    while len(remaining) > limit:
        window = remaining[:limit]
        newline_at = window.rfind("\n")
        split_at = newline_at + 1 if newline_at != -1 else limit
        chunks.append(remaining[:split_at])
        remaining = remaining[split_at:]
    if remaining:
        chunks.append(remaining)
    return chunks


async def _send_with_retry(
    coro_factory: Callable[[], Awaitable[_T]], *, attempts: int = 4
) -> _T:
    """Call ``coro_factory()`` and retry on transient Telegram errors.

    ``coro_factory`` is a zero-arg callable that returns a fresh awaitable
    each time, so a failed send can be re-invoked (a plain coroutine object
    can only be awaited once).

    * ``RetryAfter`` (flood control / 429) -> sleep ``retry_after`` seconds
      plus a small jitter, then retry.
    * ``TimedOut`` / other ``NetworkError`` -> exponential backoff
      (0.5s, 1s, 2s, ...) then retry.
    * ``BadRequest`` (e.g. malformed entities) is NOT transient -> re-raise
      immediately without retrying.
    * Once ``attempts`` tries are exhausted, the last error is raised.

    This is implemented manually (no dependency on the optional
    ``AIORateLimiter`` extra) so tests need no extra deps and no network.
    """
    last_exc: Exception | None = None
    for attempt in range(attempts):
        try:
            return await coro_factory()
        except RetryAfter as exc:
            last_exc = exc
            if attempt == attempts - 1:
                break
            # PTB returns retry_after as int seconds today but as a
            # datetime.timedelta under PTB_TIMEDELTA (a future default);
            # normalize to float seconds so the sleep never raises TypeError.
            delay = exc.retry_after
            if isinstance(delay, datetime.timedelta):
                delay = delay.total_seconds()
            await asyncio.sleep(delay + random.uniform(0.05, 0.25))
        except BadRequest:
            raise
        except (TimedOut, NetworkError) as exc:
            last_exc = exc
            if attempt == attempts - 1:
                break
            await asyncio.sleep(_RETRY_BACKOFF[min(attempt, len(_RETRY_BACKOFF) - 1)])
    assert last_exc is not None  # pragma: no branch - loop always sets it before break
    raise last_exc


def _next(seq: list[str], current: str) -> str:
    """Return the element after ``current`` in ``seq``, wrapping around."""
    try:
        i = seq.index(current)
    except ValueError:
        return seq[0]
    return seq[(i + 1) % len(seq)]


# I2: 1-based ordinal words (EN + LT) a spoken answer might use instead of a
# bare number ("the first one" / "pirmas"). Keys are ALREADY diacritic-folded
# (see approvals._fold) because the answer's tokens are folded before lookup,
# so "trečias" -> "trecias" -> 3 and every diacritic spelling collapses to one
# entry. Kept small on purpose: a multiple-choice ask rarely has >5 options.
_ORDINAL_WORDS: dict[str, int] = {
    # English (word + "1st"/"2nd"/… numeric-ordinal forms)
    "first": 1, "1st": 1,
    "second": 2, "2nd": 2,
    "third": 3, "3rd": 3,
    "fourth": 4, "4th": 4,
    "fifth": 5, "5th": 5,
    # Lithuanian (folded)
    "pirmas": 1, "pirma": 1, "pirmasis": 1, "pirmoji": 1,
    "antras": 2, "antra": 2, "antrasis": 2, "antroji": 2,
    "trecias": 3, "trecia": 3, "treciasis": 3, "trecioji": 3,
    "ketvirtas": 4, "ketvirta": 4, "ketvirtasis": 4,
    "penktas": 5, "penkta": 5, "penktasis": 5,
}


def _match_choice(answer_text: str, choices: list[str]) -> str | None:
    """Map a free-text/voice answer to one of *choices*, or None if none fits.

    This is what makes an ask answerable hands-free: the user hears the options
    and just talks. Matching is tried most-specific first so an explicit pick
    always wins over a fuzzy one:

    1. a bare 1-based number ("2") -> that choice (ignored if out of range);
    2. an ordinal word, EN or LT, diacritics folded ("second" / "trečias") ->
       that choice — only when the answer names EXACTLY ONE in-range ordinal
       (ambiguous "first or second" falls through);
    3. a case-insensitive, diacritic-folded EXACT match to a choice label;
    4. the answer is a diacritic-folded UNIQUE substring of exactly one label
       (the user typed PART of a label, e.g. "deploy" -> "Deploy to prod").

    Only that one direction is used. Matching a label that appears INSIDE a
    longer answer is deliberately NOT done: it would snap a whole sentence —
    including a NEGATED one ("no, do not deploy") or an ambiguous one ("first or
    second") — onto a choice label. Anything that is not a clean number /
    ordinal / label / label-substring instead returns None, so
    :meth:`TelegramIO.resolve_ask` passes the RAW answer through as free-form
    and the agent interprets the user's actual words (preserving negation).

    Returns the matched choice's ORIGINAL label (not the folded form), or None.
    Pure; never raises."""
    stripped = answer_text.strip()
    if not stripped:
        return None
    lowered = stripped.lower()

    # 1. bare number (1-based).
    if lowered.isdigit():
        idx = int(lowered) - 1
        return choices[idx] if 0 <= idx < len(choices) else None

    # 2. ordinal word(s) — only an unambiguous single ordinal counts.
    folded_tokens = [_fold(tok) for tok in _TOKEN_RE.findall(lowered)]
    ordinals = {_ORDINAL_WORDS[tok] for tok in folded_tokens if tok in _ORDINAL_WORDS}
    if len(ordinals) == 1:
        idx = next(iter(ordinals)) - 1
        if 0 <= idx < len(choices):
            return choices[idx]

    # 3. exact label (case + diacritic folded).
    folded_answer = _fold(lowered)
    for choice in choices:
        if _fold(choice.lower()) == folded_answer:
            return choice

    # 4. the answer is a UNIQUE substring of exactly one label. ONLY this
    #    direction (answer inside label): matching a label inside the answer
    #    would snap a negated/ambiguous sentence onto a choice. Require length
    #    >= 2 so a stray single char can't match. Everything else -> free-form.
    if len(folded_answer) >= 2:
        substring_hits = [
            choice
            for choice in choices
            if (folded_choice := _fold(choice.lower()))
            and folded_answer in folded_choice
        ]
        if len(substring_hits) == 1:
            return substring_hits[0]
    return None


class TelegramIO:
    def __init__(
        self,
        cfg: Config,
        on_user_message: Callable[[dict], Awaitable[None]],
        controls: Controls,
        on_approval: Callable[[int, bool], bool] | None = None,
        on_always_allow: Callable[[int], Awaitable[bool]] | None = None,
        on_sent: Callable[[int, str], Awaitable] | None = None,
        on_speak: Callable[[str], Awaitable[bytes | None]] | None = None,
    ) -> None:
        self.cfg = cfg
        self.on_user_message = on_user_message
        self.controls = controls
        # Claude editor sessions and Codex app-server are separate runtimes.
        # One Telegram channel must never switch between them implicitly.
        self._claude_live_enabled = (
            getattr(cfg, "agent_backend", "claude") == "claude"
        )
        # Text -> voice bytes for the /live stream, which has no project and so
        # cannot go through the project-scoped outbound path. Wired in bridge to
        # the same TTS backend everything else uses; None means text-only.
        self._on_speak = on_speak
        # Whether the last message driving the live session came in as a voice
        # note. Mirrors the user's own channel back at them: talk to it and it
        # talks back, type at it and it stays quiet. Starts False so attaching
        # from the keyboard does not start reading the stream aloud.
        self._live_spoken = False
        # Resolver for inline Allow/Deny taps: returns True if a live pending
        # approval was resolved, False if it was already answered / timed out.
        # Wired in bridge to ApprovalManager.resolve_token.
        self._on_approval = on_approval
        # "Always allow" (apv:{token}:2) persist hook: given the resolved
        # token, records an always-allow policy so future MATCHING calls
        # auto-approve. Wired in bridge; a failure here must never break the
        # (already-resolved-as-allow) approval — it degrades to allow-once.
        self._on_always_allow = on_always_allow
        # Records (message_id, project) so a quote-reply to something WE sent
        # outside the normal outbound path (the /live stream, a permission
        # prompt) routes back to the right project instead of silently falling
        # back to whatever was last active.
        self._on_sent = on_sent
        # /agent rewrites AGENT_BACKEND here, then asks the process to stop so
        # systemd (Restart=always) starts it again on the new backend. The
        # service's WorkingDirectory is the repo root, where .env lives.
        self._env_path = ".env"
        self._restart = lambda: os.kill(os.getpid(), signal.SIGTERM)
        self._agent_switched: str | None = None
        # Where the last message was delivered (see note_route).
        self._last_route: str | None = None
        # Message ids this process sent: the hook-button watcher skips them.
        self._own_sent: set = set()
        self._hook_buttons_task: asyncio.Task | None = None
        # Which Claude account was logged in when, and its limits over time:
        # what /usage needs to tell this PC's part from the account's total.
        self.usage_ledger = usage.ledger_path(cfg.db_path)
        self.app: Application | None = None
        self._pending_off_sends: dict[str, tuple[str, str]] = {}
        self._pending_off_seq = 0
        self._pending_asks: dict[str, tuple[asyncio.Future[str], list[str]]] = {}
        self._pending_ask_seq = 0
        # I2: token -> the ask question's message_id, so a quote-reply (text or
        # voice) to THAT message can resolve THAT specific ask. Kept in lockstep
        # with _pending_asks (set after a successful send in ask_user, popped by
        # ask_user's finally on the button/timeout path and by resolve_ask on
        # the phone-answer path).
        self._ask_msg_ids: dict[str, int] = {}
        # Live references to the fire-and-forget "Answered: …" edit tasks
        # spawned by resolve_ask (which is sync). Held so the tasks are not
        # garbage-collected mid-flight; each drops itself on completion.
        self._ask_edit_tasks: set[asyncio.Task] = set()
        # /live: the already-running Claude Code session this chat is currently
        # driving (None = not attached, everything routes to bridge projects as
        # usual), plus the task tailing its transcript back to Telegram.
        self._live_session = None
        self._live_task: asyncio.Task | None = None
        # Editor permission prompts relayed here for an ✅/❌ tap. The IDE hook
        # drops a request file and BLOCKS on the answer file we write back, so
        # an unanswered request simply falls back to the editor's own prompt.
        self._perm_task: asyncio.Task | None = None
        self._perm_seen: set[str] = set()
        # ident -> the buttons message, so it can be marked expired once the
        # editor session stops waiting (otherwise a late tap looks accepted
        # while nothing is listening any more).
        self._perm_pending: dict[str, object] = {}

    # --- whitelist -------------------------------------------------------
    def _allowed(self, user_id: int | None) -> bool:
        return user_id == self.cfg.telegram_allowed_user_id

    @property
    def _chat_id(self) -> int:
        # Single-chat bot: the only authorized user is also the chat target.
        return self.cfg.telegram_allowed_user_id

    # --- inbound handlers ------------------------------------------------
    @staticmethod
    def _reply_to(msg) -> int | None:
        if msg.reply_to_message is not None:
            return msg.reply_to_message.message_id
        return None

    async def _handle_text(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        msg = update.message
        if msg is None or not self._allowed(msg.from_user.id):
            return
        await self.on_user_message({
            "message_id": msg.message_id,
            "reply_to": self._reply_to(msg),
            "text": msg.text or "",
            "is_voice": False,
            "audio": None,
        })

    async def _handle_voice(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        msg = update.message
        if msg is None or not self._allowed(msg.from_user.id):
            return
        # Bug fix: a voice note over Telegram's ~20 MB getFile cap raises
        # BadRequest ("file is too big") with no handler here, so it used to
        # vanish silently. Tell the user instead of letting the handler
        # crash / the message disappear.
        #
        # Review fix: only the oversize case is a BadRequest we can safely
        # swallow -- an unrelated BadRequest (expired/invalid file_id, etc.)
        # must not be mislabeled as "too big", so it re-raises instead
        # (mirrors this file's `_answer_quietly`/`_edit_callback_markup`
        # convention of checking `str(exc).lower()` before handling).
        try:
            tg_file = await msg.voice.get_file()
            audio = bytes(await tg_file.download_as_bytearray())
        except BadRequest as exc:
            if "too big" not in str(exc).lower() and "too large" not in str(exc).lower():
                raise
            logger.warning(
                "voice download failed (likely >20MB Telegram cap), message %s",
                msg.message_id,
            )
            await msg.reply_text(t("file.too_large"))
            return
        await self.on_user_message({
            "message_id": msg.message_id,
            "reply_to": self._reply_to(msg),
            "text": "",
            "is_voice": True,
            "audio": audio,
        })

    async def _handle_attachment(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        msg = update.message
        if msg is None or not self._allowed(msg.from_user.id):
            return
        # Bug fix: a document/photo/audio/video over Telegram's ~20 MB
        # getFile cap raises BadRequest ("file is too big") with no handler
        # here, so it used to vanish silently. Tell the user instead of
        # letting the handler crash / the attachment disappear.
        #
        # Review fix: only the oversize case is a BadRequest we can safely
        # swallow -- an unrelated BadRequest (expired/invalid file_id, etc.)
        # must not be mislabeled as "too big", so it re-raises instead
        # (mirrors this file's `_answer_quietly`/`_edit_callback_markup`
        # convention of checking `str(exc).lower()` before handling).
        try:
            attachment = await _download_attachment(msg)
        except BadRequest as exc:
            if "too big" not in str(exc).lower() and "too large" not in str(exc).lower():
                raise
            logger.warning(
                "attachment download failed (likely >20MB Telegram cap), message %s",
                msg.message_id,
            )
            await msg.reply_text(t("file.too_large"))
            return
        if attachment is None:
            return
        await self.on_user_message({
            "message_id": msg.message_id,
            "reply_to": self._reply_to(msg),
            "text": msg.caption or "",
            "is_voice": False,
            "audio": None,
            "attachments": [attachment],
        })

    # --- outbound --------------------------------------------------------
    async def _send_text_chunks(self, project: str, text: str) -> list[int]:
        """Send ``text`` as one or more <=4096-char messages.

        The ``[{project}] `` prefix is added once, to the FIRST chunk only
        (subsequent chunks are plain continuations of the same message).
        Every resulting message_id is returned so the caller can map ALL of
        them to the project for reply routing.
        """
        bot = self.app.bot
        ids: list[int] = []
        for chunk in _chunk_text(f"[{project}] {text}"):
            msg = await _send_with_retry(
                lambda chunk=chunk: bot.send_message(
                    chat_id=self._chat_id, text=chunk
                )
            )
            ids.append(msg.message_id)
        return ids

    async def send_update(
        self,
        project: str,
        voice_label: str,
        text: str,
        voice_bytes: bytes | None,
    ) -> list[int]:
        """Send a TEXT message (full, may contain code) and, if voice_bytes
        is provided, a VOICE message. Return the message_ids sent.

        ``text`` is chunked (see ``_send_text_chunks``) since Telegram caps a
        single message at 4096 chars; every chunk's message_id comes back so
        the caller maps ALL of them to the project. Transient Telegram
        errors are retried via ``_send_with_retry``.
        """
        bot = self.app.bot
        ids = await self._send_text_chunks(project, text)
        if voice_bytes is not None:
            voice_msg = await _send_with_retry(
                lambda: bot.send_voice(
                    chat_id=self._chat_id,
                    voice=voice_bytes,
                    caption=f"{project} · {voice_label}",
                )
            )
            ids.append(voice_msg.message_id)
        row = _find_project_row(self.controls.snapshot(), project)
        cwd = (row or {}).get("cwd") or ""
        for mid in ids:
            try:
                sent_log.record(mid, None, cwd)
            except OSError:
                logger.exception("could not record message %s", mid)
        return ids

    async def send_question(
        self,
        project: str,
        text: str,
        *,
        voice_label: str | None = None,
        voice_bytes: bytes | None = None,
        approval_token: int | None = None,
    ) -> int:
        """Send one message and return its (text) message_id (keys approvals).

        When ``approval_token`` is given, the message carries inline
        ✅ Leisti / ❌ Neleisti buttons whose ``callback_data`` encodes the
        token (``apv:{token}:1`` / ``apv:{token}:0``). When ``voice_bytes`` is
        given, an accompanying VOICE message is sent (used for the ALERT-voiced
        spoken approval line); the returned id is still the text message's, so
        both the quote-reply and inline-button paths key the same approval.
        """
        bot = self.app.bot
        reply_markup = None
        if approval_token is not None:
            # Allow-once (1) / Deny (0) on the first row; the more powerful
            # always-allow (2) sits on its own row so it is harder to fat-finger.
            reply_markup = InlineKeyboardMarkup([
                [
                    InlineKeyboardButton(
                        t("approval.allow"), callback_data=f"apv:{approval_token}:1"),
                    InlineKeyboardButton(
                        t("approval.deny"), callback_data=f"apv:{approval_token}:0"),
                ],
                [
                    InlineKeyboardButton(
                        t("approval.always"), callback_data=f"apv:{approval_token}:2"),
                ],
            ])
        text = _truncate_approval_preview(text)
        msg = await _send_with_retry(
            lambda: bot.send_message(
                chat_id=self._chat_id,
                text=f"[{project}] {text}",
                reply_markup=reply_markup,
            )
        )
        if voice_bytes is not None:
            await _send_with_retry(
                lambda: bot.send_voice(
                    chat_id=self._chat_id,
                    voice=voice_bytes,
                    caption=f"{project} · {voice_label}",
                )
            )
        return msg.message_id

    async def ask_user(self, project: str, question: str, choices: list[str]) -> str:
        """Ask a tappable multiple-choice question; return the chosen label.

        Reliability fix (audit-confirmed): the pending future used to be
        registered in ``_pending_asks`` BEFORE the send, so a transient send
        failure (RetryAfter/network) would both drop the question AND leak a
        pending entry that could never resolve. The send now goes through
        ``_send_with_retry`` and the pending entry is registered only AFTER
        it succeeds; a persistent send failure returns ``""`` (same sentinel
        as a timeout) instead of leaving phantom state or raising.

        Review fix: the except clause used to list only
        ``(BadRequest, NetworkError, RetryAfter, TimedOut)``, but
        ``telegram.error.Forbidden`` (bot blocked by the user), ``Conflict``,
        ``InvalidToken``, etc. are siblings of ``NetworkError`` under
        ``TelegramError`` and fell through uncaught -- contradicting this
        docstring's "never raises" claim. Catching ``TelegramError`` (the
        common base) keeps the same clean return for every Telegram failure.
        """
        clean_choices = _clean_choices(choices)
        if not clean_choices:
            clean_choices = ["Yes", "No"]
        self._pending_ask_seq += 1
        token = str(self._pending_ask_seq)
        rows = [
            [InlineKeyboardButton(choice, callback_data=f"ask:{token}:{idx}")]
            for idx, choice in enumerate(clean_choices)
        ]
        bot = self.app.bot
        try:
            msg = await _send_with_retry(
                lambda: bot.send_message(
                    chat_id=self._chat_id,
                    text=f"[{project}] {question}",
                    reply_markup=InlineKeyboardMarkup(rows),
                )
            )
        except TelegramError:
            logger.exception(
                "ask_user: send failed after retries for %s; no pending registered",
                project,
            )
            return ""

        loop = asyncio.get_running_loop()
        future: asyncio.Future[str] = loop.create_future()
        self._pending_asks[token] = (future, clean_choices)
        # I2: map the question's message_id -> token so a quote-reply to THIS
        # message resolves THIS ask (see resolve_ask / the inbound router).
        # Defensive: if the send returned no usable message_id, skip the mapping
        # — the single-pending fallback still lets a plain reply answer the sole
        # outstanding question.
        message_id = getattr(msg, "message_id", None)
        if message_id is not None:
            self._ask_msg_ids[token] = message_id
        try:
            return await asyncio.wait_for(future, timeout=self.cfg.approval_timeout)
        except asyncio.TimeoutError:
            return ""
        finally:
            self._pending_asks.pop(token, None)
            self._ask_msg_ids.pop(token, None)

    # --- I2: answer a pending ask from the phone (text/voice), not just taps --
    def pending_ask_token_for_message(self, message_id: int) -> str | None:
        """Token of the pending ask whose question is *message_id*, or None.

        Reverse lookup for the inbound router: a quote-reply to a specific
        ``ask_user`` question resolves THAT question. A mapping whose token is
        no longer in ``_pending_asks`` (already answered / expired) is ignored
        so a stale message_id never matches. Pure; never raises."""
        for token, mid in self._ask_msg_ids.items():
            if mid == message_id and token in self._pending_asks:
                return token
        return None

    def single_pending_ask_token(self) -> str | None:
        """The sole outstanding ask token, or None unless EXACTLY one is pending.

        Lets the inbound router treat a plain reply (no quote-reply, no
        name-prefix) as the answer when there is no ambiguity about which
        question it answers. With zero or several pending, the message routes
        normally. Pure."""
        if len(self._pending_asks) == 1:
            return next(iter(self._pending_asks))
        return None

    def has_pending_asks(self) -> bool:
        """True if any ``ask_user`` question is currently awaiting an answer."""
        return bool(self._pending_asks)

    def resolve_ask(self, token: str, answer_text: str) -> bool:
        """Resolve a pending ask from a phone reply. Return True if resolved.

        SYNC by design: the inbound router calls it without ``await`` (like the
        approval interception it mirrors), so the cosmetic "Answered: …" edit is
        fired-and-forgotten rather than awaited. Returns False (message falls
        through to normal routing) when the token is unknown / already answered,
        or the answer is blank — never on a mere no-match, since a free-text
        answer is itself a valid, useful reply for a hands-free user.

        *answer_text* is mapped to a choice via :func:`_match_choice`; on a match
        the future resolves to that choice's LABEL, otherwise to the RAW answer
        (free-form passthrough — ``ask_user`` returns a plain string to the
        agent). The ``_pending_asks`` pop stays with ``ask_user``'s finally (the
        awaiter owns it, same as the button path); only ``_ask_msg_ids`` is
        popped here."""
        pending = self._pending_asks.get(token)
        if pending is None:
            return False
        future, choices = pending
        if future.done():
            return False
        if not answer_text or not answer_text.strip():
            return False
        matched = _match_choice(answer_text, choices)
        resolved = matched if matched is not None else answer_text
        if not future.done():
            future.set_result(resolved)
        message_id = self._ask_msg_ids.pop(token, None)
        if message_id is not None:
            self._schedule_ask_edit(message_id, resolved)
        return True

    def _schedule_ask_edit(self, message_id: int, resolved: str) -> None:
        """Fire-and-forget the "Answered: …" message edit (best-effort).

        resolve_ask is sync, so the async edit is scheduled as a task instead of
        awaited. With no running loop (unusual — resolve_ask runs inside the
        inbound coroutine) the edit is simply skipped; the answer is already
        delivered to the agent, which is what matters."""
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return
        task = loop.create_task(self._edit_ask_message(message_id, resolved))
        self._ask_edit_tasks.add(task)
        task.add_done_callback(self._ask_edit_tasks.discard)

    async def _edit_ask_message(self, message_id: int, resolved: str) -> None:
        try:
            await self.app.bot.edit_message_text(
                chat_id=self._chat_id,
                message_id=message_id,
                text=t("ask.answered", answer=resolved),
            )
        except TelegramError:
            # Cosmetic only (mark the question answered); the agent already has
            # its answer, so a failed edit must never surface as a failure.
            logger.debug("resolve_ask: message edit failed for %s", message_id)

    async def send_file(
        self,
        project: str,
        voice_label: str,
        text: str,
        voice_bytes: bytes | None,
        file_path: str,
    ) -> list[int]:
        """Send a project-produced file and optional voice summary.

        Telegram caps a media caption at ~1024 chars. When the full
        ``"[{project}] {text}"`` caption would exceed that, the file goes
        out with a short caption instead and the full text follows as a
        separate chunked message (see ``_send_text_chunks``).
        """
        bot = self.app.bot
        ids: list[int] = []
        path = Path(file_path)
        full_caption = f"[{project}] {text}".strip()
        suffix = path.suffix.lower()

        if len(full_caption) <= _CAPTION_LIMIT:
            caption = full_caption
            overflow_text: str | None = None
        else:
            caption = f"[{project}]"
            overflow_text = text

        with path.open("rb") as fh:
            def _seek_and(factory):
                def _call():
                    fh.seek(0)
                    return factory()
                return _call

            if suffix in _PHOTO_SUFFIXES:
                msg = await _send_with_retry(_seek_and(
                    lambda: bot.send_photo(
                        chat_id=self._chat_id, photo=fh, caption=caption
                    )
                ))
            elif suffix in _AUDIO_SUFFIXES:
                msg = await _send_with_retry(_seek_and(
                    lambda: bot.send_audio(
                        chat_id=self._chat_id, audio=fh, caption=caption
                    )
                ))
            elif suffix in _VIDEO_SUFFIXES:
                msg = await _send_with_retry(_seek_and(
                    lambda: bot.send_video(
                        chat_id=self._chat_id, video=fh, caption=caption
                    )
                ))
            else:
                msg = await _send_with_retry(_seek_and(
                    lambda: bot.send_document(
                        chat_id=self._chat_id,
                        document=fh,
                        caption=caption,
                        filename=path.name,
                    )
                ))
        ids.append(msg.message_id)

        if overflow_text is not None:
            ids.extend(await self._send_text_chunks(project, overflow_text))

        if voice_bytes is not None:
            voice_msg = await _send_with_retry(
                lambda: bot.send_voice(
                    chat_id=self._chat_id,
                    voice=voice_bytes,
                    caption=f"{project} · {voice_label}",
                )
            )
            ids.append(voice_msg.message_id)
        return ids

    async def send_disabled_project_prompt(self, project: str, text: str) -> int | None:
        """Ask whether to enable a disabled project and send the pending turn.

        Reliability fix (audit-confirmed): the pending entry used to be
        registered in ``_pending_off_sends`` BEFORE the send, so a transient
        send failure (RetryAfter/network) would both drop the prompt AND leak
        a pending entry nothing could ever resolve. The send now goes through
        ``_send_with_retry`` and the pending entry is registered only AFTER
        it succeeds. A persistent send failure returns ``None`` instead of
        leaving phantom state or raising (never-crash posture, matching the
        rest of this module's send paths).

        Review fix: the except clause used to list only
        ``(BadRequest, NetworkError, RetryAfter, TimedOut)``, but
        ``telegram.error.Forbidden`` (bot blocked by the user), ``Conflict``,
        ``InvalidToken``, etc. are siblings of ``NetworkError`` under
        ``TelegramError`` and fell through uncaught -- contradicting the
        never-crash posture above. Catching ``TelegramError`` (the common
        base) keeps the same clean return for every Telegram failure.
        """
        self._pending_off_seq += 1
        token = str(self._pending_off_seq)
        markup = InlineKeyboardMarkup([
            [
                InlineKeyboardButton(
                    t("off.enable_and_send"), callback_data=f"offsend:{token}"
                )
            ],
            [InlineKeyboardButton(t("off.cancel"), callback_data=f"offcancel:{token}")],
        ])
        bot = self.app.bot
        try:
            msg = await _send_with_retry(
                lambda: bot.send_message(
                    chat_id=self._chat_id,
                    text=t("off.prompt", project=project),
                    reply_markup=markup,
                )
            )
        except TelegramError:
            logger.exception(
                "send_disabled_project_prompt: send failed after retries for "
                "%s; no pending registered",
                project,
            )
            return None
        self._pending_off_sends[token] = (project, text)
        return msg.message_id

    # --- /panel + callbacks ---------------------------------------------
    async def _cmd_panel(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        msg = update.message
        if msg is None or not self._allowed(msg.from_user.id):
            return
        markup = build_panel_markup(self.controls.snapshot())
        await msg.reply_text(t("panel.title"), reply_markup=markup)

    async def _cmd_menu(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        msg = update.message
        if msg is None or not self._allowed(msg.from_user.id):
            return
        # The bot's own Telegram name, so every install shows its own.
        name = getattr(getattr(self.app, "bot", None), "first_name", None)
        title = f"🏠 {name}" if isinstance(name, str) and name else t("menu.title")
        await msg.reply_text(title, reply_markup=build_menu_markup())

    async def _handle_callback(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        query = update.callback_query
        if query is None or not self._allowed(query.from_user.id):
            return
        action, index_str = parse_callback(query.data)

        # Inline approval taps own their query.answer() (a stale token shows a
        # toast), so they are dispatched BEFORE the generic acknowledgement.
        if action == "apv":
            await self._handle_approval_callback(query, index_str)
            return

        try:
            await query.answer()
        except BadRequest as exc:
            if "query is too old" in str(exc).lower():
                return
            raise

        if action == "noop":
            return
        if action == "back":
            await self._edit_callback_markup(
                query, build_panel_markup(self.controls.snapshot())
            )
            return
        if action in {"offsend", "offcancel"}:
            pending = self._pending_off_sends.pop(index_str, None)
            if pending is None:
                await query.edit_message_text(t("off.expired"))
                return
            project, text = pending
            if action == "offcancel":
                await query.edit_message_text(t("off.cancelled", project=project))
                return
            await self.controls.enable_and_deliver(project, text)
            await query.edit_message_text(t("off.sent", project=project))
            return
        if action == "menu":
            await self._handle_menu_callback(query, index_str)
            return
        if action == "ask":
            try:
                token, choice_idx = index_str.split(":", 1)
                idx = int(choice_idx)
            except (ValueError, TypeError):
                return
            pending = self._pending_asks.get(token)
            if pending is None:
                await query.edit_message_text(t("ask.expired"))
                return
            future, choices = pending
            if idx < 0 or idx >= len(choices):
                return
            choice = choices[idx]
            if not future.done():
                future.set_result(choice)
            await query.edit_message_text(t("ask.selected", choice=choice))
            return
        if action == "live":
            await query.edit_message_text(await self._attach_live(index_str))
            if self._live_session is not None:
                # "🔗 Prisijungta" is what one naturally replies to next.
                await self._remember_sent(
                    query.message, self._live_session.cwd, self._live_session.session_id
                )
            return
        if action == "agent":
            await query.edit_message_text(self._switch_agent(index_str))
            self._restart_if_switched(index_str)
            return
        if action == "perm":
            if not self._claude_live_enabled:
                await query.edit_message_text(t("codex.no_claude_permissions"))
                return
            ident, _, code = index_str.rpartition(":")
            allow = code == "1"
            if self.answer_permission(ident, allow):
                await query.edit_message_text(
                    t("approval.allowed") if allow else t("approval.denied"))
            else:
                # Almost always: the editor session stopped waiting before the
                # tap landed, so it is already asking there instead.
                await query.edit_message_text(t("approval.too_late"))
            return
        if action == "ans":
            await self._answer_from_button(query, index_str)
            return
        if action == "liveans":
            # Answer a live session's plain-text question by sending the chosen
            # number, exactly as typing it would.
            if self.live_target() is None:
                await query.edit_message_text(t("live.not_attached"))
                return
            if await self.live_send(index_str):
                await query.edit_message_text(t("answer.sent", answer=index_str))
            else:
                await query.edit_message_text(t("answer.failed"))
            return
        if action in {"pc", "pcgo", "pcno"}:
            await self._handle_pc_callback(query, action, index_str)
            return
        if action == "cost":
            # Info action: reply with a fresh message, do not touch the panel.
            await query.message.reply_text(await asyncio.to_thread(usage.format_usage, self.usage_ledger))
            return
        if action == "recap":
            await query.message.reply_text(self.controls.recap())
            return

        # Global actions do not need a project index.
        if action == "allon":
            await self.controls.toggle(None, True)
        elif action == "alloff":
            await self.controls.toggle(None, False)
        elif action == "engine":
            snap_list = self.controls.snapshot()
            cur = snap_list[0]["engine"] if snap_list else _ENGINES[0]
            await self.controls.set_engine(_next(_ENGINES, cur))
        else:
            # Per-project actions: resolve project by index from a fresh snapshot.
            value = ""
            if action in {"mset", "vset"}:
                try:
                    index_str, value = index_str.split(":", 1)
                except ValueError:
                    return
            try:
                idx = int(index_str)
            except (ValueError, TypeError):
                return
            snap_list = self.controls.snapshot()
            if idx < 0 or idx >= len(snap_list):
                return  # guard against out-of-range
            row = snap_list[idx]
            project = row["project"]

            if action == "tog":
                await self.controls.toggle(project, not row["enabled"])
            elif action == "verb":
                await self.controls.set_verbose(project, not row.get("verbose", False))
            elif action in {"sel"}:
                await self.controls.select(project)
                snap = self.controls.snapshot()
                await self._edit_callback_text(
                    query,
                    format_projects(snap, open_projects=self._open_projects()),
                    build_projects_list_markup(snap),
                )
                return
            elif action == "ptgl":
                turning_off = row["enabled"]
                await self.controls.toggle(project, not row["enabled"])
                snap = self.controls.snapshot()
                text = format_projects(snap, open_projects=self._open_projects())
                if turning_off:
                    # Disabling drops this project's queued turns; note it
                    # instead of a silent redraw (audit finding #2).
                    text = t("projects.off_dropped", project=html.escape(project)) + "\n\n" + text
                await self._edit_callback_text(
                    query,
                    text,
                    build_projects_list_markup(snap),
                )
                return
            elif action == "mode":
                await self._edit_callback_markup(query, build_mode_markup(snap_list, idx))
                return
            elif action == "voice":
                await self._edit_callback_markup(query, build_voice_markup(snap_list, idx))
                return
            elif action == "mset":
                if value not in _MODES:
                    return
                await self.controls.set_mode(project, value)
            elif action == "vset":
                if value not in available_voices(row.get("engine", "openai")):
                    return
                await self.controls.set_voice(project, value)
            else:
                return

        new_markup = build_panel_markup(self.controls.snapshot())
        await self._edit_callback_markup(query, new_markup)

    async def _handle_approval_callback(self, query, index_str: str) -> None:
        """Resolve an inline approval tap by token.

        ``index_str`` is ``"{token}:{code}"`` where ``code`` is ``0`` = deny,
        ``1`` = allow-once, ``2`` = allow + persist an always-allow policy. A
        token with no live pending (already answered via quote-reply, or timed
        out) answers with a "no longer relevant" toast and leaves the message
        untouched — and persists NOTHING. A resolved tap edits the message to
        show the outcome and removes the buttons.
        """
        try:
            token_str, code_str = index_str.split(":", 1)
            token = int(token_str)
        except (ValueError, TypeError):
            await self._answer_quietly(query)
            return
        always = code_str == "2"
        approved = code_str in ("1", "2")
        resolved = (
            self._on_approval(token, approved)
            if self._on_approval is not None
            else False
        )
        if not resolved:
            # Stale/already-answered: no policy is persisted for a tap that did
            # not actually resolve a live approval.
            await self._answer_quietly(query, "nebeaktualu")
            return
        persisted = False
        if always and self._on_always_allow is not None:
            # Persist the policy. The approval is ALREADY resolved as allow, so
            # a persist failure degrades to allow-once — it must never raise
            # out of the callback (never-crash posture). The hook returns False
            # when the call is not policy-eligible (signature None), so the
            # label stays honest rather than claiming a persistent grant.
            try:
                persisted = await self._on_always_allow(token)
            except Exception:  # noqa: BLE001 - persist is best-effort
                logger.exception("always-allow persist failed for token %s", token)
        await self._answer_quietly(query)
        if always and persisted:
            label = t("approval.always_done")
        elif always:
            label = t("approval.once_only")
        elif approved:
            label = t("approval.allowed_label")
        else:
            label = t("approval.denied_label")
        try:
            await query.edit_message_text(label)
        except BadRequest as exc:
            if "message is not modified" not in str(exc).lower():
                raise

    @staticmethod
    async def _answer_quietly(query, text: str | None = None) -> None:
        try:
            await query.answer(text) if text is not None else await query.answer()
        except BadRequest as exc:
            if "query is too old" not in str(exc).lower():
                raise

    async def _handle_menu_callback(self, query, action: str) -> None:
        snapshot = self.controls.snapshot()
        if action == "projects":
            await self._edit_callback_text(
                query,
                format_projects(snapshot, open_projects=self._open_projects()),
                build_projects_list_markup(snapshot),
            )
        elif action.startswith("projects_all"):
            _, _, page_str = action.partition(":")
            page = int(page_str) if page_str.isdigit() else 0
            await self._edit_callback_text(
                query,
                format_projects(snapshot, show_all=True, page=page, open_projects=self._open_projects()),
                build_projects_list_markup(snapshot, show_all=True, page=page),
            )
        elif action == "panel":
            await self._edit_callback_text(query, t("panel.title"), build_panel_markup(snapshot))
        elif action == "refresh":
            added = await self.controls.refresh_projects()
            snapshot = self.controls.snapshot()
            await self._edit_callback_text(
                query,
                t("projects.added", n=added) + "\n\n"
                + format_projects(snapshot, show_all=True, open_projects=self._open_projects()),
                build_projects_list_markup(snapshot, show_all=True),
            )
        elif action == "stop":
            await self._edit_callback_text(
                query,
                await self.controls.interrupt(None),
                build_menu_markup(),
            )
        elif action == "handoff":
            await self._edit_callback_text(
                query,
                self._format_handoff_text(""),
                build_menu_markup(),
            )
        elif action == "policies":
            # Reply with a fresh PLAIN message (like cost/recap) so the
            # HTML-free policy text is never parsed as HTML, and the menu stays.
            policies = await self.controls.list_policies()
            await query.message.reply_text(_format_policies(policies))

    async def _edit_callback_markup(self, query, new_markup: InlineKeyboardMarkup) -> None:
        try:
            await query.edit_message_reply_markup(reply_markup=new_markup)
        except BadRequest as exc:
            if "message is not modified" not in str(exc).lower():
                raise

    async def _edit_callback_text(
        self, query, text: str, markup: InlineKeyboardMarkup
    ) -> None:
        try:
            await query.edit_message_text(
                text=text,
                parse_mode="HTML",
                reply_markup=markup,
            )
        except BadRequest as exc:
            if "message is not modified" not in str(exc).lower():
                raise

    # --- project-arg validation ------------------------------------------
    # Shared by every command that takes an optional ``[project]`` arg
    # (/on /off /stop /mode /effort /verbose /voice). Before this helper
    # existed, a typo'd or stale project name was passed straight to the
    # Controls mutator, which silently no-ops on an unknown key while the
    # command still replied as if it worked (audit finding #1).
    def _known_projects(self) -> list[str]:
        return [row["project"] for row in self.controls.snapshot()]

    def _resolve_project_arg(
        self, name: str | None
    ) -> tuple[str | None, str | None]:
        """Validate an optional project-name slash-command argument.

        ``name`` is ``None`` when the command was given no project arg at
        all (meaning "all projects" — always valid, no lookup needed).
        When a name IS given but doesn't match any ``project`` key in
        ``controls.snapshot()``, this returns an error reply naming it
        unknown and listing the known projects; the caller MUST send that
        reply and return WITHOUT calling any Controls mutator.

        Returns ``(project, error)`` where exactly one of the two is not
        ``None``.
        """
        if name is None:
            return None, None
        known = self._known_projects()
        if name not in known:
            return None, t("projects.unknown", name=name, known=", ".join(known))
        return name, None

    def _voice_choices_for_engine(self, engine: str) -> list[str]:
        """Voices accepted by ``/voice <name>`` for the given ``engine``.

        ``available_voices("auto")`` only returns the OpenAI list (AutoTTS's
        preferred choice), but at runtime "auto" can fall back to piper or
        together, so validating an "auto" project's voice against just the
        OpenAI list would reject perfectly legitimate names. Union the
        concrete backends' voices instead.
        """
        if engine != "auto":
            return available_voices(engine)
        choices: list[str] = []
        for backend in ("openai", "piper", "together"):
            for voice in available_voices(backend):
                if voice not in choices:
                    choices.append(voice)
        return choices

    # --- text slash commands --------------------------------------------
    async def _cmd_projects(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        msg = update.message
        if msg is None or not self._allowed(msg.from_user.id):
            return
        show_all = bool(context.args and context.args[0] == "all")
        snapshot = self.controls.snapshot()
        await msg.reply_text(
            format_projects(snapshot, show_all=show_all, open_projects=self._open_projects()),
            parse_mode="HTML",
            reply_markup=build_projects_list_markup(snapshot, show_all=show_all),
        )

    async def _cmd_projects_all(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        msg = update.message
        if msg is None or not self._allowed(msg.from_user.id):
            return
        snapshot = self.controls.snapshot()
        await msg.reply_text(
            format_projects(snapshot, show_all=True, open_projects=self._open_projects()),
            parse_mode="HTML",
            reply_markup=build_projects_list_markup(snapshot, show_all=True),
        )

    async def _cmd_projects_refresh(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        msg = update.message
        if msg is None or not self._allowed(msg.from_user.id):
            return
        added = await self.controls.refresh_projects()
        snapshot = self.controls.snapshot()
        await msg.reply_text(
            t("projects.added", n=added) + "\n\n"
            + format_projects(snapshot, show_all=True, open_projects=self._open_projects()),
            parse_mode="HTML",
            reply_markup=build_projects_list_markup(snapshot, show_all=True),
        )

    async def _cmd_newproject(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        """``/newproject <name>``: create a brand-new project folder and
        switch to it so the user's next message routes straight there."""
        msg = update.message
        if msg is None or not self._allowed(msg.from_user.id):
            return
        if not context.args:
            await msg.reply_text(t("newproject.usage"))
            return
        name = context.args[0]
        result = await self.controls.create_project(name)
        await msg.reply_text(result)

    async def _cmd_on(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        msg = update.message
        if msg is None or not self._allowed(msg.from_user.id):
            return
        arg = context.args[0] if context.args else None
        project, error = self._resolve_project_arg(arg)
        if error:
            await msg.reply_text(error)
            return
        await self.controls.toggle(project, True)
        await msg.reply_text(t("projects.on", project=project or t("projects.all")))

    async def _cmd_off(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        msg = update.message
        if msg is None or not self._allowed(msg.from_user.id):
            return
        arg = context.args[0] if context.args else None
        project, error = self._resolve_project_arg(arg)
        if error:
            await msg.reply_text(error)
            return
        await self.controls.toggle(project, False)
        # Disabling drops that project's queued turns; say so instead of a
        # bare "x off" that hides the fact that pending work was discarded
        # (audit finding #2).
        await msg.reply_text(t("projects.off_dropped", project=project or t("projects.all")))

    async def _cmd_stop(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        msg = update.message
        if msg is None or not self._allowed(msg.from_user.id):
            return
        arg = context.args[0] if context.args else None
        project, error = self._resolve_project_arg(arg)
        if error:
            await msg.reply_text(error)
            return
        result = await self.controls.interrupt(project)
        await msg.reply_text(result)

    async def _cmd_mode(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        msg = update.message
        if msg is None or not self._allowed(msg.from_user.id):
            return
        if not context.args or context.args[0] not in _MODES:
            await msg.reply_text(t("mode.usage"))
            return
        mode = context.args[0]
        arg = context.args[1] if len(context.args) > 1 else None
        project, error = self._resolve_project_arg(arg)
        if error:
            await msg.reply_text(error)
            return
        await self.controls.set_mode(project, mode)
        await msg.reply_text(t("mode.set", mode=mode, project=project or t("projects.all")))

    async def _cmd_effort(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        """Set per-project reasoning effort: ``/effort <level> [project]``.

        No project arg targets all projects. A live change restarts the running
        session so the new effort applies (mirrors /mode)."""
        msg = update.message
        if msg is None or not self._allowed(msg.from_user.id):
            return
        if not context.args or context.args[0] not in _EFFORTS:
            await msg.reply_text(t("effort.usage", levels="|".join(_EFFORTS)))
            return
        level = context.args[0]
        arg = context.args[1] if len(context.args) > 1 else None
        project, error = self._resolve_project_arg(arg)
        if error:
            await msg.reply_text(error)
            return
        await self.controls.set_effort(project, level)
        await msg.reply_text(t("effort.set", level=level, project=project or t("projects.all")))

    async def _cmd_info(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        """Per-project model (config + real), effort, mode, voice, verbose."""
        msg = update.message
        if msg is None or not self._allowed(msg.from_user.id):
            return
        await msg.reply_text(self.controls.info())

    async def _cmd_verbose(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        """Toggle live tool-activity streaming: ``/verbose [on|off] [project]``.

        A leading ``on``/``off`` sets the state (defaults to ``on`` when
        omitted); a trailing token is the project (else all projects)."""
        msg = update.message
        if msg is None or not self._allowed(msg.from_user.id):
            return
        args = list(context.args or [])
        on = True
        if args and args[0].lower() in {"on", "off"}:
            on = args.pop(0).lower() == "on"
        arg = args[0] if args else None
        project, error = self._resolve_project_arg(arg)
        if error:
            await msg.reply_text(error)
            return
        await self.controls.set_verbose(project, on)
        state = "on" if on else "off"
        await msg.reply_text(t("verbose.set", state=state, project=project or t("projects.all")))

    async def _cmd_voice(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        msg = update.message
        if msg is None or not self._allowed(msg.from_user.id):
            return
        args = context.args
        if not args or args[0] == "list":
            snapshot = self.controls.snapshot()
            current = snapshot[0]["engine"] if snapshot else "openai"
            engine = args[1] if len(args) >= 2 else current
            await msg.reply_text(t("voice.list", voices=", ".join(available_voices(engine))))
            return
        voice = args[0]
        arg = None
        if len(args) >= 3 and args[1] == "for":
            arg = args[2]
        project, error = self._resolve_project_arg(arg)
        if error:
            await msg.reply_text(error)
            return
        # An invalid voice name would set silently and make every later
        # synth fail with no feedback, so validate against the target
        # engine's known voices before calling the mutator (audit finding
        # #3). ``project=None`` means "all", so fall back to whichever
        # project's row would be picked as "active" (or the first one) to
        # find the engine, matching how /voice list already resolves it.
        snapshot = self.controls.snapshot()
        row = _find_project_row(snapshot, project or "")
        engine = row.get("engine", "openai") if row else "openai"
        valid_voices = self._voice_choices_for_engine(engine)
        if voice not in valid_voices:
            await msg.reply_text(t("voice.unknown", voice=voice, known=", ".join(valid_voices)))
            return
        await self.controls.set_voice(project, voice)
        await msg.reply_text(t("voice.set", voice=voice, project=project or t("projects.all")))

    async def _cmd_engine(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        msg = update.message
        if msg is None or not self._allowed(msg.from_user.id):
            return
        if not context.args or context.args[0] not in _ENGINES:
            await msg.reply_text(t("engine.usage", engines="|".join(_ENGINES)))
            return
        name = context.args[0]
        await self.controls.set_engine(name)
        await msg.reply_text(t("engine.set", engine=name))

    async def _cmd_agent(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        """Show which agent answers here, or switch it: `/agent claude|codex`."""
        msg = update.message
        if msg is None or not self._allowed(msg.from_user.id):
            return
        if context.args:
            name = context.args[0].strip().lower()
            await msg.reply_text(self._switch_agent(name))
            self._restart_if_switched(name)
            return
        current = self.cfg.agent_backend
        buttons = [
            InlineKeyboardButton(
                ("✅ " if name == current else "") + name.capitalize(),
                callback_data=f"agent:{name}",
            )
            for name in AGENT_BACKENDS
        ]
        await msg.reply_text(
            t("agent.current", agent=current.capitalize()),
            reply_markup=InlineKeyboardMarkup([buttons]),
        )

    def _switch_agent(self, name: str) -> str:
        """Persist AGENT_BACKEND=*name*; return the text to show the user."""
        if name not in AGENT_BACKENDS:
            return t("agent.usage", agents="|".join(AGENT_BACKENDS))
        if name == self.cfg.agent_backend:
            return t("agent.already", agent=name.capitalize())
        try:
            set_env_value(self._env_path, "AGENT_BACKEND", name)
        except OSError:
            logger.exception("agent: could not rewrite %s", self._env_path)
            return t("agent.write_failed")
        self._agent_switched = name
        return t("agent.switching", agent=name.capitalize())

    def _restart_if_switched(self, name: str) -> None:
        # Only after the reply is out, and only if .env really changed: a failed
        # write must not restart into the same backend and pretend it switched.
        if self._agent_switched == name:
            self._restart()

    async def _cmd_pc(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        """/pc: suspend, power off or reboot this machine, each confirmed."""
        msg = update.message
        if msg is None or not self._allowed(msg.from_user.id):
            return
        if not getattr(self.cfg, "pc_power_commands", False):
            await msg.reply_text(t("pc.disabled"))
            return
        await msg.reply_text(t("pc.title"), reply_markup=InlineKeyboardMarkup([
            [InlineKeyboardButton(t(f"pc.{action}"), callback_data=f"pc:{action}")]
            for action in _PC_ACTIONS
        ]))

    async def _handle_pc_callback(self, query, action: str, arg: str) -> None:
        if not getattr(self.cfg, "pc_power_commands", False) or (
            action != "pcno" and arg not in _PC_ACTIONS
        ):
            await query.edit_message_text(t("pc.disabled"))
            return
        if action == "pcno":
            await query.edit_message_text(t("pc.cancelled"))
            return
        if action == "pc":
            # Nothing happens on the first tap: switching a PC off by a stray
            # touch in a pocket is exactly what must not be possible.
            await query.edit_message_text(
                t(f"pc.confirm_{arg}"),
                reply_markup=InlineKeyboardMarkup([[
                    InlineKeyboardButton(t("answer.yes_button"), callback_data=f"pcgo:{arg}"),
                    InlineKeyboardButton(t("answer.no_button"), callback_data="pcno:"),
                ]]),
            )
            return
        await query.edit_message_text(t(f"pc.doing_{arg}"))
        error = await self._power(arg)
        if error:
            await self._send_plain(t("pc.failed", error=error))

    async def _power(self, action: str) -> str | None:
        """Run ``systemctl <action>`` through logind; the error text, or None.

        A moment's pause first, so the confirmation edit reaches Telegram
        before the network goes down with the machine."""
        await asyncio.sleep(1.5)
        try:
            proc = await asyncio.create_subprocess_exec(
                "systemctl", action,
                stdout=asyncio.subprocess.DEVNULL, stderr=asyncio.subprocess.PIPE,
            )
            _, err = await proc.communicate()
        except OSError as exc:
            return str(exc)
        return (err.decode(errors="replace").strip() or f"exit {proc.returncode}") if proc.returncode else None

    async def _cmd_status(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        msg = update.message
        if msg is None or not self._allowed(msg.from_user.id):
            return
        project = context.args[0] if context.args else ""
        text = f"{project} status please".strip() or "status please"
        await self.on_user_message({
            "message_id": msg.message_id,
            "reply_to": None,
            "text": text,
            "is_voice": False,
            "audio": None,
        })

    async def _cmd_recap(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        """B3b: "what changed while I was gone" — cheap, synchronous, no LLM."""
        msg = update.message
        if msg is None or not self._allowed(msg.from_user.id):
            return
        await msg.reply_text(self.controls.recap())

    async def _cmd_cost(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        """/usage (and the old /cost): Claude limits in % plus this PC's sessions.

        Dollars were meaningless here: on a subscription the SDK reports no
        cost, so the old summary always said "n/a"."""
        msg = update.message
        if msg is None or not self._allowed(msg.from_user.id):
            return
        await msg.reply_text(await asyncio.to_thread(usage.format_usage, self.usage_ledger))

    async def _cmd_policies(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        """Show or revoke always-allow policies (SECURITY visibility).

        ``/policies`` lists every ``(project, signature)`` the user has granted
        via the "✅♾ Visada leisti" button. ``/policies clear`` revokes ALL of
        them; ``/policies clear <project>`` revokes just that project's. This is
        the escape hatch for a security setting, so it is owner-gated like every
        other command.
        """
        msg = update.message
        if msg is None or not self._allowed(msg.from_user.id):
            return
        args = list(context.args or [])
        if args and args[0] == "clear":
            target = args[1] if len(args) > 1 else None
            await self.controls.clear_policies(target)
            await msg.reply_text(t("policies.cleared", scope=target or t("projects.all")))
            return
        policies = await self.controls.list_policies()
        await msg.reply_text(_format_policies(policies))

    async def _cmd_schedule(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        """Manage daily recurring prompts (I4), owner-gated like every command.

        Sub-commands (dispatched on the FIRST arg):

        * (none) / ``list`` → list every schedule (plain text, HTML-free).
        * ``remove``/``rm``/``del`` ``<id>`` → delete one schedule.
        * ``on``/``off`` ``<id>`` → enable/disable one schedule.
        * ``<project> <HH:MM> <prompt...>`` → add a daily schedule; the project
          is validated with the same ``_resolve_project_arg`` the other commands
          use, and the time via :func:`~voice_bridge.scheduler.parse_hhmm`.

        Any bad input replies with usage and calls no mutator. The output is
        HTML-free so a scheduled prompt's ``<``/``>``/``&`` can never break
        Telegram parsing.
        """
        msg = update.message
        if msg is None or not self._allowed(msg.from_user.id):
            return
        args = list(context.args or [])
        # A first arg that is a KNOWN project WITH the 3-arg add shape is always
        # an add — so a project literally named "list"/"on"/"remove" is never
        # shadowed by a subcommand keyword. Only when it is NOT such an add do we
        # interpret args[0] as a subcommand.
        first_is_add = (
            len(args) >= 3 and self._resolve_project_arg(args[0])[1] is None
        )
        if not first_is_add:
            if not args or args[0] == "list":
                schedules = await self.controls.list_schedules()
                await msg.reply_text(_format_schedules(schedules))
                return

            sub = args[0]
            if sub in {"remove", "rm", "del"}:
                sid = _parse_schedule_id(args[1] if len(args) > 1 else None)
                if sid is None:
                    await msg.reply_text(t("schedule.usage_remove"))
                    return
                removed = await self.controls.remove_schedule(sid)
                await msg.reply_text(
                    t("schedule.removed", id=sid) if removed else t("schedule.not_found", id=sid)
                )
                return

            if sub in {"on", "off"}:
                sid = _parse_schedule_id(args[1] if len(args) > 1 else None)
                if sid is None:
                    await msg.reply_text(t("schedule.usage_toggle", sub=sub))
                    return
                enabled = sub == "on"
                ok = await self.controls.set_schedule_enabled(sid, enabled)
                state = t("schedule.enabled") if enabled else t("schedule.disabled")
                await msg.reply_text(
                    t("schedule.toggled", id=sid, state=state) if ok
                    else t("schedule.not_found", id=sid)
                )
                return

        # Otherwise: add. Needs <project> <HH:MM> <prompt...>.
        usage_text = t("schedule.usage_add")
        if len(args) < 3:
            await msg.reply_text(usage_text)
            return
        project, error = self._resolve_project_arg(args[0])
        if error:
            await msg.reply_text(error)
            return
        hhmm = parse_hhmm(args[1])
        if hhmm is None:
            await msg.reply_text(t("schedule.bad_time", time=args[1], usage=usage_text))
            return
        prompt = " ".join(args[2:]).strip()
        if not prompt:
            await msg.reply_text(usage_text)
            return
        # If the time has already passed for today (local), seed last_run=today
        # so a morning schedule added in the afternoon first fires TOMORROW,
        # not on the next tick (~30s later).
        now = datetime.datetime.now()
        first_tomorrow = hhmm <= now.strftime("%H:%M")
        last_run = now.date().isoformat() if first_tomorrow else None
        await self.controls.add_schedule(project, hhmm, prompt, last_run=last_run)
        suffix = t("schedule.first_tomorrow") if first_tomorrow else ""
        await msg.reply_text(t("schedule.added", project=project, time=hhmm, suffix=suffix))

    async def _cmd_help(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        """Document the routing rules + command reference (owner-gated).

        Sent as PLAIN text (no ``parse_mode``): :func:`_format_help` is kept
        strictly HTML-free, so nothing in it can be mis-parsed as markup. The
        content covers what the command menu cannot — how a message is routed to
        a project and how a phone reply answers an approval / ask_user question.
        """
        msg = update.message
        if msg is None or not self._allowed(msg.from_user.id):
            return
        await msg.reply_text(_format_help())

    # --- /live: drive an already-running Claude Code session ---------------
    #
    # These sessions are NOT ours: they belong to the editor or a CLI the user
    # started, and Claude Code cannot open a second copy of one (both processes
    # would load the .jsonl at their own start and overwrite each other's tail).
    # So instead of resuming, we join: a line written to the session's unix
    # socket arrives in it as a user turn, and its own transcript is tailed for
    # the reply. While attached, plain messages go THERE instead of to a bridge
    # project (see bridge.make_inbound).

    def live_target(self):
        """The live session this chat is driving, or None."""
        return self._live_session if self._claude_live_enabled else None

    async def live_send(self, text: str, spoken: bool = False) -> bool:
        """Deliver *text* into the attached live session. False if not attached
        or the socket refused (the envelope is Claude Code's internal wire
        format, so a future release can break it without warning).

        ``spoken`` says the message arrived as a voice note, which is how the
        stream decides whether to answer out loud: away from the keyboard you
        talk and want to be talked back to, at the desk a voice note is just
        noise over the text you are already reading."""
        if not self._claude_live_enabled:
            return False
        session = self._live_session
        if session is None or not text.strip():
            return False
        self._live_spoken = spoken
        try:
            await live.send(session.socket_path, text)
            return True
        except Exception:  # noqa: BLE001 - never crash the inbound path
            logger.exception("live: send failed for pid %s", session.pid)
            await self._send_plain(t("live.unreachable", pid=session.pid))
            return False

    async def live_route(self, cwd: str, text: str, spoken: bool = False) -> bool:
        """Send *text* to the editor session already open on *cwd*, if any.

        This is what keeps a project's work in ONE place: when a session for
        that directory is already running in the editor, a Telegram message
        joins it instead of the bridge spawning a second, invisible session
        nobody is watching. Attaches on first use so replies stream back too.
        Returns False when no such session is running, and the caller falls
        back to the bridge's own session exactly as before.
        """
        if not self._claude_live_enabled or not cwd or not text.strip():
            return False
        try:
            sessions = live.list_sessions(Path.home() / ".claude" / "sessions")
        except Exception:  # noqa: BLE001 - never break routing over this
            logger.exception("live: could not list sessions")
            return False
        target = Path(cwd).resolve()
        best = None
        for session in sessions:
            try:
                path = Path(session.cwd).resolve()
            except (OSError, ValueError):
                continue
            # The session may sit in a subdirectory of the project. With
            # several open, the one worked in last is the one being watched.
            if path == target or target in path.parents:
                if best is None or session.last_active > best.last_active:
                    best = session
        if best is None:
            return False
        return await self._send_to(best, text, spoken)

    async def live_send_to(self, session_id: str, text: str, spoken: bool = False) -> bool:
        """Send *text* into the session *session_id* if it is open on this PC.

        This is how a reply goes back to the conversation that produced the
        message. False when that session is not running (closed editor), and
        the caller routes by project instead.
        """
        if not self._claude_live_enabled or not session_id or not text.strip():
            return False
        try:
            sessions = live.list_sessions(Path.home() / ".claude" / "sessions")
        except Exception:  # noqa: BLE001 - never break routing over this
            logger.exception("live: could not list sessions")
            return False
        match = next((x for x in sessions if x.session_id == session_id), None)
        return match is not None and await self._send_to(match, text, spoken)

    async def _send_to(self, session, text: str, spoken: bool) -> bool:
        """Attach to *session* (so its answer streams back) and send *text*."""
        if self._live_session is None or self._live_session.pid != session.pid:
            await self._attach_live(str(session.pid))
        sent = await self.live_send(text, spoken=spoken)
        if sent:
            await self.note_route(
                session.session_id, self.session_label(session),
                session_id=session.session_id, cwd=session.cwd,
            )
        return sent

    async def _answer_from_button(self, query, value: str) -> None:
        """Send a tapped answer ("taip", "ne", "2") to the session that asked.

        The session is the one that sent the message carrying the buttons
        (sent_log), falling back to the attached one. The question stays
        readable: only the buttons change, to say what was answered."""
        entry = sent_log.lookup(query.message.message_id)
        if entry and entry.get("s"):
            sent = await self.live_send_to(entry["s"], value)
        else:
            sent = self.live_target() is not None and await self.live_send(value)
        label = {
            t("answer.yes"): t("answer.sent_yes"), t("answer.no"): t("answer.sent_no"),
        }.get(value, t("answer.sent", answer=value)) if sent else t("answer.session_closed")
        await self._edit_callback_markup(
            query, InlineKeyboardMarkup([[InlineKeyboardButton(label, callback_data="noop:")]])
        )

    async def _buttons_for_hook_messages(self) -> None:
        """Put answer buttons under the IDE hooks' own Telegram messages.

        The hooks post "finished" notices straight to Telegram and record
        which session each came from. When that session's turn ended on a
        yes/no question or numbered options, the same bot adds the buttons
        afterwards, so the hooks stay plain curl. Runs until cancelled."""
        seen_until = time.time()
        while True:
            await asyncio.sleep(2)
            try:
                seen_until = await self._add_hook_buttons(seen_until)
            except asyncio.CancelledError:
                raise
            except Exception:  # noqa: BLE001 - a missed button is not worth dying for
                logger.exception("hook buttons: pass failed")

    async def _add_hook_buttons(self, seen_until: float, root: Path | None = None) -> float:
        """One pass of :meth:`_buttons_for_hook_messages`; returns the new mark."""
        root = root or Path.home() / ".claude" / "projects"
        entries = [
            e for e in sent_log._tail(None)
            if (e.get("t") or 0) > seen_until and e.get("s")
            and e.get("m") not in self._own_sent
        ]
        for entry in sorted(entries, key=lambda e: e["t"]):
            seen_until = entry["t"]
            text = live.last_assistant_text(live.transcript_of(root, entry["s"]))
            markup = _answer_markup(text)
            if markup is not None and self.app is not None:
                await self.app.bot.edit_message_reply_markup(
                    chat_id=self._chat_id, message_id=entry["m"], reply_markup=markup
                )
        return seen_until

    def session_label(self, session) -> str:
        """``Project · conversation title`` for a live session."""
        project = self.project_for_cwd(session.cwd)
        row = _find_project_row(self.controls.snapshot(), project) if project else None
        name = (row or {}).get("display_name") or project or Path(session.cwd).name
        title = live.title_of(Path.home() / ".claude" / "projects", session.session_id)
        if title and len(title) > 40:
            title = title[:40] + "…"
        return f"{name} · {title}" if title else name

    async def note_route(
        self, key: str, label: str, session_id: str | None = None, cwd: str = ""
    ) -> None:
        """Say where a message went -- once per change of destination.

        Every message confirmed would be noise; never confirming left the user
        guessing which project a message reached. Saying it when the target
        changes answers exactly that."""
        if key == self._last_route:
            return
        self._last_route = key
        message = await self._send_plain(f"➡️ {label}")
        # A reply to the notice itself must reach the same place.
        await self._remember_sent(message, cwd, session_id)

    def _open_projects(self) -> set[str]:
        """Projects that have a session open in the editor right now."""
        if not self._claude_live_enabled:
            return set()
        try:
            sessions = live.list_sessions(Path.home() / ".claude" / "sessions")
        except Exception:  # noqa: BLE001 - a listing must not break /projects
            return set()
        return {p for p in (self.project_for_cwd(x.cwd) for x in sessions) if p}

    def project_for_cwd(self, cwd: str) -> str | None:
        """The bridge project whose directory contains *cwd*, if any."""
        if not cwd:
            return None
        try:
            target = Path(cwd).resolve()
        except (OSError, ValueError):
            return None
        best = None
        for row in self.controls.snapshot():
            raw = row.get("cwd") or ""
            if not raw:
                continue
            try:
                path = Path(raw).resolve()
            except (OSError, ValueError):
                continue
            if path == target or path in target.parents:
                if best is None or len(str(path)) > len(str(best[1])):
                    best = (row["project"], path)
        return best[0] if best else None

    async def _remember_sent(
        self, message, cwd: str, session_id: str | None = None
    ) -> None:
        """Map a message we sent to its project and session; never raises."""
        mid = getattr(message, "message_id", None)
        if message is None or mid is None:
            return
        self._own_sent.add(mid)
        try:
            sent_log.record(mid, session_id, cwd)
        except (OSError, TypeError, ValueError):
            logger.exception("could not record message %s", mid)
        if self._on_sent is None:
            return
        project = self.project_for_cwd(cwd)
        if project is None:
            return
        try:
            await self._on_sent(mid, project)
        except Exception:  # noqa: BLE001 - routing memory is best-effort
            logger.exception("could not map message %s to %s", mid, project)

    async def send_notice(self, text: str) -> None:
        """A plain notice from the bridge itself (e.g. a limit running out)."""
        await self._send_plain(text)

    async def _send_plain(self, text: str, reply_markup=None):
        """Plain message to the owner (optionally with buttons); never raises.

        Returns the sent Message so the caller can edit it later, or None when
        the send failed."""
        try:
            return await _send_with_retry(
                lambda: self.app.bot.send_message(
                    chat_id=self._chat_id, text=text, reply_markup=reply_markup
                )
            )
        except TelegramError:
            logger.exception("live: could not send notice")
            return None

    async def _send_plain_voice(self, voice_bytes: bytes):
        """Voice note to the owner, outside the project-scoped path; never raises.

        Mirrors :meth:`_send_plain`: the /live stream has no project, so it
        cannot use ``send_update``'s caption."""
        try:
            return await _send_with_retry(
                lambda: self.app.bot.send_voice(
                    chat_id=self._chat_id, voice=voice_bytes
                )
            )
        except TelegramError:
            logger.exception("live: could not send voice")
            return None

    async def _cmd_live(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        """Attach to a running Claude Code session, or `/live off` to detach."""
        msg = update.message
        if msg is None or not self._allowed(msg.from_user.id):
            return
        if not self._claude_live_enabled:
            await msg.reply_text(t("codex.no_live"))
            return
        args = list(context.args or [])
        if args and args[0] in {"off", "stop", "detach"}:
            self._detach_live()
            await msg.reply_text(t("live.detached"))
            return

        sessions = live.list_sessions(Path.home() / ".claude" / "sessions")
        if not sessions:
            await msg.reply_text(t("live.none"))
            return
        root = Path.home() / ".claude" / "projects"
        rows = []
        for s in sessions:
            title = live.title_of(root, s.session_id) or t("live.untitled")
            rows.append([InlineKeyboardButton(
                f"{title[:40]} · {Path(s.cwd).name}", callback_data=f"live:{s.pid}"
            )])
        await msg.reply_text(
            t("live.pick"),
            reply_markup=InlineKeyboardMarkup(rows),
        )

    async def _attach_live(self, pid_str: str) -> str:
        """Attach to the session with this pid; returns the reply text."""
        if not self._claude_live_enabled:
            return t("codex.no_live")
        try:
            pid = int(pid_str)
        except (TypeError, ValueError):
            return t("live.bad_id")
        session = live.find(pid, Path.home() / ".claude" / "sessions")
        if session is None:
            return t("live.gone", pid=pid_str)
        self._detach_live()
        self._live_session = session
        self._write_live_marker(session.session_id)
        self._live_task = asyncio.create_task(self._tail_live(session))
        title = live.title_of(Path.home() / ".claude" / "projects", session.session_id)
        return t("live.attached", title=title or session.cwd)

    @staticmethod
    def live_marker() -> Path:
        """File naming the session we are streaming, read by the Stop hook.

        While we tail a session, its output already reaches Telegram, so the
        editor's "finished" notification would be the same text twice. The hook
        reads this and stays quiet for exactly that session."""
        return Path.home() / ".claude" / ".voice-bridge-live"

    def _write_live_marker(self, session_id: str | None) -> None:
        """Publish (or clear) the attached session id; never raises."""
        try:
            marker = self.live_marker()
            if session_id:
                marker.parent.mkdir(parents=True, exist_ok=True)
                marker.write_text(session_id)
            elif marker.exists():
                marker.unlink()
        except OSError:
            logger.exception("live: could not update the marker file")

    def _detach_live(self) -> None:
        """Stop tailing and forget the attachment (idempotent)."""
        task, self._live_task = self._live_task, None
        self._live_session = None
        self._write_live_marker(None)
        if task is not None:
            task.cancel()

    async def _tail_live(self, session) -> None:
        """Tail the attached session's transcript back into Telegram.

        Nothing returns over the socket, so the session's own .jsonl is the
        reply channel. Starts at the CURRENT end so attaching does not replay
        the whole history."""
        root = Path.home() / ".claude" / "projects"
        path = live.transcript_of(root, session.session_id)
        offset = live.end_of(path)
        try:
            while True:
                await asyncio.sleep(1.5)
                if self._live_session is not session:
                    return
                if path is None:
                    path = live.transcript_of(root, session.session_id)
                    offset = live.end_of(path)
                    continue
                try:
                    # Already rendered to Telegram lines by read_new.
                    start = offset
                    lines, offset = live.read_new(path, offset)
                    # Typed at the keyboard means sitting at the screen: stop
                    # reading answers aloud. Only a Telegram message can turn
                    # it back on (live_send), so the voice follows wherever the
                    # user last actually spoke from -- not just the phone.
                    if live.typed_here(path, start, offset):
                        self._live_spoken = False
                except Exception:  # noqa: BLE001 - a bad read must not kill the tail
                    logger.exception("live: transcript read failed")
                    continue
                if not lines:
                    continue
                # One message per poll, not per line: a busy session emits a
                # tool line every second or two, and separate messages are a
                # wall of notifications with the real answer buried in it.
                body = "\n".join(lines)
                # A plain-text question with numbered options becomes real
                # buttons: tapping sends the NUMBER back down the socket, the
                # same thing the user would have typed. (An AskUserQuestion
                # picker is NOT answerable this way — it is answered in the
                # editor that raised it — so live.render says exactly that
                # instead, and parse_options finds nothing there to tap.)
                markup = _answer_markup(body)
                # Every message says which conversation it is from: with more
                # than one session working, an unlabeled "🔧 Bash ..." could be
                # any of them.
                chunks = _chunk_text(f"💬 {self.session_label(session)}\n{body}")
                for i, chunk in enumerate(chunks):
                    # Buttons ride the LAST chunk, right under the options.
                    message = await self._send_plain(
                        chunk, markup if i == len(chunks) - 1 else None
                    )
                    await self._remember_sent(message, session.cwd, session.session_id)
                # Text first, voice after: the text is the record, the voice is
                # for when you are away from the screen. Only the assistant's
                # own words are spoken -- see live.spoken_of.
                await self._speak_live(body, session.cwd)
        except asyncio.CancelledError:
            raise
        except Exception:  # noqa: BLE001 - the tail must never take the bot down
            logger.exception("live: tail loop stopped for pid %s", session.pid)

    async def _speak_live(self, body: str, cwd: str) -> None:
        """Read the speakable part of a /live body aloud; never raises.

        No speak hook, nothing worth speaking, or a TTS failure all degrade to
        the text that was already sent -- the tail must survive either way."""
        if self._on_speak is None or not self._live_spoken:
            return
        try:
            spoken = live.spoken_of(body)
            if not spoken:
                return
            voice_bytes = await self._on_speak(spoken)
            if voice_bytes is None:
                return
            message = await self._send_plain_voice(voice_bytes)
            await self._remember_sent(message, cwd)
        except Exception:  # noqa: BLE001 - voice is a nicety, the text already went
            logger.exception("live: could not voice the stream line")

    # --- editor permission prompts, answerable from here ------------------

    @staticmethod
    def alive_marker() -> Path:
        """Heartbeat the editor-side gate checks before it blocks anything.

        Without it a gate would stall a tool call for its full timeout whenever
        the bridge happens to be down — the worst possible failure for someone
        sitting at the keyboard. Refreshed every watcher tick; the gate treats
        a stale or missing file as "bridge not running, do not interfere".
        """
        return Path.home() / ".claude" / ".voice-bridge-alive"

    @staticmethod
    def perm_dir() -> Path:
        """Where the IDE hook drops permission requests and reads answers."""
        return Path.home() / ".claude" / ".voice-bridge-perm"

    async def _watch_permissions(self) -> None:
        """Relay editor permission prompts here with ✅/❌ buttons.

        A `PermissionRequest` hook in the editor session writes `<id>.req.json`
        and then BLOCKS, polling for `<id>.ans`. Whatever we write there becomes
        its decision. Never answering is safe: the hook times out and the editor
        asks the user itself, exactly as before this existed.
        """
        directory = self.perm_dir()
        while True:
            try:
                await asyncio.sleep(1.0)
                try:
                    marker = self.alive_marker()
                    marker.parent.mkdir(parents=True, exist_ok=True)
                    marker.write_text(str(int(time.time())))
                except OSError:
                    logger.exception("perm: could not refresh the heartbeat")
                try:
                    requests = sorted(directory.glob("*.req.json"))
                except OSError:
                    continue
                for path in requests:
                    ident = path.name[: -len(".req.json")]
                    if ident in self._perm_seen:
                        continue
                    try:
                        data = json.loads(path.read_text())
                    except (OSError, ValueError):
                        self._perm_seen.add(ident)
                        continue
                    self._perm_seen.add(ident)
                    await self._ask_permission(ident, data)
                await self._expire_permissions(directory)
            except asyncio.CancelledError:
                raise
            except Exception:  # noqa: BLE001 - the watcher must never die
                logger.exception("perm: watcher iteration failed")

    async def _ask_permission(self, ident: str, data: dict) -> None:
        project = str(data.get("project") or "IDE")
        tool = str(data.get("tool") or "?")
        detail = str(data.get("detail") or "").strip()
        body = t("approval.editor_prompt", project=project, tool=tool)
        if detail:
            body += f"\n{_truncate_approval_preview(detail)}"
        markup = InlineKeyboardMarkup([[
            InlineKeyboardButton(t("approval.allow"), callback_data=f"perm:{ident}:1"),
            InlineKeyboardButton(t("approval.deny"), callback_data=f"perm:{ident}:0"),
        ]])
        message = await self._send_plain(body, markup)
        if message is not None:
            self._perm_pending[ident] = message
            await self._remember_sent(
                message, str(data.get("cwd") or ""), data.get("session_id")
            )

    async def _expire_permissions(self, directory: Path) -> None:
        """Mark buttons dead once the editor session stops waiting.

        The hook deletes its request file when it gives up (its timeout is
        shorter than a phone is patient). Leaving the buttons live after that
        is the worst outcome: the tap looks accepted while nothing is listening
        and the editor has already raised its own dialog. Say so instead.
        """
        for ident, message in list(self._perm_pending.items()):
            if (directory / f"{ident}.req.json").exists():
                continue
            self._perm_pending.pop(ident, None)
            try:
                await message.edit_text(t("approval.too_late"))
            except Exception:  # noqa: BLE001 - a failed edit must not stop the watcher
                logger.exception("perm: could not mark %s expired", ident)

    def answer_permission(self, ident: str, allow: bool) -> bool:
        """Write the decision the blocked editor hook is waiting for."""
        # Only ever name a file we generated an id for; never interpolate a
        # callback payload into a path without this guard.
        if not ident or not _SAFE_ID_RE.fullmatch(ident):
            return False
        try:
            directory = self.perm_dir()
            # The hook deletes its request the moment it gives up. Writing an
            # answer nobody is waiting for would report success while the editor
            # dialog stays open — exactly the confusing case. Check first.
            if not (directory / f"{ident}.req.json").exists():
                self._perm_pending.pop(ident, None)
                return False
            directory.mkdir(parents=True, exist_ok=True)
            (directory / f"{ident}.ans").write_text("allow" if allow else "deny")
            self._perm_pending.pop(ident, None)   # answered, never expire it
            return True
        except OSError:
            logger.exception("perm: could not write the answer for %s", ident)
            return False

    async def _cmd_handoff(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        msg = update.message
        if msg is None or not self._allowed(msg.from_user.id):
            return
        # parse_mode=HTML matches the /panel Handoff button edit; the content
        # is html.escape'd inside _format_handoff_text so this never raises.
        await msg.reply_text(
            self._format_handoff_text(context.args[0] if context.args else ""),
            parse_mode="HTML",
        )

    def _format_handoff_text(self, project: str) -> str:
        # This text is rendered with parse_mode='HTML' (the /panel Handoff
        # button edits that way). Transcripts routinely contain <, >, & from
        # code, so every dynamic value below is html.escape'd; a raw '<' would
        # make Telegram reject the edit and the button would silently do
        # nothing. The static structure has no HTML metacharacters.
        row = _find_project_row(self.controls.snapshot(), project)
        if row is None:
            return t("handoff.not_found")
        path = transcript_path(row.get("cwd") or "")
        label = html.escape(row.get("display_name") or row["project"])
        if not path.exists():
            return t("handoff.none", project=label)
        text = path.read_text(encoding="utf-8", errors="replace").strip()
        if not text:
            return t("handoff.empty", project=label)
        tail = html.escape(_tail_for_telegram(text))
        friendly = html.escape(_friendly_path(str(path)))
        return t("handoff.text", project=label, path=friendly, tail=tail)

    # --- lifecycle -------------------------------------------------------
    async def run(self) -> None:
        """Build the Application, register handlers, start polling, RETURN.

        Per C3 the bridge main() owns the run-forever wait; this method must
        not block. ``stop()`` performs the symmetric shutdown.
        """
        app = Application.builder().token(self.cfg.telegram_bot_token).build()
        self.app = app

        only_me = filters.User(user_id=self.cfg.telegram_allowed_user_id)

        app.add_handler(
            CommandHandler("menu", self._cmd_menu, filters=only_me))
        app.add_handler(
            CommandHandler("panel", self._cmd_panel, filters=only_me))
        app.add_handler(
            CommandHandler("projects", self._cmd_projects, filters=only_me))
        app.add_handler(
            CommandHandler("projects_all", self._cmd_projects_all, filters=only_me))
        app.add_handler(
            CommandHandler("projects_refresh", self._cmd_projects_refresh, filters=only_me))
        app.add_handler(
            CommandHandler("newproject", self._cmd_newproject, filters=only_me))
        app.add_handler(
            CommandHandler("handoff", self._cmd_handoff, filters=only_me))
        app.add_handler(
            CommandHandler("on", self._cmd_on, filters=only_me))
        app.add_handler(
            CommandHandler("off", self._cmd_off, filters=only_me))
        app.add_handler(
            CommandHandler("stop", self._cmd_stop, filters=only_me))
        app.add_handler(
            CommandHandler("mode", self._cmd_mode, filters=only_me))
        app.add_handler(
            CommandHandler("effort", self._cmd_effort, filters=only_me))
        app.add_handler(
            CommandHandler("info", self._cmd_info, filters=only_me))
        app.add_handler(
            CommandHandler("voice", self._cmd_voice, filters=only_me))
        app.add_handler(
            CommandHandler("verbose", self._cmd_verbose, filters=only_me))
        app.add_handler(
            CommandHandler("engine", self._cmd_engine, filters=only_me))
        app.add_handler(
            CommandHandler("agent", self._cmd_agent, filters=only_me))
        app.add_handler(
            CommandHandler("pc", self._cmd_pc, filters=only_me))
        app.add_handler(
            CommandHandler("status", self._cmd_status, filters=only_me))
        app.add_handler(
            CommandHandler("recap", self._cmd_recap, filters=only_me))
        app.add_handler(
            CommandHandler(["usage", "cost"], self._cmd_cost, filters=only_me))
        app.add_handler(
            CommandHandler("policies", self._cmd_policies, filters=only_me))
        app.add_handler(
            CommandHandler("schedule", self._cmd_schedule, filters=only_me))
        app.add_handler(
            CommandHandler("help", self._cmd_help, filters=only_me))
        app.add_handler(
            CommandHandler("live", self._cmd_live, filters=only_me))
        app.add_handler(CallbackQueryHandler(self._handle_callback))
        app.add_handler(MessageHandler(
            only_me & filters.VOICE, self._handle_voice))
        app.add_handler(MessageHandler(
            only_me
            & (
                filters.PHOTO
                | filters.Document.ALL
                | filters.AUDIO
                | filters.VIDEO
                | filters.VIDEO_NOTE
            ),
            self._handle_attachment,
        ))
        app.add_handler(MessageHandler(
            only_me & filters.TEXT & ~filters.COMMAND, self._handle_text))

        await app.initialize()
        commands = (
            bot_commands()
            if self._claude_live_enabled
            else [command for command in bot_commands() if command.command != "live"]
        )
        await app.bot.set_my_commands(commands)
        await app.start()
        await app.updater.start_polling()
        if self._claude_live_enabled:
            self._perm_task = asyncio.create_task(self._watch_permissions())
            self._hook_buttons_task = asyncio.create_task(self._buttons_for_hook_messages())
            await self._restore_live()
        else:
            self._detach_live()
            try:
                self.alive_marker().unlink(missing_ok=True)
            except OSError:
                logger.exception("could not clear disabled Claude relay heartbeat")

    async def _restore_live(self) -> None:
        """Re-attach to the session we were tailing before a restart.

        The attachment lives in memory, so a bridge restart drops it -- but the
        marker file stays, and the Stop hook reads that marker to stay quiet for
        the streamed session. Left alone, the two together are a silent hole:
        nothing tails the session AND its finish notifications are suppressed,
        so someone on their phone simply stops hearing from it with no error
        anywhere. Re-attach if that session is still up, clear the marker if it
        is not. Never raises: a failure here must not stop the bot starting."""
        if not self._claude_live_enabled:
            self._detach_live()
            return
        try:
            marker = self.live_marker()
            session_id = marker.read_text().strip() if marker.exists() else ""
            if not session_id:
                return
            for session in live.list_sessions():
                if session.session_id == session_id:
                    await self._attach_live(str(session.pid))
                    return
            marker.unlink(missing_ok=True)
        except Exception:  # noqa: BLE001 - startup must survive a bad marker
            logger.exception("live: could not restore the attachment")

    async def stop(self) -> None:
        """Stop polling and shut the Application down (idempotent)."""
        for name in ("_perm_task", "_hook_buttons_task"):
            task = getattr(self, name)
            setattr(self, name, None)
            if task is not None:
                task.cancel()
        app = self.app
        if app is None:
            return
        updater = getattr(app, "updater", None)
        if updater is not None and getattr(updater, "running", False):
            await updater.stop()
        if getattr(app, "running", False):
            await app.stop()
        await app.shutdown()


async def _download_attachment(msg) -> dict | None:
    kind = "file"
    file_name = ""
    mime_type = None
    source = None

    if getattr(msg, "photo", None):
        kind = "photo"
        source = msg.photo[-1]
        file_name = "photo.jpg"
    elif getattr(msg, "document", None) is not None:
        doc = msg.document
        kind = "document"
        source = doc
        file_name = doc.file_name or "document.bin"
        mime_type = getattr(doc, "mime_type", None)
    elif getattr(msg, "audio", None) is not None:
        audio = msg.audio
        kind = "audio"
        source = audio
        file_name = audio.file_name or "audio.bin"
        mime_type = getattr(audio, "mime_type", None)
    elif getattr(msg, "video", None) is not None:
        video = msg.video
        kind = "video"
        source = video
        file_name = video.file_name or "video.mp4"
        mime_type = getattr(video, "mime_type", None)
    elif getattr(msg, "video_note", None) is not None:
        kind = "video_note"
        source = msg.video_note
        file_name = "video_note.mp4"

    if source is None:
        return None
    tg_file = await source.get_file()
    data = bytes(await tg_file.download_as_bytearray())
    return {
        "kind": kind,
        "file_name": _clean_telegram_filename(file_name),
        "mime_type": mime_type,
        "data": data,
    }


def _clean_telegram_filename(name: str) -> str:
    return Path(name or "file.bin").name or "file.bin"
