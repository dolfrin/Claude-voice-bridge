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
import logging
import random
from pathlib import Path
from typing import Awaitable, Callable, Protocol, TypeVar

from telegram import (
    InlineKeyboardButton,
    InlineKeyboardMarkup,
    Update,
)
from telegram.error import BadRequest, NetworkError, RetryAfter, TimedOut
from telegram.ext import (
    Application,
    CallbackQueryHandler,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    filters,
)

from .config import Config
from .transcript import transcript_path
from .tts import available_voices

# Pure view/format/parse helpers were extracted into telegram_views for
# maintainability. Re-export them here so existing references
# (telegram_io.build_panel_markup, .parse_callback, ...) and the test import
# surface stay identical after the split. telegram_views must NOT import this
# module (would create a cycle).
from .telegram_views import (
    _BOT_COMMANDS,
    _EFFORTS,
    _ENGINES,
    _MODES,
    _clean_choices,
    _find_project_row,
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
    async def cost_summary(self) -> str: ...


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


class TelegramIO:
    def __init__(
        self,
        cfg: Config,
        on_user_message: Callable[[dict], Awaitable[None]],
        controls: Controls,
        on_approval: Callable[[int, bool], bool] | None = None,
    ) -> None:
        self.cfg = cfg
        self.on_user_message = on_user_message
        self.controls = controls
        # Resolver for inline Allow/Deny taps: returns True if a live pending
        # approval was resolved, False if it was already answered / timed out.
        # Wired in bridge to ApprovalManager.resolve_token.
        self._on_approval = on_approval
        self.app: Application | None = None
        self._pending_off_sends: dict[str, tuple[str, str]] = {}
        self._pending_off_seq = 0
        self._pending_asks: dict[str, tuple[asyncio.Future[str], list[str]]] = {}
        self._pending_ask_seq = 0

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
        tg_file = await msg.voice.get_file()
        audio = bytes(await tg_file.download_as_bytearray())
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
        attachment = await _download_attachment(msg)
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
            reply_markup = InlineKeyboardMarkup([[
                InlineKeyboardButton(
                    "✅ Leisti", callback_data=f"apv:{approval_token}:1"),
                InlineKeyboardButton(
                    "❌ Neleisti", callback_data=f"apv:{approval_token}:0"),
            ]])
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
        clean_choices = _clean_choices(choices)
        if not clean_choices:
            clean_choices = ["Yes", "No"]
        self._pending_ask_seq += 1
        token = str(self._pending_ask_seq)
        loop = asyncio.get_running_loop()
        future: asyncio.Future[str] = loop.create_future()
        self._pending_asks[token] = (future, clean_choices)
        rows = [
            [InlineKeyboardButton(choice, callback_data=f"ask:{token}:{idx}")]
            for idx, choice in enumerate(clean_choices)
        ]
        await self.app.bot.send_message(
            chat_id=self._chat_id,
            text=f"[{project}] {question}",
            reply_markup=InlineKeyboardMarkup(rows),
        )
        try:
            return await asyncio.wait_for(future, timeout=self.cfg.approval_timeout)
        except asyncio.TimeoutError:
            return ""
        finally:
            self._pending_asks.pop(token, None)

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

    async def send_disabled_project_prompt(self, project: str, text: str) -> int:
        """Ask whether to enable a disabled project and send the pending turn."""
        self._pending_off_seq += 1
        token = str(self._pending_off_seq)
        self._pending_off_sends[token] = (project, text)
        markup = InlineKeyboardMarkup([
            [
                InlineKeyboardButton(
                    "Enable and send", callback_data=f"offsend:{token}"
                )
            ],
            [InlineKeyboardButton("Cancel", callback_data=f"offcancel:{token}")],
        ])
        msg = await self.app.bot.send_message(
            chat_id=self._chat_id,
            text=(
                f"[bridge] {project} is disabled.\n"
                "Enable the project and send the last message?"
            ),
            reply_markup=markup,
        )
        return msg.message_id

    # --- /panel + callbacks ---------------------------------------------
    async def _cmd_panel(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        msg = update.message
        if msg is None or not self._allowed(msg.from_user.id):
            return
        markup = build_panel_markup(self.controls.snapshot())
        await msg.reply_text("Control panel", reply_markup=markup)

    async def _cmd_menu(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        msg = update.message
        if msg is None or not self._allowed(msg.from_user.id):
            return
        await msg.reply_text("🏠 Alex for Claude", reply_markup=build_menu_markup())

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
                await query.edit_message_text("This request has expired.")
                return
            project, text = pending
            if action == "offcancel":
                await query.edit_message_text(f"Cancelled: {project}")
                return
            await self.controls.enable_and_deliver(project, text)
            await query.edit_message_text(f"Enabled and sent to {project}.")
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
                await query.edit_message_text("This choice has expired.")
                return
            future, choices = pending
            if idx < 0 or idx >= len(choices):
                return
            choice = choices[idx]
            if not future.done():
                future.set_result(choice)
            await query.edit_message_text(f"Selected: {choice}")
            return
        if action == "cost":
            # Info action: reply with a fresh message, do not touch the panel.
            await query.message.reply_text(await self.controls.cost_summary())
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
                    format_projects(snap),
                    build_projects_list_markup(snap),
                )
                return
            elif action == "ptgl":
                turning_off = row["enabled"]
                await self.controls.toggle(project, not row["enabled"])
                snap = self.controls.snapshot()
                text = format_projects(snap)
                if turning_off:
                    # Disabling drops this project's queued turns; note it
                    # instead of a silent redraw (audit finding #2).
                    text = f"{project} off — laukusios užduotys atmestos.\n\n" + text
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
        """Resolve an inline Allow/Deny tap by approval token.

        ``index_str`` is ``"{token}:{approved}"`` (``1`` = allow, ``0`` = deny).
        A token with no live pending (already answered via quote-reply, or timed
        out) answers with a "no longer relevant" toast and leaves the message
        untouched. A resolved tap edits the message to show the outcome and
        removes the buttons.
        """
        try:
            token_str, approved_str = index_str.split(":", 1)
            token = int(token_str)
        except (ValueError, TypeError):
            await self._answer_quietly(query)
            return
        approved = approved_str == "1"
        resolved = (
            self._on_approval(token, approved)
            if self._on_approval is not None
            else False
        )
        if not resolved:
            await self._answer_quietly(query, "nebeaktualu")
            return
        await self._answer_quietly(query)
        label = "✅ Leista" if approved else "❌ Neleista"
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
                format_projects(snapshot),
                build_projects_list_markup(snapshot),
            )
        elif action == "projects_all":
            await self._edit_callback_text(
                query,
                format_projects(snapshot, show_all=True),
                build_projects_list_markup(snapshot, show_all=True),
            )
        elif action == "panel":
            await self._edit_callback_text(query, "Control panel", build_panel_markup(snapshot))
        elif action == "refresh":
            added = await self.controls.refresh_projects()
            snapshot = self.controls.snapshot()
            await self._edit_callback_text(
                query,
                f"New projects added: {added}\n\n"
                + format_projects(snapshot, show_all=True),
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
            return None, f"Nežinomas projektas: {name}. Yra: {', '.join(known)}"
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
            format_projects(snapshot, show_all=show_all),
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
            format_projects(snapshot, show_all=True),
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
            f"New projects added: {added}\n\n"
            + format_projects(snapshot, show_all=True),
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
            await msg.reply_text("Naudojimas: /newproject <vardas>")
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
        await msg.reply_text(f"{project or 'all'} on")

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
        await msg.reply_text(
            f"{project or 'all'} off — laukusios užduotys atmestos."
        )

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
            await msg.reply_text("usage: /mode <full|safe|ask> [project]")
            return
        mode = context.args[0]
        arg = context.args[1] if len(context.args) > 1 else None
        project, error = self._resolve_project_arg(arg)
        if error:
            await msg.reply_text(error)
            return
        await self.controls.set_mode(project, mode)
        await msg.reply_text(f"mode {mode} for {project or 'all'}")

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
            await msg.reply_text(
                "usage: /effort <" + "|".join(_EFFORTS) + "> [project]"
            )
            return
        level = context.args[0]
        arg = context.args[1] if len(context.args) > 1 else None
        project, error = self._resolve_project_arg(arg)
        if error:
            await msg.reply_text(error)
            return
        await self.controls.set_effort(project, level)
        await msg.reply_text(f"effort {level} for {project or 'all'}")

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
        await msg.reply_text(f"verbose {state} for {project or 'all'}")

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
            await msg.reply_text("voices: " + ", ".join(available_voices(engine)))
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
            await msg.reply_text(
                f"Nežinomas balsas: {voice}. Yra: " + ", ".join(valid_voices)
            )
            return
        await self.controls.set_voice(project, voice)
        await msg.reply_text(f"voice {voice} for {project or 'all'}")

    async def _cmd_engine(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        msg = update.message
        if msg is None or not self._allowed(msg.from_user.id):
            return
        if not context.args or context.args[0] not in _ENGINES:
            await msg.reply_text("usage: /engine <auto|openai|piper|together>")
            return
        name = context.args[0]
        await self.controls.set_engine(name)
        await msg.reply_text(f"engine {name}")

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
        """B3c: per-project + total token/cost usage summary."""
        msg = update.message
        if msg is None or not self._allowed(msg.from_user.id):
            return
        await msg.reply_text(await self.controls.cost_summary())

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
            return "Project not found. Use /projects_all."
        path = transcript_path(row.get("cwd") or "")
        label = html.escape(row.get("display_name") or row["project"])
        if not path.exists():
            return f"{label}: no handoff history yet."
        text = path.read_text(encoding="utf-8", errors="replace").strip()
        if not text:
            return f"{label}: handoff history is empty."
        tail = html.escape(_tail_for_telegram(text))
        friendly = html.escape(_friendly_path(str(path)))
        return f"{label} handoff\n{friendly}\n\n{tail}"

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
            CommandHandler("status", self._cmd_status, filters=only_me))
        app.add_handler(
            CommandHandler("recap", self._cmd_recap, filters=only_me))
        app.add_handler(
            CommandHandler("cost", self._cmd_cost, filters=only_me))
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
        await app.bot.set_my_commands(_BOT_COMMANDS)
        await app.start()
        await app.updater.start_polling()

    async def stop(self) -> None:
        """Stop polling and shut the Application down (idempotent)."""
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
