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
``{"project", "enabled", "mode", "voice", "engine", "last_active"}``.
C3: ``run()`` starts polling and RETURNS; ``bridge.main()`` owns the
run-forever wait. ``stop()`` shuts the Application down.
"""

from __future__ import annotations

import asyncio
import html
from pathlib import Path
from typing import Awaitable, Callable, Protocol

from telegram import (
    BotCommand,
    InlineKeyboardButton,
    InlineKeyboardMarkup,
    Update,
)
from telegram.error import BadRequest
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


class Controls(Protocol):
    """State surface implemented by bridge.py (Task 10).

    ``snapshot`` is synchronous; the mutators are coroutines. ``project=None``
    means "all projects".
    """

    def snapshot(self) -> list[dict]:
        # each dict keyed EXACTLY:
        # {"project": str, "enabled": bool, "mode": str, "voice": str,
        #  "engine": str, "last_active": bool}
        ...

    async def toggle(self, project: str | None, on: bool) -> None: ...
    async def select(self, project: str) -> None: ...
    async def enable_and_deliver(self, project: str, text: str) -> None: ...
    async def refresh_projects(self) -> int: ...
    async def set_mode(self, project: str | None, mode: str) -> None: ...
    async def set_voice(self, project: str | None, voice: str) -> None: ...
    async def set_engine(self, name: str) -> None: ...
    async def interrupt(self, project: str | None) -> str: ...


_MODES = ["safe", "full", "ask"]
_ENGINES = ["auto", "openai", "piper", "together"]
_PHOTO_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}
_AUDIO_SUFFIXES = {".mp3", ".m4a", ".ogg", ".opus", ".wav", ".flac"}
_VIDEO_SUFFIXES = {".mp4", ".mov", ".mkv", ".webm"}
_BOT_COMMANDS = [
    BotCommand("menu", "🏠 Pagrindinis meniu"),
    BotCommand("panel", "🎛 Valdymo panelė"),
    BotCommand("projects", "🟢 Aktyvūs projektai"),
    BotCommand("projects_all", "📚 Visi projektai"),
    BotCommand("projects_refresh", "🔎 Ieškoti naujų projektų"),
    BotCommand("handoff", "🧾 Paskutinė projekto istorija"),
    BotCommand("status", "📡 Paklausti projekto statuso"),
    BotCommand("on", "▶️ Įjungti projektą arba visus"),
    BotCommand("off", "⏸ Išjungti projektą arba visus"),
    BotCommand("stop", "⛔ Nutraukti einamą darbą"),
    BotCommand("mode", "🛡 Keisti safe/full/ask režimą"),
    BotCommand("voice", "🔊 Balsai ir TTS balsas"),
    BotCommand("engine", "🧠 Keisti TTS variklį"),
]


def _next(seq: list[str], current: str) -> str:
    """Return the element after ``current`` in ``seq``, wrapping around."""
    try:
        i = seq.index(current)
    except ValueError:
        return seq[0]
    return seq[(i + 1) % len(seq)]


def parse_callback(data: str) -> tuple[str, str]:
    """Decode ``"<action>:<index_or_empty>"`` callback data.

    Returns ``(action, index_str)`` where ``index_str`` is the project index
    (as a string) for per-project actions, or ``""`` for global actions.
    Global actions: ``allon``, ``alloff``, ``engine``.
    Per-project actions: ``tog``, ``sel``, ``ptgl``, ``mode``, ``voice``, ``noop``.
    """
    parts = data.split(":", 1)
    action = parts[0]
    index_str = parts[1] if len(parts) > 1 else ""
    return action, index_str


def format_projects(snapshot: list[dict], show_all: bool = False) -> str:
    """Render /projects as a scannable HTML summary."""
    rows = _project_list_rows(snapshot, show_all=show_all)
    if not rows:
        return "no active projects\nUse /projects_all to show every project."

    lines: list[str] = []
    for _idx, row in rows:
        status = "\U0001F7E2" if row["enabled"] else "\u26AA"
        active = " \u2B50" if row.get("last_active") else ""
        project = html.escape(row.get("display_name") or row["project"])
        cwd = _friendly_path(row.get("cwd") or "")
        path_part = html.escape(cwd) if cwd else "-"
        settings = html.escape(
            f"{row['mode']} · {row['voice']} · {row['engine']}"
        )
        lines.extend([
            f"{status} <b>{project}</b>{active}",
            f"  \U0001F4C1 {path_part} · {settings}",
            "",
        ])
    return "\n".join(lines).strip()


def build_projects_list_markup(
    snapshot: list[dict], show_all: bool = False
) -> InlineKeyboardMarkup:
    """Project picker with separate select-target and on/off controls."""
    rows: list[list[InlineKeyboardButton]] = []
    for idx, row in _project_list_rows(snapshot, show_all=show_all):
        status = "\U0001F7E2" if row["enabled"] else "\u26AA"
        active = " \u2B50" if row.get("last_active") else ""
        name = row.get("display_name") or row["project"]
        toggle_label = "ON" if row["enabled"] else "OFF"
        rows.append([
            InlineKeyboardButton(
                f"\u270D {status} {name}{active}",
                callback_data=f"sel:{idx}",
            ),
            InlineKeyboardButton(toggle_label, callback_data=f"ptgl:{idx}"),
        ])
    return InlineKeyboardMarkup(rows)


def build_menu_markup() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup([
        [
            InlineKeyboardButton("🟢 Aktyvūs", callback_data="menu:projects"),
            InlineKeyboardButton("📚 Visi", callback_data="menu:projects_all"),
        ],
        [
            InlineKeyboardButton("🎛 Panelė", callback_data="menu:panel"),
            InlineKeyboardButton("🧾 Handoff", callback_data="menu:handoff"),
        ],
        [
            InlineKeyboardButton("⛔ Stop", callback_data="menu:stop"),
            InlineKeyboardButton("🔎 Ieškoti naujų", callback_data="menu:refresh"),
        ],
    ])


def _project_list_rows(
    snapshot: list[dict], show_all: bool = False
) -> list[tuple[int, dict]]:
    rows = [
        (idx, row)
        for idx, row in enumerate(snapshot)
        if show_all or row.get("enabled") or row.get("last_active")
    ]
    return sorted(rows, key=lambda item: (0 if item[1].get("last_active") else 1, item[0]))


def _friendly_path(path: str) -> str:
    if path.startswith("/home/home/"):
        return "~/" + path[len("/home/home/"):]
    return path


def _find_project_row(snapshot: list[dict], project: str) -> dict | None:
    if project:
        for row in snapshot:
            if row["project"] == project or row.get("display_name") == project:
                return row
        return None
    for row in snapshot:
        if row.get("last_active"):
            return row
    return snapshot[0] if snapshot else None


def _tail_for_telegram(text: str, limit: int = 3500) -> str:
    if len(text) <= limit:
        return text
    return "...\n" + text[-limit:]


def _clean_choices(choices: list[str], limit: int = 6) -> list[str]:
    cleaned: list[str] = []
    for choice in choices:
        value = " ".join(str(choice).split())
        if not value:
            continue
        cleaned.append(value[:48])
        if len(cleaned) >= limit:
            break
    return cleaned


def build_panel_markup(snapshot: list[dict]) -> InlineKeyboardMarkup:
    """Render the /panel inline keyboard from a controls snapshot.

    Pure function: maps a snapshot (list of dicts keyed by ``"project"``) to an
    ``InlineKeyboardMarkup`` with one row per project plus a global row.

    Per-project buttons encode the project's INDEX into the snapshot list as
    callback_data (e.g. ``"tog:0"``). This avoids any dependency on project-name
    characters (especially ``:``) and keeps callback_data well under the 64-byte
    Telegram limit. Index order is stable (projects come from static config).
    """
    rows: list[list[InlineKeyboardButton]] = []
    for i, row in enumerate(snapshot):
        proj = row.get("display_name") or row["project"]
        dot = "\U0001F7E2" if row["enabled"] else "\U0001F534"  # green/red
        on_label = "ON" if row["enabled"] else "OFF"
        rows.append([
            InlineKeyboardButton(
                f"{dot} {proj}", callback_data=f"noop:{i}"),
            InlineKeyboardButton(
                on_label, callback_data=f"tog:{i}"),
            InlineKeyboardButton(
                f"{row['mode']} ▾", callback_data=f"mode:{i}"),
            InlineKeyboardButton(
                f"{row['voice']} ▾", callback_data=f"voice:{i}"),
        ])
    engine = snapshot[0]["engine"] if snapshot else "openai"
    rows.append([
        InlineKeyboardButton("▶ ALL ON", callback_data="allon"),
        InlineKeyboardButton("⏸ ALL OFF", callback_data="alloff"),
        InlineKeyboardButton(
            f"engine: {engine} ▾", callback_data="engine"),
    ])
    return InlineKeyboardMarkup(rows)


def build_mode_markup(snapshot: list[dict], idx: int) -> InlineKeyboardMarkup:
    """Render explicit mode choices for one project."""
    row = snapshot[idx]
    buttons = [
        InlineKeyboardButton(
            f"{'✓ ' if mode == row['mode'] else ''}{mode}",
            callback_data=f"mset:{idx}:{mode}",
        )
        for mode in _MODES
    ]
    return InlineKeyboardMarkup([
        [InlineKeyboardButton(f"{row.get('display_name') or row['project']} mode", callback_data=f"noop:{idx}")],
        buttons,
        [InlineKeyboardButton("back", callback_data="back")],
    ])


def build_voice_markup(snapshot: list[dict], idx: int) -> InlineKeyboardMarkup:
    """Render explicit voice choices for one project."""
    row = snapshot[idx]
    voices = available_voices(row.get("engine", "openai"))
    rows: list[list[InlineKeyboardButton]] = [
        [InlineKeyboardButton(f"{row.get('display_name') or row['project']} voice", callback_data=f"noop:{idx}")]
    ]
    for start in range(0, len(voices), 2):
        pair = voices[start:start + 2]
        rows.append([
            InlineKeyboardButton(
                f"{'✓ ' if voice == row['voice'] else ''}{voice}",
                callback_data=f"vset:{idx}:{voice}",
            )
            for voice in pair
        ])
    rows.append([InlineKeyboardButton("back", callback_data="back")])
    return InlineKeyboardMarkup(rows)


class TelegramIO:
    def __init__(
        self,
        cfg: Config,
        on_user_message: Callable[[dict], Awaitable[None]],
        controls: Controls,
    ) -> None:
        self.cfg = cfg
        self.on_user_message = on_user_message
        self.controls = controls
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
    async def send_update(
        self,
        project: str,
        voice_label: str,
        text: str,
        voice_bytes: bytes | None,
    ) -> list[int]:
        """Send a TEXT message (full, may contain code) and, if voice_bytes
        is provided, a VOICE message. Return the message_ids sent."""
        bot = self.app.bot
        ids: list[int] = []
        text_msg = await bot.send_message(
            chat_id=self._chat_id,
            text=f"[{project}] {text}",
        )
        ids.append(text_msg.message_id)
        if voice_bytes is not None:
            voice_msg = await bot.send_voice(
                chat_id=self._chat_id,
                voice=voice_bytes,
                caption=f"{project} · {voice_label}",
            )
            ids.append(voice_msg.message_id)
        return ids

    async def send_question(self, project: str, text: str) -> int:
        """Send one message and return its message_id (keys approvals)."""
        bot = self.app.bot
        msg = await bot.send_message(
            chat_id=self._chat_id,
            text=f"[{project}] {text}",
        )
        return msg.message_id

    async def ask_user(self, project: str, question: str, choices: list[str]) -> str:
        clean_choices = _clean_choices(choices)
        if not clean_choices:
            clean_choices = ["Taip", "Ne"]
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
        """Send a project-produced file and optional voice summary."""
        bot = self.app.bot
        ids: list[int] = []
        path = Path(file_path)
        caption = f"[{project}] {text}".strip()
        suffix = path.suffix.lower()

        with path.open("rb") as fh:
            if suffix in _PHOTO_SUFFIXES:
                msg = await bot.send_photo(
                    chat_id=self._chat_id,
                    photo=fh,
                    caption=caption,
                )
            elif suffix in _AUDIO_SUFFIXES:
                msg = await bot.send_audio(
                    chat_id=self._chat_id,
                    audio=fh,
                    caption=caption,
                )
            elif suffix in _VIDEO_SUFFIXES:
                msg = await bot.send_video(
                    chat_id=self._chat_id,
                    video=fh,
                    caption=caption,
                )
            else:
                msg = await bot.send_document(
                    chat_id=self._chat_id,
                    document=fh,
                    caption=caption,
                    filename=path.name,
                )
        ids.append(msg.message_id)

        if voice_bytes is not None:
            voice_msg = await bot.send_voice(
                chat_id=self._chat_id,
                voice=voice_bytes,
                caption=f"{project} · {voice_label}",
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
                    "Įjungti ir siųsti", callback_data=f"offsend:{token}"
                )
            ],
            [InlineKeyboardButton("Atšaukti", callback_data=f"offcancel:{token}")],
        ])
        msg = await self.app.bot.send_message(
            chat_id=self._chat_id,
            text=(
                f"[bridge] {project} išjungtas.\n"
                "Įjungti projektą ir išsiųsti paskutinę žinutę?"
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
        try:
            await query.answer()
        except BadRequest as exc:
            if "query is too old" in str(exc).lower():
                return
            raise
        action, index_str = parse_callback(query.data)

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
                await query.edit_message_text("Šita užklausa nebegalioja.")
                return
            project, text = pending
            if action == "offcancel":
                await query.edit_message_text(f"Atšaukta: {project}")
                return
            await self.controls.enable_and_deliver(project, text)
            await query.edit_message_text(f"Įjungta ir išsiųsta į {project}.")
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
                await query.edit_message_text("Šitas pasirinkimas nebegalioja.")
                return
            future, choices = pending
            if idx < 0 or idx >= len(choices):
                return
            choice = choices[idx]
            if not future.done():
                future.set_result(choice)
            await query.edit_message_text(f"Pasirinkta: {choice}")
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
            elif action in {"sel", "ptog"}:
                await self.controls.select(project)
                snap = self.controls.snapshot()
                await self._edit_callback_text(
                    query,
                    format_projects(snap),
                    build_projects_list_markup(snap),
                )
                return
            elif action == "ptgl":
                await self.controls.toggle(project, not row["enabled"])
                snap = self.controls.snapshot()
                await self._edit_callback_text(
                    query,
                    format_projects(snap),
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
                f"Pridėta naujų projektų: {added}\n\n"
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
            f"Pridėta naujų projektų: {added}\n\n"
            + format_projects(snapshot, show_all=True),
            parse_mode="HTML",
            reply_markup=build_projects_list_markup(snapshot, show_all=True),
        )

    async def _cmd_on(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        msg = update.message
        if msg is None or not self._allowed(msg.from_user.id):
            return
        project = context.args[0] if context.args else None
        await self.controls.toggle(project, True)
        await msg.reply_text(f"{project or 'all'} on")

    async def _cmd_off(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        msg = update.message
        if msg is None or not self._allowed(msg.from_user.id):
            return
        project = context.args[0] if context.args else None
        await self.controls.toggle(project, False)
        await msg.reply_text(f"{project or 'all'} off")

    async def _cmd_stop(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        msg = update.message
        if msg is None or not self._allowed(msg.from_user.id):
            return
        project = context.args[0] if context.args else None
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
        project = context.args[1] if len(context.args) > 1 else None
        await self.controls.set_mode(project, mode)
        await msg.reply_text(f"mode {mode} for {project or 'all'}")

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
        project = None
        if len(args) >= 3 and args[1] == "for":
            project = args[2]
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

    async def _cmd_handoff(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        msg = update.message
        if msg is None or not self._allowed(msg.from_user.id):
            return
        await msg.reply_text(self._format_handoff_text(context.args[0] if context.args else ""))

    def _format_handoff_text(self, project: str) -> str:
        row = _find_project_row(self.controls.snapshot(), project)
        if row is None:
            return "Neradau projekto. Naudok /projects_all."
        path = transcript_path(row.get("cwd") or "")
        label = row.get("display_name") or row["project"]
        if not path.exists():
            return f"{label}: istorijos dar nėra."
        text = path.read_text(encoding="utf-8", errors="replace").strip()
        if not text:
            return f"{label}: istorija tuščia."
        tail = _tail_for_telegram(text)
        return f"{label} handoff\n{_friendly_path(str(path))}\n\n{tail}"

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
            CommandHandler("voice", self._cmd_voice, filters=only_me))
        app.add_handler(
            CommandHandler("engine", self._cmd_engine, filters=only_me))
        app.add_handler(
            CommandHandler("status", self._cmd_status, filters=only_me))
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
