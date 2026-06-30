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

from typing import Awaitable, Callable, Protocol

from telegram import (
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
    async def set_mode(self, project: str | None, mode: str) -> None: ...
    async def set_voice(self, project: str | None, voice: str) -> None: ...
    async def set_engine(self, name: str) -> None: ...


_MODES = ["safe", "full", "ask"]
_ENGINES = ["openai", "piper"]


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
    Per-project actions: ``tog``, ``mode``, ``voice``, ``noop``.
    """
    parts = data.split(":", 1)
    action = parts[0]
    index_str = parts[1] if len(parts) > 1 else ""
    return action, index_str


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
        proj = row["project"]
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

    # --- /panel + callbacks ---------------------------------------------
    async def _cmd_panel(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        msg = update.message
        if msg is None or not self._allowed(msg.from_user.id):
            return
        markup = build_panel_markup(self.controls.snapshot())
        await msg.reply_text("Control panel", reply_markup=markup)

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
            elif action == "mode":
                await self.controls.set_mode(
                    project, _next(_MODES, row["mode"]))
            elif action == "voice":
                await self.controls.set_voice(
                    project, _next(available_voices("openai"), row["voice"]))
            else:
                return

        new_markup = build_panel_markup(self.controls.snapshot())
        try:
            await query.edit_message_reply_markup(reply_markup=new_markup)
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
        lines = []
        for r in self.controls.snapshot():
            state = "on" if r["enabled"] else "off"
            star = " *" if r["last_active"] else ""
            lines.append(
                f"{r['project']}: {state} · {r['mode']} · "
                f"{r['voice']} · {r['engine']}{star}"
            )
        await msg.reply_text("\n".join(lines) if lines else "no projects")

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
            await msg.reply_text("voices: " + ", ".join(available_voices("openai")))
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
            await msg.reply_text("usage: /engine <openai|piper>")
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
            CommandHandler("panel", self._cmd_panel, filters=only_me))
        app.add_handler(
            CommandHandler("projects", self._cmd_projects, filters=only_me))
        app.add_handler(
            CommandHandler("on", self._cmd_on, filters=only_me))
        app.add_handler(
            CommandHandler("off", self._cmd_off, filters=only_me))
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
            only_me & filters.TEXT & ~filters.COMMAND, self._handle_text))

        await app.initialize()
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
