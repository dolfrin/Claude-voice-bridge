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
import logging
import time
from pathlib import Path
from typing import Awaitable, Callable, Protocol

from telegram import (
    BotCommand,
    InlineKeyboardButton,
    InlineKeyboardMarkup,
    MessageEntity,
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
from .routing import project_of
from .transcript import transcript_path
from .tts import available_voices

logger = logging.getLogger(__name__)


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

    # conversations (one Claude session each, one sub-topic each)
    async def open_conversation(
        self, project: str, resume: str | None = None, fork: bool = False
    ) -> str | None: ...
    async def stage_conversation(self, project: str, resume: str) -> str | None: ...
    async def attach_conversation(self, key: str) -> bool: ...
    def project_sessions(self, project: str, limit: int | None = None) -> tuple[list, int]: ...
    def session_history_text(self, project: str, uuid: str) -> str: ...
    def session_transcript_file(self, project: str, uuid: str): ...
    async def close_conversation(self, key: str) -> bool: ...
    async def reload_conversations(self) -> None: ...
    def conversation_rows(self) -> list[dict]: ...
    def resume_options(self, project: str) -> list: ...
    def history_text(self, key: str, limit: int = 12) -> str: ...


_MODES = ["auto", "safe", "full", "ask"]
_ENGINES = ["auto", "openai", "piper", "together"]
_PHOTO_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}
_AUDIO_SUFFIXES = {".mp3", ".m4a", ".ogg", ".opus", ".wav", ".flac"}
_VIDEO_SUFFIXES = {".mp4", ".mov", ".mkv", ".webm"}
_BOT_COMMANDS = [
    BotCommand("menu", "🏠 Main menu"),
    BotCommand("new", "➕ New session sub-topic"),
    BotCommand("resume", "🔄 Attach an existing Claude session"),
    BotCommand("sessions", "📋 Open sessions"),
    BotCommand("history", "📜 Session transcript"),
    BotCommand("close", "🗑 Close this session"),
    BotCommand("panel", "🎛 Control panel"),
    BotCommand("projects", "🟢 Active projects"),
    BotCommand("projects_all", "📚 All projects"),
    BotCommand("projects_refresh", "🔎 Discover new projects"),
    BotCommand("handoff", "🧾 Latest project handoff"),
    BotCommand("status", "📡 Ask project status"),
    BotCommand("on", "▶️ Enable one project or all"),
    BotCommand("off", "⏸ Disable one project or all"),
    BotCommand("stop", "⛔ Interrupt current work"),
    BotCommand("mode", "🛡 Change safe/full/ask mode"),
    BotCommand("voice", "🔊 List or set TTS voice"),
    BotCommand("engine", "🧠 Change TTS backend"),
]

# Text posted into a project's own topic is not a turn: that topic is the
# control desk, the agent lives in the numbered sub-topics.
_HUB_HINT = (
    "\U0001F5C2 <b>{name}</b> — control topic.\n"
    "Agent work happens in the <b>#N</b> sub-topics."
)


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
            InlineKeyboardButton("➕ New session", callback_data="menu:new"),
            InlineKeyboardButton("🔄 Resume", callback_data="menu:resume"),
        ],
        [
            InlineKeyboardButton("📋 Sessions", callback_data="clist:"),
            InlineKeyboardButton("📜 History", callback_data="menu:history"),
        ],
        [
            InlineKeyboardButton("🟢 Active", callback_data="menu:projects"),
            InlineKeyboardButton("📚 All", callback_data="menu:projects_all"),
        ],
        [
            InlineKeyboardButton("🎛 Panel", callback_data="menu:panel"),
            InlineKeyboardButton("🧾 Handoff", callback_data="menu:handoff"),
        ],
        [
            InlineKeyboardButton("⛔ Stop", callback_data="menu:stop"),
            InlineKeyboardButton("🔎 Refresh", callback_data="menu:refresh"),
        ],
    ])


def build_project_pick_markup(
    snapshot: list[dict], action: str, show_all: bool = False
) -> InlineKeyboardMarkup:
    """Pick a project, then run *action* on it.

    ``/new`` and ``/resume`` need a project, and the main menu is opened from
    General where no topic names one — so the menu asks first.
    """
    rows = []
    for idx, row in _project_list_rows(snapshot, show_all=show_all):
        status = "\U0001F7E2" if row["enabled"] else "⚪"
        name = row.get("display_name") or row["project"]
        rows.append([
            InlineKeyboardButton(f"{status} {name}", callback_data=f"{action}:{idx}")
        ])
    rows.append([InlineKeyboardButton("« menu", callback_data="menu:home")])
    return InlineKeyboardMarkup(rows)


def format_session_index(label: str, sessions: list, total: int) -> str:
    """Header for a project's session list, naming what the cap left out."""
    if not sessions:
        return f"{html.escape(label)}: no Claude sessions on disk."
    shown = (
        f"showing {len(sessions)} of {total}" if total > len(sessions)
        else f"{total} session{'s' if total != 1 else ''}"
    )
    return f"\U0001F4DC <b>{html.escape(label)}</b> — {shown}"


def build_session_pick_markup(idx: int, sessions: list, action: str) -> InlineKeyboardMarkup:
    rows = [
        [
            InlineKeyboardButton(
                f"{'🟢 ' if s.live else ''}{(s.title or s.uuid)[:44]}",
                callback_data=f"{action}:{idx}:{s.uuid}",
            )
        ]
        for s in sessions
    ]
    rows.append([InlineKeyboardButton("« menu", callback_data="menu:home")])
    return InlineKeyboardMarkup(rows)


def build_transcript_markup(idx: int, uuid: str) -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup([
        [InlineKeyboardButton("📄 Full transcript", callback_data=f"cfull:{idx}:{uuid}")],
        [InlineKeyboardButton("« back", callback_data=f"chp:{idx}")],
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


def tail_for_telegram(text: str, limit: int = 3500) -> str:
    if len(text) <= limit:
        return text
    return "...\n" + text[-limit:]


# Telegram rejects a sendMessage body longer than this.
TELEGRAM_MAX_MESSAGE = 4096

def _utf16_len(text: str) -> int:
    """Telegram entity offsets/lengths count UTF-16 code units, not chars."""
    return len(text.encode("utf-16-le")) // 2


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


def build_hub_markup(idx: int) -> InlineKeyboardMarkup:
    """Buttons of a project's control topic."""
    return InlineKeyboardMarkup([
        [
            InlineKeyboardButton("➕ New session", callback_data=f"cnew:{idx}"),
            InlineKeyboardButton("🔄 Resume", callback_data=f"cres:{idx}"),
        ],
        [
            InlineKeyboardButton("📋 Sessions", callback_data=f"clist:{idx}"),
            InlineKeyboardButton("🎛 Panel", callback_data="menu:panel"),
        ],
    ])


def topic_link(chat_id: int, thread_id: int) -> str | None:
    """``t.me`` deep link to a forum topic, for supergroups only."""
    text = str(chat_id)
    if not text.startswith("-100"):
        return None
    return f"https://t.me/c/{text[4:]}/{thread_id}"


def format_conversations(rows: list[dict]) -> str:
    """Render the open-sessions list.

    Two names per row on purpose: the topic it lives in, and what Claude calls
    the session. The topic name alone is just a number.
    """
    if not rows:
        return "No open sessions. Use /new in a project topic."
    lines: list[str] = []
    for row in rows:
        if row.get("busy"):
            status = "⚡"
        elif row.get("running"):
            status = "\U0001F7E2"
        else:
            status = "⚪"
        title = html.escape(row.get("title") or row["key"])
        lines.append(f"{status} <b>{title}</b>")
        session_title = row.get("session_title") or ""
        lines.append(
            f"    ↳ <i>{html.escape(session_title[:60])}</i>"
            if session_title
            else "    ↳ <i>new session, no name yet</i>"
        )
    lines.append("\n⚡ working now · 🟢 live, waiting · ⚪ stopped")
    return "\n".join(lines)


def conversation_label(row: dict, limit: int = 42) -> str:
    """``Paprika ASR #2 · Fix topic routing bug``, clipped to fit a button."""
    label = row.get("title") or row["key"]
    session_title = row.get("session_title") or ""
    if not session_title:
        return label
    room = limit - len(label) - 3
    if room < 8:
        return label
    return f"{label} · {session_title[:room]}"


def build_conversations_markup(rows: list[dict], chat_id: int | None) -> InlineKeyboardMarkup:
    buttons: list[list[InlineKeyboardButton]] = []
    for row in rows:
        label = conversation_label(row)
        link = (
            topic_link(chat_id, row["thread_id"])
            if chat_id is not None and row.get("thread_id")
            else None
        )
        open_button = (
            InlineKeyboardButton(f"\U0001F5C2 {label}", url=link)
            if link
            else InlineKeyboardButton(f"\U0001F5C2 {label}", callback_data="noop:0")
        )
        buttons.append([
            open_button,
            InlineKeyboardButton("🗑", callback_data=f"cdel:{row['key']}"),
        ])
    return InlineKeyboardMarkup(buttons)


def format_resume_options(project_label: str, sessions: list) -> str:
    """Render the Claude-session picker: title, when, where we left off."""
    if not sessions:
        return f"{html.escape(project_label)}: no Claude sessions on disk yet."
    lines = [f"\U0001F4C2 <b>{html.escape(project_label)}</b>\n"]
    for i, session in enumerate(sessions, 1):
        when = time.strftime("%m-%d %H:%M", time.localtime(session.mtime))
        live = (
            f" \U0001F7E2 <i>open in {html.escape(session.live)}</i>"
            if session.live
            else ""
        )
        lines.append(f"<b>{i}. {html.escape(session.title[:60])}</b>{live}")
        lines.append(f"    <i>{when}</i>")
        if session.last_prompt:
            lines.append(f"    ↳ <i>{html.escape(session.last_prompt[:70])}</i>")
    lines.append(
        "\n\U0001F7E2 = open elsewhere → attaches as a <b>fork</b> "
        "(the original is untouched)."
    )
    return "\n".join(lines)


def build_resume_markup(idx: int, sessions: list) -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup([
        [
            InlineKeyboardButton(
                f"{i}. {'🟢 ' if s.live else ''}{(s.title or s.uuid)[:40]}",
                callback_data=f"rs:{idx}:{s.uuid}",
            )
        ]
        for i, s in enumerate(sessions, 1)
    ])


def format_attach_card(key: str, session, transcript: str) -> str:
    """The card posted INTO a staged session's own topic.

    Everything needed to recognise a session before reopening it: what Claude
    calls it, where it runs, when it was last touched, and how it ended.
    """
    when = time.strftime("%m-%d %H:%M", time.localtime(session.mtime))
    lines = [
        f"\U0001F504 <b>{html.escape(session.title or session.uuid)}</b>",
        f"\U0001F4C2 <code>{html.escape(session.cwd)}</code>",
        f"\U0001F551 {when} · <code>{html.escape(session.uuid[:8])}</code>",
    ]
    if session.last_prompt:
        lines.append(f"↳ <i>{html.escape(session.last_prompt[:120])}</i>")
    if session.live:
        lines.append(
            f"\U0001F7E2 open in <b>{html.escape(session.live)}</b> → "
            "will attach as a <b>fork</b>, the original stays untouched"
        )
    if transcript:
        lines.append("")
        lines.append(transcript)
    return "\n".join(lines)


def build_attach_markup(key: str) -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup([[
        InlineKeyboardButton("🔗 Attach", callback_data=f"cok:{key}"),
        InlineKeyboardButton("🗑 Cancel", callback_data=f"cdel:{key}"),
    ]])


def build_progress_markup(key: str) -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup([
        [InlineKeyboardButton("⛔ Stop", callback_data=f"pstop:{key}")]
    ])


class TelegramIO:
    def __init__(
        self,
        cfg: Config,
        on_user_message: Callable[[dict], Awaitable[None]],
        controls: Controls,
        store=None,
        projects=None,
    ) -> None:
        self.cfg = cfg
        self.on_user_message = on_user_message
        self.controls = controls
        # Both only needed in group mode, to create and persist the per-project
        # forum topics once the Application exists.
        self._store = store
        self._projects = list(projects or [])
        # project name -> control topic; conversation key -> its own sub-topic.
        self._topics: dict[str, int] = {}
        self._conv_topics: dict[str, int] = {}
        # conversation key -> the live progress message being edited in place.
        self._progress_msgs: dict[str, int] = {}
        self.app: Application | None = None
        self._pending_off_sends: dict[str, tuple[str, str]] = {}
        self._pending_off_seq = 0
        # token -> (future, labels, bodies). bodies is None for ask_user
        # (single-message flow, no Edit) and carries the untruncated option
        # bodies for ask_per_message so the Edit tap can dump the text.
        self._pending_asks: dict[
            str, tuple[asyncio.Future[str | None], list[str], list[str] | None]
        ] = {}
        self._pending_ask_seq = 0

    # --- whitelist -------------------------------------------------------
    def _allowed(self, user_id: int | None) -> bool:
        return user_id == self.cfg.telegram_allowed_user_id

    @property
    def _chat_id(self) -> int:
        # Group mode posts into the configured supergroup; otherwise the only
        # authorized user is also the chat target.
        if self.cfg.telegram_chat_id is not None:
            return self.cfg.telegram_chat_id
        return self.cfg.telegram_allowed_user_id

    def _dest(self, target: str | None) -> dict:
        """Destination kwargs for any send: the chat, and the right topic.

        *target* is either a conversation key (``"paprika#1"`` — its own
        sub-topic) or a project name (its control topic). In private-chat mode
        there are no topics and this is just the chat id. Anything without a
        topic yet falls back to the group's General thread rather than failing
        the send.
        """
        dest = {"chat_id": self._chat_id}
        thread_id = None
        if target:
            thread_id = self._conv_topics.get(target)
            if thread_id is None:
                thread_id = self._topics.get(project_of(target))
        if thread_id is not None:
            dest["message_thread_id"] = thread_id
        return dest

    def project_for_thread(self, thread_id: int | None) -> str | None:
        """Which project's CONTROL topic this is, if any."""
        if thread_id is None:
            return None
        for name, tid in self._topics.items():
            if tid == thread_id:
                return name
        return None

    def conversation_for_thread(self, thread_id: int | None) -> str | None:
        """Which conversation owns an incoming sub-topic, if any."""
        if thread_id is None:
            return None
        for key, tid in self._conv_topics.items():
            if tid == thread_id:
                return key
        return None

    async def ensure_topics(self, projects: list | None = None) -> None:
        """Create one forum topic per project, reusing any already recorded.

        No-op outside group mode. A failure to create one topic is logged and
        skipped: that project simply keeps landing in General, which is far
        better than refusing to start the bot.
        """
        chat_id = self.cfg.telegram_chat_id
        if chat_id is None or self._store is None:
            return
        self._topics = await self._store.topics_for_chat(chat_id)
        for project in self._projects if projects is None else projects:
            if project.name in self._topics:
                continue
            title = (project.display_name or project.name)[:128]
            try:
                topic = await self.app.bot.create_forum_topic(
                    chat_id=chat_id, name=title
                )
            except Exception:  # noqa: BLE001 - one bad topic must not stop startup
                logger.exception("could not create a forum topic for %s", project.name)
                continue
            self._topics[project.name] = topic.message_thread_id
            await self._store.set_topic(
                project.name, chat_id, topic.message_thread_id
            )

    async def load_conversation_topics(self) -> None:
        """Remember the sub-topics of every open conversation (restart-safe)."""
        if self._store is None:
            return
        self._conv_topics = {
            conv.key: conv.thread_id
            for conv in await self._store.conversations()
            if conv.thread_id is not None
        }

    async def open_conversation_topic(
        self, key: str, project: str, ordinal: int
    ) -> int | None:
        """Create the ``<project> #N`` sub-topic for a new conversation.

        Outside group mode there is nothing to create; the conversation simply
        shares the private chat, which is the pre-topics behaviour.
        """
        chat_id = self.cfg.telegram_chat_id
        if chat_id is None or self.app is None:
            return None
        base = next(
            (p.display_name or p.name for p in self._projects if p.name == project),
            project,
        )
        title = f"{base} #{ordinal}"[:128]
        try:
            topic = await self.app.bot.create_forum_topic(chat_id=chat_id, name=title)
        except Exception:  # noqa: BLE001 - fall back to General, never fail the open
            logger.exception("could not create a sub-topic for %s", key)
            return None
        self._conv_topics[key] = topic.message_thread_id
        if self._store is not None:
            await self._store.set_conversation_topic(
                key, chat_id, topic.message_thread_id
            )
        return topic.message_thread_id

    async def close_conversation_topic(self, key: str) -> None:
        """Delete a closed conversation's sub-topic; a left-behind one only
        clutters the forum, and its conversation can never be reopened."""
        thread_id = self._conv_topics.pop(key, None)
        self._progress_msgs.pop(key, None)
        chat_id = self.cfg.telegram_chat_id
        if thread_id is None or chat_id is None or self.app is None:
            return
        try:
            await self.app.bot.delete_forum_topic(
                chat_id=chat_id, message_thread_id=thread_id
            )
        except Exception:  # noqa: BLE001 - already gone / no rights
            logger.info("could not delete sub-topic for %s", key, exc_info=True)

    async def send_progress(self, key: str, text: str, final: bool) -> None:
        """Show what the agent is doing right now, editing ONE message.

        A new message per tool call would bury the conversation, so the live
        view is a single message edited in place — and dropped once the turn
        ends, leaving only the final one-line summary.
        """
        bot = self.app.bot
        message_id = self._progress_msgs.get(key)
        markup = None if final else build_progress_markup(key)
        if message_id is None:
            if final:
                return
            msg = await bot.send_message(
                **self._dest(key), text=text, reply_markup=markup
            )
            self._progress_msgs[key] = msg.message_id
            return
        try:
            await bot.edit_message_text(
                chat_id=self._chat_id,
                message_id=message_id,
                text=text,
                reply_markup=markup,
            )
        except BadRequest as exc:
            if "message is not modified" not in str(exc).lower():
                logger.debug("progress edit failed for %s: %s", key, exc)
        if final:
            self._progress_msgs.pop(key, None)

    # --- inbound handlers ------------------------------------------------
    @staticmethod
    def _reply_to(msg) -> int | None:
        if msg.reply_to_message is not None:
            return msg.reply_to_message.message_id
        return None

    async def _hub_of(self, msg) -> str | None:
        """The project whose CONTROL topic this message landed in, if any.

        A hit means the message is not a turn: it is answered with the hub's
        buttons instead of being forwarded to an agent.
        """
        thread_id = getattr(msg, "message_thread_id", None)
        if self.conversation_for_thread(thread_id) is not None:
            return None
        return self.project_for_thread(thread_id)

    async def _reply_hub(self, msg, project: str) -> None:
        snapshot = self.controls.snapshot()
        idx = next(
            (i for i, row in enumerate(snapshot) if row["project"] == project), None
        )
        row = snapshot[idx] if idx is not None else None
        name = (row.get("display_name") or project) if row else project
        await msg.reply_text(
            _HUB_HINT.format(name=html.escape(name)),
            parse_mode="HTML",
            reply_markup=build_hub_markup(idx) if idx is not None else None,
        )

    def _inbound(self, msg, key: str | None, **extra) -> dict:
        payload = {
            "message_id": msg.message_id,
            "reply_to": self._reply_to(msg),
            "project": key,
            "text": "",
            "is_voice": False,
            "audio": None,
        }
        payload.update(extra)
        return payload

    async def _handle_text(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        msg = update.message
        if msg is None or not self._allowed(msg.from_user.id):
            return
        hub = await self._hub_of(msg)
        if hub is not None:
            await self._reply_hub(msg, hub)
            return
        key = self.conversation_for_thread(getattr(msg, "message_thread_id", None))
        await self.on_user_message(self._inbound(msg, key, text=msg.text or ""))

    async def _handle_voice(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        msg = update.message
        if msg is None or not self._allowed(msg.from_user.id):
            return
        hub = await self._hub_of(msg)
        if hub is not None:
            await self._reply_hub(msg, hub)
            return
        tg_file = await msg.voice.get_file()
        audio = bytes(await tg_file.download_as_bytearray())
        key = self.conversation_for_thread(getattr(msg, "message_thread_id", None))
        await self.on_user_message(
            self._inbound(msg, key, is_voice=True, audio=audio)
        )

    async def _handle_attachment(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        msg = update.message
        if msg is None or not self._allowed(msg.from_user.id):
            return
        hub = await self._hub_of(msg)
        if hub is not None:
            await self._reply_hub(msg, hub)
            return
        attachment = await _download_attachment(msg)
        if attachment is None:
            return
        key = self.conversation_for_thread(getattr(msg, "message_thread_id", None))
        await self.on_user_message(
            self._inbound(
                msg, key, text=msg.caption or "", attachments=[attachment]
            )
        )

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
            **self._dest(project),
            text=f"[{project}] {text}",
        )
        ids.append(text_msg.message_id)
        if voice_bytes is not None:
            voice_msg = await bot.send_voice(
                **self._dest(project),
                voice=voice_bytes,
                caption=f"{project} · {voice_label}",
            )
            ids.append(voice_msg.message_id)
        return ids

    async def send_question(self, project: str, text: str) -> int:
        """Send one message and return its message_id (keys approvals)."""
        bot = self.app.bot
        msg = await bot.send_message(
            **self._dest(project),
            text=f"[{project}] {text}",
        )
        return msg.message_id

    async def ask_per_message(
        self,
        project: str,
        options: list[tuple[str, str]],
        button: str = "Accept this",
    ) -> str | None:
        """Send one message per option, each carrying its own accept button.

        Unlike :meth:`ask_user`, which puts every choice on one message, this
        gives each option a full message of its own so long bodies stay
        readable. Returns the accepted option's label, or ``""`` if nothing was
        tapped before ``approval_timeout``. The first tap resolves the choice
        and expires the rest.
        Returns ``None`` when the user taps Edit: the tapped message is
        replaced with the raw transcript as a tap-to-copy code block and the
        caller must abort silently — the user resends corrected text as a
        normal message.
        """
        labels = [
            " ".join(str(label).split())[:48] or f"option {idx + 1}"
            for idx, (label, _body) in enumerate(options)
        ]
        self._pending_ask_seq += 1
        token = str(self._pending_ask_seq)
        loop = asyncio.get_running_loop()
        future: asyncio.Future[str] = loop.create_future()
        self._pending_asks[token] = (
            future,
            labels,
            [body for _label, body in options],
        )
        for idx, (_label, body) in enumerate(options):
            header = f"[{project}] {labels[idx]}\n\n"
            await self.app.bot.send_message(
                **self._dest(project),
                text=(header + body)[:TELEGRAM_MAX_MESSAGE],
                reply_markup=InlineKeyboardMarkup([[
                    InlineKeyboardButton(button, callback_data=f"ask:{token}:{idx}"),
                    InlineKeyboardButton("Edit", callback_data=f"ask:{token}:e{idx}"),
                ]]),
            )
        try:
            return await asyncio.wait_for(future, timeout=self.cfg.approval_timeout)
        except asyncio.TimeoutError:
            return ""
        finally:
            self._pending_asks.pop(token, None)

    async def ask_user(self, project: str, question: str, choices: list[str]) -> str:
        clean_choices = _clean_choices(choices)
        if not clean_choices:
            clean_choices = ["Yes", "No"]
        self._pending_ask_seq += 1
        token = str(self._pending_ask_seq)
        loop = asyncio.get_running_loop()
        future: asyncio.Future[str] = loop.create_future()
        self._pending_asks[token] = (future, clean_choices, None)
        rows = [
            [InlineKeyboardButton(choice, callback_data=f"ask:{token}:{idx}")]
            for idx, choice in enumerate(clean_choices)
        ]
        await self.app.bot.send_message(
            **self._dest(project),
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
                    **self._dest(project),
                    photo=fh,
                    caption=caption,
                )
            elif suffix in _AUDIO_SUFFIXES:
                msg = await bot.send_audio(
                    **self._dest(project),
                    audio=fh,
                    caption=caption,
                )
            elif suffix in _VIDEO_SUFFIXES:
                msg = await bot.send_video(
                    **self._dest(project),
                    video=fh,
                    caption=caption,
                )
            else:
                msg = await bot.send_document(
                    **self._dest(project),
                    document=fh,
                    caption=caption,
                    filename=path.name,
                )
        ids.append(msg.message_id)

        if voice_bytes is not None:
            voice_msg = await bot.send_voice(
                **self._dest(project),
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
                    "Enable and send", callback_data=f"offsend:{token}"
                )
            ],
            [InlineKeyboardButton("Cancel", callback_data=f"offcancel:{token}")],
        ])
        msg = await self.app.bot.send_message(
            **self._dest(project),
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
        if action in {"cnew", "cres", "clist", "chp", "chs", "cfull",
                      "rs", "cok", "cdel", "pstop"}:
            await self._handle_conversation_callback(query, action, index_str)
            return
        if action == "ask":
            try:
                token, choice_idx = index_str.split(":", 1)
            except (ValueError, TypeError):
                return
            pending = self._pending_asks.get(token)
            if pending is None:
                await query.edit_message_text("This choice has expired.")
                return
            future, choices, bodies = pending
            if choice_idx.startswith("e"):
                # Edit: dump the raw body as a tap-to-copy code block and
                # abort the choice silently (future resolves to None).
                try:
                    idx = int(choice_idx[1:])
                except ValueError:
                    return
                if bodies is None or not 0 <= idx < len(bodies):
                    return
                body = bodies[idx][:TELEGRAM_MAX_MESSAGE]
                await query.edit_message_text(
                    body,
                    entities=[
                        MessageEntity(
                            type=MessageEntity.CODE,
                            offset=0,
                            length=_utf16_len(body),
                        )
                    ],
                )
                if not future.done():
                    future.set_result(None)
                return
            try:
                idx = int(choice_idx)
            except ValueError:
                return
            if idx < 0 or idx >= len(choices):
                return
            choice = choices[idx]
            if not future.done():
                future.set_result(choice)
            await query.edit_message_text(f"Selected: {choice}")
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

    async def _handle_conversation_callback(
        self, query, action: str, payload: str
    ) -> None:
        if action == "clist":
            await self.controls.reload_conversations()
            rows = self.controls.conversation_rows()
            await self._edit_callback_text(
                query,
                format_conversations(rows),
                build_conversations_markup(rows, self.cfg.telegram_chat_id),
            )
            return
        if action in {"chp", "chs", "cfull"}:
            await self._handle_history_callback(query, action, payload)
            return
        if action == "cdel":
            closed = await self.controls.close_conversation(payload)
            await self._edit_callback_text(
                query,
                f"Closed {payload}." if closed else f"{payload} is already gone.",
                build_menu_markup(),
            )
            return
        if action == "pstop":
            await self._edit_callback_text(
                query, await self.controls.interrupt(payload), build_menu_markup()
            )
            return
        if action == "rs":
            await self._stage_session(query, payload)
            return
        if action == "cok":
            started = await self.controls.attach_conversation(payload)
            await self._edit_callback_text(
                query,
                f"✅ <b>Attached</b> — write here to continue {html.escape(payload)}."
                if started
                else "Could not attach that session.",
                None,
            )
            return

        project = self._project_by_index(payload)
        if project is None:
            return
        if action == "cnew":
            key = await self.controls.open_conversation(project)
            await self._edit_callback_text(
                query,
                f"Opened {key}." if key else "Could not open a session.",
                build_menu_markup(),
            )
        elif action == "cres":
            view = self._resume_view(project, int(payload))
            await self._edit_callback_text(
                query, view["text"], view["reply_markup"] or build_menu_markup()
            )

    async def _handle_history_callback(self, query, action: str, payload: str) -> None:
        """Browse a project's real Claude history: project -> session -> text.

        This reads what is on disk, not the conversations the bridge happens to
        have open — a project can have dozens of sessions the bridge never
        touched, and those are usually the ones you want to look back at.
        """
        index_str, _, uuid = payload.partition(":")
        project = self._project_by_index(index_str)
        if project is None:
            return
        idx = int(index_str)
        row = _find_project_row(self.controls.snapshot(), project)
        label = (row.get("display_name") or project) if row else project

        if action == "chp":
            sessions, total = self.controls.project_sessions(
                project, self.cfg.history_limit
            )
            await self._edit_callback_text(
                query,
                format_session_index(label, sessions, total),
                build_session_pick_markup(idx, sessions, "chs")
                if sessions
                else build_menu_markup(),
            )
            return

        if action == "chs":
            await self._edit_callback_text(
                query,
                self.controls.session_history_text(project, uuid),
                build_transcript_markup(idx, uuid),
            )
            return

        # cfull: the message can only ever hold a fragment; send the whole thing.
        transcript = self.controls.session_transcript_file(project, uuid)
        if transcript is None:
            await query.answer("nothing on disk", show_alert=True)
            return
        filename, data = transcript
        await self.app.bot.send_document(
            **self._dest_of(query),
            document=data,
            filename=filename,
            caption=f"📄 {label} · {uuid[:8]}",
        )

    def _dest_of(self, query) -> dict:
        """Reply in the topic the button was tapped in, whichever that is."""
        dest = {"chat_id": self._chat_id}
        message = getattr(query, "message", None)
        thread_id = getattr(message, "message_thread_id", None) if message else None
        if thread_id is not None:
            dest["message_thread_id"] = thread_id
        return dest

    def _project_by_index(self, index_str: str) -> str | None:
        try:
            idx = int(index_str)
        except (TypeError, ValueError):
            return None
        snapshot = self.controls.snapshot()
        if idx < 0 or idx >= len(snapshot):
            return None
        return snapshot[idx]["project"]

    async def _handle_menu_callback(self, query, action: str) -> None:
        snapshot = self.controls.snapshot()
        if action == "home":
            await self._edit_callback_text(
                query, "🏠 Alex for Claude", build_menu_markup()
            )
        elif action == "new":
            await self._edit_callback_text(
                query,
                "➕ New session in which project?",
                build_project_pick_markup(snapshot, "cnew", show_all=True),
            )
        elif action == "resume":
            await self._edit_callback_text(
                query,
                "🔄 Attach an existing Claude session — which project?",
                build_project_pick_markup(snapshot, "cres", show_all=True),
            )
        elif action == "history":
            await self._edit_callback_text(
                query,
                "📜 History of which project?",
                build_project_pick_markup(snapshot, "chp", show_all=True),
            )
        elif action == "projects":
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
            reason = str(exc).lower()
            # "not modified" is a no-op; "not found" means the message went with
            # the topic we just deleted, which is exactly what was asked for.
            if "message is not modified" in reason or "not found" in reason:
                logger.debug("callback edit skipped: %s", exc)
                return
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
            f"New projects added: {added}\n\n"
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
            await msg.reply_text("usage: /mode <auto|full|safe|ask> [project]")
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
            return "Project not found. Use /projects_all."
        path = transcript_path(row.get("cwd") or "")
        label = row.get("display_name") or row["project"]
        if not path.exists():
            return f"{label}: no handoff history yet."
        text = path.read_text(encoding="utf-8", errors="replace").strip()
        if not text:
            return f"{label}: handoff history is empty."
        tail = tail_for_telegram(text)
        return f"{label} handoff\n{_friendly_path(str(path))}\n\n{tail}"

    # --- conversations ---------------------------------------------------
    def _project_index(self, project: str) -> int | None:
        return next(
            (
                i
                for i, row in enumerate(self.controls.snapshot())
                if row["project"] == project
            ),
            None,
        )

    def _target_project(self, msg, context) -> str | None:
        """Which project a hub command means: argument, then topic, then active."""
        if context.args:
            row = _find_project_row(self.controls.snapshot(), context.args[0])
            return row["project"] if row else None
        thread_id = getattr(msg, "message_thread_id", None)
        hub = self.project_for_thread(thread_id)
        if hub is not None:
            return hub
        key = self.conversation_for_thread(thread_id)
        if key is not None:
            return project_of(key)
        row = _find_project_row(self.controls.snapshot(), "")
        return row["project"] if row else None

    def _target_conversation(self, msg) -> str | None:
        key = self.conversation_for_thread(getattr(msg, "message_thread_id", None))
        if key is not None:
            return key
        rows = self.controls.conversation_rows()
        return rows[0]["key"] if rows else None

    def _topic_link_markup(self, key: str) -> InlineKeyboardMarkup | None:
        """A tap-through to a conversation's own topic, when there is one."""
        thread_id = self._conv_topics.get(key)
        chat_id = self.cfg.telegram_chat_id
        link = (
            topic_link(chat_id, thread_id)
            if chat_id is not None and thread_id is not None
            else None
        )
        if link is None:
            return None
        return InlineKeyboardMarkup([
            [InlineKeyboardButton(f"\U0001F5C2 {key}", url=link)]
        ])

    async def _announce_opened(self, target, key: str | None) -> None:
        if key is None:
            await target.reply_text("Could not open a session.")
            return
        await target.reply_text(
            f"Opened {key}.", reply_markup=self._topic_link_markup(key)
        )

    async def _stage_session(self, query, payload: str) -> None:
        """Reserve a topic for a picked session and put its card inside it.

        The picker lives wherever it was opened — often General — but the card
        and the Attach button belong in the topic that will hold the session, so
        nothing about it is left behind in a shared thread.
        """
        try:
            index_str, uuid = payload.split(":", 1)
        except ValueError:
            return
        project = self._project_by_index(index_str)
        if project is None:
            return
        session = next(
            (s for s in self.controls.resume_options(project) if s.uuid == uuid), None
        )
        if session is None:
            await self._edit_callback_text(
                query, "That session is no longer on disk.", build_menu_markup()
            )
            return
        key = await self.controls.stage_conversation(project, uuid)
        if key is None:
            await self._edit_callback_text(
                query, "Could not reserve a session.", build_menu_markup()
            )
            return
        transcript = self.controls.history_text(key, 6, 3000)
        if not transcript.startswith("\U0001F4DC"):
            transcript = ""  # a "no transcript" notice adds nothing to the card
        await self.app.bot.send_message(
            **self._dest(key),
            text=format_attach_card(key, session, transcript),
            parse_mode="HTML",
            reply_markup=build_attach_markup(key),
        )
        await self._edit_callback_text(
            query,
            f"→ <b>{html.escape(key)}</b> — confirm the attach in its topic.",
            self._topic_link_markup(key) or build_menu_markup(),
        )

    async def _cmd_new(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        msg = update.message
        if msg is None or not self._allowed(msg.from_user.id):
            return
        project = self._target_project(msg, context)
        if project is None:
            await msg.reply_text("usage: /new <project> (or run it in a project topic)")
            return
        key = await self.controls.open_conversation(project)
        await self._announce_opened(msg, key)

    async def _cmd_resume(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        msg = update.message
        if msg is None or not self._allowed(msg.from_user.id):
            return
        project = self._target_project(msg, context)
        idx = self._project_index(project) if project else None
        if project is None or idx is None:
            await msg.reply_text("usage: /resume <project> (or run it in a project topic)")
            return
        await msg.reply_text(**self._resume_view(project, idx))

    def _resume_view(self, project: str, idx: int) -> dict:
        sessions = self.controls.resume_options(project)
        row = _find_project_row(self.controls.snapshot(), project)
        label = (row.get("display_name") or project) if row else project
        return {
            "text": format_resume_options(label, sessions),
            "parse_mode": "HTML",
            "reply_markup": build_resume_markup(idx, sessions) if sessions else None,
        }

    async def _cmd_sessions(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        msg = update.message
        if msg is None or not self._allowed(msg.from_user.id):
            return
        await self.controls.reload_conversations()
        rows = self.controls.conversation_rows()
        await msg.reply_text(
            format_conversations(rows),
            parse_mode="HTML",
            reply_markup=build_conversations_markup(rows, self.cfg.telegram_chat_id),
        )

    async def _cmd_history(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        msg = update.message
        if msg is None or not self._allowed(msg.from_user.id):
            return
        # The session id is only known after the first turn answers, so refresh
        # before reading rather than showing "no transcript yet" for a session
        # that has one.
        await self.controls.reload_conversations()
        key = self._target_conversation(msg)
        if key is None:
            await msg.reply_text("No open session. Use /new in a project topic.")
            return
        limit = 12
        if context.args:
            try:
                limit = max(1, min(50, int(context.args[0])))
            except ValueError:
                pass
        await msg.reply_text(
            self.controls.history_text(key, limit), parse_mode="HTML"
        )

    async def _cmd_close(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        msg = update.message
        if msg is None or not self._allowed(msg.from_user.id):
            return
        key = self.conversation_for_thread(getattr(msg, "message_thread_id", None))
        if key is None:
            await msg.reply_text("Run /close inside a session topic.")
            return
        await msg.reply_text(
            f"Close {key} and delete this topic?",
            reply_markup=InlineKeyboardMarkup([[
                InlineKeyboardButton("🗑 Yes", callback_data=f"cdel:{key}"),
                InlineKeyboardButton("No", callback_data="noop:0"),
            ]]),
        )

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
            CommandHandler("new", self._cmd_new, filters=only_me))
        app.add_handler(
            CommandHandler("resume", self._cmd_resume, filters=only_me))
        app.add_handler(
            CommandHandler("sessions", self._cmd_sessions, filters=only_me))
        app.add_handler(
            CommandHandler("history", self._cmd_history, filters=only_me))
        app.add_handler(
            CommandHandler("close", self._cmd_close, filters=only_me))
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
        # block=False: a voice message may pause mid-handler to ask which
        # transcript to accept. Updates are processed sequentially by default,
        # so a blocking handler would stall the queue and the very button press
        # it is waiting for could never arrive.
        app.add_handler(MessageHandler(
            only_me & filters.VOICE, self._handle_voice, block=False))
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
        await self.ensure_topics()
        await self.load_conversation_topics()
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
