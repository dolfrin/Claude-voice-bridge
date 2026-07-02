"""Pure view/format/parse helpers for the Telegram front end.

Extracted from ``telegram_io.py`` for maintainability. Everything here is a
module-level, ``self``/``Bot``-independent helper: it takes plain data (a
controls snapshot, callback strings, paths) and returns Telegram markup or
strings, with no network and no shared state. ``telegram_io`` re-exports these
names so existing ``telegram_io.build_panel_markup`` references keep resolving.

This module MUST NOT import ``telegram_io`` (that would create a cycle).
"""

from __future__ import annotations

import html
from pathlib import Path

from telegram import BotCommand, InlineKeyboardButton, InlineKeyboardMarkup

from .config import AUTONOMY_MODES, TTS_BACKENDS
from .tts import available_voices

# Local aliases (list, not tuple) kept for minimal churn at call sites below;
# config.py is the single source of truth for the order and the members.
_MODES = list(AUTONOMY_MODES)
_ENGINES = list(TTS_BACKENDS)
_BOT_COMMANDS = [
    BotCommand("menu", "🏠 Main menu"),
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
    BotCommand("verbose", "🔧 Toggle live tool activity"),
    BotCommand("engine", "🧠 Change TTS backend"),
    BotCommand("recap", "🗒 What happened while away"),
    BotCommand("cost", "💰 Token & cost usage"),
]


def parse_callback(data: str) -> tuple[str, str]:
    """Decode ``"<action>:<index_or_empty>"`` callback data.

    Returns ``(action, index_str)`` where ``index_str`` is the project index
    (as a string) for per-project actions, or ``""`` for global actions.
    Global actions: ``allon``, ``alloff``, ``engine``, ``cost``, ``recap``.
    Per-project actions: ``tog``, ``sel``, ``ptgl``, ``mode``, ``voice``,
    ``verb``, ``noop``.
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
    """Shorten a path under the current user's home dir to a ``~/`` prefix.

    Portable across hosts: reads ``Path.home()`` at call time rather than
    hardcoding a dev-machine path. Trailing-sep-safe (works whether
    ``Path.home()`` itself ends in ``/`` or not).
    """
    home = str(Path.home()).rstrip("/") + "/"
    if path.startswith(home):
        return "~/" + path[len(home):]
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
        verbose_label = "\U0001F527✓" if row.get("verbose") else "\U0001F527·"
        rows.append([
            InlineKeyboardButton(
                f"{dot} {proj}", callback_data=f"noop:{i}"),
            InlineKeyboardButton(
                on_label, callback_data=f"tog:{i}"),
            InlineKeyboardButton(
                f"{row['mode']} ▾", callback_data=f"mode:{i}"),
            InlineKeyboardButton(
                f"{row['voice']} ▾", callback_data=f"voice:{i}"),
            InlineKeyboardButton(
                verbose_label, callback_data=f"verb:{i}"),
        ])
    engine = snapshot[0]["engine"] if snapshot else "openai"
    rows.append([
        InlineKeyboardButton("▶ ALL ON", callback_data="allon"),
        InlineKeyboardButton("⏸ ALL OFF", callback_data="alloff"),
        InlineKeyboardButton(
            f"engine: {engine} ▾", callback_data="engine"),
    ])
    rows.append([
        InlineKeyboardButton("💰 Cost", callback_data="cost"),
        InlineKeyboardButton("🗒 Recap", callback_data="recap"),
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
