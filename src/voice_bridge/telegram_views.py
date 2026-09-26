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

from .config import AUTONOMY_MODES, EFFORT_LEVELS, TTS_BACKENDS
from .i18n import t
from .tts import available_voices

# Local aliases (list, not tuple) kept for minimal churn at call sites below;
# config.py is the single source of truth for the order and the members.
_MODES = list(AUTONOMY_MODES)
_ENGINES = list(TTS_BACKENDS)
_EFFORTS = list(EFFORT_LEVELS)
_COMMAND_NAMES = (
    "menu",
    "panel",
    "projects",
    "projects_all",
    "projects_refresh",
    "newproject",
    "handoff",
    "status",
    "info",
    "on",
    "off",
    "stop",
    "mode",
    "effort",
    "voice",
    "verbose",
    "engine",
    "agent",
    "recap",
    "usage",
    "policies",
    "schedule",
    "help",
    "live",
)


def bot_commands() -> list[BotCommand]:
    """The command menu, described in the bot's language."""
    return [BotCommand(name, t(f"cmd.{name}")) for name in _COMMAND_NAMES]

# A scheduled prompt can be arbitrarily long; the plain-text listing truncates
# it so one runaway schedule cannot blow past Telegram's message limit.
_SCHEDULE_PROMPT_MAX = 80


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


# Telegram rejects a message over 4096 characters or ~100 buttons outright, so
# a list that grows with the project count must be paged: at 50+ projects the
# full list (and the old all-projects panel) silently never arrived.
_PAGE_SIZE = 15


def _paged(rows: list, page: int) -> tuple[list, int, int]:
    """``(rows_on_page, page, page_count)`` with *page* clamped into range."""
    pages = max(1, -(-len(rows) // _PAGE_SIZE))
    page = min(max(page, 0), pages - 1)
    return rows[page * _PAGE_SIZE:(page + 1) * _PAGE_SIZE], page, pages


def format_projects(
    snapshot: list[dict], show_all: bool = False, page: int = 0,
    open_projects: set[str] | None = None,
) -> str:
    """Render /projects as a scannable HTML summary.

    Each project shows what to type to reach it and whether a session for it
    is open in the editor right now (then messages go straight in there)."""
    rows = _project_list_rows(snapshot, show_all=show_all)
    if not rows:
        return t("projects.none_active")
    rows, page, pages = _paged(rows, page)

    lines: list[str] = [t("projects.page", page=page + 1, pages=pages), ""] if pages > 1 else []
    for _idx, row in rows:
        status = "\U0001F7E2" if row["enabled"] else "\u26AA"
        active = " \u2B50" if row.get("last_active") else ""
        project = html.escape(row.get("display_name") or row["project"])
        cwd = _friendly_path(row.get("cwd") or "")
        path_part = html.escape(cwd) if cwd else "-"
        settings = html.escape(
            f"{row['mode']} · {row['voice']} · {row['engine']}"
        )
        where = (
            t("projects.open_in_editor") if row["project"] in (open_projects or set())
            else t("projects.bridge_session")
        )
        lines.extend([
            f"{status} <b>{project}</b>{active} — {where}",
            "  " + t("projects.address", name=html.escape(row["project"])),
            f"  \U0001F4C1 {path_part} · {settings}",
            "",
        ])
    return "\n".join(lines).strip()


def build_projects_list_markup(
    snapshot: list[dict], show_all: bool = False, page: int = 0
) -> InlineKeyboardMarkup:
    """Project picker with separate select-target and on/off controls."""
    rows: list[list[InlineKeyboardButton]] = []
    listed, page, pages = _paged(_project_list_rows(snapshot, show_all=show_all), page)
    for idx, row in listed:
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
    if pages > 1:
        nav = []
        if page > 0:
            nav.append(InlineKeyboardButton("◀", callback_data=f"menu:projects_all:{page - 1}"))
        nav.append(InlineKeyboardButton(f"{page + 1}/{pages}", callback_data="noop:"))
        if page < pages - 1:
            nav.append(InlineKeyboardButton("▶", callback_data=f"menu:projects_all:{page + 1}"))
        rows.append(nav)
    return InlineKeyboardMarkup(rows)


def build_menu_markup() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup([
        [
            InlineKeyboardButton(t("menu.active"), callback_data="menu:projects"),
            InlineKeyboardButton(t("menu.all"), callback_data="menu:projects_all"),
        ],
        [
            InlineKeyboardButton(t("menu.panel"), callback_data="menu:panel"),
            InlineKeyboardButton(t("menu.handoff"), callback_data="menu:handoff"),
        ],
        [
            InlineKeyboardButton(t("menu.stop"), callback_data="menu:stop"),
            InlineKeyboardButton(t("menu.refresh"), callback_data="menu:refresh"),
        ],
        [
            InlineKeyboardButton(t("menu.policies"), callback_data="menu:policies"),
        ],
    ])


def _format_policies(policies: list[tuple[str, str]]) -> str:
    """Render the always-allow policy list as a plain-text (no-HTML) message.

    Kept HTML-free so it can be sent with no ``parse_mode``: a signature like
    ``"echo > /etc/y"`` carries ``>``/``&`` metacharacters that would break an
    HTML-parsed send. Each line is ``• {project}: {signature}``.
    """
    if not policies:
        return (
            t("policies.none")
        )
    lines = [t("policies.title")]
    for project, signature in policies:
        lines.append(f"• {project}: {signature}")
    lines.append("")
    lines.append(t("policies.revoke_hint"))
    return "\n".join(lines)


def _format_schedules(schedules: list[dict]) -> str:
    """Render the /schedule listing as a plain-text (no-HTML) message.

    Kept HTML-free (sent with no ``parse_mode``) so a scheduled prompt carrying
    ``<``/``>``/``&`` can never break Telegram parsing. Each line is
    ``{id}  {project}  {HH:MM}  [off]  {prompt}`` with the ``[off]`` marker only
    for disabled schedules and the prompt truncated to keep the message bounded.
    """
    if not schedules:
        return (
            t("schedule.none")
        )
    lines = [t("schedule.title")]
    for s in schedules:
        prompt = str(s.get("prompt") or "")
        if len(prompt) > _SCHEDULE_PROMPT_MAX:
            prompt = prompt[:_SCHEDULE_PROMPT_MAX] + "…"
        off = " [off]" if not s.get("enabled", True) else ""
        lines.append(
            f"{s.get('id')}  {s.get('project')}  {s.get('hhmm')}{off}  {prompt}"
        )
    lines.append("")
    lines.append(t("schedule.hint"))
    return "\n".join(lines)


def _format_help() -> str:
    """Render the /help reference as a plain-text (no-HTML) message.

    Kept strictly HTML-free — no ``<``/``>``/``&`` — so it is sent with no
    ``parse_mode`` and nothing in it can ever be mis-parsed as markup. It
    documents the two things that are not discoverable from the command menu:
    how a message is ROUTED to a project, and how a phone reply ANSWERS an
    approval or question. The command list mirrors the registered commands, one
    line each.
    """
    return t("help.text")


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
    # Only running (or last-used) projects: five buttons per project for every
    # known project blew past Telegram's button limit and the panel never
    # opened. The rest are switched on from /projects_all.
    for i, row in enumerate(snapshot):
        if not (row["enabled"] or row.get("last_active")):
            continue
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
        InlineKeyboardButton(t("panel.all_on"), callback_data="allon"),
        InlineKeyboardButton(t("panel.all_off"), callback_data="alloff"),
        InlineKeyboardButton(
            t("panel.engine", engine=engine), callback_data="engine"),
    ])
    rows.append([
        InlineKeyboardButton(t("panel.limits"), callback_data="cost"),
        InlineKeyboardButton(t("panel.recap"), callback_data="recap"),
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
        [InlineKeyboardButton(t("panel.mode_of", project=row.get("display_name") or row["project"]), callback_data=f"noop:{idx}")],
        buttons,
        [InlineKeyboardButton(t("panel.back"), callback_data="back")],
    ])


def build_voice_markup(snapshot: list[dict], idx: int) -> InlineKeyboardMarkup:
    """Render explicit voice choices for one project."""
    row = snapshot[idx]
    voices = available_voices(row.get("engine", "openai"))
    rows: list[list[InlineKeyboardButton]] = [
        [InlineKeyboardButton(t("panel.voice_of", project=row.get("display_name") or row["project"]), callback_data=f"noop:{idx}")]
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
    rows.append([InlineKeyboardButton(t("panel.back"), callback_data="back")])
    return InlineKeyboardMarkup(rows)
