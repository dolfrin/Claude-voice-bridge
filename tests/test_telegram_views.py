"""Sanity checks for the telegram_io -> telegram_views split.

These guard the extraction itself (importability, no cycle, and the
telegram_io re-export surface), not the view logic — that stays covered by
tests/test_telegram_io.py.
"""

from telegram import InlineKeyboardMarkup

from voice_bridge import telegram_io, telegram_views

# Names moved to telegram_views and re-exported from telegram_io.
_REEXPORTED = [
    "build_panel_markup",
    "build_projects_list_markup",
    "build_menu_markup",
    "build_mode_markup",
    "build_voice_markup",
    "parse_callback",
    "format_projects",
    "_project_list_rows",
    "_friendly_path",
    "_find_project_row",
    "_tail_for_telegram",
    "_clean_choices",
    "_MODES",
    "_ENGINES",
    "_BOT_COMMANDS",
]


def test_reexports_are_the_same_objects():
    for name in _REEXPORTED:
        assert getattr(telegram_io, name) is getattr(telegram_views, name), name


def test_views_module_has_no_cycle_back_to_telegram_io():
    # telegram_views must stay independent of telegram_io.
    assert "telegram_io" not in telegram_views.__dict__


def test_pure_helpers_smoke():
    assert telegram_views.parse_callback("tog:3") == ("tog", "3")
    assert telegram_views.parse_callback("allon") == ("allon", "")
    snapshot = [{
        "project": "demo",
        "display_name": "demo",
        "enabled": True,
        "mode": "safe",
        "voice": "alloy",
        "engine": "openai",
        "last_active": False,
        "cwd": "",
        "verbose": False,
    }]
    assert isinstance(telegram_views.build_panel_markup(snapshot), InlineKeyboardMarkup)
    assert "demo" in telegram_views.format_projects(snapshot)
