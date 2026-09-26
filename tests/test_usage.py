"""Account-scoped Claude usage: this PC's part vs the account's total."""

import json
import os
from datetime import datetime, timezone

import pytest

import voice_bridge.usage as usage

HOUR = 3600


def _iso(ts):
    return datetime.fromtimestamp(ts, timezone.utc).isoformat()


def _turn(ts, msg_id, out_tokens, cwd="/nowhere/proj"):
    return json.dumps({
        "type": "assistant", "timestamp": _iso(ts).replace("+00:00", "Z"),
        "requestId": "r" + msg_id, "cwd": cwd,
        "message": {"id": msg_id, "usage": {"output_tokens": out_tokens}},
    })


@pytest.fixture
def home(tmp_path):
    (tmp_path / ".claude" / "projects" / "p").mkdir(parents=True)
    (tmp_path / ".claude" / ".credentials.json").write_text(
        json.dumps({"claudeAiOauth": {"accessToken": "x"}}))
    return tmp_path


def _login(home, account, at):
    (home / ".claude.json").write_text(json.dumps({"oauthAccount": {
        "accountUuid": account, "emailAddress": f"{account}@x",
        "organizationRateLimitTier": "default_claude_max_20x"}}))
    os.utime(home / ".claude" / ".credentials.json", (at, at))


def _limits(monkeypatch, u5, u7, now):
    monkeypatch.setattr(usage, "fetch_limits", lambda home: {
        "five_hour": {"utilization": u5, "resets_at": _iso(now + 4 * HOUR)},
        "seven_day": {"utilization": u7, "resets_at": _iso(now + 6 * 24 * HOUR)},
    })


def test_other_accounts_turns_are_not_counted_and_estimate_waits(home, monkeypatch):
    now = 1_800_000_000.0
    transcript = home / ".claude" / "projects" / "p" / "s1.jsonl"
    # 200 tokens before the login (another account), 100 after.
    transcript.write_text(
        _turn(now - 50 * 60, "a", 40) + "\n" + _turn(now - 10 * 60, "b", 20) + "\n")
    os.utime(transcript, (now, now))
    _login(home, "acc-B", now - 30 * 60)
    _limits(monkeypatch, 2, 30, now)
    ledger = home / "ledger.jsonl"

    sample = usage.take_sample(home, ledger, now)
    assert sample["l5"] == 100.0  # 20 output tokens * 5, the pre-login 40 excluded

    text = usage.format_usage(ledger, home, now)
    assert "acc-B@x (max 20x)" in text
    assert "Bendrai paskyroj (visi įrenginiai): 2 %" in text  # account total, exact
    assert "(prisijungimo):" in text  # the PC's list starts at the login, not the window
    assert "sužinosiu nuo kito lango" in text  # window began before the login


def test_estimate_after_calibration_is_capped_by_account_total(home, monkeypatch):
    now = 1_800_000_000.0
    transcript = home / ".claude" / "projects" / "p" / "s1.jsonl"
    transcript.write_text(_turn(now - 10 * 60, "a", 200) + "\n")  # 1000 weighted
    os.utime(transcript, (now, now))
    _login(home, "acc-A", now - 30 * 24 * HOUR)  # logged in long before both windows
    ledger = home / "ledger.jsonl"

    # First sample: this PC alone used 1000 tokens = 10 %  ->  0.01 %/token.
    _limits(monkeypatch, 10, 10, now)
    usage.take_sample(home, ledger, now)
    # Later: another device joined; account at 40 %, this PC still 1000 tokens.
    _limits(monkeypatch, 40, 40, now + 60)
    text = usage.format_usage(ledger, home, now + 60)

    assert "Bendrai paskyroj (visi įrenginiai): 40 %" in text
    assert "Šis PC: ≈ 10 %" in text
    assert "(lango pradžios):" in text


def test_switching_account_restarts_attribution(home, monkeypatch):
    now = 1_800_000_000.0
    ledger = home / "ledger.jsonl"
    _login(home, "acc-A", now - 3 * HOUR)
    _limits(monkeypatch, 1, 1, now)
    first = usage.take_sample(home, ledger, now)
    _login(home, "acc-B", now + 600)
    second = usage.take_sample(home, ledger, now + 900)

    assert first["since"] == now - 3 * HOUR
    assert second["since"] == now + 600
