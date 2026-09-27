"""The spare account: the bridge's own sessions use a second subscription's
allowance and fall back to the main login when it runs out."""

import time

from claude_agent_sdk import AssistantMessage, RateLimitEvent, ResultMessage, TextBlock
from claude_agent_sdk.types import RateLimitInfo

from voice_bridge import spare_account as sa
from voice_bridge import usage
from voice_bridge.sessions import Outbound

from test_sessions import (  # noqa: F401 - the autouse fixtures patch the SDK
    FakeClaudeSDKClient, FakeStore, _neutralize_catchup, _patch_sdk, _wait_for, make_cfg, make_project, make_sm,
)


def limit(status, resets_at=None, kind="five_hour", utilization=None):
    return RateLimitInfo(status=status, resets_at=resets_at, rate_limit_type=kind, utilization=utilization)


def test_usable_until_used_up_then_again_after_the_reset(tmp_path):
    clock = {"t": 1000.0}
    spare = sa.SpareAccount(tmp_path, clock=lambda: clock["t"])
    assert not spare.usable()                                  # no token yet
    sa.save_token(tmp_path, "sk-ant-oat01-x")
    assert spare.usable() and (tmp_path / sa.TOKEN_FILE).stat().st_mode & 0o777 == 0o600

    assert spare.record_limit(limit("allowed_warning", 5000, utilization=0.8)) is False
    assert spare.usable()
    assert spare.record_limit(limit("rejected", 5000)) is True
    assert not spare.usable()
    clock["t"] = 5000
    assert spare.usable()


def test_paid_overage_counts_as_used_up(tmp_path):
    spare = sa.SpareAccount(tmp_path, clock=lambda: 1000.0)
    sa.save_token(tmp_path, "sk-ant-oat01-x")
    assert spare.record_limit(limit("allowed", 9000, kind="overage")) is True
    assert not spare.usable()


def test_a_failed_turn_without_a_reset_time_waits_an_hour(tmp_path):
    spare = sa.SpareAccount(tmp_path, clock=lambda: 1000.0)
    sa.save_token(tmp_path, "sk-ant-oat01-x")
    spare.mark_spent()
    assert sa.status(tmp_path)["until"] == 4600


def test_the_token_is_found_in_wrapped_terminal_output():
    out = (b"\x1b[32mYour OAuth token:\x1b[0m\r\n"
           b"sk-ant-oat01-AbCdEfGhIjKlMnOp\r\nQrStUvWxYz_0123-456789\r\n\r\nStore it safely.")
    assert sa.capture_token(out) == "sk-ant-oat01-AbCdEfGhIjKlMnOpQrStUvWxYz_0123-456789"
    assert sa.capture_token(b"Login cancelled") is None


async def test_bridge_session_runs_on_the_spare_account_then_moves_to_main(tmp_path):
    sa.save_token(tmp_path, "sk-ant-oat01-spare")
    outbound: list[Outbound] = []

    async def on_outbound(o):
        outbound.append(o)

    sm = make_sm([make_project("qwing")], FakeStore(enabled={"qwing": True}), on_outbound,
                 cfg=make_cfg(db_path=str(tmp_path / "state.db")))
    await sm.deliver("qwing", "do it")          # starts the session on first use
    first = FakeClaudeSDKClient.instances[0]
    assert first.options.env["CLAUDE_CODE_OAUTH_TOKEN"] == "sk-ant-oat01-spare"

    # (the scripted turn is set after connect; the queued turn waits for it)
    first.scripted_turns = [[
        # no "rejected" reading: only the failed turn says the limit is hit
        RateLimitEvent(rate_limit_info=limit("allowed"), uuid="u", session_id="s"),
        AssistantMessage(content=[TextBlock(text="You've hit your limit")], model="m", error="rate_limit"),
        ResultMessage(subtype="error", duration_ms=1, duration_api_ms=1, is_error=True,
                      num_turns=1, session_id="s", result="You've hit your limit"),
    ]]
    assert await _wait_for(lambda: len(FakeClaudeSDKClient.instances) == 2 and outbound)
    second = FakeClaudeSDKClient.instances[1]
    assert "CLAUDE_CODE_OAUTH_TOKEN" not in (second.options.env or {})   # the main login
    assert second.queries and second.queries[-1].endswith("do it")        # the same turn, again
    assert [o.text for o in outbound if "limit" in o.text] == []          # no limit error shown
    assert first.disconnected
    await sm.stop_all()


def test_usage_does_not_count_spare_turns_against_the_main_account(tmp_path, monkeypatch):
    now = time.time()
    turns = [("spare-sess", now - 60, "claude-x", 100.0, "/p"), ("main-sess", now - 60, "claude-x", 50.0, "/p")]
    monkeypatch.setattr(usage, "_turns", lambda root, since: turns)
    monkeypatch.setattr(usage, "fetch_limits", lambda home: {})
    monkeypatch.setattr(usage, "parse_limits", lambda data: [
        {"key": "session", "pct": 10.0, "reset": now + 3600, "span": 5 * 3600, "model": None}])
    monkeypatch.setattr(usage, "current_account", lambda home: ("acc", "a@x", ""))
    sa.SpareAccount(tmp_path).record_turn("spare-sess", now - 120, now)

    sample = usage.take_sample(tmp_path, tmp_path / "ledger.jsonl", now)

    assert [x[0] for x in sample["turns"]] == ["main-sess"]
