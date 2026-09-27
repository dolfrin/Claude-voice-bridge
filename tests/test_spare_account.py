"""Spare accounts: their allowance is used before it expires, by the bridge's
own sessions and by background workers, falling back to the main login."""

import io
import json
import time

from claude_agent_sdk import AssistantMessage, RateLimitEvent, ResultMessage, TextBlock
from claude_agent_sdk.types import RateLimitInfo

from voice_bridge import spare_account as sa
from voice_bridge import usage
from voice_bridge.sessions import Outbound

from test_sessions import (  # noqa: F401 - the autouse fixtures patch the SDK
    FakeClaudeSDKClient, FakeStore, _neutralize_catchup, _patch_sdk, _wait_for, make_cfg, make_project, make_sm,
)


def limit(status, resets_at=None, kind="five_hour", utilization=None, windows=None):
    return RateLimitInfo(status=status, resets_at=resets_at, rate_limit_type=kind,
                         utilization=utilization, raw={"unifiedWindows": windows} if windows else {})


def window(pct, reset):
    return {"utilization": pct, "resetsAt": reset}


def test_usable_until_used_up_then_again_after_the_reset(tmp_path):
    clock = {"t": 1000.0}
    spares = sa.Spares(tmp_path, clock=lambda: clock["t"])
    assert spares.pick() is None                                # none set up
    path = sa.save_token(tmp_path, "a", "sk-ant-oat01-x")
    assert path.stat().st_mode & 0o777 == 0o600
    assert spares.pick() == "a"

    assert spares.record_limit("a", limit("allowed_warning", 5000, utilization=0.8)) is False
    assert spares.record_limit("a", limit("rejected", 5000)) is True
    assert spares.pick() is None
    clock["t"] = 5000
    assert spares.pick() == "a"


def test_the_allowance_that_expires_first_is_used_first(tmp_path):
    spares = sa.Spares(tmp_path, clock=lambda: 1000.0)
    for name in ("a", "b", "c"):
        sa.save_token(tmp_path, name, "sk-ant-oat01-" + name)
    spares.record_limit("a", limit("allowed", windows={"seven_day": window(0.2, 900_000)}))
    spares.record_limit("b", limit("allowed", windows={"seven_day": window(0.5, 200_000)}))
    # c: never used, nothing known -- after the ones whose reset is known
    assert spares.pick() == "b"                 # resets soonest: use it before it is lost
    spares.record_limit("b", limit("rejected", 200_000, kind="seven_day"))
    assert spares.pick() == "a"
    spares.record_limit("a", limit("rejected", 900_000, kind="seven_day"))
    assert spares.pick() == "c"


def test_paid_overage_counts_as_used_up(tmp_path):
    spares = sa.Spares(tmp_path, clock=lambda: 1000.0)
    sa.save_token(tmp_path, "a", "sk-ant-oat01-x")
    assert spares.record_limit("a", limit("allowed", 9000, kind="overage")) is True
    assert spares.pick() is None


def test_the_first_single_account_layout_becomes_account_spare(tmp_path):
    (tmp_path / "claude-spare-token").write_text("sk-ant-oat01-old")
    (tmp_path / "claude-spare.json").write_text(json.dumps({"until": 99, "turns": [["s", 1, 2]]}))
    spares = sa.Spares(tmp_path, clock=lambda: 1000.0)
    assert spares.names() == ["spare"] and spares.token("spare") == "sk-ant-oat01-old"
    assert spares.account("spare")["until"] == 99
    assert sa.spans(tmp_path) == [("s", 1, 2)]
    assert not (tmp_path / "claude-spare-token").exists()


def test_the_token_is_found_in_wrapped_terminal_output():
    out = (b"\x1b[32mYour OAuth token:\x1b[0m\r\n"
           b"sk-ant-oat01-AbCdEfGhIjKlMnOp\r\nQrStUvWxYz_0123-456789\r\n\r\nStore it safely.")
    assert sa.capture_token(out) == "sk-ant-oat01-AbCdEfGhIjKlMnOpQrStUvWxYz_0123-456789"
    assert sa.capture_token(b"Login cancelled") is None


def test_a_worker_does_not_inherit_the_session_it_was_started_from():
    env = sa.worker_env("sk-ant-oat01-w", {
        "PATH": "/bin", "CLAUDE_CODE_SESSION_ID": "me", "CLAUDE_CODE_MESSAGING_SOCKET": "/s",
        "CLAUDE_CODE_ENTRYPOINT": "claude-vscode", "CLAUDECODE": "1", "ANTHROPIC_API_KEY": "k",
        "CLAUDE_CONFIG_DIR": "/cfg",
    })
    assert env == {"PATH": "/bin", "CLAUDE_CONFIG_DIR": "/cfg",
                   "CLAUDE_CODE_OAUTH_TOKEN": "sk-ant-oat01-w", "CLAUDE_CODE_ENTRYPOINT": "sdk-cli"}


class FakeProc:
    """A ``claude -p --output-format stream-json`` run, scripted."""

    def __init__(self, lines):
        self.stdout = io.StringIO("".join(json.dumps(x) + "\n" for x in lines))

    def wait(self):
        return 0


def test_worker_continues_on_the_next_account_when_one_runs_out(tmp_path):
    for name in ("a", "b"):
        sa.save_token(tmp_path, name, "sk-ant-oat01-" + name)
    spares = sa.Spares(tmp_path, clock=lambda: 1000.0)
    spares.record_limit("a", limit("allowed", windows={"seven_day": window(0.9, 5000)}))
    calls = []
    script = [
        [{"type": "rate_limit_event", "rate_limit_info": {"status": "rejected", "resetsAt": 5000,
                                                          "rateLimitType": "seven_day"}},
         {"type": "result", "session_id": "w1", "is_error": True, "result": "You've hit your limit"}],
        [{"type": "result", "session_id": "w1", "is_error": False, "result": "Done: 3 files"}],
    ]

    def spawn(cmd, **kw):
        calls.append((cmd, kw["env"]["CLAUDE_CODE_OAUTH_TOKEN"]))
        return FakeProc(script[len(calls) - 1])

    out = io.StringIO()
    code = sa.run("refactor x", "/p", tmp_path, spawn=spawn, clock=lambda: 1000.0, out=out)

    assert code == 0
    assert [token for _, token in calls] == ["sk-ant-oat01-a", "sk-ant-oat01-b"]
    assert calls[1][0][calls[1][0].index("--resume") + 1] == "w1"      # the same session
    assert "Done: 3 files" in out.getvalue() and "account=b session=w1" in out.getvalue()
    assert "--settings" in calls[0][0]                                   # hooks off: no Telegram noise


def test_worker_says_so_when_no_account_has_allowance(tmp_path):
    sa.save_token(tmp_path, "a", "sk-ant-oat01-a")
    sa.Spares(tmp_path, clock=lambda: 1000.0).record_limit("a", limit("rejected", 5000))
    out = io.StringIO()
    code = sa.run("x", "/p", tmp_path, spawn=lambda *a, **k: 1 / 0, clock=lambda: 1000.0, out=out)
    assert code == sa.NO_ACCOUNT and "main account" in out.getvalue()


async def test_bridge_session_runs_on_a_spare_account_then_moves_off_it(tmp_path):
    sa.save_token(tmp_path, "a", "sk-ant-oat01-spare")
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
    sa.Spares(tmp_path).record_turn("spare-sess", now - 120, now)

    sample = usage.take_sample(tmp_path, tmp_path / "ledger.jsonl", now)

    assert [x[0] for x in sample["turns"]] == ["main-sess"]


def test_usage_shows_each_spare_account_and_which_is_next(tmp_path):
    now = 1_790_496_000.0
    for name in ("a", "b"):
        sa.save_token(tmp_path, name, "sk-ant-oat01-" + name)
    spares = sa.Spares(tmp_path, clock=lambda: now)
    spares.record_limit("a", limit("rejected", int(now + 86400), kind="seven_day", windows={
        "five_hour": window(0, now + 3600), "seven_day": window(1, now + 86400)}))

    lines = [usage._spare_line(n, acc, now, n == "b") for n, acc in sa.status(tmp_path)]

    assert "a: išnaudota iki" in lines[0]
    assert "5 val. langas 0 %" in lines[0] and "Savaitė 100 %" in lines[0]
    assert lines[1].startswith("• b") and lines[1].endswith("← kita")
