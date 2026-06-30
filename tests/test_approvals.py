"""TDD tests for voice_bridge.approvals — risk classification, yes/no parsing,
ApprovalManager, and make_can_use_tool."""

import asyncio

import pytest

from voice_bridge.approvals import (
    ApprovalManager,
    is_risky,
    make_can_use_tool,
    parse_yes_no,
)
from voice_bridge.config import Config, ProjectConfig


CWD = "/home/home/Projects/qwing"


def _cfg(autonomy_mode: str = "safe", approval_timeout: int = 300) -> Config:
    return Config(
        telegram_bot_token="t",
        telegram_allowed_user_id=1,
        anthropic_api_key="a",
        openai_api_key="o",
        tts_backend="openai",
        tts_voice="nova",
        piper_voice_path="/x.onnx",
        whisper_model="large-v3",
        autonomy_mode=autonomy_mode,
        approval_timeout=approval_timeout,
        db_path=":memory:",
    )


def _proj(autonomy=None) -> ProjectConfig:
    return ProjectConfig(name="qwing", cwd=CWD, autonomy=autonomy)


# ---------------------------------------------------------------------------
# is_risky — True cases
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "tool_name,tool_input",
    [
        ("Bash", {"command": "git push origin master"}),
        ("Bash", {"command": "rm -rf build"}),
        ("Bash", {"command": "ssh root@server uptime"}),
        ("Bash", {"command": "npm install left-pad"}),
        ("Bash", {"command": "pip install requests"}),
        ("Bash", {"command": "kubectl apply -f deploy.yaml"}),
        ("Bash", {"command": "vercel deploy"}),
        ("Bash", {"command": "send 0.5 ETH to my wallet"}),
        ("Write", {"file_path": "/etc/hosts", "content": "x"}),
        ("Edit", {"file_path": "/home/home/Projects/other/a.py"}),
    ],
)
def test_is_risky_true(tool_name, tool_input):
    assert is_risky(tool_name, tool_input, CWD) is True


# ---------------------------------------------------------------------------
# is_risky — False cases
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "tool_name,tool_input",
    [
        ("Read", {"file_path": "/home/home/Projects/qwing/a.py"}),
        ("Grep", {"pattern": "def main"}),
        ("Bash", {"command": "pytest -q"}),
        ("Bash", {"command": "ls -la"}),
        ("Bash", {"command": "git status"}),
        ("Bash", {"command": "npm run build"}),
        ("Edit", {"file_path": "/home/home/Projects/qwing/src/a.py"}),
        ("Write", {"file_path": "/home/home/Projects/qwing/new.py", "content": "x"}),
    ],
)
def test_is_risky_false(tool_name, tool_input):
    assert is_risky(tool_name, tool_input, CWD) is False


# ---------------------------------------------------------------------------
# parse_yes_no
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "text",
    ["taip", "Taip!", "  jo ", "davai", "gerai", "ok", "okay", "yes", "yep", "y", "sure", "varom"],
)
def test_parse_yes_no_true(text):
    assert parse_yes_no(text) is True


@pytest.mark.parametrize(
    "text",
    ["ne", "Ne.", "stop", "no", "nope", "n", "atšauk", "neleisk"],
)
def test_parse_yes_no_false(text):
    assert parse_yes_no(text) is False


@pytest.mark.parametrize(
    "text",
    ["", "   ", "gal but", "what do you mean", "kažkas neaiškaus", "run the tests first"],
)
def test_parse_yes_no_none(text):
    assert parse_yes_no(text) is None


# ---------------------------------------------------------------------------
# ApprovalManager
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_approval_manager_approve():
    sent: list[tuple[str, str]] = []

    async def send_question(project: str, text: str) -> int:
        sent.append((project, text))
        return 42

    mgr = ApprovalManager(send_question, timeout=5)

    async def approve_soon():
        await asyncio.sleep(0)
        assert mgr.has_pending(42) is True
        assert mgr.resolve(42, True) is True

    approver = asyncio.create_task(approve_soon())
    result = await mgr.request("qwing", "Bash", {"command": "git push"})
    await approver

    assert result is True
    assert sent and sent[0][0] == "qwing"
    assert "git push" in sent[0][1]
    assert mgr.has_pending(42) is False


@pytest.mark.asyncio
async def test_approval_manager_deny():
    async def send_question(project: str, text: str) -> int:
        return 7

    mgr = ApprovalManager(send_question, timeout=5)

    async def deny_soon():
        await asyncio.sleep(0)
        assert mgr.resolve(7, False) is True

    asyncio.create_task(deny_soon())
    result = await mgr.request("qwing", "Bash", {"command": "rm -rf x"})
    assert result is False


@pytest.mark.asyncio
async def test_approval_manager_timeout_denies():
    async def send_question(project: str, text: str) -> int:
        return 99

    mgr = ApprovalManager(send_question, timeout=0.05)
    result = await mgr.request("qwing", "Bash", {"command": "git push"})
    assert result is False
    assert mgr.has_pending(99) is False


@pytest.mark.asyncio
async def test_resolve_unknown_returns_false():
    async def send_question(project: str, text: str) -> int:
        return 1

    mgr = ApprovalManager(send_question, timeout=5)
    assert mgr.resolve(123456, True) is False
    assert mgr.has_pending(123456) is False


# ---------------------------------------------------------------------------
# make_can_use_tool
# ---------------------------------------------------------------------------

class _FakeManager:
    """Records request() calls; returns a preset decision."""

    def __init__(self, decision: bool):
        self.decision = decision
        self.calls: list[tuple[str, str, dict]] = []

    async def request(self, project: str, tool_name: str, tool_input: dict) -> bool:
        self.calls.append((project, tool_name, tool_input))
        return self.decision


def _decision_kind(result) -> str:
    return type(result).__name__


@pytest.mark.asyncio
async def test_can_use_tool_full_allows_without_asking():
    mgr = _FakeManager(decision=False)
    fn = make_can_use_tool(_proj(autonomy="full"), _cfg(), mgr)
    result = await fn("Bash", {"command": "git push origin master"}, None)
    assert _decision_kind(result) == "PermissionResultAllow"
    assert mgr.calls == []


@pytest.mark.asyncio
async def test_can_use_tool_ask_requests_even_safe():
    mgr = _FakeManager(decision=True)
    fn = make_can_use_tool(_proj(autonomy="ask"), _cfg(), mgr)
    result = await fn("Read", {"file_path": f"{CWD}/a.py"}, None)
    assert _decision_kind(result) == "PermissionResultAllow"
    assert len(mgr.calls) == 1


@pytest.mark.asyncio
async def test_can_use_tool_ask_deny_maps_to_deny():
    mgr = _FakeManager(decision=False)
    fn = make_can_use_tool(_proj(autonomy="ask"), _cfg(), mgr)
    result = await fn("Read", {"file_path": f"{CWD}/a.py"}, None)
    assert _decision_kind(result) == "PermissionResultDeny"
    assert len(mgr.calls) == 1


@pytest.mark.asyncio
async def test_can_use_tool_safe_allows_safe_without_asking():
    mgr = _FakeManager(decision=False)
    fn = make_can_use_tool(_proj(autonomy="safe"), _cfg(), mgr)
    result = await fn("Read", {"file_path": f"{CWD}/a.py"}, None)
    assert _decision_kind(result) == "PermissionResultAllow"
    assert mgr.calls == []


@pytest.mark.asyncio
async def test_can_use_tool_safe_asks_for_risky():
    mgr = _FakeManager(decision=True)
    fn = make_can_use_tool(_proj(autonomy="safe"), _cfg(), mgr)
    result = await fn("Bash", {"command": "git push origin master"}, None)
    assert _decision_kind(result) == "PermissionResultAllow"
    assert len(mgr.calls) == 1
    assert mgr.calls[0][0] == "qwing"


@pytest.mark.asyncio
async def test_can_use_tool_uses_project_autonomy_over_global():
    # global cfg is full, but project override is safe -> risky must be asked
    mgr = _FakeManager(decision=False)
    fn = make_can_use_tool(_proj(autonomy="safe"), _cfg(autonomy_mode="full"), mgr)
    result = await fn("Bash", {"command": "rm -rf build"}, None)
    assert _decision_kind(result) == "PermissionResultDeny"
    assert len(mgr.calls) == 1
