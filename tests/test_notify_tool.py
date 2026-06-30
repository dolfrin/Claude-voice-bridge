import sys
import types

import pytest

# --- Install a fake `claude_agent_sdk` module BEFORE importing notify_tool. ---
# `tool` records each decorated handler so the test can invoke it directly.
# `create_sdk_mcp_server` records its kwargs and returns a sentinel object.

_REGISTERED: dict[str, dict] = {}


def _fake_tool(name, description, input_schema, annotations=None):
    def _decorator(handler):
        _REGISTERED[name] = {
            "name": name,
            "description": description,
            "input_schema": input_schema,
            "annotations": annotations,
            "handler": handler,
        }
        return handler

    return _decorator


_CREATED: dict[str, object] = {}


def _fake_create_sdk_mcp_server(name, version="1.0.0", tools=None):
    server = object()
    _CREATED["last"] = {
        "name": name,
        "version": version,
        "tools": tools,
        "server": server,
    }
    return server


_fake_sdk = types.ModuleType("claude_agent_sdk")
_fake_sdk.tool = _fake_tool
_fake_sdk.create_sdk_mcp_server = _fake_create_sdk_mcp_server

# Save real SDK, install fake only for notify_tool import, then restore real SDK
# so downstream tests that need the real claude_agent_sdk are not polluted.
_real_sdk = sys.modules.get("claude_agent_sdk")
sys.modules["claude_agent_sdk"] = _fake_sdk

from voice_bridge import notify_tool  # noqa: E402

if _real_sdk is not None:
    sys.modules["claude_agent_sdk"] = _real_sdk
else:
    del sys.modules["claude_agent_sdk"]


@pytest.fixture(autouse=True)
def _reset_registry():
    _REGISTERED.clear()
    _CREATED.clear()
    yield
    _REGISTERED.clear()
    _CREATED.clear()


def test_notify_tool_name_is_fully_qualified():
    assert notify_tool.NOTIFY_TOOL_NAME == "mcp__bridge__notify_user"


def test_make_notify_server_registers_notify_user_on_bridge_server():
    async def on_notify(summary, detail):
        pass

    server = notify_tool.make_notify_server(on_notify)

    # Server identity matches what create_sdk_mcp_server returned.
    assert server is _CREATED["last"]["server"]
    # Server name must be "bridge" so the FQ tool name resolves correctly.
    assert _CREATED["last"]["name"] == "bridge"
    # The notify_user tool was registered with summary+detail in its schema.
    assert "notify_user" in _REGISTERED
    schema = _REGISTERED["notify_user"]["input_schema"]
    assert schema == {"summary": str, "detail": str}
    # The single registered tool was passed to create_sdk_mcp_server.
    assert _CREATED["last"]["tools"] == [_REGISTERED["notify_user"]["handler"]]


@pytest.mark.asyncio
async def test_handler_invokes_callback_with_summary_and_detail():
    calls = []

    async def on_notify(summary, detail):
        calls.append((summary, detail))

    notify_tool.make_notify_server(on_notify)
    handler = _REGISTERED["notify_user"]["handler"]

    result = await handler({"summary": "build done", "detail": "12 files changed"})

    assert calls == [("build done", "12 files changed")]
    assert result["content"][0]["type"] == "text"
    assert isinstance(result["content"][0]["text"], str)


@pytest.mark.asyncio
async def test_handler_defaults_detail_to_empty_string():
    calls = []

    async def on_notify(summary, detail):
        calls.append((summary, detail))

    notify_tool.make_notify_server(on_notify)
    handler = _REGISTERED["notify_user"]["handler"]

    await handler({"summary": "tests passed"})

    assert calls == [("tests passed", "")]
