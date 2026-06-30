import pytest

from voice_bridge.notify_tool import NOTIFY_TOOL_NAME, _build_notify_tool, make_notify_server


def test_notify_tool_name_is_fully_qualified():
    assert NOTIFY_TOOL_NAME == "mcp__bridge__notify_user"


def test_build_notify_tool_returns_sdk_tool_with_correct_name():
    async def spy(summary, detail):
        pass

    tool_obj = _build_notify_tool(spy)
    assert tool_obj.name == "notify_user"


def test_build_notify_tool_has_summary_detail_schema():
    async def spy(summary, detail):
        pass

    tool_obj = _build_notify_tool(spy)
    assert tool_obj.input_schema == {"summary": str, "detail": str}


@pytest.mark.asyncio
async def test_handler_invokes_callback_with_summary_and_detail():
    calls = []

    async def spy(summary, detail):
        calls.append((summary, detail))

    tool_obj = _build_notify_tool(spy)
    result = await tool_obj.handler({"summary": "Tests praėjo", "detail": "diff"})

    assert calls == [("Tests praėjo", "diff")]
    assert result["content"][0]["type"] == "text"
    assert result["content"][0]["text"] == "delivered"


@pytest.mark.asyncio
async def test_handler_no_summary_key_does_not_raise():
    calls = []

    async def spy(summary, detail):
        calls.append((summary, detail))

    tool_obj = _build_notify_tool(spy)
    result = await tool_obj.handler({})

    assert calls == [("", "")]
    assert result["content"][0]["text"] == "delivered"


@pytest.mark.asyncio
async def test_handler_defaults_detail_to_empty_string():
    calls = []

    async def spy(summary, detail):
        calls.append((summary, detail))

    tool_obj = _build_notify_tool(spy)
    await tool_obj.handler({"summary": "tests passed"})

    assert calls == [("tests passed", "")]


def test_make_notify_server_returns_non_none():
    async def spy(summary, detail):
        pass

    server = make_notify_server(spy)
    assert server is not None
