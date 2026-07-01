import pytest

from voice_bridge.notify_tool import (
    ASK_USER_TOOL_NAME,
    NOTIFY_TOOL_NAME,
    SEND_FILE_TOOL_NAME,
    _build_ask_user_tool,
    _build_notify_tool,
    _build_send_file_tool,
    make_notify_server,
)


def test_notify_tool_name_is_fully_qualified():
    assert NOTIFY_TOOL_NAME == "mcp__bridge__notify_user"
    assert SEND_FILE_TOOL_NAME == "mcp__bridge__send_file"
    assert ASK_USER_TOOL_NAME == "mcp__bridge__ask_user"


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


def test_build_send_file_tool_has_path_caption_schema():
    async def spy(path, caption):
        return "delivered"

    tool_obj = _build_send_file_tool(spy)
    assert tool_obj.name == "send_file"
    assert tool_obj.input_schema == {"path": str, "caption": str}


def test_build_ask_user_tool_has_question_choices_schema():
    async def spy(question, choices):
        return "A"

    tool_obj = _build_ask_user_tool(spy)
    assert tool_obj.name == "ask_user"
    assert tool_obj.input_schema == {"question": str, "choices": list}


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


@pytest.mark.asyncio
async def test_send_file_handler_invokes_callback():
    calls = []

    async def spy(path, caption):
        calls.append((path, caption))
        return "delivered"

    tool_obj = _build_send_file_tool(spy)
    result = await tool_obj.handler({"path": "dist/out.zip", "caption": "build"})

    assert calls == [("dist/out.zip", "build")]
    assert result["content"][0]["text"] == "delivered"


@pytest.mark.asyncio
async def test_send_file_handler_defaults_args_to_empty_string():
    calls = []

    async def spy(path, caption):
        calls.append((path, caption))
        return "denied"

    tool_obj = _build_send_file_tool(spy)
    result = await tool_obj.handler({})

    assert calls == [("", "")]
    assert result["content"][0]["text"] == "denied"


@pytest.mark.asyncio
async def test_ask_user_handler_invokes_callback():
    calls = []

    async def spy(question, choices):
        calls.append((question, choices))
        return "B"

    tool_obj = _build_ask_user_tool(spy)
    result = await tool_obj.handler({"question": "Rinktis?", "choices": ["A", "B"]})

    assert calls == [("Rinktis?", ["A", "B"])]
    assert result["content"][0]["text"] == "B"


@pytest.mark.asyncio
async def test_ask_user_handler_ignores_non_list_choices():
    calls = []

    async def spy(question, choices):
        calls.append((question, choices))
        return ""

    tool_obj = _build_ask_user_tool(spy)
    await tool_obj.handler({"question": "?", "choices": "A,B"})

    assert calls == [("?", [])]


def test_make_notify_server_returns_non_none():
    async def spy(summary, detail):
        pass

    server = make_notify_server(spy)
    assert server is not None
