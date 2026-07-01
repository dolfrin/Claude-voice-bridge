"""In-process Claude Agent SDK MCP server exposing bridge tools.

This lets a running agent ping the user mid-turn and send project-local files
back to Telegram. Tool callbacks are injected by the bridge and stay async, so
the single event loop is never stalled.
"""

from __future__ import annotations

from typing import Awaitable, Callable

from claude_agent_sdk import create_sdk_mcp_server, tool

# Server name "bridge" + tool name "notify_user" => SDK fully-qualified name.
NOTIFY_TOOL_NAME = "mcp__bridge__notify_user"
SEND_FILE_TOOL_NAME = "mcp__bridge__send_file"
ASK_USER_TOOL_NAME = "mcp__bridge__ask_user"


def _build_notify_tool(on_notify: Callable[[str, str], Awaitable[None]]):
    """Build the ``notify_user`` tool bound to *on_notify*.

    Returns:
        An ``SdkMcpTool`` instance ready for use with ``create_sdk_mcp_server``.
    """

    @tool(
        "notify_user",
        "Send a short status/question to the user. 'summary' is spoken aloud (no code); 'detail' is text-only.",
        {"summary": str, "detail": str},
    )
    async def notify_user(args: dict) -> dict:
        summary = args.get("summary", "")
        detail = args.get("detail", "")
        await on_notify(summary, detail)
        return {"content": [{"type": "text", "text": "delivered"}]}

    return notify_user


def _build_send_file_tool(on_send_file: Callable[[str, str], Awaitable[str]]):
    """Build the ``send_file`` tool bound to *on_send_file*."""

    @tool(
        "send_file",
        "Send a project-local file to the user on Telegram. 'path' must be inside the project directory; 'caption' is optional.",
        {"path": str, "caption": str},
    )
    async def send_file(args: dict) -> dict:
        path = args.get("path", "")
        caption = args.get("caption", "")
        result = await on_send_file(path, caption)
        return {"content": [{"type": "text", "text": result}]}

    return send_file


def _build_ask_user_tool(on_ask_user: Callable[[str, list[str]], Awaitable[str]]):
    """Build the ``ask_user`` tool bound to *on_ask_user*."""

    @tool(
        "ask_user",
        "Ask the user a question on Telegram with tappable choices. 'choices' must be a short list of button labels.",
        {"question": str, "choices": list},
    )
    async def ask_user(args: dict) -> dict:
        question = args.get("question", "")
        raw_choices = args.get("choices", [])
        choices = [str(choice) for choice in raw_choices] if isinstance(raw_choices, list) else []
        result = await on_ask_user(str(question), choices)
        return {"content": [{"type": "text", "text": result}]}

    return ask_user


def make_notify_server(
    on_notify: Callable[[str, str], Awaitable[None]],
    on_send_file: Callable[[str, str], Awaitable[str]] | None = None,
    on_ask_user: Callable[[str, list[str]], Awaitable[str]] | None = None,
):
    """Build the in-process MCP server exposing bridge tools.

    Args:
        on_notify: async callback invoked as ``on_notify(summary, detail)`` each
            time the agent calls the tool.
        on_send_file: async callback invoked as ``on_send_file(path, caption)``.
        on_ask_user: async callback invoked as ``on_ask_user(question, choices)``.

    Returns:
        The SDK MCP server config (from ``create_sdk_mcp_server``) for use in
        ``ClaudeAgentOptions.mcp_servers``.
    """
    tools = [_build_notify_tool(on_notify)]
    if on_send_file is not None:
        tools.append(_build_send_file_tool(on_send_file))
    if on_ask_user is not None:
        tools.append(_build_ask_user_tool(on_ask_user))
    return create_sdk_mcp_server("bridge", tools=tools)
