"""In-process Claude Agent SDK MCP server exposing ``notify_user``.

This lets a running agent ping the user mid-turn. The single tool forwards its
arguments to an injected async ``on_notify(summary, detail)`` callback, which the
bridge wires to the Telegram + TTS outbound path. No blocking work happens here,
so the single event loop is never stalled.
"""

from __future__ import annotations

from typing import Awaitable, Callable

from claude_agent_sdk import create_sdk_mcp_server, tool

# Server name "bridge" + tool name "notify_user" => SDK fully-qualified name.
NOTIFY_TOOL_NAME = "mcp__bridge__notify_user"


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


def make_notify_server(on_notify: Callable[[str, str], Awaitable[None]]):
    """Build the in-process MCP server exposing ``notify_user``.

    Args:
        on_notify: async callback invoked as ``on_notify(summary, detail)`` each
            time the agent calls the tool.

    Returns:
        The SDK MCP server config (from ``create_sdk_mcp_server``) for use in
        ``ClaudeAgentOptions.mcp_servers``.
    """
    return create_sdk_mcp_server("bridge", tools=[_build_notify_tool(on_notify)])
