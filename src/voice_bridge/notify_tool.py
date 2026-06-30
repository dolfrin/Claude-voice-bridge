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


def make_notify_server(on_notify: Callable[[str, str], Awaitable[None]]):
    """Build the in-process MCP server exposing ``notify_user``.

    Args:
        on_notify: async callback invoked as ``on_notify(summary, detail)`` each
            time the agent calls the tool.

    Returns:
        The SDK MCP server config (from ``create_sdk_mcp_server``) for use in
        ``ClaudeAgentOptions.mcp_servers``.
    """

    @tool(
        "notify_user",
        "Send the user a short status update mid-turn. "
        "summary is a one-line spoken-friendly message; detail is optional "
        "longer context shown in text.",
        {"summary": str, "detail": str},
    )
    async def notify_user(args: dict) -> dict:
        summary = args["summary"]
        detail = args.get("detail", "")
        await on_notify(summary, detail)
        return {"content": [{"type": "text", "text": "Notification sent to user."}]}

    return create_sdk_mcp_server(
        name="bridge",
        version="1.0.0",
        tools=[notify_user],
    )
