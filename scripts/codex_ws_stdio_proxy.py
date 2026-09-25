#!/usr/bin/env python3
"""Bridge Codex app-server JSONL stdio to its Unix WebSocket endpoint."""

from __future__ import annotations

import asyncio
import os
import sys
from urllib.parse import unquote, urlparse

from websockets.asyncio.client import unix_connect

# Keep in sync with ``voice_bridge.codex_app_server.MAX_FRAME_BYTES``: the shared
# endpoint pushes whole thread payloads, and the ``websockets`` 1 MiB default
# closes the proxy with 1009 once a thread grows past it.
MAX_FRAME_BYTES = 64 * 1024 * 1024


def socket_path() -> str:
    endpoint = os.environ.get("CODEX_APP_SERVER_URL", "").strip()
    if not endpoint:
        runtime = os.environ.get("XDG_RUNTIME_DIR") or f"/run/user/{os.getuid()}"
        endpoint = f"unix://{runtime}/codex-shared/app-server.sock"
    parsed = urlparse(endpoint)
    if parsed.scheme != "unix" or parsed.netloc or not parsed.path:
        raise ValueError("CODEX_APP_SERVER_URL must be unix:///absolute/path")
    path = unquote(parsed.path)
    if not os.path.isabs(path):
        raise ValueError("Codex app-server socket path must be absolute")
    return path


async def stdin_to_socket(socket) -> None:
    while True:
        line = await asyncio.to_thread(sys.stdin.buffer.readline)
        if not line:
            await socket.close()
            return
        await socket.send(line.decode("utf-8").rstrip("\r\n"))


async def socket_to_stdout(socket) -> None:
    async for message in socket:
        if isinstance(message, bytes):
            message = message.decode("utf-8")
        sys.stdout.write(message + "\n")
        sys.stdout.flush()


async def main() -> None:
    async with unix_connect(
        socket_path(),
        uri="ws://localhost/",
        compression=None,
        max_size=MAX_FRAME_BYTES,
    ) as socket:
        tasks = {
            asyncio.create_task(stdin_to_socket(socket)),
            asyncio.create_task(socket_to_stdout(socket)),
        }
        done, pending = await asyncio.wait(
            tasks, return_when=asyncio.FIRST_COMPLETED
        )
        for task in pending:
            task.cancel()
        await asyncio.gather(*pending, return_exceptions=True)
        for task in done:
            task.result()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
    except Exception as exc:
        print(f"codex shared proxy failed: {exc}", file=sys.stderr)
        raise SystemExit(1)
