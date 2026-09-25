from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
import sys

import pytest
from websockets.asyncio.server import unix_serve

from voice_bridge.codex_app_server import (
    CodexAppServerClient,
    CodexAppServerError,
    CodexRequestError,
    build_codex_env,
)


_SERVER = r'''
import asyncio
import json
import os
import sys

async def send(value):
    sys.stdout.write(json.dumps(value, separators=(",", ":")) + "\n")
    sys.stdout.flush()

async def main():
    while True:
        line = await asyncio.to_thread(sys.stdin.readline)
        if not line:
            return
        request = json.loads(line)
        method = request.get("method")
        ident = request.get("id")
        params = request.get("params", {})
        if method == "initialize":
            await send({"jsonrpc":"2.0","id":ident,"result":{
                "userAgent":"fake","codexHome":"/tmp/codex","platformFamily":"unix","platformOs":"linux"
            }})
        elif method == "initialized":
            continue
        elif method == "echo":
            if params.get("delay"):
                await asyncio.sleep(params["delay"])
            await send({"jsonrpc":"2.0","id":ident,"result":params})
        elif method == "error":
            await send({"jsonrpc":"2.0","id":ident,"error":{"code":123,"message":"bad"}})
        elif method == "notify":
            await send({"jsonrpc":"2.0","method":"turn/started","params":{"turn":params}})
            await send({"jsonrpc":"2.0","id":ident,"result":{}})
        elif method == "ask-client":
            await send({"jsonrpc":"2.0","id":"server-1","method":params["method"],"params":params.get("payload",{})})
        elif ident == "server-1":
            await send({"jsonrpc":"2.0","id":1_000_000,"result":request})
        elif method == "ask-roundtrip":
            await send({"jsonrpc":"2.0","id":"server-1","method":params["method"],"params":params.get("payload",{})})
            response_line = await asyncio.to_thread(sys.stdin.readline)
            response = json.loads(response_line)
            await send({"jsonrpc":"2.0","id":ident,"result":response})
        elif method == "malformed":
            sys.stdout.write("not-json\n")
            sys.stdout.flush()
            await send({"jsonrpc":"2.0","id":ident,"result":{"ok":True}})
        elif method == "stderr":
            sys.stderr.write("token=super-secret-value\n")
            sys.stderr.flush()
            await send({"jsonrpc":"2.0","id":ident,"result":{}})
        elif method == "hang":
            continue
        elif method == "exit":
            return
        elif method == "env":
            await send({"jsonrpc":"2.0","id":ident,"result":dict(os.environ)})

asyncio.run(main())
'''


@pytest.fixture
def fake_server(tmp_path: Path) -> tuple[str, ...]:
    path = tmp_path / "fake_app_server.py"
    path.write_text(_SERVER)
    return (sys.executable, str(path))


@pytest.mark.asyncio
async def test_start_initializes_and_close_is_idempotent(fake_server):
    client = CodexAppServerClient(command=fake_server)
    result = await client.start()
    assert result["userAgent"] == "fake"
    assert client.running
    assert client.pid is not None
    await client.close()
    await client.close()
    assert not client.running


@pytest.mark.asyncio
async def test_request_and_error_response(fake_server):
    client = CodexAppServerClient(command=fake_server)
    await client.start()
    assert await client.request("echo", {"value": 7}) == {"value": 7}
    with pytest.raises(CodexRequestError) as error:
        await client.request("error")
    assert error.value.code == 123
    await client.close()


@pytest.mark.asyncio
async def test_concurrent_requests_are_correlated(fake_server):
    client = CodexAppServerClient(command=fake_server)
    await client.start()
    slow = asyncio.create_task(client.request("echo", {"value": "slow", "delay": 0.04}))
    fast = asyncio.create_task(client.request("echo", {"value": "fast"}))
    assert (await fast)["value"] == "fast"
    assert (await slow)["value"] == "slow"
    await client.close()


@pytest.mark.asyncio
async def test_notification_handler_receives_event(fake_server):
    seen = []
    ready = asyncio.Event()

    async def handler(method, params):
        seen.append((method, params))
        ready.set()

    client = CodexAppServerClient(command=fake_server)
    client.add_notification_handler(handler)
    await client.start()
    await client.request("notify", {"id": "turn-1"})
    await asyncio.wait_for(ready.wait(), 1)
    assert seen == [("turn/started", {"turn": {"id": "turn-1"}})]
    await client.close()


@pytest.mark.asyncio
async def test_server_request_handler_returns_result(fake_server):
    client = CodexAppServerClient(command=fake_server)
    client.add_request_handler("approve", lambda params: {"decision": params["choice"]})
    await client.start()
    response = await client.request(
        "ask-roundtrip", {"method": "approve", "payload": {"choice": "accept"}}
    )
    assert response["id"] == "server-1"
    assert response["result"] == {"decision": "accept"}
    await client.close()


@pytest.mark.asyncio
async def test_unknown_server_request_is_method_not_found(fake_server):
    client = CodexAppServerClient(command=fake_server)
    await client.start()
    response = await client.request(
        "ask-roundtrip", {"method": "unknown", "payload": {}}
    )
    assert response["error"]["code"] == -32601
    await client.close()


@pytest.mark.asyncio
async def test_handler_error_is_redacted(fake_server):
    def fail(_params):
        raise RuntimeError("token=do-not-leak")

    client = CodexAppServerClient(command=fake_server)
    client.add_request_handler("approve", fail)
    await client.start()
    response = await client.request(
        "ask-roundtrip", {"method": "approve", "payload": {}}
    )
    assert "do-not-leak" not in response["error"]["message"]
    assert "[REDACTED]" in response["error"]["message"]
    await client.close()


@pytest.mark.asyncio
async def test_malformed_line_does_not_break_following_response(fake_server):
    client = CodexAppServerClient(command=fake_server)
    await client.start()
    assert await client.request("malformed") == {"ok": True}
    await client.close()


@pytest.mark.asyncio
async def test_timeout_and_cancellation_remove_pending(fake_server):
    client = CodexAppServerClient(command=fake_server)
    await client.start()
    with pytest.raises(CodexAppServerError, match="timed out"):
        await client.request("hang", timeout=0.02)
    task = asyncio.create_task(client.request("hang", timeout=10))
    await asyncio.sleep(0)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert client._pending == {}
    await client.close()


@pytest.mark.asyncio
async def test_initialization_timeout_closes_without_deadlock(fake_server):
    client = CodexAppServerClient(command=fake_server, request_timeout=0.000001)
    with pytest.raises(CodexAppServerError, match="timed out"):
        await asyncio.wait_for(client.start(), timeout=3)
    assert not client.running


@pytest.mark.asyncio
async def test_eof_fails_pending_request(fake_server):
    client = CodexAppServerClient(command=fake_server)
    await client.start()
    with pytest.raises(CodexAppServerError, match="stdout closed"):
        await client.request("exit", timeout=1)
    await client.close()


@pytest.mark.asyncio
async def test_stderr_is_bounded_and_redacted(fake_server):
    client = CodexAppServerClient(command=fake_server, stderr_lines=1)
    await client.start()
    await client.request("stderr")
    for _ in range(20):
        if client.stderr_tail:
            break
        await asyncio.sleep(0.01)
    assert len(client.stderr_tail) == 1
    assert "super-secret-value" not in client.stderr_tail[0]
    assert "[REDACTED]" in client.stderr_tail[0]
    await client.close()


def test_environment_is_allowlisted_and_has_path():
    source = {
        "HOME": "/home/test",
        "PATH": "/bin",
        "LANG": "lt_LT.UTF-8",
        "LC_TIME": "lt_LT.UTF-8",
        "TELEGRAM_BOT_TOKEN": "secret",
        "ANTHROPIC_API_KEY": "secret",
        "DEEPSEEK_API_KEY": "secret",
        "OPENAI_API_KEY": "secret",
        "UNRELATED": "value",
    }
    env = build_codex_env(source)
    assert env == {
        "HOME": "/home/test",
        "PATH": "/bin",
        "LANG": "lt_LT.UTF-8",
        "LC_TIME": "lt_LT.UTF-8",
    }
    assert build_codex_env({})["PATH"] == os.defpath


@pytest.mark.asyncio
async def test_child_receives_only_sanitized_environment(fake_server):
    client = CodexAppServerClient(
        command=fake_server,
        env={"HOME": "/tmp", "PATH": os.defpath, "TELEGRAM_BOT_TOKEN": "no"},
    )
    await client.start()
    env = await client.request("env")
    assert env["HOME"] == "/tmp"
    assert "TELEGRAM_BOT_TOKEN" not in env
    await client.close()


@pytest.mark.asyncio
async def test_missing_binary_is_actionable():
    client = CodexAppServerClient(command=("/definitely/missing/codex",))
    with pytest.raises(CodexAppServerError, match="could not start"):
        await client.start()


@pytest.mark.asyncio
async def test_shared_unix_websocket_transport(tmp_path):
    socket_path = tmp_path / "app-server.sock"

    async def handler(socket):
        async for raw in socket:
            request = json.loads(raw)
            if request.get("method") == "initialize":
                await socket.send(json.dumps({
                    "jsonrpc": "2.0",
                    "id": request["id"],
                    "result": {"userAgent": "shared-fake"},
                }))
            elif request.get("method") == "echo":
                await socket.send(json.dumps({
                    "jsonrpc": "2.0",
                    "id": request["id"],
                    "result": request.get("params", {}),
                }))

    async with unix_serve(handler, str(socket_path), compression=None):
        client = CodexAppServerClient(endpoint=f"unix://{socket_path}")
        result = await client.start()
        assert result["userAgent"] == "shared-fake"
        assert client.running
        assert client.pid is None
        assert await client.request("echo", {"shared": True}) == {"shared": True}
        await client.close()
        assert not client.running


@pytest.mark.asyncio
async def test_shared_endpoint_requires_absolute_unix_url():
    client = CodexAppServerClient(endpoint="tcp://localhost:1234")
    with pytest.raises(CodexAppServerError, match="unix:///absolute/path"):
        await client.start()


@pytest.mark.asyncio
async def test_shared_endpoint_receives_message_over_one_mib(tmp_path):
    """Regression: the websockets 1 MiB default killed the shared connection.

    codex-cli 0.154.0 was observed pushing one 7,654,245-byte message; the
    client then closed with 1009 (message too big) and every later turn failed
    with "app-server reader failed" until the service was restarted.
    """

    socket_path = tmp_path / "app-server.sock"
    blob_size = 2 * 1024 * 1024

    async def handler(socket):
        async for raw in socket:
            request = json.loads(raw)
            method = request.get("method")
            if method == "initialize":
                await socket.send(json.dumps({
                    "jsonrpc": "2.0",
                    "id": request["id"],
                    "result": {"userAgent": "shared-fake"},
                }))
            elif method == "big":
                await socket.send(json.dumps({
                    "jsonrpc": "2.0",
                    "id": request["id"],
                    "result": {"blob": "y" * request["params"]["size"]},
                }))
            elif method == "echo":
                await socket.send(json.dumps({
                    "jsonrpc": "2.0",
                    "id": request["id"],
                    "result": request.get("params", {}),
                }))

    async with unix_serve(
        handler, str(socket_path), compression=None, max_size=None
    ):
        client = CodexAppServerClient(endpoint=f"unix://{socket_path}")
        await client.start()
        result = await client.request("big", {"size": blob_size}, timeout=30)
        assert len(result["blob"]) == blob_size
        assert client.running
        assert await client.request("echo", {"still": "working"}) == {
            "still": "working"
        }
        await client.close()


@pytest.mark.asyncio
async def test_start_rebuilds_a_transport_that_died(fake_server):
    client = CodexAppServerClient(command=fake_server)
    await client.start()
    with pytest.raises(CodexAppServerError, match="stdout closed"):
        await client.request("exit", timeout=1)
    for _ in range(200):
        if not client.running:
            break
        await asyncio.sleep(0.01)
    assert not client.running

    await client.start()

    assert client.running
    assert await client.request("echo", {"value": "again"}) == {"value": "again"}
    await client.close()
