"""Small asyncio client for the Codex app-server JSON-RPC protocol.

Only the transport and lifecycle live here.  Project/thread semantics belong
to :mod:`voice_bridge.codex_sessions`.
"""

from __future__ import annotations

import asyncio
from collections import deque
import inspect
import json
import logging
import os
import re
from collections.abc import Awaitable, Callable, Mapping, Sequence
from typing import Any
from urllib.parse import unquote, urlparse

from websockets.asyncio.client import ClientConnection, unix_connect

logger = logging.getLogger(__name__)

JsonObject = dict[str, Any]
NotificationHandler = Callable[[str, JsonObject], Awaitable[None] | None]
RequestHandler = Callable[[JsonObject], Awaitable[Any] | Any]


class CodexAppServerError(RuntimeError):
    """Base error for app-server transport failures."""


class CodexProtocolError(CodexAppServerError):
    """The peer sent an invalid JSON-RPC envelope."""


class CodexRequestError(CodexAppServerError):
    """A JSON-RPC request completed with an error response."""

    def __init__(self, code: int | None, message: str, data: Any = None) -> None:
        super().__init__(f"app-server request failed ({code}): {message}")
        self.code = code
        self.message = message
        self.data = data


_ENV_ALLOWLIST = frozenset(
    {
        "HOME",
        "USER",
        "LOGNAME",
        "PATH",
        "SHELL",
        "TERM",
        "TMPDIR",
        "LANG",
        "LC_ALL",
        "LC_CTYPE",
        "XDG_CONFIG_HOME",
        "XDG_CACHE_HOME",
        "XDG_DATA_HOME",
        "CODEX_HOME",
    }
)

_SECRET_TEXT_RE = re.compile(
    r"(?i)(bearer\s+)[^\s]+|"
    r"((?:api[_-]?key|token|secret|password)\s*[=:]\s*)[^\s,;]+|"
    r"\bsk-[A-Za-z0-9_-]{8,}\b"
)

# Codex sends whole JSON-RPC messages over the shared WebSocket: a resume
# payload, a large thread item, a diff, or captured command output.  The
# ``websockets`` default of 1 MiB is far too small -- codex-cli 0.154.0 was
# observed sending a single 7,654,245-byte message, after which the local
# endpoint closed with 1009 (message too big) and every later turn failed with
# "app-server reader failed" until the service was restarted.  64 MiB keeps a
# bounded ceiling with head-room for large threads.
MAX_FRAME_BYTES = 64 * 1024 * 1024


def build_codex_env(source: Mapping[str, str] | None = None) -> dict[str, str]:
    """Return the small environment inherited by the Codex child.

    App secrets loaded for Telegram, speech providers, or model providers are
    deliberately not inherited.  Codex account authentication is discovered
    through ``HOME``/``CODEX_HOME`` instead.
    """

    values = os.environ if source is None else source
    env = {
        key: value
        for key, value in values.items()
        if key in _ENV_ALLOWLIST or key.startswith("LC_")
    }
    env.setdefault("PATH", os.defpath)
    return env


def _redact(text: str) -> str:
    def replace(match: re.Match[str]) -> str:
        prefix = match.group(1) or match.group(2) or ""
        return f"{prefix}[REDACTED]"

    return _SECRET_TEXT_RE.sub(replace, text)


class CodexAppServerClient:
    """JSON-RPC client for a child or shared Codex app-server.

    With ``endpoint=unix:///path`` the client connects to a long-running
    app-server and leaves its lifecycle to the service manager.  Without an
    endpoint, the backwards-compatible private ``--stdio`` child is used.
    """

    def __init__(
        self,
        *,
        command: Sequence[str] = ("codex", "app-server", "--stdio"),
        endpoint: str | None = None,
        env: Mapping[str, str] | None = None,
        request_timeout: float = 30.0,
        client_name: str = "claude-voice-bridge",
        client_title: str = "Telegram Voice Bridge",
        client_version: str = "0.1.0",
        stderr_lines: int = 50,
    ) -> None:
        if not command:
            raise ValueError("app-server command cannot be empty")
        self._command = tuple(command)
        self._endpoint = endpoint.strip() if endpoint else None
        self._socket: ClientConnection | None = None
        self._env_source = dict(env) if env is not None else None
        self._request_timeout = request_timeout
        self._client_info = {
            "name": client_name,
            "title": client_title,
            "version": client_version,
        }
        self._process: asyncio.subprocess.Process | None = None
        self._reader_task: asyncio.Task | None = None
        self._stderr_task: asyncio.Task | None = None
        self._handler_tasks: set[asyncio.Task] = set()
        self._pending: dict[int | str, asyncio.Future] = {}
        self._request_handlers: dict[str, RequestHandler] = {}
        self._notification_handlers: list[NotificationHandler] = []
        self._write_lock = asyncio.Lock()
        self._lifecycle_lock = asyncio.Lock()
        self._next_id = 0
        self._closed = False
        self._started = False
        self._connection_error: BaseException | None = None
        self._stderr_tail: deque[str] = deque(maxlen=max(1, stderr_lines))

    @property
    def running(self) -> bool:
        if not self._started or self._closed:
            return False
        if self._endpoint:
            return self._socket is not None and self._socket.state.name == "OPEN"
        process = self._process
        return process is not None and process.returncode is None

    @property
    def pid(self) -> int | None:
        return self._process.pid if self._process is not None else None

    @property
    def stderr_tail(self) -> tuple[str, ...]:
        return tuple(self._stderr_tail)

    def add_notification_handler(self, handler: NotificationHandler) -> None:
        self._notification_handlers.append(handler)

    def add_request_handler(self, method: str, handler: RequestHandler) -> None:
        if not method:
            raise ValueError("request method cannot be empty")
        self._request_handlers[method] = handler

    async def start(self) -> JsonObject:
        """Spawn, initialize, and mark the connection ready."""

        async with self._lifecycle_lock:
            if self.running:
                return {}
            if self._process is not None or self._socket is not None:
                if self._connection_error is None:
                    raise CodexAppServerError("app-server client cannot be restarted")
                # The transport died (an oversized frame, a killed child, a
                # dropped socket).  Bury it and rebuild here, so one bad message
                # cannot poison every later turn until the service is restarted.
                await self._close_locked()
            self._closed = False
            self._connection_error = None
            try:
                if self._endpoint:
                    self._socket = await self._connect_unix(self._endpoint)
                else:
                    self._process = await asyncio.create_subprocess_exec(
                        *self._command,
                        stdin=asyncio.subprocess.PIPE,
                        stdout=asyncio.subprocess.PIPE,
                        stderr=asyncio.subprocess.PIPE,
                        env=build_codex_env(self._env_source),
                    )
            except (OSError, ValueError) as exc:
                self._closed = True
                action = "connect to" if self._endpoint else "start"
                raise CodexAppServerError(
                    f"could not {action} app-server: {exc}"
                ) from exc

            self._reader_task = asyncio.create_task(
                self._reader_loop(), name="codex-app-server-reader"
            )
            if self._process is not None:
                self._stderr_task = asyncio.create_task(
                    self._stderr_loop(), name="codex-app-server-stderr"
                )
            try:
                result = await self.request(
                    "initialize",
                    {"clientInfo": self._client_info, "capabilities": None},
                )
                if not isinstance(result, dict):
                    raise CodexProtocolError(
                        "initialize result must be a JSON object"
                    )
                await self.notify("initialized")
                self._started = True
                return result
            except BaseException:
                # We already hold _lifecycle_lock.  Calling close() here would
                # try to acquire it again and deadlock precisely on failed
                # initialization.
                await self._close_locked()
                raise

    @staticmethod
    async def _connect_unix(endpoint: str) -> ClientConnection:
        parsed = urlparse(endpoint)
        if parsed.scheme != "unix" or parsed.netloc or not parsed.path:
            raise ValueError("app-server endpoint must be unix:///absolute/path")
        path = unquote(parsed.path)
        if not os.path.isabs(path):
            raise ValueError("app-server Unix socket path must be absolute")
        # Codex doesn't negotiate permessage-deflate on its Unix endpoint.
        return await unix_connect(
            path,
            uri="ws://localhost/",
            compression=None,
            max_size=MAX_FRAME_BYTES,
        )

    async def request(
        self,
        method: str,
        params: JsonObject | None = None,
        *,
        timeout: float | None = None,
    ) -> Any:
        process = self._process
        if self._closed or (self._socket is None and process is None):
            raise CodexAppServerError("app-server is not connected")
        if self._socket is None and (process is None or process.stdin is None):
            raise CodexAppServerError("app-server is not connected")
        if self._connection_error is not None:
            raise CodexAppServerError(str(self._connection_error))

        self._next_id += 1
        request_id = self._next_id
        loop = asyncio.get_running_loop()
        future = loop.create_future()
        self._pending[request_id] = future
        envelope: JsonObject = {
            "jsonrpc": "2.0",
            "id": request_id,
            "method": method,
        }
        if params is not None:
            envelope["params"] = params
        try:
            await self._write(envelope)
            wait = self._request_timeout if timeout is None else timeout
            return await asyncio.wait_for(future, timeout=wait)
        except asyncio.TimeoutError as exc:
            raise CodexAppServerError(
                f"app-server request timed out: {method}"
            ) from exc
        finally:
            self._pending.pop(request_id, None)

    async def notify(self, method: str, params: JsonObject | None = None) -> None:
        envelope: JsonObject = {"jsonrpc": "2.0", "method": method}
        if params is not None:
            envelope["params"] = params
        await self._write(envelope)

    async def _write(self, envelope: JsonObject) -> None:
        process = self._process
        socket = self._socket
        if self._closed or (socket is None and process is None):
            raise CodexAppServerError("app-server is not connected")
        wire = json.dumps(envelope, separators=(",", ":"), ensure_ascii=False)
        async with self._write_lock:
            try:
                if socket is not None:
                    await socket.send(wire)
                else:
                    if process is None or process.stdin is None:
                        raise CodexAppServerError("app-server is not connected")
                    process.stdin.write((wire + "\n").encode("utf-8"))
                    await process.stdin.drain()
            except (BrokenPipeError, ConnectionError, OSError) as exc:
                error = CodexAppServerError(f"app-server write failed: {exc}")
                self._fail_pending(error)
                raise error from exc

    async def _reader_loop(self) -> None:
        error: BaseException | None = None
        try:
            if self._socket is not None:
                async for raw in self._socket:
                    self._decode_and_dispatch(raw)
                error = CodexAppServerError("app-server WebSocket closed")
            else:
                process = self._process
                assert process is not None and process.stdout is not None
                while True:
                    raw = await process.stdout.readline()
                    if not raw:
                        error = CodexAppServerError("app-server stdout closed")
                        return
                    self._decode_and_dispatch(raw)
        except asyncio.CancelledError:
            raise
        except BaseException as exc:  # pragma: no cover - defensive loop guard
            error = CodexAppServerError(f"app-server reader failed: {exc}")
            if not self._closed:
                logger.exception("app-server reader failed")
        finally:
            if error is not None and not self._closed:
                self._connection_error = error
                self._fail_pending(error)

    def _decode_and_dispatch(self, raw: str | bytes) -> None:
        try:
            value = json.loads(raw)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            logger.warning("invalid app-server JSON ignored: %s", exc)
            return
        try:
            self._dispatch(value)
        except CodexProtocolError as exc:
            logger.warning("invalid app-server envelope ignored: %s", exc)

    def _dispatch(self, value: Any) -> None:
        if not isinstance(value, dict):
            raise CodexProtocolError("top-level JSON-RPC value is not an object")
        if value.get("jsonrpc") not in (None, "2.0"):
            raise CodexProtocolError("unsupported JSON-RPC version")

        if "id" in value and "method" not in value:
            request_id = value.get("id")
            future = self._pending.get(request_id)
            if future is None or future.done():
                logger.debug("late or unknown app-server response id=%r", request_id)
                return
            if "error" in value:
                error = value.get("error")
                if not isinstance(error, dict):
                    future.set_exception(CodexProtocolError("invalid error response"))
                else:
                    future.set_exception(
                        CodexRequestError(
                            error.get("code"),
                            str(error.get("message") or "unknown error"),
                            error.get("data"),
                        )
                    )
            elif "result" in value:
                future.set_result(value.get("result"))
            else:
                future.set_exception(
                    CodexProtocolError("response has neither result nor error")
                )
            return

        method = value.get("method")
        if not isinstance(method, str) or not method:
            raise CodexProtocolError("event has no method")
        params = value.get("params", {})
        if not isinstance(params, dict):
            raise CodexProtocolError("event params are not an object")

        if "id" in value:
            self._track_handler_task(
                self._handle_server_request(value.get("id"), method, params),
                f"codex-request-{method}",
            )
            return
        for handler in tuple(self._notification_handlers):
            self._track_handler_task(
                self._call_notification(handler, method, params),
                f"codex-notification-{method}",
            )

    def _track_handler_task(self, coro: Awaitable[None], name: str) -> None:
        task = asyncio.create_task(coro, name=name)
        self._handler_tasks.add(task)
        task.add_done_callback(self._handler_tasks.discard)

    @staticmethod
    async def _call_notification(
        handler: NotificationHandler, method: str, params: JsonObject
    ) -> None:
        try:
            result = handler(method, params)
            if inspect.isawaitable(result):
                await result
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("app-server notification handler failed for %s", method)

    async def _handle_server_request(
        self, request_id: int | str | None, method: str, params: JsonObject
    ) -> None:
        handler = self._request_handlers.get(method)
        if handler is None:
            await self._send_response(
                request_id,
                error={"code": -32601, "message": f"unsupported method: {method}"},
            )
            return
        try:
            result = handler(params)
            if inspect.isawaitable(result):
                result = await result
            await self._send_response(request_id, result=result)
        except asyncio.CancelledError:
            try:
                await self._send_response(
                    request_id,
                    error={"code": -32000, "message": "request cancelled"},
                )
            except CodexAppServerError:
                pass
            raise
        except Exception as exc:
            logger.exception("app-server request handler failed for %s", method)
            await self._send_response(
                request_id,
                error={"code": -32000, "message": _redact(str(exc))},
            )

    async def _send_response(
        self,
        request_id: int | str | None,
        *,
        result: Any = None,
        error: JsonObject | None = None,
    ) -> None:
        envelope: JsonObject = {"jsonrpc": "2.0", "id": request_id}
        if error is None:
            envelope["result"] = result
        else:
            envelope["error"] = error
        await self._write(envelope)

    async def _stderr_loop(self) -> None:
        process = self._process
        assert process is not None and process.stderr is not None
        try:
            while True:
                raw = await process.stderr.readline()
                if not raw:
                    return
                line = _redact(raw.decode("utf-8", errors="replace").rstrip())
                self._stderr_tail.append(line[:2000])
                logger.debug("app-server stderr: %s", line[:2000])
        except asyncio.CancelledError:
            raise
        except Exception:  # pragma: no cover - diagnostics must not kill client
            logger.exception("app-server stderr reader failed")

    def _fail_pending(self, error: BaseException) -> None:
        for future in tuple(self._pending.values()):
            if not future.done():
                future.set_exception(error)

    async def close(self) -> None:
        """Close stdin, stop the child, and fail/cancel all outstanding work."""

        async with self._lifecycle_lock:
            await self._close_locked()

    async def _close_locked(self) -> None:
        """Close implementation; caller must hold ``_lifecycle_lock``."""

        if self._closed and self._process is None and self._socket is None:
            return
        self._closed = True
        self._started = False
        error = CodexAppServerError("app-server closed")
        self._fail_pending(error)

        for task in tuple(self._handler_tasks):
            task.cancel()
        if self._handler_tasks:
            await asyncio.gather(*self._handler_tasks, return_exceptions=True)
        self._handler_tasks.clear()

        socket = self._socket
        if socket is not None:
            await socket.close()

        process = self._process
        if process is not None:
            if process.stdin is not None:
                process.stdin.close()
                try:
                    await process.stdin.wait_closed()
                except (AttributeError, BrokenPipeError, ConnectionError):
                    pass
            if process.returncode is None:
                try:
                    await asyncio.wait_for(process.wait(), timeout=2.0)
                except asyncio.TimeoutError:
                    process.terminate()
                    try:
                        await asyncio.wait_for(process.wait(), timeout=2.0)
                    except asyncio.TimeoutError:
                        process.kill()
                        await process.wait()

        current = asyncio.current_task()
        tasks = [
            task
            for task in (self._reader_task, self._stderr_task)
            if task is not None and task is not current
        ]
        for task in tasks:
            if not task.done():
                task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        self._reader_task = None
        self._stderr_task = None
        self._process = None
        self._socket = None

    async def __aenter__(self) -> "CodexAppServerClient":
        await self.start()
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        await self.close()
