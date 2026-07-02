"""Per-project long-lived ClaudeSDKClient sessions in streaming-input mode.

One :class:`SessionManager` owns one ``ClaudeSDKClient`` per project. Each
project has its own :class:`asyncio.Queue` of inbound user turns and a single
background task that:

* drains the queue,
* forwards each turn to the SDK via ``client.query(...)``,
* streams the assistant's response, emitting :class:`~voice_bridge.types.Outbound`
  for any assistant text, and
* persists the SDK ``session_id`` so the session can be resumed across restarts.

Constraints honored:

* **C6** — the notify MCP server is built *per project*; its ``on_notify``
  closure emits ``Outbound(project.name, detail or summary, summary)`` so the
  user always sees which project pinged them. Never a literal ``"bridge"``.
* **C8** — each turn is processed under try/except; a crashing session emits a
  user-facing error Outbound and marks itself stopped instead of taking down the
  whole service.
* **C12** — uses the verified SDK API: ``ClaudeSDKClient`` /
  ``ClaudeAgentOptions`` / ``AssistantMessage`` / ``TextBlock`` / ``ResultMessage``,
  ``permission_mode="bypassPermissions"`` for ``full`` mode (no ``can_use_tool``),
  ``can_use_tool`` from :func:`make_can_use_tool` otherwise, and ``resume`` from
  the stored ``session_id``.
"""

from __future__ import annotations

import asyncio
import logging
import shutil
from pathlib import Path
from typing import Awaitable, Callable

from claude_agent_sdk import (
    AssistantMessage,
    ClaudeAgentOptions,
    ClaudeSDKClient,
    ResultMessage,
    TextBlock,
)

from .approvals import ApprovalManager, make_can_use_tool
from .config import Config, ProjectConfig, effective_autonomy
from .notify_tool import (
    ASK_USER_TOOL_NAME,
    NOTIFY_TOOL_NAME,
    SEND_FILE_TOOL_NAME,
    make_notify_server,
)
from .routing import Store
from .transcript import append_transcript
from .types import Outbound

logger = logging.getLogger(__name__)

# Appended to the agent's system prompt so its user-facing messages are
# voice-friendly and split cleanly into a spoken line + technical detail. This
# mirrors the bridge's prepare_outbound split on the ``---`` separator.
_VOICE_SPLIT_INSTRUCTION = (
    "When you send a user-facing message, make the FIRST line a short, "
    "spoken-friendly summary or question with NO code, paths, or commands. "
    "Then a line that is exactly '---'. Then put any code, diffs, paths, or "
    "commands below it. When you need the user to choose between options, use "
    "the bridge ask_user tool with short button labels."
)

# Sentinel pushed onto a session queue to ask its loop to exit cleanly.
_SHUTDOWN = None

_ERROR_SPOKEN = "The session crashed. Check the text."
_SILENT_SPOKEN = " "
_TURN_ERROR_SPOKEN = "Turas baigėsi klaida."
_GIVEUP_SPOKEN = "Sesija nekyla, ją išjungiau."

# Per-turn "still working" heartbeat. A multi-minute tool-running turn emits no
# user-facing text until it finishes, which reads as total silence to a walking
# user. A watchdog emits ONE brief Outbound after each interval of genuine
# silence (reset whenever the loop receives assistant text), so it never fires
# on a fast turn and stays non-spammy on a slow one.
_HEARTBEAT_INTERVAL = 60.0
_HEARTBEAT_TEXT = "Vis dar dirbu…"
_HEARTBEAT_SPOKEN = "Vis dar dirbu…"

# Supervised auto-restart policy. Backoff grows exponentially from *base*
# seconds, doubling per attempt, capped at *cap*. After *max* consecutive
# restart cycles without a successful turn the project is disabled.
_RESTART_BACKOFF_BASE = 1.0
_RESTART_BACKOFF_CAP = 30.0
_RESTART_MAX_ATTEMPTS = 5


class _Session:
    """Live state for one project's ClaudeSDKClient."""

    def __init__(self, project: ProjectConfig) -> None:
        self.project = project
        self.client: ClaudeSDKClient | None = None
        self.queue: asyncio.Queue = asyncio.Queue()
        self.task: asyncio.Task | None = None


class SessionManager:
    """Owns one long-lived streaming Claude session per project."""

    def __init__(
        self,
        projects: list[ProjectConfig],
        cfg: Config,
        store: Store,
        on_outbound: Callable[[Outbound], Awaitable[None]],
        approvals: ApprovalManager,
        ask_user: Callable[[str, str, list[str]], Awaitable[str]] | None = None,
    ) -> None:
        self._projects: dict[str, ProjectConfig] = {p.name: p for p in projects}
        self._cfg = cfg
        self._store = store
        self._on_outbound = on_outbound
        self._approvals = approvals
        self._ask_user = ask_user
        self._sessions: dict[str, _Session] = {}

        # Supervision state (crash recovery / deliver recovery).
        # _restart_tasks: pending backoff->restart supervisor task per project.
        # _attempts: consecutive restart cycles since the last successful turn.
        # _locks: per-project start lock guarding double-start races.
        # _stopping: projects whose stop is intentional (must not auto-restart).
        # _closed: set once by stop_all so no new restart is ever scheduled.
        self._restart_tasks: dict[str, asyncio.Task] = {}
        self._attempts: dict[str, int] = {}
        self._locks: dict[str, asyncio.Lock] = {}
        self._stopping: set[str] = set()
        self._closed = False
        self._restart_backoff_base = _RESTART_BACKOFF_BASE
        self._restart_backoff_cap = _RESTART_BACKOFF_CAP
        self._max_restart_attempts = _RESTART_MAX_ATTEMPTS
        # Seconds of user-facing silence within a turn before the "still
        # working" heartbeat fires. An instance attribute so tests set it tiny.
        self.heartbeat_interval = _HEARTBEAT_INTERVAL

    # ------------------------------------------------------------------ #
    # Lookup
    # ------------------------------------------------------------------ #

    def project(self, name: str) -> ProjectConfig | None:
        """Return the ProjectConfig for *name*, or None if unknown."""
        return self._projects.get(name)

    def names(self) -> list[str]:
        """Return configured project names in projects.yaml order."""
        return list(self._projects)

    def add_projects(self, projects: list[ProjectConfig]) -> int:
        """Add newly discovered projects without starting their sessions."""
        added = 0
        for project in projects:
            if project.name in self._projects:
                continue
            self._projects[project.name] = project
            added += 1
        return added

    def is_running(self, name: str) -> bool:
        """Return True if a live session task exists for *name*."""
        return name in self._sessions

    # ------------------------------------------------------------------ #
    # Lifecycle
    # ------------------------------------------------------------------ #

    async def start_all(self) -> None:
        """Start a session task for each enabled project, in isolation.

        One project's ``client.connect()`` failure must never take down the
        whole boot: each start is wrapped so a failure is logged, surfaced to
        the user, and the next project still starts.
        """
        for name in self._projects:
            if not await self._store.is_enabled(name):
                continue
            try:
                await self._start(name)
            except Exception as err:  # noqa: BLE001 - boot must not die
                logger.exception("start_all: failed to start %s", name)
                try:
                    await self._on_outbound(
                        Outbound(
                            project=name,
                            text=f"{name}: nepavyko paleisti — {err}",
                            spoken="nepavyko paleisti",
                        )
                    )
                except Exception:  # pragma: no cover - defensive
                    logger.exception(
                        "start_all: failed to emit start failure for %s", name
                    )

    async def deliver(self, project: str, text: str) -> None:
        """Enqueue a user turn.

        If the project has no live session but is still enabled (e.g. it is in
        the down window after a crash, before the supervisor has resumed it),
        (re)start the session first so the turn is not black-holed. Unknown or
        disabled projects remain a no-op.
        """
        sess = self._sessions.get(project)
        if sess is None:
            if project not in self._projects:
                return
            if not await self._store.is_enabled(project):
                return
            await self._ensure_started(project)
            sess = self._sessions.get(project)
            if sess is None:
                return
        position = sess.queue.qsize() + 1
        await sess.queue.put(text)
        if position > 1:
            await self._emit_status(project, f"Queued: {position}.")

    async def _ensure_started(self, name: str) -> None:
        """Start *name* if not running, guarding against double-start races.

        A per-project lock serializes concurrent recovery starts; any pending
        restart supervisor is cancelled first so it cannot start a second
        client. A failed start is logged (deliver stays best-effort).
        """
        if name in self._sessions:
            return
        lock = self._locks.setdefault(name, asyncio.Lock())
        async with lock:
            if name in self._sessions:
                return
            await self._cancel_restart(name)
            if name in self._sessions:
                return
            try:
                # Already holding the per-project lock: call the locked variant
                # directly (self._start would re-acquire and deadlock).
                await self._start_locked(name)
            except Exception:  # noqa: BLE001 - deliver recovery is best-effort
                logger.exception("deliver recovery: failed to start %s", name)

    async def interrupt(self, project: str) -> bool:
        """Cancel the running session, drop queued turns, and restart if enabled."""
        if project not in self._projects:
            return False
        was_running = project in self._sessions
        await self._stop(project)
        if await self._store.is_enabled(project):
            await self._start(project)
        await self._emit_status(project, "Interrupted.")
        return was_running

    async def set_enabled(self, project: str, enabled: bool) -> None:
        """Persist the enabled flag and start (resume) or stop the session."""
        if project not in self._projects:
            return
        await self._store.set_enabled(project, enabled)
        if enabled:
            if self._cfg.open_vscode_on_enable:
                await self._open_vscode(self._projects[project])
            await self._start(project)
        else:
            await self._stop(project)
            if self._cfg.close_vscode_on_disable:
                await self._close_vscode(self._projects[project])

    async def set_mode(self, project: str, mode: str) -> None:
        """Update a project's autonomy; restart the running session so the new
        ``permission_mode`` / ``can_use_tool`` take effect. Resume preserves
        context across the restart."""
        cfg = self._projects.get(project)
        if cfg is None:
            return
        if mode not in {"full", "safe", "ask"}:
            logger.warning("set_mode: invalid mode %r for project %r; ignored", mode, project)
            return
        cfg.autonomy = mode
        if project in self._sessions:
            await self._stop(project)
            await self._start(project)

    async def stop_all(self) -> None:
        """Stop every running session and cancel every pending restart.

        Sets ``_closed`` first so a crash racing with shutdown cannot schedule
        a new restart that would resurrect a session after we have torn down.
        """
        self._closed = True
        for name in set(self._sessions) | set(self._restart_tasks):
            await self._stop(name)

    # ------------------------------------------------------------------ #
    # Internals
    # ------------------------------------------------------------------ #

    def _build_options(
        self, project: ProjectConfig, resume: str | None, notify_server
    ) -> ClaudeAgentOptions:
        mode = effective_autonomy(project, self._cfg)

        append_text = "\n\n".join(
            p for p in [project.system_prompt_extra, _VOICE_SPLIT_INSTRUCTION] if p
        )

        if mode == "full":
            permission_mode = "bypassPermissions"
            can_use_tool = None
        else:
            permission_mode = "default"
            can_use_tool = make_can_use_tool(project, self._cfg, self._approvals)

        return ClaudeAgentOptions(
            cwd=project.cwd,
            model=project.model,
            system_prompt={"type": "preset", "preset": "claude_code", "append": append_text},
            permission_mode=permission_mode,
            can_use_tool=can_use_tool,
            mcp_servers={"bridge": notify_server},
            allowed_tools=[NOTIFY_TOOL_NAME, SEND_FILE_TOOL_NAME, ASK_USER_TOOL_NAME],
            resume=resume,
        )

    def _make_on_notify(
        self, project_name: str
    ) -> Callable[[str, str], Awaitable[None]]:
        """C6: per-project notify closure. Emits Outbound tagged with the
        project so the user knows who is pinging them."""

        async def on_notify(summary: str, detail: str) -> None:
            await self._on_outbound(
                Outbound(
                    project=project_name,
                    text=detail or summary,
                    spoken=summary,
                )
            )

        return on_notify

    def _make_on_send_file(
        self, project_name: str
    ) -> Callable[[str, str], Awaitable[str]]:
        project = self._projects[project_name]

        async def on_send_file(path: str, caption: str) -> str:
            resolved = _resolve_project_file(project.cwd, path)
            if resolved is None:
                return "denied: path must be inside the project directory"
            if not resolved.is_file():
                return "not found: file does not exist"
            await self._on_outbound(
                Outbound(
                    project=project_name,
                    text=caption.strip() or resolved.name,
                    spoken="",
                    file_path=str(resolved),
                )
            )
            return "delivered"

        return on_send_file

    def _make_on_ask_user(
        self, project_name: str
    ) -> Callable[[str, list[str]], Awaitable[str]]:

        async def on_ask_user(question: str, choices: list[str]) -> str:
            if self._ask_user is None:
                return ""
            return await self._ask_user(project_name, question, choices)

        return on_ask_user

    async def _start(self, name: str) -> None:
        """Start a session for *name*, serialized by the per-project lock.

        The lock closes the double-start window: two concurrent starts for one
        project cannot both pass the ``name in _sessions`` guard and build two
        CLI subprocesses. The fast path for an already-running project returns
        before touching the lock.
        """
        if name in self._sessions:
            return
        lock = self._locks.setdefault(name, asyncio.Lock())
        async with lock:
            await self._start_locked(name)

    async def _start_locked(self, name: str) -> None:
        """Build and register a session. Caller must hold the per-project lock.

        Connecting the client is wrapped so a cancellation (e.g. a supervisor
        task cancelled mid-connect) or a connect failure disconnects the
        already-connected CLI subprocess instead of orphaning it — the client
        is only registered into ``_sessions`` once it is fully connected.
        """
        if name in self._sessions:
            return
        project = self._projects[name]
        sess = _Session(project)

        notify_server = make_notify_server(
            self._make_on_notify(name),
            self._make_on_send_file(name),
            self._make_on_ask_user(name),
        )
        resume = await self._store.get_session_id(name)
        options = self._build_options(project, resume, notify_server)

        client = ClaudeSDKClient(options)
        try:
            await client.connect()
        except BaseException:
            try:
                await client.disconnect()
            except Exception:  # pragma: no cover - defensive
                logger.exception("cleanup after failed connect for %s", name)
            raise
        sess.client = client

        self._sessions[name] = sess
        sess.task = asyncio.create_task(self._run_loop(sess))

    async def _open_vscode(self, project: ProjectConfig) -> None:
        code = shutil.which("code")
        if code is None:
            logger.warning("OPEN_VSCODE_ON_ENABLE is set but 'code' is not on PATH")
            return
        try:
            proc = await asyncio.create_subprocess_exec(
                code,
                project.cwd,
                stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.DEVNULL,
            )
            await proc.wait()
            if proc.returncode != 0:
                logger.warning("code %s exited with %s", project.cwd, proc.returncode)
        except OSError:
            logger.exception("failed to open VS Code for %s", project.cwd)

    async def _close_vscode(self, project: ProjectConfig) -> None:
        wmctrl = shutil.which("wmctrl")
        if wmctrl is None:
            logger.warning("CLOSE_VSCODE_ON_DISABLE is set but 'wmctrl' is not on PATH")
            return
        basename = Path(project.cwd).name
        try:
            list_proc = await asyncio.create_subprocess_exec(
                wmctrl,
                "-l",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.DEVNULL,
            )
            out, _ = await list_proc.communicate()
            if list_proc.returncode != 0:
                logger.warning("wmctrl -l exited with %s", list_proc.returncode)
                return
            for line in out.decode("utf-8", "replace").splitlines():
                parts = line.split(None, 3)
                if len(parts) < 4:
                    continue
                window_id, title = parts[0], parts[3]
                if "Visual Studio Code" not in title:
                    continue
                if f" - {basename} - Visual Studio Code" not in title:
                    continue
                close_proc = await asyncio.create_subprocess_exec(
                    wmctrl,
                    "-ic",
                    window_id,
                    stdout=asyncio.subprocess.DEVNULL,
                    stderr=asyncio.subprocess.DEVNULL,
                )
                await close_proc.wait()
        except OSError:
            logger.exception("failed to close VS Code for %s", project.cwd)

    async def _stop(self, name: str) -> None:
        # Mark the stop intentional for the whole duration so a crash racing
        # in a sibling coroutine cannot be misread as needing a restart, and
        # cancel any pending restart supervisor before touching the session.
        self._stopping.add(name)
        try:
            await self._cancel_restart(name)
            sess = self._sessions.pop(name, None)
            if sess is None:
                return
            # Ask the loop to exit cleanly, then cancel as a fallback.
            try:
                sess.queue.put_nowait(_SHUTDOWN)
            except asyncio.QueueFull:  # pragma: no cover - unbounded queue
                pass
            if sess.task is not None:
                sess.task.cancel()
                try:
                    await sess.task
                except asyncio.CancelledError:
                    pass
                except Exception:  # pragma: no cover - defensive
                    logger.exception("session %s task raised during stop", name)
            if sess.client is not None:
                try:
                    await sess.client.disconnect()
                except Exception:  # pragma: no cover - defensive
                    logger.exception("session %s disconnect failed", name)
        finally:
            self._stopping.discard(name)
            self._attempts.pop(name, None)

    async def _cancel_restart(self, name: str) -> None:
        """Cancel and await a project's pending restart supervisor, if any."""
        task = self._restart_tasks.pop(name, None)
        if task is None or task.done():
            return
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass
        except Exception:  # pragma: no cover - defensive
            logger.exception("restart task for %s raised during cancel", name)

    async def _run_loop(self, sess: _Session) -> None:
        """Drain the queue, forward turns to the SDK, emit assistant output.

        Wrapped per-turn in try/except (C8): a crashing turn emits an error
        Outbound and stops this session without affecting any other project.
        """
        name = sess.project.name
        assert sess.client is not None
        client = sess.client
        while True:
            text = await sess.queue.get()
            if text is _SHUTDOWN:
                return
            try:
                await self._emit_status(name, "Working.")
                await append_transcript(sess.project.cwd, "user", text)
                await client.query(text)
                parts: list[str] = []
                result_error = False
                result_subtype: str | None = None
                result_detail: str | None = None
                # Per-turn "still working" watchdog. It fires during genuine
                # silence and is reset whenever we receive assistant text. It is
                # always cancelled AND awaited in the finally so it can never
                # leak, fire after the turn, or fire during a cancellation.
                activity = asyncio.Event()
                watchdog = asyncio.create_task(
                    self._heartbeat_loop(name, activity)
                )
                try:
                    async for msg in client.receive_response():
                        if isinstance(msg, AssistantMessage):
                            for block in msg.content:
                                if isinstance(block, TextBlock):
                                    parts.append(block.text)
                                    activity.set()  # reset the heartbeat timer
                        elif isinstance(msg, ResultMessage):
                            session_id = getattr(msg, "session_id", None)
                            if session_id:
                                await self._store.set_session_id(name, session_id)
                            if getattr(msg, "is_error", False):
                                result_error = True
                                result_subtype = getattr(msg, "subtype", None)
                                result_detail = getattr(msg, "result", None)
                            await self._capture_usage(name, msg)
                finally:
                    watchdog.cancel()
                    try:
                        await watchdog
                    except asyncio.CancelledError:
                        pass
                joined = "\n".join(p for p in parts if p).strip()
                if joined:
                    await append_transcript(sess.project.cwd, "assistant", joined)
                    await self._on_outbound(
                        Outbound(project=name, text=joined, spoken="")
                    )
                elif result_error:
                    # The SDK ended the turn in error with no assistant text;
                    # surface something so the user is not left in silence.
                    detail = result_detail or f"Turas baigėsi klaida: {result_subtype}"
                    await self._on_outbound(
                        Outbound(project=name, text=detail, spoken=_TURN_ERROR_SPOKEN)
                    )
                # A turn completed without raising: the session is healthy,
                # so reset the consecutive-restart counter.
                self._attempts.pop(name, None)
            except asyncio.CancelledError:
                raise
            except Exception as err:  # noqa: BLE001 - C8: never crash the service
                logger.exception("session %s crashed on a turn", name)
                await self._emit_crash(sess, err)
                self._schedule_restart(name)
                return

    async def _capture_usage(self, name: str, msg: ResultMessage) -> None:
        """Persist one turn's token/cost usage from a ResultMessage (B3c).

        Defensive by design: ``usage`` may be a plain dict (the normal shape,
        keyed ``input_tokens`` / ``output_tokens`` /
        ``cache_read_input_tokens`` / ``cache_creation_input_tokens``) or, in
        principle, an object exposing the same names as attributes — both are
        read via a getter that never raises on a missing key/attr. A
        malformed payload (unexpected type, non-numeric value) is caught here
        and logged; it must never propagate into the per-turn loop and be
        mistaken for a turn crash (C8 corollary). ``total_cost_usd is None``
        (Claude Code subscription auth reports no cost) is passed through
        unchanged — ``Store.add_usage`` stores that as 0.
        """
        try:
            usage = getattr(msg, "usage", None) or {}

            def _field(key: str) -> int:
                value = usage.get(key) if isinstance(usage, dict) else getattr(usage, key, None)
                return int(value or 0)

            await self._store.add_usage(
                name,
                cost_usd=getattr(msg, "total_cost_usd", None),
                input_tokens=_field("input_tokens"),
                output_tokens=_field("output_tokens"),
                cache_read_tokens=_field("cache_read_input_tokens"),
                cache_creation_tokens=_field("cache_creation_input_tokens"),
            )
        except Exception:  # noqa: BLE001 - never let a bad usage payload
            # crash the turn loop; it would otherwise be misread as a
            # session crash by _run_loop's outer except.
            logger.exception("usage capture failed for %s", name)

    async def _heartbeat_loop(self, name: str, activity: asyncio.Event) -> None:
        """Emit a "still working" Outbound after each interval of silence.

        Sleeps one ``heartbeat_interval`` at a time. If *activity* was set during
        that interval (the turn loop received assistant text) the timer is reset
        and no heartbeat fires; otherwise — genuine silence — it emits ONE
        heartbeat. So a turn that keeps streaming text never fires, and a long
        silence fires at most once per interval. Built on ``asyncio.sleep`` (not
        ``wait_for``, which mis-handles cancellation on Python < 3.11) so the
        caller's cancel+await tears it down cleanly: it never fires after the
        turn ends nor during an intentional cancellation.
        """
        while True:
            await asyncio.sleep(self.heartbeat_interval)
            if activity.is_set():
                activity.clear()  # progress this interval: reset, do not fire
                continue
            try:
                await self._on_outbound(
                    Outbound(
                        project=name,
                        text=_HEARTBEAT_TEXT,
                        spoken=_HEARTBEAT_SPOKEN,
                    )
                )
            except asyncio.CancelledError:
                raise
            except Exception:
                # Heartbeat is best-effort: a failure here must never latch and
                # poison the turn (its exception would re-raise on the turn's
                # `await watchdog` and discard the turn's real output).
                logger.exception("heartbeat outbound failed for %s", name)

    async def _emit_crash(self, sess: _Session, err: Exception) -> None:
        name = sess.project.name
        # Mark the session stopped without a re-entrant cancel of this task.
        self._sessions.pop(name, None)
        if sess.client is not None:
            try:
                await sess.client.disconnect()
            except Exception:  # pragma: no cover - defensive
                logger.exception("session %s disconnect after crash failed", name)
        await append_transcript(sess.project.cwd, "system", f"Sesija krito: {err}")
        try:
            await self._on_outbound(
                Outbound(
                    project=name,
                    text=f"Sesija krito: {err}",
                    spoken=_ERROR_SPOKEN,
                    alert=True,
                )
            )
        except Exception:  # pragma: no cover - defensive
            logger.exception("failed to emit crash Outbound for %s", name)

    # ------------------------------------------------------------------ #
    # Supervised restart
    # ------------------------------------------------------------------ #

    def _schedule_restart(self, name: str) -> None:
        """Schedule a supervised restart for a crashed project (idempotent).

        Never schedules for an intentional stop or after shutdown. At most one
        supervisor per project runs at a time.
        """
        if self._closed or name in self._stopping:
            return
        existing = self._restart_tasks.get(name)
        if existing is not None and not existing.done():
            return
        self._restart_tasks[name] = asyncio.create_task(
            self._supervise_restart(name)
        )

    def _restart_backoff(self, attempt: int) -> float:
        return min(
            self._restart_backoff_base * (2 ** (attempt - 1)),
            self._restart_backoff_cap,
        )

    async def _supervise_restart(self, name: str) -> None:
        """Back off, then resume a crashed session while it is still wanted.

        ``_attempts[name]`` counts consecutive restart cycles since the last
        successful turn (a healthy turn resets it in ``_run_loop``). After
        ``_max_restart_attempts`` cycles the project is disabled and the user
        is told. Bails out (no restart) if the project was disabled, is being
        intentionally stopped, or was already brought back up meanwhile.
        """
        try:
            while True:
                attempt = self._attempts.get(name, 0) + 1
                self._attempts[name] = attempt
                if attempt > self._max_restart_attempts:
                    await self._store.set_enabled(name, False)
                    await self._emit_giveup(name)
                    self._attempts.pop(name, None)
                    return

                await asyncio.sleep(self._restart_backoff(attempt))

                if self._closed or name in self._stopping:
                    return
                if not await self._store.is_enabled(name):
                    return
                if name in self._sessions:
                    # deliver-recovery (or another path) already resumed it.
                    return

                try:
                    await self._start(name)
                except Exception:  # noqa: BLE001 - retry with longer backoff
                    logger.exception(
                        "restart %s failed (attempt %d)", name, attempt
                    )
                    continue
                else:
                    # Session is back up. If it crashes again a fresh
                    # supervisor is scheduled by _run_loop.
                    return
        finally:
            if self._restart_tasks.get(name) is asyncio.current_task():
                self._restart_tasks.pop(name, None)

    async def _emit_giveup(self, name: str) -> None:
        n = self._max_restart_attempts
        try:
            await self._on_outbound(
                Outbound(
                    project=name,
                    text=(
                        f"po {n} bandymų sesija nekyla, išjungiau — "
                        f"/on {name} kai pataisysi"
                    ),
                    spoken=_GIVEUP_SPOKEN,
                )
            )
        except Exception:  # pragma: no cover - defensive
            logger.exception("failed to emit give-up Outbound for %s", name)

    async def _emit_status(self, project: str, text: str) -> None:
        await self._on_outbound(
            Outbound(project=project, text=text, spoken=_SILENT_SPOKEN)
        )


def _resolve_project_file(cwd: str, requested: str) -> Path | None:
    if not requested.strip():
        return None
    root = Path(cwd).resolve()
    path = Path(requested).expanduser()
    if not path.is_absolute():
        path = root / path
    try:
        resolved = path.resolve()
        resolved.relative_to(root)
    except (OSError, ValueError):
        return None
    return resolved
