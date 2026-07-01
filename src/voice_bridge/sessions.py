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
from .notify_tool import NOTIFY_TOOL_NAME, make_notify_server
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
    "commands below it."
)

# Sentinel pushed onto a session queue to ask its loop to exit cleanly.
_SHUTDOWN = None

_ERROR_SPOKEN = "Sesija krito, žiūrėk tekstą."


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
    ) -> None:
        self._projects: dict[str, ProjectConfig] = {p.name: p for p in projects}
        self._cfg = cfg
        self._store = store
        self._on_outbound = on_outbound
        self._approvals = approvals
        self._sessions: dict[str, _Session] = {}

    # ------------------------------------------------------------------ #
    # Lookup
    # ------------------------------------------------------------------ #

    def project(self, name: str) -> ProjectConfig | None:
        """Return the ProjectConfig for *name*, or None if unknown."""
        return self._projects.get(name)

    def names(self) -> list[str]:
        """Return configured project names in projects.yaml order."""
        return list(self._projects)

    def is_running(self, name: str) -> bool:
        """Return True if a live session task exists for *name*."""
        return name in self._sessions

    # ------------------------------------------------------------------ #
    # Lifecycle
    # ------------------------------------------------------------------ #

    async def start_all(self) -> None:
        """Start a session task for each project where store.is_enabled is True."""
        for name in self._projects:
            if await self._store.is_enabled(name):
                await self._start(name)

    async def deliver(self, project: str, text: str) -> None:
        """Enqueue a user turn. No-op if the session is not running."""
        sess = self._sessions.get(project)
        if sess is None:
            return
        await sess.queue.put(text)

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
        """Stop and disconnect every running session."""
        for name in list(self._sessions):
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
            allowed_tools=[NOTIFY_TOOL_NAME],
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

    async def _start(self, name: str) -> None:
        if name in self._sessions:
            return
        project = self._projects[name]
        sess = _Session(project)

        notify_server = make_notify_server(self._make_on_notify(name))
        resume = await self._store.get_session_id(name)
        options = self._build_options(project, resume, notify_server)

        client = ClaudeSDKClient(options)
        await client.connect()
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
                await append_transcript(sess.project.cwd, "user", text)
                await client.query(text)
                parts: list[str] = []
                async for msg in client.receive_response():
                    if isinstance(msg, AssistantMessage):
                        for block in msg.content:
                            if isinstance(block, TextBlock):
                                parts.append(block.text)
                    elif isinstance(msg, ResultMessage):
                        session_id = getattr(msg, "session_id", None)
                        if session_id:
                            await self._store.set_session_id(name, session_id)
                joined = "\n".join(p for p in parts if p).strip()
                if joined:
                    await append_transcript(sess.project.cwd, "assistant", joined)
                    await self._on_outbound(
                        Outbound(project=name, text=joined, spoken="")
                    )
            except asyncio.CancelledError:
                raise
            except Exception as err:  # noqa: BLE001 - C8: never crash the service
                logger.exception("session %s crashed on a turn", name)
                await self._emit_crash(sess, err)
                return

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
                )
            )
        except Exception:  # pragma: no cover - defensive
            logger.exception("failed to emit crash Outbound for %s", name)
