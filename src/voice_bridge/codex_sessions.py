"""Codex app-server backed project sessions.

The public methods intentionally mirror ``sessions.SessionManager`` so the
Telegram/routing layers can select a backend without branching everywhere.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
import logging
from pathlib import Path
from typing import Any, Awaitable, Callable

from .approvals import ApprovalManager, is_risky, signature_for
from .codex_app_server import (
    CodexAppServerClient,
    CodexAppServerError,
    CodexRequestError,
)
from .config import (
    AUTONOMY_MODES,
    EFFORT_LEVELS,
    Config,
    ProjectConfig,
    effective_autonomy,
)
from .routing import Store
from .transcript import append_transcript
from .types import Outbound

logger = logging.getLogger(__name__)

_SHUTDOWN = object()
_SILENT_SPOKEN = " "
_TURN_ERROR_SPOKEN = "Turas baigėsi klaida."
_VOICE_INSTRUCTION = (
    "When you send a user-facing final answer, make the first line a short, "
    "spoken-friendly summary with no code, paths, or commands. Then put a "
    "line containing exactly '---' before technical detail."
)


@dataclass
class _CodexSession:
    project: ProjectConfig
    thread_id: str
    queue: asyncio.Queue = field(default_factory=asyncio.Queue)
    task: asyncio.Task | None = None
    active_turn_id: str | None = None


def _short(value: object, limit: int = 80) -> str:
    text = " ".join(str(value or "").split())
    return text if len(text) <= limit else text[: limit - 1] + "…"


def _activity_line(item: dict[str, Any]) -> str | None:
    kind = item.get("type")
    if kind == "commandExecution":
        return f"🔧 Bash: {_short(item.get('command'))}".rstrip()
    if kind == "fileChange":
        count = len(item.get("changes") or [])
        return f"🔧 Edit: {count} file{'s' if count != 1 else ''}"
    if kind == "mcpToolCall":
        server = _short(item.get("server"), 30)
        tool = _short(item.get("tool"), 40)
        return f"🔧 {server}/{tool}".rstrip("/")
    if kind == "webSearch":
        return "🔧 Web search"
    if kind == "imageView":
        return f"🔧 View: {Path(str(item.get('path') or '')).name}"
    return None


class CodexSessionManager:
    """Own per-project Codex threads on one local app-server process."""

    def __init__(
        self,
        projects: list[ProjectConfig],
        cfg: Config,
        store: Store,
        on_outbound: Callable[[Outbound], Awaitable[None]],
        approvals: ApprovalManager,
        ask_user: Callable[[str, str, list[str]], Awaitable[str]] | None = None,
        *,
        client: CodexAppServerClient | None = None,
    ) -> None:
        self._projects = {project.name: project for project in projects}
        self._cfg = cfg
        self._store = store
        self._on_outbound = on_outbound
        self._approvals = approvals
        self._ask_user = ask_user
        self._client = client or CodexAppServerClient()
        self._sessions: dict[str, _CodexSession] = {}
        self._thread_projects: dict[str, str] = {}
        self._last_model: dict[str, str] = {}
        self._completion_waiters: dict[
            tuple[str, str], asyncio.Future[dict[str, Any]]
        ] = {}
        self._early_completions: dict[tuple[str, str], dict[str, Any]] = {}
        self._delta_text: dict[tuple[str, str], list[str]] = {}
        self._start_lock = asyncio.Lock()
        self._closed = False

        self._client.add_notification_handler(self._on_notification)
        self._client.add_request_handler(
            "item/commandExecution/requestApproval",
            self._on_command_approval,
        )
        self._client.add_request_handler(
            "item/fileChange/requestApproval",
            self._on_file_approval,
        )
        self._client.add_request_handler(
            "item/permissions/requestApproval",
            self._on_permissions_approval,
        )
        self._client.add_request_handler(
            "item/tool/requestUserInput",
            self._on_user_input,
        )

    def project(self, name: str) -> ProjectConfig | None:
        return self._projects.get(name)

    def names(self) -> list[str]:
        return list(self._projects)

    def last_model(self, project: str) -> str | None:
        return self._last_model.get(project)

    def add_projects(self, projects: list[ProjectConfig]) -> int:
        added = 0
        for project in projects:
            if project.name not in self._projects:
                self._projects[project.name] = project
                added += 1
        return added

    def is_running(self, name: str) -> bool:
        session = self._sessions.get(name)
        return session is not None and session.task is not None and not session.task.done()

    async def start_all(self) -> None:
        enabled = [
            name for name in self._projects if await self._store.is_enabled(name)
        ]
        if not enabled:
            return
        try:
            await self._ensure_client()
        except Exception as exc:  # noqa: BLE001 - keep Telegram service alive
            logger.exception("could not start Codex app-server")
            for name in enabled:
                await self._safe_outbound(
                    Outbound(
                        project=name,
                        text=f"{name}: nepavyko paleisti Codex — {exc}",
                        spoken="nepavyko paleisti Codex",
                        alert=True,
                    )
                )
            return
        for name in enabled:
            try:
                await self._start(name)
            except Exception as exc:  # noqa: BLE001 - isolate projects
                logger.exception("could not start Codex project %s", name)
                await self._safe_outbound(
                    Outbound(
                        project=name,
                        text=f"{name}: nepavyko atidaryti Codex gijos — {exc}",
                        spoken="nepavyko atidaryti Codex gijos",
                        alert=True,
                    )
                )

    async def _ensure_client(self) -> None:
        if self._client.running:
            return
        async with self._start_lock:
            if not self._client.running:
                await self._client.start()

    async def _start(self, name: str) -> None:
        if name in self._sessions or name not in self._projects:
            return
        await self._ensure_client()
        project = self._projects[name]
        stored = await self._store.get_agent_session_id(name, "codex")
        result: dict[str, Any]
        if stored:
            try:
                result = await self._client.request(
                    "thread/resume",
                    self._thread_params(project, thread_id=stored),
                )
            except CodexRequestError as exc:
                if "active writer" in exc.message.lower():
                    raise CodexAppServerError(
                        "Codex thread is still open in the old IDE process; "
                        "reload the VS Code window, then retry"
                    ) from exc
                logger.warning("stored Codex thread for %s cannot resume; starting new", name)
                result = await self._client.request(
                    "thread/start", self._thread_params(project)
                )
        else:
            result = await self._client.request(
                "thread/start", self._thread_params(project)
            )
        thread = result.get("thread") if isinstance(result, dict) else None
        thread_id = thread.get("id") if isinstance(thread, dict) else None
        if not isinstance(thread_id, str) or not thread_id:
            raise CodexAppServerError("thread response has no thread.id")
        await self._store.set_agent_session_id(name, "codex", thread_id)
        model = result.get("model") if isinstance(result, dict) else None
        if isinstance(model, str) and model:
            self._last_model[name] = model
        session = _CodexSession(project=project, thread_id=thread_id)
        session.task = asyncio.create_task(
            self._run_loop(session), name=f"codex-session-{name}"
        )
        self._sessions[name] = session
        self._thread_projects[thread_id] = name

    def _thread_params(
        self, project: ProjectConfig, *, thread_id: str | None = None
    ) -> dict[str, Any]:
        params: dict[str, Any] = {
            "cwd": project.cwd,
            "approvalPolicy": self._approval_policy(project),
            "approvalsReviewer": "user",
            "sandbox": self._sandbox_mode(project),
        }
        if thread_id is not None:
            params["threadId"] = thread_id
            params["excludeTurns"] = True
        if project.model:
            params["model"] = project.model
        instructions = "\n\n".join(
            value
            for value in (project.system_prompt_extra, _VOICE_INSTRUCTION)
            if value
        )
        if instructions:
            params["developerInstructions"] = instructions
        return params

    def _approval_policy(self, project: ProjectConfig) -> str:
        mode = effective_autonomy(project, self._cfg)
        if mode == "full":
            return "never"
        if mode == "ask":
            return "untrusted"
        return "on-request"

    def _sandbox_mode(self, project: ProjectConfig) -> str:
        return (
            "danger-full-access"
            if effective_autonomy(project, self._cfg) == "full"
            else "workspace-write"
        )

    async def deliver(self, project: str, text: str) -> None:
        if project not in self._projects or not await self._store.is_enabled(project):
            return
        if project not in self._sessions:
            try:
                await self._start(project)
            except Exception as exc:  # noqa: BLE001 - user-facing, not service-fatal
                logger.exception("Codex deliver start failed for %s", project)
                await self._safe_outbound(
                    Outbound(
                        project=project,
                        text=f"{project}: Codex nepasiekiamas — {exc}",
                        spoken="Codex nepasiekiamas",
                        alert=True,
                    )
                )
                return
        session = self._sessions.get(project)
        if session is None:
            return
        position = session.queue.qsize() + 1
        await session.queue.put(text)
        if position > 1:
            await self._safe_outbound(
                Outbound(project=project, text=f"Queued: {position}.", spoken=" ", transient=True)
            )

    async def _run_loop(self, session: _CodexSession) -> None:
        while True:
            text = await session.queue.get()
            if text is _SHUTDOWN:
                return
            try:
                await self._run_turn(session, str(text))
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # noqa: BLE001 - one turn must not kill service
                logger.exception("Codex turn failed for %s", session.project.name)
                await self._safe_outbound(
                    Outbound(
                        project=session.project.name,
                        text=f"Codex turas nutrūko: {exc}",
                        spoken=_TURN_ERROR_SPOKEN,
                        alert=True,
                    )
                )

    async def _run_turn(self, session: _CodexSession, text: str) -> None:
        name = session.project.name
        if not self._client.running:
            # A transport that died mid-run (an oversized frame, a killed child)
            # is rebuilt here so the next Telegram turn recovers on its own
            # instead of failing until the service is restarted.
            logger.warning("reconnecting Codex app-server before turn for %s", name)
        await self._ensure_client()
        await self._safe_outbound(
            Outbound(project=name, text="Working.", spoken=_SILENT_SPOKEN, transient=True)
        )
        await append_transcript(session.project.cwd, "user", text)
        params: dict[str, Any] = {
            "threadId": session.thread_id,
            "input": [{"type": "text", "text": text, "text_elements": []}],
            "cwd": session.project.cwd,
            "approvalPolicy": self._approval_policy(session.project),
            "approvalsReviewer": "user",
            "sandboxPolicy": self._sandbox_policy(session.project),
        }
        if session.project.model:
            params["model"] = session.project.model
        if session.project.effort:
            params["effort"] = session.project.effort
        result = await self._client.request("turn/start", params)
        turn = result.get("turn") if isinstance(result, dict) else None
        turn_id = turn.get("id") if isinstance(turn, dict) else None
        if not isinstance(turn_id, str) or not turn_id:
            raise CodexAppServerError("turn/start response has no turn.id")
        session.active_turn_id = turn_id
        key = (session.thread_id, turn_id)
        loop = asyncio.get_running_loop()
        future: asyncio.Future[dict[str, Any]] = loop.create_future()
        self._completion_waiters[key] = future
        early = self._early_completions.pop(key, None)
        if early is not None and not future.done():
            future.set_result(early)
        try:
            completed = await future
        finally:
            self._completion_waiters.pop(key, None)
            session.active_turn_id = None
        completed_turn = completed.get("turn")
        if not isinstance(completed_turn, dict):
            raise CodexAppServerError("turn/completed has no turn")
        status = completed_turn.get("status")
        final_text = self._final_text(completed_turn, key)
        self._delta_text.pop(key, None)
        if final_text:
            await append_transcript(session.project.cwd, "assistant", final_text)
            await self._safe_outbound(Outbound(project=name, text=final_text, spoken=""))
        elif status == "failed":
            error = completed_turn.get("error")
            detail = error.get("message") if isinstance(error, dict) else None
            await self._safe_outbound(
                Outbound(
                    project=name,
                    text=detail or "Codex turas baigėsi klaida.",
                    spoken=_TURN_ERROR_SPOKEN,
                    alert=True,
                )
            )

    def _sandbox_policy(self, project: ProjectConfig) -> dict[str, Any]:
        if effective_autonomy(project, self._cfg) == "full":
            return {"type": "dangerFullAccess"}
        return {
            "type": "workspaceWrite",
            "writableRoots": [project.cwd],
            "networkAccess": False,
            "excludeTmpdirEnvVar": False,
            "excludeSlashTmp": False,
        }

    def _final_text(
        self, turn: dict[str, Any], key: tuple[str, str]
    ) -> str:
        finals: list[str] = []
        unknown: list[str] = []
        for item in turn.get("items") or []:
            if not isinstance(item, dict) or item.get("type") != "agentMessage":
                continue
            text = item.get("text")
            if not isinstance(text, str) or not text.strip():
                continue
            if item.get("phase") == "final_answer":
                finals.append(text.strip())
            elif item.get("phase") in (None, ""):
                unknown.append(text.strip())
        if finals:
            return "\n".join(finals).strip()
        if unknown:
            return unknown[-1]
        return "".join(self._delta_text.get(key, [])).strip()

    async def _on_notification(self, method: str, params: dict[str, Any]) -> None:
        thread_id = params.get("threadId")
        turn_id = params.get("turnId")
        if method == "item/agentMessage/delta":
            delta = params.get("delta")
            if isinstance(thread_id, str) and isinstance(turn_id, str) and isinstance(delta, str):
                self._delta_text.setdefault((thread_id, turn_id), []).append(delta)
            return
        if method == "item/completed":
            if not isinstance(thread_id, str):
                return
            name = self._thread_projects.get(thread_id)
            session = self._sessions.get(name) if name else None
            item = params.get("item")
            if session is not None and session.project.verbose and isinstance(item, dict):
                line = _activity_line(item)
                if line:
                    await self._safe_outbound(
                        Outbound(project=name, text=line, spoken=_SILENT_SPOKEN, transient=True)
                    )
            return
        if method == "turn/completed":
            turn = params.get("turn")
            if not isinstance(thread_id, str) or not isinstance(turn, dict):
                return
            actual_turn_id = turn.get("id")
            if not isinstance(actual_turn_id, str):
                return
            key = (thread_id, actual_turn_id)
            waiter = self._completion_waiters.get(key)
            if waiter is not None and not waiter.done():
                waiter.set_result(params)
            else:
                self._early_completions[key] = params

    async def _on_command_approval(self, params: dict[str, Any]) -> dict[str, str]:
        project = self._project_for_request(params)
        if project is None:
            return {"decision": "decline"}
        command = params.get("command")
        tool_input = {
            "command": command if isinstance(command, str) else "",
            "cwd": params.get("cwd") or project.cwd,
        }
        approved = await self._approve(project, "Bash", tool_input)
        return {"decision": "accept" if approved else "decline"}

    async def _on_file_approval(self, params: dict[str, Any]) -> dict[str, str]:
        project = self._project_for_request(params)
        if project is None:
            return {"decision": "decline"}
        tool_input = {
            "file_path": params.get("grantRoot") or project.cwd,
            "reason": params.get("reason") or "",
        }
        approved = await self._approve(project, "Write", tool_input)
        return {"decision": "accept" if approved else "decline"}

    async def _on_permissions_approval(
        self, params: dict[str, Any]
    ) -> dict[str, Any]:
        """Relay broader filesystem/network permission requests to Telegram.

        These requests are never silently accepted in safe mode: unlike a
        concrete command, a permission profile grants a wider capability and
        cannot be classified safely by the legacy command risk matcher.
        """
        project = self._project_for_request(params)
        requested = params.get("permissions")
        if project is None or not isinstance(requested, dict):
            return {"permissions": {}, "scope": "turn"}
        if effective_autonomy(project, self._cfg) == "full":
            approved = True
        else:
            approved = await self._approvals.request(
                project.name,
                "Permissions",
                {
                    "cwd": params.get("cwd") or project.cwd,
                    "reason": params.get("reason") or "",
                    "permissions": requested,
                },
                policy_signature=None,
            )
        if not approved:
            return {"permissions": {}, "scope": "turn"}
        granted = {
            key: value
            for key, value in requested.items()
            if key in {"network", "fileSystem"} and value is not None
        }
        return {"permissions": granted, "scope": "turn"}

    async def _on_user_input(self, params: dict[str, Any]) -> dict[str, Any]:
        project = self._project_for_request(params)
        questions = params.get("questions")
        if project is None or not isinstance(questions, list) or self._ask_user is None:
            return {"answers": {}}
        answers: dict[str, dict[str, list[str]]] = {}
        for question in questions[:3]:
            if not isinstance(question, dict):
                continue
            question_id = question.get("id")
            prompt = question.get("question")
            # Telegram is not an appropriate secret-input surface. Returning no
            # answer lets Codex continue or ask through another trusted route.
            if (
                not isinstance(question_id, str)
                or not isinstance(prompt, str)
                or question.get("isSecret") is True
            ):
                continue
            options = question.get("options")
            choices = [
                item["label"]
                for item in (options or [])
                if isinstance(item, dict)
                and isinstance(item.get("label"), str)
                and item["label"]
            ]
            answer = await self._ask_user(project.name, prompt, choices)
            if isinstance(answer, str) and answer:
                answers[question_id] = {"answers": [answer]}
        return {"answers": answers}

    def _project_for_request(self, params: dict[str, Any]) -> ProjectConfig | None:
        thread_id = params.get("threadId")
        if not isinstance(thread_id, str):
            return None
        name = self._thread_projects.get(thread_id)
        return self._projects.get(name) if name else None

    async def _approve(
        self, project: ProjectConfig, tool_name: str, tool_input: dict[str, Any]
    ) -> bool:
        mode = effective_autonomy(project, self._cfg)
        if mode == "full":
            return True
        risky = is_risky(tool_name, tool_input, project.cwd)
        if mode == "safe" and not risky:
            return True
        signature = signature_for(tool_name, tool_input, project.cwd)
        if signature is not None:
            try:
                if await self._store.has_policy(project.name, signature):
                    return True
            except Exception:  # noqa: BLE001 - fail closed into prompt
                logger.exception("Codex approval policy lookup failed")
        return await self._approvals.request(
            project.name,
            tool_name,
            tool_input,
            policy_signature=signature,
        )

    async def interrupt(self, project: str) -> bool:
        session = self._sessions.get(project)
        if session is None:
            return False
        was_active = session.active_turn_id is not None
        if session.active_turn_id is not None:
            await self._client.request(
                "turn/interrupt",
                {"threadId": session.thread_id, "turnId": session.active_turn_id},
            )
        while not session.queue.empty():
            try:
                session.queue.get_nowait()
            except asyncio.QueueEmpty:
                break
        await self._safe_outbound(
            Outbound(project=project, text="Interrupted.", spoken=_SILENT_SPOKEN, transient=True)
        )
        return was_active

    async def set_enabled(self, project: str, enabled: bool) -> None:
        if project not in self._projects:
            return
        await self._store.set_enabled(project, enabled)
        if enabled:
            await self._start(project)
        else:
            await self._stop(project)

    async def set_mode(self, project: str, mode: str) -> None:
        config = self._projects.get(project)
        if config is not None and mode in AUTONOMY_MODES:
            config.autonomy = mode

    async def set_effort(self, project: str, level: str) -> None:
        config = self._projects.get(project)
        if config is not None and level in EFFORT_LEVELS:
            config.effort = level

    async def set_verbose(self, project: str | None, on: bool) -> None:
        targets = [project] if project is not None else list(self._projects)
        for name in targets:
            config = self._projects.get(name)
            if config is not None:
                config.verbose = on

    async def _stop(self, name: str) -> None:
        session = self._sessions.pop(name, None)
        if session is None:
            return
        self._thread_projects.pop(session.thread_id, None)
        if session.active_turn_id is not None:
            try:
                await self._client.request(
                    "turn/interrupt",
                    {"threadId": session.thread_id, "turnId": session.active_turn_id},
                    timeout=5,
                )
            except Exception:  # noqa: BLE001 - shutdown must continue
                logger.warning("could not interrupt Codex turn during stop", exc_info=True)
        if session.task is not None:
            session.task.cancel()
            await asyncio.gather(session.task, return_exceptions=True)

    async def stop_all(self) -> None:
        self._closed = True
        for name in list(self._sessions):
            await self._stop(name)
        for waiter in tuple(self._completion_waiters.values()):
            if not waiter.done():
                waiter.cancel()
        self._completion_waiters.clear()
        self._early_completions.clear()
        self._delta_text.clear()
        await self._client.close()

    async def _safe_outbound(self, outbound: Outbound) -> None:
        try:
            await self._on_outbound(outbound)
        except Exception:  # noqa: BLE001 - outbound failure cannot kill backend
            logger.exception("Codex outbound failed for %s", outbound.project)
