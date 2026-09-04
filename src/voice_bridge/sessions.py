"""Long-lived ClaudeSDKClient conversations, one per forum sub-topic.

A project is a place on disk; a *conversation* is one agent session inside it,
keyed ``"<project>#<n>"``. A project can have several at once — ``paprika#1``
and ``paprika#2`` are two independent Claude sessions in the same directory,
each with its own Telegram topic. The project's own topic is a control hub and
never carries agent turns.

Each conversation owns one ``ClaudeSDKClient``, one :class:`asyncio.Queue` of
inbound user turns, and one background task that:

* drains the queue,
* forwards each turn to the SDK via ``client.query(...)``,
* streams the response — emitting live tool-by-tool progress while the agent
  works, then the assistant text as an :class:`~voice_bridge.types.Outbound`,
* persists the SDK ``session_id`` so the conversation resumes across restarts.

Constraints honored:

* **C6** — the notify MCP server is built *per conversation*; its ``on_notify``
  closure emits ``Outbound(key, detail or summary, summary)`` so the user always
  sees which conversation pinged them. Never a literal ``"bridge"``.
* **C8** — each turn is processed under try/except; a crashing conversation
  emits a user-facing error Outbound and marks itself stopped instead of taking
  down the whole service.
* **C12** — uses the verified SDK API: ``ClaudeSDKClient`` /
  ``ClaudeAgentOptions`` / ``AssistantMessage`` / ``TextBlock`` /
  ``ResultMessage``, ``can_use_tool`` from :func:`make_can_use_tool`, and
  ``resume`` / ``fork_session`` from the stored session id.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import shutil
import time
import warnings
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Awaitable, Callable

from claude_agent_sdk import (
    AssistantMessage,
    ClaudeAgentOptions,
    ClaudeSDKClient,
    ResultMessage,
    SystemMessage,
    TextBlock,
    ThinkingBlock,
    ToolResultBlock,
    ToolUseBlock,
    UserMessage,
)

from . import claude_history
from .approvals import ApprovalManager, make_can_use_tool
from .config import (
    Config,
    ProjectConfig,
    claude_sessions_path,
    effective_autonomy,
)
from .notify_tool import (
    ASK_USER_TOOL_NAME,
    NOTIFY_TOOL_NAME,
    SEND_FILE_TOOL_NAME,
    make_notify_server,
)
from .routing import Store, conversation_key, project_of
from .transcript import append_transcript
from .types import Outbound

logger = logging.getLogger(__name__)

# The SDK warns that `allowed_tools` auto-approves our own bridge MCP tools
# before can_use_tool runs. That is deliberate — notify_user / send_file /
# ask_user are how the bridge talks to its user, and gating them behind an
# approval the user could only answer through those same tools would deadlock.
# Without this the warning repeats on every session start.
try:  # pragma: no cover - the class is new in recent SDKs
    from claude_agent_sdk import CanUseToolShadowedWarning

    warnings.filterwarnings("ignore", category=CanUseToolShadowedWarning)
except ImportError:  # pragma: no cover
    pass

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

AUTONOMY_MODES = ("full", "auto", "safe", "ask")


def permission_mode(autonomy: str) -> str:
    """The CLI permission mode behind one of our autonomy settings.

    ``auto`` hands the judgement to Claude Code itself — the same thing the VS
    Code extension does — so routine reads run without a Telegram round trip and
    only what the CLI escalates reaches ``can_use_tool``. Every other mode keeps
    the CLI neutral and lets our own callback decide.
    """
    return "auto" if autonomy == "auto" else "default"

# How many activity lines the live progress message keeps.
PROGRESS_LINES = 14
_THINKING = "\U0001F4AD thinking…"

EFFORT_LEVELS = ("low", "medium", "high", "xhigh", "max")


@lru_cache(maxsize=1)
def _claude_settings() -> dict:
    """Claude Code's own ``settings.json``, or ``{}`` if it cannot be read.

    Neither the model nor the effort is knowable before a session has answered
    once — nothing announces them at connect — so an untouched session takes
    its labels from the same file the CLI reads. Cached: a restart picks up an
    edit, same as for the CLI itself.
    """
    path = Path(os.environ.get("CLAUDE_CONFIG_DIR") or "~/.claude").expanduser()
    try:
        settings = json.loads((path / "settings.json").read_text())
    except (OSError, ValueError):  # missing, unreadable, or not JSON
        return {}
    return settings if isinstance(settings, dict) else {}


def settings_effort() -> str | None:
    """The effort level Claude Code is configured with, if it is a valid one."""
    level = _claude_settings().get("effortLevel")
    return level if level in EFFORT_LEVELS else None


def settings_model() -> str | None:
    """The model Claude Code is configured with, e.g. ``opus[1m]``."""
    model = _claude_settings().get("model")
    return model if isinstance(model, str) and model else None


def short_model(model: str | None) -> str:
    """``claude-opus-4-5-20251101`` -> ``opus-4-5``: the part that identifies it.

    The date suffix and the vendor prefix are the same on every answer, so they
    only cost width in a footer that is meant to be glanceable.
    """
    if not model:
        return "default"
    name = re.sub(r"-\d{8}$", "", model.strip())
    return name.removeprefix("claude-") or model


def with_model_footer(text: str, model: str | None, effort: str | None = None) -> str:
    """Append the answering model to a turn, below the voice split marker.

    It goes last so ``prepare_outbound`` keeps it out of the spoken summary —
    the voice never reads a model id.
    """
    label = short_model(model or settings_model())
    return f"{text}\n\n— {label} · {effort or settings_effort() or 'high'}"


def _entrypoint_env() -> dict[str, str]:
    """Override the marker the SDK stamps on every session it starts.

    The SDK sets ``CLAUDE_CODE_ENTRYPOINT=sdk-py``, and the Claude Code VS Code
    extension hides any session whose entrypoint is ``sdk-cli``/``sdk-ts``/
    ``sdk-py``, so programmatic runs do not clutter the human session list. That
    also hides these Telegram conversations, which we do want listed. The SDK
    merges ``options.env`` over its own default, so setting the variable is
    enough — no patching the SDK, and no rewriting session files afterwards.

    Unset or empty keeps the SDK default, i.e. the sessions stay hidden.
    """
    value = os.environ.get("CLAUDE_CODE_ENTRYPOINT", "").strip()
    return {"CLAUDE_CODE_ENTRYPOINT": value} if value else {}


# --------------------------------------------------------------------------- #
# Live progress rendering (pure)
# --------------------------------------------------------------------------- #

_TOOL_ICONS = {
    "Bash": "\U0001F527",
    "Read": "\U0001F4D6",
    "Edit": "✏️",
    "Write": "\U0001F4DD",
    "Glob": "\U0001F50D",
    "Grep": "\U0001F50D",
    "WebFetch": "\U0001F310",
    "WebSearch": "\U0001F310",
    "Task": "\U0001F916",
    "TodoWrite": "\U0001F4CB",
}
# Per tool, the input field worth showing. Everything else falls back to the
# first short string value, which covers MCP tools we know nothing about.
_TOOL_ARGS = {
    "Bash": "command",
    "Read": "file_path",
    "Edit": "file_path",
    "Write": "file_path",
    "Glob": "pattern",
    "Grep": "pattern",
    "WebFetch": "url",
    "WebSearch": "query",
    "Task": "description",
}


def _tool_detail(name: str, tool_input: dict) -> str:
    key = _TOOL_ARGS.get(name)
    value = tool_input.get(key) if key else None
    if not isinstance(value, str) or not value.strip():
        value = next(
            (v for v in tool_input.values() if isinstance(v, str) and v.strip()),
            "",
        )
    return " ".join(str(value).split())


def format_tool(name: str, tool_input: dict) -> str:
    """One compact activity line for a tool call, e.g. ``🔧 Bash: pytest -q``."""
    icon = _TOOL_ICONS.get(name, "⚙️")
    short = name.rsplit("__", 1)[-1]  # mcp__bridge__notify_user -> notify_user
    detail = _tool_detail(name, tool_input)
    if len(detail) > 70:
        detail = detail[:69] + "…"
    return f"{icon} {short}: {detail}" if detail else f"{icon} {short}"


@dataclass
class _Progress:
    """Rolling activity log for one in-flight turn."""

    started: float
    lines: list[str] = field(default_factory=list)
    index: dict[str, int] = field(default_factory=dict)  # tool_use_id -> line
    tools: int = 0
    last_sent: float = 0.0
    last_hash: str = ""

    def add_tool(self, block: ToolUseBlock) -> None:
        self.index[block.id] = len(self.lines)
        self.lines.append(format_tool(block.name, block.input or {}))
        self.tools += 1

    def add_thinking(self) -> None:
        if self.lines[-1:] != [_THINKING]:
            self.lines.append(_THINKING)

    def mark_result(self, block: ToolResultBlock) -> None:
        i = self.index.get(block.tool_use_id)
        if i is None or i >= len(self.lines):
            return
        mark = " ✗" if block.is_error else " ✓"
        if not self.lines[i].endswith(("✓", "✗")):
            self.lines[i] += mark

    def render(self, elapsed: float) -> str:
        tail = self.lines[-PROGRESS_LINES:]
        body = "\n".join(tail) if tail else "starting…"
        return f"⏳ {_elapsed(elapsed)} · {self.tools} tools\n{body}"

    def render_final(self, elapsed: float) -> str:
        return f"✅ done · {_elapsed(elapsed)} · {self.tools} tools"


def _elapsed(seconds: float) -> str:
    total = int(seconds)
    return f"{total}s" if total < 60 else f"{total // 60}m{total % 60:02d}s"


# --------------------------------------------------------------------------- #
# Session manager
# --------------------------------------------------------------------------- #


class _Session:
    """Live state for one conversation's ClaudeSDKClient."""

    def __init__(self, key: str, project: ProjectConfig) -> None:
        self.key = key
        self.project = project
        self.client: ClaudeSDKClient | None = None
        self.queue: asyncio.Queue = asyncio.Queue()
        self.task: asyncio.Task | None = None
        self.progress: _Progress | None = None
        self.model: str | None = None  # what actually answered, per AssistantMessage
        self.effort: str | None = None  # only settable at connect, so kept per session


class SessionManager:
    """Owns every live conversation across every project."""

    def __init__(
        self,
        projects: list[ProjectConfig],
        cfg: Config,
        store: Store,
        on_outbound: Callable[[Outbound], Awaitable[None]],
        approvals: ApprovalManager,
        ask_user: Callable[[str, str, list[str]], Awaitable[str]] | None = None,
        on_open: Callable[[str, str, int], Awaitable[None]] | None = None,
        on_close: Callable[[str], Awaitable[None]] | None = None,
        on_progress: Callable[[str, str, bool], Awaitable[None]] | None = None,
    ) -> None:
        self._projects: dict[str, ProjectConfig] = {p.name: p for p in projects}
        self._cfg = cfg
        self._store = store
        self._on_outbound = on_outbound
        self._approvals = approvals
        self._ask_user = ask_user
        self._on_open = on_open
        self._on_close = on_close
        self._on_progress = on_progress
        self._sessions: dict[str, _Session] = {}
        # Effort is a connect-time CLI flag, so a per-conversation choice has to
        # outlive the client it was made on and be re-applied on every start.
        self._effort: dict[str, str] = {}

    # ------------------------------------------------------------------ #
    # Lookup
    # ------------------------------------------------------------------ #

    def project(self, target: str | None) -> ProjectConfig | None:
        """ProjectConfig for a project name OR a conversation key."""
        if not target:
            return None
        return self._projects.get(project_of(target))

    def names(self) -> list[str]:
        """Configured project names in projects.yaml order."""
        return list(self._projects)

    def add_projects(self, projects: list[ProjectConfig]) -> int:
        """Add newly discovered projects without opening conversations."""
        added = 0
        for project in projects:
            if project.name in self._projects:
                continue
            self._projects[project.name] = project
            added += 1
        return added

    def is_running(self, key: str) -> bool:
        """True if a live session task exists for a conversation key."""
        return key in self._sessions

    def running_keys(self) -> list[str]:
        return list(self._sessions)

    def is_busy(self, key: str) -> bool:
        """True while a turn is in flight (a progress log is open)."""
        sess = self._sessions.get(key)
        return sess is not None and sess.progress is not None

    # ------------------------------------------------------------------ #
    # Lifecycle
    # ------------------------------------------------------------------ #

    async def start_all(self) -> None:
        """Restore the conversations that already exist, and only those.

        Deliberately creates nothing: a session appearing in a topic nobody
        asked for is indistinguishable from a stray one, and ten enabled
        projects meant ten empty topics and ten idle agents on every start. New
        conversations come from ``/new`` or ``/resume``.
        """
        for name in self._projects:
            if not await self._store.is_enabled(name):
                continue
            for conv in await self._store.conversations(name):
                await self._start(conv.key, resume=conv.session_id)

    async def stage(self, project: str, *, resume: str | None = None) -> str | None:
        """Reserve a conversation (``#N``) and its topic WITHOUT starting it.

        Attaching an old session shows the user what they are about to reopen,
        inside the topic that will hold it — which needs the topic to exist
        before anything runs. :meth:`attach` then starts it, or :meth:`close`
        throws both away.
        """
        if project not in self._projects:
            return None
        # Disabled means the user switched this project off; opening a session
        # in it anyway would quietly undo that. start_all used to be the only
        # caller and carried this check itself.
        if not await self._store.is_enabled(project):
            logger.info("refusing to open a conversation in disabled project %s", project)
            return None
        ordinal = await self._store.next_ordinal(project)
        key = conversation_key(project, ordinal)
        await self._store.add_conversation(
            key, project, ordinal, session_id=resume, created=time.time()
        )
        await self._ensure_topic(key)
        return key

    async def attach(self, key: str, *, fork: bool = False) -> bool:
        """Start a staged conversation, resuming whatever session it holds."""
        if key in self._sessions:
            return False
        conv = await self._store.conversation(key)
        if conv is None or conv.closed:
            return False
        await self._start(key, resume=conv.session_id, fork=fork)
        return key in self._sessions

    async def open(
        self,
        project: str,
        *,
        resume: str | None = None,
        fork: bool = False,
    ) -> str | None:
        """Stage a new conversation and start it in one go.

        ``resume`` attaches an existing Claude session; ``fork`` branches it
        instead of writing to it, which is what an already-open session needs.
        Returns the conversation key, or None for an unknown project.
        """
        key = await self.stage(project, resume=resume)
        if key is not None:
            await self._start(key, resume=resume, fork=fork)
        return key

    async def close(self, key: str) -> bool:
        """Stop a conversation and mark it closed; its topic can be removed."""
        conv = await self._store.conversation(key)
        if conv is None:
            return False
        await self._stop(key)
        await self._store.close_conversation(key)
        if self._on_close is not None:
            await self._on_close(key)
        return True

    async def deliver(self, key: str, text: str) -> None:
        """Enqueue a user turn. No-op if the conversation is not running."""
        sess = self._sessions.get(key)
        if sess is None:
            return
        position = sess.queue.qsize() + 1
        await sess.queue.put(text)
        if position > 1:
            await self._emit_status(key, f"Queued: {position}.")

    async def interrupt(self, key: str) -> bool:
        """Cancel the in-flight turn and drop anything queued behind it.

        Prefers the SDK's own ``interrupt`` — it stops the turn while keeping
        the session and its context alive. Only if that fails do we fall back to
        tearing the client down and resuming it.
        """
        sess = self._sessions.get(key)
        if sess is None:
            return False
        while not sess.queue.empty():
            try:
                sess.queue.get_nowait()
            except asyncio.QueueEmpty:  # pragma: no cover - drained concurrently
                break
        interrupted = False
        if sess.client is not None:
            try:
                await sess.client.interrupt()
                interrupted = True
            except Exception:  # noqa: BLE001 - no turn running, or transport gone
                logger.info("SDK interrupt failed for %s; restarting", key, exc_info=True)
        if not interrupted:
            conv = await self._store.conversation(key)
            await self._stop(key)
            await self._start(key, resume=conv.session_id if conv else None)
        await self._emit_status(key, "Interrupted.")
        return True

    async def set_enabled(self, project: str, enabled: bool) -> None:
        """Persist the enabled flag and start or stop that project's work."""
        if project not in self._projects:
            return
        await self._store.set_enabled(project, enabled)
        if enabled:
            if self._cfg.open_vscode_on_enable:
                await self._open_vscode(self._projects[project])
            # Same rule as start_all: enabling a project makes it available,
            # it does not conjure a conversation the user never asked for.
            for conv in await self._store.conversations(project):
                await self._start(conv.key, resume=conv.session_id)
        else:
            for key in list(self._sessions):
                if project_of(key) == project:
                    await self._stop(key)
            if self._cfg.close_vscode_on_disable:
                await self._close_vscode(self._projects[project])

    async def set_mode(self, project: str, mode: str) -> None:
        """Update a project's autonomy, live.

        ``make_can_use_tool`` reads the mode per call, and the CLI's own
        permission mode is pushed to every running client — so nothing restarts
        and no in-flight turn is lost.
        """
        cfg = self._projects.get(project)
        if cfg is None:
            return
        if mode not in AUTONOMY_MODES:
            logger.warning("set_mode: invalid mode %r for project %r; ignored", mode, project)
            return
        cfg.autonomy = mode
        wanted = permission_mode(mode)
        for key, sess in self._sessions.items():
            if project_of(key) != project or sess.client is None:
                continue
            try:
                await sess.client.set_permission_mode(wanted)
            except Exception:  # noqa: BLE001 - a stale client must not block the switch
                logger.exception("set_permission_mode failed for %s", key)

    async def set_model(self, target: str, model: str | None) -> None:
        """Switch the model live.

        ``target`` is either a project name (every conversation of it, and the
        default for new ones) or a single conversation key like ``app#2`` —
        sub-topics exist so two tasks do not share a context, so a switch made
        inside one must not reach its siblings.
        """
        if "#" in target:
            sess = self._sessions.get(target)
            if sess is None or sess.client is None:
                return
            try:
                await sess.client.set_model(model)
            except Exception:  # noqa: BLE001 - a stale client must not break the switch
                logger.exception("set_model failed for %s", target)
            return
        cfg = self._projects.get(target)
        if cfg is None:
            return
        cfg.model = model
        for key, sess in self._sessions.items():
            if project_of(key) != target or sess.client is None:
                continue
            try:
                await sess.client.set_model(model)
            except Exception:  # noqa: BLE001 - a stale client must not break the switch
                logger.exception("set_model failed for %s", key)

    async def set_effort(self, target: str, effort: str | None) -> str:
        """Change the reasoning effort, reconnecting the affected sessions.

        Unlike the model, effort is only a connect-time CLI flag — there is no
        control request for it — so the client is torn down and resumed on the
        same session id, which keeps the context. A conversation that is mid-turn
        is refused rather than reconnected: that would throw away the work in
        flight, and the user can ask again a minute later.
        """
        if effort is not None and effort not in EFFORT_LEVELS:
            return "usage: /effort <low|medium|high|xhigh|max|default> [project]"
        keys = (
            [target]
            if "#" in target
            else [k for k in self._sessions if project_of(k) == target]
        )
        if "#" not in target:
            cfg = self._projects.get(target)
            if cfg is None:
                return f"unknown project: {target}"
            cfg.effort = effort
        busy = [k for k in keys if self.is_busy(k)]
        for key in keys:
            if key in busy:
                continue
            if effort is None:
                self._effort.pop(key, None)
            else:
                self._effort[key] = effort
            if key in self._sessions:
                await self._restart(key)
        done = f"effort {effort or 'default'} for {target}"
        if busy:
            return f"{done} — still working, so {', '.join(busy)} kept the old one; ask again when idle."
        return done

    async def _restart(self, key: str) -> None:
        """Reconnect one conversation on its own session id, keeping context."""
        conv = await self._store.conversation(key)
        await self._stop(key)
        await self._start(key, resume=conv.session_id if conv else None)

    def effort_of(self, target: str) -> str | None:
        """The effort a conversation runs with, or a project's default."""
        sess = self._sessions.get(target)
        if sess is not None:
            return sess.effort
        project = self._projects.get(project_of(target))
        return self._effort.get(target) or (project.effort if project else None)

    def model_of(self, target: str) -> str | None:
        """The model a conversation last answered with.

        Before the first answer there is nothing to report from the wire, so
        this falls back to the project's override and then to Claude Code's own
        configured model — otherwise a freshly restarted session says
        "default", which tells the user nothing.
        """
        sess = self._sessions.get(target)
        if sess is not None and sess.model:
            return sess.model
        project = self._projects.get(project_of(target))
        return (project.model if project else None) or settings_model()

    async def stop_all(self) -> None:
        """Stop and disconnect every running conversation."""
        for key in list(self._sessions):
            await self._stop(key)

    # ------------------------------------------------------------------ #
    # Internals
    # ------------------------------------------------------------------ #

    def _build_options(
        self,
        project: ProjectConfig,
        key: str,
        resume: str | None,
        fork: bool,
        notify_server,
    ) -> ClaudeAgentOptions:
        append_text = "\n\n".join(
            p for p in [project.system_prompt_extra, _VOICE_SPLIT_INSTRUCTION] if p
        )

        # can_use_tool is always installed and decides per call, including the
        # "full" allow-everything case. bypassPermissions would skip the
        # callback entirely, and then switching out of full mode on a live
        # session would leave prompts with nobody to answer them.
        return ClaudeAgentOptions(
            cwd=project.cwd,
            model=project.model,
            effort=self._effort.get(key, project.effort),
            system_prompt={"type": "preset", "preset": "claude_code", "append": append_text},
            permission_mode=permission_mode(effective_autonomy(project, self._cfg)),
            can_use_tool=make_can_use_tool(project, self._cfg, self._approvals, key),
            mcp_servers={"bridge": notify_server},
            allowed_tools=[NOTIFY_TOOL_NAME, SEND_FILE_TOOL_NAME, ASK_USER_TOOL_NAME],
            resume=resume,
            fork_session=fork,
            env=_entrypoint_env(),
        )

    def _make_on_notify(self, key: str) -> Callable[[str, str], Awaitable[None]]:
        """C6: per-conversation notify closure. Emits Outbound tagged with the
        conversation so the user knows which topic is pinging them."""

        async def on_notify(summary: str, detail: str) -> None:
            await self._on_outbound(
                Outbound(project=key, text=detail or summary, spoken=summary)
            )

        return on_notify

    def _make_on_send_file(self, key: str) -> Callable[[str, str], Awaitable[str]]:
        project = self._projects[project_of(key)]

        async def on_send_file(path: str, caption: str) -> str:
            resolved = _resolve_project_file(project.cwd, path)
            if resolved is None:
                return "denied: path must be inside the project directory"
            if not resolved.is_file():
                return "not found: file does not exist"
            await self._on_outbound(
                Outbound(
                    project=key,
                    text=caption.strip() or resolved.name,
                    spoken="",
                    file_path=str(resolved),
                )
            )
            return "delivered"

        return on_send_file

    def _make_on_ask_user(self, key: str) -> Callable[[str, list[str]], Awaitable[str]]:

        async def on_ask_user(question: str, choices: list[str]) -> str:
            if self._ask_user is None:
                return ""
            return await self._ask_user(key, question, choices)

        return on_ask_user

    async def _start(
        self, key: str, *, resume: str | None = None, fork: bool = False
    ) -> None:
        if key in self._sessions:
            return
        project = self._projects.get(project_of(key))
        if project is None:
            logger.warning("cannot start %s: unknown project", key)
            return
        await self._ensure_topic(key)
        fork = fork or self._should_fork(resume)
        sess = _Session(key, project)

        notify_server = make_notify_server(
            self._make_on_notify(key),
            self._make_on_send_file(key),
            self._make_on_ask_user(key),
        )
        sess.effort = self._effort.get(key, project.effort)
        options = self._build_options(project, key, resume, fork, notify_server)

        client = ClaudeSDKClient(options)
        await client.connect()
        sess.client = client

        self._sessions[key] = sess
        sess.task = asyncio.create_task(self._run_loop(sess))

    def _should_fork(self, resume: str | None) -> bool:
        """Is the session we are about to resume already open somewhere else?

        Checked on EVERY start, not just when the user picks one from the list:
        a restart restores conversations too, and by then the session may have
        been opened in VS Code. Two processes on one .jsonl silently lose each
        other's last messages, so the only safe answer is to branch.
        """
        if not resume:
            return False
        try:
            live = claude_history.live_sessions(claude_sessions_path(self._cfg))
        except OSError:  # pragma: no cover - unreadable registry
            logger.exception("could not read the live-session registry")
            return False
        return resume in live

    async def _ensure_topic(self, key: str) -> None:
        """Give a conversation its sub-topic if it has none yet.

        Every start goes through here, not just a fresh ``open``: a conversation
        carried over from an older database — or one whose topic could not be
        created last time — would otherwise keep posting into General forever.
        """
        if self._on_open is None:
            return
        conv = await self._store.conversation(key)
        if conv is None or conv.thread_id is not None:
            return
        await self._on_open(key, conv.project, conv.ordinal)

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

    async def _stop(self, key: str) -> None:
        sess = self._sessions.pop(key, None)
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
                logger.exception("session %s task raised during stop", key)
        if sess.client is not None:
            try:
                await sess.client.disconnect()
            except Exception:  # pragma: no cover - defensive
                logger.exception("session %s disconnect failed", key)

    async def _run_loop(self, sess: _Session) -> None:
        """Drain the queue, forward turns to the SDK, emit output.

        Wrapped per-turn in try/except (C8): a crashing turn emits an error
        Outbound and stops this conversation without affecting any other.
        """
        key = sess.key
        assert sess.client is not None
        client = sess.client
        while True:
            text = await sess.queue.get()
            if text is _SHUTDOWN:
                return
            try:
                await self._begin_turn(sess)
                await append_transcript(sess.project.cwd, "user", text)
                await client.query(text)
                parts: list[str] = []
                async for msg in client.receive_response():
                    await self._consume(sess, msg, parts)
                await self._end_turn(sess)
                joined = "\n".join(p for p in parts if p).strip()
                if joined:
                    await append_transcript(sess.project.cwd, "assistant", joined)
                    await self._on_outbound(
                        Outbound(
                            project=key,
                            text=with_model_footer(joined, sess.model, sess.effort),
                            spoken="",
                        )
                    )
            except asyncio.CancelledError:
                sess.progress = None
                raise
            except Exception as err:  # noqa: BLE001 - C8: never crash the service
                logger.exception("session %s crashed on a turn", key)
                sess.progress = None
                await self._emit_crash(sess, err)
                return

    async def _consume(self, sess: _Session, msg, parts: list[str]) -> None:
        """Fold one streamed SDK message into the turn's text and progress log."""
        progress = sess.progress
        if isinstance(msg, AssistantMessage):
            sess.model = getattr(msg, "model", None) or sess.model
            changed = False
            for block in msg.content:
                if isinstance(block, TextBlock):
                    parts.append(block.text)
                elif progress is None:
                    continue
                elif isinstance(block, ToolUseBlock):
                    progress.add_tool(block)
                    changed = True
                elif isinstance(block, ThinkingBlock):
                    progress.add_thinking()
                    changed = True
            if changed:
                await self._push_progress(sess)
        elif isinstance(msg, UserMessage) and progress is not None:
            content = msg.content
            if isinstance(content, list):
                for block in content:
                    if isinstance(block, ToolResultBlock):
                        progress.mark_result(block)
                await self._push_progress(sess)
        elif isinstance(msg, SystemMessage) and msg.subtype == "init":
            # The CLI names the model it connected with; that beats guessing
            # from settings.json, and it lands before the first text block.
            sess.model = msg.data.get("model") or sess.model
        elif isinstance(msg, ResultMessage):
            session_id = getattr(msg, "session_id", None)
            if session_id:
                await self._store.set_conversation_session(sess.key, session_id)

    async def _begin_turn(self, sess: _Session) -> None:
        if not self._cfg.stream_progress or self._on_progress is None:
            await self._emit_status(sess.key, "Working.")
            return
        sess.progress = _Progress(started=time.monotonic())
        await self._push_progress(sess, force=True)

    async def _end_turn(self, sess: _Session) -> None:
        progress, sess.progress = sess.progress, None
        if progress is None or self._on_progress is None:
            return
        elapsed = time.monotonic() - progress.started
        try:
            await self._on_progress(sess.key, progress.render_final(elapsed), True)
        except Exception:  # noqa: BLE001 - cosmetic; the answer still ships
            logger.debug("final progress update failed for %s", sess.key, exc_info=True)

    async def _push_progress(self, sess: _Session, force: bool = False) -> None:
        """Send the live activity view, rate-limited and de-duplicated.

        Telegram will not take an edit per tool call on a busy turn, so an
        unchanged render or one inside ``stream_interval`` is dropped — the same
        trade-off a terminal mirror makes when it polls instead of following.
        """
        progress = sess.progress
        if progress is None or self._on_progress is None:
            return
        now = time.monotonic()
        text = progress.render(now - progress.started)
        if not force:
            if now - progress.last_sent < self._cfg.stream_interval:
                return
            if text == progress.last_hash:
                return
        progress.last_sent = now
        progress.last_hash = text
        try:
            await self._on_progress(sess.key, text, False)
        except Exception:  # noqa: BLE001 - progress is cosmetic, never fatal
            logger.debug("progress update failed for %s", sess.key, exc_info=True)

    async def _emit_crash(self, sess: _Session, err: Exception) -> None:
        key = sess.key
        # Mark the conversation stopped without a re-entrant cancel of this task.
        self._sessions.pop(key, None)
        if sess.client is not None:
            try:
                await sess.client.disconnect()
            except Exception:  # pragma: no cover - defensive
                logger.exception("session %s disconnect after crash failed", key)
        await append_transcript(sess.project.cwd, "system", f"Sesija krito: {err}")
        try:
            await self._on_outbound(
                Outbound(project=key, text=f"Sesija krito: {err}", spoken=_ERROR_SPOKEN)
            )
        except Exception:  # pragma: no cover - defensive
            logger.exception("failed to emit crash Outbound for %s", key)

    async def _emit_status(self, key: str, text: str) -> None:
        await self._on_outbound(Outbound(project=key, text=text, spoken=_SILENT_SPOKEN))


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
