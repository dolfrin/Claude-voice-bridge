"""Risk classification, yes/no parsing, voice-approval futures, canUseTool factory."""

from __future__ import annotations

import asyncio
import logging
import os
import re
from pathlib import Path
from typing import Awaitable, Callable

from voice_bridge.config import Config, ProjectConfig, effective_autonomy
from voice_bridge.notify_tool import SEND_FILE_TOOL_NAME

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Risk classification
# ---------------------------------------------------------------------------

# Bash command verbs / phrases that are always risky.
_RISKY_COMMAND_PATTERNS = [
    re.compile(r"\bgit\s+push\b"),
    re.compile(r"\brm\b"),
    re.compile(r"\bssh\b"),
    re.compile(r"\bscp\b"),
    re.compile(r"\bsftp\b"),
    re.compile(r"\bnc\b"),
    re.compile(r"\bnetcat\b"),
    re.compile(r"\bncat\b"),
    re.compile(r"\bmv\b"),
    re.compile(r"\bchmod\b"),
    re.compile(r"\bchown\b"),
    re.compile(r"\brsync\b"),
    re.compile(r"\bdeploy\b"),
    re.compile(r"\bvercel\b"),
    re.compile(r"\bnetlify\b"),
    re.compile(r"\bkubectl\b"),
    re.compile(r"\bdocker\s+push\b"),
    re.compile(r"\bterraform\s+apply\b"),
    re.compile(r"\bnpm\s+install\b"),
    re.compile(r"\byarn\s+add\b"),
    re.compile(r"\bpnpm\s+(add|install)\b"),
    re.compile(r"\bpip\s+install\b"),
    re.compile(r"\bapt(-get)?\s+install\b"),
    re.compile(r"\bbrew\s+install\b"),
    re.compile(r"\bsnap\s+install\b"),
    re.compile(r"\bcurl\b.*\|"),
    re.compile(r"\bwget\b.*\|"),
    re.compile(r"\bwallet\b"),
    re.compile(r"\bsend\b.*\b(eth|btc|usdc|sol)\b"),
    re.compile(r"\bmkfs\b"),
    re.compile(r"\bdd\b\s+if="),
    re.compile(r"\bshutdown\b"),
    re.compile(r"\breboot\b"),
    re.compile(r"\bsystemctl\b"),
    # --- Data-exfiltration: curl/wget with a spelled-out upload/POST-style
    # payload flag. Long flags are unambiguous so this is safe to match
    # case-insensitively (checked against `lowered` below).
    re.compile(
        r"\b(curl|wget)\b.*("
        r"--data-binary\b|--data-raw\b|--data\b|"
        r"--form\b|--upload-file\b|"
        r"--post-data\b|--post-file\b|"
        r"--request[=\s]+(post|put)\b"
        r")"
    ),
]

# Single-letter curl exfil flags (-d, -F, -T, -X POST/PUT) are matched
# case-SENSITIVELY against the original (non-lowered) command, because
# curl assigns unrelated meanings to the opposite case: -D dumps headers
# (not -d, which sends data), -f fails silently on HTTP errors (not -F,
# multipart form), -x sets a proxy (not -X, custom request method). Case
# sensitivity here avoids flagging everyday safe curl usage like `curl -f`.
_EXFIL_SHORT_FLAG_RE = re.compile(
    r"\b(?:curl|wget)\b.*(-d\b|-F\b|-T\b|-X\s*(?:POST|PUT)\b)"
)

# Reader/dump commands that, combined with a sensitive-looking path, should
# be flagged even though the bare command (e.g. `cat README.md`) is safe.
_SENSITIVE_TOKEN_RE = (
    # `.env` (and `.env.local` etc.) is sensitive, but the committed
    # `.env.example`/`.sample`/`.template`/`.dist` templates are not.
    r"(?:\.env(?!\.(?:example|sample|template|dist))\b|\.pem\b|id_rsa|id_ed25519|"
    r"\.ssh/|\.aws/|\.gnupg|credentials|\.netrc|\.git-credentials|secret|token|"
    r"password|\.key\b)"
)
_READER_CMD_RE = r"\b(?:cat|less|more|head|tail|xxd|od|base64|strings|grep|awk|sed|cp|mv)\b"

_RISKY_COMMAND_PATTERNS.append(
    re.compile(rf"{_READER_CMD_RE}.*{_SENSITIVE_TOKEN_RE}", re.IGNORECASE)
)

# Redirection target: the first non-whitespace token after `>`/`>>`, stopping
# at shell metacharacters that would end the token (pipe, semicolon, `&`,
# another redirection). This intentionally excludes fd-duplication targets
# like `>&1` — those aren't file paths.
_REDIRECT_TARGET_RE = re.compile(r">>?\s*([^\s|;&<>]+)")

# Pseudo-files that discard/duplicate output rather than persist or exfil
# data. `cmd > /dev/null 2>&1` is one of the most common shell idioms and
# would otherwise be flagged purely for being an absolute path.
_SAFE_REDIRECT_TARGETS = {"/dev/null", "/dev/stdout", "/dev/stderr"}


def _has_risky_redirect(command: str) -> bool:
    """Flag `>`/`>>` redirection to an absolute path, a path that escapes
    cwd (`../`), or a sensitive-looking target. Plain relative targets like
    `> out.txt` stay safe, and so do the standard /dev/null-style sinks."""
    for match in _REDIRECT_TARGET_RE.finditer(command):
        target = match.group(1)
        if target in _SAFE_REDIRECT_TARGETS:
            continue
        if target.startswith("/"):
            return True
        if target.startswith("..") or "../" in target:
            return True
        if re.search(_SENSITIVE_TOKEN_RE, target, re.IGNORECASE):
            return True
    return False


# Tools that touch the filesystem with an explicit path.
_PATH_INPUT_KEYS = ("file_path", "path", "notebook_path")

# send_file's input schema uses "path" (see notify_tool._build_send_file_tool),
# but both keys are checked so this stays robust to a caller-side rename.
_SEND_FILE_PATH_KEYS = ("path", "file_path")


def _resolve(path: str, cwd: str) -> str:
    """Resolve `path` against `cwd` to a real, symlink-free absolute path.

    Uses realpath resolution (not lexical normalization) so a symlink that
    lives inside cwd but points outside of it is not mistaken for an
    in-cwd path. Non-existent paths are still resolved (as far as their
    existing ancestors allow) so a Write to a brand-new file stays safe.
    """
    return str(Path(cwd, path).resolve())


def _inside_cwd(path: str, cwd: str) -> bool:
    try:
        base = str(Path(cwd).resolve())
        target = _resolve(path, base)
    except (OSError, RuntimeError, ValueError):
        # Can't prove containment (symlink loop, unreadable link, bad path).
        # Fail closed: treat as outside cwd so the action is flagged risky
        # (asked in safe mode) rather than letting the exception break the turn.
        return False
    return target == base or target.startswith(base + os.sep)


def is_risky(tool_name: str, tool_input: dict, cwd: str) -> bool:
    """Return True if the tool call is risky: push/deploy/rm/ssh/install/out-of-cwd/
    wallet/exfiltration/secret-read/sensitive-redirect."""
    command = tool_input.get("command")
    if isinstance(command, str):
        lowered = command.lower()
        if any(p.search(lowered) for p in _RISKY_COMMAND_PATTERNS):
            return True
        if _EXFIL_SHORT_FLAG_RE.search(command):
            return True
        if _has_risky_redirect(command):
            return True

    for key in _PATH_INPUT_KEYS:
        path = tool_input.get(key)
        if isinstance(path, str) and path:
            if not _inside_cwd(path, cwd):
                return True

    # send_file uploads a project file to the user's Telegram — it's an
    # egress channel, not a local read/write, so an in-cwd path is not
    # automatically safe the way it is for Read/Write/Edit: `.env`, keys,
    # and credentials live inside cwd too. Flag it if the target LOOKS
    # sensitive (same token regex as the Bash reader-command check above)
    # even though there is no `command` key here to match against, OR if it
    # resolves outside cwd (belt-and-suspenders; already caught above for
    # the "path"/"file_path" keys, but explicit here for clarity/robustness).
    #
    # The token regex alone is trivially bypassed by an innocuously-named
    # symlink/hardlink that points at a sensitive file: `ln -s .env
    # innocuous.txt` sits inside cwd (passes containment) and its own name
    # doesn't match the sensitive-token regex (passes the raw check), so the
    # RAW string is not enough — the target must be resolved (symlink-
    # followed) once and checked too. A symlink's resolved path reveals the
    # real name (`.env`); a HARDLINK's resolved path does NOT change (a hard
    # link is just another directory entry for the same inode, not a
    # pointer), so it is instead caught via `st_nlink > 1` — a project file
    # sent via send_file essentially never legitimately has multiple hard
    # links, so this stays fail-safe (over-flagging is acceptable; a silent
    # bypass is not).
    if tool_name == SEND_FILE_TOOL_NAME:
        for key in _SEND_FILE_PATH_KEYS:
            target = tool_input.get(key)
            if isinstance(target, str) and target:
                if re.search(_SENSITIVE_TOKEN_RE, target, re.IGNORECASE):
                    return True
                try:
                    base = str(Path(cwd).resolve())
                    resolved = _resolve(target, base)
                except (OSError, RuntimeError, ValueError):
                    # Can't prove anything about the target (symlink loop,
                    # unreadable link, bad path) — fail closed to risky.
                    return True
                if re.search(_SENSITIVE_TOKEN_RE, resolved, re.IGNORECASE):
                    return True
                try:
                    if os.stat(resolved).st_nlink > 1:
                        return True
                except OSError:
                    # Doesn't exist (yet) or unreadable: nothing to alias.
                    pass
                if not (resolved == base or resolved.startswith(base + os.sep)):
                    return True

    return False


# ---------------------------------------------------------------------------
# Yes/No parsing
# ---------------------------------------------------------------------------

_YES_WORDS = {
    "ok", "okay", "yes", "yep", "yeah", "y", "sure", "go", "allow",
    "approve", "approved",
}
_NO_WORDS = {
    "stop", "no", "nope", "n", "cancel", "deny", "denied",
}

_TOKEN_RE = re.compile(r"[a-z]+", re.IGNORECASE)


def parse_yes_no(text: str) -> bool | None:
    """Return True for yes, False for no, None if undecidable."""
    if not text or not text.strip():
        return None
    tokens = _TOKEN_RE.findall(text.lower())
    if not tokens:
        return None
    saw_yes = any(t in _YES_WORDS for t in tokens)
    saw_no = any(t in _NO_WORDS for t in tokens)
    if saw_no and not saw_yes:
        return False
    if saw_yes and not saw_no:
        return True
    return None


# ---------------------------------------------------------------------------
# Pending-approval manager
# ---------------------------------------------------------------------------


# Max characters of file content / edit strings shown inline in the preview.
_PREVIEW_SNIPPET = 400
_PREVIEW_EDIT = 200
_PREVIEW_VALUE = 120
_TRUNCATE_MARKER = "…"

# Generic, code-free spoken verb per tool. The spoken channel never names the
# command or path (that would leak code into voice); it stays a short action so
# a walking user hears *what kind* of thing is being asked, then reads the text.
_SPOKEN_ACTIONS = {
    "Bash": "paleisti komandą",
    "Write": "įrašyti failą",
    "Edit": "redaguoti failą",
    "MultiEdit": "redaguoti failą",
    "Read": "perskaityti failą",
    "NotebookEdit": "redaguoti užrašinę",
}
# Tool-input keys worth surfacing (in order) for tools without a bespoke branch.
_OTHER_PREVIEW_KEYS = (
    "file_path", "path", "notebook_path", "pattern", "url",
    "command", "query", "prompt", "description",
)


def _truncate(text, limit: int) -> str:
    # Coerce defensively: model-generated tool_input may put non-str values in
    # old_string/new_string/content, and a preview must never raise into the
    # permission flow.
    text = str(text) if text else ""
    if len(text) <= limit:
        return text
    return text[:limit].rstrip() + _TRUNCATE_MARKER


def _first_path(tool_input: dict) -> str:
    for key in _PATH_INPUT_KEYS:
        value = tool_input.get(key)
        if isinstance(value, str) and value:
            return value
    return ""


def format_approval_preview(tool_name: str, tool_input: dict) -> str:
    """Render a compact, human-readable preview of *what* a tool call will do.

    This is the TEXT side of an approval (the user reads it to decide); it may
    contain code/paths. The spoken side stays code-free (see
    :func:`format_approval_spoken`).

    * Bash -> the command in a code block.
    * Write -> file_path + a short snippet of content.
    * Edit / MultiEdit -> file_path + a compact ``old → new`` preview.
    * other tools -> tool name + a compact repr of the key inputs.
    """
    if tool_name == "Bash":
        command = (tool_input.get("command") or "").strip()
        return f"```\n{command}\n```" if command else "Bash"

    if tool_name == "Write":
        path = _first_path(tool_input)
        snippet = _truncate(tool_input.get("content") or "", _PREVIEW_SNIPPET)
        if snippet:
            return f"{path}\n```\n{snippet}\n```"
        return path or "Write"

    if tool_name in ("Edit", "MultiEdit"):
        path = _first_path(tool_input)
        if tool_name == "MultiEdit":
            edits = tool_input.get("edits") or []
            first = edits[0] if edits else None
            if not isinstance(first, dict):
                return path or "MultiEdit"
            old = _truncate(first.get("old_string") or "", _PREVIEW_EDIT)
            new = _truncate(first.get("new_string") or "", _PREVIEW_EDIT)
            more = f" (+{len(edits) - 1} more)" if len(edits) > 1 else ""
            return f"{path}\n{old} → {new}{more}"
        old = _truncate(tool_input.get("old_string") or "", _PREVIEW_EDIT)
        new = _truncate(tool_input.get("new_string") or "", _PREVIEW_EDIT)
        return f"{path}\n{old} → {new}"

    parts = [
        f"{key}={_truncate(str(tool_input[key]), _PREVIEW_VALUE)}"
        for key in _OTHER_PREVIEW_KEYS
        if tool_input.get(key)
    ]
    if parts:
        return f"{tool_name}: " + ", ".join(parts)
    return f"{tool_name}: {_truncate(repr(tool_input), _PREVIEW_EDIT)}"


def format_approval_spoken(project: str, tool_name: str, tool_input: dict) -> str:
    """Return the code-free spoken approval line (no command/path leaks)."""
    action = _SPOKEN_ACTIONS.get(tool_name, "atlikti veiksmą")
    return f"{project} nori {action} — leidžiu?"


def _format_question(project: str, preview: str) -> str:
    """Build the approval message TEXT (carries the preview for the user)."""
    return f"{project} — approval reikalingas:\n\n{preview}"


class ApprovalManager:
    """Holds asyncio futures keyed by BOTH a stable approval token and the
    question message_id; timeout -> deny.

    The token is an incrementing per-manager int generated *before* the question
    is sent (the inline Allow/Deny buttons need it at send time, but the
    message_id is only known *after* send). Registering the same future under
    both keys lets an inline-button tap resolve by token
    (:meth:`resolve_token`) and a quote-reply resolve by message_id
    (:meth:`resolve`) — either path resolves the same waiter, and both are
    idempotent (a second resolve is a no-op).
    """

    def __init__(
        self,
        send_question: Callable[[str, str, str, int], Awaitable[int]],
        timeout: int,
    ) -> None:
        self._send_question = send_question
        self._timeout = timeout
        # Same future object stored in both maps for dual-key resolution. The
        # numeric spaces are disjoint by construction only in intent (a token
        # and a message_id could coincide), so they are kept in SEPARATE dicts.
        self._pending: dict[int, asyncio.Future[bool]] = {}
        self._pending_by_token: dict[int, asyncio.Future[bool]] = {}
        self._token_counter = 0

    def _next_token(self) -> int:
        self._token_counter += 1
        return self._token_counter

    async def request(self, project: str, tool_name: str, tool_input: dict) -> bool:
        """Ask user for permission; returns True if approved, False if denied or timed out."""
        preview = format_approval_preview(tool_name, tool_input)
        text = _format_question(project, preview)
        spoken = format_approval_spoken(project, tool_name, tool_input)
        token = self._next_token()
        message_id = await self._send_question(project, text, spoken, token)
        loop = asyncio.get_running_loop()
        future: asyncio.Future[bool] = loop.create_future()
        existing = self._pending.get(message_id)
        if existing is not None and not existing.done():
            logger.warning(
                "ApprovalManager: duplicate message_id %s — resolving old future to False",
                message_id,
            )
            existing.set_result(False)
        self._pending[message_id] = future
        self._pending_by_token[token] = future
        try:
            return await asyncio.wait_for(future, timeout=self._timeout)
        except asyncio.TimeoutError:
            return False
        finally:
            self._pending.pop(message_id, None)
            self._pending_by_token.pop(token, None)

    def resolve(self, message_id: int, approved: bool) -> bool:
        """Resolve a pending future by message_id (quote-reply path).

        Returns True if a live future was resolved, False otherwise (unknown or
        already resolved)."""
        return self._resolve_future(self._pending.get(message_id), approved)

    def resolve_token(self, token: int, approved: bool) -> bool:
        """Resolve a pending future by approval token (inline-button path).

        Returns True if a live future was resolved, False otherwise (unknown or
        already resolved). A stale token (already answered / timed out) -> False
        so the caller can show a "no longer relevant" toast instead of crashing.
        """
        return self._resolve_future(self._pending_by_token.get(token), approved)

    @staticmethod
    def _resolve_future(future: asyncio.Future[bool] | None, approved: bool) -> bool:
        if future is None or future.done():
            return False
        future.set_result(approved)
        return True

    def has_pending(self, message_id: int) -> bool:
        """Return True if there is an unresolved future for this message_id."""
        future = self._pending.get(message_id)
        return future is not None and not future.done()


# ---------------------------------------------------------------------------
# canUseTool factory
# ---------------------------------------------------------------------------


def make_can_use_tool(
    project: ProjectConfig,
    cfg: Config,
    manager: "ApprovalManager",
) -> Callable:
    """Build an SDK canUseTool callback honoring effective_autonomy.

    full -> allow all (no question); ask -> request all; safe -> request only risky.
    Signature follows the claude-agent-sdk C12 API:
        async def can_use_tool(tool_name, tool_input, context) -> Allow | Deny
    """
    from claude_agent_sdk import PermissionResultAllow, PermissionResultDeny  # noqa: PLC0415

    mode = effective_autonomy(project, cfg)

    async def can_use_tool(tool_name: str, tool_input: dict, context):
        if mode == "full":
            return PermissionResultAllow()

        if mode == "safe" and not is_risky(tool_name, tool_input, project.cwd):
            return PermissionResultAllow()

        # mode == "ask", or mode == "safe" with a risky tool
        approved = await manager.request(project.name, tool_name, tool_input)
        if approved:
            return PermissionResultAllow()
        return PermissionResultDeny(message="User denied or timed out")

    return can_use_tool
