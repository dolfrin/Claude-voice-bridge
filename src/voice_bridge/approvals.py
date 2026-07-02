"""Risk classification, yes/no parsing, voice-approval futures, canUseTool factory."""

from __future__ import annotations

import asyncio
import logging
import os
import re
from pathlib import Path
from typing import Awaitable, Callable

from voice_bridge.config import Config, ProjectConfig, effective_autonomy

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
    r"(?:\.env\b|\.pem\b|id_rsa|id_ed25519|\.ssh/|\.aws/|\.gnupg|credentials|"
    r"\.netrc|\.git-credentials|secret|token|password|\.key\b)"
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


def _resolve(path: str, cwd: str) -> str:
    """Resolve `path` against `cwd` to a real, symlink-free absolute path.

    Uses realpath resolution (not lexical normalization) so a symlink that
    lives inside cwd but points outside of it is not mistaken for an
    in-cwd path. Non-existent paths are still resolved (as far as their
    existing ancestors allow) so a Write to a brand-new file stays safe.
    """
    return str(Path(cwd, path).resolve())


def _inside_cwd(path: str, cwd: str) -> bool:
    base = str(Path(cwd).resolve())
    target = _resolve(path, base)
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


def _format_question(project: str, tool_name: str, tool_input: dict) -> str:
    command = tool_input.get("command")
    if isinstance(command, str) and command.strip():
        action = command.strip()
    else:
        path = ""
        for key in _PATH_INPUT_KEYS:
            value = tool_input.get(key)
            if isinstance(value, str) and value:
                path = value
                break
        action = f"{tool_name} {path}".strip() if path else tool_name
    return f"{project} wants to run: {action}. Allow?"


class ApprovalManager:
    """Holds asyncio futures keyed by the question message_id; timeout -> deny."""

    def __init__(
        self,
        send_question: Callable[[str, str], Awaitable[int]],
        timeout: int,
    ) -> None:
        self._send_question = send_question
        self._timeout = timeout
        self._pending: dict[int, asyncio.Future[bool]] = {}

    async def request(self, project: str, tool_name: str, tool_input: dict) -> bool:
        """Ask user for permission; returns True if approved, False if denied or timed out."""
        text = _format_question(project, tool_name, tool_input)
        message_id = await self._send_question(project, text)
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
        try:
            return await asyncio.wait_for(future, timeout=self._timeout)
        except asyncio.TimeoutError:
            return False
        finally:
            self._pending.pop(message_id, None)

    def resolve(self, message_id: int, approved: bool) -> bool:
        """Set the result of a pending future. Returns True if matched, False otherwise."""
        future = self._pending.get(message_id)
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
