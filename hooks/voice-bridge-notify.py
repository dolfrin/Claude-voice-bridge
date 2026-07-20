#!/usr/bin/env python3
"""Unified IDE-notification hook for the Claude Voice Bridge (I3).

Replaces the three per-event bash hooks (``notify-stop.sh``,
``notify-notification.sh``, ``notify-question.sh``) — each of which POSTed to
Telegram directly — with ONE Python script that instead drops the event into a
spool directory the bridge drains. Routing through the bridge is what gives IDE
notifications the same TTS, consistent formatting and project attribution the
bridge already gives project turns; a hook running in a separate Claude Code
process cannot speak on its own (the bridge owns TTS).

It runs on every relevant Claude Code hook event (Stop, Notification,
PreToolUse[AskUserQuestion], PermissionRequest — one settings.json wiring, see
README). It reads the hook JSON from stdin, classifies the event, formats a
plain-text message, computes a content hash for dedup, and writes ONE spool
file ATOMICALLY (``.tmp`` then ``os.rename``) so the drainer never sees a
half-written file.

Fire-and-forget contract: it must never block, modify or fail the tool call —
so it prints nothing and ALWAYS exits 0, wrapping every step in try/except.
If the bridge is down the spool just accumulates and drains when it returns.

Spool contract (mirrored by ``voice_bridge.inbox``):
* dir: ``$VOICE_BRIDGE_INBOX_DIR`` or ``~/.claude/.voice-bridge-inbox/``
* one JSON file ``<ts>-<rand>.json`` per event, fields:
  ``kind`` (question|permission|stop|notification), ``project`` (basename cwd),
  ``cwd``, ``text`` (preformatted plain text), ``urgent`` (bool — question/
  permission → True → spoken; stop/notification → False → text only),
  ``hash`` (content dedup), ``ts`` (epoch).
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
import time
import uuid
from pathlib import Path

# A permission hint / notification can be arbitrarily long; cap it so one event
# can never blow past Telegram's message limit once the bridge prefixes it.
_HINT_MAX = 300
_DESC_MAX = 160


def resolve_spool_dir() -> Path:
    """Spool directory: the env override, else ``~/.claude/.voice-bridge-inbox``.

    Read at call time (not import) so a test can point it at a tmp dir via the
    environment exactly as the real settings.json wiring would.
    """
    override = os.environ.get("VOICE_BRIDGE_INBOX_DIR")
    if override:
        return Path(override)
    return Path.home() / ".claude" / ".voice-bridge-inbox"


def _format_questions(tool_input: dict) -> str:
    """Format an AskUserQuestion's questions + numbered options as plain text.

    Mirrors the old bash hooks' rendering (``❓ question`` then ``1) label — desc``)
    but WITHOUT HTML escaping: the bridge sends this via a plain-text path (no
    ``parse_mode``), so raw ``<``/``>``/``&`` are fine and must be preserved.
    """
    out: list[str] = []
    for q in (tool_input.get("questions") or []):
        text = (q.get("question") or "").strip()
        if not text:
            continue
        out.append("❓ " + text)
        for i, o in enumerate(q.get("options") or [], 1):
            label = (o.get("label") or "").strip()
            desc = (o.get("description") or "").strip()
            if len(desc) > _DESC_MAX:
                desc = desc[:_DESC_MAX].rstrip() + "…"
            out.append(f"{i}) {label} — {desc}" if desc else f"{i}) {label}")
    return "\n".join(out)


def _format_permission(tool: str, tool_input: dict) -> str:
    """Format a PermissionRequest as ``🔐 Prašo leidimo: <tool>`` + a hint line.

    The hint is the first non-empty of command/description/file_path/prompt —
    the same precedence the old bash question hook used.
    """
    hint = ""
    for key in ("command", "description", "file_path", "prompt"):
        v = tool_input.get(key)
        if isinstance(v, str) and v.strip():
            hint = v.strip()
            break
    if len(hint) > _HINT_MAX:
        hint = hint[:_HINT_MAX].rstrip() + "…"
    line = f"🔐 Prašo leidimo: {tool or '?'}"
    return f"{line}\n{hint}" if hint else line


def classify(hook: dict) -> tuple[str, bool, str] | None:
    """Classify a hook event into ``(kind, urgent, text)`` or None (not spoolable).

    Precedence:

    * any event whose tool is ``AskUserQuestion`` → a full question (urgent),
      regardless of whether it arrived as PreToolUse or PermissionRequest;
    * ``PermissionRequest`` → a permission ask (urgent);
    * ``Notification`` → the generic message (not urgent);
    * ``Stop`` → "✅ baigė" (not urgent).

    A generic (non-Ask) ``PreToolUse`` or any other event returns None so we do
    not spool a notification on every tool call. ``urgent`` drives TTS in the
    bridge: only questions and permission asks are spoken.
    """
    event = hook.get("hook_event_name") or ""
    tool = hook.get("tool_name") or ""
    tool_input = hook.get("tool_input") or {}

    if tool == "AskUserQuestion":
        text = _format_questions(tool_input)
        return ("question", True, text) if text else None
    if event == "PermissionRequest":
        return ("permission", True, _format_permission(tool, tool_input))
    if event == "Notification":
        message = (hook.get("message") or "").strip()
        return ("notification", False, message) if message else None
    if event == "Stop":
        return ("stop", False, "✅ baigė")
    return None


def _content_hash(kind: str, project: str, text: str) -> str:
    """Stable content hash for dedup: identical events hash identically.

    Includes ``kind`` so a stop and a notification with the same text differ,
    and ``project`` so the same message from two repos is not collapsed.
    """
    raw = f"{kind}\x00{project}\x00{text}".encode("utf-8")
    return hashlib.sha256(raw).hexdigest()[:16]


def build_event(hook: dict) -> dict | None:
    """Build the full spool event dict from a hook payload, or None if nothing
    should be spooled for it."""
    classified = classify(hook)
    if classified is None:
        return None
    kind, urgent, text = classified
    cwd = hook.get("cwd") or os.getcwd()
    project = os.path.basename(cwd.rstrip("/")) or cwd
    return {
        "kind": kind,
        "project": project,
        "cwd": cwd,
        "text": text,
        "urgent": urgent,
        "hash": _content_hash(kind, project, text),
        "ts": time.time(),
    }


def write_event(event: dict, directory) -> Path:
    """Atomically write ``event`` as one JSON file into ``directory``.

    Writes to a ``.tmp`` sibling then ``os.rename``s to ``<ts>-<rand>.json`` so
    the drainer (which only globs ``*.json``) can never read a half-written
    file. The temp name ends in ``.tmp`` (not ``.json``) so it is invisible to
    that glob even for the instant before the rename.
    """
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    base = f"{event['ts']:.6f}-{uuid.uuid4().hex[:8]}"
    tmp = directory / f"{base}.tmp"
    final = directory / f"{base}.json"
    tmp.write_text(json.dumps(event, ensure_ascii=False), encoding="utf-8")
    os.rename(tmp, final)
    return final


def main() -> int:
    """Read the hook JSON from stdin, spool it, and ALWAYS return 0.

    Every step is guarded: a malformed payload, an unspoolable event, or a
    filesystem error must never crash — a hook that raises would surface an
    error in the user's IDE session. Prints nothing.
    """
    try:
        raw = sys.stdin.read()
    except Exception:
        return 0
    try:
        hook = json.loads(raw)
    except Exception:
        return 0
    if not isinstance(hook, dict):
        return 0
    try:
        event = build_event(hook)
        if event is not None:
            write_event(event, resolve_spool_dir())
    except Exception:
        return 0
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised via subprocess/import
    sys.exit(main())
