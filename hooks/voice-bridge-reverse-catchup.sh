#!/usr/bin/env bash
# Voice Bridge — reverse catch-up hook.
# Registered in ~/.claude/settings.json under SessionStart + UserPromptSubmit.
# On a Claude Code session start (or a prompt in an already-open session) it
# injects a summary of what the Telegram bridge did in THIS project since the
# last time it was seen (dedup-gated), as read-only reference context.
# It must NEVER fail a session start: it always exits 0 and prints nothing on
# any error or when there is no new bridge activity.
PY="/home/home/Projects/claude-voice-bridge/.venv/bin/python"
[ -x "$PY" ] || exit 0
"$PY" -m voice_bridge.catchup --hook 2>/dev/null || true
exit 0
