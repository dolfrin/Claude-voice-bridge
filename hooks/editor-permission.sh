#!/usr/bin/env bash
# Voice Bridge — editor permission hook.
# Registered in ~/.claude/settings.json under PermissionRequest (timeout 3600).
# For a VS Code session while the bridge runs, it puts the prompt on Telegram
# with ✅/❌ buttons and waits; the editor's own dialog stays usable meanwhile
# (see voice_bridge.editor_permission). Anything else -- a terminal session,
# the bridge down, an error -- gets the plain notification as before.
# It must NEVER block the editor by failing: on any error it prints nothing
# and the editor decides.
PY="/home/home/Projects/claude-voice-bridge/.venv/bin/python"
INPUT=$(cat)
if [ -x "$PY" ]; then
    OUT=$(printf '%s' "$INPUT" | "$PY" -m voice_bridge.editor_permission 2>>"$HOME/.claude/perm-debug.log")
    if [ $? -eq 0 ]; then
        printf '%s' "$OUT"
        exit 0
    fi
fi
printf '%s' "$INPUT" | "$HOME/.claude/notify-question.sh"
exit 0
