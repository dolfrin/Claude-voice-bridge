#!/usr/bin/env python3
"""Record which Claude Code session sent a Telegram message.

Usage: curl ... sendMessage | telegram-record-sent.py SESSION_ID CWD

The voice bridge reads ~/.claude/.voice-bridge-sent.jsonl so that replying to
this message -- or writing right after it -- goes back into the session that
sent it. Same line format as voice_bridge.sent_log.record. Never fails the hook.
"""
import json
import os
import sys
import time

try:
    mid = json.load(sys.stdin)["result"]["message_id"]
    sid = sys.argv[1] if len(sys.argv) > 1 and sys.argv[1] not in ("", "-") else None
    cwd = sys.argv[2] if len(sys.argv) > 2 and sys.argv[2] != "-" else ""
    line = json.dumps({"m": mid, "s": sid, "c": cwd, "t": time.time()})
    with open(os.path.expanduser("~/.claude/.voice-bridge-sent.jsonl"), "a") as fh:
        fh.write(line + "\n")
except Exception:
    pass
