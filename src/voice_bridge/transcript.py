"""Project-local Markdown transcript mirror for Telegram bridge turns."""
from __future__ import annotations

import asyncio
import logging
from datetime import datetime
from pathlib import Path

_TRANSCRIPT_PATH = Path(".claude") / "voice-bridge-chat.md"
logger = logging.getLogger(__name__)


async def append_transcript(cwd: str, role: str, text: str) -> None:
    """Append one bridge-visible chat turn under the project directory."""
    if not text.strip():
        return
    try:
        await asyncio.get_running_loop().run_in_executor(
            None, _append_sync, cwd, role, text
        )
    except OSError:
        logger.exception("failed to append voice bridge transcript for %s", cwd)


def transcript_path(cwd: str) -> Path:
    """Return the Markdown transcript path for a project cwd."""
    return Path(cwd) / _TRANSCRIPT_PATH


def _append_sync(cwd: str, role: str, text: str) -> None:
    path = transcript_path(cwd)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        path.write_text(
            "# Voice Bridge Chat\n\n"
            "This file mirrors Telegram voice/text turns for IDE visibility.\n\n",
            encoding="utf-8",
        )
    stamp = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S %Z")
    label = _label(role)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(f"## {stamp} - {label}\n\n")
        fh.write(text.strip())
        fh.write("\n\n")


def _label(role: str) -> str:
    if role == "user":
        return "Telegram"
    if role == "assistant":
        return "Claude"
    return role.title()
