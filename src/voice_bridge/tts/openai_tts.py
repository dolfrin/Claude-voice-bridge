"""OpenAI TTS backend."""
from __future__ import annotations

import asyncio

from openai import OpenAI

_MODEL = "gpt-4o-mini-tts-2025-12-15"
_INSTRUCTIONS = (
    "Speak Lithuanian naturally, like a calm human assistant in a private voice "
    "message. Avoid announcer, robotic, overly formal, or synthetic intonation. "
    "Use natural pacing, warm tone, and clear articulation."
)


class OpenAITTS:
    """OpenAI text-to-speech, emitting OGG/Opus bytes."""

    def __init__(self, api_key: str) -> None:
        self._client = OpenAI(api_key=api_key)

    async def synthesize(self, text: str, voice: str) -> bytes:
        def _call() -> bytes:
            response = self._client.audio.speech.create(
                model=_MODEL,
                voice=voice,
                input=text,
                instructions=_INSTRUCTIONS,
                response_format="opus",
            )
            return response.read()

        return await asyncio.get_running_loop().run_in_executor(None, _call)
