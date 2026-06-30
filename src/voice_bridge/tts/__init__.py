"""TTS backend protocol, factory, and voice listing."""
from __future__ import annotations

from typing import Protocol, runtime_checkable

from voice_bridge.config import Config

_OPENAI_VOICES = [
    "alloy",
    "ash",
    "ballad",
    "cedar",
    "coral",
    "echo",
    "marin",
    "sage",
    "shimmer",
    "verse",
]


@runtime_checkable
class TTSBackend(Protocol):
    """A text-to-speech backend that emits OGG/Opus bytes."""

    async def synthesize(self, text: str, voice: str) -> bytes:
        """Return OGG/Opus-encoded audio for ``text`` in ``voice``."""
        ...


def get_tts(cfg: Config) -> TTSBackend:
    """Construct the configured TTS backend, dispatching on ``cfg.tts_backend``."""
    backend = cfg.tts_backend
    if backend == "openai":
        from voice_bridge.tts.openai_tts import OpenAITTS

        return OpenAITTS(cfg.openai_api_key)
    if backend == "piper":
        from voice_bridge.tts.piper_tts import PiperTTS

        return PiperTTS(cfg.piper_voice_path)
    raise ValueError(f"unknown TTS backend: {backend!r}")


def available_voices(backend: str) -> list[str]:
    """List selectable voices for ``backend`` (empty list if unknown)."""
    if backend == "openai":
        return list(_OPENAI_VOICES)
    if backend == "piper":
        return ["default"]
    return []
