"""Speech-to-text via faster-whisper.

Accepts Telegram OGG/Opus voice bytes and returns a transcript. The blocking
faster-whisper model load and inference run off the event loop in a worker
thread so the single asyncio loop is never blocked. Default language is
Lithuanian (``lt``).
"""

from __future__ import annotations

import asyncio
import os
import tempfile

from faster_whisper import WhisperModel


class Transcriber:
    """Wraps a faster-whisper model for OGG/Opus -> text transcription."""

    def __init__(self, model_name: str, language: str = "lt") -> None:
        self.model_name = model_name
        self.language = language
        self._model: WhisperModel | None = None

    def _get_model(self) -> WhisperModel:
        if self._model is None:
            self._model = WhisperModel(self.model_name)
        return self._model

    def _transcribe_sync(self, audio: bytes) -> str:
        fd, path = tempfile.mkstemp(suffix=".ogg")
        try:
            with os.fdopen(fd, "wb") as fh:
                fh.write(audio)
            segments, _info = self._get_model().transcribe(
                path, language=self.language
            )
            text = "".join(segment.text for segment in segments)
            return text.strip()
        finally:
            try:
                os.remove(path)
            except OSError:
                pass

    async def transcribe(self, audio: bytes) -> str:
        """Transcribe OGG/Opus ``audio`` bytes to text (off the event loop)."""
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self._transcribe_sync, audio)
