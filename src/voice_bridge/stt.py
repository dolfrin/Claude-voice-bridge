"""Speech-to-text via a remote Paprika ASR service, or local faster-whisper.

Accepts Telegram OGG/Opus voice bytes and returns a transcript. When
``PAPRIKA_URL`` is set the audio is posted to that service and nothing is
transcribed on this machine; otherwise a local faster-whisper model is loaded.
The blocking local model load and inference run off the event loop in a worker
thread so the single asyncio loop is never blocked. Language is auto-detected by
default.
"""

from __future__ import annotations

import asyncio
import os
import tempfile


class Transcriber:
    """OGG/Opus -> text, remote over HTTP or local via faster-whisper."""

    def __init__(self, model_name: str, language: str | None = None) -> None:
        self.model_name = model_name
        self.language = language
        self._model: object | None = None

    def _get_model(self) -> object:
        if self._model is None:
            from faster_whisper import WhisperModel

            self._model = WhisperModel(self.model_name)
        return self._model

    def _transcribe_sync(self, audio: bytes) -> str:
        fd, path = tempfile.mkstemp(suffix=".ogg")
        try:
            with os.fdopen(fd, "wb") as fh:
                fh.write(audio)
            kwargs = {"language": self.language} if self.language else {}
            segments, _info = self._get_model().transcribe(path, **kwargs)
            text = "".join(segment.text for segment in segments)
            return text.strip()
        finally:
            try:
                os.remove(path)
            except OSError:
                pass

    async def _remote_results(self, base_url: str, audio: bytes) -> list[dict]:
        import httpx

        token = os.environ.get("PAPRIKA_TOKEN", "")
        timeout = float(os.environ.get("PAPRIKA_TIMEOUT", "120"))
        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.post(
                f"{base_url}/transcribe",
                headers={"Authorization": f"Bearer {token}"},
                files={"file": ("voice.ogg", audio, "audio/ogg")},
            )
        response.raise_for_status()
        # One entry per model loaded on the server; the first is the primary.
        # Order is decided by PAPRIKA_MODELS there, not here.
        return [
            {"model": r.get("model") or "?", "text": (r.get("text") or "").strip()}
            for r in (response.json().get("results") or [])
        ]

    async def transcribe_all(self, audio: bytes) -> list[dict]:
        """Every available transcript for ``audio``, primary first.

        A local model yields exactly one; the remote service yields one per
        model it has loaded, which is what lets the caller offer a choice.
        """
        base_url = os.environ.get("PAPRIKA_URL", "").rstrip("/")
        if base_url:
            return await self._remote_results(base_url, audio)
        loop = asyncio.get_running_loop()
        text = await loop.run_in_executor(None, self._transcribe_sync, audio)
        return [{"model": self.model_name, "text": text}]

    async def transcribe(self, audio: bytes) -> str:
        """Transcribe OGG/Opus ``audio`` bytes to text (off the event loop)."""
        results = await self.transcribe_all(audio)
        return results[0]["text"] if results else ""
