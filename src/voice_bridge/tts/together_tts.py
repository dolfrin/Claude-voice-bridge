"""Together AI TTS backend."""
from __future__ import annotations

import asyncio
import json
import urllib.error
import urllib.request

_ENDPOINT = "https://api.together.ai/v1/audio/speech"
_RESPONSE_FORMAT = "mp3"


class TogetherTTS:
    """Together text-to-speech, emitting OGG/Opus bytes for Telegram voice."""

    def __init__(
        self,
        api_key: str,
        model: str = "cartesia/sonic",
        language: str = "lt",
    ) -> None:
        self._api_key = api_key
        self._model = model
        self._language = language

    async def synthesize(self, text: str, voice: str) -> bytes:
        if not self._api_key:
            raise RuntimeError("TOGETHER_API_KEY is required for TTS_BACKEND=together")
        mp3 = await asyncio.get_running_loop().run_in_executor(
            None, self._request_mp3, text, voice
        )
        return await _mp3_to_ogg_opus(mp3)

    def _request_mp3(self, text: str, voice: str) -> bytes:
        payload = {
            "model": self._model,
            "input": text,
            "voice": voice,
            "response_format": _RESPONSE_FORMAT,
        }
        if self._language:
            payload["language"] = self._language
        body = json.dumps(payload).encode("utf-8")
        request = urllib.request.Request(
            _ENDPOINT,
            data=body,
            headers={
                "Authorization": f"Bearer {self._api_key}",
                "Content-Type": "application/json",
                "User-Agent": "voice-bridge/0.1",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                return response.read()
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", "replace")
            raise RuntimeError(
                f"Together TTS failed ({exc.code}): {detail}"
            ) from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(f"Together TTS failed: {exc.reason}") from exc


async def _mp3_to_ogg_opus(mp3: bytes) -> bytes:
    ffmpeg = await asyncio.create_subprocess_exec(
        "ffmpeg",
        "-hide_banner",
        "-loglevel", "error",
        "-i", "pipe:0",
        "-c:a", "libopus",
        "-f", "ogg",
        "pipe:1",
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    ogg, err = await ffmpeg.communicate(mp3)
    if ffmpeg.returncode != 0:
        raise RuntimeError(
            f"ffmpeg failed ({ffmpeg.returncode}): {err.decode('utf-8', 'replace')}"
        )
    return ogg
