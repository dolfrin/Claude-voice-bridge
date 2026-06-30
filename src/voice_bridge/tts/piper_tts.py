"""Local Piper TTS backend; emits OGG/Opus via ffmpeg."""
from __future__ import annotations

import asyncio


class PiperTTS:
    """Piper text-to-speech; raw PCM piped through ffmpeg to OGG/Opus."""

    def __init__(self, voice_path: str) -> None:
        self._voice_path = voice_path

    async def synthesize(self, text: str, voice: str) -> bytes:
        piper = await asyncio.create_subprocess_exec(
            "piper",
            "--model",
            self._voice_path,
            "--output-raw",
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        pcm, piper_err = await piper.communicate(text.encode("utf-8"))
        if piper.returncode != 0:
            raise RuntimeError(
                f"piper failed ({piper.returncode}): {piper_err.decode('utf-8', 'replace')}"
            )

        ffmpeg = await asyncio.create_subprocess_exec(
            "ffmpeg",
            "-f",
            "s16le",
            "-ar",
            "22050",
            "-ac",
            "1",
            "-i",
            "pipe:0",
            "-c:a",
            "libopus",
            "-f",
            "ogg",
            "pipe:1",
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        ogg, ffmpeg_err = await ffmpeg.communicate(pcm)
        if ffmpeg.returncode != 0:
            raise RuntimeError(
                f"ffmpeg failed ({ffmpeg.returncode}): {ffmpeg_err.decode('utf-8', 'replace')}"
            )
        return ogg
