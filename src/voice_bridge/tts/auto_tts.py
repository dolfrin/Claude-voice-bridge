"""Language-aware TTS router.

English-looking text goes to local Piper to save OpenAI credits. Non-English or
uncertain text goes to OpenAI for better pronunciation.
"""
from __future__ import annotations

import re

from voice_bridge.tts.openai_tts import OpenAITTS
from voice_bridge.tts.piper_tts import PiperTTS

_EN_WORDS = {
    "a",
    "about",
    "again",
    "all",
    "and",
    "are",
    "as",
    "back",
    "build",
    "can",
    "check",
    "complete",
    "completed",
    "done",
    "error",
    "failed",
    "file",
    "fix",
    "fixed",
    "for",
    "from",
    "go",
    "has",
    "have",
    "hello",
    "in",
    "is",
    "it",
    "next",
    "no",
    "now",
    "ok",
    "okay",
    "passed",
    "please",
    "project",
    "ready",
    "run",
    "status",
    "stop",
    "test",
    "tests",
    "that",
    "the",
    "this",
    "to",
    "updated",
    "was",
    "what",
    "with",
    "work",
    "working",
    "yes",
    "you",
}


class AutoTTS:
    """Route each utterance to Piper or OpenAI based on likely language."""

    def __init__(self, openai_api_key: str, piper_voice_path: str) -> None:
        self._openai = OpenAITTS(openai_api_key)
        self._piper = PiperTTS(piper_voice_path)

    async def synthesize(self, text: str, voice: str) -> bytes:
        if _looks_english(text):
            try:
                return await self._piper.synthesize(text, voice)
            except Exception:
                return await self._openai.synthesize(text, voice)
        return await self._openai.synthesize(text, voice)


def _looks_english(text: str) -> bool:
    if any(ch.isalpha() and not ch.isascii() for ch in text):
        return False

    words = re.findall(r"[A-Za-z']+", text.lower())
    if not words:
        return False

    hits = sum(1 for word in words if word in _EN_WORDS)
    if len(words) == 1:
        return hits >= 1
    if len(words) <= 4:
        return hits / len(words) >= 0.5
    return hits / len(words) >= 0.25
