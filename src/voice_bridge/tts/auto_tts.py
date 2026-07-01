"""Language-aware TTS router.

English-looking text goes to local Piper to save OpenAI credits. Lithuanian or
mixed Lithuanian text goes to OpenAI for better pronunciation.
"""
from __future__ import annotations

import re

from voice_bridge.tts.openai_tts import OpenAITTS
from voice_bridge.tts.piper_tts import PiperTTS

_LT_CHARS = set("ąčęėįšųūžĄČĘĖĮŠŲŪŽ")
_LT_WORDS = {
    "aš",
    "ar",
    "bet",
    "čia",
    "dar",
    "dabar",
    "dėl",
    "gerai",
    "gali",
    "jei",
    "kad",
    "kaip",
    "ką",
    "kur",
    "labai",
    "man",
    "ne",
    "nes",
    "nu",
    "reikia",
    "su",
    "tai",
    "taip",
    "tavo",
    "tu",
    "už",
    "veikia",
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
    if any(ch in _LT_CHARS for ch in text):
        return False

    words = re.findall(r"[A-Za-zĄČĘĖĮŠŲŪŽąčęėįšųūž']+", text.lower())
    if not words:
        return False

    lt_hits = sum(1 for word in words if word in _LT_WORDS)
    ascii_words = sum(1 for word in words if word.isascii())
    return lt_hits == 0 and ascii_words / len(words) >= 0.85

