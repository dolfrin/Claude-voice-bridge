"""Local Lithuanian TTS: the reginute1 Piper voice with its own phonemizer.

Released Piper cannot load this voice from the catalogue config: it declares
``phoneme_type: lithuanian``, a phonemizer that only exists in a piper1-gpl
contribution, not in any published piper-tts (1.8.0 knows espeak, text, pinyin,
hebrew, japanese, thai). The voice ships a second config with
``phoneme_type: text`` for exactly this case -- we phonemize ourselves and hand
Piper IPA.

Everything below the phonemizer is the voice author's own reference synthesis
(``synth_reginute.ReginuteSynth``): fragment splitting, pause insertion, rate
levelling and ``normalize_audio=False`` (the default True clips this voice).
Reimplementing that would sound worse for no reason, so it is loaded from the
voice directory rather than copied.

Model dir layout (PIPER_LT_DIR), as downloaded from the voice repo:

    lt_LT-reginute1-medium.onnx        (or a symlink to it)
    lt_LT-reginute1-medium.onnx.json   phoneme_type: text
    lt_kirciai.tsv  lt_raides.tsv  lt_kreipiniai.tsv
    phonemize_lithuanian.py  skaiciu_pletiklis.py  synth_reginute.py
"""
from __future__ import annotations

import asyncio
import sys
from pathlib import Path

_MODEL = "lt_LT-reginute1-medium.onnx"
_CONFIG = "lt_LT-reginute1-medium.onnx.json"


class LithuanianTTS:
    """reginute1 via its own phonemizer; raw PCM piped through ffmpeg to OGG.

    The voice (~63 MB of weights plus a 3 MB stress dictionary) is loaded on
    the first synthesize and kept, so a bridge that never speaks Lithuanian
    pays nothing. Synthesis is CPU-bound and blocking, so it runs in a worker
    thread -- the event loop still has a bot to answer.
    """

    def __init__(self, model_dir: str) -> None:
        self._dir = Path(model_dir).expanduser()
        self._synth = None
        self._pcm16 = None
        self._sample_rate = 22050
        self._lock = asyncio.Lock()

    def _load(self) -> None:
        """Import the voice's modules and build the synth. Blocking; called once."""
        from piper import PiperVoice

        d = self._dir
        # The voice's three modules import each other by plain name, so the
        # directory has to be importable, not just readable.
        if str(d) not in sys.path:
            sys.path.insert(0, str(d))
        import phonemize_lithuanian as pl
        import synth_reginute as sr

        phonemizer = pl.LithuanianPhonemizer(
            dictionary_path=d / "lt_kirciai.tsv",
            letters_path=d / "lt_raides.tsv",
            vocatives_path=d / "lt_kreipiniai.tsv",
        )
        voice = PiperVoice.load(str(d / _MODEL), config_path=str(d / _CONFIG))
        self._synth = sr.ReginuteSynth(voice, phonemizer)
        self._pcm16 = sr.i_int16
        self._sample_rate = self._synth.sr

    def _render(self, text: str) -> bytes:
        """Text -> s16le PCM. Blocking."""
        if self._synth is None:
            self._load()
        return self._pcm16(self._synth.synthesize(text))

    async def synthesize(self, text: str, voice: str) -> bytes:
        """OGG/Opus bytes. ``voice`` is ignored: this model has one speaker."""
        # One loader at a time: two concurrent first-sends would otherwise both
        # pay the 63 MB load, and both mutate sys.path.
        async with self._lock:
            pcm = await asyncio.to_thread(self._render, text)

        ffmpeg = await asyncio.create_subprocess_exec(
            "ffmpeg",
            "-f", "s16le",
            "-ar", str(self._sample_rate),
            "-ac", "1",
            "-i", "pipe:0",
            "-c:a", "libopus",
            "-f", "ogg",
            "pipe:1",
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        ogg, err = await ffmpeg.communicate(pcm)
        if ffmpeg.returncode != 0:
            raise RuntimeError(
                f"ffmpeg failed ({ffmpeg.returncode}): {err.decode('utf-8', 'replace')}"
            )
        return ogg
