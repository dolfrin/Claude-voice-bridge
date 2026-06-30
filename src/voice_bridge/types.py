# src/voice_bridge/types.py
from dataclasses import dataclass


@dataclass
class Outbound:
    project: str
    text: str    # full content; may contain code/diffs
    spoken: str  # code-free line for TTS
