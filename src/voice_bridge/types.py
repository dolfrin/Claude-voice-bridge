from dataclasses import dataclass


@dataclass
class Outbound:
    project: str
    text: str    # full content; may contain code/diffs
    spoken: str  # code-free line for TTS
    file_path: str | None = None
    # ALERT-class message (approval question / crash notice): make_outbound
    # synthesizes it with cfg.tts_alert_voice (when set) so it sounds distinct
    # from routine status. Normal turns leave this False.
    alert: bool = False
