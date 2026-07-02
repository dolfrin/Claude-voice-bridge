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
    # NOISE-class message (per-turn "Working." status, the 60s "still
    # working" heartbeat, verbose tool-activity flushes): make_outbound skips
    # recording it into the /recap buffer so recap counts only substantive
    # updates. Never set on assistant text, notify_user, crash notices, or
    # approval questions.
    transient: bool = False
