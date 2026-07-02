"""Load and validate environment Config and projects.yaml into typed dataclasses."""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass

import yaml

# Canonical, ORDERED source of truth for these two small enums. Order
# matters here: telegram_io's /panel cycles the engine button and mode
# picker through these tuples in this exact preferred order. Validation
# sets are derived below so accepted values stay in sync automatically.
AUTONOMY_MODES = ("safe", "full", "ask")
TTS_BACKENDS = ("auto", "openai", "piper", "together")
_VALID_TTS_BACKENDS = set(TTS_BACKENDS)
_VALID_AUTONOMY_MODES = set(AUTONOMY_MODES)


@dataclass
class Config:
    telegram_bot_token: str
    telegram_allowed_user_id: int
    anthropic_api_key: str
    openai_api_key: str
    together_api_key: str
    together_tts_model: str
    together_tts_language: str
    tts_backend: str
    tts_voice: str
    piper_voice_path: str
    whisper_model: str
    autonomy_mode: str
    approval_timeout: int
    db_path: str
    # Optional distinct voice for ALERT-class TTS (approval questions + crash
    # notices). Empty string -> fall back to the project/default voice.
    tts_alert_voice: str = ""
    auto_discover_projects: bool = False
    auto_discover_limit: int = 12
    open_vscode_on_enable: bool = False
    close_vscode_on_disable: bool = False


@dataclass
class ProjectConfig:
    name: str
    cwd: str
    display_name: str | None = None
    enabled: bool = True
    autonomy: str | None = None
    voice: str | None = None
    model: str | None = None
    system_prompt_extra: str = ""
    # Opt-in live tool-activity streaming (default OFF). When True the session
    # emits compact, text-only, coalesced tool-activity Outbounds during a turn
    # so a glance at the phone shows progress. Toggled live via /verbose.
    verbose: bool = False


def _require(env: Mapping[str, str], key: str) -> str:
    value = env.get(key)
    if value is None or value == "":
        raise ValueError(f"Missing required config key: {key}")
    return value


def _require_int(env: Mapping[str, str], key: str) -> int:
    raw = _require(env, key)
    try:
        return int(raw)
    except (TypeError, ValueError):
        raise ValueError(f"Config key {key} must be an integer, got: {raw!r}")


def _optional_int(env: Mapping[str, str], key: str, default: int) -> int:
    raw = env.get(key)
    if raw is None or raw == "":
        return default
    try:
        return int(raw)
    except (TypeError, ValueError):
        raise ValueError(f"Config key {key} must be an integer, got: {raw!r}")


def _optional_bool(env: Mapping[str, str], key: str, default: bool) -> bool:
    raw = env.get(key)
    if raw is None or raw == "":
        return default
    normalized = raw.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"Config key {key} must be a boolean, got: {raw!r}")


def _together_language(env: Mapping[str, str]) -> str:
    raw = env.get("TOGETHER_TTS_LANGUAGE")
    if raw is None:
        return ""
    if raw.strip().lower() == "auto":
        return ""
    return raw


def load_config(env: Mapping[str, str] | None = None) -> Config:
    """Build a validated Config from a mapping (defaults to os.environ)."""
    env = os.environ if env is None else env

    tts_backend = env.get("TTS_BACKEND") or "openai"
    if tts_backend not in _VALID_TTS_BACKENDS:
        raise ValueError(
            f"Config key TTS_BACKEND must be one of "
            f"{sorted(_VALID_TTS_BACKENDS)}, got: {tts_backend!r}"
        )

    autonomy_mode = env.get("AUTONOMY_MODE") or "safe"
    if autonomy_mode not in _VALID_AUTONOMY_MODES:
        raise ValueError(
            f"Config key AUTONOMY_MODE must be one of "
            f"{sorted(_VALID_AUTONOMY_MODES)}, got: {autonomy_mode!r}"
        )

    return Config(
        telegram_bot_token=_require(env, "TELEGRAM_BOT_TOKEN"),
        telegram_allowed_user_id=_require_int(env, "TELEGRAM_ALLOWED_USER_ID"),
        anthropic_api_key=env.get("ANTHROPIC_API_KEY") or "",
        openai_api_key=(
            _require(env, "OPENAI_API_KEY") if tts_backend in {"auto", "openai"}
            else env.get("OPENAI_API_KEY") or ""
        ),
        together_api_key=env.get("TOGETHER_API_KEY") or "",
        together_tts_model=env.get("TOGETHER_TTS_MODEL") or "cartesia/sonic",
        together_tts_language=_together_language(env),
        tts_backend=tts_backend,
        tts_voice=env.get("TTS_VOICE") or "alloy",
        tts_alert_voice=env.get("TTS_ALERT_VOICE") or "",
        piper_voice_path=env.get("PIPER_VOICE_PATH") or "",
        whisper_model=env.get("WHISPER_MODEL") or "large-v3",
        autonomy_mode=autonomy_mode,
        approval_timeout=_optional_int(env, "APPROVAL_TIMEOUT", 300),
        db_path=env.get("DB_PATH") or "voice-bridge.db",
        auto_discover_projects=_optional_bool(
            env, "AUTO_DISCOVER_PROJECTS", False
        ),
        auto_discover_limit=_optional_int(env, "AUTO_DISCOVER_LIMIT", 12),
        open_vscode_on_enable=_optional_bool(env, "OPEN_VSCODE_ON_ENABLE", False),
        close_vscode_on_disable=_optional_bool(env, "CLOSE_VSCODE_ON_DISABLE", False),
    )


def load_projects(path: str = "projects.yaml") -> list[ProjectConfig]:
    """Parse projects.yaml into a list of validated ProjectConfig."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"projects file not found: {path}")

    with open(path, "r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}

    raw_projects = data.get("projects")
    if not raw_projects:
        raise ValueError("projects file must contain at least one project")
    projects: list[ProjectConfig] = []
    for idx, raw in enumerate(raw_projects):
        name = raw.get("name")
        if not name:
            raise ValueError(f"project at index {idx} is missing required field: name")
        cwd = raw.get("cwd")
        if not cwd:
            raise ValueError(f"project {name!r} is missing required field: cwd")

        enabled = raw.get("enabled")
        projects.append(
            ProjectConfig(
                name=name,
                cwd=cwd,
                display_name=raw.get("display_name"),
                enabled=True if enabled is None else bool(enabled),
                autonomy=raw.get("autonomy"),
                voice=raw.get("voice"),
                model=raw.get("model"),
                system_prompt_extra=raw.get("system_prompt_extra") or "",
                verbose=bool(raw.get("verbose", False)),
            )
        )
    return projects


def effective_autonomy(project: ProjectConfig, cfg: Config) -> str:
    """Project-level autonomy override, falling back to the global mode."""
    return project.autonomy or cfg.autonomy_mode


def effective_voice(project: ProjectConfig, cfg: Config) -> str:
    """Project-level voice override, falling back to the global voice."""
    return project.voice or cfg.tts_voice
