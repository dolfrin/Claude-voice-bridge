import textwrap

import pytest

from voice_bridge.config import (
    Config,
    ProjectConfig,
    effective_autonomy,
    effective_voice,
    load_config,
    load_projects,
)
from voice_bridge.types import Outbound


def _full_env() -> dict[str, str]:
    return {
        "TELEGRAM_BOT_TOKEN": "123:abc",
        "TELEGRAM_ALLOWED_USER_ID": "42",
        "ANTHROPIC_API_KEY": "sk-ant-test",
        "OPENAI_API_KEY": "sk-openai-test",
        "TTS_BACKEND": "openai",
        "TTS_VOICE": "alloy",
        "PIPER_VOICE_PATH": "/opt/piper/lt.onnx",
        "WHISPER_MODEL": "large-v3",
        "AUTONOMY_MODE": "safe",
        "APPROVAL_TIMEOUT": "300",
        "DB_PATH": "/var/lib/voice-bridge/state.db",
    }


def test_load_config_parses_all_fields_with_correct_types():
    cfg = load_config(_full_env())
    assert isinstance(cfg, Config)
    assert cfg.telegram_bot_token == "123:abc"
    assert cfg.telegram_allowed_user_id == 42
    assert isinstance(cfg.telegram_allowed_user_id, int)
    assert cfg.anthropic_api_key == "sk-ant-test"
    assert cfg.openai_api_key == "sk-openai-test"
    assert cfg.tts_backend == "openai"
    assert cfg.tts_voice == "alloy"
    assert cfg.piper_voice_path == "/opt/piper/lt.onnx"
    assert cfg.whisper_model == "large-v3"
    assert cfg.autonomy_mode == "safe"
    assert cfg.approval_timeout == 300
    assert isinstance(cfg.approval_timeout, int)
    assert cfg.db_path == "/var/lib/voice-bridge/state.db"


def test_load_config_applies_defaults_for_optional_keys():
    env = {
        "TELEGRAM_BOT_TOKEN": "123:abc",
        "TELEGRAM_ALLOWED_USER_ID": "42",
        "ANTHROPIC_API_KEY": "sk-ant-test",
        "OPENAI_API_KEY": "sk-openai-test",
    }
    cfg = load_config(env)
    assert cfg.tts_backend == "openai"
    assert cfg.tts_voice == "alloy"
    assert cfg.piper_voice_path == ""
    assert cfg.whisper_model == "large-v3"
    assert cfg.autonomy_mode == "safe"
    assert cfg.approval_timeout == 300
    assert cfg.db_path == "voice-bridge.db"


def test_load_config_missing_required_key_raises_clear_error():
    env = _full_env()
    del env["TELEGRAM_BOT_TOKEN"]
    with pytest.raises(ValueError) as exc:
        load_config(env)
    assert "TELEGRAM_BOT_TOKEN" in str(exc.value)


def test_load_config_non_int_user_id_raises_clear_error():
    env = _full_env()
    env["TELEGRAM_ALLOWED_USER_ID"] = "not-a-number"
    with pytest.raises(ValueError) as exc:
        load_config(env)
    assert "TELEGRAM_ALLOWED_USER_ID" in str(exc.value)


def test_load_config_non_int_timeout_raises_clear_error():
    env = _full_env()
    env["APPROVAL_TIMEOUT"] = "soon"
    with pytest.raises(ValueError) as exc:
        load_config(env)
    assert "APPROVAL_TIMEOUT" in str(exc.value)


def test_load_config_invalid_backend_raises_clear_error():
    env = _full_env()
    env["TTS_BACKEND"] = "espeak"
    with pytest.raises(ValueError) as exc:
        load_config(env)
    assert "TTS_BACKEND" in str(exc.value)


def test_load_config_invalid_autonomy_raises_clear_error():
    env = _full_env()
    env["AUTONOMY_MODE"] = "yolo"
    with pytest.raises(ValueError) as exc:
        load_config(env)
    assert "AUTONOMY_MODE" in str(exc.value)


def test_load_projects_parses_fields_and_defaults(tmp_path):
    yaml_text = textwrap.dedent(
        """
        projects:
          - name: qwing
            cwd: /home/home/Projects/WhisperX
            enabled: true
            autonomy: safe
            voice: alloy
            model: claude-opus-4-8
            system_prompt_extra: "be terse"
          - name: bridge
            cwd: /home/home/Projects/claude-voice-bridge
        """
    )
    path = tmp_path / "projects.yaml"
    path.write_text(yaml_text)

    projects = load_projects(str(path))
    assert [p.name for p in projects] == ["qwing", "bridge"]

    qwing = projects[0]
    assert isinstance(qwing, ProjectConfig)
    assert qwing.cwd == "/home/home/Projects/WhisperX"
    assert qwing.enabled is True
    assert qwing.autonomy == "safe"
    assert qwing.voice == "alloy"
    assert qwing.model == "claude-opus-4-8"
    assert qwing.system_prompt_extra == "be terse"

    bridge = projects[1]
    assert bridge.enabled is True
    assert bridge.autonomy is None
    assert bridge.voice is None
    assert bridge.model is None
    assert bridge.system_prompt_extra == ""


def test_load_projects_missing_name_raises_clear_error(tmp_path):
    path = tmp_path / "projects.yaml"
    path.write_text("projects:\n  - cwd: /tmp/x\n")
    with pytest.raises(ValueError) as exc:
        load_projects(str(path))
    assert "name" in str(exc.value)


def test_load_projects_missing_cwd_raises_clear_error(tmp_path):
    path = tmp_path / "projects.yaml"
    path.write_text("projects:\n  - name: x\n")
    with pytest.raises(ValueError) as exc:
        load_projects(str(path))
    assert "cwd" in str(exc.value)


def test_load_projects_missing_file_raises_clear_error(tmp_path):
    missing = tmp_path / "nope.yaml"
    with pytest.raises(FileNotFoundError) as exc:
        load_projects(str(missing))
    assert "nope.yaml" in str(exc.value)


def test_load_projects_empty_list_raises_clear_error(tmp_path):
    path = tmp_path / "projects.yaml"
    path.write_text("projects: []\n")
    with pytest.raises(ValueError) as exc:
        load_projects(str(path))
    assert "at least one project" in str(exc.value)


def test_effective_autonomy_prefers_project_override():
    cfg = load_config(_full_env())  # autonomy_mode == "safe"
    proj = ProjectConfig(name="x", cwd="/tmp/x", autonomy="full")
    assert effective_autonomy(proj, cfg) == "full"


def test_effective_autonomy_falls_back_to_global():
    cfg = load_config(_full_env())  # autonomy_mode == "safe"
    proj = ProjectConfig(name="x", cwd="/tmp/x", autonomy=None)
    assert effective_autonomy(proj, cfg) == "safe"


def test_effective_voice_prefers_project_override():
    cfg = load_config(_full_env())  # tts_voice == "alloy"
    proj = ProjectConfig(name="x", cwd="/tmp/x", voice="echo")
    assert effective_voice(proj, cfg) == "echo"


def test_effective_voice_falls_back_to_global():
    cfg = load_config(_full_env())  # tts_voice == "alloy"
    proj = ProjectConfig(name="x", cwd="/tmp/x", voice=None)
    assert effective_voice(proj, cfg) == "alloy"


def test_outbound_fields():
    o = Outbound(project="qwing", text="full", spoken="say")
    assert (o.project, o.text, o.spoken) == ("qwing", "full", "say")
