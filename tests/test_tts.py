import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from voice_bridge.config import Config
from voice_bridge.tts import TTSBackend, available_voices, get_tts
from voice_bridge.tts.openai_tts import OpenAITTS
from voice_bridge.tts.piper_tts import PiperTTS
from voice_bridge.tts.together_tts import TogetherTTS


def _cfg(**overrides) -> Config:
    base = dict(
        telegram_bot_token="t",
        telegram_allowed_user_id=1,
        anthropic_api_key="a",
        openai_api_key="sk-test",
        together_api_key="tk-test",
        together_tts_model="cartesia/sonic",
        together_tts_language="lt",
        tts_backend="openai",
        tts_voice="alloy",
        piper_voice_path="/opt/piper/lt_LT.onnx",
        whisper_model="large-v3",
        autonomy_mode="safe",
        approval_timeout=300,
        db_path=":memory:",
    )
    base.update(overrides)
    return Config(**base)


def test_available_voices_openai_lists_known_voices():
    voices = available_voices("openai")
    assert isinstance(voices, list)
    assert "alloy" in voices
    assert "alloy" in voices
    assert all(isinstance(v, str) for v in voices)


def test_available_voices_piper_returns_default_list():
    voices = available_voices("piper")
    assert voices == ["default"]


def test_available_voices_together_lists_known_voices():
    voices = available_voices("together")
    assert "friendly sidekick" in voices
    assert "af_bella" in voices


def test_available_voices_unknown_backend_returns_empty():
    assert available_voices("bogus") == []


def test_ttsbackend_is_runtime_checkable_protocol():
    assert isinstance(OpenAITTS("sk-test"), TTSBackend)
    assert isinstance(PiperTTS("/opt/piper/lt_LT.onnx"), TTSBackend)
    assert isinstance(TogetherTTS("tk-test"), TTSBackend)
    assert not isinstance(object(), TTSBackend)


def test_get_tts_openai_builds_openai_backend():
    with patch("voice_bridge.tts.openai_tts.OpenAI") as mock_openai:
        backend = get_tts(_cfg(tts_backend="openai", openai_api_key="sk-xyz"))
    assert isinstance(backend, OpenAITTS)
    mock_openai.assert_called_once_with(api_key="sk-xyz")


def test_get_tts_piper_builds_piper_backend():
    backend = get_tts(_cfg(tts_backend="piper", piper_voice_path="/v/lt.onnx"))
    assert isinstance(backend, PiperTTS)
    assert backend._voice_path == "/v/lt.onnx"


def test_get_tts_together_builds_together_backend():
    backend = get_tts(_cfg(
        tts_backend="together",
        together_api_key="tk-xyz",
        together_tts_model="cartesia/sonic",
        together_tts_language="lt",
    ))
    assert isinstance(backend, TogetherTTS)
    assert backend._api_key == "tk-xyz"
    assert backend._model == "cartesia/sonic"
    assert backend._language == "lt"


def test_get_tts_unknown_backend_raises():
    with pytest.raises(ValueError, match="unknown TTS backend"):
        get_tts(_cfg(tts_backend="bogus"))


@pytest.mark.asyncio
async def test_openai_synthesize_requests_opus_and_returns_bytes():
    fake_response = MagicMock()
    fake_response.read.return_value = b"OggS-opus-bytes"

    fake_client = MagicMock()
    fake_client.audio.speech.create.return_value = fake_response

    with patch("voice_bridge.tts.openai_tts.OpenAI", return_value=fake_client) as mock_openai:
        backend = OpenAITTS("sk-abc")
        out = await backend.synthesize("Sveiki, viskas gerai.", "alloy")

    assert out == b"OggS-opus-bytes"
    mock_openai.assert_called_once_with(api_key="sk-abc")
    fake_client.audio.speech.create.assert_called_once_with(
        model="gpt-4o-mini-tts-2025-12-15",
        voice="alloy",
        input="Sveiki, viskas gerai.",
        instructions=(
            "Speak Lithuanian naturally, like a calm human assistant in a private voice "
            "message. Avoid announcer, robotic, overly formal, or synthetic intonation. "
            "Use natural pacing, warm tone, and clear articulation."
        ),
        response_format="opus",
    )


@pytest.mark.asyncio
async def test_openai_synthesize_runs_off_the_event_loop():
    fake_response = MagicMock()
    fake_response.read.return_value = b"x"
    fake_client = MagicMock()
    fake_client.audio.speech.create.return_value = fake_response

    loop = asyncio.get_running_loop()
    calls: dict = {}
    _real_run_in_executor = loop.run_in_executor

    def spy_executor(executor, func, *args):
        calls["used_executor"] = True
        return _real_run_in_executor(executor, func, *args)

    with patch("voice_bridge.tts.openai_tts.OpenAI", return_value=fake_client):
        backend = OpenAITTS("sk-abc")
        with patch.object(loop, "run_in_executor", side_effect=spy_executor):
            await backend.synthesize("labas", "echo")

    assert calls.get("used_executor") is True


@pytest.mark.asyncio
async def test_together_synthesize_requests_mp3_and_converts_to_opus(monkeypatch):
    captured = {}

    class FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def read(self):
            return b"MP3"

    def fake_urlopen(request, timeout):
        captured["url"] = request.full_url
        captured["headers"] = dict(request.header_items())
        captured["body"] = request.data
        captured["timeout"] = timeout
        return FakeResponse()

    monkeypatch.setattr("voice_bridge.tts.together_tts.urllib.request.urlopen", fake_urlopen)
    monkeypatch.setattr(
        "voice_bridge.tts.together_tts._mp3_to_ogg_opus",
        AsyncMock(return_value=b"OggS-together-opus"),
    )

    backend = TogetherTTS("tk-abc", model="cartesia/sonic", language="lt")
    out = await backend.synthesize("Labas, patikrinam balsą.", "friendly sidekick")

    assert out == b"OggS-together-opus"
    assert captured["url"] == "https://api.together.ai/v1/audio/speech"
    assert captured["headers"]["Authorization"] == "Bearer tk-abc"
    assert captured["timeout"] == 60
    assert b'"model": "cartesia/sonic"' in captured["body"]
    assert b'"voice": "friendly sidekick"' in captured["body"]
    assert b'"language": "lt"' in captured["body"]


@pytest.mark.asyncio
async def test_together_synthesize_requires_api_key():
    backend = TogetherTTS("")
    with pytest.raises(RuntimeError, match="TOGETHER_API_KEY"):
        await backend.synthesize("x", "friendly sidekick")


def _fake_proc(stdout: bytes, stderr: bytes = b"", returncode: int = 0):
    proc = MagicMock()
    proc.communicate = AsyncMock(return_value=(stdout, stderr))
    proc.returncode = returncode
    return proc


@pytest.mark.asyncio
async def test_piper_synthesize_pipes_pcm_through_ffmpeg_to_opus():
    piper_proc = _fake_proc(stdout=b"RAWPCM")
    ffmpeg_proc = _fake_proc(stdout=b"OggS-piper-opus")

    created = []

    async def fake_exec(*args, **kwargs):
        created.append(args)
        return piper_proc if args[0] == "piper" else ffmpeg_proc

    with patch(
        "voice_bridge.tts.piper_tts.asyncio.create_subprocess_exec",
        side_effect=fake_exec,
    ):
        backend = PiperTTS("/opt/piper/lt_LT.onnx")
        out = await backend.synthesize("Sveiki", "default")

    assert out == b"OggS-piper-opus"
    # piper invoked with the configured model path
    assert created[0][0] == "piper"
    assert "/opt/piper/lt_LT.onnx" in created[0]
    assert "--output-raw" in created[0]
    # ffmpeg invoked to encode opus in an ogg container
    assert created[1][0] == "ffmpeg"
    assert "libopus" in created[1]
    assert "ogg" in created[1]
    # text fed to piper stdin; piper pcm fed to ffmpeg stdin
    piper_proc.communicate.assert_awaited_once_with(b"Sveiki")
    ffmpeg_proc.communicate.assert_awaited_once_with(b"RAWPCM")


@pytest.mark.asyncio
async def test_piper_synthesize_raises_when_piper_fails():
    piper_proc = _fake_proc(stdout=b"", stderr=b"model missing", returncode=1)

    async def fake_exec(*args, **kwargs):
        return piper_proc

    with patch(
        "voice_bridge.tts.piper_tts.asyncio.create_subprocess_exec",
        side_effect=fake_exec,
    ):
        backend = PiperTTS("/bad.onnx")
        with pytest.raises(RuntimeError, match="piper failed"):
            await backend.synthesize("x", "default")


@pytest.mark.asyncio
async def test_piper_synthesize_raises_when_ffmpeg_fails():
    piper_proc = _fake_proc(stdout=b"RAWPCM")
    ffmpeg_proc = _fake_proc(stdout=b"", stderr=b"enc error", returncode=1)

    async def fake_exec(*args, **kwargs):
        return piper_proc if args[0] == "piper" else ffmpeg_proc

    with patch(
        "voice_bridge.tts.piper_tts.asyncio.create_subprocess_exec",
        side_effect=fake_exec,
    ):
        backend = PiperTTS("/opt/piper/lt_LT.onnx")
        with pytest.raises(RuntimeError, match="ffmpeg failed"):
            await backend.synthesize("x", "default")
