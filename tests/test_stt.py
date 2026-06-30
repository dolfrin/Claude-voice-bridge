import asyncio
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from voice_bridge.stt import Transcriber


def _fake_model_factory(captured):
    """Return a MagicMock WhisperModel class that records construction + calls."""
    def transcribe(path, **kwargs):
        captured["path"] = path
        captured["kwargs"] = kwargs
        # faster-whisper returns (segments_iterable, info)
        segments = [
            SimpleNamespace(text=" Labas "),
            SimpleNamespace(text="pasauli"),
        ]
        info = SimpleNamespace(language="lt", language_probability=0.99)
        return iter(segments), info

    instance = MagicMock()
    instance.transcribe.side_effect = transcribe

    model_cls = MagicMock(return_value=instance)
    captured["model_cls"] = model_cls
    return model_cls


@pytest.mark.asyncio
async def test_transcribe_joins_segments_and_strips():
    captured = {}
    model_cls = _fake_model_factory(captured)
    with patch("faster_whisper.WhisperModel", model_cls):
        t = Transcriber("large-v3")
        result = await t.transcribe(b"OggS-fake-opus-bytes")
    assert result == "Labas pasauli"


@pytest.mark.asyncio
async def test_transcribe_passes_language_lt_by_default():
    captured = {}
    model_cls = _fake_model_factory(captured)
    with patch("faster_whisper.WhisperModel", model_cls):
        t = Transcriber("large-v3")
        await t.transcribe(b"OggS-fake")
    assert captured["kwargs"]["language"] == "lt"


@pytest.mark.asyncio
async def test_transcribe_honors_custom_language():
    captured = {}
    model_cls = _fake_model_factory(captured)
    with patch("faster_whisper.WhisperModel", model_cls):
        t = Transcriber("large-v3", language="en")
        await t.transcribe(b"OggS-fake")
    assert captured["kwargs"]["language"] == "en"


@pytest.mark.asyncio
async def test_transcribe_constructs_model_with_name_lazily():
    captured = {}
    model_cls = _fake_model_factory(captured)
    with patch("faster_whisper.WhisperModel", model_cls):
        t = Transcriber("medium")
        # not constructed at __init__ time
        captured["model_cls"].assert_not_called()
        await t.transcribe(b"OggS-fake")
    captured["model_cls"].assert_called_once_with("medium")


@pytest.mark.asyncio
async def test_transcribe_reuses_loaded_model():
    captured = {}
    model_cls = _fake_model_factory(captured)
    with patch("faster_whisper.WhisperModel", model_cls):
        t = Transcriber("large-v3")
        await t.transcribe(b"OggS-1")
        await t.transcribe(b"OggS-2")
    # model loaded exactly once across two transcriptions
    captured["model_cls"].assert_called_once()


@pytest.mark.asyncio
async def test_transcribe_writes_ogg_temp_file_and_cleans_up(tmp_path):
    seen_paths = []

    def transcribe(path, **kwargs):
        seen_paths.append(path)
        # file must exist with the ogg bytes while transcribing
        with open(path, "rb") as fh:
            assert fh.read() == b"OggS-payload"
        return iter([SimpleNamespace(text="ok")]), SimpleNamespace(language="lt")

    instance = MagicMock()
    instance.transcribe.side_effect = transcribe
    model_cls = MagicMock(return_value=instance)

    with patch("faster_whisper.WhisperModel", model_cls):
        t = Transcriber("large-v3")
        result = await t.transcribe(b"OggS-payload")

    assert result == "ok"
    assert seen_paths and seen_paths[0].endswith(".ogg")
    # temp file removed after transcription
    import os
    assert not os.path.exists(seen_paths[0])


@pytest.mark.asyncio
async def test_transcribe_runs_off_event_loop():
    """The blocking model call must not run on the main loop thread."""
    main_thread_id = None

    import threading
    main_thread_id = threading.get_ident()
    call_thread_ids = []

    def transcribe(path, **kwargs):
        call_thread_ids.append(threading.get_ident())
        return iter([SimpleNamespace(text="x")]), SimpleNamespace(language="lt")

    instance = MagicMock()
    instance.transcribe.side_effect = transcribe
    model_cls = MagicMock(return_value=instance)

    with patch("faster_whisper.WhisperModel", model_cls):
        t = Transcriber("large-v3")
        await t.transcribe(b"OggS-fake")

    assert call_thread_ids
    assert call_thread_ids[0] != main_thread_id


@pytest.mark.asyncio
async def test_transcribe_cleans_up_temp_file_on_error():
    seen_paths = []

    def transcribe(path, **kwargs):
        seen_paths.append(path)
        raise RuntimeError("decode failed")

    instance = MagicMock()
    instance.transcribe.side_effect = transcribe
    model_cls = MagicMock(return_value=instance)

    with patch("faster_whisper.WhisperModel", model_cls):
        t = Transcriber("large-v3")
        with pytest.raises(RuntimeError, match="decode failed"):
            await t.transcribe(b"OggS-fake")

    import os
    assert seen_paths and not os.path.exists(seen_paths[0])
