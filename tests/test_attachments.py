import io
import zipfile
from types import SimpleNamespace

import pytest

import voice_bridge.attachments as attachments_mod
from voice_bridge.attachments import format_attachment_prompt, save_attachments


@pytest.mark.asyncio
async def test_save_attachments_writes_file_and_formats_prompt(tmp_path):
    saved = await save_attachments(str(tmp_path), [{
        "kind": "document",
        "file_name": "../bad name.txt",
        "data": b"hello",
    }])

    assert len(saved) == 1
    assert saved[0].kind == "document"
    assert saved[0].path.startswith(".claude/voice-bridge-inbox/")
    assert ".." not in saved[0].path
    assert (tmp_path / saved[0].path).read_bytes() == b"hello"

    prompt = format_attachment_prompt("patikrink", saved)
    assert "patikrink" in prompt
    assert saved[0].path in prompt


@pytest.mark.asyncio
async def test_save_attachments_extracts_zip_safely(tmp_path):
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as archive:
        archive.writestr("inside/readme.txt", "ok")
        archive.writestr("../evil.txt", "bad")

    saved = await save_attachments(str(tmp_path), [{
        "kind": "document",
        "file_name": "archive.zip",
        "data": buf.getvalue(),
    }])

    assert saved[0].extracted_to is not None
    extracted = tmp_path / saved[0].extracted_to
    assert (extracted / "inside" / "readme.txt").read_text() == "ok"
    assert not (tmp_path / "evil.txt").exists()


@pytest.mark.asyncio
async def test_save_attachments_extracts_first_video_frame(tmp_path, monkeypatch):
    monkeypatch.setattr(attachments_mod.shutil, "which", lambda name: "/usr/bin/ffmpeg")

    def fake_run(args, **kwargs):
        frame = tmp_path / ".claude" / "voice-bridge-inbox"
        matches = list(frame.glob("*.mp4.frames"))
        assert matches
        (matches[0] / "frame-0001.jpg").write_bytes(b"JPG")
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(attachments_mod.subprocess, "run", fake_run)

    saved = await save_attachments(str(tmp_path), [{
        "kind": "video",
        "file_name": "demo.mp4",
        "data": b"MP4",
    }])

    assert saved[0].extracted_to is not None
    frames = tmp_path / saved[0].extracted_to
    assert (frames / "frame-0001.jpg").read_bytes() == b"JPG"


def test_format_attachment_prompt_adds_visual_guidance():
    prompt = format_attachment_prompt(
        "",
        [
            attachments_mod.SavedAttachment("photo", ".claude/voice-bridge-inbox/shot.jpg"),
            attachments_mod.SavedAttachment(
                "video",
                ".claude/voice-bridge-inbox/demo.mp4",
                ".claude/voice-bridge-inbox/demo.mp4.frames",
            ),
        ],
    )

    assert "analizuok matomą UI" in prompt
    assert "video kadrai" in prompt
