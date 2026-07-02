import io
import subprocess
import tarfile
import zipfile
from pathlib import Path
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

    assert "inspect the visible UI" in prompt
    assert "video frames" in prompt


# ---------------------------------------------------------------------------
# Fix 1: ffmpeg frame extraction must not hang forever
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_save_attachments_ffmpeg_timeout_skips_frame_gracefully(tmp_path, monkeypatch):
    monkeypatch.setattr(attachments_mod.shutil, "which", lambda name: "/usr/bin/ffmpeg")

    def fake_run(args, **kwargs):
        assert "timeout" in kwargs
        assert kwargs["timeout"] == attachments_mod._FFMPEG_TIMEOUT_SECONDS
        raise subprocess.TimeoutExpired(cmd=args, timeout=kwargs["timeout"])

    monkeypatch.setattr(attachments_mod.subprocess, "run", fake_run)

    saved = await save_attachments(str(tmp_path), [{
        "kind": "video",
        "file_name": "demo.mp4",
        "data": b"MP4",
    }])

    assert len(saved) == 1
    assert saved[0].extracted_to is None


# ---------------------------------------------------------------------------
# Fix 2: archive extraction — decompression bomb / zip-slip protection
# ---------------------------------------------------------------------------


def _zip_bytes(entries: dict[str, bytes]) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as archive:
        for name, data in entries.items():
            archive.writestr(name, data)
    return buf.getvalue()


def _tar_bytes(entries: dict[str, bytes], symlinks: dict[str, str] | None = None) -> bytes:
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w") as archive:
        for name, data in entries.items():
            info = tarfile.TarInfo(name=name)
            info.size = len(data)
            archive.addfile(info, io.BytesIO(data))
        for name, target in (symlinks or {}).items():
            info = tarfile.TarInfo(name=name)
            info.type = tarfile.SYMTYPE
            info.linkname = target
            archive.addfile(info)
    return buf.getvalue()


@pytest.mark.asyncio
async def test_save_attachments_extracts_tar_safely(tmp_path):
    data = _tar_bytes({"inside/readme.txt": b"ok"})

    saved = await save_attachments(str(tmp_path), [{
        "kind": "document",
        "file_name": "archive.tar",
        "data": data,
    }])

    assert saved[0].extracted_to is not None
    extracted = tmp_path / saved[0].extracted_to
    assert (extracted / "inside" / "readme.txt").read_text() == "ok"


@pytest.mark.asyncio
async def test_extract_archive_zip_slip_absolute_path_not_written_outside(tmp_path):
    data = _zip_bytes({"/etc/evil.txt": b"bad", "safe.txt": b"ok"})

    saved = await save_attachments(str(tmp_path), [{
        "kind": "document",
        "file_name": "archive.zip",
        "data": data,
    }])

    assert not (tmp_path / "etc" / "evil.txt").exists()
    assert not Path("/etc/evil.txt").exists()
    extracted = tmp_path / saved[0].extracted_to
    assert (extracted / "safe.txt").read_text() == "ok"


@pytest.mark.asyncio
async def test_extract_archive_tar_slip_parent_escape_not_written_outside(tmp_path):
    data = _tar_bytes({"../evil.txt": b"bad", "safe.txt": b"ok"})

    saved = await save_attachments(str(tmp_path), [{
        "kind": "document",
        "file_name": "archive.tar",
        "data": data,
    }])

    assert not (tmp_path / "evil.txt").exists()
    extracted = tmp_path / saved[0].extracted_to
    assert (extracted / "safe.txt").read_text() == "ok"


@pytest.mark.asyncio
async def test_extract_archive_tar_symlink_member_not_followed(tmp_path):
    data = _tar_bytes({"safe.txt": b"ok"}, symlinks={"evil-link": "/etc/passwd"})

    saved = await save_attachments(str(tmp_path), [{
        "kind": "document",
        "file_name": "archive.tar",
        "data": data,
    }])

    extracted = tmp_path / saved[0].extracted_to
    assert not (extracted / "evil-link").exists()
    assert (extracted / "safe.txt").read_text() == "ok"


@pytest.mark.asyncio
async def test_extract_archive_aborts_when_total_size_exceeds_cap(tmp_path, monkeypatch):
    monkeypatch.setattr(attachments_mod, "_ARCHIVE_MAX_TOTAL_BYTES", 1_000_000)
    # Highly compressible payload: small on disk, large once decompressed —
    # a classic decompression-bomb shape.
    bomb = b"\x00" * 5_000_000
    data = _zip_bytes({"bomb.bin": bomb, "small.txt": b"tiny"})
    assert len(data) < 100_000  # the zip itself stays small

    saved = await save_attachments(str(tmp_path), [{
        "kind": "document",
        "file_name": "bomb.zip",
        "data": data,
    }])

    extracted = tmp_path / saved[0].extracted_to
    total_written = sum(f.stat().st_size for f in extracted.rglob("*") if f.is_file())
    assert total_written <= 1_000_000
    assert saved[0].note is not None
    assert "total" in saved[0].note.lower() or "cap" in saved[0].note.lower()


@pytest.mark.asyncio
async def test_extract_archive_stops_after_member_count_cap(tmp_path, monkeypatch):
    monkeypatch.setattr(attachments_mod, "_ARCHIVE_MAX_MEMBERS", 5)
    entries = {f"file-{i}.txt": b"x" for i in range(10)}
    data = _zip_bytes(entries)

    saved = await save_attachments(str(tmp_path), [{
        "kind": "document",
        "file_name": "many.zip",
        "data": data,
    }])

    extracted = tmp_path / saved[0].extracted_to
    written = [f for f in extracted.rglob("*") if f.is_file()]
    assert len(written) <= 5
    assert saved[0].note is not None


@pytest.mark.asyncio
async def test_extract_archive_skips_member_over_per_member_cap(tmp_path, monkeypatch):
    monkeypatch.setattr(attachments_mod, "_ARCHIVE_MAX_MEMBER_BYTES", 100)
    data = _zip_bytes({"huge.bin": b"x" * 1000, "small.txt": b"tiny"})

    saved = await save_attachments(str(tmp_path), [{
        "kind": "document",
        "file_name": "mixed.zip",
        "data": data,
    }])

    extracted = tmp_path / saved[0].extracted_to
    assert not (extracted / "huge.bin").exists()
    assert (extracted / "small.txt").read_text() == "tiny"


@pytest.mark.asyncio
async def test_save_attachments_extracts_small_zip_regression(tmp_path):
    data = _zip_bytes({"inside/readme.txt": b"ok", "top.txt": b"hi"})

    saved = await save_attachments(str(tmp_path), [{
        "kind": "document",
        "file_name": "archive.zip",
        "data": data,
    }])

    extracted = tmp_path / saved[0].extracted_to
    assert (extracted / "inside" / "readme.txt").read_text() == "ok"
    assert (extracted / "top.txt").read_text() == "hi"
    assert saved[0].note is None


# ---------------------------------------------------------------------------
# Fix 3: unbounded saved-file size
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_save_attachments_refuses_payload_over_max_saved_bytes(tmp_path, monkeypatch):
    monkeypatch.setattr(attachments_mod, "_MAX_SAVED_BYTES", 1000)

    saved = await save_attachments(str(tmp_path), [{
        "kind": "document",
        "file_name": "huge.bin",
        "data": b"x" * 2000,
    }])

    assert len(saved) == 1
    assert saved[0].path is None
    assert saved[0].note is not None

    inbox = tmp_path / ".claude" / "voice-bridge-inbox"
    written_files = [f for f in inbox.rglob("*") if f.is_file()] if inbox.exists() else []
    assert written_files == []
