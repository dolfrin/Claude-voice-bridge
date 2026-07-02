import gzip
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


def _targz_bytes(entries: dict[str, bytes]) -> bytes:
    """Build a genuinely gzip-compressed .tar.gz (not just a .tar with a
    misleading name) so tests exercise the compressed-stream code path that
    tarfile.open() auto-detects."""
    return gzip.compress(_tar_bytes(entries))


def _corrupted_deflate_zip_bytes() -> bytes:
    """A real, well-formed zip (valid central directory, valid local header)
    whose DEFLATE payload bytes have been corrupted in place. zipfile.open()
    succeeds and infolist() works fine -- the corruption only surfaces as a
    zlib.error when the compressed stream is actually decompressed."""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("bad.bin", b"corrupt me please " * 200)
    raw = bytearray(buf.getvalue())
    sig = b"PK\x03\x04"
    idx = raw.index(sig)
    fn_len = int.from_bytes(raw[idx + 26 : idx + 28], "little")
    extra_len = int.from_bytes(raw[idx + 28 : idx + 30], "little")
    data_start = idx + 30 + fn_len + extra_len
    corrupt_at = data_start + 5
    for i in range(corrupt_at, corrupt_at + 10):
        raw[i] = 0xFF
    return bytes(raw)


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


# ---------------------------------------------------------------------------
# Fix pass: compressed-tar member enumeration must be lazy/bounded, and
# corrupted archives must not crash the whole attachment batch.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_extract_tar_does_not_call_getmembers(tmp_path, monkeypatch):
    """getmembers() forces tarfile to decompress the entire stream up front.
    The fix requires lazy `for member in archive:` iteration instead -- assert
    getmembers() is never touched by exercising a real, valid .tar.gz."""

    def boom(self):
        raise AssertionError("getmembers() must not be called -- forces full decompression")

    monkeypatch.setattr(tarfile.TarFile, "getmembers", boom)

    data = _targz_bytes({"a.txt": b"1", "b.txt": b"2"})

    saved = await save_attachments(str(tmp_path), [{
        "kind": "document",
        "file_name": "archive.tar.gz",
        "data": data,
    }])

    extracted = tmp_path / saved[0].extracted_to
    assert (extracted / "a.txt").read_text() == "1"
    assert (extracted / "b.txt").read_text() == "2"
    assert saved[0].note is None


@pytest.mark.asyncio
async def test_extract_targz_skips_member_over_per_member_cap(tmp_path, monkeypatch):
    """A compressed-tar member whose header declares a size over the
    per-member cap must be rejected from the header alone -- without the code
    ever calling extractfile()/reading its payload."""
    monkeypatch.setattr(attachments_mod, "_ARCHIVE_MAX_MEMBER_BYTES", 100)
    data = _targz_bytes({"huge.bin": b"x" * 1000, "small.txt": b"tiny"})

    saved = await save_attachments(str(tmp_path), [{
        "kind": "document",
        "file_name": "archive.tar.gz",
        "data": data,
    }])

    extracted = tmp_path / saved[0].extracted_to
    assert not (extracted / "huge.bin").exists()
    assert (extracted / "small.txt").read_text() == "tiny"


@pytest.mark.asyncio
async def test_extract_targz_aborts_when_total_size_exceeds_cap(tmp_path, monkeypatch):
    """Multiple members whose cumulative size exceeds the total cap must
    abort extraction partway through -- not all members get written, and the
    call returns promptly (no hang)."""
    monkeypatch.setattr(attachments_mod, "_ARCHIVE_MAX_TOTAL_BYTES", 1000)
    entries = {f"file-{i}.bin": b"x" * 500 for i in range(10)}
    data = _targz_bytes(entries)

    saved = await save_attachments(str(tmp_path), [{
        "kind": "document",
        "file_name": "archive.tar.gz",
        "data": data,
    }])

    extracted = tmp_path / saved[0].extracted_to
    written = [f for f in extracted.rglob("*") if f.is_file()]
    total_written = sum(f.stat().st_size for f in written)
    assert total_written <= 1000
    assert len(written) < len(entries)
    assert saved[0].note is not None
    assert "cap" in saved[0].note.lower() or "total" in saved[0].note.lower()


@pytest.mark.asyncio
async def test_save_attachments_extracts_targz_multiple_members_regression(tmp_path):
    """Lazy iteration must not break the happy path: a normal multi-file
    .tar.gz (with a subdirectory) still extracts every member correctly."""
    data = _targz_bytes({"a.txt": b"1", "sub/b.txt": b"2", "c.txt": b"3"})

    saved = await save_attachments(str(tmp_path), [{
        "kind": "document",
        "file_name": "multi.tar.gz",
        "data": data,
    }])

    extracted = tmp_path / saved[0].extracted_to
    assert (extracted / "a.txt").read_text() == "1"
    assert (extracted / "sub" / "b.txt").read_text() == "2"
    assert (extracted / "c.txt").read_text() == "3"
    assert saved[0].note is None


@pytest.mark.asyncio
async def test_extract_archive_targz_truncated_returns_graceful_failure(tmp_path):
    """A truncated .tar.gz body raises EOFError from gzip -- this must be
    caught and turned into a graceful failure result, not an exception."""
    data = _targz_bytes({"a.txt": b"x" * 1000})
    truncated = data[: len(data) // 2]

    saved = await save_attachments(str(tmp_path), [{
        "kind": "document",
        "file_name": "broken.tar.gz",
        "data": truncated,
    }])

    assert len(saved) == 1
    assert saved[0].path is not None  # the raw upload itself is still saved
    assert saved[0].extracted_to is None
    assert saved[0].note is not None
    assert "failed" in saved[0].note.lower()


@pytest.mark.asyncio
async def test_save_attachments_corrupted_zip_does_not_drop_valid_sibling(tmp_path):
    """A corrupted DEFLATE stream raises zlib.error, not zipfile.BadZipFile.
    That must be caught inside extraction (graceful failure for the bad zip)
    AND must not abort the rest of the batch -- the valid sibling attachment
    in the same message still gets saved."""
    bad_zip = _corrupted_deflate_zip_bytes()

    saved = await save_attachments(str(tmp_path), [
        {"kind": "document", "file_name": "bad.zip", "data": bad_zip},
        {"kind": "document", "file_name": "good.txt", "data": b"hello world"},
    ])

    assert len(saved) == 2
    bad_result, good_result = saved

    assert bad_result.path is not None  # raw upload saved even though extraction failed
    assert bad_result.extracted_to is None
    assert bad_result.note is not None
    assert "failed" in bad_result.note.lower()

    assert good_result.path is not None
    assert (tmp_path / good_result.path).read_bytes() == b"hello world"


@pytest.mark.asyncio
async def test_save_attachments_batch_survives_unexpected_exception_in_one_item(tmp_path, monkeypatch):
    """Backstop for the per-attachment batch guard: even an exception type
    unrelated to archive parsing must not drop the rest of the batch."""
    original_extract_archive = attachments_mod._extract_archive

    def flaky_extract_archive(path):
        if path.name.endswith("boom.bin"):
            raise RuntimeError("simulated unexpected failure")
        return original_extract_archive(path)

    monkeypatch.setattr(attachments_mod, "_extract_archive", flaky_extract_archive)

    saved = await save_attachments(str(tmp_path), [
        {"kind": "document", "file_name": "boom.bin", "data": b"whatever"},
        {"kind": "document", "file_name": "fine.txt", "data": b"hello"},
    ])

    assert len(saved) == 2
    assert saved[0].path is None
    assert saved[0].note is not None
    assert saved[1].path is not None
    assert (tmp_path / saved[1].path).read_bytes() == b"hello"
