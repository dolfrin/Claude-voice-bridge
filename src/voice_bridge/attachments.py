"""Project-local storage for Telegram attachments."""
from __future__ import annotations

import asyncio
import logging
import os
import re
import shutil
import subprocess
import tarfile
import uuid
import zipfile
import zlib
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)

_INBOX_DIR = Path(".claude") / "voice-bridge-inbox"

# Fix 1: ffmpeg must never be allowed to block an executor thread forever.
_FFMPEG_TIMEOUT_SECONDS = 30

# Fix 3: refuse to save payloads larger than this. Telegram bot API downloads
# are capped around 20 MB, but stay defensive against a misbehaving client.
_MAX_SAVED_BYTES = 50 * 1024 * 1024  # 50 MB

# Fix 2: bound archive extraction against decompression bombs / resource abuse.
_ARCHIVE_MAX_TOTAL_BYTES = 100 * 1024 * 1024  # cumulative uncompressed bytes
_ARCHIVE_MAX_MEMBERS = 1000  # max number of members extracted from one archive
_ARCHIVE_MAX_MEMBER_BYTES = 50 * 1024 * 1024  # max uncompressed size of one member
_ARCHIVE_READ_CHUNK_BYTES = 1024 * 1024  # chunk size used while capping actual reads


@dataclass(frozen=True)
class SavedAttachment:
    kind: str
    path: str | None
    extracted_to: str | None = None
    note: str | None = None


async def save_attachments(cwd: str, attachments: list[dict]) -> list[SavedAttachment]:
    if not attachments:
        return []
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, _save_sync, cwd, attachments)


def format_attachment_prompt(text: str, saved: list[SavedAttachment]) -> str:
    if not saved:
        return text

    lines = [text.strip()] if text.strip() else []
    lines.append("Telegram attachments saved in this project:")
    for item in saved:
        if item.path is None:
            lines.append(f"- {item.kind}: NOT SAVED ({item.note or 'skipped'})")
            continue
        lines.append(f"- {item.kind}: {item.path}")
        if item.extracted_to:
            label = "video frames" if item.kind in {"video", "video_note"} else "extracted to"
            lines.append(f"  {label}: {item.extracted_to}")
        if item.note:
            lines.append(f"  note: {item.note}")
    if any(item.kind == "photo" for item in saved):
        lines.append("If this is a screenshot or photo, inspect the visible UI/text.")
    if any(item.kind in {"video", "video_note"} for item in saved):
        lines.append("If video frames are attached, use them for a quick review; inspect the full video if needed.")
    lines.append("Review these files and continue according to my message.")
    return "\n".join(lines).strip()


def inbox_path(cwd: str) -> Path:
    return Path(cwd) / _INBOX_DIR


def _save_sync(cwd: str, attachments: list[dict]) -> list[SavedAttachment]:
    root = inbox_path(cwd)
    root.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().astimezone().strftime("%Y%m%d-%H%M%S")
    saved: list[SavedAttachment] = []

    for index, item in enumerate(attachments, start=1):
        kind = str(item.get("kind") or "file")
        try:
            raw_name = str(item.get("file_name") or f"{kind}-{index}.bin")
            data = bytes(item.get("data") or b"")

            if len(data) > _MAX_SAVED_BYTES:
                logger.warning(
                    "attachment %r (%d bytes) exceeds max saved size of %d bytes; refusing to save",
                    raw_name,
                    len(data),
                    _MAX_SAVED_BYTES,
                )
                saved.append(
                    SavedAttachment(
                        kind=kind,
                        path=None,
                        note=f"skipped: {len(data)} bytes exceeds {_MAX_SAVED_BYTES} byte save limit",
                    )
                )
                continue

            # Bug fix: `stamp` is computed once per BATCH and `index` is the
            # position within that batch, so two attachments arriving in
            # DIFFERENT Telegram messages (e.g. an album, or two quick
            # messages) within the same second each start their own
            # `_save_sync` call with index=1 and an identical `stamp`,
            # producing the identical filename and silently overwriting one
            # attachment with the other. A short uuid4 component makes every
            # saved filename unique regardless of clock resolution or which
            # batch produced it, while the human-readable timestamp+index
            # prefix (and the original extension, via `_safe_filename`) is
            # kept for a human skimming the folder.
            unique = uuid.uuid4().hex[:8]
            filename = f"{stamp}-{index:02d}-{unique}-{_safe_filename(raw_name)}"
            path = root / filename
            path.write_bytes(data)

            archive_target, archive_note = _extract_archive(path)
            extracted_to = archive_target or _extract_video_frame(path, kind)

            saved.append(
                SavedAttachment(
                    kind=kind,
                    path=_project_relative(path, cwd),
                    extracted_to=_project_relative(extracted_to, cwd) if extracted_to else None,
                    note=archive_note,
                )
            )
        except Exception:
            # One malformed/unexpected attachment must never drop the rest of
            # the batch -- log it and keep processing the remaining items.
            logger.exception("attachment %d (kind=%r) failed to process; skipping", index, kind)
            saved.append(
                SavedAttachment(
                    kind=kind,
                    path=None,
                    note="skipped: failed to process attachment (unexpected error)",
                )
            )
    return saved


def _extract_archive(path: Path) -> tuple[Path | None, str | None]:
    suffixes = [suffix.lower() for suffix in path.suffixes]
    if path.suffix.lower() == ".zip":
        return _extract_zip(path)

    if path.suffix.lower() == ".tar" or suffixes[-2:] in ([".tar", ".gz"], [".tar", ".xz"], [".tar", ".bz2"]):
        return _extract_tar(path)

    return None, None


def _extract_zip(path: Path) -> tuple[Path | None, str | None]:
    target = path.with_suffix(path.suffix + ".extracted")
    target.mkdir(exist_ok=True)
    target_root = os.path.realpath(target)

    total_bytes = 0
    member_count = 0
    note: str | None = None

    try:
        with zipfile.ZipFile(path) as archive:
            for member in archive.infolist():
                if member.is_dir():
                    continue

                member_count += 1
                if member_count > _ARCHIVE_MAX_MEMBERS:
                    note = f"archive extraction stopped: exceeded {_ARCHIVE_MAX_MEMBERS} member limit"
                    logger.warning("%s: %s", path, note)
                    break

                if member.file_size > _ARCHIVE_MAX_MEMBER_BYTES:
                    logger.warning(
                        "%s: skipping member %r (%d bytes exceeds %d byte per-member cap)",
                        path,
                        member.filename,
                        member.file_size,
                        _ARCHIVE_MAX_MEMBER_BYTES,
                    )
                    continue

                if total_bytes + member.file_size > _ARCHIVE_MAX_TOTAL_BYTES:
                    note = f"archive extraction stopped: exceeded {_ARCHIVE_MAX_TOTAL_BYTES} byte total cap"
                    logger.warning("%s: %s", path, note)
                    break

                dest = _safe_extract_path(target_root, member.filename)
                if dest is None:
                    logger.warning("%s: skipping member %r (escapes extraction directory)", path, member.filename)
                    continue

                dest.parent.mkdir(parents=True, exist_ok=True)
                with archive.open(member) as src:
                    member_bytes = _read_capped(src, _ARCHIVE_MAX_MEMBER_BYTES)
                if member_bytes is None:
                    logger.warning(
                        "%s: skipping member %r (actual size exceeds %d byte per-member cap)",
                        path,
                        member.filename,
                        _ARCHIVE_MAX_MEMBER_BYTES,
                    )
                    continue

                dest.write_bytes(member_bytes)
                total_bytes += len(member_bytes)
    except (zipfile.BadZipFile, EOFError, OSError, zlib.error) as exc:
        logger.warning("%s: not a valid zip archive (%s); skipping extraction", path, exc)
        return None, "archive extraction failed: invalid zip file"

    return target, note


def _extract_tar(path: Path) -> tuple[Path | None, str | None]:
    target = path.with_suffix(path.suffix + ".extracted")
    target.mkdir(exist_ok=True)
    target_root = os.path.realpath(target)

    total_bytes = 0
    member_count = 0
    note: str | None = None

    try:
        with tarfile.open(path) as archive:
            # Iterate the archive LAZILY (the TarFile object itself is an
            # iterator that yields one member header at a time) instead of
            # calling archive.getmembers(), which forces tarfile to read
            # through -- and for compressed streams, fully decompress -- the
            # ENTIRE archive up front just to build the member list, before
            # any of the caps below are ever consulted. A ~50 MB crafted
            # .tar.gz can expand at up to ~1000:1, so getmembers() alone can
            # block this thread for a long time on a hostile input. Lazy
            # iteration lets every cap below take effect member-by-member,
            # so the loop stops as soon as a cap trips without paying to
            # enumerate (or decompress) anything after that point.
            for member in archive:
                # isfile() excludes directories, symlinks, hardlinks, devices,
                # and fifos -- only plain regular files are ever extracted.
                if not member.isfile():
                    continue

                member_count += 1
                if member_count > _ARCHIVE_MAX_MEMBERS:
                    note = f"archive extraction stopped: exceeded {_ARCHIVE_MAX_MEMBERS} member limit"
                    logger.warning("%s: %s", path, note)
                    break

                # Consult the header-declared size BEFORE calling
                # extractfile()/reading any payload bytes -- this is what
                # lets an oversized (or bomb) member be rejected purely from
                # its header, without paying to decompress its content.
                if member.size > _ARCHIVE_MAX_MEMBER_BYTES:
                    logger.warning(
                        "%s: skipping member %r (%d bytes exceeds %d byte per-member cap)",
                        path,
                        member.name,
                        member.size,
                        _ARCHIVE_MAX_MEMBER_BYTES,
                    )
                    continue

                if total_bytes + member.size > _ARCHIVE_MAX_TOTAL_BYTES:
                    note = f"archive extraction stopped: exceeded {_ARCHIVE_MAX_TOTAL_BYTES} byte total cap"
                    logger.warning("%s: %s", path, note)
                    break

                dest = _safe_extract_path(target_root, member.name)
                if dest is None:
                    logger.warning("%s: skipping member %r (escapes extraction directory)", path, member.name)
                    continue

                src = archive.extractfile(member)
                if src is None:
                    continue
                member_bytes = _read_capped(src, _ARCHIVE_MAX_MEMBER_BYTES)
                if member_bytes is None:
                    logger.warning(
                        "%s: skipping member %r (actual size exceeds %d byte per-member cap)",
                        path,
                        member.name,
                        _ARCHIVE_MAX_MEMBER_BYTES,
                    )
                    continue

                dest.parent.mkdir(parents=True, exist_ok=True)
                dest.write_bytes(member_bytes)
                total_bytes += len(member_bytes)
    except (tarfile.TarError, EOFError, OSError, zlib.error) as exc:
        logger.warning("%s: not a valid tar archive (%s); skipping extraction", path, exc)
        return None, "archive extraction failed: invalid tar file"

    return target, note


def _read_capped(fileobj, limit: int) -> bytes | None:
    """Read fileobj fully, but bail out as soon as more than `limit` bytes has
    been read. Never trust a member's declared size header alone."""
    chunks: list[bytes] = []
    total = 0
    while True:
        chunk = fileobj.read(_ARCHIVE_READ_CHUNK_BYTES)
        if not chunk:
            break
        total += len(chunk)
        if total > limit:
            return None
        chunks.append(chunk)
    return b"".join(chunks)


def _extract_video_frame(path: Path, kind: str) -> Path | None:
    if kind not in {"video", "video_note"} or shutil.which("ffmpeg") is None:
        return None
    target = path.with_suffix(path.suffix + ".frames")
    target.mkdir(exist_ok=True)
    frame = target / "frame-0001.jpg"
    try:
        result = subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-i",
                str(path),
                "-frames:v",
                "1",
                str(frame),
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
            timeout=_FFMPEG_TIMEOUT_SECONDS,
        )
    except subprocess.TimeoutExpired:
        logger.warning(
            "%s: ffmpeg frame extraction timed out after %ss; skipping frame extraction",
            path,
            _FFMPEG_TIMEOUT_SECONDS,
        )
        return None

    if result.returncode == 0 and frame.exists():
        return target
    return None


def _safe_extract_path(root: str, member_name: str) -> Path | None:
    """Resolve `member_name` against `root` (the realpath of the extraction
    directory) and reject anything that would land outside of it -- absolute
    paths, `..` escapes, or symlink tricks resolved away by realpath."""
    if not member_name or member_name.startswith("/") or member_name.startswith("\\"):
        return None

    candidate = os.path.join(root, member_name)
    dest_real = os.path.realpath(candidate)

    if dest_real != root and not dest_real.startswith(root + os.sep):
        return None

    return Path(dest_real)


def _safe_filename(name: str) -> str:
    name = Path(name).name
    name = re.sub(r"[^A-Za-z0-9._-]+", "-", name).strip(".-")
    return name or "file.bin"


def _project_relative(path: Path, cwd: str) -> str:
    try:
        return str(path.relative_to(Path(cwd)))
    except ValueError:
        return str(path)
