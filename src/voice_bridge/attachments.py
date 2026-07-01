"""Project-local storage for Telegram attachments."""
from __future__ import annotations

import asyncio
import re
import tarfile
import zipfile
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

_INBOX_DIR = Path(".claude") / "voice-bridge-inbox"


@dataclass(frozen=True)
class SavedAttachment:
    kind: str
    path: str
    extracted_to: str | None = None


async def save_attachments(cwd: str, attachments: list[dict]) -> list[SavedAttachment]:
    if not attachments:
        return []
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, _save_sync, cwd, attachments)


def format_attachment_prompt(text: str, saved: list[SavedAttachment]) -> str:
    if not saved:
        return text

    lines = [text.strip()] if text.strip() else []
    lines.append("Telegram attachmentai išsaugoti projekte:")
    for item in saved:
        lines.append(f"- {item.kind}: {item.path}")
        if item.extracted_to:
            lines.append(f"  išskleista į: {item.extracted_to}")
    lines.append("Peržiūrėk šiuos failus ir tęsk pagal mano žinutę.")
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
        raw_name = str(item.get("file_name") or f"{kind}-{index}.bin")
        filename = f"{stamp}-{index:02d}-{_safe_filename(raw_name)}"
        path = root / filename
        path.write_bytes(bytes(item.get("data") or b""))

        extracted_to = _extract_archive(path)
        saved.append(
            SavedAttachment(
                kind=kind,
                path=_project_relative(path, cwd),
                extracted_to=_project_relative(extracted_to, cwd) if extracted_to else None,
            )
        )
    return saved


def _extract_archive(path: Path) -> Path | None:
    suffixes = [suffix.lower() for suffix in path.suffixes]
    if path.suffix.lower() == ".zip":
        target = path.with_suffix(path.suffix + ".extracted")
        target.mkdir(exist_ok=True)
        with zipfile.ZipFile(path) as archive:
            for member in archive.infolist():
                dest = _safe_extract_path(target, member.filename)
                if dest is None or member.is_dir():
                    continue
                dest.parent.mkdir(parents=True, exist_ok=True)
                with archive.open(member) as src:
                    dest.write_bytes(src.read())
        return target

    if path.suffix.lower() == ".tar" or suffixes[-2:] in ([".tar", ".gz"], [".tar", ".xz"], [".tar", ".bz2"]):
        target = path.with_suffix(path.suffix + ".extracted")
        target.mkdir(exist_ok=True)
        with tarfile.open(path) as archive:
            for member in archive.getmembers():
                if not member.isfile():
                    continue
                dest = _safe_extract_path(target, member.name)
                if dest is None:
                    continue
                dest.parent.mkdir(parents=True, exist_ok=True)
                src = archive.extractfile(member)
                if src is not None:
                    dest.write_bytes(src.read())
        return target
    return None


def _safe_extract_path(root: Path, member_name: str) -> Path | None:
    dest = (root / member_name).resolve()
    try:
        dest.relative_to(root.resolve())
    except ValueError:
        return None
    return dest


def _safe_filename(name: str) -> str:
    name = Path(name).name
    name = re.sub(r"[^A-Za-z0-9._-]+", "-", name).strip(".-")
    return name or "file.bin"


def _project_relative(path: Path, cwd: str) -> str:
    try:
        return str(path.relative_to(Path(cwd)))
    except ValueError:
        return str(path)

