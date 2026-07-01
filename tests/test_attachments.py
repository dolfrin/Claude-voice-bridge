import io
import zipfile

import pytest

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

