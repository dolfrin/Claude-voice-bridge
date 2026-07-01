from voice_bridge.transcript import append_transcript, transcript_path


async def test_append_transcript_creates_project_local_markdown(tmp_path):
    await append_transcript(str(tmp_path), "user", "Labas")
    await append_transcript(str(tmp_path), "assistant", "Atsakymas")

    path = transcript_path(str(tmp_path))
    text = path.read_text(encoding="utf-8")

    assert path == tmp_path / ".claude" / "voice-bridge-chat.md"
    assert "# Voice Bridge Chat" in text
    assert "Telegram" in text
    assert "Claude" in text
    assert "Labas" in text
    assert "Atsakymas" in text


async def test_append_transcript_ignores_blank_text(tmp_path):
    await append_transcript(str(tmp_path), "user", "   ")

    assert not transcript_path(str(tmp_path)).exists()
