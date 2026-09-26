from voice_bridge import sent_log


def test_lookup_last_and_prune(tmp_path, monkeypatch):
    path = tmp_path / "log.jsonl"
    sent_log.record(1, "s1", "/a", path)
    sent_log.record(2, None, "/b", path)
    with path.open("a") as fh:
        fh.write("not json\n")  # a hook died mid-write

    assert sent_log.lookup(1, path)["s"] == "s1"
    assert sent_log.lookup(3, path) is None
    assert sent_log.last(path)["m"] == 2

    monkeypatch.setattr(sent_log, "_KEEP_LINES", 2)
    sent_log.prune(path)
    assert len(path.read_text().splitlines()) == 2
