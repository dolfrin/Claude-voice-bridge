import sys
import types


def _ensure(name):
    if name not in sys.modules:
        sys.modules[name] = types.ModuleType(name)
    return sys.modules[name]


fw = _ensure("faster_whisper")
if not hasattr(fw, "WhisperModel"):
    fw.WhisperModel = object
_ensure("piper")


import pytest


@pytest.fixture(autouse=True)
def _no_real_usage_calls(monkeypatch):
    """Tests must never call Anthropic with the real login (the bridge's
    background sampler would, from any test that runs the main loop)."""
    import voice_bridge.usage as usage

    def _refuse(home, timeout=10):
        raise RuntimeError("no network in tests")

    monkeypatch.setattr(usage, "fetch_limits", _refuse)


@pytest.fixture(autouse=True)
def _private_sent_log(monkeypatch, tmp_path):
    """Keep tests out of the real ~/.claude message->session log."""
    import voice_bridge.sent_log as sent_log

    target = tmp_path / "sent.jsonl"
    monkeypatch.setattr(sent_log, "_path", lambda path=None: path or target)


@pytest.fixture(autouse=True)
def _lithuanian_texts():
    """The suite asserts the Lithuanian texts; the English ones are covered by
    test_i18n (same keys, same fields)."""
    from voice_bridge import i18n

    i18n.set_language("lt")
    yield
    i18n.set_language("lt")
