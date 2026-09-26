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
