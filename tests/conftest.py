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
