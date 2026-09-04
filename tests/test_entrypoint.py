"""Sessions started from Telegram must not be filtered out of the IDE list.

The Claude Code VS Code extension hides any session whose entrypoint is
sdk-cli/sdk-ts/sdk-py, which is exactly what the SDK stamps by default.
"""

import pytest

from voice_bridge.sessions import _entrypoint_env

HIDDEN_BY_THE_EXTENSION = {"sdk-cli", "sdk-ts", "sdk-py"}


def test_unset_keeps_the_sdk_default(monkeypatch):
    monkeypatch.delenv("CLAUDE_CODE_ENTRYPOINT", raising=False)

    # Nothing to merge means the SDK keeps stamping sdk-py.
    assert _entrypoint_env() == {}


def test_blank_is_treated_as_unset(monkeypatch):
    monkeypatch.setenv("CLAUDE_CODE_ENTRYPOINT", "   ")

    assert _entrypoint_env() == {}


def test_configured_value_overrides_the_default(monkeypatch):
    monkeypatch.setenv("CLAUDE_CODE_ENTRYPOINT", "voice-bridge")

    env = _entrypoint_env()

    assert env == {"CLAUDE_CODE_ENTRYPOINT": "voice-bridge"}
    assert env["CLAUDE_CODE_ENTRYPOINT"] not in HIDDEN_BY_THE_EXTENSION


@pytest.mark.parametrize("value", sorted(HIDDEN_BY_THE_EXTENSION))
def test_the_hidden_values_are_still_allowed_if_asked_for(monkeypatch, value):
    # Not our default, but the knob should not silently rewrite what was set.
    monkeypatch.setenv("CLAUDE_CODE_ENTRYPOINT", value)

    assert _entrypoint_env() == {"CLAUDE_CODE_ENTRYPOINT": value}
