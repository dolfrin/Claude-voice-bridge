"""Saving Claude logins and switching between them."""

import json
import time

import pytest

from voice_bridge import accounts


def _login(home, uuid, token, *, expires=None, extra=None):
    (home / ".claude").mkdir(exist_ok=True)
    creds = home / ".claude" / ".credentials.json"
    creds.write_text(json.dumps({"claudeAiOauth": {
        "accessToken": f"a-{token}", "refreshToken": f"r-{token}",
        "refreshTokenExpiresAt": int((expires or time.time() + 86400) * 1000),
    }, "organizationUuid": f"org-{uuid}"}))
    creds.chmod(0o600)
    config = {"oauthAccount": {"accountUuid": uuid, "emailAddress": f"{uuid}@x"}}
    config.update(extra or {})
    (home / ".claude.json").write_text(json.dumps(config))


def test_switch_back_to_a_saved_account_keeps_everything_else(tmp_path):
    vault = tmp_path / "state" / "vault.json"
    _login(tmp_path, "A", "1")
    assert accounts.remember(tmp_path, vault) == "A"
    _login(tmp_path, "B", "2", extra={"projects": {"/p": {"hasTrustDialogAccepted": True}}})

    email = accounts.switch(tmp_path, vault, "A")

    assert email == "A@x"
    creds = json.loads((tmp_path / ".claude" / ".credentials.json").read_text())
    assert creds["claudeAiOauth"]["refreshToken"] == "r-1"
    assert creds["organizationUuid"] == "org-A"
    assert (tmp_path / ".claude" / ".credentials.json").stat().st_mode & 0o777 == 0o600
    config = json.loads((tmp_path / ".claude.json").read_text())
    assert config["oauthAccount"]["accountUuid"] == "A"
    assert config["projects"]["/p"]["hasTrustDialogAccepted"] is True  # untouched
    assert {a["uuid"] for a in accounts.known(vault)} == {"A", "B"}  # B saved on the way out
    assert vault.stat().st_mode & 0o777 == 0o600


def test_expired_or_unknown_account_changes_nothing(tmp_path):
    vault = tmp_path / "vault.json"
    _login(tmp_path, "A", "1", expires=time.time() - 60)
    accounts.remember(tmp_path, vault)
    _login(tmp_path, "B", "2")
    before = (tmp_path / ".claude" / ".credentials.json").read_text()

    for uuid, reason in (("A", "account.expired"), ("Z", "account.unknown")):
        with pytest.raises(accounts.SwitchError, match=reason):
            accounts.switch(tmp_path, vault, uuid)
    assert (tmp_path / ".claude" / ".credentials.json").read_text() == before
    assert [a["usable"] for a in accounts.known(vault) if a["uuid"] == "A"] == [False]


def test_remember_needs_a_real_login(tmp_path):
    assert accounts.remember(tmp_path, tmp_path / "vault.json") is None
    assert not (tmp_path / "vault.json").exists()
