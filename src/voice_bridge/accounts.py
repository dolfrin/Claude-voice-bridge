"""The Claude logins seen on this PC, and switching between them (opt-in).

People who hold several Claude accounts switch as each runs out. Claude Code
keeps exactly one login: ``~/.claude/.credentials.json`` (the OAuth tokens)
plus the ``oauthAccount`` block of ``~/.claude.json`` (who that is). When
``CLAUDE_ACCOUNT_SWITCHING`` is on, the bridge copies the current login into
its own vault every few minutes, so any account logged in here once can be
put back later from Telegram -- until its refresh token expires, after which
it needs one normal ``/login``.

The vault holds live tokens for every account: it is written 0600, next to
the bridge's database, and only when the feature is switched on. Linux only:
elsewhere Claude Code keeps its login in the system keychain, not in a file.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path


class SwitchError(Exception):
    """Why a switch did not happen: an i18n key (see ``i18n.t``)."""


def vault_path(db_path: str) -> Path:
    return Path(db_path).parent / "claude-accounts.json"


def _read(path: Path) -> dict:
    try:
        data = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return {}
    return data if isinstance(data, dict) else {}


def _write(path: Path, data: dict, mode: int) -> None:
    """Replace *path* atomically with *data*, created with *mode*."""
    tmp = path.with_name(path.name + ".bridge-tmp")
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, mode)
    with os.fdopen(fd, "w") as fh:
        json.dump(data, fh, indent=2)
    os.chmod(tmp, mode)
    os.replace(tmp, path)


def _mode(path: Path, default: int) -> int:
    try:
        return path.stat().st_mode & 0o777
    except OSError:
        return default


def _current(home: Path) -> tuple[str, dict] | None:
    """``(account_uuid, entry)`` for the login in place now, or None."""
    creds = _read(home / ".claude" / ".credentials.json")
    oauth = creds.get("claudeAiOauth")
    account = _read(home / ".claude.json").get("oauthAccount") or {}
    uuid = account.get("accountUuid")
    if not isinstance(oauth, dict) or not oauth.get("refreshToken") or not uuid:
        return None
    return uuid, {
        "email": account.get("emailAddress") or "?",
        "oauth": oauth,
        "organizationUuid": creds.get("organizationUuid"),
        "account": account,
    }


def remember(home: Path, vault: Path) -> str | None:
    """Save the current login into the vault; its account uuid, or None."""
    current = _current(home)
    if current is None:
        return None
    uuid, entry = current
    saved = _read(vault)
    if (saved.get(uuid) or {}).get("oauth") != entry["oauth"]:
        saved[uuid] = {**entry, "saved": time.time()}
        vault.parent.mkdir(parents=True, exist_ok=True)
        _write(vault, saved, 0o600)
    return uuid


def known(vault: Path, now: float | None = None) -> list[dict]:
    """Every saved account: ``{uuid, email, usable, expires}``, by email."""
    now = time.time() if now is None else now
    out = []
    for uuid, entry in _read(vault).items():
        expires = (entry.get("oauth") or {}).get("refreshTokenExpiresAt") or 0
        expires = expires / 1000 if expires > 1e12 else expires
        out.append({
            "uuid": uuid, "email": entry.get("email", "?"),
            "usable": not expires or expires > now, "expires": expires,
        })
    return sorted(out, key=lambda a: a["email"])


def switch(home: Path, vault: Path, uuid: str, now: float | None = None) -> str:
    """Put the saved login *uuid* in place; returns its email.

    The current login is saved first, so the account being left keeps its
    newest tokens. Raises :class:`SwitchError` without touching anything when
    the target is unknown or its refresh token has expired.
    """
    now = time.time() if now is None else now
    remember(home, vault)
    target = _read(vault).get(uuid)
    if not target:
        raise SwitchError("account.unknown")
    if not next((a["usable"] for a in known(vault, now) if a["uuid"] == uuid), False):
        raise SwitchError("account.expired")

    creds_path = home / ".claude" / ".credentials.json"
    creds = _read(creds_path)
    creds["claudeAiOauth"] = target["oauth"]
    if target.get("organizationUuid"):
        creds["organizationUuid"] = target["organizationUuid"]
    _write(creds_path, creds, _mode(creds_path, 0o600))

    # Re-read right before writing: running Claude Code processes rewrite
    # this file often, and everything but the account block must survive.
    config_path = home / ".claude.json"
    config = _read(config_path)
    config["oauthAccount"] = target["account"]
    _write(config_path, config, _mode(config_path, 0o644))

    if (_current(home) or ("",))[0] != uuid:
        raise SwitchError("account.not_applied")
    return target.get("email", "?")
