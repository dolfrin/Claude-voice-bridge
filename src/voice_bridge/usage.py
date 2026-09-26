"""Claude subscription limits: the logged-in account's total, and this PC's part.

The account numbers come from the same endpoint Claude Code's own ``/usage``
reads. It is not a documented public API, so a failure is reported as "no
data", never guessed. They cover EVERY device using the account.

Anthropic does not say how much of that one PC used, and the transcripts in
``~/.claude/projects`` do not record which account a turn ran under (one PC may
switch between several). So the bridge keeps its own ledger: every few minutes
it records the logged-in account, the account's percentages, and this PC's
price-weighted tokens under that account in each window. From it:

* only turns made while THIS account was logged in are counted as this PC's;
* the cost of 1 % in tokens is learned as the smallest percent-per-token ratio
  seen. Other devices only ever push the ratio up, so the minimum comes from
  the moments when this PC was the only user. It is an estimate, shown with ≈,
  and not shown at all until there is enough data.
"""

from __future__ import annotations

import json
import logging
import os
import time
import urllib.error
import urllib.request
from collections import defaultdict
from datetime import datetime
from pathlib import Path

from . import claude_history

logger = logging.getLogger(__name__)

_USAGE_URL = "https://api.anthropic.com/api/oauth/usage"
# (API key, label, window length, ledger field suffix)
_WINDOWS = (("five_hour", "5 val.", 5 * 3600, "5"), ("seven_day", "Savaitė", 7 * 86400, "7"))
_TOP = 5
# Percentages arrive as whole numbers, so a ratio taken at 1-4 % is mostly
# rounding error. Calibrate only from samples at or above this.
_MIN_CALIBRATION_PCT = 5
_KEEP_DAYS = 30

# Relative token prices (input = 1), the same ratios across current Claude
# models. Cache reads dominate raw token counts but cost a tenth, so an
# unweighted sum would crown whichever session re-read the longest context.
# ponytail: one weight set for every model; add per-model factors if the PC
# mixes Opus/Sonnet/Haiku heavily and the ≈ drifts.
_WEIGHTS = {
    "input_tokens": 1.0,
    "cache_creation_input_tokens": 1.25,
    "cache_read_input_tokens": 0.1,
    "output_tokens": 5.0,
}


def ledger_path(db_path: str) -> Path:
    """Where the account/limits ledger lives: next to the bridge's database."""
    return Path(db_path).parent / "claude-usage.jsonl"


def current_account(home: Path) -> tuple[str, str, str]:
    """``(account_uuid, email, rate_limit_tier)`` of the logged-in Claude account."""
    data = json.loads((home / ".claude.json").read_text())
    account = data.get("oauthAccount") or {}
    return (
        account.get("accountUuid") or "",
        account.get("emailAddress") or "?",
        account.get("organizationRateLimitTier") or "",
    )


def fetch_limits(home: Path, timeout: float = 10) -> dict:
    """``{"five_hour": {...}, "seven_day": {...}}`` from Anthropic. Raises on failure."""
    creds = json.loads((home / ".claude" / ".credentials.json").read_text())
    token = creds["claudeAiOauth"]["accessToken"]
    req = urllib.request.Request(_USAGE_URL, headers={
        "Authorization": f"Bearer {token}",
        "anthropic-beta": "oauth-2025-04-20",
    })
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.load(resp)


def session_weights(root: Path, since: float) -> dict[str, tuple[float, str, float]]:
    """``{session_uuid: (weighted_tokens, cwd, last_ts)}`` for turns after *since*.

    Subagent transcripts (``<uuid>/subagents/*.jsonl``) count toward their
    parent session. One API response is written as several lines (one per
    content block) carrying the same usage, hence the dedup on message id.
    """
    out: dict[str, list] = defaultdict(lambda: [0.0, "", 0.0])
    seen: set[str] = set()
    for path in root.glob("*/**/*.jsonl"):
        try:
            if path.stat().st_mtime < since:
                continue
        except OSError:
            continue
        rel = path.relative_to(root).parts
        uuid = rel[1] if len(rel) > 2 else path.stem
        with path.open(errors="replace") as fh:
            for line in fh:
                if '"usage"' not in line:
                    continue
                try:
                    entry = json.loads(line)
                    ts = datetime.fromisoformat(
                        entry["timestamp"].replace("Z", "+00:00")
                    ).timestamp()
                except (ValueError, KeyError, TypeError):
                    continue
                msg = entry.get("message") or {}
                usage = msg.get("usage")
                if entry.get("type") != "assistant" or not usage or ts < since:
                    continue
                key = f"{msg.get('id')}:{entry.get('requestId')}"
                if key in seen:
                    continue
                seen.add(key)
                row = out[uuid]
                row[0] += sum(float(usage.get(k) or 0) * w for k, w in _WEIGHTS.items())
                row[1] = row[1] or entry.get("cwd") or ""
                row[2] = max(row[2], ts)
    return {k: (v[0], v[1], v[2]) for k, v in out.items() if v[0] > 0}


def _load(ledger: Path) -> list[dict]:
    try:
        lines = ledger.read_text().splitlines()
    except OSError:
        return []
    out = []
    for line in lines:
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return out


def _append(ledger: Path, sample: dict, samples: list[dict]) -> None:
    ledger.parent.mkdir(parents=True, exist_ok=True)
    cutoff = sample["ts"] - _KEEP_DAYS * 86400
    if samples and samples[0]["ts"] < cutoff - 86400:
        # Prune about once a day instead of rewriting on every sample.
        kept = [s for s in samples if s["ts"] >= cutoff] + [sample]
        tmp = ledger.with_suffix(".tmp")
        tmp.write_text("".join(json.dumps(s) + "\n" for s in kept))
        os.replace(tmp, ledger)
        return
    with ledger.open("a") as fh:
        fh.write(json.dumps(sample) + "\n")


def _account_since(samples: list[dict], account: str, home: Path) -> float:
    """Since when this PC has been logged in to *account*, as far as we know.

    The credentials file is rewritten on login (and on token refresh, which
    only moves the time later): turns before it may belong to another account
    and are left out rather than misattributed.
    """
    if samples and samples[-1]["account"] == account:
        return samples[-1]["since"]
    try:
        logged_in = (home / ".claude" / ".credentials.json").stat().st_mtime
    except OSError:
        logged_in = time.time()
    return max(logged_in, samples[-1]["ts"]) if samples else logged_in


def take_sample(home: Path, ledger: Path, now: float | None = None) -> dict:
    """Record the account's limits and this PC's share of them; return the sample."""
    now = time.time() if now is None else now
    samples = _load(ledger)
    account, email, tier = current_account(home)
    limits = fetch_limits(home)
    since = _account_since(samples, account, home)
    sample = {"ts": now, "account": account, "email": email, "tier": tier, "since": since}
    root = home / ".claude" / "projects"
    for key, _, span, n in _WINDOWS:
        window = limits.get(key) or {}
        resets = window.get("resets_at")
        reset_ts = datetime.fromisoformat(resets).timestamp() if resets else now + span
        start = max(reset_ts - span, since)
        sample[f"u{n}"] = window.get("utilization")
        sample[f"r{n}"] = reset_ts
        sample[f"s{n}"] = start
        sample[f"l{n}"] = sum(w for w, _, _ in session_weights(root, start).values())
    _append(ledger, sample, samples)
    return sample


def _pct_per_token(samples: list[dict], account: str, n: str) -> float | None:
    """Smallest %-per-weighted-token seen for this account and window."""
    ratios = [
        s[f"u{n}"] / s[f"l{n}"]
        for s in samples
        if s["account"] == account
        and (s.get(f"u{n}") or 0) >= _MIN_CALIBRATION_PCT
        and (s.get(f"l{n}") or 0) > 0
        # Only windows fully inside the known login; otherwise part of the
        # percentage came from before we could attribute anything.
        and s[f"s{n}"] <= s[f"r{n}"] - _span(n) + 1
    ]
    return min(ratios) if ratios else None


def _span(n: str) -> int:
    return next(span for _, _, span, m in _WINDOWS if m == n)


def _project(cwd: str) -> str:
    path = Path(cwd)
    for parent in (path, *path.parents):
        if (parent / ".git").exists():
            return parent.name
    return path.name or "?"


def _until(ts: float, now: float) -> str:
    left = ts - now
    hours = left / 3600
    if hours < 1:
        return f"po {int(left // 60)} min"
    return f"po {hours:.0f} val." if hours < 48 else f"po {hours / 24:.0f} d."


def _stamp(ts: float, now: float) -> str:
    """Local time; the date only when it is not today."""
    local = datetime.fromtimestamp(ts)
    same_day = local.date() == datetime.fromtimestamp(now).date()
    return local.strftime("%H:%M" if same_day else "%m-%d %H:%M")


def _session_lines(root: Path, since: float, now: float, top: int = _TOP) -> list[str]:
    weights = session_weights(root, since)
    total = sum(w for w, _, _ in weights.values())
    if not total:
        return ["  (šiame PC su šita paskyra dar nedirbta)"]
    ranked = sorted(weights.items(), key=lambda kv: kv[1][0], reverse=True)
    lines = []
    for uuid, (weight, cwd, last) in ranked[:top]:
        path = next(root.glob(f"*/{uuid}.jsonl"), None)
        name = claude_history.title(path) if path else ""
        if len(name) > 40:
            name = name[:40] + "…"
        ago = max(0, int((now - last) // 60))
        ago_text = f"prieš {ago} min" if ago < 90 else f"prieš {ago // 60} val."
        label = _project(cwd) + (f" · {name}" if name else "")
        lines.append(f"  {weight / total * 100:.0f} % — {label} ({ago_text})")
    if len(ranked) > top:
        rest = sum(w for _, (w, _, _) in ranked[top:])
        lines.append(f"  {rest / total * 100:.0f} % — dar {len(ranked) - top} sesijos")
    return lines


def format_usage(ledger: Path, home: Path | None = None, now: float | None = None) -> str:
    """The /usage message: this account's total, this PC's part, its sessions."""
    home = home or Path.home()
    now = time.time() if now is None else now
    try:
        sample = take_sample(home, ledger, now)
    except urllib.error.HTTPError as exc:
        logger.warning("usage: HTTP %s", exc.code)
        hint = " — prisijungimas pasenęs, atidaryk Claude Code" if exc.code == 401 else ""
        return f"⚠️ Anthropic limitų negavau (HTTP {exc.code}){hint}."
    except Exception:  # noqa: BLE001 - never break the command over this
        logger.exception("usage: sampling failed")
        return "⚠️ Anthropic limitų negavau."

    samples = _load(ledger)
    tier = sample["tier"].replace("default_claude_", "").replace("_", " ")
    login = sample["since"]
    lines = [
        f"📊 {sample['email']}" + (f" ({tier})" if tier else ""),
        f"Šis PC prie jos prisijungęs nuo {_stamp(login, now)}.",
    ]
    for (_, _, span, n), title in zip(_WINDOWS, ("⏱ 5 val. langas", "📅 Savaitės langas")):
        reset = sample[f"r{n}"]
        start = reset - span
        counted_from = sample[f"s{n}"]
        partial = counted_from > start + 1
        lines.append("")
        lines.append(
            f"{title}: {_stamp(start, now)} – {_stamp(reset, now)} "
            f"(atsinaujins {_until(reset, now)})"
        )
        if sample[f"u{n}"] is not None:
            lines.append(f"• Bendrai paskyroj (visi įrenginiai): {sample[f'u{n}']:.0f} %")
        k = _pct_per_token(samples, sample["account"], n)
        if k is not None:
            mine = min(sample[f"u{n}"] or 0, k * sample[f"l{n}"])
            lines.append(f"• Šis PC: ≈ {mine:.0f} %")
        elif partial:
            # Part of this window's percentage predates the login, so it cannot
            # be split; the next window starts clean.
            lines.append(
                "• Šis PC: sužinosiu nuo kito lango — šitas prasidėjo "
                "prieš prisijungiant šia paskyra"
            )
        else:
            lines.append(
                f"• Šis PC: dar mokausi — reikia bent {_MIN_CALIBRATION_PCT} % naudojimo"
            )
        why = " (prisijungimo)" if partial else " (lango pradžios)"
        lines.append(f"• Šio PC sesijos nuo {_stamp(counted_from, now)}{why}:")
        lines.extend(_session_lines(home / ".claude" / "projects", counted_from, now, top=3))
    lines.append("")
    lines.append(
        "Sesijų % — dalis nuo šio PC darbo tame lange. Skaičiuojamos visos šio "
        "PC Claude Code sesijos: VS Code, terminalas, tiltas, agentai. "
        "claude.ai naršyklėje ar programėlėje — ne, jos patenka į „kitus“."
    )
    return "\n".join(lines)
