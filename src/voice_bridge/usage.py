"""Claude subscription limits: the logged-in account's total, and this PC's part.

The account numbers come from the same endpoint Claude Code's own ``/usage``
reads: every limit the account has (the 5-hour session, the week, and
model-scoped weeks such as Fable). It is not a documented public API, so a
failure is reported as "no data", never guessed. The numbers cover EVERY
device using the account.

Anthropic does not say how much of that one PC used, and the transcripts in
``~/.claude/projects`` do not record which account a turn ran under (one PC may
switch between several). So the bridge keeps its own ledger: every few minutes
it records the logged-in account, each limit's percentage, and this PC's
price-weighted tokens under that account in each limit's window. From it:

* only turns made while THIS account was logged in are counted as this PC's;
* the cost of 1 % in tokens is learned from how far a limit rose against this
  PC's tokens. Other devices only ever push that ratio up, so the smallest one
  seen comes from when this PC was the only user. It is an estimate (≈);
* until there is enough rise to learn from, the rise since the first reading
  is shown as a ceiling: this PC cannot have used more than the account did.
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
from .i18n import t

logger = logging.getLogger(__name__)

_USAGE_URL = "https://api.anthropic.com/api/oauth/usage"
_SPANS = {"session": 5 * 3600, "weekly": 7 * 86400}
_TOP = 3
# Percentages arrive as whole numbers, so a ratio over a 1-2 point rise is
# mostly rounding error. Price tokens only once the account rose this much.
_MIN_RISE_PCT = 3
_KEEP_DAYS = 30
# Tell the user when the logged-in account crosses these, once per window.
_ALERT_AT = (80, 95)

# Relative token prices (input = 1), the same ratios across current Claude
# models. Cache reads dominate raw token counts but cost a tenth, so an
# unweighted sum would crown whichever session re-read the longest context.
# ponytail: one weight set for every model; add per-model factors if the PC
# mixes models heavily and the ≈ drifts.
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
    """The raw usage document from Anthropic. Raises on failure."""
    creds = json.loads((home / ".claude" / ".credentials.json").read_text())
    token = creds["claudeAiOauth"]["accessToken"]
    req = urllib.request.Request(_USAGE_URL, headers={
        "Authorization": f"Bearer {token}",
        "anthropic-beta": "oauth-2025-04-20",
    })
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.load(resp)


def parse_limits(data: dict) -> list[dict]:
    """Every session/weekly limit as ``{key, model, pct, reset, span}``.

    Read from the generic ``limits`` list so a new model-scoped limit shows up
    without a code change; the older fixed keys are the fallback.
    """
    out = []
    for item in data.get("limits") or []:
        span = _SPANS.get(item.get("group"))
        if span is None or item.get("percent") is None or not item.get("resets_at"):
            continue
        model = ((item.get("scope") or {}).get("model") or {}).get("display_name")
        out.append({
            "key": item.get("kind", "?") + (f":{model.lower()}" if model else ""),
            "model": model,
            "pct": float(item["percent"]),
            "reset": datetime.fromisoformat(item["resets_at"]).timestamp(),
            "span": span,
        })
    if out:
        return out
    for key, kind, group in (("five_hour", "session", "session"), ("seven_day", "weekly_all", "weekly")):
        window = data.get(key) or {}
        if window.get("utilization") is not None and window.get("resets_at"):
            out.append({
                "key": kind, "model": None, "pct": float(window["utilization"]),
                "reset": datetime.fromisoformat(window["resets_at"]).timestamp(),
                "span": _SPANS[group],
            })
    return out


def _turns(root: Path, since: float) -> list[tuple[str, float, str, float, str]]:
    """``(session_uuid, ts, model, weighted_tokens, cwd)`` for turns after *since*.

    Subagent transcripts (``<uuid>/subagents/*.jsonl``) count toward their
    parent session. One API response is written as several lines (one per
    content block) carrying the same usage, hence the dedup on message id.
    """
    out = []
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
                weight = sum(float(usage.get(k) or 0) * w for k, w in _WEIGHTS.items())
                if weight > 0:
                    out.append((uuid, ts, str(msg.get("model") or ""), weight, entry.get("cwd") or ""))
    return out


def _matches(turn_model: str, limit_model: str | None) -> bool:
    """Does a turn count toward a limit? Model-scoped limits count only their model."""
    return limit_model is None or limit_model.lower() in turn_model.lower()


def _load(ledger: Path) -> list[dict]:
    try:
        lines = ledger.read_text().splitlines()
    except OSError:
        return []
    out = []
    for line in lines:
        try:
            sample = json.loads(line)
        except json.JSONDecodeError:
            continue
        if "w" not in sample and "u5" in sample:
            # First ledger format: fixed 5-hour and weekly fields.
            sample["w"] = {
                key: {"u": sample[f"u{n}"], "r": sample[f"r{n}"], "s": sample[f"s{n}"], "l": sample[f"l{n}"]}
                for key, n in (("session", "5"), ("weekly_all", "7"))
                if sample.get(f"u{n}") is not None
            }
        out.append(sample)
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


def take_sample(home: Path, ledger: Path, now: float | None = None, full: bool = False) -> dict:
    """Record each limit and this PC's tokens toward it; return the sample.

    The sample also carries the parsed limits and the scanned turns under
    ``"limits"`` / ``"turns"`` for the caller; only the ledger fields are saved.
    ``full`` scans from each window's start rather than from the login, for
    /usage to show work it cannot attribute; the 5-minute sampler skips it.
    """
    now = time.time() if now is None else now
    samples = _load(ledger)
    account, email, tier = current_account(home)
    data = fetch_limits(home)
    limits = parse_limits(data)
    since = _account_since(samples, account, home)
    for limit in limits:
        limit["start"] = max(limit["reset"] - limit["span"], since)
    root = home / ".claude" / "projects"
    scan_from = min(
        (l["reset"] - l["span"] if full else l["start"] for l in limits), default=now
    )
    turns = _turns(root, scan_from)
    record = {
        "ts": now, "account": account, "email": email, "tier": tier, "since": since,
        "w": {
            l["key"]: {
                "u": l["pct"], "r": l["reset"], "s": l["start"],
                "l": sum(w for _, ts, m, w, _ in turns if ts >= l["start"] and _matches(m, l["model"])),
            }
            for l in limits
        },
    }
    _append(ledger, record, samples)
    others = other_accounts(samples + [record], account, now)
    prev = next((x for x in reversed(samples) if x["account"] == account), None)
    breakdown = [
        (row.get("display_name"), row.get("percent"))
        for row in ((data.get("seven_day_breakdown") or {}).get("rows") or [])
        if row.get("percent")
    ]
    return {
        **record, "limits": limits, "turns": turns, "breakdown": breakdown,
        "others": others, "alerts": _alerts(prev, record, others, now),
    }


def _title(key: str) -> str:
    """Lithuanian name of a limit, for sentences."""
    if key == "session":
        return t("usage.limit_session")
    if key.startswith("weekly_scoped:"):
        return t("usage.limit_week_model", model=key.split(":", 1)[1].capitalize())
    return t("usage.limit_week")


def other_accounts(samples: list[dict], current: str, now: float) -> list[dict]:
    """The other accounts this PC has been logged in to, freest first.

    Each from its LAST reading here, with every window whose reset time has
    passed counted as empty again. Other devices may have used them since:
    this is the best this PC can know without holding their logins.
    """
    latest: dict[str, dict] = {}
    for sample in sorted(samples, key=lambda x: x["ts"]):
        if sample["account"] != current:
            latest[sample["account"]] = sample
    out = []
    for sample in latest.values():
        limits = {
            key: (0.0 if w["r"] <= now else float(w["u"] or 0), w["r"])
            for key, w in sample.get("w", {}).items()
        }
        worst = max((v[0] for v in limits.values()), default=0.0)
        out.append({"uuid": sample["account"], "email": sample.get("email", "?"), "seen": sample["ts"],
                    "limits": limits, "free": 100 - worst})
    return sorted(out, key=lambda a: -a["free"])


def _account_line(acc: dict, now: float) -> str:
    parts = []
    for key, (pct, reset) in sorted(acc["limits"].items()):
        if pct == 0 and reset <= now:
            parts.append(t("usage.account_reset", limit=_title(key)))
        else:
            parts.append(t("usage.account_until", limit=_title(key), pct=f"{pct:.0f}", when=_stamp(reset, now)))
    return f"{acc['email']} — " + ", ".join(parts)


def _alerts(prev: dict | None, record: dict, others: list[dict], now: float) -> list[str]:
    """A notice per limit that crossed 80 % or 95 % since the previous reading.

    No previous reading (first start) means no alerts, so a restart never
    repeats one. A new window starts from 0 %.
    """
    if prev is None:
        return []
    out = []
    for key, w in record["w"].items():
        before = (prev.get("w") or {}).get(key)
        was = float(before["u"] or 0) if before and abs(before["r"] - w["r"]) < 300 else 0.0
        crossed = [x for x in _ALERT_AT if was < x <= float(w["u"] or 0)]
        if not crossed:
            continue
        text = t(
            "usage.alert", email=record["email"], limit=_title(key),
            pct=f"{float(w['u']):.0f}", when=_stamp(w["r"], now), until=_until(w["r"], now),
        )
        better = [a for a in others if a["free"] > 100 - float(w["u"] or 0)]
        text += "\n" + (
            t("usage.alert_freest", account=_account_line(better[0], now))
            if better else t("usage.alert_none_freer")
        )
        out.append(text)
    return out


def _window_samples(samples: list[dict], account: str, key: str, reset: float, start: float) -> list[dict]:
    """Readings of one limit within one window of one login, oldest first."""
    group = [
        s["w"][key] | {"ts": s["ts"]}
        for s in samples
        if s["account"] == account and key in s.get("w", {})
        # resets_at jitters by a second between reads.
        and abs(s["w"][key]["r"] - reset) < 300 and s["w"][key]["s"] == start
    ]
    return sorted(group, key=lambda x: x["ts"])


def _pct_per_token(samples: list[dict], account: str, key: str) -> float | None:
    """Smallest %-per-weighted-token seen for this account and limit.

    Within one window (same reset, same login) the limit's percentage and this
    PC's cumulative tokens both only grow, so the rise of one against the rise
    of the other prices a token even when the window began before the login.
    When the window lies wholly inside the login, its start (0 %, 0 tokens) is
    a valid origin too. Other devices only ever make a ratio larger, hence the
    minimum across windows.
    """
    windows: dict[tuple, list[dict]] = defaultdict(list)
    for s in samples:
        w = s.get("w", {}).get(key)
        if s["account"] == account and w and w.get("u") is not None:
            windows[(round(w["r"] / 300), w["s"])].append(w | {"ts": s["ts"]})
    ratios = []
    for group in windows.values():
        group.sort(key=lambda x: x["ts"])
        first, last = group[0], group[-1]
        rise, spent = last["u"] - first["u"], last["l"] - first["l"]
        if rise >= _MIN_RISE_PCT and spent > 0:
            ratios.append(rise / spent)
        span = _SPANS["session"] if key == "session" else _SPANS["weekly"]
        if first["s"] <= first["r"] - span + 1:
            ratios.extend(x["u"] / x["l"] for x in group if x["u"] >= _MIN_RISE_PCT and x["l"] > 0)
    return min(ratios) if ratios else None


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
        return t("usage.in_minutes", n=int(left // 60))
    return t("usage.in_hours", n=f"{hours:.0f}") if hours < 48 else t("usage.in_days", n=f"{hours / 24:.0f}")


def _stamp(ts: float, now: float) -> str:
    """Local time with its day: "šiandien 18:45", "vakar 09:10", "09-24 10:00"."""
    local = datetime.fromtimestamp(ts)
    days = (datetime.fromtimestamp(now).date() - local.date()).days
    day = {0: t("usage.today"), 1: t("usage.yesterday"), -1: t("usage.tomorrow")}.get(
        days, local.strftime("%m-%d")
    )
    return f"{day} {local.strftime('%H:%M')}"


def _bar(total: float, mine: float | None) -> str:
    """Ten cells: 🟦 this PC, then other devices (🟩, 🟨 past half, 🟥 past
    80 %), ⬜ what is left. Any use shows at least one cell."""
    used = max(1, round(total / 10)) if total > 0 else 0
    used = min(used, 10)
    own = min(used, max(1, round(mine / 10)) if mine and mine > 0 else 0)
    other = "🟥" if total >= 80 else "🟨" if total >= 50 else "🟩"
    return "🟦" * own + other * (used - own) + "⬜" * (10 - used)


def _approx(value: float) -> str:
    """"≈ 3.5 %", but just "< 0.1 %" -- an approximate bound reads oddly."""
    text = _pct(value)
    return text if text.startswith("<") else f"≈ {text}"


def _pct(value: float) -> str:
    if 0 < value < 0.1:
        return "< 0.1 %"
    return f"{value:.1f} %" if value < 10 else f"{value:.0f} %"


def _session_lines(root: Path, turns: list, now: float, k: float | None) -> list[str]:
    """This PC's sessions among *turns*, each as ≈ % of the limit when priced."""
    per: dict[str, list] = defaultdict(lambda: [0.0, "", 0.0])
    for uuid, ts, _, weight, cwd in turns:
        row = per[uuid]
        row[0] += weight
        row[1] = row[1] or cwd
        row[2] = max(row[2], ts)
    if not per:
        return ["  " + t("usage.no_sessions")]
    ranked = sorted(per.items(), key=lambda kv: kv[1][0], reverse=True)
    lines = []
    for uuid, (weight, cwd, last) in ranked[:_TOP]:
        path = next(root.glob(f"*/{uuid}.jsonl"), None)
        name = claude_history.title(path) if path else ""
        if len(name) > 40:
            name = name[:40] + "…"
        ago = max(0, int((now - last) // 60))
        ago_text = t("usage.ago_minutes", n=ago) if ago < 90 else t("usage.ago_hours", n=ago // 60)
        label = _project(cwd) + (f" · {name}" if name else "")
        amount = f"{_approx(k * weight)} — " if k is not None else "• "
        lines.append(f"  {amount}{label} ({ago_text})")
    if len(ranked) > _TOP:
        rest = sum(w for _, (w, _, _) in ranked[_TOP:])
        amount = f"{_approx(k * rest)} — " if k is not None else "• "
        lines.append(f"  {amount}" + t("usage.more_sessions", n=len(ranked) - _TOP))
    return lines


def format_usage(ledger: Path, home: Path | None = None, now: float | None = None) -> str:
    """The /usage message: every limit, this PC's part of it, and its sessions."""
    home = home or Path.home()
    now = time.time() if now is None else now
    try:
        sample = take_sample(home, ledger, now, full=True)
    except urllib.error.HTTPError as exc:
        logger.warning("usage: HTTP %s", exc.code)
        hint = t("usage.login_stale") if exc.code == 401 else ""
        return t("usage.fetch_failed_http", code=exc.code, hint=hint)
    except Exception:  # noqa: BLE001 - never break the command over this
        logger.exception("usage: sampling failed")
        return t("usage.fetch_failed")

    samples = _load(ledger)
    root = home / ".claude" / "projects"
    tier = sample["tier"].replace("default_claude_", "").replace("_", " ")
    lines = [
        f"📊 {sample['email']}" + (f" ({tier})" if tier else ""),
        t("usage.logged_in_since", when=_stamp(sample["since"], now)),
    ]
    for limit in sample["limits"]:
        start, reset = limit["reset"] - limit["span"], limit["reset"]
        counted_from = limit["start"]
        if limit["span"] == _SPANS["session"]:
            title = t("usage.window_session")
        else:
            title = (
                t("usage.window_week_model", model=limit["model"]) if limit["model"]
                else t("usage.window_week")
            )
        end = _stamp(reset, now)
        if end.split()[0] == _stamp(start, now).split()[0]:
            end = end.split()[1]  # same day: "šiandien 18:30 – 23:30"
        lines += ["", t("usage.window", title=title, start=_stamp(start, now), end=end, until=_until(reset, now))]
        k = _pct_per_token(samples, sample["account"], limit["key"])
        mine = sample["w"][limit["key"]]["l"]
        mine_pct = min(limit["pct"], k * mine) if k is not None else None
        total = t("usage.total", bar=_bar(limit["pct"], mine_pct), pct=f"{limit['pct']:.0f}")
        if limit["key"] == "weekly_all" and sample["breakdown"]:
            total += " (" + ", ".join(f"{name} {pct} %" for name, pct in sample["breakdown"]) + ")"
        lines.append(total)

        since = _stamp(counted_from, now)
        if mine_pct is not None:
            lines.append(t("usage.pc_estimate", since=since, pct=_approx(mine_pct)))
        else:
            readings = _window_samples(samples, sample["account"], limit["key"], reset, counted_from)
            if len(readings) > 1:
                first = readings[0]
                rise = limit["pct"] - first["u"]
                ceiling = "< 1 %" if rise < 1 else f"≤ ~{rise:.0f} %"
                lines.append(t(
                    "usage.pc_ceiling", since=since, ceiling=ceiling,
                    first=_stamp(first["ts"], now), rise=_MIN_RISE_PCT,
                ))
            else:
                lines.append(t("usage.pc_unknown", since=since))
        turns = [x for x in sample["turns"] if x[1] >= counted_from and _matches(x[2], limit["model"])]
        lines.extend(_session_lines(root, turns, now, k))
        # Work on this PC earlier in the window, before the ledger knew the
        # account: said out loud instead of silently left out.
        before = [
            x for x in sample["turns"]
            if start <= x[1] < counted_from and _matches(x[2], limit["model"])
        ]
        if before:
            sessions = len({x[0] for x in before})
            guess = (
                t("usage.unattributed_guess", pct=_approx(k * sum(x[3] for x in before)))
                if k is not None else ""
            )
            lines.append("  " + t(
                "usage.unattributed", until=_stamp(counted_from, now), n=sessions, guess=guess,
            ))
    lines += ["", t("usage.others_title")]
    if sample["others"]:
        for i, acc in enumerate(sample["others"]):
            star = "⭐ " if i == 0 else ""
            lines.append(t("usage.other_line", star=star, account=_account_line(acc, now), seen=_stamp(acc["seen"], now)))
    else:
        lines.append("  " + t("usage.others_none"))
    lines += [
        "",
        t("usage.legend"),
        t("usage.footnote"),
    ]
    return "\n".join(lines)
