# src/voice_bridge/sanitizer.py
"""Deterministic code-free voice sanitizer.

Guarantees the spoken (voice) channel never contains code, regardless of
agent cooperation. Strips fenced + inline code, hex colors, dimensions/units,
file paths, URLs, and snake_case/camelCase/CONSTANT identifiers, collapses
whitespace, and caps length (global_constraints: "Voice channel NEVER
contains code").

Pure stdlib (re) — no external dependencies, no I/O.

Identifier handling decisions:
- snake_case  : stripped  (load_config, effective_voice)
- camelCase   : stripped  (getSessionId, SessionManager — has internal upper after lower)
- CONSTANT    : stripped  (TELEGRAM_BOT_TOKEN — two+ ALL_CAPS segments joined by _)
- Single word with initial cap (Telegram, Claude, OAuth): KEPT — not a code identifier
- Numbers without units (42, 100): KEPT
"""

from __future__ import annotations

import re

TRUNCATION_MARKER = " Detalės tekste."

# Fenced code blocks: ``` ... ``` (any/no language tag), across lines.
_FENCED = re.compile(r"```.*?```", re.DOTALL)

# Inline code spans: `code`.
_INLINE_CODE = re.compile(r"`[^`]*`")

# URLs (http/https/ftp scheme or bare www.).
_URL = re.compile(r"\b(?:https?|ftp)://\S+|\bwww\.\S+", re.IGNORECASE)

# Hex colors: #fff, #ffffff, #1a2b3c4d (3/4/6/8 hex digits).
_HEX_COLOR = re.compile(r"#[0-9a-fA-F]{3,8}\b")

# Dimensions / units: 10px, 2rem, 100vh, 1.5em, 50%, 12pt, 3ex, 0.5vw ...
_UNIT = re.compile(
    r"\b\d+(?:\.\d+)?(?:px|rem|em|ex|vh|vw|vmin|vmax|pt|pc|cm|mm|in|ch|fr|deg|ms|s)\b"
    r"|\b\d+(?:\.\d+)?%",
    re.IGNORECASE,
)

# File paths: absolute (/a/b/c), relative (a/b/c), or any token with a
# file extension (main.py, README.md). Must contain a '/' or a '.<ext>'.
_PATH = re.compile(
    r"(?:\.{0,2}/)?[\w.\-]+(?:/[\w.\-]+)+/?"          # has at least one '/'
    r"|\b[\w\-]+\.[A-Za-z][\w]{0,7}\b"                # filename.ext
)

# CONSTANT_CASE: two+ segments of UPPERCASE/digits joined by underscores.
# A single all-caps word (like ALTER, TABLE) does NOT match — requires underscore.
_CONSTANT = re.compile(r"\b[A-Z][A-Z0-9]*(?:_[A-Z0-9]+)+\b")

# snake_case: lowercase/digit segments joined by underscores (load_config).
_SNAKE = re.compile(r"\b[a-z0-9]+(?:_[a-z0-9]+)+\b")

# camelCase / PascalCase with an internal capital transition (lower→upper).
# Matches: getSessionId, SessionManager (has lowercase then uppercase inside)
# Does NOT match: OAuth (O→A is upper→upper, no lower→upper transition)
# Does NOT match: Telegram, Claude (no internal lower→upper transition)
_CAMEL = re.compile(r"\b[A-Za-z]+[a-z][A-Z][A-Za-z0-9]*\b")

# Bare separator line: a line that is exactly '---' (optional surrounding ws).
_SEPARATOR_LINE = re.compile(r"^\s*---\s*$")

# Whitespace runs (incl. newlines/tabs) to collapse to a single space.
_WS = re.compile(r"\s+")

# Leftover lone punctuation tokens (orphaned by stripping) e.g. " : ", " | ".
_LONE_SYMBOL = re.compile(r"(?<=\s)[|:;=<>~^*_+/\\]+(?=\s)")


def to_spoken(text: str, max_chars: int = 600) -> str:
    """Return a code-free, voice-friendly version of ``text``.

    Strips fenced + inline code, URLs, hex colors, units, file paths, and
    code identifiers (snake_case/camelCase/CONSTANT), collapses whitespace,
    and caps the result at ``max_chars`` (appending ``TRUNCATION_MARKER`` if
    truncated or if content was dropped leaving it over the cap).
    """
    s = text

    # Order matters: remove fenced blocks before anything else can match inside.
    s = _FENCED.sub(" ", s)
    s = _INLINE_CODE.sub(" ", s)
    s = _URL.sub(" ", s)
    s = _HEX_COLOR.sub(" ", s)
    s = _UNIT.sub(" ", s)
    s = _PATH.sub(" ", s)
    s = _CONSTANT.sub(" ", s)
    s = _SNAKE.sub(" ", s)
    s = _CAMEL.sub(" ", s)
    s = _LONE_SYMBOL.sub(" ", s)

    # Collapse whitespace and trim.
    s = _WS.sub(" ", s).strip()

    # Tidy spaces left before sentence punctuation by removed tokens.
    s = re.sub(r"\s+([.,!?;:])", r"\1", s)
    s = _WS.sub(" ", s).strip()

    if len(s) > max_chars:
        s = s[:max_chars].rstrip()
        # Avoid cutting mid-word: drop a trailing partial word if present.
        if " " in s:
            s = s[: s.rstrip().rfind(" ")].rstrip()
        s = s.rstrip(".,!?;: ") + TRUNCATION_MARKER

    return s


def prepare_outbound(message: str) -> tuple[str, str]:
    """Split an outbound message into ``(full_text, spoken)``.

    ``full_text`` is the entire message unchanged. ``spoken`` is
    ``to_spoken`` applied to the part before the first line that is exactly
    ``'---'`` (or the whole message if there is no such line).
    """
    lines = message.split("\n")
    spoken_source = message
    for i, line in enumerate(lines):
        if _SEPARATOR_LINE.match(line):
            spoken_source = "\n".join(lines[:i])
            break
    return message, to_spoken(spoken_source)
