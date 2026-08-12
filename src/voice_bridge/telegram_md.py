"""Turn the Markdown Claude writes into the small HTML subset Telegram renders.


Telegram has no Markdown of its own worth using: legacy ``Markdown`` breaks on
any unbalanced ``*`` or ``_``, and ``MarkdownV2`` demands every one of
``_*[]()~`>#+-=|{}.!`` be escaped, which prose about code fails constantly. HTML
is the forgiving option — a fixed tag set, and only ``&<>`` need escaping.

Supported, because that is what Claude actually emits: fenced and inline code,
``**bold**``, headings, ``---`` rules, ``-``/``*`` bullets, and links. Single
``*``/``_`` are left alone on purpose — they appear inside identifiers and maths
far more often than they mean italics.

Pure stdlib, no I/O.
"""

from __future__ import annotations

import html
import re

# Telegram rejects a sendMessage body longer than this.
TELEGRAM_MAX_MESSAGE = 4096

_FENCE = re.compile(r"```([\w+.#-]*)[ \t]*\r?\n?(.*?)```", re.DOTALL)
_INLINE_CODE = re.compile(r"`([^`\n]+)`")
_BOLD = re.compile(r"\*\*(.+?)\*\*", re.DOTALL)
_LINK = re.compile(r"\[([^\]\n]+)\]\((https?://[^)\s]+)\)")
_HEADING = re.compile(r"^ {0,3}#{1,6}\s+(.*)$")
_RULE = re.compile(r"^\s*(?:-{3,}|_{3,}|\*{3,})\s*$")
_BULLET = re.compile(r"^(\s*)[-*]\s+(.*)$")

_RULE_LINE = "─" * 12
_STASH = re.compile("\x00(\\d+)\x00")


def to_html(text: str) -> str:
    """Render Markdown as Telegram-flavoured HTML.

    Code is lifted out first and put back last, so nothing inside a code block
    is mistaken for formatting and nothing formats its way into code.
    """
    blocks: list[str] = []

    def stash(rendered: str) -> str:
        blocks.append(rendered)
        return f"\x00{len(blocks) - 1}\x00"

    def fence(match: re.Match) -> str:
        language, body = match.group(1), match.group(2)
        escaped = html.escape(body.rstrip("\n"), quote=False)
        attr = (
            f' class="language-{html.escape(language, quote=True)}"' if language else ""
        )
        return stash(f"<pre><code{attr}>{escaped}</code></pre>")

    def inline(match: re.Match) -> str:
        return stash(f"<code>{html.escape(match.group(1), quote=False)}</code>")

    s = _FENCE.sub(fence, text)
    s = _INLINE_CODE.sub(inline, s)
    s = html.escape(s, quote=False)
    s = _LINK.sub(
        lambda m: f'<a href="{html.escape(m.group(2), quote=True)}">{m.group(1)}</a>', s
    )
    s = _BOLD.sub(r"<b>\1</b>", s)

    lines: list[str] = []
    for line in s.split("\n"):
        heading = _HEADING.match(line)
        if heading:
            lines.append(f"<b>{heading.group(1).strip()}</b>")
            continue
        if _RULE.match(line):
            lines.append(_RULE_LINE)
            continue
        bullet = _BULLET.match(line)
        if bullet:
            lines.append(f"{bullet.group(1)}• {bullet.group(2)}")
            continue
        lines.append(line)

    return _STASH.sub(lambda m: blocks[int(m.group(1))], "\n".join(lines))


def split_markdown(text: str, limit: int = 3000) -> list[str]:
    """Split Markdown into chunks that each survive on their own.

    Telegram drops any message over 4096 characters — a long answer used to fail
    outright rather than arrive in pieces. Splitting happens on line boundaries,
    and a fenced block spanning a boundary is closed and reopened so neither
    half renders as prose. The budget is under the hard limit because escaping
    and tags grow the text on the way to HTML.
    """
    if len(text) <= limit:
        return [text]

    chunks: list[str] = []
    current: list[str] = []
    size = 0
    fence_lang: str | None = None  # the language of the fence we are inside

    def flush() -> None:
        nonlocal current, size
        if not current:
            return
        body = "\n".join(current)
        if fence_lang is not None:
            body += "\n```"  # close what this chunk opened
        chunks.append(body)
        current = ["```" + fence_lang] if fence_lang is not None else []
        size = len(current[0]) + 1 if current else 0

    for line in text.split("\n"):
        if size and size + len(line) + 1 > limit:
            flush()
        current.append(line)
        size += len(line) + 1
        stripped = line.lstrip()
        if stripped.startswith("```"):
            fence_lang = None if fence_lang is not None else stripped[3:].strip()

    if current:
        chunks.append("\n".join(current))
    return [c for c in chunks if c.strip()]
