"""Markdown -> Telegram HTML, and splitting answers that exceed the size limit."""

from voice_bridge.telegram_md import (
    TELEGRAM_MAX_MESSAGE,
    split_markdown,
    to_html,
)


# --------------------------------------------------------------------------
# formatting
# --------------------------------------------------------------------------
def test_bold_becomes_a_tag_not_literal_asterisks():
    # The bug this exists for: Telegram showed "**Kas įvyko**" verbatim.
    assert to_html("**Kas įvyko**") == "<b>Kas įvyko</b>"


def test_headings_become_bold():
    assert to_html("## Kas įvyko") == "<b>Kas įvyko</b>"
    assert to_html("# Top") == "<b>Top</b>"


def test_bullets_become_dots():
    assert to_html("- first\n- second") == "• first\n• second"
    assert to_html("  - nested") == "  • nested"


def test_a_horizontal_rule_becomes_a_visible_line():
    # Telegram has no <hr>; a bare "---" would otherwise read as stray dashes.
    assert to_html("a\n---\nb").splitlines()[1].startswith("─")


def test_inline_code_becomes_a_code_tag():
    assert to_html("run `git push` now") == "run <code>git push</code> now"


def test_a_fenced_block_keeps_its_language():
    out = to_html("```py\nx = 1\n```")

    assert out == '<pre><code class="language-py">x = 1</code></pre>'


def test_a_fence_without_a_language_still_works():
    assert to_html("```\nplain\n```") == "<pre><code>plain</code></pre>"


def test_links_become_anchors():
    out = to_html("see [docs](https://example.com/a)")

    assert out == 'see <a href="https://example.com/a">docs</a>'


# --------------------------------------------------------------------------
# escaping — a stray angle bracket must not break the whole message
# --------------------------------------------------------------------------
def test_angle_brackets_in_prose_are_escaped():
    assert to_html("a < b & c > d") == "a &lt; b &amp; c &gt; d"


def test_code_content_is_escaped_too():
    out = to_html("`if a<b && c>d`")

    assert out == "<code>if a&lt;b &amp;&amp; c&gt;d</code>"


def test_markdown_inside_code_is_left_alone():
    # Otherwise a shell glob turns half the message bold.
    out = to_html("`ls **/*.py`")

    assert out == "<code>ls **/*.py</code>"


def test_markdown_inside_a_fence_is_left_alone():
    out = to_html("```\n# not a heading\n- not a bullet\n**not bold**\n```")

    assert "# not a heading" in out
    assert "- not a bullet" in out
    assert "**not bold**" in out
    assert "<b>" not in out


def test_a_lone_asterisk_or_underscore_is_literal():
    # snake_case identifiers and maths appear far more often than italics.
    assert to_html("a * b and rent_cost") == "a * b and rent_cost"


def test_an_unclosed_bold_marker_is_left_as_text():
    assert to_html("**unfinished") == "**unfinished"


# --------------------------------------------------------------------------
# splitting
# --------------------------------------------------------------------------
def test_a_short_message_is_not_split():
    assert split_markdown("hello") == ["hello"]


def test_a_long_message_is_split_into_sendable_chunks():
    # It used to be dropped whole: over 4096 characters Telegram just refuses.
    text = "\n".join(f"line {i} " + "x" * 60 for i in range(200))

    chunks = split_markdown(text, limit=1000)

    assert len(chunks) > 1
    assert all(len(c) <= TELEGRAM_MAX_MESSAGE for c in chunks)
    assert all(len(to_html(c)) <= TELEGRAM_MAX_MESSAGE for c in chunks)


def test_splitting_loses_no_lines():
    text = "\n".join(f"line {i}" for i in range(300))

    rejoined = "\n".join(split_markdown(text, limit=200))

    assert rejoined.split("\n") == text.split("\n")


def test_a_fence_split_across_chunks_is_closed_and_reopened():
    # Half a fenced block renders as prose, and the rest of the answer with it.
    body = "\n".join(f"code line {i}" for i in range(100))
    chunks = split_markdown(f"intro\n```py\n{body}\n```", limit=300)

    assert len(chunks) > 1
    for chunk in chunks:
        assert chunk.count("```") % 2 == 0, chunk
    assert chunks[1].startswith("```py")
    # Every chunk still renders as code, not as prose.
    assert all("<pre>" in to_html(c) for c in chunks[1:])


def test_splitting_a_very_long_single_line_still_returns_it():
    # Nothing to split on; better one oversized chunk the sender can truncate
    # than a silently dropped answer.
    chunks = split_markdown("x" * 5000, limit=1000)

    assert len(chunks) == 1
