# tests/test_sanitizer.py
"""Tests for the deterministic code-free voice sanitizer.

The voice channel must NEVER contain code: fenced blocks, inline code,
hex colors, dimensions/units, file paths, URLs, or
snake_case/camelCase/CONSTANT identifiers (global_constraints).
"""

import pytest

from voice_bridge.sanitizer import prepare_outbound, to_spoken


# --------------------------------------------------------------------------
# to_spoken: fenced code blocks
# --------------------------------------------------------------------------

def test_to_spoken_strips_fenced_block():
    text = (
        "Pataisiau klaidą.\n"
        "```python\n"
        "def foo(x):\n"
        "    return x + 1\n"
        "```\n"
        "Viskas veikia."
    )
    out = to_spoken(text)
    assert "def" not in out
    assert "foo" not in out
    assert "return" not in out
    assert "```" not in out
    assert "Pataisiau klaidą." in out
    assert "Viskas veikia." in out


def test_to_spoken_strips_fenced_block_without_language_tag():
    text = "Prieš.\n```\nrm -rf /tmp/x\n```\nPo."
    out = to_spoken(text)
    assert "rm" not in out
    assert "tmp" not in out
    assert "Prieš." in out
    assert "Po." in out


def test_to_spoken_strips_multiple_fenced_blocks():
    text = "A\n```\ncode1\n```\nB\n```\ncode2\n```\nC"
    out = to_spoken(text)
    assert "code1" not in out
    assert "code2" not in out
    assert "A" in out and "B" in out and "C" in out


# --------------------------------------------------------------------------
# to_spoken: inline code
# --------------------------------------------------------------------------

def test_to_spoken_strips_inline_code():
    text = "Paleidau `pytest -q` ir viskas žalia."
    out = to_spoken(text)
    assert "pytest" not in out
    assert "`" not in out
    assert "Paleidau" in out
    assert "viskas žalia" in out


# --------------------------------------------------------------------------
# to_spoken: hex colors
# --------------------------------------------------------------------------

def test_to_spoken_strips_hex_colors():
    text = "Pakeičiau fono spalvą į #fff ir tekstą į #1a2b3c."
    out = to_spoken(text)
    assert "#fff" not in out
    assert "#1a2b3c" not in out
    assert "Pakeičiau fono spalvą" in out


# --------------------------------------------------------------------------
# to_spoken: dimensions / units
# --------------------------------------------------------------------------

def test_to_spoken_strips_units():
    text = "Nustačiau paraštę į 10px ir šriftą į 2rem, plotis 100vh."
    out = to_spoken(text)
    assert "10px" not in out
    assert "2rem" not in out
    assert "100vh" not in out
    assert "Nustačiau paraštę" in out


# --------------------------------------------------------------------------
# to_spoken: file paths
# --------------------------------------------------------------------------

def test_to_spoken_strips_file_paths():
    text = "Redagavau failą /home/home/Projects/app/main.py ir baigiau."
    out = to_spoken(text)
    assert "/home" not in out
    assert "main.py" not in out
    assert ".py" not in out
    assert "Redagavau failą" in out
    assert "baigiau" in out


def test_to_spoken_strips_relative_paths_and_filenames():
    text = "Atnaujinau src/voice_bridge/config.py ir README.md."
    out = to_spoken(text)
    assert "config.py" not in out
    assert "src/voice_bridge" not in out
    assert "README.md" not in out
    assert "Atnaujinau" in out


# --------------------------------------------------------------------------
# to_spoken: URLs
# --------------------------------------------------------------------------

def test_to_spoken_strips_urls():
    text = "Paskelbiau čia https://example.com/deploy?id=7 — pažiūrėk."
    out = to_spoken(text)
    assert "http" not in out
    assert "example.com" not in out
    assert "Paskelbiau" in out
    assert "pažiūrėk" in out


# --------------------------------------------------------------------------
# to_spoken: code identifiers
# --------------------------------------------------------------------------

def test_to_spoken_strips_snake_case():
    text = "Pridėjau load_config funkciją ir effective_voice pagalbinę."
    out = to_spoken(text)
    assert "load_config" not in out
    assert "effective_voice" not in out
    assert "Pridėjau" in out
    assert "funkciją" in out


def test_to_spoken_strips_camel_case():
    text = "Klasė SessionManager kviečia getSessionId metodą."
    out = to_spoken(text)
    assert "SessionManager" not in out
    assert "getSessionId" not in out
    assert "Klasė" in out


def test_to_spoken_strips_constant_identifiers():
    text = "Perskaičiau TELEGRAM_BOT_TOKEN ir APPROVAL_TIMEOUT reikšmes."
    out = to_spoken(text)
    assert "TELEGRAM_BOT_TOKEN" not in out
    assert "APPROVAL_TIMEOUT" not in out
    assert "Perskaičiau" in out
    assert "reikšmes" in out


def test_to_spoken_keeps_normal_capitalized_words():
    # A single capitalized word (sentence start, proper noun) is NOT a CONSTANT
    # and must survive.
    text = "Telegram žinutė išsiųsta. Claude atsakė."
    out = to_spoken(text)
    assert "Telegram" in out
    assert "Claude" in out


# --------------------------------------------------------------------------
# to_spoken: whitespace collapse
# --------------------------------------------------------------------------

def test_to_spoken_collapses_whitespace():
    text = "Pirma   eilutė.\n\n\nAntra\teilutė."
    out = to_spoken(text)
    assert "   " not in out
    assert "\n" not in out
    assert "\t" not in out
    assert out == "Pirma eilutė. Antra eilutė."


# --------------------------------------------------------------------------
# to_spoken: length cap
# --------------------------------------------------------------------------

def test_to_spoken_truncates_and_appends_marker():
    text = "žodis " * 300  # ~1800 chars of clean prose
    out = to_spoken(text, max_chars=100)
    assert len(out) <= 100 + len(" Detalės tekste.")
    assert out.endswith(" Detalės tekste.")


def test_to_spoken_no_marker_when_under_cap():
    text = "Trumpa žinutė."
    out = to_spoken(text, max_chars=600)
    assert out == "Trumpa žinutė."
    assert "Detalės tekste." not in out


def test_to_spoken_empty_after_stripping():
    text = "```\nonly code here\n```"
    out = to_spoken(text)
    assert out == ""


# --------------------------------------------------------------------------
# to_spoken: adversarial mixed code + prose
# --------------------------------------------------------------------------

def test_to_spoken_adversarial_mixed():
    text = (
        "Baigiau migraciją.\n"
        "```sql\n"
        "ALTER TABLE messages ADD COLUMN project TEXT;\n"
        "```\n"
        "Pakeičiau spalvą į #00ff00, paraštę 12px, faile "
        "/var/lib/voice-bridge/state.db, žiūrėk https://x.io/a, "
        "funkcija map_message ir klasė Store veikia. Liko testai."
    )
    out = to_spoken(text)
    # code / technical fragments gone
    assert "ALTER" not in out
    assert "TABLE" not in out
    assert "#00ff00" not in out
    assert "12px" not in out
    assert "/var/lib" not in out
    assert "state.db" not in out
    assert "http" not in out
    assert "x.io" not in out
    assert "map_message" not in out
    # prose survives
    assert "Baigiau migraciją" in out
    assert "Liko testai" in out
    # spoken line must read cleanly (no leftover backticks or pipes)
    assert "`" not in out


# --------------------------------------------------------------------------
# adversarial: edge cases for identifiers
# --------------------------------------------------------------------------

def test_to_spoken_strips_colon_unit_fragment():
    # ": 10px" is a common CSS snippet — must NOT appear in voice output
    text = "Buvo nustatytas stilius: 10px kraštinė."
    out = to_spoken(text)
    assert "10px" not in out


def test_to_spoken_keeps_oauth_word():
    # "OAuth" is a single capitalized word (not snake/camel/CONSTANT) — keep it
    text = "Naudojame OAuth autentifikacijai."
    out = to_spoken(text)
    assert "OAuth" in out


def test_to_spoken_strips_multipart_camel():
    # Multi-segment camelCase like getUserById must be stripped
    text = "Iškviečiau getUserById ir parseResponseData metodus."
    out = to_spoken(text)
    assert "getUserById" not in out
    assert "parseResponseData" not in out


def test_to_spoken_strips_dot_extension_only_filenames():
    # Even filenames without a path prefix must go
    text = "Atnaujinau failą config.yaml ir schema.json."
    out = to_spoken(text)
    assert "config.yaml" not in out
    assert "schema.json" not in out


def test_to_spoken_keeps_numbers_without_units():
    # Plain numbers (no unit) must survive
    text = "Yra 42 testai ir 100 eilučių."
    out = to_spoken(text)
    assert "42" in out
    assert "100" in out


# --------------------------------------------------------------------------
# prepare_outbound: splitting on a bare '---' line
# --------------------------------------------------------------------------

def test_prepare_outbound_splits_on_triple_dash():
    message = (
        "Testai žali, gali tęsti.\n"
        "---\n"
        "```python\n"
        "def f(): pass\n"
        "```\n"
        "Pakeisti failai: main.py"
    )
    full, spoken = prepare_outbound(message)
    assert full == message  # full text is the WHOLE message, unchanged
    assert spoken == "Testai žali, gali tęsti."
    assert "def f" not in spoken
    assert "main.py" not in spoken


def test_prepare_outbound_no_separator_uses_whole_message():
    message = "Trumpas atnaujinimas be techninės dalies."
    full, spoken = prepare_outbound(message)
    assert full == message
    assert spoken == "Trumpas atnaujinimas be techninės dalies."


def test_prepare_outbound_only_exact_dash_line_splits():
    # An en-dash sentence or '----' (4 dashes) is NOT the separator;
    # only a line that is EXACTLY '---'.
    message = "Pirma dalis — su brūkšniu.\nAntra eilutė tame pačiame bloke."
    full, spoken = prepare_outbound(message)
    assert full == message
    assert "Pirma dalis" in spoken
    assert "Antra eilutė tame pačiame bloke" in spoken


def test_prepare_outbound_dash_with_surrounding_whitespace_splits():
    # A '---' line may have trailing/leading spaces; still the separator.
    message = "Spoken dalis.\n   ---   \nTechninė dalis su code()."
    full, spoken = prepare_outbound(message)
    assert full == message
    assert spoken == "Spoken dalis."
    assert "Techninė" not in spoken


def test_prepare_outbound_splits_on_first_separator_only():
    message = "Antraštė.\n---\nVidurys.\n---\nGalas."
    full, spoken = prepare_outbound(message)
    assert full == message
    assert spoken == "Antraštė."
    assert "Vidurys" not in spoken
