"""Both languages must say everything, with the same fill-in fields."""

import string

import pytest

from voice_bridge import i18n


def _fields(text):
    return {name for _, name, _, _ in string.Formatter().parse(text) if name}


def test_catalogs_have_the_same_keys():
    assert set(i18n.EN) == set(i18n.LT)


@pytest.mark.parametrize("key", sorted(i18n.EN))
def test_same_fields_in_both_languages(key):
    assert _fields(i18n.EN[key]) == _fields(i18n.LT[key]), key


def test_set_language_switches_and_rejects_unknown():
    i18n.set_language("en")
    assert i18n.t("usage.today") == "today"
    i18n.set_language("lt")
    assert i18n.t("usage.today") == "šiandien"
    with pytest.raises(ValueError):
        i18n.set_language("de")
