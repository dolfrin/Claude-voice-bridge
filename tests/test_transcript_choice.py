"""The user picks which model's transcript reaches Claude."""

from voice_bridge.bridge import _pick_transcript


class FakeTelegram:
    """Records ask_per_message calls and replays a canned answer."""

    def __init__(self, answer=""):
        self.answer = answer
        self.calls: list[tuple[str, list[tuple[str, str]]]] = []

    async def ask_per_message(self, project, options, button="Accept this"):
        self.calls.append((project, options))
        return self.answer


async def test_accepting_the_second_transcript_sends_that_one():
    telegram = FakeTelegram(answer="base-turbo")
    results = [
        {"model": "paprika-lt", "text": "autistinę žinutę"},
        {"model": "base-turbo", "text": "testinę žinutę"},
    ]

    assert await _pick_transcript(results, telegram) == "testinę žinutę"
    project, options = telegram.calls[0]
    assert project == "stt"
    # One message per model, each carrying its own full transcript.
    assert options == [
        ("paprika-lt", "autistinę žinutę"),
        ("base-turbo", "testinę žinutę"),
    ]


async def test_no_answer_falls_back_to_the_primary():
    telegram = FakeTelegram(answer="")  # what ask_per_message returns on timeout
    results = [
        {"model": "paprika-lt", "text": "pirmas"},
        {"model": "base-turbo", "text": "antras"},
    ]

    assert await _pick_transcript(results, telegram) == "pirmas"


async def test_single_model_is_not_worth_asking_about():
    telegram = FakeTelegram(answer="whatever")

    result = await _pick_transcript([{"model": "solo", "text": "vienas"}], telegram)

    assert result == "vienas"
    assert telegram.calls == []


async def test_blank_transcripts_never_become_choices():
    telegram = FakeTelegram(answer="")
    results = [
        {"model": "paprika-lt", "text": "   "},
        {"model": "base-turbo", "text": "tik šitas"},
    ]

    # Only one usable transcript left, so there is nothing to choose between.
    assert await _pick_transcript(results, telegram) == "tik šitas"
    assert telegram.calls == []


async def test_nothing_transcribed_yields_empty():
    assert await _pick_transcript([], FakeTelegram()) == ""


async def test_edit_aborts_the_choice_silently():
    telegram = FakeTelegram(answer=None)  # what ask_per_message returns on Edit
    results = [
        {"model": "paprika-lt", "text": "pirmas"},
        {"model": "base-turbo", "text": "antras"},
    ]

    assert await _pick_transcript(results, telegram) is None
