import pytest

from lyricaligner.formatters import Word, WordList


@pytest.fixture
def word_list():
    return [
        {"start": 0, "end": 1, "text": "hello"},
        {"start": 1, "end": 2, "text": "world"},
    ]


def lyrics():
    return """When the truth is found to be lies"""


def lyrics_wordlist():
    return WordList.from_list(
        [
            Word(text="When", start=0.04, end=0.16),
            Word(text="the", start=0.16, end=0.82),
            Word(text="truth", start=0.82, end=1.29),
            Word(text="is", start=1.29, end=1.63),
            Word(text="found", start=1.63, end=3.09),
            Word(text="to", start=3.09, end=3.37),
            Word(text="be", start=3.37, end=5.92),
            Word(text="lies", start=5.92, end=6.47),
        ]
    )


def test_word_list(word_list):
    word_list = WordList.from_list(word_list)
    assert len(word_list) == 2
    assert word_list[0].start == 0


def test_word_list_lrc(word_list):
    word_list = WordList.from_list(word_list)
    lrc = word_list.as_lrc_full("Hello world")
    assert lrc == """[00:00.00] <00:00.000> Hello <00:01.000> world""", lrc
