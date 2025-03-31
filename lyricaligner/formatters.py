import json
import logging
import re
from dataclasses import asdict, dataclass
from typing import List

import pandas as pd

logger = logging.getLogger(__name__)


def to_lrc_time(seconds: float) -> str:
    """Convert seconds to LRC format time (MM:SS.mmm)"""
    minutes = int(seconds // 60)
    seconds_remainder = seconds % 60
    seconds_int = int(seconds_remainder)
    milliseconds = round((seconds_remainder - seconds_int) * 1000)
    return f"{minutes:02d}:{seconds_int:02d}.{milliseconds:03d}"


def to_srt_time(seconds: float) -> str:
    """Convert seconds to SRT format time (HH:MM:SS,mmm)"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    seconds_remainder = seconds % 60
    seconds_int = int(seconds_remainder)
    milliseconds = round((seconds_remainder - seconds_int) * 1000)
    return f"{hours:02d}:{minutes:02d}:{seconds_int:02d},{milliseconds:03d}"


@dataclass
class Word:
    text: str
    start: float
    end: float

    def as_lrc_tag(self, is_word: bool = True) -> str:
        tag = to_lrc_time(self.start)
        return f"<{tag}> {self.text}" if is_word else f"[{tag}]"

    def as_srt_tag(self) -> str:
        return f"{to_srt_time(self.start)} --> {to_srt_time(self.end)} {self.text}"

    def __repr__(self):
        return f"[{to_lrc_time(self.start)} --> {to_lrc_time(self.end)}] {self.text}\n"

    def as_str(self):
        return f"[{to_lrc_time(self.start)} --> {to_lrc_time(self.end)}] {self.text}\n"


@dataclass
class WordList:
    words: List[Word]

    @classmethod
    def from_list(cls, word_list: List[dict]) -> "WordList":
        """Create a WordList from a list of dictionaries or Word objects"""
        return cls(
            words=[
                Word(**asdict(item)) if isinstance(item, Word) else Word(**item)
                for item in word_list
            ]
        )

    def __len__(self):
        return len(self.words)

    def __repr__(self):
        return "".join(word.as_str() for word in self.words)

    def as_srt_full(self, original_lyrics: str) -> str:
        """Format as SRT with full line text and line-level timings.
        Falls back to punctuation-based splitting if no newlines are found.
        """
        srt = ""
        counter = 1
        word_index = 0

        if "\n" in original_lyrics:
            segments = [
                line.strip() for line in original_lyrics.splitlines() if line.strip()
            ]
        else:
            logger.warning(
                "No newlines found in original lyrics, using punctuation-based splitting"
            )
            segments = re.split(r"(?<=[.!?])\s+", original_lyrics.strip())
            segments = [seg.strip() for seg in segments if seg]

        for segment in segments:
            words_in_segment = segment.split()
            num_words = len(words_in_segment)

            if word_index + num_words > len(self.words):
                raise IndexError("Not enough words in word list to match the lyrics.")

            start_time = self.words[word_index].start
            end_time = self.words[word_index + num_words - 1].end

            srt += f"{counter}\n"
            srt += f"{to_srt_time(start_time)} --> {to_srt_time(end_time)}\n"
            srt += f"{segment}\n\n"

            word_index += num_words
            counter += 1

        return srt

    def as_lrc_tags_only(self) -> str:
        """Return LRC tags without line timing"""
        return "".join(word.as_lrc_tag() for word in self.words)

    def as_lrc_full(self, original_lyrics: str) -> str:
        """Format as complete LRC with line and word timings.
        Falls back to punctuation-based splitting if no newlines are found.
        """
        lrc = ""
        counter = 0

        if "\n" in original_lyrics:
            segments = [
                line.strip() for line in original_lyrics.splitlines() if line.strip()
            ]
        else:
            logger.warning(
                "No newlines found in original lyrics, using punctuation-based splitting"
            )
            segments = re.split(r"(?<=[.!?])\s+", original_lyrics.strip())
            segments = [seg.strip() for seg in segments if seg]

        for segment in segments:
            words_in_segment = segment.split()
            num_words = len(words_in_segment)

            if counter + num_words > len(self.words):
                raise IndexError("Not enough words in word list to match the lyrics.")

            start_time = self.words[counter].start
            lrc += f"[{to_lrc_time(start_time)}]"

            for word_text in words_in_segment:
                word = self.words[counter]
                word.text = word_text
                lrc += f" {word.as_lrc_tag()}"
                counter += 1

            lrc += "\n"

        return lrc

    def as_srt(self) -> str:
        """Format as SRT subtitle format"""
        srt = ""
        for i, word in enumerate(self.words):
            srt += f"{i + 1}\n{to_srt_time(word.start)} --> {to_srt_time(word.end)}\n{word.text}\n\n"
        return srt

    def as_json(self) -> str:
        """Format as JSON"""
        return json.dumps([asdict(word) for word in self.words], indent=4)

    def as_df(self, with_length: bool = True) -> pd.DataFrame:
        """Format as CSV"""
        df = pd.DataFrame([asdict(word) for word in self.words])
        if with_length:
            df["length"] = df["end"] - df["start"]
        return df

    def __iter__(self):
        return iter(self.words)

    def __getitem__(self, index):
        return self.words[index]


@dataclass
class Phrase:
    """Represents a phrase (line) of text with start and end times"""

    text: str
    start: float
    end: float
    words: List[Word]

    def __repr__(self):
        return f"[{to_lrc_time(self.start)} --> {to_lrc_time(self.end)}] {self.text}\n"

    def as_str(self):
        return f"[{to_lrc_time(self.start)} --> {to_lrc_time(self.end)}] {self.text}\n"

    def as_lrc_line(self) -> str:
        """Format as LRC line with timestamp"""
        return f"[{to_lrc_time(self.start)}]{self.text}\n"

    def as_lrc_line_with_word_tags(self) -> str:
        """Format as LRC line with both line timestamp and word timestamps"""
        line = f"[{to_lrc_time(self.start)}]"
        for word in self.words:
            line += f" {word.as_lrc_tag()}"
        return line + "\n"

    def as_srt_entry(self, index: int) -> str:
        """Format as an SRT entry"""
        return f"{index}\n{to_srt_time(self.start)} --> {to_srt_time(self.end)}\n{self.text}\n\n"


@dataclass
class PhraseList:
    """Collection of phrases with methods to create and format phrase-level alignments"""

    phrases: List[Phrase]

    def __len__(self):
        return len(self.phrases)

    def __iter__(self):
        return iter(self.phrases)

    def __getitem__(self, index):
        return self.phrases[index]

    def __repr__(self):
        return "".join(phrase.as_str() for phrase in self.phrases)

    @classmethod
    def from_wordlist(cls, word_list: WordList, original_lyrics: str) -> "PhraseList":
        """Create a PhraseList from a WordList and original lyrics text

        Args:
            word_list: The word-level alignments
            original_lyrics: Original text with phrase-level structure

        Returns:
            PhraseList with phrase-level alignments
        """
        phrases = []
        word_index = 0

        # Try to split by newlines first, otherwise use punctuation
        if "\n" in original_lyrics:
            segments = [
                line.strip() for line in original_lyrics.splitlines() if line.strip()
            ]
        else:
            logger.warning(
                "No newlines found in original lyrics, using punctuation-based splitting"
            )
            segments = re.split(r"(?<=[.!?])\s+", original_lyrics.strip())
            segments = [seg.strip() for seg in segments if seg]

        for segment in segments:
            words_in_segment = segment.split()
            num_words = len(words_in_segment)

            if word_index + num_words > len(word_list):
                raise IndexError("Not enough words in word list to match the lyrics.")

            # Get the words for this phrase
            phrase_words = word_list.words[word_index : word_index + num_words]

            # Create a phrase with start time from the first word and end time from the last word
            phrase = Phrase(
                text=segment,
                start=phrase_words[0].start,
                end=phrase_words[-1].end,
                words=phrase_words,
            )

            phrases.append(phrase)
            word_index += num_words

        return cls(phrases=phrases)

    def as_lrc(self, include_word_tags: bool = True) -> str:
        """Format as LRC with optional word-level tags

        Args:
            include_word_tags: Whether to include word-level timestamps

        Returns:
            String in LRC format
        """
        if include_word_tags:
            return "".join(
                phrase.as_lrc_line_with_word_tags() for phrase in self.phrases
            )
        else:
            return "".join(phrase.as_lrc_line() for phrase in self.phrases)

    def as_srt(self) -> str:
        """Format as SRT subtitle format

        Returns:
            String in SRT format
        """
        return "".join(
            phrase.as_srt_entry(i + 1) for i, phrase in enumerate(self.phrases)
        )

    def as_json(self) -> str:
        """Format as JSON

        Returns:
            JSON string with phrase data
        """
        phrases_data = []
        for phrase in self.phrases:
            phrase_dict = {
                "text": phrase.text,
                "start": phrase.start,
                "end": phrase.end,
                "words": [asdict(word) for word in phrase.words],
            }
            phrases_data.append(phrase_dict)

        return json.dumps(phrases_data, indent=4)

    def as_df(self) -> pd.DataFrame:
        """Format as DataFrame

        Returns:
            Pandas DataFrame with phrase data
        """
        phrases_data = []
        for phrase in self.phrases:
            phrase_dict = {
                "text": phrase.text,
                "start": phrase.start,
                "end": phrase.end,
                "duration": phrase.end - phrase.start,
                "word_count": len(phrase.words),
            }
            phrases_data.append(phrase_dict)

        return pd.DataFrame(phrases_data)


if __name__ == "__main__":
    words = [
        Word(text="When", start=0.04, end=0.16),
        Word(text="the", start=0.16, end=0.82),
        Word(text="truth", start=0.82, end=1.29),
        Word(text="is", start=1.29, end=1.63),
        Word(text="found", start=1.63, end=3.09),
        Word(text="to", start=3.09, end=3.37),
        Word(text="be", start=3.37, end=5.92),
        Word(text="lies", start=5.92, end=6.47),
        Word(text="And", start=7.67, end=7.94),
        Word(text="all", start=7.94, end=8.36),
        Word(text="the", start=8.36, end=8.63),
        Word(text="joy", start=8.63, end=10.28),
        Word(text="within", start=10.28, end=10.53),
        Word(text="you", start=10.53, end=13.09),
        Word(text="dies", start=13.09, end=13.34),
        Word(text="Don't", start=14.32, end=14.73),
        Word(text="you", start=14.73, end=15.14),
        Word(text="want", start=15.14, end=15.57),
        Word(text="somebody", start=15.57, end=16.09),
        Word(text="to", start=16.09, end=16.46),
        Word(text="love", start=16.46, end=17.0),
    ]
    original_lyrics = """When the truth is found to be lies
And all the joy within you dies
Don't you want somebody to love"""

    word_list = WordList.from_list(words)
    phrase_list = PhraseList.from_wordlist(word_list, original_lyrics)

    print("WordList representation:")
    print(word_list)

    print("\nPhraseList representation:")
    print(phrase_list)

    print("\nLRC format with word tags:")
    print(phrase_list.as_lrc(include_word_tags=True))

    print("\nLRC format without word tags:")
    print(phrase_list.as_lrc(include_word_tags=False))

    print("\nSRT format:")
    print(phrase_list.as_srt())

    print("\nJSON format:")
    print(phrase_list.as_json())
