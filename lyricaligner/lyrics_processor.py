from dataclasses import dataclass
from itertools import groupby

from lyricaligner.alignment import Point
from lyricaligner.config import SEPARATOR
from lyricaligner.formatters import Word
from lyricaligner.utils import read_text


@dataclass
class CharacterSegment:
    character: str
    start: int
    end: int


class LyricsProcessor:
    def __init__(self, separator: str = SEPARATOR, is_upper=True) -> None:
        self.separator = separator
        self.is_upper = is_upper

    def process(self, text_path):
        """Process lyrics text file for alignment"""
        text = read_text(text_path)
        return self._preprocess_text(text)

    def _preprocess_text(self, text: str, is_upper=True):
        """Preprocess text for alignment by converting to lowercase and adding separators"""
        if is_upper:
            text = text.upper()
        else:
            text = text.lower()
        text = text.replace(" ", self.separator)
        text = text.replace("\n", self.separator)
        text = text.replace("'", "'")
        return text

    def get_words_from_path(
        self, source_text: str, alignment_path: list[Point], frame_duration: float
    ):
        """Extract words from alignment path"""
        # Skip repeating characters
        character_segments = []
        for i in range(len(alignment_path)):
            if (
                i == 0
                or alignment_path[i].token_index != alignment_path[i - 1].token_index
            ):
                # Start of a new character segment
                start_idx = i
            if (
                i == len(alignment_path) - 1
                or alignment_path[i].token_index != alignment_path[i + 1].token_index
            ):
                # End of current character segment
                character_segments.append(
                    CharacterSegment(
                        character=source_text[alignment_path[i].token_index],
                        start=alignment_path[start_idx].time_index,
                        end=alignment_path[i].time_index + 1,
                    )
                )

        return self._merge_segments_to_words(character_segments, frame_duration)

    def to_word(self, character_segments: list[CharacterSegment]):
        return "".join(seg.character for seg in character_segments)

    def _merge_segments_to_words(
        self, character_segments: list[CharacterSegment], frame_duration: float
    ):
        """Merge character segments into words"""
        words = []
        for is_separator, group in groupby(
            character_segments, key=lambda seg: seg.character == self.separator
        ):
            if not is_separator:
                segments = list(group)
                word = self.to_word(segments)
                word_start = segments[0].start * frame_duration
                word_end = segments[-1].end * frame_duration
                words.append(
                    Word(
                        text=word,
                        start=word_start,
                        end=word_end,
                    )
                )

        return words


if __name__ == "__main__":
    lp = LyricsProcessor()
    text = "Hello|world|"
    path = [Point(0, 0, 0), Point(1, 1, 0), Point(2, 2, 0)]
    print(lp.get_words_from_path(text, path, 1))
