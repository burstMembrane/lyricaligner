from dataclasses import dataclass

from lyricaligner.alignment import Point
from lyricaligner.config import SEPARATOR
from lyricaligner.formatters import Word
from lyricaligner.utils import read_text


@dataclass
class Segment:
    word: str
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

    def get_words_from_path(self, text, path, frame_duration):
        """Extract words from alignment path"""
        # Skip repeating characters
        segments = []
        i1, i2 = 0, 0

        while i1 < len(path):
            while i2 < len(path) and path[i1].token_index == path[i2].token_index:
                i2 += 1
            segments.append(
                Segment(
                    text[path[i1].token_index],
                    path[i1].time_index,
                    path[i2 - 1].time_index + 1,
                )
            )
            i1 = i2

        return self._merge_segments(segments, frame_duration)

    def _merge_segments(self, segments, frame_duration):
        """Merge character segments into words"""
        words = []
        i1, i2 = 0, 0

        while i1 < len(segments):
            if i2 >= len(segments) or segments[i2].word == self.separator:
                if i1 != i2:
                    segs = segments[i1:i2]
                    word = "".join([seg.word for seg in segs])
                    start = segs[0].start * frame_duration
                    end = segs[-1].end * frame_duration
                    words.append(
                        Word(
                            text=word,
                            start=start,
                            end=end,
                        )
                    )
                i1 = i2 + 1
                i2 = i1
            else:
                i2 += 1

        return words


if __name__ == "__main__":
    lp = LyricsProcessor()
    text = "Hello|world|"
    path = [Point(0, 0, 0), Point(1, 1, 0), Point(2, 2, 0)]
    print(lp.get_words_from_path(text, path, 1))
