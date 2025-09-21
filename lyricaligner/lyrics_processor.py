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
    confidence: float = 0.0


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
                or alignment_path[i].token_position
                != alignment_path[i - 1].token_position
            ):
                # Start of a new character segment
                start_idx = i
            if (
                i == len(alignment_path) - 1
                or alignment_path[i].token_position
                != alignment_path[i + 1].token_position
            ):
                # Calculate average confidence for this character segment
                segment_confidence = sum(
                    alignment_path[j].confidence for j in range(start_idx, i + 1)
                ) / (i - start_idx + 1)
                
                character_segments.append(
                    CharacterSegment(
                        character=source_text[alignment_path[i].token_position],
                        start=alignment_path[start_idx].frame_index,
                        end=alignment_path[i].frame_index + 1,
                        confidence=segment_confidence,
                    )
                )

        return self._merge_segments_to_words(character_segments, frame_duration)

    def to_word(self, character_segments: list[CharacterSegment]):
        word = "".join(seg.character for seg in character_segments)
        # convert back to original case
        if self.is_upper:
            word = word.lower()
        return word

    def _merge_segments_to_words(
        self, character_segments: list[CharacterSegment], frame_duration: float
    ):
        """Merge character segments into words with timing corrections"""
        from lyricaligner.timing_utils import (
            detect_timing_drift, 
            smooth_timing_with_confidence,
            apply_timing_anchors
        )
        
        words = []
        for is_separator, group in groupby(
            character_segments, key=lambda seg: seg.character == self.separator
        ):
            if not is_separator:
                segments = list(group)
                word = self.to_word(segments)
                word_start = segments[0].start * frame_duration
                word_end = segments[-1].end * frame_duration
                
                # Calculate average confidence for the word
                avg_confidence = sum(getattr(seg, 'confidence', 0.5) for seg in segments) / len(segments)
                
                words.append(
                    Word(
                        text=word,
                        start=word_start,
                        end=word_end,
                        confidence=avg_confidence
                    )
                )

        # Apply timing corrections if we have enough words
        if len(words) > 5:
            # Check for timing drift
            if detect_timing_drift(words):
                # Apply smoothing and anchoring
                words = smooth_timing_with_confidence(words)
                words = apply_timing_anchors(words)

        return words


if __name__ == "__main__":
    lp = LyricsProcessor()
    text = "Hello|world|"
    path = [Point(0, 0, 0), Point(1, 1, 0), Point(2, 2, 0)]
    print(lp.get_words_from_path(text, path, 1))
