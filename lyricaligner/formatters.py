import json
from dataclasses import asdict, dataclass
from typing import List

import pandas as pd


def to_lrc_time(seconds: float) -> str:
    """Convert seconds to LRC format time (MM:SS.mmm)"""
    minutes = int(seconds // 60)
    seconds = seconds % 60
    milliseconds = int((seconds % 1) * 1000)
    seconds = int(seconds)
    return f"{minutes:02d}:{seconds:02d}.{milliseconds:03d}"


def to_srt_time(seconds: float) -> str:
    """Convert seconds to SRT format time (HH:MM:SS,mmm)"""
    hours = int(seconds / 3600)
    minutes = int((seconds % 3600) / 60)
    seconds = int(seconds % 60)
    milliseconds = int((seconds % 1) * 1000)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d},{milliseconds:03d}"


@dataclass
class Word:
    text: str
    start: float
    end: float

    def as_lrc_tag(self, is_word: bool = True) -> str:
        tag = to_lrc_time(self.start)
        return f"<{tag}> {self.text}" if is_word else f"[{tag}]"

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

    def as_lrc_tags_only(self) -> str:
        """Return LRC tags without line timing"""
        return "".join(word.as_lrc_tag() for word in self.words)

    def as_lrc_full(self, original_lyrics: str) -> str:
        """Format as complete LRC with line and word timings"""
        lrc = ""
        counter = 0
        word_end = None

        for line in original_lyrics.splitlines():
            if not line.strip():
                continue

            if word_end:
                lrc += f"\n[{to_lrc_time(word_end)}]"
            else:
                lrc += "[00:00.00]"

            for word_text in line.split():
                if not word_text:
                    continue

                word = self.words[counter]
                word.text = word_text
                lrc += f" {word.as_lrc_tag()}"
                word_end = word.end
                counter += 1

        return lrc

    def to_srt(self) -> str:
        """Format as SRT subtitle format"""
        srt = ""
        for i, word in enumerate(self.words):
            srt += f"{i + 1}\n{to_srt_time(word.start)} --> {to_srt_time(word.end)}\n{word.text}\n\n"
        return srt

    def to_json(self) -> str:
        """Format as JSON"""
        return json.dumps([asdict(word) for word in self.words], indent=4)

    def to_df(self) -> pd.DataFrame:
        """Format as CSV"""
        df = pd.DataFrame([asdict(word) for word in self.words])
        df["length"] = df["end"] - df["start"]
        return df

    def __iter__(self):
        return iter(self.words)

    def __getitem__(self, index):
        return self.words[index]


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
    print("LRC format:")
    print(word_list.as_lrc_full(original_lyrics))
    print("\nSRT format:")
    print(word_list.to_srt())
    print("\nJSON format:")
    print(word_list.to_json())
