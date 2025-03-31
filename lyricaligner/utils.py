"""Utility functions for audio processing and file operations"""

import logging
from pathlib import Path

import librosa
import numpy as np

from lyricaligner.config import TARGET_SR, WINDOW_LENGTH
from lyricaligner.formatters import WordList

logger = logging.getLogger(__name__)

# Audio segmentation parameters
# 15 seconds of audio at TARGET_SR
window_size = int(TARGET_SR * WINDOW_LENGTH)
hop_length = window_size


def get_audio_segments(
    audio: np.ndarray, window_size=window_size, hop_length=hop_length
):
    """Split audio into fixed-length segments for processing

    Args:
        audio: Audio array to segment

    Returns:
        List of audio segments
    """
    # Handle short audio files
    if len(audio) < window_size:
        return [audio]

    return librosa.util.frame(
        audio, frame_length=window_size, hop_length=hop_length, axis=0
    )


def get_audio_segments_by_onsets(audio):
    onset_times = librosa.onset.onset_detect(y=audio,
                                             sr=TARGET_SR,
                                             backtrack=True)
    onset_boundaries = np.concatenate([onset_times, [len(audio)]])
    segments = []
    start_onset = 0
    for onset in onset_boundaries:
        segments.append(audio[start_onset:onset])
    return segments


def read_text(text_path):
    """Read text from a file

    Args:
        text_path: Path to the text file

    Returns:
        Content of the text file as string
    """
    with open(text_path, "r") as file:
        return file.read()


def save_srt(words: WordList, output_dir: Path, name: str, original_lyrics: str = None):
    """Save alignment as SRT subtitle file

    Args:
        words: WordList object
        output_dir: Output directory
        name: Base name for the output file
        original_lyrics: Optional original lyrics text for line-level SRT
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    if original_lyrics is not None and ".word" not in name:
        # Line-level SRT
        srt = words.as_srt_full(original_lyrics)
    else:
        # Word-level SRT
        srt = words.as_srt()

    logger.info(f"Saving SRT to {output_dir}/{name}.srt")
    with open(f"{output_dir}/{name}.srt", "w", encoding="utf-8") as f:
        f.write(srt)


def save_json(words: WordList, output_dir: Path, name: str):
    """Save alignment as JSON file

    Args:
        words: WordList object
        output_dir: Output directory
        name: Base name for the output file
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    json_data = words.as_json()

    logger.info(f"Saving JSON to {output_dir}/{name}.json")
    with open(f"{output_dir}/{name}.json", "w", encoding="utf-8") as f:
        f.write(json_data)


def save_lrc(words: WordList, output_dir: Path, name: str, original_lyrics: str = None):
    """Save alignment as LRC lyrics file

    Args:
        words: WordList object
        output_dir: Output directory
        name: Base name for the output file
        original_lyrics: Optional original lyrics text for full LRC
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    if original_lyrics is not None:
        # Full LRC with line and word timings
        lrc = words.as_lrc_full(original_lyrics)
    else:
        # LRC with word timings only
        lrc = words.as_lrc_tags_only()

    logger.info(f"Saving LRC to {output_dir}/{name}.lrc")
    with open(f"{output_dir}/{name}.lrc", "w", encoding="utf-8") as f:
        f.write(lrc)


def save_csv(words: WordList, output_dir: Path, name: str, lyrics_text: str):
    """Save word timing information to a CSV file

    Args:
        words: WordList object
        name: Base name for the output file
    """
    df = words.as_df()
    logger.info(f"Saving CSV to {output_dir}/{name}.csv")
    output_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(f"{output_dir}/{name}.csv", index=False)
