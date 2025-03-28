"""Utility functions for audio processing and file operations"""

import logging
from pathlib import Path

import librosa
import numpy as np

from lyricaligner.config import TARGET_SR, WINDOW_LENGTH
from lyricaligner.formatters import WordList

logger = logging.getLogger(__name__)

# Audio segmentation parameters
window_size = int(TARGET_SR * WINDOW_LENGTH)  # 15 seconds of audio at TARGET_SR
hop_length = window_size


def get_audio_segments(audio, window_size=window_size, hop_length=hop_length):
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
    onset_times = librosa.onset.onset_detect(y=audio, sr=TARGET_SR, backtrack=True)
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


def save_srt(srt: str, output_dir: Path, name: str):
    """Save SRT format lyrics to a file

    Args:
        srt: SRT format string
        name: Base name for the output file
    """
    logger.info(f"Saving SRT to {output_dir}/{name}.srt")

    output_dir.mkdir(parents=True, exist_ok=True)

    with open(f"{output_dir}/{name}.srt", "w+") as fp:
        fp.write(srt)


def save_lrc(lrc: str, output_dir: Path, name: str):
    """Save LRC format lyrics to a file

    Args:
        lrc: LRC format string
        name: Base name for the output file
    """
    logger.info(f"Saving LRC to {output_dir}/{name}.lrc")

    output_dir.mkdir(parents=True, exist_ok=True)

    with open(f"{output_dir}/{name}.lrc", "w+") as fp:
        fp.write(lrc)


def save_csv(words: WordList, output_dir: Path, name: str):
    """Save word timing information to a CSV file

    Args:
        words: WordList object
        name: Base name for the output file
    """
    df = words.to_df()
    logger.info(f"Saving CSV to {output_dir}/{name}.csv")
    output_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(f"{output_dir}/{name}.csv", index=False)
