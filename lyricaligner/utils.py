"""Utility functions for audio processing and file operations"""

import csv
import logging
from dataclasses import asdict
from pathlib import Path

import numpy as np

from lyricaligner.config import TARGET_SR, WINDOW_LENGTH
from lyricaligner.formatters import WordList

logger = logging.getLogger(__name__)

# Audio segmentation parameters
# 15 seconds of audio at TARGET_SR
window_size = int(TARGET_SR * WINDOW_LENGTH)
hop_length = window_size


def get_audio_segments(
    audio: np.ndarray, window_size: int = window_size, hop_length: int = hop_length
):
    """Split audio into fixed-length segments for processing.

    Args:
        audio: 1D numpy array containing audio samples
        window_size: Size of each frame/segment
        hop_length: Step size between successive frames

    Returns:
        A 2D numpy array of shape (num_frames, window_size)
    """
    n_samples = len(audio)

    # Handle short audio files
    if n_samples < window_size:
        return audio[np.newaxis, :]  # return shape (1, N)

    # Calculate number of frames that fit
    num_frames = 1 + (n_samples - window_size) // hop_length

    # Compute strides (bytes to step for each axis)
    stride = audio.strides[0]
    return np.lib.stride_tricks.as_strided(
        audio,
        shape=(num_frames, window_size),
        strides=(hop_length * stride, stride),
        writeable=False,  # safer: returned view is read-only
    )


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


def save_csv(words, output_dir: Path, name: str):
    """Save word timing information to a CSV file

    Args:
        words: WordList object
        output_dir: Output directory Path
        name: Base name for the output file
    """
    # Convert to a list of dictionaries or tuples
    records = [asdict(word) for word in words.words]

    if not records:
        logger.warning("No words to save.")
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"{name}.csv"

    fieldnames = list(records[0].keys())
    logger.info(f"Saving CSV to {out_path}")
    with out_path.open(mode="w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)
