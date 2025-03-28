# LyricAligner

A Python tool for synchronizing lyrics with audio files by using a Wav2Vec2 model to auto-aligning text with word-level timestamps.

## Installation

```bash
pip install lyricaligner
```

## Requirements

- Python >= 3.12
- librosa >= 0.10.2
- numba >= 0.54
- pandas >= 2.2.3
- torch >= 2.6.0
- transformers >= 4.50.2

## Usage

### Command Line Interface

```bash
# Basic usage
lyricaligner --audio path/to/song.mp3 --text path/to/lyrics.txt --output path/to/output/dir

# Arguments:
#   --audio    Path to the audio file to be aligned (required)
#   --text     Path to the text file containing lyrics (required)
#   --output   Path to the output directory (default: current directory)
```

The tool will generate:

- CSV file with word-level timing information
- LRC file compatible with media players

### Using as a Python Module

```python
from lyricaligner import LyricsAligner

# Initialize the aligner
aligner = LyricsAligner()

# Method 1: Align using file paths
words, lrc = aligner.sync(
    audio_fn="path/to/song.mp3",
    text_fn="path/to/lyrics.txt", 
    output_dir="output"
)

# Method 2: Align using audio array and text string
import numpy as np

# Make sure audio is at the target sample rate
audio_array = np.array([...])  # Audio samples
lyrics = "These are the lyrics to align"

words, lrc = aligner.sync_from_array(
    audio_array=audio_array,
    lyrics_text=lyrics,
    output_name="alignment"
)

# The words object contains timing information for each word
# The lrc string contains the formatted LRC file content
```

## Project Structure

The package includes modules for:

- ASR transcription
- Audio processing
- Alignment algorithms
- Export formatting
