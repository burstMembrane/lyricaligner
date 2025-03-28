# LyricAligner

A Python tool for synchronizing lyrics with audio files by using a Wav2Vec2 model to auto-align text with word-level timestamps.

Forked & refactored from [lyrics-sync](https://github.com/mikezzb/lyrics-sync)

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
    audio_path = "./path/to/audio"
    transcript_path = "./path/to/transcript"
    output_path = "./output"
    aligner = LyricsAligner()
    # pass save to save output files e.g CSV, LRC
    words = aligner.sync(
        audio_fn=audio_path, text_fn=transcript_path, output_dir=output_path, save=True
    )
    print(words)

# Method 2: Align using audio array and text string
    audio, sr = librosa.load(audio_path, sr=None)
    # resample to 16000
    audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)
    with open(transcript_path, "r") as f:
        text = f.read()
    # we need to pass in the output name
    output_name = Path(audio_path).stem
    words = aligner.sync_from_array(
        audio_array=audio,
        lyrics_text=text,
        output_dir=output_path,
        output_name=output_name,
        save=True,
    )
    print()
    print(words)

# The words object contains timing information for each word
```

## Project Structure

The package includes modules for:

- ASR transcription
- Audio processing
- Alignment algorithms
- Export formatting
