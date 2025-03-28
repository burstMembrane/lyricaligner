import argparse
import logging
from pathlib import Path

from lyricaligner.lyricsaligner import LyricsAligner

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--audio", type=Path, required=True, help="Path to the audio file to be aligned"
    )
    parser.add_argument(
        "--text",
        type=Path,
        required=True,
        help="Path to the transcription text file to be aligned",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=".",
        help="Path to the output directory",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if not all([args.audio.exists(), args.text.exists()]):
        raise FileNotFoundError(f"File not found: {args.audio} or {args.text}")
    logger.info(f"Aligning audio: {args.audio} with text: {args.text}")
    aligner = LyricsAligner()
    words = aligner.sync(
        audio_fn=args.audio, text_fn=args.text, output_dir=args.output, save=True
    )
    logger.info(f"Alignment complete. Saved to {args.output}")
    logger.info(words)


if __name__ == "__main__":
    main()
