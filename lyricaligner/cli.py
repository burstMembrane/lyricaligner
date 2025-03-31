import argparse
import logging
from pathlib import Path
from typing import List

from lyricaligner.lyricsaligner import LyricsAligner

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-a",
        "--audio",
        type=Path,
        required=True,
        help="Path to the audio file to be aligned",
    )
    parser.add_argument(
        "-t",
        "--text",
        type=Path,
        required=True,
        help="Path to the transcription text file to be aligned",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=".",
        help="Path to the output directory",
    )
    parser.add_argument(
        "-d",
        "--device",
        type=str,
        default="cpu",
        help="Device to run the model on",
    )
    parser.add_argument(
        "-b",
        "--batch_size",
        type=int,
        default=2,
        help="Batch size for ASR model",
    )
    parser.add_argument(
        "-m",
        "--model_id",
        type=str,
        default="facebook/wav2vec2-large-960h-lv60-self",
        help="The huggingface model id to use. Must be compatible with the Wav2Vec2Processor",
    )
    parser.add_argument(
        "-f",
        "--output_formats",
        type=List[str],
        default=["lrc", "srt", "json", "csv"],
        help="The formats to save the output in: default is lrc, srt, json, csv",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if not all([args.audio.exists(), args.text.exists()]):
        raise FileNotFoundError(f"File not found: {args.audio} or {args.text}")
    logger.info(f"Aligning audio: {args.audio} with text: {args.text}")
    aligner = LyricsAligner(
        device=args.device, batch_size=args.batch_size, model_id=args.model_id
    )
    words = aligner.sync(
        audio_fn=args.audio, text_fn=args.text, output_dir=args.output, save=True
    )
    logger.info(f"Alignment complete. Saved to {Path(args.output).resolve()}")
    logger.info(words)


if __name__ == "__main__":
    main()
