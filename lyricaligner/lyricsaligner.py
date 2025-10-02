"""Lyrics Synchronization library for aligning lyrics with audio"""

import logging
from pathlib import Path

import numpy as np
import soundfile as sf
import soxr
from line_profiler import profile

from lyricaligner.alignment import ForcedAligner
from lyricaligner.asr_transcriber import ASRTranscriber
from lyricaligner.config import DEFAULT_BLANK_TOKEN_ID, DEFAULT_MODEL, TARGET_SR
from lyricaligner.formatters import WordList
from lyricaligner.lyrics_processor import LyricsProcessor
from lyricaligner.utils import read_text, save_csv, save_json, save_lrc, save_srt

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


@profile
class LyricsAligner:
    """Main class for aligning lyrics with audio"""

    def __init__(
        self,
        model_id=DEFAULT_MODEL,
        device="cpu",
        blank_id=DEFAULT_BLANK_TOKEN_ID,
        batch_size=2,
    ) -> None:
        """Initialize the LyricsAligner

        Args:
            model_id: ASR model ID to use (defaults to facebook/wav2vec2-large-960h-lv60-self)
            blank_id: ID of the blank token in the ASR model
        """
        # the transcriber module
        self.asr = ASRTranscriber(
            model_id=model_id, device=device, batch_size=batch_size
        )
        self.lp = LyricsProcessor()
        self.blank_id = blank_id

    def load_audio(self, audio_fn: str, normalize=True):
        """Load input audio file and prepare for processing

        Args:
            audio_fn: Path to the audio file

        Returns:
            Normalized audio at TARGET_SR sample rate
        """
        # Load audio
        audio, sr = sf.read(audio_fn, dtype=np.float32)

        # Convert to mono if stereo
        if len(audio.shape) > 1:
            audio = np.mean(audio, axis=1)

        # Normalize audio
        if normalize:
            audio = audio / np.max(np.abs(audio))

        # Resample audio if needed
        if sr != TARGET_SR:
            logger.info(f"Resampling audio from {sr}Hz to {TARGET_SR}Hz")
            audio = soxr.resample(audio, in_rate=sr, out_rate=TARGET_SR)

        logger.info(f"Loaded audio from {audio_fn} with shape {audio.shape}")
        return audio

    def _align(self, emission, tokens, processed_lyrics, frame_duration):
        """Internal helper function for alignment logic

        Args:
            emission: ASR emission probabilities
            tokens: Tokenized lyrics
            processed_lyrics: Processed lyrics text
            frame_duration: Duration of each frame

        Returns:
            WordList object with aligned words
        """
        # Align audio with lyrics
        path = ForcedAligner.align(
            emission_log_probs=emission,
            token_ids=tokens,
            blank_token_id=self.blank_id,
        )
        words = self.lp.get_words_from_path(processed_lyrics, path, frame_duration)
        return WordList.from_list(words)

    def _save_results(self, words, lyrics_text, output_dir, output_name):
        """Save alignment results to files

        Args:
            words: WordList object
            lyrics_text: Original lyrics text
            output_dir: Output directory
            output_name: Base name for output files
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        save_csv(words, output_dir, output_name)
        save_lrc(words, output_dir, output_name, lyrics_text)
        # save the word level srt
        save_srt(words, output_dir, f"{output_name}.word")
        # save the line level srt
        save_srt(words, output_dir, output_name, lyrics_text)
        # save as JSON (TODO: add phrase level timestamps with lyrics text)
        save_json(words, output_dir, output_name)

    def sync(self, audio_fn: str, text_fn: str, output_dir: str = "output", save=False):
        """Synchronize lyrics with audio

        Args:
            audio_fn: Path to the audio file
            text_fn: Path to the lyrics text file
            output_dir: Directory to save output files to
            save: Whether to save the results to files

        Returns:
            WordList object with aligned words
        """
        # Extract base filename
        audio_name = Path(audio_fn).stem
        output_dir = Path(output_dir)
        # create output directory if it doesn't exist
        output_dir.mkdir(parents=True, exist_ok=True)

        # Load and process audio
        vocals = self.load_audio(audio_fn)
        emission, _, _, frame_duration = self.asr.transcribe(vocals)

        # Process and tokenize lyrics
        tokens = self.asr.tokenize(text_fn)
        processed_lyrics = self.lp.process(text_fn)

        # Align audio with lyrics
        words = self._align(emission, tokens, processed_lyrics, frame_duration)

        # Save results if requested
        if save:
            original_lyrics = read_text(text_fn)
            self._save_results(words, original_lyrics, output_dir, audio_name)

        return words

    def sync_from_array(
        self,
        audio_array: np.ndarray,
        lyrics_text: str,
        output_dir: str = "output",
        output_name: str = "output",
        save=True,
    ):
        """Synchronize lyrics with audio directly from audio array and lyrics string

        Args:
            audio_array: Audio array (should be mono and at 16000 Hz for compatibility with ASR model)
            lyrics_text: Raw lyrics text as a string
            output_dir: Directory to save output files to
            output_name: Base name for output files if saving
            save: Whether to save the results to files

        Returns:
            WordList object with aligned words
        """
        # Normalize audio if needed
        if np.max(np.abs(audio_array)) > 1.0:
            audio_array = audio_array / np.max(np.abs(audio_array))

        # Process audio to get emission
        emission, _, _, frame_duration = self.asr.transcribe(audio_array)

        # Preprocess lyrics
        processed_lyrics = self.lp._preprocess_text(lyrics_text)

        # Tokenize the processed lyrics
        tokens = self.asr.processor.tokenizer(processed_lyrics).input_ids

        # Align audio with lyrics
        words = self._align(emission, tokens, processed_lyrics, frame_duration)

        # Save results if requested
        if save:
            self._save_results(words, lyrics_text, output_dir, output_name)

        return words
