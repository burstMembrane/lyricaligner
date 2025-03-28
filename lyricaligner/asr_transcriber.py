import logging

import torch
from torch import autocast
from tqdm import tqdm
from transformers import (
    Wav2Vec2ForCTC,
    Wav2Vec2Processor,
)
from transformers import (
    logging as transformers_logging,
)

from lyricaligner.config import DEFAULT_MODEL, TARGET_SR
from lyricaligner.lyrics_processor import LyricsProcessor
from lyricaligner.utils import get_audio_segments

# Suppress transformer warnings
transformers_logging.set_verbosity_error()

logger = logging.getLogger(__name__)


class ASRTranscriber:
    def __init__(self, model_id=None, device="cpu", batch_size=2) -> None:
        """Initialize the ASR Transcriber with a Wav2Vec2 model"""
        self.model_id = model_id or DEFAULT_MODEL
        self.device = device
        self.batch_size = batch_size
        self.processor = Wav2Vec2Processor.from_pretrained(
            self.model_id, device=self.device
        )
        self.model = Wav2Vec2ForCTC.from_pretrained(self.model_id)
        self.model.to(self.device)
        logger.info(f"Loaded model {self.model_id} on device {self.device}")
        self.lp = LyricsProcessor()
        self.blank_token_id = self.model.config.pad_token_id
        logger.info(f"Blank token ID: {self.blank_token_id}")

    @torch.inference_mode()
    def transcribe(self, audio):
        """Transcribe audio into text with timing information"""
        # Segment the audio into overlapping windows
        with autocast(device_type=self.device, dtype=torch.bfloat16):
            segs = get_audio_segments(audio)
            logger.info(f"Segmenting audio into {len(segs)} segments")

            logits = []
            for i in tqdm(
                range(0, len(segs), self.batch_size), desc="Processing batches"
            ):
                batch_segs = segs[i : i + self.batch_size]
                logits_batch = self._process_segment(batch_segs)
                # Extend list with individual sequences from batch
                for j in range(logits_batch.shape[0]):
                    logits.append(logits_batch[j : j + 1])
            logits = torch.cat(logits, dim=1)
            print(logits.shape)
            # Calculate log probabilities for CTC decoder
            emission = torch.log_softmax(logits, dim=-1)[0].cpu().detach()

            # Get predicted token ids and transcription
            pred = torch.argmax(logits, dim=-1)
            transcription = self.processor.batch_decode(pred)

            # Calculate frame duration for timing
            frame_duration = self.model.config.inputs_to_logits_ratio / TARGET_SR

            return emission, pred, transcription, frame_duration

    def tokenize(self, text_path):
        """Tokenize text into input IDs for the model"""
        text = self.lp.process(text_path)
        return self.processor.tokenizer(text).input_ids

    def _process_segment(self, audio_segments):
        """Process a batch of audio segments through the ASR model"""
        inputs = self.processor(
            audio_segments,
            return_tensors="pt",
            padding="longest",
            sampling_rate=TARGET_SR,
        ).input_values.to(self.device)

        logits = self.model(inputs).logits

        return logits.cpu().detach()

    def get_labels(self):
        """Get sorted token labels used by the tokenizer"""
        return [
            k
            for k, v in sorted(
                self.processor.tokenizer.get_vocab().items(), key=lambda x: x[1]
            )
        ]
