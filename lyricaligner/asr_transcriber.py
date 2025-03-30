import logging

import numpy as np
import torch
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

transformers_logging.set_verbosity_warning()

logger = logging.getLogger(__name__)


class ASRTranscriber:
    def __init__(
        self, model_id=None, device="cpu", batch_size=1, is_upper=True
    ) -> None:
        self.model_id = model_id or DEFAULT_MODEL
        self.device = device
        self.processor = Wav2Vec2Processor.from_pretrained(self.model_id)
        self.model = Wav2Vec2ForCTC.from_pretrained(self.model_id)
        is_upper = self.expects_uppercase()
        if is_upper:
            self.lp = LyricsProcessor(is_upper=True)
        else:
            self.lp = LyricsProcessor(is_upper=False)
        self.model.to(self.device)
        logger.info(f"Loaded model {self.model_id} on device {self.device}")
        self.blank_token_id = self.model.config.pad_token_id
        self.blank_token = self.processor.tokenizer.decode([self.blank_token_id])
        self.total_duration_s = 0
        logger.info(f"Blank token ID: {self.blank_token_id}")

    def expects_uppercase(self):
        """Check if the model expects uppercase tokens"""
        vocab = self.processor.tokenizer.get_vocab()
        tokens = list(vocab.keys())
        has_upper = any(t.isupper() for t in tokens if t.isalpha())
        has_lower = any(t.islower() for t in tokens if t.isalpha())
        return has_upper and not has_lower

    def transcribe(self, audio: np.ndarray, batch_size: int = 1):
        """Transcribe audio into text with timing information"""
        self.total_duration_s = 0  # Reset duration
        segs = get_audio_segments(audio)
        logger.info(f"Segmenting audio into {len(segs)} segments")
        segs = [segs[i : i + batch_size] for i in range(0, len(segs), batch_size)]
        # Process all segments
        logits = torch.cat(
            [
                self._recognize(seg)
                for seg in tqdm(segs, desc="Processing segments")
                if self._recognize(seg).any()
            ],
            dim=1,
        )
        emission = torch.log_softmax(logits, dim=-1)[0].cpu().detach()
        pred = torch.argmax(logits, dim=-1)
        transcription = self.processor.batch_decode(pred)

        # Compute time per frame
        frame_duration = self.model.config.inputs_to_logits_ratio / TARGET_SR

        return emission, pred, transcription, frame_duration

    def tokenize(self, text_path):
        """Tokenize text into input IDs for the model"""
        text = self.lp.process(text_path)
        return self.processor.tokenizer(text).input_ids

    def _recognize(self, audio_segment):
        """Process a single audio segment"""
        inputs = self.processor(
            audio_segment,
            return_tensors="pt",
            sampling_rate=TARGET_SR,
            padding="longest",
        ).input_values.to(self.device)

        segment_sec = inputs.shape[1] / TARGET_SR
        self.total_duration_s += segment_sec
        with torch.inference_mode():
            logits = self.model(inputs).logits
        return logits

    def get_labels(self):
        """Return sorted token labels used by the tokenizer"""
        return [
            k
            for k, v in sorted(
                self.processor.tokenizer.get_vocab().items(), key=lambda x: x[1]
            )
        ]
