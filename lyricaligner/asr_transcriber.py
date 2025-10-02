import logging

import numpy as np
from line_profiler import profile
from tqdm import tqdm

from lyricaligner.config import DEFAULT_MODEL, TARGET_SR
from lyricaligner.lyrics_processor import LyricsProcessor
from lyricaligner.utils import get_audio_segments

logger = logging.getLogger(__name__)


@profile
class ASRTranscriber:
    def __init__(
        self,
        model_id=None,
        device="cpu",
        batch_size=1,
        is_upper=True,
        use_transformers=True,
    ) -> None:
        self.batch_size = batch_size
        self.model_id = model_id or DEFAULT_MODEL
        self.device = device
        self.use_transformers = use_transformers
        if use_transformers:
            from transformers import (
                Wav2Vec2ForCTC,
                Wav2Vec2Processor,
            )
            from transformers import (
                logging as transformers_logging,
            )

            transformers_logging.set_verbosity_warning()

            self.processor = Wav2Vec2Processor.from_pretrained(self.model_id)
            self.model = Wav2Vec2ForCTC.from_pretrained(self.model_id)
        else:
            # onnxruntime implementation (faster for CPU, no GPU support)
            import onnxruntime as ort

            raise NotImplementedError("ONNX runtime not implemented yet")

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

    def transcribe(self, audio: np.ndarray):
        """Transcribe audio into text with timing information"""
        import torch

        self.total_duration_s = 0  # Reset duration
        segs = get_audio_segments(audio)
        logger.info(f"Segmenting audio into {len(segs)} segments")

        if len(segs) == 1:
            logits = self._recognize(segs[0])
        else:
            all_logits = []
            for i in tqdm(
                range(0, len(segs), self.batch_size),
                desc="Aligning segments",
                total=len(segs) / self.batch_size,
            ):
                batch = segs[i : i + self.batch_size]
                batch_logits = self._recognize_batch(batch)
                all_logits.append(batch_logits)
            logits = torch.cat(
                [x.reshape(-1, x.size(-1)) for x in all_logits], dim=0
            ).unsqueeze(0)

        emission = torch.log_softmax(logits, dim=-1)[0].cpu().detach()
        pred = torch.argmax(logits, dim=-1)
        transcription = self.processor.batch_decode(pred)

        # compute time per frame with dynamic correction
        from lyricaligner.timing_utils import calculate_dynamic_frame_duration

        model_frame_duration = self.model.config.inputs_to_logits_ratio / TARGET_SR
        audio_length = (
            self.total_duration_s
            if self.total_duration_s > 0
            else len(audio) / TARGET_SR
        )
        total_frames = emission.shape[0]
        frame_duration = calculate_dynamic_frame_duration(
            audio_length, total_frames, model_frame_duration
        )

        return emission, pred, transcription, frame_duration

    def _recognize_batch(self, audio_segments):
        """Process a batch of audio segments"""
        # Process each segment in the batch
        inputs = self.processor(
            audio_segments,
            return_tensors="pt",
            sampling_rate=TARGET_SR,
            padding="longest",
        ).input_values.to(self.device)

        # Track duration for each segment in the batch
        for i in range(len(audio_segments)):
            segment_sec = inputs[i].shape[0] / TARGET_SR
            self.total_duration_s += segment_sec
        if self.use_transformers:
            import torch

            with torch.inference_mode():
                logits = self.model(inputs).logits
        else:
            raise NotImplementedError("ONNX inference not implemented yet")
        return logits

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

    def tokenize(self, text_path):
        """Tokenize text into input IDs for the model"""
        # preprocess the input text to something the model recognizes
        text = self.lp.process(text_path)
        return self.processor.tokenizer(text).input_ids

    def get_labels(self):
        """Return sorted token labels used by the tokenizer"""
        return [
            k
            for k, _ in sorted(
                self.processor.tokenizer.get_vocab().items(), key=lambda x: x[1]
            )
        ]
