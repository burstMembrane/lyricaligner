"""Utilities for timing correction and drift detection in lyrics alignment"""

import logging
from typing import List
import numpy as np
from lyricaligner.formatters import Word

logger = logging.getLogger(__name__)


def detect_timing_drift(words: List[Word], max_drift_rate: float = 0.05) -> bool:
    """
    Detect if there's significant timing drift in the word alignments.
    
    Args:
        words: List of aligned words with timing
        max_drift_rate: Maximum acceptable drift rate (5% by default)
    
    Returns:
        True if significant drift is detected
    """
    if len(words) < 10:  # Need sufficient data points
        return False
    
    # Calculate expected vs actual timing progression
    total_duration = words[-1].end - words[0].start
    word_count = len(words)
    expected_word_duration = total_duration / word_count
    
    # Check for increasing gaps between words (drift indicator)
    gaps = []
    for i in range(1, len(words)):
        gap = words[i].start - words[i-1].end
        gaps.append(gap)
    
    # Detect if gaps are increasing significantly over time
    if len(gaps) > 5:
        early_gaps = np.mean(gaps[:len(gaps)//3])
        late_gaps = np.mean(gaps[-len(gaps)//3:])
        
        if late_gaps > early_gaps * (1 + max_drift_rate):
            logger.warning(f"Timing drift detected: early gaps {early_gaps:.3f}s, late gaps {late_gaps:.3f}s")
            return True
    
    return False


def calculate_dynamic_frame_duration(audio_length: float, total_frames: int, 
                                   model_frame_duration: float) -> float:
    """
    Calculate frame duration based on actual audio length vs expected frames.
    
    Args:
        audio_length: Actual audio length in seconds
        total_frames: Total number of frames from the model
        model_frame_duration: Frame duration from model config
    
    Returns:
        Corrected frame duration
    """
    expected_duration = total_frames * model_frame_duration
    correction_factor = audio_length / expected_duration
    
    corrected_duration = model_frame_duration * correction_factor
    
    if abs(correction_factor - 1.0) > 0.01:  # Log if correction > 1%
        logger.info(f"Frame duration corrected: {model_frame_duration:.6f}s -> {corrected_duration:.6f}s "
                   f"(factor: {correction_factor:.4f})")
    
    return corrected_duration


def smooth_timing_with_confidence(words: List[Word], confidence_threshold: float = 0.5) -> List[Word]:
    """
    Smooth word timing using confidence scores to reduce jitter.
    
    Args:
        words: List of words with timing and confidence
        confidence_threshold: Minimum confidence for anchoring timing
    
    Returns:
        List of words with smoothed timing
    """
    if len(words) <= 2:
        return words
    
    smoothed_words = words.copy()
    
    # Find high-confidence anchor points
    anchors = []
    for i, word in enumerate(words):
        if hasattr(word, 'confidence') and word.confidence > confidence_threshold:
            anchors.append((i, word.start, word.end))
    
    # Interpolate timing between anchors
    for i in range(len(anchors) - 1):
        start_idx, start_time, _ = anchors[i]
        end_idx, _, end_time = anchors[i + 1]
        
        if end_idx - start_idx > 1:  # Only interpolate if there are words in between
            # Linear interpolation between anchors
            segment_words = words[start_idx:end_idx + 1]
            segment_duration = end_time - start_time
            
            # Redistribute timing proportionally based on word length
            total_chars = sum(len(w.text) for w in segment_words)
            
            current_time = start_time
            for j, word in enumerate(segment_words[:-1]):  # Skip last word (it's the anchor)
                word_portion = len(word.text) / total_chars
                word_duration = segment_duration * word_portion
                
                smoothed_words[start_idx + j] = Word(
                    text=word.text,
                    start=current_time,
                    end=current_time + word_duration,
                    confidence=getattr(word, 'confidence', 0.0)
                )
                current_time += word_duration
    
    return smoothed_words


def apply_timing_anchors(words: List[Word], anchor_interval: float = 10.0) -> List[Word]:
    """
    Create timing anchor points at regular intervals to prevent drift accumulation.
    
    Args:
        words: List of words with timing
        anchor_interval: Seconds between anchor points
    
    Returns:
        List of words with anchored timing
    """
    if len(words) <= 2:
        return words
    
    anchored_words = words.copy()
    total_duration = words[-1].end - words[0].start
    
    if total_duration <= anchor_interval:
        return words  # No need for anchoring in short audio
    
    # Create anchor points every anchor_interval seconds
    anchor_times = []
    current_time = words[0].start
    while current_time < words[-1].end:
        anchor_times.append(current_time)
        current_time += anchor_interval
    anchor_times.append(words[-1].end)  # Always anchor the end
    
    # Find words closest to each anchor time and adjust surrounding words
    for i, anchor_time in enumerate(anchor_times[1:-1], 1):  # Skip first and last
        # Find word closest to anchor time
        closest_idx = min(range(len(words)), 
                         key=lambda x: abs(words[x].start - anchor_time))
        
        # Calculate expected time based on position in sequence
        progress = closest_idx / (len(words) - 1)
        expected_time = words[0].start + progress * total_duration
        
        # Apply small correction if drift is detected
        actual_time = words[closest_idx].start
        drift = actual_time - expected_time
        
        if abs(drift) > 0.5:  # Only correct significant drift (>0.5s)
            correction = -drift * 0.3  # Apply 30% correction to avoid overcorrection
            
            # Apply correction to surrounding words
            window_size = min(5, len(words) // 10)  # Correction window
            start_idx = max(0, closest_idx - window_size)
            end_idx = min(len(words), closest_idx + window_size + 1)
            
            for j in range(start_idx, end_idx):
                weight = 1.0 - abs(j - closest_idx) / window_size
                time_correction = correction * weight
                
                anchored_words[j] = Word(
                    text=words[j].text,
                    start=words[j].start + time_correction,
                    end=words[j].end + time_correction,
                    confidence=getattr(words[j], 'confidence', 0.0)
                )
    
    return anchored_words