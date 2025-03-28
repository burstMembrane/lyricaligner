"""Type definitions for the lsync package"""

from typing import Any, List, Tuple

# Audio processing types
AudioArray = Any  # Typically a numpy array of audio samples

# Common return types
AlignmentResult = Tuple[List["Word"], str]  # (words, lrc_string)
TranscriptionResult = Tuple[
    Any, Any, List[str], float
]  # (emission, prediction, transcription, frame_duration)
