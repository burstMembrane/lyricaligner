"""Configuration constants for lyrics synchronization"""

# Audio processing constants
ORIGINAL_SR = 44100  # Original sample rate
TARGET_SR = 16000  # Target sample rate for processing
SEG_HOP_LENGTH = 1  # Segment hop length
WINDOW_LENGTH = 15  # Window length in seconds
# Text processing constants
SEPARATOR = "|"  # Word separator for tokenization

# ASR model configuration
DEFAULT_MODEL = "facebook/wav2vec2-large-960h-lv60-self"
DEFAULT_BLANK_TOKEN_ID = 0
