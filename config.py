from pathlib import Path

# =============================================================================
# CONFIGURATION
# =============================================================================

SAMPLE_RATE = 44100
BLOCKSIZE = 256
CHANNELS = 1
MAX_LOOP_SECONDS = 120

# Export settings
EXPORT_MP3_BITRATE = "192k"
EXPORT_WAV_SAMPLE_WIDTH = 2  # 16-bit

# Sessions directory (relative to this script)
SESSIONS_DIR = Path(__file__).parent / '_sessions'
SESSIONS_DIR.mkdir(exist_ok=True)

# Layer color palette (cycles by layer id)
LAYER_COLORS = [
    '#667eea',  # Purple
    '#38a169',  # Green
    '#ed8936',  # Orange
    '#e53e3e',  # Red
    '#319795',  # Teal
    '#d53f8c',  # Pink
    '#d69e2e',  # Yellow
    '#3182ce',  # Blue
]
