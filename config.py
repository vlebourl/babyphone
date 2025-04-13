"""
Configuration settings for the babyphone application.
"""

import logging
from typing import Any, Dict

import pyaudio

# Audio settings
INITIAL_THRESHOLD = 0.09
FORMAT = pyaudio.paInt16
SHORT_NORMALIZE = 1.0 / 32768.0
CHANNELS = 1
RATE = 48000  # Sample rate
INPUT_BLOCK_TIME = 0.05  # Time in seconds for each audio block

# File paths
CLIPS_DIR = "~/babyphone/clips/"

# Default API endpoints (will be overridden by secrets.py if available)
DEFAULT_URL = "http://localhost/api/webhook/babyphone"
DEFAULT_NOISE_URL = "http://localhost/api/webhook/noise-babyphone"

# Application settings
SPEAKING_TIMEOUT = 180  # seconds before considering silence
NOISE_THRESHOLD_ADJUSTMENT = 0.05  # adjustment added to median for threshold
MIN_NOISE_DURATION = 0.11  # minimum noise duration to trigger an event
NOISE_EVENT_COUNT = 3  # number of noise events before considering speaking
NOISE_EVENT_TIMEOUT = 1.5  # seconds between noise events

# Load secrets if available
try:
    from secrets import NOISE_URL, URL

    logging.info("Loaded API endpoints from secrets.py")
except ImportError:
    logging.warning("secrets.py not found, using default API endpoints")
    URL = DEFAULT_URL
    NOISE_URL = DEFAULT_NOISE_URL


def get_config() -> Dict[str, Any]:
    """Return all configuration as a dictionary for easier access."""
    return {k: v for k, v in globals().items() if not k.startswith("_") and k.isupper()}
