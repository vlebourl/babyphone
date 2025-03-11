import logging
import os
import shutil
from datetime import datetime, timedelta

import requests
from pydub import AudioSegment
from ratelimit import limits, sleep_and_retry
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# Create a session with retries
session = requests.Session()
retries = Retry(total=5, backoff_factor=1, status_forcelist=[502, 503, 504])
adapter = HTTPAdapter(pool_connections=1, pool_maxsize=10, max_retries=retries)
session.mount("http://", adapter)
session.mount("https://", adapter)

from typing import Dict, Any, Deque

class NotificationManager:
    def __init__(self, mic_handler: 'MicrophoneHandler', webhook_url: str, 
                 noise_webhook_url: str, input_block_time: float) -> None:
        self.speaking: bool = True
        self.send_noise_level_timestamp: datetime = datetime.now()
        self.mic_handler: 'MicrophoneHandler' = mic_handler
        self.webhook_url: str = webhook_url
        self.noise_webhook_url: str = noise_webhook_url
        self.input_block_time: float = input_block_time

    @sleep_and_retry
    @limits(calls=1, period=1)
    def send_post_request(self, url: str, payload: Dict[str, Any]) -> None:
        """Send a POST request to the specified URL with the given payload."""
        response = session.post(url, json=payload)
        response.raise_for_status()
        logging.debug("Response status (%s): %s", url, response.status_code)

    def record_audio_clip(self):
        """Record a 3-second audio clip and save it as an MP3 file."""
        # ... rest of the method remains the same ...

    def notify_noise_level(self, amplitudes: Deque[float], threshold: float) -> None:
        """Notify the current noise level if enough data is available and time elapsed."""
        DURATION_WINDOW = 1.0
        required_length = int(DURATION_WINDOW / self.input_block_time)
        if len(amplitudes) < required_length:
            return
        if datetime.now() - self.send_noise_level_timestamp > timedelta(seconds=1):
            noise_amplitude = mean(list(amplitudes)[-required_length:])
            payload = {
                "noise_amplitude": noise_amplitude,
                "threshold": threshold,
            }
            self.send_post_request(self.noise_webhook_url, payload)
            logging.debug("%s %s", datetime.now().isoformat(), payload)
            self.send_noise_level_timestamp = datetime.now()

    def notify_speaking_event(self, speaking: bool, message: str = "") -> None:
        """Notify a speaking event when the speaking state changes."""
        if speaking != self.speaking:
            self.speaking = speaking
            payload = {
                "speaking": speaking,
                "time": datetime.now().isoformat(),
                "message": message,
            }
            self.send_post_request(self.webhook_url, payload)
            logging.info("%s %s", datetime.now().isoformat(), payload)
        if speaking:
            self.record_audio_clip()