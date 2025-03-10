"""
Babyphone Application

This module captures audio from the microphone, computes the RMS amplitude,
and sends notifications based on thresholds defined in config.json.
"""

import json
import logging
import math
import os
import shutil
import struct
from collections import deque
from datetime import datetime, timedelta
from statistics import mean, median

import pyaudio
import requests
from pydub import AudioSegment
from ratelimit import limits, sleep_and_retry
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s:%(levelname)s:%(message)s",
)


def load_config(config_path="config.json"):
    """Load configuration from a JSON file."""
    with open(config_path, "r") as config_file:
        return json.load(config_file)


config = load_config()

# Configuration values
URL = config.get("LOCAL_WEBHOOK_URL")
NOISE_URL = config.get("LOCAL_NOISE_WEBHOOK_URL")
INITIAL_THRESHOLD = config.get("INITIAL_THRESHOLD")
CHANNELS = config.get("CHANNELS")
RATE = config.get("RATE")
INPUT_BLOCK_TIME = config.get("INPUT_BLOCK_TIME")
INPUT_FRAMES_PER_BLOCK = int(RATE * INPUT_BLOCK_TIME)
FORMAT = pyaudio.paInt16
SHORT_NORMALIZE = 1.0 / 32768.0

# Create a session with retries
session = requests.Session()
retries = Retry(total=5, backoff_factor=1, status_forcelist=[502, 503, 504])
adapter = HTTPAdapter(pool_connections=1, pool_maxsize=10, max_retries=retries)
session.mount("http://", adapter)
session.mount("https://", adapter)


def get_rms(block):
    """
    Calculate the RMS amplitude of a block of audio.
    """
    count = len(block) // 2  # Number of 16-bit samples
    format_str = f"{count}h"
    shorts = struct.unpack(format_str, block)
    sum_squares = sum((sample * SHORT_NORMALIZE) ** 2 for sample in shorts)
    return math.sqrt(sum_squares / count)


# New Class: MicrophoneHandler
class MicrophoneHandler:
    def __init__(self):
        self.pa = pyaudio.PyAudio()
        self.stream = self.init_microphone_stream()

    def select_input_device(self):
        for i in range(self.pa.get_device_count()):
            devinfo = self.pa.get_device_info_by_index(i)
            logging.info("Device %d: %s", i, devinfo["name"])
            if any(keyword in devinfo["name"].lower() for keyword in ["mic", "input"]):
                logging.info("Found an input: device %d - %s", i, devinfo["name"])
                return i
        logging.info("No preferred input found; using default input device.")
        return None

    def init_microphone_stream(self):
        device_index = self.select_input_device()
        return self.pa.open(
            format=FORMAT,
            channels=CHANNELS,
            rate=RATE,
            input=True,
            input_device_index=device_index,
            frames_per_buffer=INPUT_FRAMES_PER_BLOCK,
        )

    def read_block(self):
        return self.stream.read(INPUT_FRAMES_PER_BLOCK, exception_on_overflow=False)

    def stop(self):
        self.stream.close()

    def reset(self):
        self.stop()
        self.pa = pyaudio.PyAudio()
        self.stream = self.init_microphone_stream()


# New Class: NotificationManager
class NotificationManager:
    def __init__(self, mic_handler: MicrophoneHandler):
        self.speaking = True
        self.send_noise_level_timestamp = datetime.now()
        self.mic_handler = mic_handler

    @sleep_and_retry
    @limits(calls=1, period=1)
    def send_post_request(self, url, payload):
        response = session.post(url, json=payload)
        response.raise_for_status()
        logging.debug("Response status (%s): %s", url, response.status_code)

    def record_audio_clip(self):
        # Use mic_handler's stream for recording
        _, _, free = shutil.disk_usage("/")
        if free < 8 * 1024**3:
            clips_dir = os.path.expanduser("~/babyphone/clips/")
            clips = os.listdir(clips_dir)
            clips.sort()  # Oldest first
            if clips:
                os.remove(os.path.join(clips_dir, clips[0]))
        audio_frames = []
        start_time = datetime.now()
        while (datetime.now() - start_time).total_seconds() < 3:
            try:
                block = self.mic_handler.stream.read(
                    INPUT_FRAMES_PER_BLOCK, exception_on_overflow=False
                )
                audio_frames.append(block)
            except IOError as e:
                logging.error("Error recording audio clip: %s", e)
                break
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        clips_dir = os.path.expanduser("~/babyphone/clips/")
        os.makedirs(clips_dir, exist_ok=True)
        wav_path = os.path.join(clips_dir, f"{timestamp}.wav")
        mp3_path = os.path.join(clips_dir, f"{timestamp}.mp3")
        with open(wav_path, "wb") as wf:
            wf.write(b"".join(audio_frames))
        sound = AudioSegment.from_wav(wav_path)
        sound.export(mp3_path, format="mp3")
        os.remove(wav_path)

    def notify_noise_level(self, amplitudes, threshold):
        DURATION_WINDOW = 1.0  # seconds
        required_length = int(DURATION_WINDOW / INPUT_BLOCK_TIME)
        if len(amplitudes) < required_length:
            return
        if datetime.now() - self.send_noise_level_timestamp > timedelta(seconds=1):
            noise_amplitude = mean(list(amplitudes)[-required_length:])
            payload = {
                "noise_amplitude": noise_amplitude,
                "threshold": threshold,
            }
            self.send_post_request(NOISE_URL, payload)
            logging.debug("%s %s", datetime.now().isoformat(), payload)
            self.send_noise_level_timestamp = datetime.now()

    def notify_speaking_event(self, speaking: bool, message: str = ""):
        if speaking != self.speaking:
            self.speaking = speaking
            payload = {
                "speaking": speaking,
                "time": datetime.now().isoformat(),
                "message": message,
            }
            self.send_post_request(URL, payload)
            logging.info("%s %s", datetime.now().isoformat(), payload)
        if speaking:
            self.record_audio_clip()


# New Class: AudioProcessor
class AudioProcessor:
    def __init__(self, notifier: NotificationManager):
        self.threshold = INITIAL_THRESHOLD
        self.noisycount = 0
        self.noise_event = 0
        self.last_events_time = datetime(1900, 1, 1)
        self.amplitudes = deque(maxlen=int(120 / INPUT_BLOCK_TIME))
        self.notifier = notifier

    def process(self, amplitude: float):
        self.amplitudes.append(amplitude)
        self.threshold = median(self.amplitudes) + 0.05
        self.notifier.notify_noise_level(self.amplitudes, self.threshold)
        if amplitude > self.threshold:
            self.noisycount += 1
        else:
            if self.noisycount * INPUT_BLOCK_TIME >= 0.11:
                if (
                    datetime.now() - self.last_events_time > timedelta(seconds=1.5)
                    and self.noise_event >= 3
                ):
                    self.notifier.notify_speaking_event(True)
                logging.info(
                    "%s NOISE, duration: %.4f count: %d",
                    datetime.now().isoformat(),
                    self.noisycount * INPUT_BLOCK_TIME,
                    self.noise_event,
                )
                self.noise_event += 1
                self.last_events_time = datetime.now()
            if datetime.now() - self.last_events_time > timedelta(seconds=180):
                self.notifier.notify_speaking_event(False)
                if self.noise_event > 0:
                    logging.info(
                        "%s SILENT, duration: %.4f count: %d",
                        datetime.now().isoformat(),
                        self.noisycount * INPUT_BLOCK_TIME,
                        self.noise_event,
                    )
                self.noise_event = 0
            self.noisycount = 0


# Updated composite class: AudioMonitor
class AudioMonitor:
    def __init__(self):
        self.mic_handler = MicrophoneHandler()
        self.notifier = NotificationManager(self.mic_handler)
        self.processor = AudioProcessor(self.notifier)
        self.notifier.notify_speaking_event(False, message="Starting")

    def monitor_audio(self):
        try:
            block = self.mic_handler.read_block()
        except IOError as e:
            logging.info("Error recording: %s", e)
            raise e
        amplitude = get_rms(block)
        self.processor.process(amplitude)

    def stop(self):
        self.mic_handler.stop()

    def reset(self):
        self.mic_handler.reset()


def main():
    monitor = AudioMonitor()
    try:
        while True:
            monitor.monitor_audio()
    except Exception as e:
        logging.exception("An error occurred during execution")
    finally:
        monitor.stop()


if __name__ == "__main__":
    main()
