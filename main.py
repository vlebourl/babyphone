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

# Import configuration from a separate file
from config import (CHANNELS, CLIPS_DIR, FORMAT, INITIAL_THRESHOLD,
                    INPUT_BLOCK_TIME, MIN_NOISE_DURATION, NOISE_EVENT_COUNT,
                    NOISE_EVENT_TIMEOUT, NOISE_THRESHOLD_ADJUSTMENT, NOISE_URL,
                    RATE, SHORT_NORMALIZE, SPEAKING_TIMEOUT, URL)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s:%(levelname)s:%(message)s",
)

# Calculate frames per block based on rate and block time
INPUT_FRAMES_PER_BLOCK = int(RATE * INPUT_BLOCK_TIME)


# Create a Session with retry capability
def create_session():
    session = requests.Session()
    retries = Retry(total=5, backoff_factor=1, status_forcelist=[502, 503, 504])
    adapter = HTTPAdapter(pool_connections=1, pool_maxsize=10, max_retries=retries)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    return session


# Global session
session = create_session()


def get_rms(block):
    """Calculate the Root Mean Square of the audio block."""
    count = len(block) / 2
    formatting = "%dh" % (count)
    shorts = struct.unpack(formatting, block)

    sum_squares = 0.0
    for sample in shorts:
        n = sample * SHORT_NORMALIZE
        sum_squares += n * n

    return math.sqrt(sum_squares / count)


class AudioRecorder:
    """Handles audio recording and clip management."""

    def __init__(self, stream, clips_dir=CLIPS_DIR):
        self.stream = stream
        self.clips_dir = os.path.expanduser(clips_dir)
        os.makedirs(self.clips_dir, exist_ok=True)

    def record_clip(self, duration=3):
        """Record an audio clip of specified duration in seconds."""
        self._ensure_disk_space()

        audio_frames = []
        start_time = datetime.now()
        while (datetime.now() - start_time).total_seconds() < duration:
            try:
                block = self.stream.read(
                    INPUT_FRAMES_PER_BLOCK, exception_on_overflow=False
                )
                audio_frames.append(block)
            except IOError as e:
                logging.error("Error recording audio clip: %s", e)
                break

        if audio_frames:
            self._save_clip(audio_frames)

    def _ensure_disk_space(self, min_space_gb=8):
        """Ensure there's enough disk space, removing old clips if necessary."""
        _, _, free = shutil.disk_usage("/")
        # Check if free space is less than min_space_gb
        if free < min_space_gb * 1024**3:
            clips = [f for f in os.listdir(self.clips_dir) if f.endswith(".mp3")]
            clips.sort()  # oldest will be first
            if clips:
                # Remove the oldest clip
                os.remove(os.path.join(self.clips_dir, clips[0]))
                logging.info(f"Removed old clip {clips[0]} to free space")

    def _save_clip(self, audio_frames):
        """Save recorded audio frames as an MP3 file."""
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        wav_path = os.path.join(self.clips_dir, f"{timestamp}.wav")
        mp3_path = os.path.join(self.clips_dir, f"{timestamp}.mp3")

        try:
            # Save as WAV first
            with open(wav_path, "wb") as wf:
                wf.write(b"".join(audio_frames))

            # Convert to MP3
            sound = AudioSegment.from_wav(wav_path)
            sound.export(mp3_path, format="mp3")

            # Delete the WAV file
            os.remove(wav_path)
            logging.info(f"Saved audio clip to {mp3_path}")
        except Exception as e:
            logging.error(f"Error saving audio clip: {e}")


class ApiClient:
    """Handles API communication."""

    def __init__(self, session):
        self.session = session

    @sleep_and_retry
    @limits(calls=1, period=1)
    def post(self, url, json_data):
        """Send a POST request to the specified URL with rate limiting."""
        try:
            response = self.session.post(url, json=json_data)
            response.raise_for_status()
            logging.info(f"Response status ({url}): {response.status_code}")
            return response
        except requests.exceptions.RequestException as e:
            logging.error(f"API request failed: {e}")
            return None


class TapTester:
    """Main class for audio monitoring and processing."""

    def __init__(self):
        self.pa = pyaudio.PyAudio()
        self.stream = self.open_mic_stream()
        self.threshold = INITIAL_THRESHOLD
        self.noisycount = 0
        self.errorcount = 0
        self.last_events_time = datetime(1900, 1, 1)
        self.speaking = True
        self.noise_event = 0
        self.amplitudes = deque(maxlen=int(120 / INPUT_BLOCK_TIME))
        self.send_noise_level_timestamp = datetime.now()

        self.api_client = ApiClient(session)
        self.recorder = AudioRecorder(self.stream)

        self.send_speaking(False, message="Starting")
        logging.info("Initialized TapTester")

    def stop(self):
        """Stop the audio stream."""
        if self.stream:
            self.stream.close()
        if self.pa:
            self.pa.terminate()

    def reset(self):
        """Reset the audio stream."""
        self.stop()
        self.pa = pyaudio.PyAudio()
        self.stream = self.open_mic_stream()

    def find_input_device(self):
        """Find a suitable input device."""
        device_index = None
        for i in range(self.pa.get_device_count()):
            devinfo = self.pa.get_device_info_by_index(i)
            logging.info("Device %d: %s" % (i, devinfo["name"]))

            for keyword in ["mic", "input"]:
                if keyword in devinfo["name"].lower():
                    logging.info(
                        "Found an input: device %d - %s" % (i, devinfo["name"])
                    )
                    return i

        logging.info("No preferred input found; using default input device.")
        return device_index

    def open_mic_stream(self):
        """Open and configure the microphone stream."""
        device_index = self.find_input_device()

        return self.pa.open(
            format=FORMAT,
            channels=CHANNELS,
            rate=RATE,
            input=True,
            input_device_index=device_index,
            frames_per_buffer=INPUT_FRAMES_PER_BLOCK,
        )

    def display_amplitude(self, amplitude):
        """Display a visual representation of the current amplitude."""
        length = 100
        max_amplitude = 0.2
        vec = [f' {"TALKING" if self.speaking else "SILENT "} [']
        for i in range(length):
            if i <= amplitude * length / max_amplitude:
                vec.append(
                    "#" if i > INITIAL_THRESHOLD * length / max_amplitude else "*"
                )
            else:
                vec.append(" ")
        vec[int(self.threshold * length / max_amplitude)] = "|"
        vec.extend(
            (
                "] ",
                f"level: {amplitude:.4f} threshold={self.threshold:.4f}, "
                f"Noise lasted for {self.noisycount * INPUT_BLOCK_TIME:.2f}s"
                "                 \r",
            )
        )
        logging.info("".join(vec))

    def send_noise_level(self):
        """Send the current noise level to the server."""
        DURATION_WINDOW = 1.0  # seconds
        if len(self.amplitudes) < DURATION_WINDOW / INPUT_BLOCK_TIME:
            return

        if datetime.now() - self.send_noise_level_timestamp > timedelta(seconds=1):
            recent_amplitudes = [
                self.amplitudes[i]
                for i in range(
                    len(self.amplitudes) - int(DURATION_WINDOW / INPUT_BLOCK_TIME),
                    len(self.amplitudes),
                )
            ]

            noise_amplitude = mean(recent_amplitudes)
            json_data = {
                "noise_amplitude": noise_amplitude,
                "threshold": self.threshold,
            }

            self.api_client.post(NOISE_URL, json_data)
            logging.debug(f"{datetime.now().isoformat()} {json_data}")
            self.send_noise_level_timestamp = datetime.now()

    def send_speaking(self, speaking: bool, message: str = ""):
        """Send a message to the server to indicate whether the user is speaking."""
        logging.info(f"send speaking {speaking}: {message}")
        if speaking != self.speaking:
            self.speaking = speaking
            json_data = {
                "speaking": speaking,
                "time": datetime.now().isoformat(),
                "noise": self.noisycount * INPUT_BLOCK_TIME,
                "message": message,
            }

            self.api_client.post(URL, json_data)
            logging.info(f"{datetime.now().isoformat()} {json_data}")

        if speaking:
            self.recorder.record_clip()

    def process_amplitude(self, amplitude: float) -> None:
        """Process the current amplitude value."""
        self.amplitudes.append(amplitude)
        self.threshold = median(self.amplitudes) + NOISE_THRESHOLD_ADJUSTMENT

        # Uncomment to display amplitude visualization
        # self.display_amplitude(amplitude)

        self.send_noise_level()

        if amplitude > self.threshold:
            # noisy block
            self.noisycount += 1
        else:
            # quiet block.
            if self.noisycount * INPUT_BLOCK_TIME >= MIN_NOISE_DURATION:
                if (
                    datetime.now() - self.last_events_time
                    > timedelta(seconds=NOISE_EVENT_TIMEOUT)
                    and self.noise_event >= NOISE_EVENT_COUNT
                ):
                    self.send_speaking(True)
                logging.info(
                    f"{datetime.now().isoformat()} NOISE,  "
                    f"duration is {self.noisycount * INPUT_BLOCK_TIME:.4f} "
                    f"count is {self.noise_event}"
                )
                self.noise_event += 1
                self.last_events_time = datetime.now()

            if datetime.now() - self.last_events_time > timedelta(
                seconds=SPEAKING_TIMEOUT
            ):
                self.send_speaking(False)
                if self.noise_event > 0:
                    logging.info(
                        f"{datetime.now().isoformat()} SILENT, "
                        f"duration is {self.noisycount * INPUT_BLOCK_TIME:.4f} "
                        f"count is {self.noise_event}"
                    )
                self.noise_event = 0

            self.noisycount = 0

    def listen(self):
        """Listen for audio input and process it."""
        try:
            block = self.stream.read(
                INPUT_FRAMES_PER_BLOCK, exception_on_overflow=False
            )
            amplitude = get_rms(block)
            self.process_amplitude(amplitude)
        except IOError as e:
            self.errorcount += 1
            logging.info("(%d) Error recording: %s" % (self.errorcount, e))
            self.noisycount = 1
            if self.errorcount > 5:
                logging.warning("Too many errors, resetting audio stream")
                self.reset()
                self.errorcount = 0
        except Exception as e:
            logging.exception(f"Unexpected error in listen: {e}")


def main():
    """Main function to run the application."""
    tt = TapTester()
    try:
        logging.info("Starting audio monitoring...")
        while True:
            tt.listen()
    except KeyboardInterrupt:
        logging.info("Application stopped by user")
    except Exception as e:
        logging.exception("An error occurred during execution")
    finally:
        tt.stop()
        logging.info("Application shutdown complete")


if __name__ == "__main__":
    main()
