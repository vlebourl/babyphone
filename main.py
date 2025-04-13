import math
import struct
import logging
from collections import deque
from datetime import datetime, timedelta
from ratelimit import limits, sleep_and_retry
from statistics import mean, median
import os
import shutil
from pydub import AudioSegment
import pyaudio
import requests

from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# Create a Session
session = requests.Session()

# Setup retries: (Optional, but a good practice)
retries = Retry(total=5, backoff_factor=1, status_forcelist=[502, 503, 504])
session.mount("https://", HTTPAdapter(max_retries=retries))

# Mount it for both HTTP and HTTPS
adapter = HTTPAdapter(pool_connections=1, pool_maxsize=10)
session.mount("http://", adapter)
session.mount("https://", adapter)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s:%(levelname)s:%(message)s",
)

URL = "https://ha.tiarkaerell.com/api/webhook/create-event-on-webhook-babyphone-81msbShN2-7pcRq_WCHZICRN"
NOISE_URL = "https://ha.tiarkaerell.com/api/webhook/noise-babyphone-81msbShN2-7pcRq_WCHZICRN"

URL = "http://192.168.1.10/api/webhook/create-event-on-webhook-babyphone-81msbShN2-7pcRq_WCHZICRN"
NOISE_URL = "http://192.168.1.10/api/webhook/noise-babyphone-81msbShN2-7pcRq_WCHZICRN"

INITIAL_THRESHOLD = 0.09
FORMAT = pyaudio.paInt16
SHORT_NORMALIZE = 1.0 / 32768.0
CHANNELS = 1
RATE = 48000#44100
INPUT_BLOCK_TIME = 0.05
INPUT_FRAMES_PER_BLOCK = int(RATE * INPUT_BLOCK_TIME)


def get_rms(block):
    # RMS amplitude is defined as the square root of the
    # mean over time of the square of the amplitude.
    # so we need to convert this string of bytes into
    # a string of 16-bit samples...

    # we will get one short out for each
    # two chars in the string.
    count = len(block) / 2
    formating = "%dh" % (count)
    shorts = struct.unpack(formating, block)

    # iterate over the block.
    sum_squares = 0.0
    for sample in shorts:
        # sample is a signed short in +/- 32768.
        # normalize it to 1.0
        n = sample * SHORT_NORMALIZE
        sum_squares += n * n

    return math.sqrt(sum_squares / count)


class TapTester(object):
    def __init__(self):
        self.pa = pyaudio.PyAudio()
        self.stream = self.open_mic_stream()
        self.sample_rate = RATE
        self.threshold = INITIAL_THRESHOLD
        self.noisycount = 0
        self.errorcount = 0
        self.last_events_time = datetime(1900, 1, 1)
        self.speaking = True
        self.send_speaking(False, message="Starting")
        self.noise_event = 0
        self.amplitudes = deque(maxlen=int(120 / INPUT_BLOCK_TIME))
        self.send_noise_level_timestamp = datetime.now()
        logging.info("Initialized TapTester")

    def stop(self):
        self.stream.close()

    def reset(self):
        self.stop()
        self.pa = pyaudio.PyAudio()
        self.stream = self.open_mic_stream()

    def find_input_device(self):
        device_index = None
        for i in range(self.pa.get_device_count()):
            devinfo = self.pa.get_device_info_by_index(i)
            logging.info("Device %d: %s" % (i, devinfo["name"]))

            for keyword in ["mic", "input"]:
                if keyword in devinfo["name"].lower():
                    logging.info("Found an input: device %d - %s" % (i, devinfo["name"]))
                    return i
        if device_index is None:
            logging.info("No preferred input found; using default input device.")

        return device_index

    def open_mic_stream(self):
        device_index = self.find_input_device()

        return self.pa.open(
            format=FORMAT,
            channels=CHANNELS,
            rate=RATE,
            input=True,
            input_device_index=device_index,
            frames_per_buffer=INPUT_FRAMES_PER_BLOCK,
        )

    def displayAmplitude(self, amplitude):
        length = 100
        max_amplitude = 0.2
        vec = [f' {"TALKING" if self.speaking else "SILENT "} [']
        for i in range(length):
            if i <= amplitude * length / max_amplitude:
                vec.append("#" if i > INITIAL_THRESHOLD * length / max_amplitude else "*")
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

    @sleep_and_retry
    @limits(calls=1, period=1)
    def post(self, url, json):
        response = session.post(url, json=json)
        response.raise_for_status()
        logging.info(f"Response status ({url}): {response.status_code}")

    def record_clip(self):
        # Check available space
        _, _, free = shutil.disk_usage("/")
        # Check if free space is less than 8GB
        if free < 8 * 1024 ** 3:
            # Get list of existing clips
            clips = os.listdir("~/babyphone/clips/")
            clips.sort()  # oldest will be first
            if clips:
                # Remove the oldest clip
                os.remove(os.path.join("~/babyphone/clips/", clips[0]))

        # Record audio for 3 seconds
        audio_frames = []
        start_time = datetime.now()
        while (datetime.now() - start_time).total_seconds() < 3:
            try:
                block = self.stream.read(INPUT_FRAMES_PER_BLOCK, exception_on_overflow=False)
                audio_frames.append(block)
            except IOError as e:
                logging.error("Error recording audio clip: %s", e)
                break
        
        # Save the recorded audio to an mp3 file
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        file_path = os.path.expanduser(f"~/babyphone/clips/{timestamp}.wav")
        with open(file_path, 'wb') as wf:
            wf.write(b''.join(audio_frames))
        
        # Convert to mp3
        sound = AudioSegment.from_wav(file_path)
        sound.export(os.path.expanduser(f"~/babyphone/clips/{timestamp}.mp3"), format="mp3")
        os.remove(file_path)  # Delete the wav file after conversion to mp3

    def send_noise_level(self):
        logging.info("send noise level")
        DURATION_WINDOW = 1.0  # seconds
        if len(self.amplitudes) < DURATION_WINDOW / INPUT_BLOCK_TIME:
            return
        if datetime.now() - self.send_noise_level_timestamp > timedelta(seconds=1):
            noise_amplitude = mean(
                [
                    self.amplitudes[i]
                    for i in range(
                        len(self.amplitudes) - int(DURATION_WINDOW / INPUT_BLOCK_TIME),
                        len(self.amplitudes),
                    )
                ]
            )
            json = {
                "noise_amplitude": noise_amplitude,
                "threshold": self.threshold,
            }
            self.post(NOISE_URL, json)
            logging.debug(f"{datetime.now().isoformat()} {json}")
            self.send_noise_level_timestamp = datetime.now()

    def send_speaking(self, speaking: bool, message: str = ""):
        """Send a message to the server to indicate whether the user is speaking."""
        logging.info(f"send speaking {speaking}: {message}")
        if speaking != self.speaking:
            self.speaking = speaking
            json = {
                "speaking": speaking,
                "time": datetime.now().isoformat(),
                "noise": self.noisycount * INPUT_BLOCK_TIME,
                "message": message,
            }
            self.post(URL, json)
            logging.info(f"{datetime.now().isoformat()} {json}")
        if speaking:
            self.record_clip()
            

    def process_amplitude(self, amplitude: float) -> None:
        self.amplitudes.append(amplitude)
        self.threshold = median(self.amplitudes) + 0.05

        # self.displayAmplitude(amplitude)
        self.send_noise_level()
        if amplitude > self.threshold:
            # noisy block
            self.noisycount += 1
        else:
            # quiet block.
            if self.noisycount * INPUT_BLOCK_TIME >= 0.11:
                if (
                    datetime.now() - self.last_events_time > timedelta(seconds=1.5)
                    and self.noise_event >= 3
                ):
                    self.send_speaking(True)
                logging.info(
                    f"{datetime.now().isoformat()} NOISE,  "
                    f"duration is {self.noisycount * INPUT_BLOCK_TIME:.4f} "
                    f"count is {self.noise_event}"
                )
                self.noise_event += 1
                self.last_events_time = datetime.now()

            if datetime.now() - self.last_events_time > timedelta(seconds=180):
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
        try:
            block = self.stream.read(INPUT_FRAMES_PER_BLOCK, exception_on_overflow=False)
        except IOError as e:
            # dammit.
            self.errorcount += 1
            logging.info("(%d) Error recording: %s" % (self.errorcount, e))
            self.noisycount = 1
            raise e

        amplitude = get_rms(block)
        self.process_amplitude(amplitude)


if __name__ == "__main__":
    tt = TapTester()
    try:
        while True:
            tt.listen()
    except Exception as e:
        logging.exception("An error occurred during execution")
    finally:
        tt.stop()
