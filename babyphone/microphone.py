import logging
import pyaudio
from typing import Optional

class MicrophoneHandler:
    def __init__(self, format: int, channels: int, rate: int, frames_per_block: int) -> None:
        self.format: int = format
        self.channels: int = channels
        self.rate: int = rate
        self.frames_per_block: int = frames_per_block
        self.pa: pyaudio.PyAudio = pyaudio.PyAudio()
        self.stream: pyaudio.Stream = self.init_microphone_stream()

    def select_input_device(self) -> Optional[int]:
        for i in range(self.pa.get_device_count()):
            devinfo = self.pa.get_device_info_by_index(i)
            logging.info("Device %d: %s", i, devinfo["name"])
            if any(keyword in devinfo["name"].lower() for keyword in ["mic", "input"]):
                logging.info("Found an input: device %d - %s", i, devinfo["name"])
                return i
        logging.info("No preferred input found; using default input device.")
        return None

    def init_microphone_stream(self) -> pyaudio.Stream:
        device_index = self.select_input_device()
        return self.pa.open(
            format=self.format,
            channels=self.channels,
            rate=self.rate,
            input=True,
            input_device_index=device_index,
            frames_per_buffer=self.frames_per_block,
        )

    def read_block(self) -> bytes:
        return self.stream.read(self.frames_per_block, exception_on_overflow=False)

    def stop(self) -> None:
        self.stream.close()

    def reset(self) -> None:
        self.stop()
        self.pa = pyaudio.PyAudio()
        self.stream = self.init_microphone_stream()