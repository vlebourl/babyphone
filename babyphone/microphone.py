import logging
import pyaudio

class MicrophoneHandler:
    """Handles microphone operations: device selection, stream initialization, and data reading."""

    def __init__(self, format, channels, rate, frames_per_block):
        """Initialize PyAudio and open the microphone stream."""
        self.format = format
        self.channels = channels
        self.rate = rate
        self.frames_per_block = frames_per_block
        self.pa = pyaudio.PyAudio()
        self.stream = self.init_microphone_stream()

    def select_input_device(self):
        """Select and return the index of a preferred microphone (device containing 'mic' or 'input')."""
        for i in range(self.pa.get_device_count()):
            devinfo = self.pa.get_device_info_by_index(i)
            logging.info("Device %d: %s", i, devinfo["name"])
            if any(keyword in devinfo["name"].lower() for keyword in ["mic", "input"]):
                logging.info("Found an input: device %d - %s", i, devinfo["name"])
                return i
        logging.info("No preferred input found; using default input device.")
        return None

    def init_microphone_stream(self):
        """Initialize and return the microphone stream."""
        device_index = self.select_input_device()
        return self.pa.open(
            format=self.format,
            channels=self.channels,
            rate=self.rate,
            input=True,
            input_device_index=device_index,
            frames_per_buffer=self.frames_per_block,
        )

    def read_block(self):
        """Read and return a block of audio data from the stream."""
        return self.stream.read(self.frames_per_block, exception_on_overflow=False)

    def stop(self):
        """Stop the microphone stream."""
        self.stream.close()

    def reset(self):
        """Reset the microphone stream."""
        self.stop()
        self.pa = pyaudio.PyAudio()
        self.stream = self.init_microphone_stream()