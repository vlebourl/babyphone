import logging
from .microphone import MicrophoneHandler
from .notifications import NotificationManager
from .processor import AudioProcessor
from .utils import get_rms

class AudioMonitor:
    """Integrates microphone handling, audio processing, and notifications."""

    def __init__(self, config):
        """Initialize AudioMonitor and notify initial state."""
        self.config = config
        self.mic_handler = MicrophoneHandler(
            format=config["FORMAT"],
            channels=config["CHANNELS"],
            rate=config["RATE"],
            frames_per_block=int(config["RATE"] * config["INPUT_BLOCK_TIME"])
        )
        self.notifier = NotificationManager(
            self.mic_handler,
            config["LOCAL_WEBHOOK_URL"],
            config["LOCAL_NOISE_WEBHOOK_URL"],
            config["INPUT_BLOCK_TIME"]
        )
        self.processor = AudioProcessor(
            self.notifier,
            config["INITIAL_THRESHOLD"],
            config["INPUT_BLOCK_TIME"]
        )
        self.notifier.notify_speaking_event(False, message="Starting")

    def monitor_audio(self):
        """Read an audio block, compute its RMS, and process it."""
        try:
            block = self.mic_handler.read_block()
        except IOError as e:
            logging.info("Error recording: %s", e)
            raise e
        amplitude = get_rms(block)
        self.processor.process(amplitude)

    def stop(self):
        """Stop audio monitoring."""
        self.mic_handler.stop()

    def reset(self):
        """Reset audio monitoring."""
        self.mic_handler.reset()