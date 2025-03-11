from collections import deque
from datetime import datetime, timedelta
from statistics import median

class AudioProcessor:
    """Processes audio amplitude data and handles noise-event logic."""

    def __init__(self, notifier, initial_threshold, input_block_time):
        """Initialize the AudioProcessor."""
        self.threshold = initial_threshold
        self.noisycount = 0
        self.noise_event = 0
        self.last_events_time = datetime(1900, 1, 1)
        self.amplitudes = deque(maxlen=int(120 / input_block_time))
        self.notifier = notifier
        self.input_block_time = input_block_time

    def process(self, amplitude: float):
        """
        Process an audio amplitude value: update history, compute threshold, and notify.

        Args:
            amplitude (float): The current amplitude of audio input.
        """
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