"""Composition de l'application : micro → détection → domotique."""

import logging
import sys
from datetime import datetime

from audio_source import (AlreadyRunning, MicrophoneSource,
                          acquire_single_instance_lock)
from config import (INPUT_BLOCK_TIME, MIN_NOISE_DURATION, NOISE_EVENT_COUNT,
                    NOISE_EVENT_TIMEOUT, NOISE_THRESHOLD_ADJUSTMENT, NOISE_URL,
                    SPEAKING_TIMEOUT, URL)
from detection import Detection, Output, Settings, Transition
from emitter import WebhookEmitter

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s:%(levelname)s:%(message)s",
)


def main():
    """Main function to run the application."""
    try:
        # gardé ouvert pour toute la vie du processus : le fermer libère le verrou
        lock = acquire_single_instance_lock()  # noqa: F841
    except AlreadyRunning as e:
        logging.error("%s", e)
        return 1

    detection = Detection(
        Settings(
            block_time=INPUT_BLOCK_TIME,
            threshold_offset=NOISE_THRESHOLD_ADJUSTMENT,
            min_noise_duration=MIN_NOISE_DURATION,
            event_count=NOISE_EVENT_COUNT,
            event_gap=NOISE_EVENT_TIMEOUT,
            calm_timeout=SPEAKING_TIMEOUT,
        )
    )
    emitter = WebhookEmitter(URL, NOISE_URL)
    source = MicrophoneSource()

    # Signale le (re)démarrage : état calme explicite vers la domotique
    emitter.publish(
        Output(
            transitions=(
                Transition(
                    awake=False, at=datetime.now(), noise_duration=0.0, message="Starting"
                ),
            )
        )
    )

    try:
        logging.info("Starting audio monitoring...")
        for now, amplitude in source.readings():
            try:
                emitter.publish(detection.feed(amplitude, now))
            except Exception:
                logging.exception("Unexpected error in processing loop")
    except KeyboardInterrupt:
        logging.info("Application stopped by user")
    finally:
        source.close()
        logging.info("Application shutdown complete")
    return 0


if __name__ == "__main__":
    sys.exit(main())
