"""
Babyphone Application

This module captures audio from the microphone, computes the RMS amplitude,
and sends notifications based on thresholds defined in config.json.
"""

import json
import logging
import pyaudio
from babyphone.monitor import AudioMonitor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s:%(levelname)s:%(message)s",
)

def load_config(config_path="config.json"):
    """Load configuration from a JSON file."""
    try:
        with open(config_path, "r") as config_file:
            config = json.load(config_file)
            
        required_keys = [
            "LOCAL_WEBHOOK_URL",
            "LOCAL_NOISE_WEBHOOK_URL",
            "INITIAL_THRESHOLD",
            "CHANNELS",
            "RATE",
            "INPUT_BLOCK_TIME"
        ]
        if missing_keys := [key for key in required_keys if key not in config]:
            raise KeyError(f"Missing required configuration keys: {', '.join(missing_keys)}")
            
        # Add FORMAT to config
        config["FORMAT"] = pyaudio.paInt16
        return config
            
    except FileNotFoundError:
        logging.error(f"Configuration file not found: {config_path}")
        logging.info("Please copy config.template.json to config.json and configure it")
        raise
    except json.JSONDecodeError as e:
        logging.error(f"Invalid JSON in configuration file: {e}")
        raise
    except KeyError as e:
        logging.error(str(e))
        raise

def main():
    config = load_config()
    monitor = AudioMonitor(config)
    try:
        while True:
            monitor.monitor_audio()
    except Exception as e:
        logging.exception("An error occurred during execution")
    finally:
        monitor.stop()

if __name__ == "__main__":
    main()
