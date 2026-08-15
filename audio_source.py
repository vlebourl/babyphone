"""Source d'amplitudes : le micro USB, vu comme un itérable de (instant, amplitude).

Cache tout ce que la cible impose (ADR-0005) : découverte du périphérique par
heuristique ALSA, calcul du RMS, et surtout la tolérance aux coupures du flux —
mode de panne dominant sur le 3A+ (sous-tensions de l'alimentation micro-USB).
L'interface est `readings()` ; en test, n'importe quel itérable de
(datetime, float) remplit le même rôle sans adaptateur dédié.
"""

import logging
import math
import struct
from datetime import datetime
from typing import Iterator

import pyaudio

from config import CHANNELS, FORMAT, INPUT_BLOCK_TIME, RATE, SHORT_NORMALIZE

# Calculate frames per block based on rate and block time
INPUT_FRAMES_PER_BLOCK = int(RATE * INPUT_BLOCK_TIME)

MAX_READ_ERRORS = 5  # au-delà, on réinitialise complètement la pile audio


def get_rms(block: bytes) -> float:
    """Calculate the Root Mean Square of the audio block."""
    count = len(block) / 2
    formatting = "%dh" % (count)
    shorts = struct.unpack(formatting, block)

    sum_squares = 0.0
    for sample in shorts:
        n = sample * SHORT_NORMALIZE
        sum_squares += n * n

    return math.sqrt(sum_squares / count)


class MicrophoneSource:
    """Adaptateur pyaudio de la source d'amplitudes."""

    def __init__(self):
        self._pa = pyaudio.PyAudio()
        self._stream = self._open_mic_stream()
        self._error_count = 0

    def readings(self) -> Iterator[tuple[datetime, float]]:
        """Flux infini de (instant, amplitude), résilient aux coupures du micro."""
        while True:
            try:
                block = self._stream.read(
                    INPUT_FRAMES_PER_BLOCK, exception_on_overflow=False
                )
            except IOError as e:
                self._error_count += 1
                logging.info("(%d) Error recording: %s", self._error_count, e)
                if self._error_count > MAX_READ_ERRORS:
                    logging.warning("Too many errors, resetting audio stream")
                    self._reset()
                    self._error_count = 0
                continue
            yield datetime.now(), get_rms(block)

    def close(self):
        """Libère le flux et la pile audio."""
        if self._stream:
            self._stream.close()
        if self._pa:
            self._pa.terminate()

    def _reset(self):
        self.close()
        self._pa = pyaudio.PyAudio()
        self._stream = self._open_mic_stream()

    def _find_input_device(self):
        """Heuristique ALSA/Pi : premier périphérique nommé « mic » ou « input »."""
        for i in range(self._pa.get_device_count()):
            devinfo = self._pa.get_device_info_by_index(i)
            logging.info("Device %d: %s", i, devinfo["name"])

            for keyword in ["mic", "input"]:
                if keyword in devinfo["name"].lower():
                    logging.info("Found an input: device %d - %s", i, devinfo["name"])
                    return i

        logging.info("No preferred input found; using default input device.")
        return None

    def _open_mic_stream(self):
        return self._pa.open(
            format=FORMAT,
            channels=CHANNELS,
            rate=RATE,
            input=True,
            input_device_index=self._find_input_device(),
            frames_per_buffer=INPUT_FRAMES_PER_BLOCK,
        )
