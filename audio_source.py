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
import time
from datetime import datetime
from typing import Iterator

import pyaudio

from config import CHANNELS, FORMAT, INPUT_BLOCK_TIME, RATE, SHORT_NORMALIZE

try:
    # RMS en C : ~50× plus rapide que la boucle Python, crucial dans le budget
    # de 50 ms par bloc de la cible (ADR-0005). Déprécié (retiré en 3.13) mais
    # présent sur le Python 3.11 de Bookworm ; repli pur Python sinon.
    import audioop
except ImportError:  # pragma: no cover - dépend de la version de Python
    audioop = None

# Calculate frames per block based on rate and block time
INPUT_FRAMES_PER_BLOCK = int(RATE * INPUT_BLOCK_TIME)

MAX_READ_ERRORS = 5  # au-delà, on réinitialise complètement la pile audio


def get_rms(block: bytes) -> float:
    """RMS normalisé [0, 1] d'un bloc PCM 16 bits.

    L'écart entre les deux chemins est < 1/32768 (arrondi entier d'audioop),
    très en dessous de toute marge de décision.
    """
    count = len(block) // 2
    if count == 0:
        return 0.0  # lecture vide (flux en cours de coupure) : silence, pas de crash

    if audioop is not None:
        return audioop.rms(block, 2) * SHORT_NORMALIZE

    shorts = struct.unpack("%dh" % count, block)
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
            # lecture réussie : le seuil de reset compte des erreurs CONSÉCUTIVES,
            # pas un cumul sur toute la vie du processus
            self._error_count = 0
            yield datetime.now(), get_rms(block)

    def close(self):
        """Libère le flux et la pile audio."""
        if self._stream:
            self._stream.close()
        if self._pa:
            self._pa.terminate()

    def _reset(self):
        """Réinitialise la pile audio, avec retries : un micro USB en sous-tension
        peut disparaître quelques secondes — crasher ici laisserait la chambre
        sans surveillance, on insiste jusqu'à son retour."""
        self.close()
        delay = 1
        while True:
            try:
                self._pa = pyaudio.PyAudio()
                self._stream = self._open_mic_stream()
                return
            except Exception:
                logging.exception(
                    "Failed to reopen audio stream; retrying in %ds", delay
                )
                try:
                    self._pa.terminate()
                except Exception:
                    pass
                time.sleep(delay)
                delay = min(delay * 2, 30)

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
