"""Source d'amplitudes : le micro USB, vu comme un itérable de (instant, amplitude).

Cache tout ce que la cible impose (ADR-0005) : découverte du périphérique par
heuristique ALSA, calcul du RMS, et surtout la tolérance aux coupures du flux —
mode de panne dominant sur le 3A+ (sous-tensions de l'alimentation micro-USB).
L'interface est `readings()` ; en test, n'importe quel itérable de
(datetime, float) remplit le même rôle sans adaptateur dédié.
"""

import fcntl
import logging
import math
import os
import struct
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Iterator

import pyaudio

from config import (CHANNELS, FORMAT, INPUT_BLOCK_TIME, RATE, SHORT_NORMALIZE,
                    VOICE_BAND_HIGH, VOICE_BAND_LOW)

try:
    import numpy as np
except ImportError:  # pragma: no cover - la cible l'a toujours
    np = None

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

# Le micro USB n'accepte qu'un seul client ALSA : deux instances du babyphone
# se disputent le périphérique et la seconde meurt en « Device unavailable ».
# Le verrou rend le démarrage sûr quel que soit le lanceur (systemd, domotique
# par SSH, lancement manuel) — voir docs/adr/0006.
LOCK_PATH = "/tmp/babyphone.lock"


class AlreadyRunning(RuntimeError):
    """Une autre instance détient déjà le micro."""


def acquire_single_instance_lock(path: str = LOCK_PATH):
    """Prend un verrou exclusif non bloquant ; rend l'objet fichier à garder ouvert.

    Le verrou est libéré automatiquement par le noyau à la mort du processus,
    y compris sur SIGKILL — pas de fichier PID périmé à nettoyer.
    """
    handle = open(path, "w")
    try:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError as e:
        handle.close()
        raise AlreadyRunning(
            f"Une autre instance du babyphone tourne déjà (verrou {path}). "
            "Vérifier avec `systemctl status babyphone` et `pgrep -af main.py`."
        ) from e
    handle.write(str(os.getpid()))
    handle.flush()
    return handle


# Plancher du silence numérique : évite log10(0) = -inf quand le micro rend
# des blocs strictement nuls. -120 dBFS est très en dessous de tout fond réel.
SILENCE_FLOOR = 1e-6
MIN_DBFS = 20 * math.log10(SILENCE_FLOOR)


def to_dbfs(rms: float) -> float:
    """Convertit un RMS normalisé [0, 1] en dBFS (négatif, 0 = pleine échelle).

    L'échelle logarithmique est celle de la perception sonore, et elle rend
    la sensibilité du seuil indépendante du fond ambiant (ADR-0008).
    """
    return 20 * math.log10(max(rms, SILENCE_FLOOR))


# Bornes de la bande vocale exprimées en indices de raie FFT. Une raie vaut
# RATE/INPUT_FRAMES_PER_BLOCK = 20 Hz ici.
_BAND_LO = int(VOICE_BAND_LOW * INPUT_FRAMES_PER_BLOCK / RATE)
_BAND_HI = int(VOICE_BAND_HIGH * INPUT_FRAMES_PER_BLOCK / RATE)


# Sous-bandes de la voix (ADR-0011). Les pleurs concentrent leur énergie plus
# haut que la parole : une fondamentale de pleur tourne autour de 400-600 Hz
# avec des harmoniques très fortes vers 1-3 kHz, là où la parole reste plus
# grave et plus plate.
_SUB_BANDS = ((300, 800), (800, 2000), (2000, 4000))


@dataclass(frozen=True)
class Spectrum:
    """Ce qu'un bloc audio révèle, au-delà de son seul volume."""

    dbfs: float  # énergie dans la bande vocale
    centroid_hz: float  # centre de gravité spectral : « où » se situe le son
    low: float  # part de l'énergie en 300-800 Hz
    mid: float  # part en 800-2000 Hz
    high: float  # part en 2000-4000 Hz


def _bin(hz: int) -> int:
    return int(hz * INPUT_FRAMES_PER_BLOCK / RATE)


def analyse(block: bytes) -> Spectrum:
    """Analyse spectrale d'un bloc : énergie, centre de gravité, sous-bandes.

    Une seule FFT sert tout : c'est la même transformée qui donnait déjà
    l'énergie de bande (ADR-0010), on se contente d'en lire davantage. Le
    surcoût est celui de quelques sommes sur un tableau déjà calculé.
    """
    if np is None or len(block) < 4:
        return Spectrum(to_dbfs(get_rms(block)), 0.0, 0.0, 0.0, 0.0)

    x = np.frombuffer(block, dtype="<i2").astype(np.float32)
    spec = np.fft.rfft(x)
    power = spec.real**2 + spec.imag**2

    voix = power[_BAND_LO:_BAND_HI]
    total = float(voix.sum())
    rms = math.sqrt(2 * total) / len(x) * SHORT_NORMALIZE

    if total <= 0:
        return Spectrum(to_dbfs(rms), 0.0, 0.0, 0.0, 0.0)

    freqs = np.arange(_BAND_LO, _BAND_HI) * (RATE / len(x))
    centroid = float((freqs * voix).sum() / total)
    parts = [float(power[_bin(lo):_bin(hi)].sum()) / total for lo, hi in _SUB_BANDS]
    return Spectrum(to_dbfs(rms), centroid, *parts)


def band_rms(block: bytes) -> float:
    """RMS normalisé de la seule bande vocale (ADR-0010).

    On ne veut pas le signal filtré, seulement son énergie : une FFT la donne
    directement par Parseval, sans filtre récursif ni état à maintenir. Un
    biquad équivalent en Python pur coûtait 25 ms par bloc sur la cible — la
    moitié du budget ; cette voie en coûte 0,9 ms.

    Repli sur le RMS large bande si numpy est absent : le dispositif reste
    fonctionnel, simplement moins sélectif.
    """
    if np is None or len(block) < 4:
        return get_rms(block)
    x = np.frombuffer(block, dtype="<i2").astype(np.float32)
    spec = np.fft.rfft(x)[_BAND_LO:_BAND_HI]
    power = float((spec.real**2 + spec.imag**2).sum())
    return math.sqrt(2 * power) / len(x) * SHORT_NORMALIZE


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

    def readings(self) -> Iterator[tuple[datetime, "Spectrum"]]:
        """Flux infini de (instant, analyse spectrale), résilient aux coupures.

        L'amplitude sort en dBFS et non en RMS linéaire (ADR-0008), accompagnée
        de la forme du spectre — de quoi distinguer une voix d'un pleur, ce
        qu'un volume seul ne peut pas dire (ADR-0011).
        """
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
            yield datetime.now(), analyse(block)

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
        """Choisit le micro parmi les périphériques CAPABLES de capturer.

        Le filtre sur `maxInputChannels` est essentiel : la sortie casque
        intégrée du Pi s'appelle « bcm2835 Headphones » et un nom de
        périphérique de sortie peut contenir « mic » ou « input ». La
        sélectionner produirait un « Device unavailable » incompréhensible
        au lieu d'un message clair.
        """
        candidates = []
        for i in range(self._pa.get_device_count()):
            devinfo = self._pa.get_device_info_by_index(i)
            inputs = int(devinfo["maxInputChannels"])
            logging.info("Device %d: %s (entrées: %d)", i, devinfo["name"], inputs)
            if inputs > 0:
                candidates.append((i, devinfo["name"]))

        for keyword in ["mic", "input"]:
            for i, name in candidates:
                if keyword in name.lower():
                    logging.info("Found an input: device %d - %s", i, name)
                    return i

        if candidates:
            i, name = candidates[0]
            logging.warning("No preferred input found; falling back to %d - %s", i, name)
            return i

        raise RuntimeError(
            "Aucun périphérique de capture disponible : le micro USB est-il "
            "branché ? (vérifier avec `arecord -l`)"
        )

    def _open_mic_stream(self):
        return self._pa.open(
            format=FORMAT,
            channels=CHANNELS,
            rate=RATE,
            input=True,
            input_device_index=self._find_input_device(),
            frames_per_buffer=INPUT_FRAMES_PER_BLOCK,
        )
