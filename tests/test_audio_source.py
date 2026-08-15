"""Tests de la source d'amplitudes, sans matériel : flux scripté injecté."""

import struct

import pytest

from audio_source import INPUT_FRAMES_PER_BLOCK, MicrophoneSource, get_rms


def test_rms_d_un_bloc_vide_vaut_zero():
    # Un flux en cours de coupure peut rendre une lecture vide :
    # silence, pas de ZeroDivisionError.
    assert get_rms(b"") == 0.0


def test_rms_d_un_bloc_silencieux_vaut_zero():
    assert get_rms(b"\x00\x00" * INPUT_FRAMES_PER_BLOCK) == 0.0


def test_rms_est_normalise_entre_0_et_1():
    full_scale = struct.pack("<4h", 32767, -32767, 32767, -32767)
    assert get_rms(full_scale) == pytest.approx(1.0, abs=1e-3)


class ScriptedStream:
    """Rejoue une séquence de lectures : bytes → bloc rendu, Exception → levée."""

    def __init__(self, script):
        self.script = list(script)

    def read(self, n, exception_on_overflow=False):
        item = self.script.pop(0)
        if isinstance(item, Exception):
            raise item
        return item

    def close(self):
        pass


def make_source(script):
    """MicrophoneSource sans __init__ (pas de matériel), flux scripté."""
    src = object.__new__(MicrophoneSource)
    src._pa = None
    src._stream = ScriptedStream(script)
    src._error_count = 0
    src._resets = 0

    def fake_reset():
        src._resets += 1
        src._error_count = 0  # sans incidence, remis par l'appelant

    src._reset = fake_reset
    return src


BLOCK = b"\x00\x00" * 4


def test_des_erreurs_sporadiques_ne_declenchent_jamais_de_reset():
    # 12 erreurs isolées, chacune suivie d'une lecture réussie : le compteur
    # mesure des erreurs consécutives, pas un cumul sur la vie du processus.
    script = [IOError("glitch"), BLOCK] * 12
    src = make_source(script)
    readings = src.readings()
    for _ in range(12):
        next(readings)
    assert src._resets == 0


def test_six_erreurs_consecutives_declenchent_un_reset():
    script = [IOError("brownout")] * 6 + [BLOCK]
    src = make_source(script)
    next(src.readings())
    assert src._resets == 1
