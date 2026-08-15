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


def test_le_chemin_audioop_et_le_repli_python_donnent_le_meme_rms(monkeypatch):
    # L'optimisation C (audioop) et le repli pur Python doivent coïncider
    # à l'arrondi entier d'audioop près (1/32768), sur des blocs réalistes.
    import random

    import audio_source

    if audio_source.audioop is None:
        pytest.skip("audioop absent de ce Python")

    rng = random.Random(7)
    for _ in range(20):
        samples = [rng.randint(-32768, 32767) for _ in range(2400)]
        block = struct.pack("<2400h", *samples)
        fast = get_rms(block)
        monkeypatch.setattr(audio_source, "audioop", None)
        slow = get_rms(block)
        monkeypatch.undo()
        assert fast == pytest.approx(slow, abs=1.5 / 32768)


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


# --- Bande vocale (ADR-0010) ---

def _tone(freq, amplitude=8000, n=INPUT_FRAMES_PER_BLOCK, rate=48000):
    import math
    return struct.pack(
        f"<{n}h", *[int(amplitude * math.sin(2 * math.pi * freq * i / rate)) for i in range(n)]
    )


def test_un_son_dans_la_bande_vocale_est_mesure_comme_le_large_bande():
    from audio_source import band_rms, get_rms

    tone = _tone(700)  # au cœur de la bande 300-4000
    assert band_rms(tone) == pytest.approx(get_rms(tone), rel=0.02)


def test_un_grondement_basse_frequence_est_massivement_attenue():
    # 94 % de l'énergie du fond de la chambre vit sous 300 Hz. C'est
    # exactement ce que la mesure de bande doit cesser de voir.
    from audio_source import band_rms, get_rms

    rumble = _tone(60)
    assert get_rms(rumble) > 0.15  # bien présent en large bande
    assert band_rms(rumble) < get_rms(rumble) / 100  # au moins 40 dB plus bas


def test_un_cri_emerge_mieux_d_un_grondement_apres_selection_de_bande():
    """La propriété que l'ADR-0010 achète, sur signal synthétique.

    Fond de grondement + cri dans la bande vocale : l'écart entre « avec
    cri » et « sans cri » doit être nettement plus grand en bande vocale.
    """
    import math

    from audio_source import band_rms, get_rms

    n = INPUT_FRAMES_PER_BLOCK
    rumble = [12000 * math.sin(2 * math.pi * 60 * i / 48000) for i in range(n)]
    cry = [4000 * math.sin(2 * math.pi * 800 * i / 48000) for i in range(n)]
    fond = struct.pack(f"<{n}h", *[int(v) for v in rumble])
    avec = struct.pack(f"<{n}h", *[int(a + b) for a, b in zip(rumble, cry)])

    def emergence(f):
        return 20 * math.log10(f(avec) / f(fond))

    assert emergence(band_rms) > emergence(get_rms) + 10


def test_repli_sur_le_large_bande_si_numpy_absent(monkeypatch):
    import audio_source
    from audio_source import band_rms, get_rms

    tone = _tone(700)
    monkeypatch.setattr(audio_source, "np", None)
    assert band_rms(tone) == pytest.approx(get_rms(tone))
