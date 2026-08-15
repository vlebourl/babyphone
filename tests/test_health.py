"""Tests de la santé matérielle et du canal d'alerte hors bande."""

import pytest

from health import OutOfBandAlerter, read_undervoltage


def test_la_sous_tension_se_lit_en_sysfs(tmp_path):
    f = tmp_path / "alarm"
    f.write_text("1\n")
    assert read_undervoltage(str(f)) is True
    f.write_text("0\n")
    assert read_undervoltage(str(f)) is False


def test_hors_cible_la_sous_tension_est_inconnue_pas_fausse():
    # Sur la machine de dev le fichier n'existe pas. Rendre False laisserait
    # croire que l'alimentation est saine ; None dit « je ne sais pas ».
    assert read_undervoltage("/nexiste/pas") is None


class FakeSession:
    def __init__(self):
        self.gets = []

    def get(self, url, timeout=None):
        self.gets.append(url)


def test_inactif_sans_url():
    a = OutOfBandAlerter("", 180)
    assert a.enabled is False
    assert a.note_failure(1000.0, "peu importe") is False


def test_une_panne_courte_ne_declenche_pas():
    s = FakeSession()
    a = OutOfBandAlerter("http://sms/?msg={msg}", 180, session=s)
    a.note_failure(1000.0, "msg")   # début de la panne
    assert a.note_failure(1100.0, "msg") is False  # 100 s < 180 s
    assert s.gets == []


def test_une_panne_longue_crie_une_seule_fois():
    # Une coupure d'une nuit ne doit pas produire des milliers de messages.
    s = FakeSession()
    a = OutOfBandAlerter("http://sms/?msg={msg}", 180, session=s)
    a.note_failure(1000.0, "msg")
    assert a.note_failure(1200.0, "msg") is True   # 200 s > 180 s
    for t in range(1300, 5000, 100):
        assert a.note_failure(float(t), "msg") is False
    assert len(s.gets) == 1


def test_le_retour_de_la_domotique_rearme_le_canal():
    s = FakeSession()
    a = OutOfBandAlerter("http://sms/?msg={msg}", 180, session=s)
    a.note_failure(1000.0, "msg")
    a.note_failure(1200.0, "msg")          # crie
    a.note_success(1300.0)                 # la domotique revient
    a.note_failure(2000.0, "msg")          # nouvelle panne
    assert a.note_failure(2200.0, "msg") is True
    assert len(s.gets) == 2


def test_le_message_est_encode_pour_l_url():
    s = FakeSession()
    a = OutOfBandAlerter("http://sms/?msg={msg}", 0, session=s)
    a.note_failure(1000.0, "x")
    a.note_failure(1001.0, "chambre non surveillee")
    assert "%20" in s.gets[0] or "+" in s.gets[0]


def test_un_canal_de_secours_qui_echoue_ne_tue_jamais_la_boucle():
    class Exploding:
        def get(self, url, timeout=None):
            raise RuntimeError("réseau mort")

    a = OutOfBandAlerter("http://sms/?msg={msg}", 0, session=Exploding())
    a.note_failure(1000.0, "x")
    assert a.note_failure(1001.0, "x") is True  # ne doit pas lever
