"""Tests de la machine à états de détection, au seam `feed()` uniquement.

Chaque test rejoue un scénario sonore simulé (spécifications exécutables des
ADR-0001 et 0002) et n'observe que les valeurs retournées — jamais l'état
interne. Les amplitudes sont sur l'échelle RMS normalisée [0, 1].
"""

from datetime import datetime, timedelta

import pytest

from detection import Detection, Settings

S = Settings()  # valeurs de production
T0 = datetime(2026, 1, 1, 2, 0, 0)  # 2 h du matin, une nuit comme une autre

QUIET = 0.005  # fond sonore d'une chambre calme → seuil ≈ 0.055
LOUD = 0.5  # cri franc, très au-dessus du seuil


def quiet(seconds: float) -> list[float]:
    return [QUIET] * int(seconds / S.block_time)


def bang() -> list[float]:
    """Une salve juste assez longue pour compter comme événement de bruit
    (3 blocs = 0.15 s ≥ 0.11 s), close par un bloc calme qui la fait décider."""
    return [LOUD] * 3 + [QUIET]


class Scenario:
    """Déroule des amplitudes à la cadence d'un bloc et collecte les sorties."""

    def __init__(self):
        self.detection = Detection(S)
        self.now = T0
        self.transitions = []
        self.reports = []

    def play(self, amplitudes: list[float]) -> "Scenario":
        for amplitude in amplitudes:
            out = self.detection.feed(amplitude, self.now)
            self.transitions.extend(out.transitions)
            if out.noise_report is not None:
                self.reports.append(out.noise_report)
            self.now += timedelta(seconds=S.block_time)
        return self


# --- ADR-0002 : l'éveil est confirmé par accumulation, pas par un seul pic ---


def test_un_choc_isole_ne_reveille_pas():
    sc = Scenario().play(quiet(10) + bang() + quiet(30))
    assert sc.transitions == []


def test_des_chocs_espaces_de_plus_du_timeout_de_calme_ne_reveillent_jamais():
    # 5 portes qui claquent à 4 min d'intervalle : le compteur d'événements
    # est remis à zéro entre chaque, l'accumulation ne s'amorce jamais.
    sc = Scenario().play(quiet(10) + (bang() + quiet(240)) * 5)
    assert sc.transitions == []


def test_quatre_salves_rapprochees_reveillent_a_la_quatrieme():
    # Pleurs : salves espacées de ~5 s. La 4e déclare l'éveil
    # (3 événements accumulés + un événement déclencheur).
    sc = Scenario().play(quiet(10) + (bang() + quiet(5)) * 4)
    assert len(sc.transitions) == 1
    t = sc.transitions[0]
    assert t.awake is True
    assert t.noise_duration == pytest.approx(3 * S.block_time)


def test_l_accumulation_persiste_tant_que_le_calme_n_est_pas_declare():
    # Comportement réel (et discutable) du dispositif, capturé par le test :
    # des événements espacés de moins de 180 s s'accumulent indéfiniment —
    # 4 chocs isolés à 2 min d'intervalle finissent par déclarer un éveil.
    sc = Scenario().play(quiet(10) + (bang() + quiet(120)) * 4)
    assert [t.awake for t in sc.transitions] == [True]


def test_un_eveil_ne_produit_qu_une_transition_meme_si_le_bruit_continue():
    sc = Scenario().play(quiet(10) + (bang() + quiet(5)) * 12)
    assert [t.awake for t in sc.transitions] == [True]


# --- ADR-0002 : le retour au calme est volontairement lent ---


def test_pas_de_retour_au_calme_avant_le_timeout():
    sc = Scenario().play(quiet(10) + (bang() + quiet(5)) * 4)  # éveil déclaré
    sc.play(quiet(170))  # long silence, mais < 180 s
    assert [t.awake for t in sc.transitions] == [True]


def test_une_salve_pendant_le_silence_repousse_le_retour_au_calme():
    sc = Scenario().play(quiet(10) + (bang() + quiet(5)) * 4)
    sc.play(quiet(170) + bang() + quiet(170))  # jamais 180 s d'affilée
    assert [t.awake for t in sc.transitions] == [True]


def test_retour_au_calme_apres_le_timeout_plein():
    sc = Scenario().play(quiet(10) + (bang() + quiet(5)) * 4)
    sc.play(quiet(185))
    assert [t.awake for t in sc.transitions] == [True, False]


def test_apres_retour_au_calme_un_nouvel_eveil_doit_se_reconfirmer():
    sc = Scenario().play(quiet(10) + (bang() + quiet(5)) * 4)  # éveil
    sc.play(quiet(185))  # retour au calme
    before = len(sc.transitions)
    sc.play(bang() + quiet(5))  # un seul choc après le calme
    assert len(sc.transitions) == before  # pas de re-déclenchement immédiat
    sc.play((bang() + quiet(5)) * 3)  # ... il refaut l'accumulation complète
    assert [t.awake for t in sc.transitions] == [True, False, True]


# --- ADR-0001 : le seuil suit le fond sonore ---


def test_un_fond_sonore_qui_monte_lentement_ne_reveille_pas():
    # Le fond passe de 0.01 à 0.11 en ~4 min : la médiane suit, le seuil
    # monte avec elle, aucun bloc ne la dépasse jamais de plus de la marge.
    ramp = [0.01 + i * 2e-5 for i in range(5000)]
    sc = Scenario().play(quiet(10) + ramp)
    assert sc.transitions == []
    assert sc.detection.threshold > 0.12  # le seuil a bien suivi


def test_le_meme_cri_reveille_quel_que_soit_le_fond():
    # Dans une pièce bruyante (fond 0.15, seuil ≈ 0.2), un cri à 0.5
    # déclenche exactement comme dans une pièce calme.
    noisy_room = [0.15] * int(60 / S.block_time)
    sc = Scenario()
    sc.play(noisy_room)
    loud_bang = [LOUD] * 3 + [0.15]
    sc.play((loud_bang + [0.15] * 100) * 4)
    assert [t.awake for t in sc.transitions] == [True]


# --- Télémétrie : le niveau sonore, cadencé et moyenné ---


def test_au_plus_un_rapport_par_seconde():
    sc = Scenario().play(quiet(5))
    # amorce au premier bloc, puis un rapport par tranche de ~1.05 s
    assert len(sc.reports) == 4


def test_le_rapport_reflete_la_moyenne_recente_et_le_seuil():
    sc = Scenario().play(quiet(3))
    report = sc.reports[-1]
    assert report.amplitude == pytest.approx(QUIET)
    assert report.threshold == pytest.approx(QUIET + S.threshold_offset)


def test_les_rapports_continuent_pendant_un_eveil():
    sc = Scenario().play(quiet(10) + (bang() + quiet(5)) * 4)
    before = len(sc.reports)
    sc.play(quiet(10))
    assert len(sc.reports) > before
