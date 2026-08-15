"""Tests de l'étiquetage des sons (ADR-0011)."""

import pytest

from classification import (BRUIT, CALME, CRI, PLEURS, VOIX,
                            EMERGENCE_CRI_DB, EMERGENCE_MIN_DB, classify)


def test_sous_le_seuil_d_emergence_c_est_l_ambiance_pas_un_evenement():
    k = classify(1500, 0.2, 0.5, 0.3, emergence_db=EMERGENCE_MIN_DB - 1)
    assert k.label == CALME


def test_une_fondamentale_grave_est_une_voix():
    # Gazouillis, babil, parole : énergie surtout en 300-800 Hz.
    k = classify(600, 0.8, 0.15, 0.05, emergence_db=12)
    assert k.label == VOIX


def test_un_spectre_haut_et_un_medium_charge_sont_des_pleurs():
    # Les harmoniques qui rendent un pleur perçant vivent en 800-2000 Hz.
    k = classify(1600, 0.2, 0.55, 0.25, emergence_db=15)
    assert k.label == PLEURS


def test_un_centre_de_gravite_tres_haut_suffit_pour_des_pleurs():
    k = classify(2200, 0.1, 0.2, 0.7, emergence_db=15)
    assert k.label == PLEURS


def test_tres_fort_et_aigu_c_est_un_cri():
    k = classify(1700, 0.15, 0.5, 0.35, emergence_db=EMERGENCE_CRI_DB + 3)
    assert k.label == CRI


def test_un_choc_grave_sans_harmoniques_n_est_pas_une_voix():
    # Une porte qui claque est forte et grave, mais son énergie est massée
    # dans le bas sans les partiels qu'aurait une voix. L'attribuer à
    # l'enfant serait un faux positif de nature.
    k = classify(400, 0.97, 0.02, 0.01, emergence_db=30)
    assert k.label == BRUIT


def test_une_voix_grave_garde_ses_harmoniques_donc_reste_une_voix():
    k = classify(600, 0.80, 0.15, 0.05, emergence_db=12)
    assert k.label == VOIX


def test_un_son_fort_sans_signature_vocale_est_dit_bruit_pas_pleurs():
    # Assez fort pour compter, spectre plat entre les deux bornes, médium
    # pauvre : on le nomme plutôt que de l'attribuer à l'enfant.
    k = classify(1200, 0.45, 0.2, 0.35, emergence_db=12)
    assert k.label == BRUIT


def test_l_etiquette_transporte_ce_qui_l_a_justifiee():
    k = classify(1600, 0.2, 0.55, 0.25, emergence_db=15)
    assert k.emergence_db == 15
    assert k.centroid_hz == 1600


@pytest.mark.parametrize("centroid,low,mid,high,em", [
    (0, 0, 0, 0, 0), (4000, 0, 0, 1, 50), (300, 1, 0, 0, 6),
])
def test_aucune_entree_ne_fait_planter_l_etiquetage(centroid, low, mid, high, em):
    assert classify(centroid, low, mid, high, em).label in {
        CALME, VOIX, PLEURS, CRI, BRUIT}
