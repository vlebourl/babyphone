"""Tests de l'adaptateur webhook, avec une session HTTP factice au seam transport."""

from datetime import datetime

import pytest

from detection import NoiseReport, Output, Transition
from emitter import WebhookEmitter, create_session


class FakeResponse:
    status_code = 200

    def raise_for_status(self):
        pass


class FakeSession:
    def __init__(self):
        self.calls = []

    def post(self, url, json=None, timeout=None):
        self.calls.append({"url": url, "json": json, "timeout": timeout})
        return FakeResponse()


T = datetime(2026, 1, 1, 2, 30, 0)


def test_la_telemetrie_part_avant_la_transition_dans_un_meme_bloc():
    # Ordre historique du dispositif : niveau sonore émis avant la décision.
    session = FakeSession()
    emitter = WebhookEmitter("http://ha/wake", "http://ha/noise", session=session)
    emitter.publish(
        Output(
            transitions=(Transition(awake=True, at=T, noise_duration=0.15),),
            noise_report=NoiseReport(amplitude=0.02, threshold=0.055),
        )
    )
    assert [c["url"] for c in session.calls] == ["http://ha/noise", "http://ha/wake"]


def test_le_vocabulaire_filaire_est_traduit_au_seam():
    # Le code interne dit `awake` ; le contrat avec la domotique dit "speaking".
    session = FakeSession()
    emitter = WebhookEmitter("http://ha/wake", "http://ha/noise", session=session)
    emitter.publish(
        Output(transitions=(Transition(awake=True, at=T, noise_duration=0.15),))
    )
    payload = session.calls[0]["json"]
    assert payload == {
        "speaking": True,
        "time": "2026-01-01T02:30:00",
        "noise": 0.15,
        "message": "",
    }


def test_chaque_post_est_borne_par_un_timeout():
    # Sans timeout, une domotique qui accepte la connexion sans répondre
    # gèlerait la boucle d'écoute pour toujours.
    session = FakeSession()
    emitter = WebhookEmitter("http://ha/wake", "http://ha/noise", session=session)
    emitter.publish(Output(noise_report=NoiseReport(amplitude=0.02, threshold=0.055)))
    assert session.calls[0]["timeout"] is not None


def test_les_retries_http_s_appliquent_bien_aux_post():
    # urllib3 exclut POST des retries par défaut : le status_forcelist
    # historique n'a jamais servi. Vérifie que la session le réactive.
    session = create_session()
    retries = session.get_adapter("http://192.168.1.10/").max_retries
    assert "POST" in {m.upper() for m in retries.allowed_methods}
    assert retries.total == 5
    assert 503 in retries.status_forcelist


def test_un_echec_reseau_ne_remonte_pas_dans_la_boucle():
    class ExplodingSession:
        def post(self, url, json=None, timeout=None):
            import requests

            raise requests.exceptions.ConnectionError("HA injoignable")

    emitter = WebhookEmitter("http://ha/wake", "http://ha/noise", session=ExplodingSession())
    # ne doit pas lever : l'écoute continue même domotique coupée
    emitter.publish(
        Output(transitions=(Transition(awake=True, at=T, noise_duration=0.15),))
    )
