"""Émission vers la domotique : adaptateur webhook du port `publish(output)`.

Seul endroit du code où vit le vocabulaire filaire : la clé JSON `"speaking"`
est le contrat établi avec la domotique (ADR-0003) ; partout ailleurs, le code
parle d'éveil (`awake`, cf. CONTEXT.md). En test, une simple liste de captures
remplit le même port.
"""

import logging
import time

import requests
from ratelimit import limits, sleep_and_retry
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from detection import Output


# Les deux flux n'ont pas la même valeur, donc pas la même obstination.
#
# La télémétrie est jetable : un rapport de niveau perdu est remplacé par le
# suivant une seconde plus tard. Insister dessus n'apporte rien et coûte cher —
# les envois sont synchrones dans la boucle d'écoute (dette de l'ADR-0003),
# donc chaque seconde d'attente est une seconde où le micro n'est PAS lu.
#
# Une transition d'éveil, elle, ne repassera pas : elle n'est émise qu'au
# changement d'état, une notification perdue laisse la domotique désynchronisée
# jusqu'à la suivante. Elle mérite quelques tentatives.
#
# Budget de blocage au pire : ~5 s pour la télémétrie, ~19 s pour une
# transition. Sans cette séparation, une domotique injoignable bloquait la
# boucle ~90 s — assez pour que le dispositif se déclare lui-même hors ligne,
# et surtout assez pour être sourd pendant une minute et demie.
TELEMETRY_TIMEOUT = (2, 3)
TRANSITION_TIMEOUT = (2, 4)


def _redact(url: str) -> str:
    """Masque le secret que porte l'URL de webhook avant de la journaliser."""
    head, sep, _ = url.partition("/api/webhook/")
    return f"{head}{sep}***" if sep else url


def create_session(total: int, backoff_factor: float) -> requests.Session:
    session = requests.Session()
    retries = Retry(
        total=total,
        backoff_factor=backoff_factor,
        status_forcelist=[502, 503, 504],
        # par défaut urllib3 exclut POST des retries : sans ceci,
        # status_forcelist est lettre morte pour nos webhooks
        allowed_methods=frozenset(["POST"]),
    )
    adapter = HTTPAdapter(pool_connections=1, pool_maxsize=2, max_retries=retries)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    return session


class WebhookEmitter:
    """Publie transitions d'éveil et niveaux sonores en POST sortants, avec
    limitation de débit (1 req/s, partagée entre les deux webhooks) et retries."""

    def __init__(
        self,
        url: str,
        noise_url: str,
        session: "requests.Session | None" = None,
        telemetry_session: "requests.Session | None" = None,
    ):
        self._url = url
        self._noise_url = noise_url
        # transitions : quelques tentatives, elles ne repasseront pas
        self._session = session or create_session(total=2, backoff_factor=0.3)
        # télémétrie : aucune tentative, la suivante arrive dans une seconde
        self._telemetry_session = telemetry_session or session or create_session(
            total=0, backoff_factor=0
        )
        # Publier la durée de vie du processus est le seul moyen fiable de
        # détecter une boucle de redémarrage depuis la domotique : un service
        # qui redémarre toutes les quelques secondes continue d'émettre de la
        # télémétrie, donc paraît « en ligne », alors que la détection repart
        # de zéro à chaque cycle et ne peut plus jamais confirmer un éveil
        # (ADR-0002). Une durée qui DÉCROÎT trahit le redémarrage.
        self._started_at = time.monotonic()

    def publish(self, output: Output) -> None:
        # télémétrie d'abord, transitions ensuite : ordre historique du dispositif
        # (le niveau sonore était émis avant la décision dans chaque bloc)
        if (r := output.noise_report) is not None:
            # `noise_amplitude` et `threshold` sont le contrat historique avec
            # la domotique : ne pas renommer sans déployer les deux tiers
            # ensemble (ADR-0007). Les autres clés sont purement additives.
            self._post(
                self._noise_url,
                {
                    "noise_amplitude": r.amplitude,
                    "threshold": r.threshold,
                    "peak": r.peak,
                    "floor": r.floor,
                    "noisy_ratio": r.noisy_ratio,
                    "uptime_s": round(time.monotonic() - self._started_at, 1),
                },
                session=self._telemetry_session,
                timeout=TELEMETRY_TIMEOUT,
            )

        for t in output.transitions:
            json_data = {
                "speaking": t.awake,  # vocabulaire filaire, ne pas renommer sans HA
                "time": t.at.isoformat(),
                "noise": t.noise_duration,
                "message": t.message,
            }
            # rare et précieux : c'est la trace qu'on relit pour comprendre une nuit
            logging.info("transition d'éveil : %s", json_data)
            self._post(self._url, json_data)

    @sleep_and_retry
    @limits(calls=1, period=1)
    def _post(self, url: str, json_data: dict, session=None, timeout=None):
        session = session or self._session
        try:
            response = session.post(
                url, json=json_data, timeout=timeout or TRANSITION_TIMEOUT
            )
            response.raise_for_status()
            # DEBUG et pas INFO : un POST par seconde, soit ~86 000 lignes par
            # jour sur une microSD (ADR-0005). Et l'URL porte le secret
            # d'authentification du webhook (ADR-0003) : l'écrire en clair à
            # chaque succès le recopie sans fin dans les journaux.
            logging.debug("Response status (%s): %s", _redact(url), response.status_code)
            return response
        except requests.exceptions.RequestException as e:
            logging.error("API request failed (%s): %s", _redact(url), e)
            return None
