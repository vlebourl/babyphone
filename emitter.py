"""Émission vers la domotique : adaptateur webhook du port `publish(output)`.

Seul endroit du code où vit le vocabulaire filaire : la clé JSON `"speaking"`
est le contrat établi avec la domotique (ADR-0003) ; partout ailleurs, le code
parle d'éveil (`awake`, cf. CONTEXT.md). En test, une simple liste de captures
remplit le même port.
"""

import logging

import requests
from ratelimit import limits, sleep_and_retry
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from detection import Output


# (connexion, lecture) en secondes : borne le blocage de la boucle d'écoute
# quand la domotique accepte la connexion mais ne répond pas
POST_TIMEOUT = (3.05, 10)


def _redact(url: str) -> str:
    """Masque le secret que porte l'URL de webhook avant de la journaliser."""
    head, sep, _ = url.partition("/api/webhook/")
    return f"{head}{sep}***" if sep else url


def create_session() -> requests.Session:
    session = requests.Session()
    retries = Retry(
        total=5,
        backoff_factor=1,
        status_forcelist=[502, 503, 504],
        # par défaut urllib3 exclut POST des retries : sans ceci,
        # status_forcelist est lettre morte pour nos webhooks
        allowed_methods=frozenset(["POST"]),
    )
    adapter = HTTPAdapter(pool_connections=1, pool_maxsize=10, max_retries=retries)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    return session


class WebhookEmitter:
    """Publie transitions d'éveil et niveaux sonores en POST sortants, avec
    limitation de débit (1 req/s, partagée entre les deux webhooks) et retries."""

    def __init__(self, url: str, noise_url: str, session: "requests.Session | None" = None):
        self._url = url
        self._noise_url = noise_url
        self._session = session or create_session()

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
                },
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
    def _post(self, url: str, json_data: dict):
        try:
            response = self._session.post(url, json=json_data, timeout=POST_TIMEOUT)
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
