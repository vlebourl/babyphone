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
        if output.noise_report is not None:
            self._post(
                self._noise_url,
                {
                    "noise_amplitude": output.noise_report.amplitude,
                    "threshold": output.noise_report.threshold,
                },
            )

        for t in output.transitions:
            json_data = {
                "speaking": t.awake,  # vocabulaire filaire, ne pas renommer sans HA
                "time": t.at.isoformat(),
                "noise": t.noise_duration,
                "message": t.message,
            }
            logging.info(f"transition d'éveil : {json_data}")
            self._post(self._url, json_data)

    @sleep_and_retry
    @limits(calls=1, period=1)
    def _post(self, url: str, json_data: dict):
        try:
            response = self._session.post(url, json=json_data, timeout=POST_TIMEOUT)
            response.raise_for_status()
            logging.info(f"Response status ({url}): {response.status_code}")
            return response
        except requests.exceptions.RequestException as e:
            logging.error(f"API request failed: {e}")
            return None
