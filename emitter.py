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
from classification import classify
from health import read_undervoltage


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
        out_of_band=None,
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
        # Canal de secours : prévenu de chaque succès et de chaque échec, il
        # décide seul quand la domotique est absente depuis trop longtemps.
        self._out_of_band = out_of_band

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
                    "undervoltage": read_undervoltage(),
                    "centroid_hz": round(r.centroid_hz, 0),
                    "band_low": round(r.low, 3),
                    "band_mid": round(r.mid, 3),
                    "band_high": round(r.high, 3),
                    "kind": classify(r.centroid_hz, r.low, r.mid, r.high,
                                     r.peak - r.floor).label,
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
            if self._out_of_band is not None:
                self._out_of_band.note_success(time.monotonic())
            # DEBUG et pas INFO : un POST par seconde, soit ~86 000 lignes par
            # jour sur une microSD (ADR-0005). Et l'URL porte le secret
            # d'authentification du webhook (ADR-0003) : l'écrire en clair à
            # chaque succès le recopie sans fin dans les journaux.
            logging.debug("Response status (%s): %s", _redact(url), response.status_code)
            return response
        except requests.exceptions.RequestException as e:
            logging.error("API request failed (%s): %s", _redact(url), e)
            if self._out_of_band is not None:
                self._out_of_band.note_failure(
                    time.monotonic(),
                    "Babyphone: domotique injoignable, la chambre n'est plus surveillee",
                )
            return None


class Heartbeat:
    """Battement vers un chien de garde externe (ADR-0009).

    Ne transporte aucune donnée : un GET vide, périodique. C'est le tiers qui
    alerte quand le battement cesse — donc il couvre les pannes que la
    domotique ne peut pas signaler, la sienne comprise.

    Inactif si l'URL est vide, ce qui est le cas par défaut : activer un
    service externe est une décision de l'utilisateur, pas du code.
    """

    def __init__(self, url: str, period: float, session=None):
        self._url = url
        self._period = period
        self._session = session or create_session(total=0, backoff_factor=0)
        self._last = None

    @property
    def enabled(self) -> bool:
        return bool(self._url)

    def beat(self, now: float) -> bool:
        """Envoie un battement si la période est écoulée. Rend True si envoyé."""
        if not self.enabled:
            return False
        if self._last is not None and now - self._last < self._period:
            return False
        self._last = now
        try:
            # timeout court : le battement ne doit jamais retarder l'écoute
            self._session.get(self._url, timeout=(2, 3))
        except requests.exceptions.RequestException as e:
            # un battement perdu est sans conséquence : le tiers tolère une
            # période de grâce, et le suivant repart dans une minute
            logging.debug("Heartbeat failed: %s", e)
        return True
