"""Santé matérielle de la cible, et alerte hors bande.

Deux préoccupations que la domotique ne peut pas couvrir toute seule :

- La **sous-tension** du Raspberry Pi est la cause du mode de panne dominant
  (coupures du micro USB, ADR-0005). Elle est invisible depuis le réseau : le
  dispositif paraît sain jusqu'à ce qu'il décroche. La publier la rend
  diagnosticable avant la panne plutôt qu'après.

- L'**alerte hors bande** existe parce qu'une alerte hébergée par le système
  surveillé ne peut pas signaler sa propre mort (ADR-0009). Si la domotique
  devient injoignable, c'est au dispositif de crier — par un chemin qui ne
  traverse ni Home Assistant, ni le réseau local.
"""

import logging

# Le noyau expose l'alarme de sous-tension en sysfs. La lire ne coûte rien,
# là où `vcgencmd get_throttled` demande un fork de 10 ms — inacceptable dans
# une boucle qui dispose de 50 ms par bloc.
UNDERVOLTAGE_PATH = "/sys/class/hwmon/hwmon1/in0_lcrit_alarm"


def read_undervoltage(path: str = UNDERVOLTAGE_PATH) -> "bool | None":
    """Alarme de sous-tension, ou None hors de la cible (machine de dev)."""
    try:
        with open(path) as f:
            return f.read().strip() == "1"
    except OSError:
        return None


class OutOfBandAlerter:
    """Crie par un chemin qui ne traverse pas la domotique.

    Compte le temps pendant lequel la domotique reste injoignable et, au-delà
    d'un seuil, envoie **une seule** alerte — pas une par échec, sans quoi une
    coupure d'une nuit produirait des milliers de messages. Le compteur se
    réarme dès que la domotique répond à nouveau.

    Inactif si l'URL est vide, ce qui est le cas par défaut.
    """

    def __init__(self, url_template: str, after_seconds: float = 180.0, session=None):
        self._template = url_template
        self._after = after_seconds
        self._session = session
        self._failing_since = None
        self._alerted = False

    @property
    def enabled(self) -> bool:
        return bool(self._template)

    def note_success(self, now: float) -> None:
        """La domotique répond : on oublie la panne en cours."""
        self._failing_since = None
        self._alerted = False

    def note_failure(self, now: float, message: str) -> bool:
        """Un envoi a échoué. Rend True si une alerte vient d'être émise."""
        if not self.enabled or self._alerted:
            return False
        if self._failing_since is None:
            self._failing_since = now
            return False
        if now - self._failing_since < self._after:
            return False
        self._alerted = True
        self._send(message)
        return True

    def _send(self, message: str) -> None:
        if self._session is None:  # import tardif : health.py reste sans requests
            from emitter import create_session

            self._session = create_session(total=1, backoff_factor=1)
        try:
            from urllib.parse import quote

            self._session.get(self._template.format(msg=quote(message)), timeout=(5, 10))
            logging.warning("Alerte hors bande émise : %s", message)
        except Exception as e:  # noqa: BLE001 - un canal de secours ne doit jamais tuer
            logging.error("Alerte hors bande impossible : %s", e)
