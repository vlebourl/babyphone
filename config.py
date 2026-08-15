"""
Configuration settings for the babyphone application.
"""

import logging

import pyaudio

# Audio settings
FORMAT = pyaudio.paInt16
SHORT_NORMALIZE = 1.0 / 32768.0
CHANNELS = 1
RATE = 48000  # Sample rate
INPUT_BLOCK_TIME = 0.05  # Time in seconds for each audio block

# Default API endpoints (will be overridden by secrets.py if available)
DEFAULT_URL = "http://localhost/api/webhook/babyphone"
DEFAULT_NOISE_URL = "http://localhost/api/webhook/noise-babyphone"

# Application settings
SPEAKING_TIMEOUT = 180  # seconds before considering silence
MIN_NOISE_DURATION = 0.11  # minimum noise duration to trigger an event
NOISE_EVENT_COUNT = 3  # number of noise events before considering speaking
NOISE_EVENT_TIMEOUT = 1.5  # seconds between noise events

# Chien de garde externe (ADR-0009). Vide = désactivé.
# Une alerte hébergée par le système surveillé ne peut pas signaler sa propre
# mort : si Home Assistant, le Wi-Fi ou le lien Internet tombent, personne
# n'est prévenu et le silence se confond avec « tout va bien ». Un tiers qui
# attend un battement régulier et alerte quand il cesse est le seul montage
# qui couvre ce cas. Le battement ne transporte AUCUNE donnée — ni audio, ni
# amplitude, ni horodatage d'éveil — ce qui reste compatible avec l'ADR-0003.
HEARTBEAT_URL = ""
HEARTBEAT_PERIOD = 60  # secondes entre deux battements

# Load local settings if available
try:
    from local_settings import NOISE_URL, URL
except ImportError:
    try:
        # Compat : ancien nom du fichier, qui masquait le module standard
        # `secrets` — à renommer en local_settings.py sur le déploiement
        from secrets import NOISE_URL, URL  # type: ignore[attr-defined]

        logging.warning(
            "secrets.py est déprécié (il masque le module standard `secrets`) : "
            "renommez-le en local_settings.py"
        )
    except ImportError:
        logging.warning("local_settings.py not found, using default API endpoints")
        URL = DEFAULT_URL
        NOISE_URL = DEFAULT_NOISE_URL

try:  # l'URL du chien de garde porte un jeton : jamais dans le dépôt
    from local_settings import HEARTBEAT_URL  # noqa: F811
except ImportError:
    pass
