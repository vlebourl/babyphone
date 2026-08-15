"""Machine à états de détection : des amplitudes entrent, des transitions d'éveil sortent.

Module pur — aucune E/S, aucune horloge interne, aucun état observable autre que
les valeurs retournées. L'interface est `feed(amplitude, now) -> Output` ; tout le
reste (seuil adaptatif médian, confirmation par accumulation, hystérésis, cadence
de télémétrie — voir docs/adr/0001 et 0002) est de l'implémentation.

Invariants d'interface :
- `amplitude` est dans [0, 1] (RMS normalisé, cf. CONTEXT.md) ;
- `now` est monotone croissant d'un appel à l'autre ;
- les appels arrivent à la cadence d'un bloc (`Settings.block_time`) — la fenêtre
  du seuil est dimensionnée en nombre de blocs, pas en temps réel ;
- les constantes de `Settings` sont couplées entre elles (ADR-0002) : ne pas les
  modifier isolément.
"""

from bisect import bisect_left, insort
from collections import deque
from dataclasses import dataclass
from datetime import datetime, timedelta
from statistics import mean


@dataclass(frozen=True)
class Settings:
    """Constantes de décision. Les défauts sont les valeurs de production."""

    block_time: float = 0.05  # durée d'un bloc (s)
    window_seconds: float = 120.0  # fenêtre de la médiane du seuil (s)
    threshold_offset: float = 0.05  # marge ajoutée à la médiane
    min_noise_duration: float = 0.11  # durée mini d'une salve pour compter comme événement (s)
    event_count: int = 3  # événements accumulés avant de pouvoir déclarer l'éveil
    event_gap: float = 1.5  # écart mini entre deux événements distincts (s)
    calm_timeout: float = 180.0  # silence continu avant retour au calme (s)
    report_period: float = 1.0  # cadence maxi des rapports de niveau sonore (s)
    report_window: float = 1.0  # fenêtre moyennée dans un rapport (s)


@dataclass(frozen=True)
class Transition:
    """Changement d'état d'éveil, à publier vers la domotique."""

    awake: bool
    at: datetime
    noise_duration: float  # durée de la salve qui a conclu la décision (s)
    message: str = ""


@dataclass(frozen=True)
class NoiseReport:
    """Télémétrie continue de la fenêtre écoulée.

    `amplitude` est la moyenne — utile pour l'ambiance, mais elle lisse
    précisément les pics qui déclenchent la décision : la décision se prend
    par bloc, la moyenne porte sur une vingtaine de blocs. `peak` et
    `noisy_ratio` sont ce qui rend la décision lisible depuis une courbe.
    """

    amplitude: float  # moyenne de la fenêtre (contrat filaire historique)
    threshold: float  # seuil courant, identique pour tous les blocs de la fenêtre
    peak: float = 0.0  # bloc le plus fort — ce que le détecteur a réellement vu
    floor: float = 0.0  # bloc le plus faible — le vrai fond sonore, non lissé
    noisy_ratio: float = 0.0  # part des blocs au-dessus du seuil, dans [0, 1]


@dataclass(frozen=True)
class Output:
    transitions: tuple[Transition, ...] = ()
    noise_report: "NoiseReport | None" = None


_EPOCH = datetime(1900, 1, 1)  # sentinelle : aucun événement encore observé


class Detection:
    """Décide de l'éveil à partir du flux d'amplitudes. État initial : calme."""

    def __init__(self, settings: Settings = Settings()):
        self._s = settings
        self._amplitudes: deque[float] = deque(
            maxlen=int(settings.window_seconds / settings.block_time)
        )
        # Miroir trié de la fenêtre, maintenu par insertions/suppressions
        # dichotomiques : la médiane devient une lecture O(1) au lieu d'un tri
        # complet de la fenêtre à chaque bloc — le budget CPU d'un bloc sur la
        # cible (ADR-0005) est de 50 ms, dépassement = overflow du flux audio.
        self._sorted_amplitudes: list[float] = []
        self._threshold = 0.0
        self._noisy_blocks = 0
        self._event_count = 0
        self._last_event_at = _EPOCH
        self._awake = False
        self._last_report_at: "datetime | None" = None

    @property
    def threshold(self) -> float:
        """Seuil courant (médiane de la fenêtre + marge)."""
        return self._threshold

    def feed(self, amplitude: float, now: datetime) -> Output:
        s = self._s
        # Le Pi n'a pas d'horloge RTC : au premier accrochage NTP après le boot,
        # l'heure peut reculer. Des repères dans le futur rendraient la détection
        # sourde (écarts négatifs) jusqu'à ce que l'heure les rattrape — on recale.
        if now < self._last_event_at:
            self._last_event_at = now
        if self._last_report_at is not None and now < self._last_report_at:
            self._last_report_at = now

        if len(self._amplitudes) == self._amplitudes.maxlen:
            evicted = self._amplitudes[0]  # va sortir de la fenêtre
            del self._sorted_amplitudes[bisect_left(self._sorted_amplitudes, evicted)]
        self._amplitudes.append(amplitude)
        insort(self._sorted_amplitudes, amplitude)
        self._threshold = self._median() + s.threshold_offset

        report = self._maybe_report(now)
        transitions: list[Transition] = []

        if amplitude > self._threshold:
            # bloc bruyant : la salve continue
            self._noisy_blocks += 1
        else:
            # bloc calme : la salve (éventuelle) se termine, on décide
            noise_duration = self._noisy_blocks * s.block_time

            if noise_duration >= s.min_noise_duration:
                # la salve est un événement de bruit
                if (
                    now - self._last_event_at > timedelta(seconds=s.event_gap)
                    and self._event_count >= s.event_count
                ):
                    if (t := self._set_awake(True, now, noise_duration)) is not None:
                        transitions.append(t)
                self._event_count += 1
                self._last_event_at = now

            if now - self._last_event_at > timedelta(seconds=s.calm_timeout):
                if (t := self._set_awake(False, now, noise_duration)) is not None:
                    transitions.append(t)
                self._event_count = 0

            self._noisy_blocks = 0

        return Output(transitions=tuple(transitions), noise_report=report)

    def _median(self) -> float:
        """Médiane de la fenêtre — mêmes conventions que statistics.median."""
        data = self._sorted_amplitudes
        n = len(data)
        if n % 2:
            return data[n // 2]
        return (data[n // 2 - 1] + data[n // 2]) / 2

    def _set_awake(self, awake: bool, now: datetime, noise_duration: float):
        if awake == self._awake:
            return None
        self._awake = awake
        return Transition(awake=awake, at=now, noise_duration=noise_duration)

    def _maybe_report(self, now: datetime):
        s = self._s
        if self._last_report_at is None:
            self._last_report_at = now  # amorce : premier rapport une période plus tard
            return None
        blocks = int(s.report_window / s.block_time)
        if len(self._amplitudes) < blocks:
            return None
        if now - self._last_report_at <= timedelta(seconds=s.report_period):
            return None
        self._last_report_at = now
        recent = list(self._amplitudes)[-blocks:]
        noisy = sum(1 for a in recent if a > self._threshold)
        return NoiseReport(
            amplitude=mean(recent),
            threshold=self._threshold,
            peak=max(recent),
            floor=min(recent),
            noisy_ratio=noisy / len(recent),
        )
