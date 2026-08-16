"""Nature d'un son : gazouillis, paroles, pleurs, cri.

Module pur — aucune E/S, aucun état. L'entrée est une analyse spectrale et
l'émergence du son au-dessus du fond de la pièce ; la sortie est une étiquette.

Le volume seul ne peut pas distinguer une porte qui claque d'un pleur : les
deux sont forts. C'est la **forme** du spectre qui les sépare (ADR-0011). Un
pleur de nourrisson concentre son énergie vers 1–3 kHz, avec un centre de
gravité spectral haut ; la parole reste plus grave et mieux répartie ; un choc
est bref et large bande, donc son centre de gravité tombe entre les deux sans
que l'énergie se concentre dans le médium.

**Le centre de gravité ne suffit jamais à déclarer un pleur.** Les consonnes
sifflantes de la parole — « ch », « s », « f » — ont un centre de gravité très
haut, souvent au-dessus de 2 kHz, avec presque toute leur énergie dans l'aigu et
un médium famélique. Un pleur, lui, charge le médium : ce sont ses harmoniques
qui le rendent perçant. C'est donc la part de médium, et elle seule, qui
distingue les deux. Mesuré le 2026-08-16 : une lecture d'histoire à voix haute
produisait 4,3 fausses étiquettes « pleurs » par minute, contre zéro pendant le
silence qui précédait.

Les seuils ci-dessous sont un **premier jeu**, posé sur la littérature acoustique
et vérifié sur signaux de synthèse. Ils demandent une calibration sur plusieurs
nuits réelles avant d'être tenus pour justes — d'où la publication des grandeurs
brutes (centre de gravité, parts de bandes) à côté de l'étiquette : on peut
recalibrer sans redéployer de code.
"""

from dataclasses import dataclass

# Émergence minimale au-dessus du fond pour qu'un son mérite une étiquette.
# Calée sur la MARGE DE DÉTECTION (ADR-0008) : en deçà, le bloc n'est même pas
# jugé bruyant par la machine à états. Aligner les deux évite qu'un son soit
# nommé « voix » alors que la surveillance le considère comme du silence — et
# supprime le clignotement calme/voix à chaque seconde qui rendait l'étiquette
# illisible sur le tableau de bord.
EMERGENCE_MIN_DB = 10.0

# Un cri se reconnaît d'abord à sa violence, quelle que soit sa couleur.
EMERGENCE_CRI_DB = 22.0

# Frontière de centre de gravité spectral, en hertz.
CENTROID_VOIX_HZ = 900.0  # en dessous : fondamentale grave, plutôt de la voix

# Part minimale d'énergie dans le médium (800–2000 Hz) pour un pleur : c'est là
# que vivent les harmoniques qui rendent un pleur perçant. C'est le SEUL
# discriminant du pleur : un centre de gravité haut ne suffit pas, parce qu'une
# consonne sifflante en a un aussi (voir la note sur les sifflantes en tête de
# module).
MID_PLEUR = 0.30

# Une voix humaine est harmonique : sa fondamentale est toujours accompagnée de
# partiels qui débordent dans le médium. Une énergie massée presque entièrement
# dans le grave, sans ce prolongement, n'est pas une voix — c'est un choc, un
# meuble, un pas. Le distinguer évite d'attribuer à l'enfant un bruit de maison.
LOW_SANS_HARMONIQUES = 0.85
MID_VOIX_MIN = 0.05

CALME = "calme"
VOIX = "voix"
PLEURS = "pleurs"
CRI = "cri"
BRUIT = "bruit"


@dataclass(frozen=True)
class Kind:
    """Étiquette d'un son, et ce qui a permis de la poser."""

    label: str
    emergence_db: float
    centroid_hz: float


def classify(centroid_hz: float, low: float, mid: float, high: float,
             emergence_db: float) -> Kind:
    """Étiquette un son à partir de sa forme spectrale et de son émergence.

    `low`, `mid`, `high` sont les parts d'énergie des trois sous-bandes, de
    somme 1. `emergence_db` est l'écart entre le son et le fond de la pièce.
    """
    if emergence_db < EMERGENCE_MIN_DB:
        return Kind(CALME, emergence_db, centroid_hz)

    # Très fort ET aigu : un cri. La conjonction compte — un choc violent est
    # fort mais son énergie ne se concentre pas dans le haut du spectre.
    if emergence_db >= EMERGENCE_CRI_DB and centroid_hz >= CENTROID_VOIX_HZ:
        return Kind(CRI, emergence_db, centroid_hz)

    if centroid_hz < CENTROID_VOIX_HZ:
        if low >= LOW_SANS_HARMONIQUES and mid < MID_VOIX_MIN:
            # grave, massif, sans harmoniques : un choc, pas quelqu'un
            return Kind(BRUIT, emergence_db, centroid_hz)
        return Kind(VOIX, emergence_db, centroid_hz)

    if mid >= MID_PLEUR:
        return Kind(PLEURS, emergence_db, centroid_hz)

    # Assez fort pour compter, mais ne ressemble à aucune voix : porte, meuble,
    # circulation. On le dit plutôt que de l'attribuer à l'enfant.
    return Kind(BRUIT, emergence_db, centroid_hz)
