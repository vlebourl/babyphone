<!-- wayfinder:task | parent: MAP.md | blocked-by: — -->
# Trancher l'implémentation du passe-bande : numpy ou audioop

## Question

Le RMS large bande est dominé par les basses fréquences — ventilation, circulation, chauffage — c'est-à-dire tout sauf l'enfant. C'est ce qui explique la dynamique faible mesurée (p99/médiane = 1,5×).

Deux voies : **numpy** (biquad vectorisé, sélectif, mais ~40 Mo de RSS sur 512 Mo) ou **audioop** (passe-haut du premier ordre en C via `sub`/`mul`, gratuit mais grossier). Le CPU actuel est à 1,7 %, la marge existe — mais la RAM et la dépendance se paient.

Trancher **par la mesure sur la cible**, pas par principe.

## Critère de résolution

Une voie retenue, justifiée par des mesures de CPU et de RAM prises sur le Pi lui-même.
