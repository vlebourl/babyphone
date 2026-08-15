<!-- wayfinder:task | parent: MAP.md | blocked-by: — -->
# Trancher l'implémentation du passe-bande : numpy ou audioop

## Question

Le RMS large bande est dominé par les basses fréquences — ventilation, circulation, chauffage — c'est-à-dire tout sauf l'enfant. C'est ce qui explique la dynamique faible mesurée (p99/médiane = 1,5×).

Deux voies : **numpy** (biquad vectorisé, sélectif, mais ~40 Mo de RSS sur 512 Mo) ou **audioop** (passe-haut du premier ordre en C via `sub`/`mul`, gratuit mais grossier). Le CPU actuel est à 1,7 %, la marge existe — mais la RAM et la dépendance se paient.

Trancher **par la mesure sur la cible**, pas par principe.

## Critère de résolution

Une voie retenue, justifiée par des mesures de CPU et de RAM prises sur le Pi lui-même.

---
## Résolution (2026-08-15) — CLOS

**Ni numpy-biquad ni audioop : FFT numpy.** On ne veut pas le signal filtré, seulement son énergie dans la bande — que Parseval donne directement depuis le spectre, sans récursion ni état.

Mesures **sur la cible** :

| Voie | ms/bloc | Budget 50 ms |
|---|---|---|
| RMS large bande (avant) | 0,05 | 0,1 % |
| Biquad Python pur | 25,4 | **51 %** |
| audioop `sub` | — | l'API n'existe pas |
| **FFT numpy** | **0,87** | **1,7 %** |

Le biquad Python consommait la moitié du budget sur une machine qui throttle déjà — écarté. numpy coûte 25 Mo de RSS, acceptable sur 512 Mo.
