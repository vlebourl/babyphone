<!-- wayfinder:task | parent: MAP.md | blocked-by: 0001 -->
# Choisir l'offset en dB par rejeu des données historiques

## Question

L'offset additif `+0.05` rend le dispositif presque 3× moins sensible la nuit que le jour (nuit p10 : il faut 3,8× le fond ; jour p90 : 1,5×). Passer en dBFS rend le rapport constant — mais **quelle valeur** ?

Trop bas : des éveils sur la ventilation. Trop haut : on rate des pleurs naissants. Il faut trancher sur des données, pas à l'intuition : rejouer l'historique disponible et la nouvelle télémétrie, et mesurer le nombre d'éveils produits par chaque offset candidat.

## Critère de résolution

Une valeur d'offset en dB justifiée par une mesure sur données réelles, avec le tableau des candidats et leur effet sur le nombre d'éveils.

---
## Résolution (2026-08-15) — CLOS

**Offset retenu : +10 dB** (seuil = 3,16× l'énergie du fond, quel que soit l'ambiant).

Méthode : rejeu de **78 532 échantillons réels** couvrant 24 h complètes (nuit incluse), en alimentant la classe `Detection` de production avec des valeurs en dB — médiane et logarithme commutent, donc un offset additif en dB *est* le mode multiplicatif, ce qui valide l'implémentation à venir en même temps que la valeur.

| Offset | Éveils nuit | Éveils jour | Total |
|---|---|---|---|
| Actuel (+0,05 linéaire) | 10 | 4 | 14 |
| 8 dB | 16 | 9 | 25 |
| 9 dB | 13 | 5 | 18 |
| **10 dB** | **12** | **2** | **14** |
| 11 dB | 11 | 1 | 12 |

À **volume d'éveils identique** (14), 10 dB redistribue : +20 % de détection la nuit, −50 % de faux positifs le jour. Aucun choc pour la maison, meilleure détection là où elle compte.

**Correction d'une estimation antérieure.** J'avais annoncé une asymétrie nuit/jour de ~2,6×, calculée sur les percentiles p10/p90 de la distribution globale. Mesurée correctement sur la médiane glissante — celle qu'utilise réellement le seuil — l'asymétrie est de **1,33×** (8,8 dB la nuit contre 6,3 dB le jour). Le défaut est réel et vaut d'être corrigé, mais il est moins spectaculaire que je ne l'ai écrit.

**Limite assumée** : le rejeu porte sur des moyennes à 1 s, alors que la production décide sur des blocs de 50 ms. Les comptages absolus ne sont pas transposables ; la comparaison entre offsets, elle, est valide — c'est ce qui fonde le choix.
