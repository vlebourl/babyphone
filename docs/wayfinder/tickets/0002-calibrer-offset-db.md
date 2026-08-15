<!-- wayfinder:task | parent: MAP.md | blocked-by: 0001 -->
# Choisir l'offset en dB par rejeu des données historiques

## Question

L'offset additif `+0.05` rend le dispositif presque 3× moins sensible la nuit que le jour (nuit p10 : il faut 3,8× le fond ; jour p90 : 1,5×). Passer en dBFS rend le rapport constant — mais **quelle valeur** ?

Trop bas : des éveils sur la ventilation. Trop haut : on rate des pleurs naissants. Il faut trancher sur des données, pas à l'intuition : rejouer l'historique disponible et la nouvelle télémétrie, et mesurer le nombre d'éveils produits par chaque offset candidat.

## Critère de résolution

Une valeur d'offset en dB justifiée par une mesure sur données réelles, avec le tableau des candidats et leur effet sur le nombre d'éveils.
