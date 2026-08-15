<!-- wayfinder:task | parent: MAP.md | blocked-by: 0002 -->
# Passer la détection en dBFS

## Question

Convertir la chaîne de décision à l'échelle logarithmique avec l'offset retenu, sans dégrader la détection ni le budget CPU. Corrige au passage le `unit_of_measurement: "dB"` des capteurs HA, aujourd'hui mensonger.

Quelle est la surface exacte du changement — `Detection` seule, ou faut-il toucher la source audio et l'émetteur ?

## Critère de résolution

Détection en dBFS déployée, tests verts, capteurs HA en dB honnêtes, comportement validé sur données réelles.
