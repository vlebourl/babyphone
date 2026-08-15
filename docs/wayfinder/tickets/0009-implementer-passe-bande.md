<!-- wayfinder:task | parent: MAP.md | blocked-by: 0008, 0003 -->
# Implémenter le passe-bande vocal

## Question

Insérer le filtre dans `audio_source.py`, en amont du RMS et invisible pour `Detection`. Vérifier le gain réel de rapport signal/bruit et l'absence de régression du budget 50 ms par bloc.

## Critère de résolution

Filtre déployé en prod, gain de dynamique mesuré avant/après, CPU et RAM dans le budget de la cible.
