<!-- wayfinder:task | parent: MAP.md | blocked-by: 0002 -->
# Passer la détection en dBFS

## Question

Convertir la chaîne de décision à l'échelle logarithmique avec l'offset retenu, sans dégrader la détection ni le budget CPU. Corrige au passage le `unit_of_measurement: "dB"` des capteurs HA, aujourd'hui mensonger.

Quelle est la surface exacte du changement — `Detection` seule, ou faut-il toucher la source audio et l'émetteur ?

## Critère de résolution

Détection en dBFS déployée, tests verts, capteurs HA en dB honnêtes, comportement validé sur données réelles.

---
## Résolution (2026-08-15) — CLOS

Conversion faite **en un seul endroit** : `audio_source.readings()` émet des dBFS, `Detection` ignore le changement d'unité (sa médiane glissante est indifférente à l'échelle). Surface minimale, seam respecté. Plancher à −120 dBFS contre `log10(0)`.

`Settings.threshold_offset = 10.0` dB. La constante linéaire `NOISE_THRESHOLD_ADJUSTMENT` disparaît de `config.py`.

**Vérifié en prod** : `noise_level = -32,3 dB`, `threshold = -16,4 dB`, `pic = -28,7 dB`, `fond = -35,2 dB`. L'`unit_of_measurement: "dB"` des capteurs est enfin exact.

Un test verrouille la propriété achetée : le même scénario translaté de 30 dB produit **les mêmes transitions aux mêmes instants**. ADR-0008 écrit, glossaire mis à jour (*Amplitude* en dBFS, nouveaux termes *Marge* et *Pic*).

**Différentiel retiré à ce point** : il garantissait l'absence de changement de comportement, or c'est désormais l'objectif. Le rejeu sur données réelles le remplace comme référence.
