<!-- wayfinder:task | parent: MAP.md | blocked-by: 0008, 0003 -->
# Implémenter le passe-bande vocal

## Question

Insérer le filtre dans `audio_source.py`, en amont du RMS et invisible pour `Detection`. Vérifier le gain réel de rapport signal/bruit et l'absence de régression du budget 50 ms par bloc.

## Critère de résolution

Filtre déployé en prod, gain de dynamique mesuré avant/après, CPU et RAM dans le budget de la cible.

---
## Résolution (2026-08-15) — CLOS

Déployé. **Vérifié en prod** : Pi à jour, service actif, **58 Mo de RAM et 4,0 % de CPU** (contre 46 Mo / 1,7 % avant) — très en dessous du budget.

Gain mesuré sur son réel de la chambre : émergence d'un cri **6,4 dB → 15,2 dB**, soit +8,8 dB. Atténuation vérifiée : −89 dB à 60 Hz, −24 dB à 150 Hz, bande passante intacte de 300 à 4000 Hz.

**Correction en cours de route** : ma première mesure concluait que le filtrage *dégradait* la dynamique. Elle était faite sur un échantillon **sans aucun pleur** — elle mesurait la fluctuation ambiante, pas la discrimination d'un cri. L'expérience refaite avec un cri superposé a inversé la conclusion.
