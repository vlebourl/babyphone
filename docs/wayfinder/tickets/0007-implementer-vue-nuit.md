<!-- wayfinder:task | parent: MAP.md | blocked-by: 0006, 0001 -->
# Implémenter la vue « nuit dernière »

## Question

Réaliser la maquette retenue avec `apexcharts-card` (déjà installé) et les capteurs template nécessaires (heure d'endormissement, plus long segment calme, nombre de réveils).

## Critère de résolution

Vue déployée dans lovelace-mobile, alimentée par des données réelles, versionnée dans `deploy/homeassistant/`.

---
## Résolution (2026-08-15) — CLOS

Déployée, 2 sections, **12 entités référencées, aucune manquante**.

Effet de bord notable : Lovelace était la dernière pièce à échapper à l'ADR-0007 (elle vit dans le stockage interne de HA). Elle est désormais décrite par un script idempotent versionné, déployé par `deploy.sh` avec sauvegarde avant réécriture.
