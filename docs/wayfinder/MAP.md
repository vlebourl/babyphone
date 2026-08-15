<!-- wayfinder:map -->
# Carte — Améliorations backend et visualisation du babyphone

## Destination

Les cinq améliorations issues de l'étude du 2026-08-15 — télémétrie enrichie, détection en dBFS, alertes de panne, vue « nuit dernière », passe-bande vocal — **implémentées, déployées en production sur les trois tiers et vérifiées en fonctionnement**.

## Notes

**Cette carte porte l'exécution, pas seulement la décision.** Override explicite du « Plan, don't do » de wayfinder : un ticket n'est résolu que lorsque son changement est déployé en prod et vérifié. Override également de « un seul ticket par session » — l'effort se mène en autonomie jusqu'au bout.

- **Domaine et vocabulaire** : `CONTEXT.md` (amplitude, seuil, événement de bruit, éveil). Décisions existantes : `docs/adr/0001` à `0007`.
- **Skills à consulter** : `codebase-design` (modules profonds, seams), `domain-modeling` (le glossaire évolue avec le code), `tdd` pour toute logique de décision.
- **Tickets de type grilling** : la contrepartie humaine est tenue par un **subagent adversarial indépendant**, mandaté pour attaquer la proposition. L'agent ne répond jamais à sa propre place.
- **Déploiement** : `./deploy/deploy.sh` (refuse un dépôt sale ou non poussé). Prod = Raspberry Pi 3A+ `babyphone.local` + Home Assistant `192.168.1.10`.
- **Garde-fous permanents** : les 31 tests doivent rester verts, et le différentiel contre le code pré-refactor reste la référence pour tout ce qui ne doit pas changer de comportement filaire.
- **Contrainte cible** : budget 50 ms par bloc, 512 Mo de RAM, microSD (ADR-0005). Toute régression de performance est un défaut fonctionnel.

## Decisions so far

- [Télémétrie enrichie : publier ce que le détecteur voit réellement](tickets/0001-telemetrie-enrichie.md) — `peak`, `floor`, `noisy_ratio` ajoutés (additif) ; capteurs de bruit rapatriés dans le package babyphone, `webhook_id` en `!secret`

- [Choisir l'offset en dB par rejeu des données historiques](tickets/0002-calibrer-offset-db.md) — **+10 dB** retenu : volume d'éveils identique, +20 % la nuit, −50 % de faux positifs le jour (rejeu de 78 532 échantillons)

- [Passer la détection en dBFS](tickets/0003-detection-dbfs.md) — conversion dans la source audio, `Detection` inchangée ; capteurs HA enfin honnêtes en dB (ADR-0008)

<!-- une ligne par ticket clos : le gist, puis zoomer le lien pour le détail -->

## Not yet specified

- **Réglage fin des temporisations de l'ADR-0002** une fois le passe-bande en place : si le rapport signal/bruit s'améliore vraiment, `min_noise_duration`, `event_count` et `event_gap` deviennent probablement trop conservateurs. Impossible de trancher avant d'avoir mesuré le gain réel sur plusieurs nuits.
- **Sort du terme filaire `speaking`** : le glossaire dit *éveil*, le contrat HA dit `speaking`. Un renommage coordonné des deux tiers est possible mais son déclencheur naturel serait un changement de contrat déjà nécessaire pour une autre raison.
- **Rétention et granularité de l'historique** : la télémétrie enrichie multiplie les séries dans TimescaleDB. À revisiter si le volume devient un sujet.

## Out of scope

<!-- travail consciemment écarté de cet effort -->
