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

- [Décider la politique d'alerte quand le babyphone tombe](tickets/0004-politique-alerte-panne.md) — durcie par grilling adversarial : canal Android, armement sur l'intention, jamais sur l'état de sommeil (circulaire) — ADR-0009
- [Implémenter les alertes de panne et l'historique de fiabilité](tickets/0005-implementer-alertes.md) — 9 automatisations, détecteurs de cécité, boucle de redémarrage **vérifiée en réel**

- [Maquetter la vue « nuit dernière »](tickets/0006-maquette-vue-nuit.md) — chiffres, puis hypnogramme, puis courbe fine traçant le **pic**, puis 7 nuits
- [Implémenter la vue « nuit dernière »](tickets/0007-implementer-vue-nuit.md) — déployée ; Lovelace passe sous Git, dernière pièce hors ADR-0007
- [Trancher l'implémentation du passe-bande](tickets/0008-choix-passe-bande.md) — **FFT numpy** (0,87 ms) contre biquad Python (25,4 ms = 51 % du budget)
- [Implémenter le passe-bande vocal](tickets/0009-implementer-passe-bande.md) — déployé, émergence d'un cri **+8,8 dB**, 58 Mo / 4 % CPU

<!-- une ligne par ticket clos : le gist, puis zoomer le lien pour le détail -->

## Not yet specified

- **Recalibrer la marge de 10 dB sur données filtrées.** Elle a été choisie par rejeu de données large bande où le fond était dominé par du grondement. Le rapport signal/bruit ayant gagné 8,8 dB (ADR-0010), l'optimum a probablement bougé. Le rejeu du ticket 0002 se rejoue à l'identique dès que quelques nuits filtrées sont accumulées.
- **Réglage fin des temporisations de l'ADR-0002.** Elles compensaient un signal bruité ; avec 8,8 dB de marge en plus, `min_noise_duration` et `event_count` sont probablement trop conservateurs et coûtent de la latence pour rien.
- **Sort du terme filaire `speaking`** : le glossaire dit *éveil*, le contrat HA dit `speaking`. Un renommage coordonné des deux tiers est possible mais son déclencheur naturel serait un changement de contrat déjà nécessaire pour une autre raison.
- **Rétention et granularité de l'historique** : la télémétrie enrichie multiplie les séries dans TimescaleDB. À revisiter si le volume devient un sujet.

## Out of scope

- **Notification à chaque éveil** — écartée au ticket 0004 : l'ADR-0002 existe précisément pour ne pas crier au premier bruit, et rebrancher une sonnette derrière ce filtre le contredirait.
- **`uptime-card` comme mesure des micro-coupures** — écartée au ticket 0005 : le capteur de vivacité a 90 s de péremption et une minute de granularité, il ne peut structurellement pas montrer des coupures de 1 à 30 s.
- **Alimentation du Raspberry Pi** (`throttled=0x50005`, sous-tension et throttling en cours) — action matérielle, hors de portée du logiciel, mais préalable réel à toute calibration fine d'alerte.
- **Inscription à un prestataire de chien de garde externe** — le mécanisme est livré et inactif ; choisir un tiers et lui ouvrir un compte appartient à l'utilisateur.
