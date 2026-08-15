# Une panne du babyphone doit se voir, et l'alerte ne peut pas vivre uniquement dans la domotique

Un babyphone qui meurt est silencieux, et le silence se confond avec « tout va bien ». C'est le pire mode de panne d'un dispositif de sécurité : il ne se signale pas. La politique ci-dessous a été écrite contre un grilling adversarial, qui a invalidé trois hypothèses de la version initiale — toutes vérifiées en production avant correction.

## Ce qui alerte, et pourquoi ces seuils

**Panne franche — plus de télémétrie pendant 180 s.** Le déclencheur porte sur le capteur de niveau sonore, *pas* sur `binary_sensor.babyphone_en_ligne` : ce dernier ajoute 90 s de péremption plus la granularité d'une minute des templates contenant `now()`, soit 3,5 à 4,5 minutes de latence réelle pour un seuil qu'on croirait à 2 minutes. Les 180 s sont posés au-dessus du plus long rétablissement autonome — backoff audio (~61 s) + redémarrage systemd (~15 s) + réassociation Wi-Fi (~20 s) ≈ 96 s au pire — avec près du double en marge. En dessous, on alerte sur de l'auto-guérison ; au-delà de 300 s, la cécité devient inacceptable.

**Armement sur l'intention, jamais sur l'heure ni sur le sommeil.** Une sieste mérite la même vigilance qu'une nuit ; une fenêtre horaire créerait un trou en plein jour. `input_boolean.babyphone_on_off` déclare « je veux être surveillé », c'est le seul critère pertinent. Surtout, l'état de sommeil est **interdit comme condition** : `binary_sensor.lenaic_night_asleep` est construit sur `input_boolean.lenaic_speaking`, lui-même alimenté par le babyphone. Conditionner le détecteur de panne à la sortie de l'organe en panne est circulaire.

**L'interrupteur dégrade l'alerte, il ne la supprime jamais.** Il est exposé à l'assistant vocal, et le service systemd est `enabled` — après une coupure de courant, le dispositif démarre quel que soit son état. En faire le muet universel de la seule alerte de sécurité créerait un point de silence unique, actionnable à la voix, sans confirmation. Interrupteur sur « éteint » + dispositif muet donne donc une notification normale, une seule fois.

**Le canal doit être celui du système réel.** Les téléphones de la maison sont des Pixel : `interruption-level: critical` est un champ **iOS**, silencieusement ignoré sur Android. L'alerte serait partie, aurait semblé fonctionner, et n'aurait réveillé personne. Le canal correct est `alarm_stream_max`, qui joue sur le flux d'alarme et traverse le Ne pas déranger.

**Escalade jusqu'à un canal qui ne dépend de rien.** Quatre notifications à cinq minutes d'intervalle, second parent à partir de la deuxième, acquittement par bouton, annulation automatique au retour du dispositif. Sans réponse au bout de vingt minutes, la lumière de la chambre parentale passe à 100 % : le seul canal qui survive à un téléphone déchargé, un cloud injoignable ou un lien Internet coupé.

## « En ligne » ne veut pas dire « voyant »

Trois pannes laissent la télémétrie couler pendant que le dispositif est sourd. Elles ont chacune leur détecteur, en dBFS ([ADR-0008](0008-detection-en-dbfs.md)) :

- **Micro muet** (pic < −80 dB pendant 10 min) : du silence numérique, pas une chambre calme.
- **Fond masquant** (creux > −11 dB pendant 10 min) : un fond très élevé pousse la médiane, donc le seuil, si haut qu'un gémissement ne peut plus le franchir. Sourd par construction, tout en se déclarant sain.
- **Boucle de redémarrage** (≥ 3 démarrages en 15 min) : le plus dangereux. Chaque cycle réémet de la télémétrie, donc tous les voyants restent verts — mais la détection repart de zéro et l'accumulation exigée par l'[ADR-0002](0002-eveil-confirme-par-accumulation.md) devient mathématiquement inatteignable, pendant que chaque démarrage envoie un faux « calme ». Détecté par la durée de service publiée dans la télémétrie : une durée qui décroît ne peut signifier qu'un redémarrage.

## Le chien de garde externe

Une alerte hébergée par le système surveillé ne peut pas signaler sa propre mort. Si Home Assistant tombe, si le Wi-Fi tombe, si le lien Internet tombe, la chaîne d'alerte tombe avec eux — et l'absence de notification reste indiscernable de « tout va bien ». C'est une propriété structurelle, pas un défaut d'implémentation.

Le seul montage qui la corrige est un battement sortant vers un tiers, ce tiers alertant quand le battement cesse. Le mécanisme est implémenté et **désactivé par défaut** : `HEARTBEAT_URL` vide. Le battement ne transporte rien — pas d'audio, pas d'amplitude, pas d'horodatage d'éveil, un simple GET vide — ce qui reste compatible avec le refus de diffusion de l'[ADR-0003](0003-webhooks-domotique-sans-flux-audio.md), qui rejette le streaming et non la connectivité sortante.

**Choisir un prestataire et lui confier un compte est une décision de l'utilisateur, pas du code.** Tant qu'aucune URL n'est fournie, ce trou reste ouvert et assumé.

## Conséquences

- **Un préalable matériel non résolu conditionne tout le reste** : le Pi rapporte `throttled=0x50005`, c'est-à-dire sous-tension et throttling *en cours*. L'ADR-0005 nomme la sous-tension comme cause du mode de panne dominant. Aucun seuil d'alerte n'est pleinement calibrable tant que `vcgencmd get_throttled` ne rend pas `0x0` : l'alimentation se corrige avant qu'on discute des seuils.
- **Budget de faux positifs : au plus une alerte critique injustifiée par 90 nuits.** Au-delà, on désarme et on rouvre l'enquête — on ne monte pas le seuil pour masquer le symptôme.
- **La notification d'éveil a été écartée** : l'ADR-0002 existe précisément pour ne pas crier au premier bruit, et rebrancher une sonnette derrière ce filtre le contredirait.
- Le volume d'historique augmente avec la télémétrie enrichie ; `threshold` et le fond sonore méritent d'être exclus du recorder si le sujet devient sensible.
