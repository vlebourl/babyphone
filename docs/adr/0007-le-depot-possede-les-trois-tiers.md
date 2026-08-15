# Le dépôt possède les trois tiers du dispositif

Le babyphone n'est pas seulement le code qui tourne sur le Raspberry Pi : c'est un système à trois tiers — le **code de détection** sur le Pi, l'**unité systemd** qui le fait vivre, et le **package Home Assistant** qui transforme ses webhooks en entités et en écrans. Ces trois morceaux n'ont de sens qu'ensemble : renommer une clé JSON casse le package, changer la cadence de télémétrie casse le capteur de vivacité, ajouter une entité sans déployer le code la laisse vide.

Historiquement, seul le premier tiers vivait dans le dépôt. L'unité systemd n'existait que sur le Pi, le package HA que sur la domotique — invisibles, non versionnés, et découverts par accident. C'est ainsi qu'un `StandardOutput=append:` sans rotation a pu accumuler 4,1 Go de journaux sur une microSD sans que rien ne le signale, et que trois lanceurs concurrents ont coexisté pendant des mois ([ADR-0006](0006-lanceur-unique-systemd.md)).

**Le dépôt est désormais la source de vérité des trois.** `deploy/babyphone.service` et `deploy/homeassistant/babyphone_monitoring.yaml` sont versionnés au même titre que `detection.py`, et `deploy/deploy.sh` les pousse vers leurs cibles respectives. Les modifications se font ici, jamais directement sur les machines.

## Conséquences

- **Le déploiement fait partie du travail.** Une correction poussée sur GitHub mais absente du Pi ne protège personne. `deploy.sh` refuse de démarrer si le dépôt est sale ou en avance sur `origin/main` — le Pi tire depuis GitHub, pas depuis la machine de dev.
- **Le contrat filaire est le point de couplage à surveiller.** Les clés JSON (`speaking`, `noise_amplitude`, `threshold`) et la cadence d'émission sont lues par le package HA. Les changer impose de déployer les deux tiers ensemble ; c'est la raison d'être du seam de traduction dans `emitter.py` ([ADR-0003](0003-webhooks-domotique-sans-flux-audio.md)).
- **Ce qui reste hors du dépôt est délibéré et limité** : les URLs de webhook porteuses de secret (`local_settings.py` côté Pi, `lenaic_sleep.yaml` côté HA), et la configuration Lovelace, qui vit dans le stockage interne de Home Assistant et se modifie surtout à la souris.
- **Le paquet HA ne contient que ce qui appartient au babyphone.** Les entités de sommeil préexistantes restent dans `lenaic_sleep.yaml` ; on ne déplace pas ce qu'on ne possède pas.
