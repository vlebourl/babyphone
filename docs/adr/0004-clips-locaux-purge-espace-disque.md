# Clips conservés localement, purgés par pression sur l'espace disque

À chaque entrée en éveil, un court clip audio est enregistré et conservé **sur la machine uniquement**, en MP3 horodaté. Il répond à une question que la télémétrie ne peut pas trancher : « c'était quoi, ce bruit à 2h ? ». Cohérent avec [ADR-0003](0003-webhooks-domotique-sans-flux-audio.md), l'audio ne quitte jamais l'appareil ; on l'écoute en s'y connectant.

La rétention n'est pas une durée mais une **pression disque** : tant qu'il reste de la place, les clips s'accumulent ; en dessous d'un seuil d'espace libre, les plus anciens sont supprimés. Sur un appareil dédié à demeure, on ne veut ni saturer le disque, ni jeter un clip encore utile parce qu'un calendrier l'a décidé.

Le stockage est une carte microSD ([ADR-0005](0005-cible-raspberry-pi-3a-plus.md)) : petite, lente, et seule ressource disque de la machine. Saturer la carte ne remplit pas juste un dossier de clips, ça met l'OS en panne — d'où une purge pilotée par l'espace libre global et non par la taille du dossier.

## Conséquences

- La durée de rétention est **imprévisible** — elle dépend de ce que le reste de la machine consomme. Il n'y a aucune garantie qu'un clip de la nuit dernière existe encore.
- L'horodatage dans le nom de fichier est ce qui définit « le plus ancien ». Renommer les clips casse l'ordre de purge.
- **Le seuil d'espace libre doit être dimensionné pour la carte réellement montée.** Un seuil supérieur à ce que la carte peut offrir rend la condition de purge toujours vraie : le mécanisme ne libère plus de place, il plafonne simplement le nombre de clips à sa valeur du moment, un supprimé pour un créé. À l'inverse, ne supprimer qu'un seul clip par éveil ne rattrape pas un disque déjà plein. Ce dimensionnement est un réglage de déploiement, pas une constante universelle.
- L'enregistrement d'un clip lit le même micro que la boucle d'écoute : pendant sa durée, aucune amplitude n'est mesurée ni envoyée. Un éveil crée donc un trou bref dans la télémétrie, juste au moment le plus intéressant. Corriger cela demanderait de dupliquer le flux audio plutôt que de le partager.
