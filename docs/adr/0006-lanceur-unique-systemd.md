# systemd est le seul lanceur ; le code se protège par un verrou d'instance

Le micro USB n'accepte **qu'un seul client ALSA** : une seconde instance du babyphone meurt en `OSError: [Errno -9985] Device unavailable`. Trois lanceurs coexistaient sur la cible — un service systemd `babyphone.service` (`enabled`, `Restart=always`), un `shell_command` de la domotique qui ouvrait sa propre session SSH avec `nohup`, et les lancements manuels en développement. Chacun ignorait les autres, et `Restart=always` transformait le conflit en boucle de redémarrage silencieuse : le service relançait toutes les 3 secondes un processus condamné, pendant qu'une autre instance tenait le micro.

Deux décisions, à deux niveaux :

**systemd est le seul lanceur.** Il survit au redémarrage, redémarre après une panne, et journalise dans un seul endroit. La domotique ne lance plus son propre processus : elle appelle `systemctl start/stop babyphone` par SSH. Un seul propriétaire du cycle de vie, un seul endroit où regarder quand ça ne marche pas.

**Le code refuse de démarrer en double.** Un verrou `flock` exclusif non bloquant est pris avant toute ouverture du micro ; s'il est déjà détenu, le processus sort proprement avec un message actionnable au lieu de mourir sur une erreur ALSA cryptique. Le verrou est libéré par le noyau à la mort du processus, y compris sur `SIGKILL` — aucun fichier PID périmé à nettoyer.

## Conséquences

- Le verrou protège quel que soit le lanceur : un lancement manuel pendant que le service tourne échoue avec une phrase compréhensible, pas avec un mur de messages ALSA.
- `Restart=always` reste souhaitable (le mode de panne dominant est la coupure micro, [ADR-0005](0005-cible-raspberry-pi-3.md)), mais il n'est plus dangereux : une instance surnuméraire s'arrête d'elle-même au lieu de boucler.
- Le fichier de verrou vit dans `/tmp` : il disparaît au redémarrage, ce qui est le comportement voulu.
- Le diagnostic d'un babyphone muet commence désormais par `systemctl status babyphone` et `journalctl -u babyphone`, plus par la recherche de processus orphelins.
