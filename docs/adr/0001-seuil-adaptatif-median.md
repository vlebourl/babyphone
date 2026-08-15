# Seuil adaptatif fondé sur la médiane glissante

Le fond sonore d'une chambre varie sur la journée (ventilation, fenêtre ouverte, rue), et un seuil fixe donne soit des faux éveils l'après-midi, soit une surdité totale la nuit. Le seuil est donc recalculé à chaque bloc comme la médiane des amplitudes des dernières minutes, plus une marge constante : il suit le fond sonore, et seul ce qui dépasse *l'ambiance du moment* compte comme bruit.

## Options envisagées

- **Seuil fixe calibré à la main** — rejeté : demande un recalibrage à chaque changement d'ambiance, et c'est exactement ce qu'on ne veut pas demander à un parent à 3h du matin.
- **Moyenne glissante** — rejetée : un cri long tire la moyenne vers le haut et finit par masquer sa propre détection. La médiane est insensible à ces valeurs extrêmes tant qu'elles restent minoritaires dans la fenêtre.

## Conséquences

- Une agitation qui occuperait **plus de la moitié** de la fenêtre glissante ferait monter la médiane et donc le seuil, jusqu'à s'auto-masquer. La fenêtre est dimensionnée assez large (quelques minutes) pour que ce cas ne se présente pas en pratique, mais la borne est réelle : raccourcir la fenêtre rapproche ce point de rupture.
- Le seuil est indéfini au démarrage et se stabilise pendant les premières secondes d'écoute. Une valeur initiale sert d'amorce, mais elle est écrasée dès le premier bloc.
