# Babyphone

Surveillance sonore d'une chambre d'enfant : un micro écoute en continu, décide si l'enfant est agité, et notifie la domotique de la maison.

## Langage

### Signal

**Bloc** :
La plus petite tranche d'audio observable, celle sur laquelle une amplitude est mesurée. Toute la chaîne de décision raisonne en blocs, jamais en échantillons.
_Éviter_ : frame, buffer, chunk

**Amplitude** :
L'énergie sonore d'un bloc **dans la bande vocale**, exprimée en dBFS — négative, 0 étant la pleine échelle. C'est la seule grandeur extraite du son ; le contenu du signal n'est jamais analysé.
_Éviter_ : volume, niveau, RMS

**Bande vocale** :
L'intervalle de fréquences où vivent les pleurs, 300 à 4000 Hz. Tout ce qui est en dehors est ignoré : dans une chambre, 94 % de l'énergie sonore est du grondement basse fréquence sans rapport avec l'enfant (voir [ADR-0010](docs/adr/0010-mesure-limitee-a-la-bande-vocale.md)).
_Éviter_ : filtre, passe-bande, spectre

**Marge** :
L'écart, en dB, qu'un bloc doit avoir au-dessus du fond sonore pour compter comme bruyant. Étant un écart logarithmique, c'est un rapport d'énergie constant : la sensibilité ne dépend pas de l'ambiance (voir [ADR-0008](docs/adr/0008-detection-en-dbfs.md)).
_Éviter_ : offset, sensibilité, décalage

**Seuil** :
L'amplitude au-dessus de laquelle un bloc est considéré comme bruyant, soit le fond sonore augmenté de la marge. Il suit le fond plutôt que d'être fixe (voir [ADR-0001](docs/adr/0001-seuil-adaptatif-median.md)).
_Éviter_ : sensibilité, trigger level

**Fond sonore** :
Le niveau ambiant de la pièce hors agitation : ventilation, circulation, appareils. C'est ce que le seuil suit.
_Éviter_ : bruit de fond, noise floor, baseline

### Décision

**Bloc bruyant** :
Un bloc dont l'amplitude dépasse le seuil. Un bloc bruyant isolé ne signifie rien.
_Éviter_ : pic, détection

**Événement de bruit** :
Une salve de blocs bruyants consécutifs assez longue pour ne pas être un simple craquement. Unité de base de la décision : c'est l'événement, pas le bloc, qui compte.
_Éviter_ : pic, détection, alerte

**Éveil** :
L'état dans lequel l'enfant est jugé agité, atteint quand plusieurs événements de bruit se succèdent assez rapprochés. C'est l'état que la domotique observe.
_Éviter_ : speaking, pleurs, réveil, alerte

**Retour au calme** :
La fin d'un éveil, déclarée après une longue période sans nouvel événement de bruit. Volontairement lente : un enfant qui se rendort n'est pas calme au premier silence.
_Éviter_ : silence, fin d'alerte

### Sorties

**Niveau sonore** :
La télémétrie continue envoyée à la domotique, indépendamment de tout éveil : amplitude moyenne, seuil courant, ainsi que le pic, le creux et la part de blocs bruyants de la dernière seconde. Sert à tracer des courbes, pas à alerter.
_Éviter_ : métrique, monitoring

**Pic** :
Le bloc le plus fort de la fenêtre de télémétrie — ce que le détecteur a réellement vu, là où la moyenne l'aurait lissé. C'est la grandeur qui rend une décision d'éveil lisible sur une courbe.
_Éviter_ : max, crête

**Domotique** :
Le système maison qui reçoit les éveils et les niveaux sonores et décide quoi en faire (notifier, allumer une lampe, tracer un graphe). Le babyphone ne sait rien de ce qui se passe en aval.
_Éviter_ : serveur, backend, Home Assistant, API
