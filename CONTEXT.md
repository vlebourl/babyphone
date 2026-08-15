# Babyphone

Surveillance sonore d'une chambre d'enfant : un micro écoute en continu, décide si l'enfant est agité, et notifie la domotique de la maison.

## Langage

### Signal

**Bloc** :
La plus petite tranche d'audio observable, celle sur laquelle une amplitude est mesurée. Toute la chaîne de décision raisonne en blocs, jamais en échantillons.
_Éviter_ : frame, buffer, chunk

**Amplitude** :
L'énergie sonore d'un bloc, normalisée entre 0 et 1. C'est la seule grandeur extraite du son ; le contenu du signal n'est jamais analysé.
_Éviter_ : volume, niveau, RMS

**Seuil** :
L'amplitude au-dessus de laquelle un bloc est considéré comme bruyant. Le seuil suit le fond sonore de la pièce plutôt que d'être fixe (voir [ADR-0001](docs/adr/0001-seuil-adaptatif-median.md)).
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
La télémétrie continue envoyée à la domotique — amplitude récente et seuil courant — indépendamment de tout éveil. Sert à tracer des courbes, pas à alerter.
_Éviter_ : métrique, monitoring

**Domotique** :
Le système maison qui reçoit les éveils et les niveaux sonores et décide quoi en faire (notifier, allumer une lampe, tracer un graphe). Le babyphone ne sait rien de ce qui se passe en aval.
_Éviter_ : serveur, backend, Home Assistant, API
