# L'éveil est confirmé par accumulation, pas par un seul pic

Un babyphone qui alerte au premier bruit est un babyphone qu'on éteint. Un éveil n'est donc déclaré qu'après une succession de conditions : une salve de bruit doit durer plus qu'un craquement pour compter comme *événement de bruit*, plusieurs événements doivent s'enchaîner à faible intervalle, et l'éveil ne retombe qu'après plusieurs minutes de calme continu.

Le paramétrage cible délibérément l'**asymétrie** : on accepte de rater les premières secondes d'une agitation réelle (elles seront rattrapées par les événements suivants) pour ne quasiment jamais déclencher sur une porte qui claque ou un camion qui passe. Le retour au calme est tout aussi délibérément lent — un enfant qui se rendort produit des silences de plusieurs dizaines de secondes, et redescendre trop vite ferait osciller l'état d'éveil.

## Conséquences

- La latence de détection est de l'ordre de quelques secondes, pas immédiate. C'est le prix assumé de l'absence de faux positifs.
- Le compteur d'événements se réinitialise dès que le retour au calme est prononcé : une agitation reprenant après une longue pause repart de zéro et doit se reconfirmer.
- Les constantes de temporisation ne sont pas indépendantes. Modifier la durée minimale d'un événement sans revoir l'intervalle entre événements peut rendre l'éveil inatteignable : à toucher ensemble.
