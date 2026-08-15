# Notification par webhooks vers la domotique, sans flux audio

Le babyphone ne diffuse pas de son sur le réseau. Il pousse deux choses vers la domotique de la maison, en HTTP sortant : les **transitions d'éveil** (l'état a changé) et le **niveau sonore** (télémétrie continue). L'intelligence de présentation — notification téléphone, lampe, courbe — appartient entièrement à la domotique ; le babyphone ne fait qu'émettre.

Ce choix découle d'une contrainte de vie privée : un micro d'enfant qui streame en continu est une surface d'exposition permanente. Envoyer deux nombres au lieu d'un flux audio réduit cette surface à presque rien, tout en restant suffisant pour tout ce qu'on veut faire en aval.

## Options envisagées

- **Serveur dédié avec streaming** — rejeté : diffusion audio permanente sur le réseau, plus un service à maintenir et à sécuriser, pour un gain nul par rapport aux deux signaux transmis.
- **Exposer un endpoint interrogeable (pull)** — rejeté : demande d'ouvrir un port sur l'appareil et de gérer son authentification. En push sortant, l'appareil n'écoute rien.

## Conséquences

- L'URL de webhook porte le secret d'authentification. Elle vit dans un fichier non versionné, avec repli sur `localhost` quand il est absent : sans configuration, le babyphone tourne sans notifier personne plutôt que d'échouer au démarrage.
- Les envois sont **synchrones dans la boucle d'écoute** : une domotique lente ou injoignable bloque l'écoute audio pendant les tentatives. Une limitation de débit et des retries bornés contiennent le problème sans le supprimer. C'est la principale dette de ce choix — le jour où elle fait mal, la sortie est d'émettre depuis un thread avec une file. Les timeouts bornés sur chaque POST en limitent depuis le pire cas ([ADR-0005](0005-cible-raspberry-pi-3.md)).
- Les transitions d'éveil ne sont envoyées qu'aux changements d'état. Une notification perdue ne sera pas retransmise : la domotique peut rester désynchronisée jusqu'à la transition suivante.
