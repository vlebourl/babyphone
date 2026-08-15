<!-- wayfinder:task | parent: MAP.md | blocked-by: — -->
# Télémétrie enrichie : publier ce que le détecteur voit réellement

## Question

La domotique reçoit la **moyenne** des amplitudes sur 1 s alors que la décision se prend par blocs de 50 ms. Mesuré sur 24 h : p99 de ce qui est envoyé = 0,1159 pour un max réel de 0,3918. Le lissage écrase les pics qui déclenchent, donc la courbe peut afficher « calme » pendant que le détecteur voit un cri — et aucun réglage de constante ne peut se valider visuellement.

Quelles grandeurs `NoiseReport` doit-il porter pour que la courbe montre la décision, sans changer la cadence d'émission ni casser le contrat existant ?

## Critère de résolution

Grandeurs publiées, entités HA créées, déployé en prod, et la courbe montre un pic là où le journal montre une transition d'éveil.
