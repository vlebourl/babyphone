# La nature d'un son se lit dans son spectre, pas dans son volume

Le dispositif ne savait dire que « bruit » ou « pas bruit ». Un parent, lui, ne réagit pas de la même façon à un gazouillis, à des pleurs et à un cri — et une porte qui claque ne devrait jamais être attribuée à l'enfant. Le volume seul ne peut pas les séparer : un choc et un cri sont tous deux forts.

La **forme** du spectre, elle, les sépare. Un pleur de nourrisson concentre son énergie vers 1–3 kHz, avec un centre de gravité spectral haut ; la parole reste plus grave et mieux répartie ; un choc est bref, massé dans le bas, et dépourvu des harmoniques qu'aurait une voix.

Le dispositif publie donc, à côté du niveau sonore, le **centre de gravité spectral** et la répartition de l'énergie en trois sous-bandes (300–800, 800–2000, 2000–4000 Hz), plus une **étiquette** : `calme`, `voix`, `pleurs`, `cri`, `bruit`.

## Le calcul est gratuit

Aucune FFT supplémentaire : c'est la même transformée qui donnait déjà l'énergie de bande ([ADR-0010](0010-mesure-limitee-a-la-bande-vocale.md)), on se contente d'en lire davantage. Le surcoût se réduit à quelques sommes sur un tableau déjà calculé — sans commune mesure avec le budget de 50 ms par bloc de la cible ([ADR-0005](0005-cible-raspberry-pi-3.md)).

La forme retenue est celle du **bloc le plus fort** de la fenêtre de télémétrie, pas une moyenne : c'est l'instant du cri qui renseigne, pas la seconde qui l'entoure.

## Deux règles qui portent l'essentiel

**Un cri est fort *et* aigu.** La conjonction compte : sans elle, tout choc violent deviendrait un cri.

**Une voix est harmonique.** Sa fondamentale s'accompagne toujours de partiels qui débordent dans le médium. Une énergie massée à plus de 85 % dans le grave, sans ce prolongement, n'est pas quelqu'un — c'est un meuble, un pas, une porte. Cette règle seule évite le faux positif le plus gênant : attribuer à l'enfant un bruit de maison.

## Conséquences

- **Les seuils sont un premier jeu, pas une vérité.** Ils s'appuient sur l'acoustique de la voix et ont été vérifiés sur signaux de synthèse — pas sur des pleurs réels de Lenaïc. Ils demandent une calibration sur plusieurs nuits. C'est pourquoi les grandeurs brutes sont publiées **à côté** de l'étiquette : on peut recalibrer en regardant l'historique, sans redéployer de code.
- **L'étiquette ne décide de rien.** La machine à états d'éveil continue de ne dépendre que de l'amplitude ([ADR-0002](0002-eveil-confirme-par-accumulation.md)) : le spectre qualifie, il ne déclenche pas. Une erreur d'étiquetage n'a donc aucune conséquence sur la surveillance elle-même.
- **La durée reste le meilleur filtre contre les transitoires.** Un choc bref est éliminé par l'exigence de durée minimale de l'ADR-0002, pas par son spectre. Les deux mécanismes se complètent plutôt qu'ils ne se doublent.
- **`Detection.feed` accepte un spectre optionnel.** Son interface s'élargit d'un argument sans que la décision en dépende, ce qui laisse tous les tests existants valides et le module aussi pur qu'avant.
