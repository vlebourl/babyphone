# La détection raisonne en dBFS, pas en amplitude linéaire

Le seuil se calculait en ajoutant une marge fixe de `0.05` à la médiane d'un RMS normalisé. Une marge **additive** sur une grandeur dont le fond varie d'un facteur 20 entre le jour et la nuit produit une sensibilité qui dérive avec l'ambiance : mesuré sur 78 532 échantillons réels, le seuil valait **2,75× le fond la nuit contre 2,06× le jour** — le dispositif était un tiers moins sensible quand il comptait le plus.

Toute la chaîne travaille désormais en **dBFS** (`20·log₁₀(rms)`, négatif, 0 = pleine échelle). En échelle logarithmique une marge additive *est* un rapport constant : `médiane + 10 dB` signifie « 3,16× l'énergie du fond », dans une chambre silencieuse comme dans une pièce bruyante. C'est aussi l'échelle de la perception sonore, donc celle qui rend une courbe lisible.

La conversion se fait dans la source audio, en un seul endroit ; `Detection` ne sait pas qu'elle a changé d'unité — sa logique de médiane glissante est indifférente à l'échelle. Un plancher à −120 dBFS évite `log₁₀(0)` sur un bloc numériquement nul.

## Choix de l'offset : +10 dB

Retenu par rejeu de 24 h de données réelles, en comparant le nombre d'éveils produits (ticket 0002) :

| Offset | Éveils nuit | Éveils jour | Total |
|---|---|---|---|
| Ancien (+0,05 linéaire) | 10 | 4 | 14 |
| 9 dB | 13 | 5 | 18 |
| **10 dB** | **12** | **2** | **14** |
| 11 dB | 11 | 1 | 12 |

À **volume d'éveils inchangé**, 10 dB redistribue la sensibilité vers la nuit (+20 %) et divise par deux les faux positifs de jour. Aucun changement d'habitude pour la maison, meilleure détection là où elle sert.

## Conséquences

- **Rupture d'échelle côté domotique.** `sensor.babyphone_noise_level` passe de ~0,09 à ~−21 : l'historique présente une discontinuité au déploiement, et toute automatisation comparant cette valeur à un nombre en dur doit être revue. En contrepartie, l'`unit_of_measurement: "dB"` que portaient déjà les capteurs cesse d'être mensonger.
- **Le différentiel contre le code pré-refactor est retiré à ce point.** Il garantissait l'absence de changement de comportement ; ici le changement est l'objectif. Le rejeu sur données réelles le remplace comme référence.
- **Les constantes de temporisation de l'ADR-0002 sont inchangées** et restent exprimées en secondes — seule l'échelle d'amplitude bouge.
- La dynamique affichée passe de 0,007–0,39 à −43…−8 dBFS, bien plus étalée : les graphes deviennent exploitables pour régler les constantes.
