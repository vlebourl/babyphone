# L'amplitude ne mesure que la bande vocale, 300–4000 Hz

Le RMS large bande est dominé par ce qui porte le plus d'énergie, et dans une chambre ce n'est pas l'enfant. Mesure spectrale faite sur 60 secondes de son réel capté par le micro de la chambre :

| Bande | Part de l'énergie du fond sonore |
|---|---|
| 20–100 Hz | 70,2 % |
| 100–300 Hz | 24,0 % |
| **300–4000 Hz (voix)** | **5,6 %** |
| > 4000 Hz | 0,1 % |

**94 % de ce que le détecteur mesurait était du grondement** — ventilation, circulation, chauffage, couplage mécanique du meuble. Ce qui explique la dynamique très faible observée jusqu'ici : le signal utile était noyé dans un fond qui n'a rien à voir avec l'enfant.

Sur le même enregistrement, avec un pleur synthétique superposé, l'émergence du cri au-dessus du fond passe de **6,4 dB en large bande à 15,2 dB en bande vocale — un gain de 8,8 dB**. C'est le plus gros gain de qualité de détection de tout le dispositif, et il conditionne la pertinence de la marge de 10 dB retenue par l'[ADR-0008](0008-detection-en-dbfs.md) : un cri qui n'émerge que de 6,4 dB ne franchirait jamais cette marge.

## L'énergie de bande se calcule par FFT, pas par filtre

On ne veut pas le signal filtré, seulement son **énergie** dans la bande — et le théorème de Parseval la donne directement depuis le spectre, sans filtre récursif ni état à maintenir entre les blocs.

Ce choix est imposé par la mesure sur la cible :

| Voie | Coût par bloc | Part du budget de 50 ms |
|---|---|---|
| RMS large bande (avant) | 0,05 ms | 0,1 % |
| Biquad en Python pur | 25,4 ms | **51 %** |
| **FFT numpy** | **0,87 ms** | **1,7 %** |

Le biquad en Python pur consommait la moitié du budget temps réel sur une machine qui throttle déjà ([ADR-0005](0005-cible-raspberry-pi-3.md)) — écarté. numpy coûte 25 Mo de RSS, acceptable sur 512 Mo, et rend la voie FFT 29 fois moins chère. Un repli sur le RMS large bande est conservé si numpy manque : le dispositif reste fonctionnel, simplement moins sélectif.

## Conséquences

- **Les niveaux absolus baissent d'environ 10 dB.** Le seuil étant relatif à la médiane, la détection n'en souffre pas — mais l'historique des capteurs présente une seconde discontinuité, après celle du passage en dBFS.
- **La marge de 10 dB mérite d'être recalibrée** après quelques nuits de données filtrées. Elle a été choisie sur des données large bande où le fond était dominé par du grondement ; le rapport signal/bruit ayant changé d'un ordre de grandeur, l'optimum a probablement bougé. Le rejeu du ticket 0002 se rejoue à l'identique sur les nouvelles données.
- **Les temporisations de l'ADR-0002 deviennent probablement trop conservatrices.** Elles compensaient un signal bruité ; avec 8,8 dB de marge supplémentaire, `event_count` et `min_noise_duration` pourraient être resserrés pour réduire la latence de détection. À trancher sur données réelles, pas maintenant.
- Les bornes de bande sont dans la configuration : les élargir vers 200 Hz capterait des pleurs plus graves au prix de plus de grondement.
