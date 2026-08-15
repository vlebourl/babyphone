# La cible d'exécution est un Raspberry Pi 3 Model A+, pas la machine de dev

Le code est développé sur macOS mais tourne en permanence sur un **Raspberry Pi 3 Model A+ Rev 1.0** (Cortex-A53 aarch64, 512 Mo de RAM, stockage microSD, Raspberry Pi OS Bookworm / Python 3.11). Cette machine n'est pas un détail de déploiement : ses limites matérielles expliquent une bonne partie de la forme du code, et rien dans le dépôt ne les rend visibles.

Trois caractéristiques du 3A+ contraignent directement la conception :

- **Un seul port USB, et pas d'entrée audio intégrée.** La prise jack du Pi est une sortie ; le micro est donc forcément USB, et il occupe l'unique port. L'appareil tourne headless par construction — aucun clavier, aucune clé USB, tout passe par le réseau.
- **Pas d'Ethernet.** Le Wi-Fi est le seul lien vers la domotique, avec les coupures que ça implique dans une maison.
- **512 Mo de RAM et une carte microSD.** Deux fois moins de mémoire que le 3B+ avec lequel on le confond, et un stockage lent, petit et à usure limitée.

## Conséquences

- **Les erreurs de lecture micro sont normales, pas exceptionnelles.** L'alimentation micro-USB du 3A+ tolère mal les pointes de courant d'un micro USB : une sous-tension provoque des coupures du flux audio. C'est la raison d'être de la logique de comptage d'erreurs et de réinitialisation du flux dans la boucle d'écoute — ce n'est pas de la paranoïa défensive, c'est le mode de panne dominant en production. Ne pas la simplifier.
- **La détection du périphérique d'entrée est spécifique à la cible.** Le micro est choisi en cherchant « mic » ou « input » dans les noms de périphériques exposés par ALSA sur le Pi. Ces noms n'ont rien d'universel : sur la machine de dev, l'heuristique tombe sur un autre périphérique ou sur rien.
- **Le Wi-Fi seul aggrave la dette de [ADR-0003](0003-webhooks-domotique-sans-flux-audio.md).** Les envois synchrones dans la boucle d'écoute bloquent sur un lien qui, ici, tombe régulièrement. C'est le facteur le plus probable de déclenchement de cette dette.
- **Toute idée de tampon audio en mémoire est hors budget.** 512 Mo partagés avec l'OS interdisent de garder plus que la fenêtre glissante d'amplitudes ; conserver de l'audio brut en RAM pour re-analyse n'est pas une option sur cette cible.
- **Aucun environnement virtuel n'est conservé dans le dépôt.** Un venv construit ici embarque des extensions compilées pour l'architecture qui l'a créé — celles du Pi sont en `aarch64-linux-gnu` et ne s'exécutent pas sur macOS, et l'inverse est tout aussi vrai. Chaque machine régénère le sien avec `uv sync` (dépendances verrouillées dans `uv.lock`, Python épinglé en 3.11 pour coller à Bookworm). Sur le Pi, la compilation de `pyaudio` demande `portaudio19-dev`, `python3-dev` et `gcc`.
- **`pyaudio` / `portaudio` ne se comportent pas pareil hors cible.** L'énumération des périphériques et le comportement du flux dépendent de la pile audio de la machine : ils ne sont pas testables ailleurs que sur le Pi.
- **Vérifier sur le Pi, pas sur le Mac.** Tout ce qui touche à l'audio, aux périphériques ou aux temporisations doit être validé sur la cible avant d'être considéré comme fonctionnel.
