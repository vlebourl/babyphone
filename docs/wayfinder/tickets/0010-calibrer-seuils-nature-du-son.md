<!-- wayfinder:grilling | parent: MAP.md | blocked-by: — -->
# Calibrer les seuils de nature du son sur des nuits réelles

## Question

L'ADR-0011 le disait sans détour : les seuils d'étiquetage sont « un premier jeu, posé sur la littérature acoustique et vérifié sur signaux de synthèse », qui « demandent une calibration sur plusieurs nuits réelles avant d'être tenus pour justes ». La première nuit de données réelles est arrivée le **2026-08-16**, et elle a déjà invalidé un seuil sur deux.

**Ce qui est déjà corrigé** (branche `fix/sifflantes-etiquetees-pleurs`) : le centre de gravité spectral ne suffit plus à déclarer un pleur. Il étiquetait « pleurs » chaque consonne sifflante de la parole — 51 fausses trames en onze minutes de lecture d'histoire à voix haute, contre zéro pendant le silence qui précédait.

**Ce qui reste ouvert, et fait l'objet de ce ticket** : la branche `cri`.

```python
if emergence_db >= EMERGENCE_CRI_DB and centroid_hz >= CENTROID_VOIX_HZ:
    return Kind(CRI, emergence_db, centroid_hz)
```

Avec `EMERGENCE_CRI_DB = 22.0` et `CENTROID_VOIX_HZ = 900.0`, **une voix d'adulte qui lit une histoire en faisant les personnages est étiquetée `cri`**. Mesuré le 2026-08-16 : profil médian centre 2027 Hz, grave 10 %, médium 55 %, aigu 35 %, émergence ≈ 27 dB. Douze trames sur la soirée. Aucun cri n'a eu lieu.

Le correctif appliqué aux pleurs ne mord pas ici, et c'est normal : ces trames ont **55 % de médium**, la signature harmonique d'une vraie voix forte. Un cri d'enfant et un adulte qui déclame sont acoustiquement voisins ; ce n'est pas un bug de seuil mal placé mais une **ambiguïté réelle**.

À trancher :

1. **900 Hz est-il la bonne borne pour un cri ?** C'est `CENTROID_VOIX_HZ`, réutilisée telle quelle — la frontière « en dessous, c'est de la voix ». S'en servir aussi pour qualifier un cri revient à dire « tout ce qui n'est pas grave et qui est fort ». Un cri d'enfant monte bien plus haut ; une borne propre, plus élevée, écarterait la déclamation adulte sans coûter de sensibilité.
2. **22 dB d'émergence est-il atteignable par une voix adulte normale à proximité du micro ?** Ce soir, oui, largement. La distance micro-source n'est pas dans le modèle : un adulte penché sur le lit est bien plus près que l'enfant couché.
3. **Faut-il une classe pour la présence d'un adulte ?** Un parent qui lit n'est ni un cri, ni un bruit de maison, ni l'enfant. La nommer réglerait le problème par le haut, mais ajoute une classe au vocabulaire (`CONTEXT.md`) et une branche à la machine.
4. **La persistance temporelle.** Sur 425 échantillons de la demi-heure du 2026-08-16, on compte **424 plages** : aucune étiquette ne tient plus d'une trame. Le mécanisme d'`EMERGENCE_MIN_DB` calé sur la marge de détection (commit `3518ddc`) a supprimé le clignotement `calme`/`voix` mais pas le reste. Faut-il un lissage — N trames sur M — avant publication ? Ce soir, exiger trois trames consécutives aurait éliminé la totalité des faux positifs, sifflantes comprises.

## Ce qu'il faut pour trancher

Des nuits réelles étiquetées, avec le contexte connu : qui était dans la chambre, à quelle heure, et ce qui s'est réellement passé. La télémétrie publie déjà tout le nécessaire — centre de gravité, trois bandes, pic, fond — donc le rejeu se fait sans redéployer de code, exactement comme au ticket 0002 pour la marge.

**Ne pas trancher à l'intuition.** Le ticket 0002 a montré ce que vaut une mesure correcte contre une estimation : l'asymétrie nuit/jour annoncée à 2,6× valait en réalité 1,33×.

## Critère de résolution

Des bornes justifiées par une mesure sur données réelles, avec le tableau des candidats et leur effet sur le nombre de fausses étiquettes — et une décision explicite sur le lissage temporel.

## Contexte rassurant

L'étiquette **ne décide de rien** : la surveillance reste pilotée par l'amplitude seule (`deploy/homeassistant/babyphone_monitoring.yaml`). Aucun de ces faux positifs n'a déclenché d'alerte. Le sujet est la lisibilité du tableau de bord et la confiance qu'on peut accorder à l'étiquette, pas la sécurité.

En revanche l'**éveil**, lui, est piloté par l'amplitude — et le « dernier réveil » du 2026-08-16 à 20:05:59 correspond à l'entrée d'un adulte dans la chambre, pas à un réveil de Lenaïc. Les 26 éveils sur 24 h sont probablement gonflés par le même mécanisme. C'est un sujet distinct de l'étiquetage, à ouvrir séparément.
