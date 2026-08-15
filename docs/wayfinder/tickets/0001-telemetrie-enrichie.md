<!-- wayfinder:task | parent: MAP.md | blocked-by: — -->
# Télémétrie enrichie : publier ce que le détecteur voit réellement

## Question

La domotique reçoit la **moyenne** des amplitudes sur 1 s alors que la décision se prend par blocs de 50 ms. Mesuré sur 24 h : p99 de ce qui est envoyé = 0,1159 pour un max réel de 0,3918. Le lissage écrase les pics qui déclenchent, donc la courbe peut afficher « calme » pendant que le détecteur voit un cri — et aucun réglage de constante ne peut se valider visuellement.

Quelles grandeurs `NoiseReport` doit-il porter pour que la courbe montre la décision, sans changer la cadence d'émission ni casser le contrat existant ?

## Critère de résolution

Grandeurs publiées, entités HA créées, déployé en prod, et la courbe montre un pic là où le journal montre une transition d'éveil.

---
## Résolution (2026-08-15) — CLOS

`NoiseReport` porte désormais `peak`, `floor` et `noisy_ratio` en plus de `amplitude` et `threshold`. Clés filaires **additives** : le différentiel contre le code pré-refactor reste identique (717 POST).

Côté Home Assistant, les capteurs de bruit ont dû être **rapatriés** de `lenaic_sleep.yaml` vers le package babyphone : HA n'accepte qu'un seul déclencheur par `webhook_id`, donc ajouter des capteurs imposait de regrouper le bloc. `unique_id` conservés à l'identique (historique préservé), et le `webhook_id` passe par `!secret` — le package est versionné dans un dépôt public.

**Vérifié en prod** : les 5 capteurs alimentés, et la première mesure confirme le diagnostic — moyenne 0,0926 / **pic 0,1213** (+31 %), écart que la courbe ne montrait pas.

**Fait apparaître pour 0002** : le seuil (0,1346) est au-dessus du pic mesuré en journée, ce qui explique une activité nulle en régime calme.
