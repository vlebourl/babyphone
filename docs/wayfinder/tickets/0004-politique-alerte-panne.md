<!-- wayfinder:grilling | parent: MAP.md | blocked-by: — -->
# Décider la politique d'alerte quand le babyphone tombe

## Question

Un babyphone qui meurt à 2 h du matin est silencieux, et le silence se confond avec « tout va bien ». Le capteur de vivacité existe mais **personne n'est prévenu**.

Une notification critique nocturne est intrusive par conception : elle traverse le mode silencieux. Quel délai avant alerte ? Quelles conditions évitent de réveiller la maison pour un redémarrage volontaire ou une micro-coupure Wi-Fi ? Faut-il un canal différent le jour et la nuit ? Et que se passe-t-il si la panne, c'est Home Assistant lui-même ?

Ticket **HITL** : la contrepartie est tenue par un subagent adversarial indépendant, mandaté pour attaquer la proposition (faux positifs, fatigue d'alerte, angles morts).

## Critère de résolution

Une politique d'alerte écrite, ayant survécu au grilling adversarial, prête à implémenter.

---
## Résolution (2026-08-15) — CLOS

Grilling adversarial mené par un subagent indépendant, qui a **vérifié ses affirmations sur les machines** plutôt que de théoriser. Trois hypothèses invalidées, toutes reconfirmées par moi avant correction :

1. **Les téléphones sont des Pixel** (Android). `interruption-level: critical` est iOS uniquement — l'alerte serait partie, aurait paru fonctionner, et n'aurait réveillé personne. Canal correct : `alarm_stream_max`.
2. **Le Pi est en sous-tension et throttling *en cours*** : `throttled=0x50005`. Préalable matériel, hors de portée du logiciel.
3. **Le générateur de faux positifs était dans mon propre code** : les retries HTTP que j'avais ajoutés pouvaient bloquer la boucle d'écoute ~90 s, assez pour que le dispositif se déclare lui-même hors ligne — et surtout pour être sourd une minute et demie.

Politique retenue : voir [ADR-0009](../adr/0009-politique-d-alerte-de-panne.md).

**Ce que j'ai retenu du contradicteur** : le déclenchement sur capteur plutôt que sur le binary_sensor (3,5–4,5 min de latence cachée), la circularité de l'état de sommeil, le point de silence unique de l'interrupteur exposé à la voix, `last_reported` contre `last_updated`, les détecteurs de cécité, et surtout la boucle de redémarrage — mode de panne où tous les voyants restent verts pendant que l'éveil devient inatteignable.

**Ce que j'ai écarté** : la notification d'éveil (contredit l'ADR-0002), et `uptime-card` comme mesure de micro-coupures (elle ne peut structurellement pas les montrer).
