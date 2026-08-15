<!-- wayfinder:task | parent: MAP.md | blocked-by: 0004 -->
# Implémenter les alertes de panne et l'historique de fiabilité

## Question

Mettre en œuvre la politique retenue : notification de panne, et visualisation de la fiabilité du dispositif (`uptime-card` est déjà installé) pour voir les micro-coupures récurrentes — mode de panne dominant de l'ADR-0005.

## Critère de résolution

Alertes déployées et **déclenchement vérifié en conditions réelles** (arrêt provoqué du service), carte de fiabilité en place.

---
## Résolution (2026-08-15) — CLOS

Déployé et **vérifié en conditions réelles** : redémarrage provoqué du service → `counter.babyphone_demarrages` passe de 0 à 1. Le mode de panne le plus dangereux du système est désormais instrumenté.

9 automatisations actives, 0 en défaut. Corrections de fond livrées avec : émetteur non bloquant (deux sessions), `last_reported`, `uptime_s` dans la télémétrie.

**Deux incidents de déploiement, corrigés** : un second bloc `automation:` en fin de fichier écrasait le premier (clé YAML dupliquée — l'automatisation de réinitialisation d'éveil était passée `unavailable`) ; et une automatisation pointait `sensor.babyphone_duree_service` alors que l'`entity_id` réel, figé par l'`unique_id`, est `sensor.babyphone_duree_de_service`.

**Reste ouvert, hors logiciel** : l'alimentation du Pi (`throttled=0x50005`) et le choix d'un prestataire pour le chien de garde externe — mécanisme prêt, inactif tant qu'aucune URL n'est fournie.
