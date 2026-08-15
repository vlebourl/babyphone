#!/usr/bin/env python3
"""Tableau de bord Babyphone — source de vérité (ADR-0007).

Une seule vue, cinq sections, du plus urgent au plus technique : ce qui se
passe maintenant, la nuit écoulée, les tendances, la santé du dispositif,
puis la télémétrie brute. Un parent s'arrête après la troisième ; on
descend jusqu'aux deux dernières quand quelque chose cloche.

Lovelace vit dans le stockage interne de Home Assistant, pas en YAML : ce
script écrit la vue de façon idempotente et remplace l'existante.
Déployé et exécuté par deploy/deploy.sh.
"""

import json

PATH = "/config/.storage/lovelace.lovelace_mobile"

# Palette : un bleu calme pour le signal, un ambre pour le seuil, un rouge
# réservé aux seules situations qui demandent une action.
BLEU, BLEU_CLAIR, AMBRE, ROUGE, VERT = "#4f8fd1", "#7fb2e5", "#e8a94b", "#e05252", "#5cb87a"

# `as_timestamp(state, 0)` rend 0 quand l'état est inconnu, ce qui affichait
# « 01:00 » — l'epoch en heure locale. Il faut tester l'état AVANT de convertir.
HEURE_ENDORMISSEMENT = (
    "{% set s = states('sensor.babyphone_heure_endormissement') %}"
    "{% if s not in ['unknown', 'unavailable', 'none', ''] %}"
    "{{ as_timestamp(s) | timestamp_custom('%H:%M', true) }}"
    "{% else %}—{% endif %}"
)

DERNIER_REVEIL = (
    "{% set s = states('sensor.babyphone_dernier_reveil') %}"
    "{% if s not in ['unknown', 'unavailable', 'none', ''] %}"
    "{{ as_timestamp(s) | timestamp_custom('%H:%M', true) }}"
    "{% else %}—{% endif %}"
)


def tuile(entity, primary, secondary, icon, color, tap=None):
    """Tuile mushroom verticale — la brique de tous les bandeaux de chiffres."""
    c = {
        "type": "custom:mushroom-template-card",
        "entity": entity,
        "primary": primary,
        "secondary": secondary,
        "icon": icon,
        "icon_color": color,
        "layout": "vertical",
        "fill_container": True,
    }
    if tap:
        c["tap_action"] = tap
    return c


def titre(t, s=None):
    c = {"type": "custom:mushroom-title-card", "title": t}
    if s:
        c["subtitle"] = s
    return c


# ══════════════════════════════════════════════════════════════════════
# Sections de surveillance — en haut de page
# ══════════════════════════════════════════════════════════════════════

ETAT_ACTUEL = {
    "type": "grid",
    "cards": [
        titre("👶 Maintenant", "Chambre de Lenaïc"),
        {
            # La carte la plus importante de la page : grande, sans ambiguïté,
            # et son icône dit l'essentiel avant même la lecture du texte.
            "type": "custom:mushroom-template-card",
            "entity": "input_boolean.lenaic_speaking",
            "primary": (
                "{{ 'Réveil en cours' if is_state('input_boolean.lenaic_speaking',"
                " 'on') else 'Tout est calme' }}"
            ),
            "secondary": (
                "{% if is_state('input_boolean.lenaic_speaking', 'on') %}"
                "depuis {{ relative_time(states.input_boolean.lenaic_speaking.last_changed) }}"
                "{% else %}"
                "{{ states('sensor.babyphone_calme_depuis') | int(0) }} min de calme"
                "{% endif %}"
            ),
            "icon": (
                "{{ 'mdi:emoticon-cry-outline' if"
                " is_state('input_boolean.lenaic_speaking', 'on') else 'mdi:sleep' }}"
            ),
            "icon_color": (
                "{{ 'red' if is_state('input_boolean.lenaic_speaking', 'on')"
                " else 'green' }}"
            ),
            "fill_container": True,
            "multiline_secondary": True,
        },
        {
            # Le niveau sonore des 30 dernières minutes, sans échelle ni
            # légende : on cherche une forme, pas une valeur.
            "type": "custom:mini-graph-card",
            "entities": [{"entity": "sensor.babyphone_pic_sonore", "name": "Niveau"}],
            "name": "Activité sonore récente",
            "hours_to_show": 0.5,
            "points_per_hour": 120,
            "line_width": 2,
            "line_color": BLEU,
            "show": {"labels": False, "legend": False, "extrema": False, "name": True},
        },
        {
            "type": "horizontal-stack",
            "cards": [
                tuile(
                    "binary_sensor.babyphone_en_ligne",
                    "{{ 'Actif' if is_state('binary_sensor.babyphone_en_ligne', 'on')"
                    " else 'HORS LIGNE' }}",
                    "Dispositif",
                    "{{ 'mdi:access-point-check' if"
                    " is_state('binary_sensor.babyphone_en_ligne', 'on')"
                    " else 'mdi:access-point-off' }}",
                    "{{ 'green' if is_state('binary_sensor.babyphone_en_ligne', 'on')"
                    " else 'red' }}",
                ),
                tuile(
                    "input_boolean.babyphone_on_off",
                    "{{ 'Armée' if is_state('input_boolean.babyphone_on_off', 'on')"
                    " else 'Éteinte' }}",
                    "Surveillance",
                    "mdi:shield-home",
                    "{{ 'amber' if is_state('input_boolean.babyphone_on_off', 'on')"
                    " else 'disabled' }}",
                    {"action": "toggle"},
                ),
                tuile(
                    "sensor.babyphone_dernier_reveil",
                    DERNIER_REVEIL,
                    "Dernier réveil",
                    "mdi:clock-alert-outline",
                    "blue",
                ),
            ],
        },
    ],
}

NUIT = {
    "type": "grid",
    "cards": [
        titre("🌙 Cette nuit", "Depuis 20 h"),
        {
            "type": "horizontal-stack",
            "cards": [
                tuile("sensor.babyphone_heure_endormissement", HEURE_ENDORMISSEMENT,
                      "Endormi à", "mdi:weather-night", "indigo"),
                tuile("sensor.babyphone_reveils_nuit",
                      "{{ states('sensor.babyphone_reveils_nuit') | int(0) }}",
                      "Réveils", "mdi:emoticon-cry-outline",
                      "{{ 'green' if states('sensor.babyphone_reveils_nuit') | int(0) < 3"
                      " else ('orange' if states('sensor.babyphone_reveils_nuit') | int(0)"
                      " < 8 else 'red') }}"),
                tuile("sensor.lenaic_night_asleep_duration",
                      "{{ states('sensor.lenaic_night_asleep_duration') | float(0) | round(1) }} h",
                      "Sommeil", "mdi:sleep", "blue"),
            ],
        },
        {
            # Hypnogramme : chaque barre est un éveil. Le trait doit être
            # visible — avec une épaisseur nulle, une aire à 0 est invisible
            # et le graphe paraissait vide alors que les données existaient.
            "type": "custom:apexcharts-card",
            "header": {"show": True, "title": "Éveils de la nuit", "show_states": False},
            "graph_span": "14h",
            "span": {"start": "day", "offset": "-4h"},
            "apex_config": {
                "chart": {"height": 120},
                "stroke": {"curve": "stepline", "width": 2},
                "fill": {"type": "solid", "opacity": 0.4},
                "legend": {"show": False},
                "yaxis": {"show": False, "min": 0, "max": 1},
            },
            "series": [
                {
                    "entity": "input_boolean.lenaic_speaking",
                    "name": "Éveil",
                    "type": "area",
                    "color": ROUGE,
                    "transform": "return x === 'on' ? 1 : 0;",
                    "extend_to": "now",
                    "group_by": {"func": "max", "duration": "1min"},
                }
            ],
        },
    ],
}

TENDANCES = {
    "type": "grid",
    "cards": [
        titre("📈 Tendances", "Une nuit isolée ne dit rien, une tendance si"),
        {
            "type": "custom:apexcharts-card",
            "header": {"show": True, "title": "Réveils par jour", "show_states": False},
            "graph_span": "7d",
            "span": {"start": "day"},
            "apex_config": {"chart": {"height": 180}, "legend": {"show": False}},
            "series": [
                {
                    "entity": "sensor.babyphone_reveils_24h",
                    "name": "Réveils",
                    "type": "column",
                    "color": AMBRE,
                    "group_by": {"func": "max", "duration": "1d"},
                }
            ],
        },
        {
            "type": "custom:apexcharts-card",
            "header": {"show": True, "title": "Agitation cumulée par jour",
                       "show_states": False},
            "graph_span": "7d",
            "span": {"start": "day"},
            "apex_config": {"chart": {"height": 180}, "legend": {"show": False}},
            "yaxis": [{"decimals": 1}],
            "series": [
                {
                    "entity": "sensor.babyphone_agitation_24h",
                    "name": "Heures",
                    "type": "column",
                    "color": BLEU,
                    "group_by": {"func": "max", "duration": "1d"},
                }
            ],
        },
    ],
}


# ══════════════════════════════════════════════════════════════════════
# Sections techniques — plus bas sur la même page
# ══════════════════════════════════════════════════════════════════════

SANTE = {
    "type": "grid",
    "cards": [
        titre("🩺 Santé du dispositif", "Raspberry Pi · babyphone.local"),
        {
            "type": "horizontal-stack",
            "cards": [
                tuile("binary_sensor.babyphone_en_ligne",
                      "{{ 'En ligne' if is_state('binary_sensor.babyphone_en_ligne', 'on')"
                      " else 'HORS LIGNE' }}",
                      "Télémétrie",
                      "{{ 'mdi:access-point-check' if"
                      " is_state('binary_sensor.babyphone_en_ligne', 'on')"
                      " else 'mdi:access-point-off' }}",
                      "{{ 'green' if is_state('binary_sensor.babyphone_en_ligne', 'on')"
                      " else 'red' }}"),
                tuile("sensor.babyphone_duree_de_service",
                      "{% set s = states('sensor.babyphone_duree_de_service') | float(0) %}"
                      "{% if s < 3600 %}{{ (s / 60) | round(0) }} min"
                      "{% else %}{{ (s / 3600) | round(1) }} h{% endif %}",
                      "Sans redémarrage", "mdi:restart",
                      "{{ 'red' if states('counter.babyphone_demarrages') | int(0) >= 3"
                      " else 'grey' }}"),
                tuile("sensor.babyphone_sous_tension",
                      "{{ states('sensor.babyphone_sous_tension') }}",
                      "Sous-tension", "mdi:flash-alert",
                      "{{ 'red' if is_state('sensor.babyphone_sous_tension', 'Oui')"
                      " else 'green' }}"),
            ],
        },
        {
            # Ce bandeau n'apparaît que lorsqu'il a quelque chose à dire.
            "type": "conditional",
            "conditions": [{"condition": "state",
                            "entity": "sensor.babyphone_sous_tension", "state": "Oui"}],
            "card": {
                "type": "markdown",
                "content": (
                    "### ⚡ Alimentation insuffisante\n"
                    "Le Pi signale une sous-tension. C'est la cause connue des "
                    "coupures du micro — aucun réglage logiciel ne la compense.\n\n"
                    "**Remède** : bloc officiel 5,1 V / 3 A et câble court."
                ),
            },
        },
        {
            "type": "horizontal-stack",
            "cards": [
                tuile("counter.babyphone_demarrages",
                      "{{ states('counter.babyphone_demarrages') | int(0) }}",
                      "Redémarrages", "mdi:restart-alert",
                      "{{ 'red' if states('counter.babyphone_demarrages') | int(0) >= 3"
                      " else 'grey' }}"),
                tuile("input_boolean.babyphone_on_off",
                      "{{ 'Armée' if is_state('input_boolean.babyphone_on_off', 'on')"
                      " else 'Éteinte' }}",
                      "Surveillance", "mdi:shield-home",
                      "{{ 'amber' if is_state('input_boolean.babyphone_on_off', 'on')"
                      " else 'disabled' }}",
                      {"action": "toggle"}),
                tuile("input_boolean.babyphone_alerte_acquittee",
                      "{{ 'Acquittée' if"
                      " is_state('input_boolean.babyphone_alerte_acquittee', 'on')"
                      " else 'Armée' }}",
                      "Alerte", "mdi:bell-check",
                      "{{ 'orange' if"
                      " is_state('input_boolean.babyphone_alerte_acquittee', 'on')"
                      " else 'green' }}"),
            ],
        },
    ],
}

TELEMETRIE = {
    "type": "grid",
    "cards": [
        titre("📡 Télémétrie", "Bande vocale 300–4000 Hz, en dBFS"),
        {
            "type": "horizontal-stack",
            "cards": [
                tuile("sensor.babyphone_pic_sonore",
                      "{{ states('sensor.babyphone_pic_sonore') | float(0) | round(0) }}",
                      "Pic (dB)", "mdi:waveform", "light-blue"),
                tuile("sensor.babyphone_noise_level",
                      "{{ states('sensor.babyphone_noise_level') | float(0) | round(0) }}",
                      "Moyenne (dB)", "mdi:volume-medium", "blue"),
                tuile("sensor.babyphone_threshold",
                      "{{ states('sensor.babyphone_threshold') | float(0) | round(0) }}",
                      "Seuil (dB)", "mdi:arrow-collapse-horizontal", "amber"),
                tuile("sensor.babyphone_activite",
                      "{{ states('sensor.babyphone_activite') | int(0) }} %",
                      "Blocs bruyants", "mdi:chart-bar", "purple"),
            ],
        },
        {
            # C'est le PIC qui est comparé au seuil, pas la moyenne : tracer
            # les deux rend une décision d'éveil lisible sur la courbe.
            "type": "custom:apexcharts-card",
            "header": {"show": True, "title": "Niveau sonore et seuil (2 h)",
                       "show_states": True, "colorize_states": True},
            "graph_span": "2h",
            "apex_config": {
                "chart": {"height": 220},
                "stroke": {"width": [1, 2, 2]},
                "legend": {"show": True},
            },
            "yaxis": [{"decimals": 0}],
            "series": [
                {"entity": "sensor.babyphone_pic_sonore", "name": "Pic", "type": "area",
                 "color": BLEU_CLAIR, "opacity": 0.2,
                 "group_by": {"func": "max", "duration": "30s"}},
                {"entity": "sensor.babyphone_noise_level", "name": "Moyenne",
                 "color": BLEU, "group_by": {"func": "avg", "duration": "30s"}},
                {"entity": "sensor.babyphone_threshold", "name": "Seuil", "color": AMBRE,
                 "group_by": {"func": "avg", "duration": "30s"}},
            ],
        },
        {
            "type": "custom:apexcharts-card",
            "header": {"show": True, "title": "Disponibilité de la télémétrie (24 h)",
                       "show_states": False},
            "graph_span": "24h",
            "apex_config": {
                "chart": {"height": 110},
                "stroke": {"curve": "stepline", "width": 2},
                "fill": {"type": "solid", "opacity": 0.4},
                "legend": {"show": False},
                "yaxis": {"show": False, "min": 0, "max": 1},
            },
            "series": [
                {"entity": "binary_sensor.babyphone_en_ligne", "name": "En ligne",
                 "type": "area", "color": VERT,
                 "transform": "return x === 'on' ? 1 : 0;", "extend_to": "now",
                 "group_by": {"func": "min", "duration": "5min"}},
            ],
        },
    ],
}


VUE = {
    "theme": "Backend-selected",
    "title": "Babyphone",
    "path": "babyphone",
    "icon": "mdi:baby-face-outline",
    "type": "sections",
    "max_columns": 2,
    "badges": [
        {"type": "entity", "show_name": False, "show_state": True, "show_icon": True,
         "entity": "binary_sensor.babyphone_en_ligne"},
        {"type": "entity", "show_name": False, "show_state": True, "show_icon": True,
         "entity": "input_boolean.lenaic_speaking"},
    ],
    "cards": [],
    "sections": [ETAT_ACTUEL, NUIT, TENDANCES, SANTE, TELEMETRIE],
}


def main():
    with open(PATH) as f:
        data = json.load(f)
    views = data["data"]["config"]["views"]

    # la sous-vue système a été fusionnée : on la retire si elle traîne encore
    avant = len(views)
    views[:] = [v for v in views if v.get("path") != "babyphone-systeme"]
    if len(views) != avant:
        print("  ancienne sous-vue « babyphone-systeme » retirée")

    for i, v in enumerate(views):
        if v.get("path") == VUE["path"]:
            views[i] = VUE
            print(f"  vue « babyphone » remplacée (position {i})")
            break
    else:
        idx = next((i for i, v in enumerate(views) if v.get("path") == "lenaic"),
                   len(views) - 1)
        views.insert(idx + 1, VUE)
        print(f"  vue « babyphone » insérée (position {idx + 1})")

    with open(PATH, "w") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
