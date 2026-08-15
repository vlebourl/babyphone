#!/usr/bin/env python3
"""Vue Babyphone du tableau de bord mobile — source de vérité (ADR-0007).

Lovelace vit dans le stockage interne de Home Assistant, pas en YAML : ce
script écrit la vue de façon idempotente. Relancé, il remplace la vue
existante au lieu d'en créer une seconde.

Déployé et exécuté par deploy/deploy.sh.
"""

import json

PATH = "/config/.storage/lovelace.lovelace_mobile"

# ── Section « Nuit dernière » — la question du matin ────────────────────
# Un parent ne veut pas lire une courbe de 12 h : il veut savoir comment
# s'est passée la nuit. Les chiffres d'abord, l'hypnogramme ensuite, la
# courbe fine en dernier pour qui veut creuser.
NUIT = {
    "type": "grid",
    "cards": [
        {
            "type": "custom:mushroom-title-card",
            "title": "🌙 Nuit dernière",
            "subtitle": "Ce qui s'est passé pendant que vous dormiez",
        },
        {
            "type": "horizontal-stack",
            "cards": [
                {
                    "type": "custom:mushroom-template-card",
                    "entity": "sensor.babyphone_heure_endormissement",
                    "primary": (
                        "{{ as_timestamp(states('sensor.babyphone_heure_endormissement'),"
                        " 0) | timestamp_custom('%H:%M', true, '—') }}"
                    ),
                    "secondary": "Endormi à",
                    "icon": "mdi:weather-night",
                    "icon_color": "indigo",
                    "layout": "vertical",
                    "fill_container": True,
                },
                {
                    "type": "custom:mushroom-template-card",
                    "entity": "sensor.babyphone_reveils_nuit",
                    "primary": "{{ states('sensor.babyphone_reveils_nuit') | int(0) }}",
                    "secondary": "Réveils",
                    "icon": "mdi:emoticon-cry-outline",
                    "icon_color": (
                        "{{ 'green' if states('sensor.babyphone_reveils_nuit')"
                        " | int(0) == 0 else 'orange' }}"
                    ),
                    "layout": "vertical",
                    "fill_container": True,
                },
                {
                    "type": "custom:mushroom-template-card",
                    "entity": "sensor.lenaic_night_asleep_duration",
                    "primary": (
                        "{{ states('sensor.lenaic_night_asleep_duration')"
                        " | float(0) | round(1) }} h"
                    ),
                    "secondary": "Sommeil",
                    "icon": "mdi:sleep",
                    "icon_color": "blue",
                    "layout": "vertical",
                    "fill_container": True,
                },
                {
                    "type": "custom:mushroom-template-card",
                    "entity": "sensor.babyphone_calme_depuis",
                    "primary": (
                        "{{ states('sensor.babyphone_calme_depuis') | int(0) }} min"
                    ),
                    "secondary": "Calme depuis",
                    "icon": "mdi:timer-sand",
                    "icon_color": "teal",
                    "layout": "vertical",
                    "fill_container": True,
                },
            ],
        },
        {
            # Hypnogramme : la nuit entière lisible d'un coup d'œil. Les
            # bandes rouges sont les éveils, le reste est calme.
            "type": "custom:apexcharts-card",
            "header": {"show": True, "title": "Éveils de la nuit", "show_states": False},
            "graph_span": "14h",
            "span": {"start": "day", "offset": "-4h"},
            "chart_type": "line",
            "apex_config": {
                "chart": {"height": 130},
                "stroke": {"curve": "stepline", "width": 0},
                "fill": {"type": "solid", "opacity": 0.85},
                "yaxis": {"show": False, "min": 0, "max": 1},
                "legend": {"show": False},
                "tooltip": {"enabled": True},
            },
            "series": [
                {
                    "entity": "input_boolean.lenaic_speaking",
                    "name": "Éveil",
                    "color": "#e05252",
                    "type": "area",
                    "transform": "return x === 'on' ? 1 : 0;",
                    "extend_to": "now",
                    "group_by": {"func": "max", "duration": "2min"},
                }
            ],
        },
        {
            # La courbe fine, pour qui veut comprendre un éveil précis.
            # Le PIC est tracé, pas seulement la moyenne : c'est lui que le
            # détecteur compare au seuil (ticket 0001).
            "type": "custom:apexcharts-card",
            "header": {"show": True, "title": "Niveau sonore et seuil", "show_states": False},
            "graph_span": "14h",
            "span": {"start": "day", "offset": "-4h"},
            "apex_config": {
                "chart": {"height": 200},
                "stroke": {"width": [1, 2, 2]},
                "legend": {"show": True},
            },
            "yaxis": [{"decimals": 0, "apex_config": {"tickAmount": 4}}],
            "series": [
                {
                    "entity": "sensor.babyphone_pic_sonore",
                    "name": "Pic",
                    "color": "#7fb2e5",
                    "type": "area",
                    "opacity": 0.25,
                    "group_by": {"func": "max", "duration": "2min"},
                },
                {
                    "entity": "sensor.babyphone_noise_level",
                    "name": "Moyenne",
                    "color": "#4f8fd1",
                    "group_by": {"func": "avg", "duration": "2min"},
                },
                {
                    "entity": "sensor.babyphone_threshold",
                    "name": "Seuil",
                    "color": "#e8a94b",
                    "stroke_width": 2,
                    "group_by": {"func": "avg", "duration": "2min"},
                },
            ],
        },
        {
            # Une nuit isolée ne dit rien ; une tendance sur sept nuits, si.
            "type": "custom:apexcharts-card",
            "header": {"show": True, "title": "Sept dernières nuits", "show_states": False},
            "graph_span": "7d",
            "span": {"start": "day"},
            # Les histogrammes se déclarent par `type: column` SUR LA SÉRIE.
            # `chart_type` n'accepte que line, scatter, pie, donut, radialBar :
            # y mettre "bar" fait rejeter toute la carte.
            "apex_config": {"chart": {"height": 170}},
            "series": [
                {
                    "entity": "sensor.lenaic_night_asleep_duration",
                    "name": "Sommeil (h)",
                    "type": "column",
                    "color": "#5b8def",
                    "group_by": {"func": "max", "duration": "1d"},
                },
                {
                    "entity": "sensor.babyphone_reveils_24h",
                    "name": "Réveils",
                    "type": "column",
                    "color": "#e0954f",
                    "group_by": {"func": "max", "duration": "1d"},
                },
            ],
        },
    ],
}

# ── Section « État » — maintenant ───────────────────────────────────────
ETAT = {
    "type": "grid",
    "cards": [
        {
            "type": "custom:mushroom-title-card",
            "title": "👶 Maintenant",
            "subtitle": "Chambre de Lenaïc",
        },
        {
            "type": "custom:mushroom-template-card",
            "entity": "input_boolean.lenaic_speaking",
            "primary": (
                "{{ 'Réveil en cours' if is_state('input_boolean.lenaic_speaking',"
                " 'on') else 'Tout est calme' }}"
            ),
            "secondary": "{{ relative_time(states.input_boolean.lenaic_speaking.last_changed) }}",
            "icon": (
                "{{ 'mdi:emoticon-cry-outline' if"
                " is_state('input_boolean.lenaic_speaking', 'on') else 'mdi:sleep' }}"
            ),
            "icon_color": (
                "{{ 'red' if is_state('input_boolean.lenaic_speaking', 'on')"
                " else 'green' }}"
            ),
            "fill_container": True,
        },
        {
            "type": "horizontal-stack",
            "cards": [
                {
                    "type": "custom:mushroom-template-card",
                    "entity": "binary_sensor.babyphone_en_ligne",
                    "primary": (
                        "{{ 'En ligne' if is_state('binary_sensor.babyphone_en_ligne',"
                        " 'on') else 'HORS LIGNE' }}"
                    ),
                    "secondary": "Dispositif",
                    "icon": (
                        "{{ 'mdi:wifi-check' if"
                        " is_state('binary_sensor.babyphone_en_ligne', 'on')"
                        " else 'mdi:wifi-off' }}"
                    ),
                    "icon_color": (
                        "{{ 'green' if is_state('binary_sensor.babyphone_en_ligne',"
                        " 'on') else 'red' }}"
                    ),
                    "layout": "vertical",
                    "fill_container": True,
                },
                {
                    "type": "custom:mushroom-template-card",
                    "entity": "input_boolean.babyphone_on_off",
                    "primary": (
                        "{{ 'Allumé' if is_state('input_boolean.babyphone_on_off',"
                        " 'on') else 'Éteint' }}"
                    ),
                    "secondary": "Surveillance",
                    "icon": "mdi:power",
                    "icon_color": (
                        "{{ 'amber' if is_state('input_boolean.babyphone_on_off',"
                        " 'on') else 'grey' }}"
                    ),
                    "layout": "vertical",
                    "fill_container": True,
                    "tap_action": {"action": "toggle"},
                },
                {
                    # Une durée de service qui repart de zéro trahit une
                    # boucle de redémarrage : dispositif sourd, voyants verts.
                    "type": "custom:mushroom-template-card",
                    "entity": "sensor.babyphone_duree_de_service",
                    "primary": (
                        "{{ (states('sensor.babyphone_duree_de_service')"
                        " | float(0) / 3600) | round(1) }} h"
                    ),
                    "secondary": "Sans redémarrage",
                    "icon": "mdi:restart",
                    "icon_color": (
                        "{{ 'red' if states('counter.babyphone_demarrages')"
                        " | int(0) >= 3 else 'grey' }}"
                    ),
                    "layout": "vertical",
                    "fill_container": True,
                },
            ],
        },
    ],
}

VIEW = {
    "theme": "Backend-selected",
    "title": "Babyphone",
    "path": "babyphone",
    "icon": "mdi:baby-face-outline",
    "type": "sections",
    "max_columns": 2,
    "badges": [
        {
            "type": "entity",
            "show_name": False,
            "show_state": True,
            "show_icon": True,
            "entity": "binary_sensor.babyphone_en_ligne",
            "name": "Babyphone",
        }
    ],
    "cards": [],
    "sections": [NUIT, ETAT],
}


def main():
    with open(PATH) as f:
        data = json.load(f)

    views = data["data"]["config"]["views"]
    for i, v in enumerate(views):
        if v.get("path") == "babyphone":
            views[i] = VIEW
            print(f"  vue « babyphone » remplacée (position {i})")
            break
    else:
        idx = next(
            (i for i, v in enumerate(views) if v.get("path") == "lenaic"), len(views) - 1
        )
        views.insert(idx + 1, VIEW)
        print(f"  vue « babyphone » insérée (position {idx + 1})")

    with open(PATH, "w") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
