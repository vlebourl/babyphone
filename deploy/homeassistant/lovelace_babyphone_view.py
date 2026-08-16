#!/usr/bin/env python3
"""Tableau de bord Babyphone — source de vérité (ADR-0007).

Suit le **modèle maison** transmis par Anima (docs/design/modele-cartes-anima.md) :

- un seul `custom:stack-in-card` vertical par bloc, qui est le seul cadre ;
- un hero `custom:button-card` transparent, rendu en HTML via `custom_fields` ;
- des tuiles qui sont de **vrais enfants** `custom:button-card` dans un `grid`,
  donc une tuile = une cible tactile = un `tap_action` ;
- jamais de `button-card` imbriqué dans le HTML d'un `custom_fields` — c'est le
  motif qui avait produit les icônes géantes sur la carte G6 ;
- jamais de second cadre (bordure/ombre) à l'intérieur du `stack-in-card` ;
- les graphes restent de **vraies cartes** HA, on ne les simule pas en HTML.

Publication par l'API websocket `lovelace/config/save` : écrire dans `.storage`
ne marche pas (Home Assistant garde sa copie mémoire et finit par l'écraser).
Une sauvegarde horodatée de la configuration précédente est déposée sur l'hôte
avant toute écriture.
"""

import base64
import datetime
import hashlib
import json
import os
import socket
import struct

WS_HOST, WS_PORT, WS_PATH = "supervisor", 80, "/core/websocket"
DASHBOARD = "lovelace-mobile"
BACKUP_DIR = "/config/lovelace_backups"

# ── Entités réelles du projet ────────────────────────────────────────────
# Le modèle d'Anima cite des identifiants indicatifs ; voici la
# correspondance avec ceux que le babyphone publie réellement.
NIVEAU = "sensor.babyphone_noise_level"
PIC = "sensor.babyphone_pic_sonore"          # Anima : babyphone_peak
FOND = "sensor.babyphone_fond_sonore"        # Anima : babyphone_noise_floor
SEUIL = "sensor.babyphone_threshold"
NATURE = "sensor.babyphone_nature_du_son"    # Anima : babyphone_detected_sound
EN_LIGNE = "binary_sensor.babyphone_en_ligne"  # Anima : babyphone_telemetry_online
UPTIME = "sensor.babyphone_duree_de_service"   # Anima : babyphone_service_uptime
# Anima suppose un binary_sensor ; le nôtre est un sensor « Oui »/« Non ».
SOUS_TENSION = "sensor.babyphone_sous_tension"
EVEIL = "input_boolean.lenaic_speaking"
ARMEE = "input_boolean.babyphone_on_off"
ACTIVITE = "sensor.babyphone_activite"
CENTROID = "sensor.babyphone_centre_spectral"
CALME_DEPUIS = "sensor.babyphone_calme_depuis"
REVEILS_NUIT = "sensor.babyphone_reveils_nuit"
REVEILS_24H = "sensor.babyphone_reveils_24h"
AGITATION_24H = "sensor.babyphone_agitation_24h"
ENDORMI = "sensor.babyphone_heure_endormissement"
SOMMEIL = "sensor.lenaic_night_asleep_duration"
REDEMARRAGES = "counter.babyphone_demarrages"

# ── Palette sémantique du modèle maison ──────────────────────────────────
INFO, INFO_CLAIR = "#0ea5e9", "#38bdf8"
CALME_V = "#22c55e"
VOIX_V = "#f59e0b"
PLEURS_V = "#f97316"
CRI_V = "#ef4444"

# Ossature commune à tous les button-card du modèle : carte transparente,
# grille à une seule zone, corps en pleine largeur.
NU = {
    "show_name": False,
    "show_icon": False,
    "show_state": False,
    "styles": {
        "card": [{"padding": "0"}, {"background": "transparent"},
                 {"box-shadow": "none"}, {"border": "none"}, {"border-radius": "0"}],
        "grid": [{"grid-template-areas": '"body"'}, {"grid-template-columns": "1fr"}],
        "custom_fields": {"body": [{"width": "100%"}]},
    },
}

# Style d'une tuile, paramétré par sa couleur d'accent.
TUILE_CSS = """
  .bp-tile {{
    min-width: 0;
    padding: 10px;
    border: 1px solid {teinte}38;
    border-radius: 14px;
    background: {teinte}1a;
    text-align: left;
  }}
  .bp-tile .k {{ font-size: 12px; font-weight: 760; color: var(--primary-text-color); }}
  .bp-tile .v {{
    overflow: hidden; margin-top: 4px; font-size: 18px; font-weight: 800;
    color: {teinte}; text-overflow: ellipsis; white-space: nowrap;
  }}
  .bp-tile .m {{
    overflow: hidden; margin-top: 2px; font-size: 10px;
    color: var(--secondary-text-color); text-overflow: ellipsis;
    text-transform: uppercase; white-space: nowrap;
  }}
"""


def tuile(entity, titre, legende, teinte, corps_js, couleur_dynamique=False):
    """Tuile acoustique : un vrai enfant du grid, avec sa propre cible tactile.

    `couleur_dynamique` laisse le JS choisir la teinte de la valeur — la
    maquette colore le mot « pleurs » ou « cri » lui-même, ce qui rend la
    tuile lisible sans lire l'étiquette.
    """
    return {
        "type": "custom:button-card",
        "name": f"Anima Babyphone - {titre.lower()}",
        "entity": entity,
        **NU,
        "tap_action": {"action": "more-info", "entity": entity},
        "extra_styles": TUILE_CSS.format(teinte=teinte),
        "custom_fields": {"body": f"""[[[
          {corps_js}
          return `<div class="bp-tile">
            <div class="k">{titre}</div>
            <div class="v"{{style}}>${{v}}</div>
            <div class="m">{legende}</div>
          </div>`;
        ]]]""".replace("${STYLE}",
                       ' style="color:${teinte}"' if couleur_dynamique else "")},
    }


def tuile_db(entity, titre, legende, teinte):
    return tuile(entity, titre, legende, teinte,
                 "const n = Number(entity?.state);"
                 " const v = Number.isFinite(n) ? n.toFixed(1)+' dB' : '—';")


# ══════════════════════════════════════════════════════════════════════
# BLOC 1 — Surveillance : hero, tuiles acoustiques, diagnostic
# ══════════════════════════════════════════════════════════════════════

HERO_CSS = """
  .bp-hero {
    position: relative; overflow: hidden; padding: 16px; color: white;
    background: linear-gradient(135deg,#111827 0%,#164e63 52%,#0f766e 100%);
  }
  .bp-hero.warn  { background: linear-gradient(135deg,#21170b 0%,#7c3f00 50%,#f08c00 100%); }
  .bp-hero.alert { background: linear-gradient(135deg,#27131a 0%,#7f1d1d 50%,#dc2626 100%); }
  .bp-hero:after {
    content: ""; position: absolute; right: -42px; top: -55px;
    width: 155px; height: 155px; border-radius: 50%; background: rgba(255,255,255,.12);
  }
  .bp-top { position: relative; z-index: 1; display: flex; align-items: center; gap: 12px; min-width: 0; }
  .bp-icon {
    flex: 0 0 44px; display: flex; align-items: center; justify-content: center;
    width: 44px; height: 44px; border-radius: 16px;
    background: rgba(255,255,255,.18); backdrop-filter: blur(8px);
  }
  .bp-icon ha-icon { --mdc-icon-size: 24px; color: white; }
  .bp-title { flex: 1; min-width: 0; text-align: left; }
  .bp-title .main { overflow: hidden; font-size: 17px; font-weight: 750; text-overflow: ellipsis; white-space: nowrap; }
  .bp-title .sub { overflow: hidden; margin-top: 2px; font-size: 12px; opacity: .78; text-overflow: ellipsis; white-space: nowrap; }
  .bp-score { min-width: 78px; padding: 7px 10px; border-radius: 999px; background: rgba(255,255,255,.18); font-weight: 800; text-align: center; }
  .bp-score .num { font-size: 21px; line-height: 21px; white-space: nowrap; }
  .bp-score .lbl { font-size: 9px; letter-spacing: .08em; opacity: .72; text-transform: uppercase; }
  .bp-stats { position: relative; z-index: 1; display: grid; grid-template-columns: repeat(3,1fr); gap: 8px; margin-top: 14px; }
  .bp-stat { min-width: 0; padding: 9px 8px; border-radius: 14px; background: rgba(255,255,255,.13); text-align: left; }
  .bp-stat .val { overflow: hidden; font-size: 15px; font-weight: 750; text-overflow: ellipsis; white-space: nowrap; }
  .bp-stat .lbl { overflow: hidden; margin-top: 2px; font-size: 10px; letter-spacing: .06em; opacity: .72; text-overflow: ellipsis; text-transform: uppercase; white-space: nowrap; }
  @media (max-width:390px) {
    .bp-score { min-width: 62px; }
    .bp-title .main { font-size: 15px; }
    .bp-stats { gap: 7px; }
    .bp-stat .val { font-size: 13px; }
  }
"""

HERO_JS = """[[[
  const E = id => states[id] || {state:'unknown', attributes:{}};
  const num = (id, unit='', d=0) => {
    const n = Number(E(id).state);
    return Number.isFinite(n) ? n.toFixed(d) + unit : '—';
  };

  const nature = String(E('%(NATURE)s').state || 'unknown').toLowerCase();
  const online = E('%(EN_LIGNE)s').state === 'on';
  const armee  = E('%(ARMEE)s').state === 'on';
  const eveil  = E('%(EVEIL)s').state === 'on';
  const sousTension = String(E('%(SOUS_TENSION)s').state) === 'Oui';

  let cls = '', label = 'Tout est calme', icon = 'mdi:sleep';

  if (nature.includes('cri')) {
    cls = 'alert'; label = 'Cri détecté'; icon = 'mdi:alert-octagram';
  } else if (nature.includes('pleur')) {
    cls = 'warn'; label = 'Pleurs détectés'; icon = 'mdi:emoticon-cry-outline';
  } else if (nature.includes('voix')) {
    cls = 'warn'; label = 'Voix détectée'; icon = 'mdi:account-voice';
  } else if (eveil) {
    cls = 'warn'; label = 'Réveil en cours'; icon = 'mdi:emoticon-cry-outline';
  } else if (!online) {
    cls = 'alert'; label = 'Télémétrie hors ligne'; icon = 'mdi:access-point-off';
  } else if (!armee) {
    cls = 'warn'; label = 'Surveillance désarmée'; icon = 'mdi:shield-off-outline';
  } else if (sousTension) {
    cls = 'warn'; label = 'Alimentation insuffisante'; icon = 'mdi:flash-alert';
  } else {
    const calme = Number(E('%(CALME_DEPUIS)s').state);
    if (Number.isFinite(calme)) label = `Calme depuis ${calme.toFixed(0)} min`;
  }

  return `<div class="bp-hero ${cls}">
    <div class="bp-top">
      <div class="bp-icon"><ha-icon icon="${icon}"></ha-icon></div>
      <div class="bp-title">
        <div class="main">Babyphone</div>
        <div class="sub">${label}</div>
      </div>
      <div class="bp-score">
        <div class="num">${online ? (armee ? 'ON' : 'VEILLE') : 'HS'}</div>
        <div class="lbl">audio</div>
      </div>
    </div>
    <div class="bp-stats">
      <div class="bp-stat"><div class="val">${num('%(NIVEAU)s',' dB',1)}</div><div class="lbl">niveau</div></div>
      <div class="bp-stat"><div class="val">${num('%(PIC)s',' dB',1)}</div><div class="lbl">pic</div></div>
      <div class="bp-stat"><div class="val">${num('%(SEUIL)s',' dB',1)}</div><div class="lbl">seuil</div></div>
    </div>
  </div>`;
]]]""" % {"NATURE": NATURE, "EN_LIGNE": EN_LIGNE, "ARMEE": ARMEE, "EVEIL": EVEIL,
          "SOUS_TENSION": SOUS_TENSION, "CALME_DEPUIS": CALME_DEPUIS,
          "NIVEAU": NIVEAU, "PIC": PIC, "SEUIL": SEUIL}

DIAG_CSS = """
  .bp-diag {
    display:grid; grid-template-columns:34px minmax(0,1fr) auto;
    align-items:center; gap:10px; padding:11px 14px;
    border-top:1px solid var(--divider-color); text-align:left;
  }
  .bp-diag .i { display:flex; align-items:center; justify-content:center; width:34px; height:34px; border-radius:12px; background:rgba(14,165,233,.12); }
  .bp-diag .i ha-icon { --mdc-icon-size:19px; color:#38bdf8; }
  .bp-diag.warn .i { background:rgba(245,158,11,.16); }
  .bp-diag.warn .i ha-icon { color:#f59e0b; }
  .bp-diag .k { overflow:hidden; font-size:13px; font-weight:760; text-overflow:ellipsis; white-space:nowrap; }
  .bp-diag .m { overflow:hidden; margin-top:2px; font-size:11px; color:var(--secondary-text-color); text-overflow:ellipsis; white-space:nowrap; }
  .bp-pill { padding:5px 9px; border-radius:999px; background:#16a34a; color:white; font-size:11px; font-weight:800; white-space:nowrap; }
  .bp-diag.warn .bp-pill { background:#f59e0b; }
"""

DIAG_JS = """[[[
  const E = id => states[id] || {state:'unknown'};
  const online = E('%(EN_LIGNE)s').state === 'on';
  const sousTension = String(E('%(SOUS_TENSION)s').state) === 'Oui';
  const boucles = Number(E('%(REDEMARRAGES)s').state) || 0;
  const s = Number(E('%(UPTIME)s').state);
  const uptime = Number.isFinite(s)
    ? (s < 3600 ? `${(s/60).toFixed(0)} min` : `${(s/3600).toFixed(1)} h`)
    : '—';

  const warn = !online || sousTension || boucles >= 3;
  const status = !online ? 'Hors ligne'
    : (boucles >= 3 ? 'Redémarre' : (sousTension ? 'Sous-tension' : 'En ligne'));
  const meta = !online ? 'La télémétrie ne répond plus'
    : (boucles >= 3 ? `${boucles} redémarrages récents, dispositif peu fiable`
    : (sousTension ? 'Alimentation insuffisante : bloc 5,1 V / 3 A requis'
    : `Service actif depuis ${uptime}`));

  return `<div class="bp-diag ${warn ? 'warn' : ''}">
    <div class="i"><ha-icon icon="${warn ? 'mdi:alert-circle-outline' : 'mdi:heart-pulse'}"></ha-icon></div>
    <div><div class="k">Diagnostic</div><div class="m">${meta}</div></div>
    <div class="bp-pill">${status}</div>
  </div>`;
]]]""" % {"EN_LIGNE": EN_LIGNE, "SOUS_TENSION": SOUS_TENSION,
          "REDEMARRAGES": REDEMARRAGES, "UPTIME": UPTIME}

SURVEILLANCE = {
    "type": "custom:stack-in-card",
    "name": "Anima Babyphone - surveillance",
    "mode": "vertical",
    "keep": {"background": True, "border_radius": True, "box_shadow": True},
    "grid_options": {"columns": "full"},
    "card_mod": {"style": (
        "ha-card {\n"
        "  overflow: hidden;\n"
        "  border-radius: 18px;\n"
        "  border: 1px solid rgba(127,127,127,.14);\n"
        "  box-shadow: 0 10px 30px rgba(0,0,0,.16);\n"
        "}\n")},
    "cards": [
        {
            "type": "custom:button-card",
            "name": "Anima Babyphone - header",
            "entity": NIVEAU,
            **NU,
            # Toute entité lue dans le JS doit figurer ici, sinon la synthèse
            # reste figée à l'écran (règle n°1 du modèle).
            "triggers_update": [NIVEAU, PIC, SEUIL, NATURE, EN_LIGNE, ARMEE,
                                EVEIL, SOUS_TENSION, CALME_DEPUIS],
            "tap_action": {"action": "more-info", "entity": NATURE},
            "hold_action": {"action": "more-info", "entity": NIVEAU},
            "extra_styles": HERO_CSS,
            "custom_fields": {"body": HERO_JS},
        },
        {
            "type": "grid",
            "columns": 2,
            "square": False,
            "card_mod": {"style": (
                "ha-card {\n"
                "  display: grid;\n  gap: 8px;\n  padding: 11px 12px 0;\n"
                "  border: none;\n  background: transparent;\n  box-shadow: none;\n"
                "}\n")},
            "cards": [
                tuile_db(NIVEAU, "Niveau sonore", "moyenne de la dernière seconde", INFO_CLAIR),
                tuile_db(PIC, "Pic récent", "bloc le plus fort vu par le détecteur", VOIX_V),
                tuile_db(FOND, "Fond sonore", "référence ambiante de la pièce", CALME_V),
                tuile(NATURE, "Son détecté", "signature spectrale", INFO_CLAIR,
                      "const raw = String(entity?.state || 'unknown').toLowerCase();"
                      " const v = ['unknown','unavailable'].includes(raw) ? '—' : raw;"
                      " const teinte = raw.includes('cri') ? '%s'"
                      " : (raw.includes('pleur') ? '%s'"
                      " : (raw.includes('voix') ? '%s'"
                      " : (raw.includes('bruit') ? '#64748b' : '%s')));" % (CRI_V, PLEURS_V, VOIX_V, CALME_V),
                      couleur_dynamique=True),
            ],
        },
        {
            "type": "custom:button-card",
            "name": "Anima Babyphone - diagnostic",
            "entity": EN_LIGNE,
            **NU,
            "triggers_update": [EN_LIGNE, UPTIME, SOUS_TENSION, REDEMARRAGES],
            "tap_action": {"action": "more-info", "entity": EN_LIGNE},
            "hold_action": {"action": "more-info", "entity": SOUS_TENSION},
            "extra_styles": DIAG_CSS,
            "custom_fields": {"body": DIAG_JS},
        },
    ],
}

# ══════════════════════════════════════════════════════════════════════
# BLOC 2 — La nuit : chiffres puis hypnogramme, en vraies cartes
# ══════════════════════════════════════════════════════════════════════

NUIT_JS = """[[[
  const E = id => states[id] || {state:'unknown'};
  const heure = s => {
    const t = E(s).state;
    if (['unknown','unavailable','none',''].includes(t)) return '—';
    const d = new Date(t);
    return isNaN(d) ? '—' : d.toLocaleTimeString('fr-FR',{hour:'2-digit',minute:'2-digit'});
  };
  const n = (id,d=0) => { const v = Number(E(id).state); return Number.isFinite(v) ? v.toFixed(d) : '—'; };
  const reveils = Number(E('%(REVEILS_NUIT)s').state) || 0;
  const teinte = reveils < 3 ? '#22c55e' : (reveils < 8 ? '#f59e0b' : '#ef4444');

  return `<div class="bp-night">
    <div class="bp-ncell"><div class="nv">${heure('%(ENDORMI)s')}</div><div class="nl">endormi à</div></div>
    <div class="bp-ncell"><div class="nv" style="color:${teinte}">${reveils}</div><div class="nl">réveils</div></div>
    <div class="bp-ncell"><div class="nv">${n('%(SOMMEIL)s',1)} h</div><div class="nl">sommeil</div></div>
    <div class="bp-ncell"><div class="nv">${n('%(CALME_DEPUIS)s')} min</div><div class="nl">calme depuis</div></div>
  </div>`;
]]]""" % {"REVEILS_NUIT": REVEILS_NUIT, "ENDORMI": ENDORMI,
          "SOMMEIL": SOMMEIL, "CALME_DEPUIS": CALME_DEPUIS}

NUIT = {
    "type": "custom:stack-in-card",
    "name": "Anima Babyphone - nuit",
    "mode": "vertical",
    "keep": {"background": True, "border_radius": True, "box_shadow": True},
    "grid_options": {"columns": "full"},
    "card_mod": {"style": (
        "ha-card {\n  overflow: hidden;\n  border-radius: 18px;\n"
        "  border: 1px solid rgba(127,127,127,.14);\n"
        "  box-shadow: 0 10px 30px rgba(0,0,0,.16);\n}\n")},
    "cards": [
        {
            "type": "custom:button-card",
            "name": "Anima Babyphone - nuit chiffres",
            "entity": REVEILS_NUIT,
            **NU,
            "triggers_update": [REVEILS_NUIT, ENDORMI, SOMMEIL, CALME_DEPUIS],
            "tap_action": {"action": "more-info", "entity": REVEILS_NUIT},
            "extra_styles": """
              .bp-night {
                display: grid; grid-template-columns: repeat(4,1fr); gap: 8px;
                padding: 14px 14px 4px;
              }
              .bp-ncell { min-width: 0; padding: 9px 8px; border-radius: 14px; background: rgba(127,127,127,.075); text-align: left; }
              .bp-ncell .nv { overflow: hidden; font-size: 16px; font-weight: 800; text-overflow: ellipsis; white-space: nowrap; }
              .bp-ncell .nl { overflow: hidden; margin-top: 2px; font-size: 10px; letter-spacing: .06em; color: var(--secondary-text-color); text-overflow: ellipsis; text-transform: uppercase; white-space: nowrap; }
              @media (max-width:390px) { .bp-night { grid-template-columns: repeat(2,1fr); } }
            """,
            "custom_fields": {"body": NUIT_JS},
        },
        {
            # LA carte qui répond à « quand Lenaïc parle-t-il ? ». Une frise
            # d'états : chaque bande colorée est une période, et son étiquette
            # dit la nature du son. Une tuile ne dit que l'instant présent ;
            # c'est cette frise qui donne l'histoire de la nuit.
            "type": "history-graph",
            "title": "Quand Lenaïc s'est manifesté",
            "hours_to_show": 14,
            "show_names": True,
            "entities": [
                {"entity": NATURE, "name": "Nature du son"},
                {"entity": EVEIL, "name": "Réveil"},
            ],
            "card_mod": {"style": (
                "ha-card {\n  margin: 0;\n  padding: 4px 8px 0;\n"
                "  border: none;\n  background: transparent;\n  box-shadow: none;\n}\n")},
        },
        {
            # Une bande colorée ne dit rien sans sa clé de lecture : la frise
            # de Home Assistant ne nomme ses états qu'au survol, inutilisable
            # sur mobile. La légende est statique, donc sans coût de calcul.
            "type": "custom:button-card",
            "name": "Anima Babyphone - legende",
            **NU,
            "tap_action": {"action": "none"},
            "extra_styles": """
              .bp-leg { display: flex; flex-wrap: wrap; gap: 6px 14px; padding: 2px 14px 12px; }
              .bp-leg span { display: inline-flex; align-items: center; gap: 6px; font-size: 11px; color: var(--secondary-text-color); }
              .bp-leg i { width: 10px; height: 10px; border-radius: 3px; display: inline-block; }
            """,
            "custom_fields": {"body": (
                '[[[ return `<div class="bp-leg">'
                '<span><i style="background:%s"></i>calme</span>'
                '<span><i style="background:%s"></i>voix</span>'
                '<span><i style="background:%s"></i>pleurs</span>'
                '<span><i style="background:%s"></i>cri</span>'
                '<span><i style="background:#64748b"></i>bruit</span>'
                '</div>`; ]]]' % (CALME_V, VOIX_V, PLEURS_V, CRI_V))},
        },
        {
            # Vraie carte : on ne simule pas un graphe en HTML (règle n°2).
            "type": "custom:apexcharts-card",
            "header": {"show": True, "title": "Intensité des éveils", "show_states": False},
            "graph_span": "14h",
            "span": {"start": "day", "offset": "-4h"},
            "apex_config": {
                "chart": {"height": 120},
                "stroke": {"curve": "stepline", "width": 2},
                "fill": {"type": "solid", "opacity": 0.4},
                "legend": {"show": False},
                "yaxis": {"show": False, "min": 0, "max": 1},
            },
            "series": [{
                "entity": EVEIL, "name": "Éveil", "type": "area", "color": CRI_V,
                "transform": "return x === 'on' ? 1 : 0;", "extend_to": "now",
                "group_by": {"func": "max", "duration": "1min"},
            }],
            "card_mod": {"style": (
                "ha-card {\n  margin: 0;\n  padding: 0 8px 8px;\n"
                "  border: none;\n  background: transparent;\n  box-shadow: none;\n}\n")},
        },
    ],
}

# ══════════════════════════════════════════════════════════════════════
# BLOC 3 — Télémétrie et tendances
# ══════════════════════════════════════════════════════════════════════

TELEMETRIE = {
    "type": "custom:stack-in-card",
    "name": "Anima Babyphone - telemetrie",
    "mode": "vertical",
    "keep": {"background": True, "border_radius": True, "box_shadow": True},
    "grid_options": {"columns": "full"},
    "card_mod": {"style": (
        "ha-card {\n  overflow: hidden;\n  border-radius: 18px;\n"
        "  border: 1px solid rgba(127,127,127,.14);\n"
        "  box-shadow: 0 10px 30px rgba(0,0,0,.16);\n}\n")},
    "cards": [
        {
            "type": "custom:apexcharts-card",
            "header": {"show": True, "title": "Niveau, pic et seuil — 2 h",
                       "show_states": True, "colorize_states": True},
            "graph_span": "2h",
            "apex_config": {"chart": {"height": 210},
                            "stroke": {"width": [1, 2, 2]},
                            "legend": {"show": True}},
            "yaxis": [{"decimals": 0}],
            "series": [
                {"entity": PIC, "name": "Pic", "type": "area", "color": INFO_CLAIR,
                 "opacity": 0.2, "group_by": {"func": "max", "duration": "30s"}},
                {"entity": NIVEAU, "name": "Moyenne", "color": INFO,
                 "group_by": {"func": "avg", "duration": "30s"}},
                {"entity": SEUIL, "name": "Seuil", "color": VOIX_V,
                 "group_by": {"func": "avg", "duration": "30s"}},
            ],
            "card_mod": {"style": (
                "ha-card {\n  margin: 0;\n  padding: 4px 8px 0;\n"
                "  border: none;\n  background: transparent;\n  box-shadow: none;\n}\n")},
        },
        {
            "type": "grid",
            "columns": 2,
            "square": False,
            "card_mod": {"style": (
                "ha-card {\n  display: grid;\n  gap: 8px;\n  padding: 8px 12px 12px;\n"
                "  border: none;\n  background: transparent;\n  box-shadow: none;\n}\n")},
            "cards": [
                tuile(ACTIVITE, "Blocs bruyants", "part au-dessus du seuil", INFO_CLAIR,
                      "const n = Number(entity?.state);"
                      " const v = Number.isFinite(n) ? n.toFixed(0)+' %' : '—';"),
                tuile(CENTROID, "Centre spectral", "aigu = plutôt un pleur", PLEURS_V,
                      "const n = Number(entity?.state);"
                      " const v = Number.isFinite(n) ? n.toFixed(0)+' Hz' : '—';"),
            ],
        },
    ],
}

TENDANCES = {
    "type": "custom:stack-in-card",
    "name": "Anima Babyphone - tendances",
    "mode": "vertical",
    "keep": {"background": True, "border_radius": True, "box_shadow": True},
    "grid_options": {"columns": "full"},
    "card_mod": {"style": (
        "ha-card {\n  overflow: hidden;\n  border-radius: 18px;\n"
        "  border: 1px solid rgba(127,127,127,.14);\n"
        "  box-shadow: 0 10px 30px rgba(0,0,0,.16);\n}\n")},
    "cards": [
        {
            "type": "custom:apexcharts-card",
            "header": {"show": True, "title": "Sept derniers jours", "show_states": False},
            "graph_span": "7d",
            "span": {"start": "day"},
            "apex_config": {"chart": {"height": 190}, "legend": {"show": True}},
            "series": [
                {"entity": REVEILS_24H, "name": "Réveils", "type": "column",
                 "color": VOIX_V, "group_by": {"func": "max", "duration": "1d"}},
                {"entity": AGITATION_24H, "name": "Agitation (h)", "type": "column",
                 "color": INFO, "group_by": {"func": "max", "duration": "1d"}},
            ],
            "card_mod": {"style": (
                "ha-card {\n  margin: 0;\n  padding: 4px 8px 8px;\n"
                "  border: none;\n  background: transparent;\n  box-shadow: none;\n}\n")},
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
        {"type": "entity", "show_name": False, "show_state": True,
         "show_icon": True, "entity": EN_LIGNE},
        {"type": "entity", "show_name": False, "show_state": True,
         "show_icon": True, "entity": EVEIL},
    ],
    "cards": [],
    "sections": [
        {"type": "grid", "cards": [SURVEILLANCE]},
        # La frise est « la carte la plus importante » de la maquette : elle
        # passe donc avant les chiffres de la nuit, pas après.
        {"type": "grid", "cards": [NUIT]},
        {"type": "grid", "cards": [TELEMETRIE]},
        {"type": "grid", "cards": [TENDANCES]},
    ],
}


class Websocket:
    """Client websocket minimal, sur la bibliothèque standard."""

    def __init__(self, host, port, path):
        self.sock = socket.create_connection((host, port), timeout=30)
        key = base64.b64encode(os.urandom(16)).decode()
        self.sock.sendall(
            f"GET {path} HTTP/1.1\r\nHost: {host}\r\nUpgrade: websocket\r\n"
            f"Connection: Upgrade\r\nSec-WebSocket-Key: {key}\r\n"
            f"Sec-WebSocket-Version: 13\r\n\r\n".encode()
        )
        expected = base64.b64encode(
            hashlib.sha1((key + "258EAFA5-E914-47DA-95CA-C5AB0DC85B11").encode()).digest()
        ).decode()
        head = b""
        while b"\r\n\r\n" not in head:
            head += self.sock.recv(1)
        assert expected in head.decode(errors="ignore"), "poignée de main refusée"
        self._buf = b""

    def _recv_exact(self, n):
        while len(self._buf) < n:
            chunk = self.sock.recv(65536)
            if not chunk:
                raise ConnectionError("connexion fermée")
            self._buf += chunk
        out, self._buf = self._buf[:n], self._buf[n:]
        return out

    def send(self, obj):
        data = json.dumps(obj).encode()
        header = bytearray([0x81])
        n = len(data)
        if n < 126:
            header.append(0x80 | n)
        elif n < 65536:
            header.append(0x80 | 126)
            header += struct.pack(">H", n)
        else:
            header.append(0x80 | 127)
            header += struct.pack(">Q", n)
        mask = os.urandom(4)
        header += mask
        self.sock.sendall(bytes(header) + bytes(b ^ mask[i % 4] for i, b in enumerate(data)))

    def recv(self):
        payload = b""
        while True:
            b0, b1 = self._recv_exact(2)
            n = b1 & 0x7F
            if n == 126:
                n = struct.unpack(">H", self._recv_exact(2))[0]
            elif n == 127:
                n = struct.unpack(">Q", self._recv_exact(8))[0]
            payload += self._recv_exact(n)
            if b0 & 0x80:
                return json.loads(payload)

    def close(self):
        try:
            self.sock.close()
        except OSError:
            pass


def main():
    ws = Websocket(WS_HOST, WS_PORT, WS_PATH)
    ws.recv()
    ws.send({"type": "auth", "access_token": os.environ["SUPERVISOR_TOKEN"]})
    if ws.recv().get("type") != "auth_ok":
        raise SystemExit("  authentification websocket refusée")

    ws.send({"id": 1, "type": "lovelace/config", "url_path": DASHBOARD})
    r = ws.recv()
    assert r.get("success"), r
    config = r["result"]

    # Sauvegarde avant écriture (règle n°4 du modèle maison).
    os.makedirs(BACKUP_DIR, exist_ok=True)
    stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    chemin = f"{BACKUP_DIR}/lovelace-mobile-{stamp}.json"
    with open(chemin, "w") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)
    print(f"  sauvegarde : {chemin}")

    views = config["views"]
    views[:] = [v for v in views if v.get("path") != "babyphone-systeme"]
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

    ws.send({"id": 2, "type": "lovelace/config/save",
             "url_path": DASHBOARD, "config": config})
    r = ws.recv()
    ws.close()
    if not r.get("success"):
        raise SystemExit(f"  échec de la publication : {r}")
    print("  vue publiée par l'API (effet immédiat, sans redémarrage)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
