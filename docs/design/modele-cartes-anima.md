# Modèle de cartes Home Assistant — réponse d'Anima

> Réponse **verbatim** d'Anima (agent Hermes sur `lyra`), obtenue le 2026-08-16
> par `hermes -z` via SSH depuis Claude Code. Non éditée.

---

Oui. Le modèle maison à reprendre est celui-ci :

- Structure : `custom:stack-in-card` vertical, **un seul cadre externe**.
- Header : `custom:button-card` transparent, rendu HTML via `custom_fields`, hero en dégradé.
- Contenu interactif : vrais enfants `custom:button-card` dans un `grid`; **une tuile = une cible tactile = un `tap_action`**.
- Ne pas mettre de `custom:button-card` imbriqué dans le HTML de `custom_fields` : c’est précisément le pattern qui a produit les icônes géantes / texte écrasé sur la carte G6.
- Ne pas ajouter de second wrapper avec bordure/ombre dans le contenu : le `stack-in-card` est le seul cadre.

Références effectivement présentes :
- `Anima Alarm Hub` : hero gradient + conteneur sombre arrondi, rayon 22 px.
- `Anima Piscine - analyse eau` : `stack-in-card`, `keep.background/border_radius/box_shadow`, header transparent, puis enfants cliquables transparents.
- Le script `fix_alarmv2_g6_stack_in_card_pool_pattern.py` est le bon exemple d’assemblage, sauvegarde, remplacement idempotent et relecture.

Conventions à garder

```yaml
# Noms stables : indispensables pour une mise à jour/replacement future.
name: Anima Babyphone - surveillance
name: Anima Babyphone - header
name: Anima Babyphone - niveau sonore
name: Anima Babyphone - pic
name: Anima Babyphone - fond
name: Anima Babyphone - hypnogramme
name: Anima Babyphone - diagnostic
```

Palette sémantique :

```text
Base sombre / neutre : #111827, #164e63, #0f766e
Information :          #0ea5e9 / #38bdf8
OK / calme :           #16a34a / #22c55e
Attention / voix :     #f59e0b
Pleurs :               #f97316
Cri / incident :       #ef4444 / #dc2626
Indisponible :         #64748b
```

Dimensions récurrentes :

```text
Cadre principal : rayon 18 px (ou 22 px pour le modèle Alarm Hub)
Hero : padding 16 px
Icône hero : 44×44 px, rayon 16 px
Tuiles : rayon 14 px, padding 10 px
Espacement grille : 8 px
Badge/pill : rayon 999 px, padding 5×9 px
Ombre externe : 0 10px 30px rgba(0,0,0,.16)
Bordure externe : 1px solid rgba(127,127,127,.14)
```

Exemple complet, prêt à adapter. Remplacer uniquement les `sensor.babyphone_*` par les IDs exacts du projet ; ne pas inventer d’entités si leur registre diffère.

```yaml
type: custom:stack-in-card
name: Anima Babyphone - surveillance
mode: vertical
keep:
  background: true
  border_radius: true
  box_shadow: true
grid_options:
  columns: full
card_mod:
  style: |
    ha-card {
      overflow: hidden;
      border-radius: 18px;
      border: 1px solid rgba(127,127,127,.14);
      box-shadow: 0 10px 30px rgba(0,0,0,.16);
    }

cards:
  # HERO — la synthèse en un regard.
  - type: custom:button-card
    name: Anima Babyphone - header
    entity: sensor.babyphone_noise_level
    show_name: false
    show_icon: false
    show_state: false
    triggers_update:
      - sensor.babyphone_noise_level
      - sensor.babyphone_peak
      - sensor.babyphone_noise_floor
      - sensor.babyphone_threshold
      - sensor.babyphone_detected_sound
      - binary_sensor.babyphone
      - binary_sensor.babyphone_telemetry_online
      - sensor.babyphone_service_uptime
      - binary_sensor.babyphone_pi_undervoltage
    tap_action:
      action: more-info
      entity: sensor.babyphone_noise_level
    hold_action:
      action: more-info
      entity: sensor.babyphone_detected_sound
    styles:
      card:
        - padding: 0
        - background: transparent
        - box-shadow: none
        - border: none
        - border-radius: 0
      grid:
        - grid-template-areas: '"body"'
        - grid-template-columns: 1fr
      custom_fields:
        body:
          - width: 100%
    extra_styles: |
      .bp-hero {
        position: relative;
        overflow: hidden;
        padding: 16px;
        color: white;
        background: linear-gradient(135deg,#111827 0%,#164e63 52%,#0f766e 100%);
      }
      .bp-hero.warn {
        background: linear-gradient(135deg,#21170b 0%,#7c3f00 50%,#f08c00 100%);
      }
      .bp-hero.alert {
        background: linear-gradient(135deg,#27131a 0%,#7f1d1d 50%,#dc2626 100%);
      }
      .bp-hero:after {
        content: "";
        position: absolute;
        right: -42px;
        top: -55px;
        width: 155px;
        height: 155px;
        border-radius: 50%;
        background: rgba(255,255,255,.12);
      }
      .bp-top {
        position: relative;
        z-index: 1;
        display: flex;
        align-items: center;
        gap: 12px;
        min-width: 0;
      }
      .bp-icon {
        flex: 0 0 44px;
        display: flex;
        align-items: center;
        justify-content: center;
        width: 44px;
        height: 44px;
        border-radius: 16px;
        background: rgba(255,255,255,.18);
        backdrop-filter: blur(8px);
      }
      .bp-icon ha-icon { --mdc-icon-size: 24px; color: white; }
      .bp-title { flex: 1; min-width: 0; text-align: left; }
      .bp-title .main {
        overflow: hidden;
        font-size: 17px;
        font-weight: 750;
        text-overflow: ellipsis;
        white-space: nowrap;
      }
      .bp-title .sub {
        overflow: hidden;
        margin-top: 2px;
        font-size: 12px;
        opacity: .78;
        text-overflow: ellipsis;
        white-space: nowrap;
      }
      .bp-score {
        min-width: 78px;
        padding: 7px 10px;
        border-radius: 999px;
        background: rgba(255,255,255,.18);
        font-weight: 800;
        text-align: center;
      }
      .bp-score .num { font-size: 21px; line-height: 21px; white-space: nowrap; }
      .bp-score .lbl {
        font-size: 9px;
        letter-spacing: .08em;
        opacity: .72;
        text-transform: uppercase;
      }
      .bp-stats {
        position: relative;
        z-index: 1;
        display: grid;
        grid-template-columns: repeat(3,1fr);
        gap: 8px;
        margin-top: 14px;
      }
      .bp-stat {
        min-width: 0;
        padding: 9px 8px;
        border-radius: 14px;
        background: rgba(255,255,255,.13);
        text-align: left;
      }
      .bp-stat .val {
        overflow: hidden;
        font-size: 15px;
        font-weight: 750;
        text-overflow: ellipsis;
        white-space: nowrap;
      }
      .bp-stat .lbl {
        overflow: hidden;
        margin-top: 2px;
        font-size: 10px;
        letter-spacing: .06em;
        opacity: .72;
        text-overflow: ellipsis;
        text-transform: uppercase;
        white-space: nowrap;
      }
      @media (max-width:390px) {
        .bp-score { min-width: 62px; }
        .bp-title .main { font-size: 15px; }
        .bp-stats { gap: 7px; }
        .bp-stat .val { font-size: 13px; }
      }
    custom_fields:
      body: |
        [[[
          const E = id => states[id] || {state:'unknown', attributes:{}};
          const valid = v => ![undefined, null, '', 'unknown', 'unavailable'].includes(v);
          const number = (id, suffix='', digits=0) => {
            const n = Number(E(id).state);
            return Number.isFinite(n) ? n.toFixed(digits) + suffix : '—';
          };

          const sound = String(E('sensor.babyphone_detected_sound').state || 'unknown').toLowerCase();
          const online = E('binary_sensor.babyphone_telemetry_online').state === 'on';
          const lowVoltage = E('binary_sensor.babyphone_pi_undervoltage').state === 'on';

          let cls = '';
          let label = 'Calme';
          let icon = 'mdi:sleep';

          if (['cri', 'cry', 'scream'].some(x => sound.includes(x))) {
            cls = 'alert'; label = 'Cri détecté'; icon = 'mdi:alert';
          } else if (['pleur', 'crying'].some(x => sound.includes(x))) {
            cls = 'warn'; label = 'Pleurs détectés'; icon = 'mdi:baby-face-outline';
          } else if (sound.includes('voix') || sound.includes('voice')) {
            cls = 'warn'; label = 'Voix détectée'; icon = 'mdi:account-voice';
          } else if (!online || lowVoltage) {
            cls = 'warn'; label = !online ? 'Télémétrie hors ligne' : 'Sous-tension Pi';
            icon = !online ? 'mdi:lan-disconnect' : 'mdi:flash-alert';
          }

          const dbfs = number('sensor.babyphone_noise_level', ' dBFS', 1);
          const peak = number('sensor.babyphone_peak', ' dBFS', 1);
          const threshold = number('sensor.babyphone_threshold', ' dBFS', 1);

          return `<div class="bp-hero ${cls}">
            <div class="bp-top">
              <div class="bp-icon"><ha-icon icon="${icon}"></ha-icon></div>
              <div class="bp-title">
                <div class="main">Babyphone</div>
                <div class="sub">${label}</div>
              </div>
              <div class="bp-score">
                <div class="num">${online ? 'ON' : 'HS'}</div>
                <div class="lbl">audio</div>
              </div>
            </div>
            <div class="bp-stats">
              <div class="bp-stat"><div class="val">${dbfs}</div><div class="lbl">niveau</div></div>
              <div class="bp-stat"><div class="val">${peak}</div><div class="lbl">pic</div></div>
              <div class="bp-stat"><div class="val">${threshold}</div><div class="lbl">seuil</div></div>
            </div>
          </div>`;
        ]]]

  # TUILES — enfants réels, donc zones tactiles réellement indépendantes.
  - type: grid
    columns: 2
    square: false
    card_mod:
      style: |
        ha-card {
          display: grid;
          gap: 8px;
          padding: 11px 12px 0;
          border: none;
          background: transparent;
          box-shadow: none;
        }
    cards:
      - type: custom:button-card
        name: Anima Babyphone - niveau sonore
        entity: sensor.babyphone_noise_level
        show_name: false
        show_icon: false
        show_state: false
        tap_action:
          action: more-info
          entity: sensor.babyphone_noise_level
        styles:
          card:
            - padding: 0
            - background: transparent
            - box-shadow: none
            - border: none
            - border-radius: 0
          grid:
            - grid-template-areas: '"body"'
            - grid-template-columns: 1fr
          custom_fields:
            body:
              - width: 100%
        extra_styles: |
          .bp-tile {
            min-width: 0;
            padding: 10px;
            border: 1px solid rgba(127,127,127,.12);
            border-radius: 14px;
            background: rgba(127,127,127,.075);
            text-align: left;
          }
          .bp-tile .k { font-size: 12px; font-weight: 760; color: var(--primary-text-color); }
          .bp-tile .v { margin-top: 4px; font-size: 18px; font-weight: 800; color: #38bdf8; }
          .bp-tile .m {
            overflow: hidden;
            margin-top: 2px;
            font-size: 10px;
            color: var(--secondary-text-color);
            text-overflow: ellipsis;
            text-transform: uppercase;
            white-space: nowrap;
          }
        custom_fields:
          body: |
            [[[
              const n = Number(entity?.state);
              const v = Number.isFinite(n) ? n.toFixed(1)+' dBFS' : '—';
              return `<div class="bp-tile">
                <div class="k">Niveau sonore</div>
                <div class="v">${v}</div>
                <div class="m">pression acoustique courante</div>
              </div>`;
            ]]]

      - type: custom:button-card
        name: Anima Babyphone - pic
        entity: sensor.babyphone_peak
        show_name: false
        show_icon: false
        show_state: false
        tap_action:
          action: more-info
          entity: sensor.babyphone_peak
        styles:
          card: [{padding: 0}, {background: transparent}, {box-shadow: none}, {border: none}, {border-radius: 0}]
          grid: [{grid-template-areas: '"body"'}, {grid-template-columns: 1fr}]
          custom_fields: {body: [{width: 100%}]}
        extra_styles: |
          .bp-tile { min-width:0; padding:10px; border-radius:14px; background:rgba(245,158,11,.10); border:1px solid rgba(245,158,11,.22); text-align:left; }
          .bp-tile .k { font-size:12px; font-weight:760; color:var(--primary-text-color); }
          .bp-tile .v { margin-top:4px; font-size:18px; font-weight:800; color:#f59e0b; }
          .bp-tile .m { margin-top:2px; font-size:10px; color:var(--secondary-text-color); text-transform:uppercase; }
        custom_fields:
          body: |
            [[[
              const n = Number(entity?.state);
              return `<div class="bp-tile">
                <div class="k">Pic récent</div>
                <div class="v">${Number.isFinite(n) ? n.toFixed(1)+' dBFS' : '—'}</div>
                <div class="m">événement le plus fort</div>
              </div>`;
            ]]]

      - type: custom:button-card
        name: Anima Babyphone - fond
        entity: sensor.babyphone_noise_floor
        show_name: false
        show_icon: false
        show_state: false
        tap_action:
          action: more-info
          entity: sensor.babyphone_noise_floor
        styles:
          card: [{padding: 0}, {background: transparent}, {box-shadow: none}, {border: none}, {border-radius: 0}]
          grid: [{grid-template-areas: '"body"'}, {grid-template-columns: 1fr}]
          custom_fields: {body: [{width: 100%}]}
        extra_styles: |
          .bp-tile { min-width:0; padding:10px; border-radius:14px; background:rgba(16,185,129,.10); border:1px solid rgba(16,185,129,.22); text-align:left; }
          .bp-tile .k { font-size:12px; font-weight:760; color:var(--primary-text-color); }
          .bp-tile .v { margin-top:4px; font-size:18px; font-weight:800; color:#22c55e; }
          .bp-tile .m { margin-top:2px; font-size:10px; color:var(--secondary-text-color); text-transform:uppercase; }
        custom_fields:
          body: |
            [[[
              const n = Number(entity?.state);
              return `<div class="bp-tile">
                <div class="k">Fond sonore</div>
                <div class="v">${Number.isFinite(n) ? n.toFixed(1)+' dBFS' : '—'}</div>
                <div class="m">référence ambiante</div>
              </div>`;
            ]]]

      - type: custom:button-card
        name: Anima Babyphone - nature son
        entity: sensor.babyphone_detected_sound
        show_name: false
        show_icon: false
        show_state: false
        tap_action:
          action: more-info
          entity: sensor.babyphone_detected_sound
        styles:
          card: [{padding: 0}, {background: transparent}, {box-shadow: none}, {border: none}, {border-radius: 0}]
          grid: [{grid-template-areas: '"body"'}, {grid-template-columns: 1fr}]
          custom_fields: {body: [{width: 100%}]}
        extra_styles: |
          .bp-tile { min-width:0; padding:10px; border-radius:14px; background:rgba(127,127,127,.075); border:1px solid rgba(127,127,127,.12); text-align:left; }
          .bp-tile .k { font-size:12px; font-weight:760; color:var(--primary-text-color); }
          .bp-tile .v { overflow:hidden; margin-top:4px; font-size:18px; font-weight:800; text-overflow:ellipsis; white-space:nowrap; }
          .bp-tile .m { margin-top:2px; font-size:10px; color:var(--secondary-text-color); text-transform:uppercase; }
        custom_fields:
          body: |
            [[[
              const raw = String(entity?.state || 'unknown').toLowerCase();
              const isCry = raw.includes('cri');
              const isTears = raw.includes('pleur');
              const isVoice = raw.includes('voix');
              const color = isCry ? '#ef4444' : (isTears ? '#f97316' : (isVoice ? '#f59e0b' : '#22c55e'));
              const value = raw === 'unknown' || raw === 'unavailable' ? '—' : raw;
              return `<div class="bp-tile">
                <div class="k">Son détecté</div>
                <div class="v" style="color:${color}">${value}</div>
                <div class="m">classification audio</div>
              </div>`;
            ]]]

  # LIGNE UTILITAIRE — statut technique, sans encombrer le hero.
  - type: custom:button-card
    name: Anima Babyphone - diagnostic
    entity: binary_sensor.babyphone_telemetry_online
    show_name: false
    show_icon: false
    show_state: false
    triggers_update:
      - binary_sensor.babyphone_telemetry_online
      - sensor.babyphone_service_uptime
      - binary_sensor.babyphone_pi_undervoltage
    tap_action:
      action: more-info
      entity: binary_sensor.babyphone_telemetry_online
    hold_action:
      action: more-info
      entity: binary_sensor.babyphone_pi_undervoltage
    styles:
      card:
        - padding: 0
        - background: transparent
        - box-shadow: none
        - border: none
        - border-radius: 0
      grid:
        - grid-template-areas: '"body"'
        - grid-template-columns: 1fr
      custom_fields:
        body:
          - width: 100%
    extra_styles: |
      .bp-diag {
        display:grid;
        grid-template-columns:34px minmax(0,1fr) auto;
        align-items:center;
        gap:10px;
        padding:11px 14px;
        border-top:1px solid var(--divider-color);
        text-align:left;
      }
      .bp-diag .i {
        display:flex; align-items:center; justify-content:center;
        width:34px; height:34px; border-radius:12px;
        background:rgba(14,165,233,.12);
      }
      .bp-diag .i ha-icon { --mdc-icon-size:19px; color:#38bdf8; }
      .bp-diag.warn .i { background:rgba(245,158,11,.16); }
      .bp-diag.warn .i ha-icon { color:#f59e0b; }
      .bp-diag .k { overflow:hidden; font-size:13px; font-weight:760; text-overflow:ellipsis; white-space:nowrap; }
      .bp-diag .m { overflow:hidden; margin-top:2px; font-size:11px; color:var(--secondary-text-color); text-overflow:ellipsis; white-space:nowrap; }
      .bp-pill { padding:5px 9px; border-radius:999px; background:#16a34a; color:white; font-size:11px; font-weight:800; white-space:nowrap; }
      .bp-diag.warn .bp-pill { background:#f59e0b; }
    custom_fields:
      body: |
        [[[
          const online = states['binary_sensor.babyphone_telemetry_online']?.state === 'on';
          const undervoltage = states['binary_sensor.babyphone_pi_undervoltage']?.state === 'on';
          const uptime = states['sensor.babyphone_service_uptime']?.state || '—';
          const warn = !online || undervoltage;
          const status = !online ? 'Hors ligne' : (undervoltage ? 'Sous-tension' : 'En ligne');
          const meta = !online
            ? 'La télémétrie babyphone ne répond plus'
            : (undervoltage ? 'Vérifier alimentation et câble du Pi' : `Service actif depuis ${uptime}`);

          return `<div class="bp-diag ${warn ? 'warn' : ''}">
            <div class="i"><ha-icon icon="${warn ? 'mdi:alert-circle-outline' : 'mdi:heart-pulse'}"></ha-icon></div>
            <div><div class="k">Diagnostic</div><div class="m">${meta}</div></div>
            <div class="bp-pill">${status}</div>
          </div>`;
        ]]]

  # Le graphe reste une vraie carte : on ne cherche pas à le simuler en HTML.
  - type: history-graph
    title: Éveils et activité — 12 h
    hours_to_show: 12
    refresh_interval: 60
    entities:
      - entity: binary_sensor.lenaic_is_sleeping
        name: Sommeil
      - entity: sensor.babyphone_detected_sound
        name: Son détecté
      - entity: sensor.babyphone_noise_level
        name: Niveau sonore
    card_mod:
      style: |
        ha-card {
          margin: 0;
          padding: 0 8px 8px;
          border: none;
          background: transparent;
          box-shadow: none;
        }
```

Points non négociables pour Claude Code :

1. Chaque entité lue dans le JavaScript doit figurer dans `triggers_update` du header ; sinon la synthèse reste visuellement figée.
2. Garder le graphe/hypnogramme comme vraie carte HA, pas comme faux rendu HTML.
3. Les tuiles acoustiques sont des enfants du `grid`, avec leur propre `more-info`, pas des `div` cliquables fictives.
4. Faire une sauvegarde complète dans `17-HomeAssistant/lovelace_backups/` avant `lovelace/config/save`, puis recharger `lovelace/config` et vérifier chemin, structure et `tap_action`.
5. Ne pas ajouter de footer du type « Toucher pour… » : les cibles tactiles sont implicites et les libellés doivent rester compacts.
