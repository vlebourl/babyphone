#!/usr/bin/env bash
# Déploie les trois tiers du babyphone depuis le dépôt (voir docs/adr/0007).
#
#   ./deploy/deploy.sh          # tout
#   ./deploy/deploy.sh pi       # code + unité systemd sur le Raspberry Pi
#   ./deploy/deploy.sh ha       # package Home Assistant
#
# Prérequis : `ssh vlb@babyphone.local` et `ssh -p 22224 root@192.168.1.10`
# fonctionnels sans mot de passe.
set -euo pipefail

PI_HOST="vlb@babyphone.local"
PI_DIR="/home/vlb/babyphone"
HA_HOST="root@192.168.1.10"
HA_PORT="22224"
HA_PACKAGE="/config/includes/packages/babyphone_monitoring.yaml"

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TARGET="${1:-all}"

say() { printf '\n\033[1m▸ %s\033[0m\n' "$*"; }

check_pushed() {
    say "Vérification : tout est poussé ?"
    git -C "$REPO" fetch -q origin
    if [ -n "$(git -C "$REPO" status --porcelain)" ]; then
        echo "  ✗ modifications non commitées — commiter avant de déployer" >&2
        exit 1
    fi
    if [ "$(git -C "$REPO" rev-list --count origin/main..HEAD)" != "0" ]; then
        echo "  ✗ commits locaux non poussés — le Pi tire depuis GitHub" >&2
        exit 1
    fi
    echo "  ✓ dépôt propre et synchronisé"
}

deploy_pi() {
    say "Raspberry Pi — code, dépendances, service"
    scp -q "$REPO/deploy/babyphone.service" "$PI_HOST:/tmp/babyphone.service"
    ssh "$PI_HOST" bash -s <<'REMOTE'
set -euo pipefail
export PATH="$HOME/.local/bin:$PATH"
cd /home/vlb/babyphone
git pull --ff-only
uv sync --quiet
# L'unité n'est réinstallée que si elle a changé (évite un daemon-reload inutile)
if ! diff -q /tmp/babyphone.service /etc/systemd/system/babyphone.service >/dev/null 2>&1; then
    sudo cp /tmp/babyphone.service /etc/systemd/system/babyphone.service
    sudo systemctl daemon-reload
    echo "  unité systemd mise à jour"
fi
sudo systemctl enable --quiet babyphone.service
sudo systemctl restart babyphone.service
sleep 5
systemctl is-active --quiet babyphone.service \
    && echo "  ✓ service actif — $(git log --oneline -1)" \
    || { echo "  ✗ service en échec :"; sudo journalctl -u babyphone -n 20 --no-pager; exit 1; }
REMOTE
}

deploy_ha() {
    say "Home Assistant — package d'entités et automatisations"
    scp -q -P "$HA_PORT" "$REPO/deploy/homeassistant/babyphone_monitoring.yaml" \
        "$HA_HOST:$HA_PACKAGE"
    ssh -p "$HA_PORT" "$HA_HOST" bash -s <<REMOTE
set -euo pipefail
ha core check >/dev/null && echo "  ✓ configuration valide"
curl -sf -X POST -H "Authorization: Bearer \$SUPERVISOR_TOKEN" \
    http://supervisor/core/api/services/homeassistant/reload_all >/dev/null \
    && echo "  ✓ rechargé (entités et automatisations)"
REMOTE
}

case "$TARGET" in
    pi)  check_pushed; deploy_pi ;;
    ha)  deploy_ha ;;
    all) check_pushed; deploy_pi; deploy_ha ;;
    *)   echo "usage: $0 [all|pi|ha]" >&2; exit 2 ;;
esac

say "Déploiement terminé"
