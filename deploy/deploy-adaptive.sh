#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────────────────────
# deploy-adaptive.sh — Déploie le Bot Adaptive Bull sur le VPS
#
# Usage : bash deploy/deploy-adaptive.sh
# ──────────────────────────────────────────────────────────────────────────────
set -euo pipefail

VPS_HOST="${VPS_HOST:-BOT-VPS}"
APP_DIR="/opt/tradex"
SERVICE="tradex-adaptive"
PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"

echo "══════════════════════════════════════════════════"
echo "  📈 Déploiement TradeX ADAPTIVE BULL → $VPS_HOST"
echo "══════════════════════════════════════════════════"
echo ""

# ── 1. Sync des fichiers ──────────────────────────────────────────────────
echo "📦 Synchronisation des fichiers..."
rsync -avz --delete \
    --filter='P .venv*/' \
    --filter='P data/' \
    --filter='P logs/' \
    --filter='P .env' \
    --filter='P private.pem' \
    --filter='P public.pem' \
    --filter='P firebase-credentials.json' \
    --filter='P firebase-key.json' \
    --exclude='.venv*' \
    --exclude='__pycache__' \
    --exclude='.git' \
    --exclude='.pytest_cache' \
    --exclude='*.pyc' \
    --exclude='.env' \
    --exclude='private.pem' \
    --exclude='public.pem' \
    --exclude='firebase-credentials.json' \
    --exclude='firebase-key.json' \
    --exclude='.DS_Store' \
    --exclude='logs/' \
    --exclude='data/' \
    "$PROJECT_DIR/" "$VPS_HOST:$APP_DIR/"

echo "   Fichiers synchronisés ✅"

# ── 2. Remote : service systemd + démarrage ───────────────────────────────
echo ""
echo "🔧 Installation service + démarrage..."
ssh "$VPS_HOST" << 'REMOTE'
    set -e
    cd /opt/tradex

    # Dépendances
    .venv/bin/pip install -r requirements.txt -q 2>/dev/null

    # Service systemd
    sudo cp deploy/tradex-adaptive.service /etc/systemd/system/
    sudo systemctl daemon-reload
    sudo systemctl enable tradex-adaptive

    # Redémarrer si déjà actif, sinon démarrer
    if sudo systemctl is-active --quiet tradex-adaptive; then
        sudo systemctl restart tradex-adaptive
        echo "   Service redémarré ✅"
    else
        sudo systemctl start tradex-adaptive
        echo "   Service démarré ✅"
    fi

    sleep 2
    sudo systemctl status tradex-adaptive --no-pager -l || true
REMOTE

echo ""
echo "══════════════════════════════════════════════════"
echo "  ✅ Adaptive Bull déployé avec succès !"
echo "  📜 Logs : ssh BOT-VPS 'sudo journalctl -u tradex-adaptive -f'"
echo "══════════════════════════════════════════════════"
