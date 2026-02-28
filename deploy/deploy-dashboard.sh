#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────────────────────
# deploy-dashboard.sh — Déploie le dashboard unifié et arrête les 3 anciens
#
# Usage :  bash deploy/deploy-dashboard.sh
# ──────────────────────────────────────────────────────────────────────────────
set -euo pipefail

VPS_HOST="${VPS_HOST:-BOT-VPS}"
APP_DIR="/opt/tradex"
PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"

echo "══════════════════════════════════════════════════"
echo "  🚀 Déploiement du Dashboard Unifié → $VPS_HOST"
echo "══════════════════════════════════════════════════"
echo ""

# ── 1. Sync des fichiers ────────────────────────────────────────────────
echo "📦 Synchronisation des fichiers..."
rsync -avz --delete \
    --exclude='.venv' \
    --exclude='__pycache__' \
    --exclude='.git' \
    --exclude='.pytest_cache' \
    --exclude='*.pyc' \
    --exclude='.env' \
    --exclude='private.pem' \
    --exclude='public.pem' \
    --exclude='test_*.py' \
    --exclude='preflight.py' \
    --exclude='.DS_Store' \
    --exclude='logs/' \
    --exclude='data/' \
    "$PROJECT_DIR/" "$VPS_HOST:$APP_DIR/"

echo "   Fichiers synchronisés ✅"

# ── 2. Stopper les 3 anciens dashboards, installer le nouveau ────────────
echo ""
echo "🔧 Migration des services dashboard..."
ssh "$VPS_HOST" << 'REMOTE'
    set -e
    cd /opt/tradex

    echo "   ⏹ Arrêt des 3 anciens dashboards..."
    sudo systemctl stop tradex-dashboard tradex-binance-dashboard tradex-binance-breakout-dashboard 2>/dev/null || true
    sudo systemctl disable tradex-dashboard tradex-binance-dashboard tradex-binance-breakout-dashboard 2>/dev/null || true

    echo "   📝 Installation du service unifié..."
    sudo cp deploy/tradex-dashboard-unified.service /etc/systemd/system/
    sudo systemctl daemon-reload
    sudo systemctl enable tradex-dashboard-unified
    sudo systemctl restart tradex-dashboard-unified

    sleep 3
    if sudo systemctl is-active --quiet tradex-dashboard-unified; then
        echo "   ✅ Dashboard unifié démarré sur le port 8502"
    else
        echo "   ❌ Erreur au démarrage — voir: sudo journalctl -u tradex-dashboard-unified -n 30"
        exit 1
    fi

    echo ""
    echo "   📊 Statut mémoire :"
    ps aux | grep streamlit | grep -v grep | awk '{printf "     PID %s → RSS %s MB\n", $2, $6/1024}'
REMOTE

echo ""
echo "══════════════════════════════════════════════════"
echo "  ✅ Dashboard unifié déployé !"
echo "══════════════════════════════════════════════════"
echo ""
echo "  🌐 URL : http://213.199.41.168:8502"
echo "  📋 Logs : ssh $VPS_HOST 'sudo journalctl -u tradex-dashboard-unified -f'"
echo ""
