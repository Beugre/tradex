#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────────────────────
# deploy-binance.sh — Déploie / met à jour TradeX Binance sur le VPS
#
# Usage depuis ta machine locale :
#   bash deploy/deploy-binance.sh
#
# Déploie le même code mais redémarre le service tradex-binance
# (le bot Revolut X est indépendant via deploy.sh / tradex.service)
# ──────────────────────────────────────────────────────────────────────────────
set -euo pipefail

# ── Configuration ──────────────────────────────────────────────────────────
VPS_HOST="${VPS_HOST:-BOT-VPS}"
APP_DIR="/opt/tradex"
SERVICE_BOT="tradex-binance"
SERVICE_DASH="tradex-binance-dashboard"

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"

echo "══════════════════════════════════════════════════"
echo "  🟡 Déploiement de TradeX BINANCE → $VPS_HOST"
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
    --exclude='firebase-credentials.json' \
    "$PROJECT_DIR/" "$VPS_HOST:$APP_DIR/"

echo "   Fichiers synchronisés ✅"

# ── 2. Installer services systemd si nécessaire ─────────────────────────
echo ""
echo "🔧 Configuration systemd + dépendances..."
ssh "$VPS_HOST" << 'REMOTE'
    set -e
    cd /opt/tradex

    # Installer les dépendances
    .venv/bin/pip install -r requirements.txt -q 2>/dev/null

    # Copier les services systemd si mis à jour
    sudo cp deploy/tradex-binance.service /etc/systemd/system/
    sudo cp deploy/tradex-binance-dashboard.service /etc/systemd/system/
    sudo systemctl daemon-reload

    # Activer les services
    sudo systemctl enable tradex-binance
    sudo systemctl enable tradex-binance-dashboard

    # Permissions
    sudo chown -R tradex:tradex /opt/tradex

    # Créer le dossier data si nécessaire
    sudo mkdir -p /opt/tradex/data
    sudo chown tradex:tradex /opt/tradex/data

    # Redémarrer le bot Binance
    sudo systemctl restart tradex-binance

    # Redémarrer le dashboard Binance
    sudo systemctl restart tradex-binance-dashboard

    sleep 2

    # Vérifications
    echo ""
    echo "── État des services Binance ──"

    if sudo systemctl is-active --quiet tradex-binance; then
        echo "   ✅ tradex-binance : actif"
    else
        echo "   ❌ tradex-binance : erreur"
        sudo journalctl -u tradex-binance -n 10 --no-pager
    fi

    if sudo systemctl is-active --quiet tradex-binance-dashboard; then
        echo "   ✅ tradex-binance-dashboard : actif (port 8503)"
    else
        echo "   ❌ tradex-binance-dashboard : erreur"
        sudo journalctl -u tradex-binance-dashboard -n 10 --no-pager
    fi

    echo ""
    echo "── État de TOUS les services TradeX ──"
    sudo systemctl status tradex --no-pager -l || true
    echo "---"
    sudo systemctl status tradex-binance --no-pager -l || true
REMOTE

echo ""
echo "══════════════════════════════════════════════════"
echo "  ✅ Déploiement Binance terminé !"
echo "══════════════════════════════════════════════════"
echo ""
echo "  Logs bot     : ssh $VPS_HOST 'sudo journalctl -u tradex-binance -f'"
echo "  Logs dashboard: ssh $VPS_HOST 'sudo journalctl -u tradex-binance-dashboard -f'"
echo "  Dashboard    : http://213.199.41.168:8503"
echo ""
