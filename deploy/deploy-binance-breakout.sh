#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────────────────────
# deploy-binance-breakout.sh — Déploie le bot Breakout sur le VPS
#
# Usage depuis ta machine locale :
#   bash deploy/deploy-binance-breakout.sh
#
# Déploie le code + redémarre le service tradex-binance-breakout
# Le bot RANGE (tradex-binance) est aussi redémarré (car le code est partagé)
# ──────────────────────────────────────────────────────────────────────────────
set -euo pipefail

# ── Configuration ──────────────────────────────────────────────────────────
VPS_HOST="${VPS_HOST:-BOT-VPS}"
APP_DIR="/opt/tradex"
SERVICE_RANGE="tradex-binance"
SERVICE_BREAKOUT="tradex-binance-breakout"
SERVICE_BREAKOUT_DASH="tradex-binance-breakout-dashboard"

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"

echo "══════════════════════════════════════════════════"
echo "  🟡 Déploiement de TradeX BINANCE BREAKOUT → $VPS_HOST"
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

# ── 2. Installer services systemd + dépendances ─────────────────────────
echo ""
echo "🔧 Configuration systemd + dépendances..."
ssh "$VPS_HOST" << 'REMOTE'
    set -e
    cd /opt/tradex

    # Installer les dépendances
    .venv/bin/pip install -r requirements.txt -q 2>/dev/null

    # Copier les services systemd
    sudo cp deploy/tradex-binance.service /etc/systemd/system/
    sudo cp deploy/tradex-binance-breakout.service /etc/systemd/system/
    sudo cp deploy/tradex-binance-breakout-dashboard.service /etc/systemd/system/
    sudo systemctl daemon-reload

    # Activer les services
    sudo systemctl enable tradex-binance-breakout
    sudo systemctl enable tradex-binance-breakout-dashboard

    # Permissions
    sudo chown -R tradex:tradex /opt/tradex

    # Créer le dossier data si nécessaire
    sudo mkdir -p /opt/tradex/data
    sudo chown tradex:tradex /opt/tradex/data

    # Redémarrer les deux bots Binance (code partagé)
    sudo systemctl restart tradex-binance
    sudo systemctl restart tradex-binance-breakout
    sudo systemctl restart tradex-binance-breakout-dashboard

    sleep 2

    # Vérifications
    echo ""
    echo "── État des services Binance ──"

    if sudo systemctl is-active --quiet tradex-binance; then
        echo "   ✅ tradex-binance (RANGE) : actif"
    else
        echo "   ❌ tradex-binance (RANGE) : erreur"
        sudo journalctl -u tradex-binance -n 5 --no-pager
    fi

    if sudo systemctl is-active --quiet tradex-binance-breakout; then
        echo "   ✅ tradex-binance-breakout : actif"
    else
        echo "   ❌ tradex-binance-breakout : erreur"
        sudo journalctl -u tradex-binance-breakout -n 10 --no-pager
    fi

    if sudo systemctl is-active --quiet tradex-binance-breakout-dashboard; then
        echo "   ✅ tradex-binance-breakout-dashboard : actif (port 8504)"
    else
        echo "   ❌ tradex-binance-breakout-dashboard : erreur"
        sudo journalctl -u tradex-binance-breakout-dashboard -n 5 --no-pager
    fi

    echo ""
    echo "── Résumé des services TradeX ──"
    for svc in tradex tradex-binance tradex-binance-breakout tradex-binance-dashboard tradex-binance-breakout-dashboard; do
        if sudo systemctl is-active --quiet "$svc"; then
            echo "   ✅ $svc"
        else
            echo "   ⚪ $svc (inactif)"
        fi
    done
REMOTE

echo ""
echo "══════════════════════════════════════════════════"
echo "  ✅ Déploiement Binance Breakout terminé !"
echo "══════════════════════════════════════════════════"
echo ""
echo "  Logs RANGE   : ssh $VPS_HOST 'sudo journalctl -u tradex-binance -f'"
echo "  Logs BREAKOUT: ssh $VPS_HOST 'sudo journalctl -u tradex-binance-breakout -f'"
echo "  Dashboard    : http://213.199.41.168:8504"
echo ""
