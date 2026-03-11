#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────────────────────
# deploy-binance-crashbot.sh — Déploie le bot CrashBot (Dip Buy) sur le VPS
#
# Usage depuis ta machine locale :
#   bash deploy/deploy-binance-crashbot.sh
#
# Déploie le code + redémarre le service tradex-binance-crashbot
# Le bot RANGE (tradex-binance) est aussi redémarré (car le code est partagé)
# ──────────────────────────────────────────────────────────────────────────────
set -euo pipefail

# ── Configuration ──────────────────────────────────────────────────────────
VPS_HOST="${VPS_HOST:-BOT-VPS}"
APP_DIR="/opt/tradex"
SERVICE_RANGE="tradex-binance"
SERVICE_CRASHBOT="tradex-binance-crashbot"

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"

echo "══════════════════════════════════════════════════"
echo "  💥 Déploiement de TradeX BINANCE CRASHBOT → $VPS_HOST"
echo "══════════════════════════════════════════════════"
echo ""

# ── 1. Sync des fichiers ────────────────────────────────────────────────
echo "📦 Synchronisation des fichiers..."
rsync -avz --delete \
    --filter='P .venv/' \
    --filter='P data/' \
    --filter='P logs/' \
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

    # S'assurer que les répertoires runtime existent (protégés côté rsync)
    sudo mkdir -p /opt/tradex/data /opt/tradex/logs

    # Installer les dépendances
    .venv/bin/pip install -r requirements.txt -q 2>/dev/null

    # ── Installer les services CrashBot ──
    echo ""
    echo "── 💥 Installation du CrashBot ──"
    sudo cp deploy/tradex-binance.service /etc/systemd/system/
    sudo cp deploy/tradex-binance-crashbot.service /etc/systemd/system/
    if [ -f deploy/tradex-binance-crashbot-dashboard.service ]; then
        sudo cp deploy/tradex-binance-crashbot-dashboard.service /etc/systemd/system/
        HAS_CRASHBOT_DASH=1
    else
        HAS_CRASHBOT_DASH=0
    fi
    sudo systemctl daemon-reload

    # Activer les services
    sudo systemctl enable tradex-binance-crashbot
    if [ "$HAS_CRASHBOT_DASH" -eq 1 ]; then
        sudo systemctl enable tradex-binance-crashbot-dashboard
    fi

    # Permissions
    sudo chown -R tradex:tradex /opt/tradex

    # Créer le dossier data si nécessaire
    sudo mkdir -p /opt/tradex/data
    sudo chown tradex:tradex /opt/tradex/data

    # Redémarrer le bot RANGE + lancer le CrashBot
    sudo systemctl restart tradex-binance
    sudo systemctl restart tradex-binance-crashbot
    if [ "$HAS_CRASHBOT_DASH" -eq 1 ]; then
        sudo systemctl restart tradex-binance-crashbot-dashboard
    fi

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

    if sudo systemctl is-active --quiet tradex-binance-crashbot; then
        echo "   ✅ tradex-binance-crashbot : actif"
    else
        echo "   ❌ tradex-binance-crashbot : erreur"
        sudo journalctl -u tradex-binance-crashbot -n 10 --no-pager
    fi

    if [ "$HAS_CRASHBOT_DASH" -eq 1 ]; then
        if sudo systemctl is-active --quiet tradex-binance-crashbot-dashboard; then
            echo "   ✅ tradex-binance-crashbot-dashboard : actif (port 8504)"
        else
            echo "   ❌ tradex-binance-crashbot-dashboard : erreur"
            sudo journalctl -u tradex-binance-crashbot-dashboard -n 5 --no-pager
        fi
    else
        echo "   ⚪ tradex-binance-crashbot-dashboard : non déployé (fichier service absent)"
    fi

    echo ""
    echo "── Résumé des services TradeX ──"
    for svc in tradex tradex-binance tradex-binance-crashbot tradex-binance-dashboard tradex-binance-crashbot-dashboard; do
        if sudo systemctl is-active --quiet "$svc"; then
            echo "   ✅ $svc"
        else
            echo "   ⚪ $svc (inactif)"
        fi
    done
REMOTE

echo ""
echo "══════════════════════════════════════════════════"
echo "  ✅ Déploiement Binance CrashBot terminé !"
echo "══════════════════════════════════════════════════"
echo ""
echo "  Logs RANGE    : ssh $VPS_HOST 'sudo journalctl -u tradex-binance -f'"
echo "  Logs CRASHBOT : ssh $VPS_HOST 'sudo journalctl -u tradex-binance-crashbot -f'"
echo "  Dashboard     : http://213.199.41.168:8504"
echo ""
