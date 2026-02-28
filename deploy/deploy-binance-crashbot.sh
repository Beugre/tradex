#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────────────────────
# deploy-binance-crashbot.sh — Déploie le bot CrashBot (Dip Buy) sur le VPS
#
# Usage depuis ta machine locale :
#   bash deploy/deploy-binance-crashbot.sh
#
# Déploie le code + redémarre le service tradex-binance-crashbot
# Le bot RANGE (tradex-binance) est aussi redémarré (car le code est partagé)
# Stoppe le bot Breakout (décommissionné, remplacé par CrashBot)
# ──────────────────────────────────────────────────────────────────────────────
set -euo pipefail

# ── Configuration ──────────────────────────────────────────────────────────
VPS_HOST="${VPS_HOST:-BOT-VPS}"
APP_DIR="/opt/tradex"
SERVICE_RANGE="tradex-binance"
SERVICE_CRASHBOT="tradex-binance-crashbot"
SERVICE_CRASHBOT_DASH="tradex-binance-crashbot-dashboard"
# Breakout décommissionné
SERVICE_BREAKOUT="tradex-binance-breakout"
SERVICE_BREAKOUT_DASH="tradex-binance-breakout-dashboard"

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"

echo "══════════════════════════════════════════════════"
echo "  💥 Déploiement de TradeX BINANCE CRASHBOT → $VPS_HOST"
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

    # ── Décommissionner le bot Breakout ──
    echo ""
    echo "── 🛑 Arrêt du bot Breakout (décommissionné) ──"
    sudo systemctl stop tradex-binance-breakout 2>/dev/null || true
    sudo systemctl stop tradex-binance-breakout-dashboard 2>/dev/null || true
    sudo systemctl disable tradex-binance-breakout 2>/dev/null || true
    sudo systemctl disable tradex-binance-breakout-dashboard 2>/dev/null || true
    echo "   Breakout arrêté et désactivé ✅"

    # ── Installer les services CrashBot ──
    echo ""
    echo "── 💥 Installation du CrashBot ──"
    sudo cp deploy/tradex-binance.service /etc/systemd/system/
    sudo cp deploy/tradex-binance-crashbot.service /etc/systemd/system/
    sudo cp deploy/tradex-binance-crashbot-dashboard.service /etc/systemd/system/
    sudo systemctl daemon-reload

    # Activer les services
    sudo systemctl enable tradex-binance-crashbot
    sudo systemctl enable tradex-binance-crashbot-dashboard

    # Permissions
    sudo chown -R tradex:tradex /opt/tradex

    # Créer le dossier data si nécessaire
    sudo mkdir -p /opt/tradex/data
    sudo chown tradex:tradex /opt/tradex/data

    # Redémarrer le bot RANGE + lancer le CrashBot
    sudo systemctl restart tradex-binance
    sudo systemctl restart tradex-binance-crashbot
    sudo systemctl restart tradex-binance-crashbot-dashboard

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

    if sudo systemctl is-active --quiet tradex-binance-crashbot-dashboard; then
        echo "   ✅ tradex-binance-crashbot-dashboard : actif (port 8504)"
    else
        echo "   ❌ tradex-binance-crashbot-dashboard : erreur"
        sudo journalctl -u tradex-binance-crashbot-dashboard -n 5 --no-pager
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

    # Vérifier que Breakout est bien arrêté
    if sudo systemctl is-active --quiet tradex-binance-breakout; then
        echo "   ⚠️  tradex-binance-breakout ENCORE ACTIF (devrait être arrêté)"
    else
        echo "   🗑️  tradex-binance-breakout : arrêté (décommissionné)"
    fi
REMOTE

echo ""
echo "══════════════════════════════════════════════════"
echo "  ✅ Déploiement Binance CrashBot terminé !"
echo "  🛑 Breakout décommissionné"
echo "══════════════════════════════════════════════════"
echo ""
echo "  Logs RANGE    : ssh $VPS_HOST 'sudo journalctl -u tradex-binance -f'"
echo "  Logs CRASHBOT : ssh $VPS_HOST 'sudo journalctl -u tradex-binance-crashbot -f'"
echo "  Dashboard     : http://213.199.41.168:8504"
echo ""
