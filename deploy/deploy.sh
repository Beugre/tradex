#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────────────────────
# deploy.sh — Déploie / met à jour TradeX sur le VPS
#
# Usage depuis ta machine locale :
#   ./deploy/deploy.sh
#
# Prérequis : alias vps-connexion configuré, ou modifier VPS_HOST ci-dessous
# ──────────────────────────────────────────────────────────────────────────────
set -euo pipefail

# ── Configuration ──────────────────────────────────────────────────────────
VPS_HOST="${VPS_HOST:-BOT-VPS}"       # Alias SSH configuré dans ~/.ssh/config
APP_DIR="/opt/tradex"
SERVICE_NAME="tradex"

# Dossier du projet (racine)
PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"

echo "══════════════════════════════════════════════════"
echo "  🚀 Déploiement de TradeX → $VPS_HOST"
echo "══════════════════════════════════════════════════"
echo ""

# ── 1. Sync des fichiers ────────────────────────────────────────────────
echo "📦 Synchronisation des fichiers..."
# --exclude : empêche le transfert local→VPS (protège aussi de --delete)
# --filter='P' : protection supplémentaire contre --delete-excluded
# ⚠ Ne JAMAIS lancer rsync manuellement sans ces flags (risque d'écraser data/, .env, .venv)
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
    --exclude='test_*.py' \
    --exclude='preflight.py' \
    --exclude='.DS_Store' \
    --exclude='logs/' \
    --exclude='data/' \
    "$PROJECT_DIR/" "$VPS_HOST:$APP_DIR/"

echo "   Fichiers synchronisés ✅"

# ── 2. Installer les dépendances et redémarrer ──────────────────────────
echo ""
echo "🔧 Installation des dépendances et redémarrage..."
ssh "$VPS_HOST" << 'REMOTE'
    set -e
    cd /opt/tradex

    # S'assurer que les répertoires runtime existent (protégés côté rsync)
    sudo mkdir -p /opt/tradex/data /opt/tradex/logs

    # Mettre à jour les dépendances
    .venv/bin/pip install -r requirements.txt -q 2>/dev/null

    # Permissions
    sudo chown -R tradex:tradex /opt/tradex

    # Redémarrer tous les services actifs (sauf le legacy 'tradex' qui est désactivé)
    SERVICES="tradex-binance tradex-binance-crashbot tradex-listing tradex-infinity tradex-london tradex-dca tradex-breakout tradex-dashboard-unified"
    for svc in $SERVICES; do
        if sudo systemctl is-enabled --quiet "$svc" 2>/dev/null; then
            sudo systemctl restart "$svc"
            echo "   ✅ $svc redémarré"
        else
            echo "   ⏭️  $svc (disabled, skip)"
        fi
    done

    # Vérifier après 3 secondes
    sleep 3
    FAILED=0
    for svc in $SERVICES; do
        if sudo systemctl is-enabled --quiet "$svc" 2>/dev/null; then
            if ! sudo systemctl is-active --quiet "$svc"; then
                echo "   ❌ $svc FAILED — voir: sudo journalctl -u $svc -n 20"
                FAILED=1
            fi
        fi
    done
    [ "$FAILED" -eq 1 ] && exit 1
REMOTE

echo ""
echo "══════════════════════════════════════════════════"
echo "  ✅ Déploiement terminé !"
echo "══════════════════════════════════════════════════"
echo ""
echo "  Voir les logs : ssh $VPS_HOST 'sudo journalctl -u tradex-dca -f'"
echo ""
