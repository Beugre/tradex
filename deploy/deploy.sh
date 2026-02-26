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

# ── 2. Installer les dépendances et redémarrer ──────────────────────────
echo ""
echo "🔧 Installation des dépendances et redémarrage..."
ssh "$VPS_HOST" << 'REMOTE'
    set -e
    cd /opt/tradex

    # Mettre à jour les dépendances
    .venv/bin/pip install -r requirements.txt -q 2>/dev/null

    # Permissions
    sudo chown -R tradex:tradex /opt/tradex

    # Redémarrer le service
    sudo systemctl restart tradex

    # Vérifier le statut
    sleep 2
    if sudo systemctl is-active --quiet tradex; then
        echo "   ✅ TradeX redémarré avec succès"
    else
        echo "   ❌ Erreur au démarrage — voir: sudo journalctl -u tradex -n 20"
        exit 1
    fi
REMOTE

echo ""
echo "══════════════════════════════════════════════════"
echo "  ✅ Déploiement terminé !"
echo "══════════════════════════════════════════════════"
echo ""
echo "  Voir les logs : ssh $VPS_HOST 'sudo journalctl -u tradex -f'"
echo ""
