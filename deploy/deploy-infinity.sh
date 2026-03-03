#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────────────────────
# deploy-infinity.sh — Déploie le bot Infinity sur le VPS
#
# Usage depuis ta machine locale :
#   ./deploy/deploy-infinity.sh
# ──────────────────────────────────────────────────────────────────────────────
set -euo pipefail

VPS_HOST="${VPS_HOST:-BOT-VPS}"
APP_DIR="/opt/tradex"
SERVICE_NAME="tradex-infinity"

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"

echo "══════════════════════════════════════════════════"
echo "  ♾️  Déploiement Infinity Bot → $VPS_HOST"
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

# ── 2. Installer le service et redémarrer ────────────────────────────────
echo ""
echo "🔧 Installation du service et redémarrage..."
ssh "$VPS_HOST" << 'REMOTE'
    set -e
    cd /opt/tradex

    # Mettre à jour les dépendances
    .venv/bin/pip install -r requirements.txt -q 2>/dev/null

    # Copier le fichier service
    sudo cp deploy/tradex-infinity.service /etc/systemd/system/
    sudo systemctl daemon-reload
    sudo systemctl enable tradex-infinity

    # Permissions
    sudo chown -R tradex:tradex /opt/tradex

    # Créer le répertoire data s'il n'existe pas
    mkdir -p /opt/tradex/data

    # Redémarrer le service
    sudo systemctl restart tradex-infinity

    # Vérifier le statut
    sleep 3
    if sudo systemctl is-active --quiet tradex-infinity; then
        echo "   ✅ Infinity Bot redémarré avec succès"
    else
        echo "   ❌ Erreur au démarrage — voir: sudo journalctl -u tradex-infinity -n 20"
        exit 1
    fi
REMOTE

echo ""
echo "══════════════════════════════════════════════════"
echo "  ✅ Déploiement Infinity Bot terminé !"
echo "══════════════════════════════════════════════════"
echo ""
echo "  Voir les logs : ssh $VPS_HOST 'sudo journalctl -u tradex-infinity -f'"
echo ""
