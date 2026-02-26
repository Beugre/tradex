#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────────────────────
# setup-vps.sh — Première installation de TradeX sur un VPS Ubuntu/Debian
#
# Usage : scp ce script + le dossier du projet sur le VPS, puis :
#   chmod +x setup-vps.sh && sudo ./setup-vps.sh
# ──────────────────────────────────────────────────────────────────────────────
set -euo pipefail

APP_DIR="/opt/tradex"
APP_USER="tradex"
PYTHON_MIN="3.9"

echo "══════════════════════════════════════════════════"
echo "  🚀 Installation de TradeX sur le VPS"
echo "══════════════════════════════════════════════════"

# ── 1. Paquets système ─────────────────────────────────────────────────────
echo ""
echo "📦 Installation des paquets système..."
apt-get update -qq
apt-get install -y -qq python3 python3-venv python3-pip git curl > /dev/null

# Vérifier la version Python
PYTHON_VERSION=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
echo "   Python $PYTHON_VERSION détecté"

# ── 2. Utilisateur dédié ──────────────────────────────────────────────────
echo ""
echo "👤 Création de l'utilisateur $APP_USER..."
if id "$APP_USER" &>/dev/null; then
    echo "   L'utilisateur $APP_USER existe déjà"
else
    useradd --system --shell /bin/bash --home-dir "$APP_DIR" "$APP_USER"
    echo "   Utilisateur $APP_USER créé"
fi

# ── 3. Répertoire de l'application ────────────────────────────────────────
echo ""
echo "📁 Préparation de $APP_DIR..."
mkdir -p "$APP_DIR/logs"

# Copier les fichiers du projet (si on est dans le dossier du projet)
SCRIPT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
if [ -f "$SCRIPT_DIR/src/bot.py" ]; then
    echo "   Copie des fichiers depuis $SCRIPT_DIR..."
    rsync -a --exclude='.venv' --exclude='__pycache__' --exclude='.git' \
          --exclude='.pytest_cache' --exclude='*.pyc' --exclude='test_*.py' \
          --exclude='preflight.py' \
          "$SCRIPT_DIR/" "$APP_DIR/"
else
    echo "   ⚠️  Fichiers source non trouvés. Copiez-les manuellement dans $APP_DIR/"
fi

# ── 4. Environnement virtuel ─────────────────────────────────────────────
echo ""
echo "🐍 Création de l'environnement virtuel..."
python3 -m venv "$APP_DIR/.venv"
"$APP_DIR/.venv/bin/pip" install --upgrade pip -q
"$APP_DIR/.venv/bin/pip" install -r "$APP_DIR/requirements.txt" -q
echo "   Dépendances installées ✅"

# ── 5. Fichier .env ──────────────────────────────────────────────────────
echo ""
if [ ! -f "$APP_DIR/.env" ]; then
    echo "⚙️  Fichier .env manquant — copie du template..."
    cp "$APP_DIR/.env.example" "$APP_DIR/.env"
    chmod 600 "$APP_DIR/.env"
    echo "   ⚠️  IMPORTANT : éditez $APP_DIR/.env avec vos clés API"
else
    echo "⚙️  Fichier .env existant conservé"
    chmod 600 "$APP_DIR/.env"
fi

# ── 6. Clé privée ────────────────────────────────────────────────────────
echo ""
if [ ! -f "$APP_DIR/private.pem" ]; then
    echo "🔑 ⚠️  Clé privée manquante : copiez votre private.pem dans $APP_DIR/"
else
    chmod 600 "$APP_DIR/private.pem"
    echo "🔑 Clé privée trouvée, permissions sécurisées"
fi

# ── 7. Permissions ───────────────────────────────────────────────────────
echo ""
echo "🔒 Application des permissions..."
chown -R "$APP_USER:$APP_USER" "$APP_DIR"
chmod 750 "$APP_DIR"

# ── 8. Service systemd ──────────────────────────────────────────────────
echo ""
echo "🔧 Installation du service systemd..."
cp "$APP_DIR/deploy/tradex.service" /etc/systemd/system/tradex.service
systemctl daemon-reload
systemctl enable tradex
echo "   Service tradex activé (démarrage auto au boot)"

# ── 9. Résumé ────────────────────────────────────────────────────────────
echo ""
echo "══════════════════════════════════════════════════"
echo "  ✅ Installation terminée !"
echo "══════════════════════════════════════════════════"
echo ""
echo "  Prochaines étapes :"
echo ""
echo "  1. Éditez la config :     sudo nano $APP_DIR/.env"
echo "  2. Copiez la clé privée : scp private.pem vps:$APP_DIR/"
echo "  3. Lancez le bot :        sudo systemctl start tradex"
echo "  4. Vérifiez les logs :    sudo journalctl -u tradex -f"
echo ""
echo "  Commandes utiles :"
echo "    sudo systemctl status tradex    # État du service"
echo "    sudo systemctl restart tradex   # Redémarrer"
echo "    sudo systemctl stop tradex      # Arrêter"
echo "    sudo journalctl -u tradex -n 50 # Dernières 50 lignes"
echo ""
