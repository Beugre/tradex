#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────────────
# logs-all.sh — Affiche les logs des 4 bots en parallèle via tmux
#
# Usage : ./scripts/logs-all.sh
# Quitter : Ctrl+B puis D (detach) ou Ctrl+C dans chaque pane
# ──────────────────────────────────────────────────────────────────────
set -euo pipefail

VPS="${VPS_HOST:-BOT-VPS}"
SESSION="tradex-logs"

# Vérifier tmux
if ! command -v tmux &>/dev/null; then
    echo "❌ tmux requis — brew install tmux"
    exit 1
fi

# Kill session existante si besoin
tmux kill-session -t "$SESSION" 2>/dev/null || true

# ── Créer la grille 2×2 ──
# Pane 0 = top-left : Range
tmux new-session -d -s "$SESSION" -n logs \
    "echo -ne '\033]2;🔄 Range\033\\'; ssh $VPS 'sudo journalctl -u tradex-binance -f --no-pager -n 30'"

# Split horizontal → top-right : CrashBot
tmux split-window -h -t "$SESSION:logs" \
    "echo -ne '\033]2;📉 CrashBot\033\\'; ssh $VPS 'sudo journalctl -u tradex-binance-crashbot -f --no-pager -n 30'"

# Revenir pane top-left (0) et split vertical → bottom-left : Momentum
tmux select-pane -t "$SESSION:logs.0"
tmux split-window -v -t "$SESSION:logs.0" \
    "echo -ne '\033]2;🚀 Momentum\033\\'; ssh $VPS 'sudo journalctl -u tradex-momentum -f --no-pager -n 30'"

# Aller au pane top-right (maintenant index 2 après insert) et split vertical → bottom-right : Infinity
tmux select-pane -t "$SESSION:logs.2"
tmux split-window -v -t "$SESSION:logs.2" \
    "echo -ne '\033]2;♾️ Infinity\033\\'; ssh $VPS 'sudo journalctl -u tradex-infinity -f --no-pager -n 30'"

# Activer les titres de panes (utilise le titre défini par echo)
tmux set -t "$SESSION" pane-border-status top
tmux set -t "$SESSION" pane-border-format " #T "

# Sélectionner le premier pane
tmux select-pane -t "$SESSION:logs.0"

# Attacher
tmux attach -t "$SESSION"
