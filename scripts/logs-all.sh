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

# Créer la session avec le premier pane (Range = top-left)
tmux new-session -d -s "$SESSION" -n logs \
    "ssh $VPS 'sudo journalctl -u tradex-binance -f --no-pager -n 30'"

# Split verticalement → pane 1 = bottom (Momentum)
tmux split-window -v -p 50 -t "$SESSION" \
    "ssh $VPS 'sudo journalctl -u tradex-momentum -f --no-pager -n 30'"

# Split le haut horizontalement → pane 2 = top-right (CrashBot)
tmux split-window -h -p 50 -t "$SESSION.0" \
    "ssh $VPS 'sudo journalctl -u tradex-binance-crashbot -f --no-pager -n 30'"

# Split le bas horizontalement → pane 3 = bottom-right (Infinity)
tmux split-window -h -p 50 -t "$SESSION.1" \
    "ssh $VPS 'sudo journalctl -u tradex-infinity -f --no-pager -n 30'"

# Titres des panes
tmux select-pane -t "$SESSION.0" -T "🔄 Range"
tmux select-pane -t "$SESSION.1" -T "🚀 Momentum"
tmux select-pane -t "$SESSION.2" -T "📉 CrashBot"
tmux select-pane -t "$SESSION.3" -T "♾️ Infinity"

# Activer les titres de panes
tmux set -t "$SESSION" pane-border-status top
tmux set -t "$SESSION" pane-border-format " #{pane_title} "

# Attacher
tmux attach -t "$SESSION"
