# TradeX — Commandes essentielles

## Connexion VPS

```bash
ssh BOT-VPS
```

---

## Logs en direct

### Un seul bot

```bash
# Range (Binance)
ssh BOT-VPS 'sudo journalctl -u tradex-binance -f'

# CrashBot (Binance)
ssh BOT-VPS 'sudo journalctl -u tradex-binance-crashbot -f'

# Momentum (Revolut X)
ssh BOT-VPS 'sudo journalctl -u tradex-momentum -f'

# Infinity (Revolut X)
ssh BOT-VPS 'sudo journalctl -u tradex-infinity -f'

# Dashboard (Streamlit)
ssh BOT-VPS 'sudo journalctl -u tradex-dashboard-unified -f'
```

### Les 4 bots en parallèle (tmux)

```bash
./scripts/logs-all.sh
```

> Raccourcis tmux :
> - `Ctrl+B` → flèches : naviguer entre les panes
> - `Ctrl+B` → `Z` : zoom/dézoom sur un pane
> - `Ctrl+B` → `D` : detach (quitter sans fermer)
> - `tmux attach -t tradex-logs` : revenir après detach

### Derniers N logs (sans follow)

```bash
ssh BOT-VPS 'sudo journalctl -u tradex-binance -n 50 --no-pager'
ssh BOT-VPS 'sudo journalctl -u tradex-binance-crashbot -n 50 --no-pager'
ssh BOT-VPS 'sudo journalctl -u tradex-momentum -n 50 --no-pager'
ssh BOT-VPS 'sudo journalctl -u tradex-infinity -n 50 --no-pager'
```

### Logs depuis une date

```bash
ssh BOT-VPS 'sudo journalctl -u tradex-binance --since "2026-03-04 10:00:00" --no-pager'
```

### Filtrer les erreurs

```bash
ssh BOT-VPS 'sudo journalctl -u tradex-binance -p err -n 30 --no-pager'
```

---

## Santé des bots

### Statut de tous les services

```bash
ssh BOT-VPS 'sudo systemctl status tradex-binance tradex-binance-crashbot tradex-momentum tradex-infinity tradex-dashboard-unified --no-pager'
```

### Vérifier si tous les bots tournent

```bash
ssh BOT-VPS 'for s in tradex-binance tradex-binance-crashbot tradex-momentum tradex-infinity tradex-dashboard-unified; do echo -n "$s: "; sudo systemctl is-active $s; done'
```

### Uptime + mémoire + CPU

```bash
ssh BOT-VPS 'uptime && echo "---" && free -h && echo "---" && ps aux --sort=-%mem | head -10'
```

### Derniers heartbeats (grep)

```bash
ssh BOT-VPS 'sudo journalctl -u tradex-binance -n 200 --no-pager | grep "💓" | tail -3'
ssh BOT-VPS 'sudo journalctl -u tradex-binance-crashbot -n 200 --no-pager | grep "💓" | tail -3'
ssh BOT-VPS 'sudo journalctl -u tradex-momentum -n 200 --no-pager | grep "💓" | tail -3'
ssh BOT-VPS 'sudo journalctl -u tradex-infinity -n 200 --no-pager | grep "💓" | tail -3'
```

### Dernier crash / restart

```bash
ssh BOT-VPS 'for s in tradex-binance tradex-binance-crashbot tradex-momentum tradex-infinity; do echo "=== $s ==="; sudo journalctl -u $s --no-pager | grep -E "Started|Stopped|error|Error|ERREUR|traceback" | tail -5; echo; done'
```

---

## Gestion des services

### Redémarrer un bot

```bash
ssh BOT-VPS 'sudo systemctl restart tradex-binance'
ssh BOT-VPS 'sudo systemctl restart tradex-binance-crashbot'
ssh BOT-VPS 'sudo systemctl restart tradex-momentum'
ssh BOT-VPS 'sudo systemctl restart tradex-infinity'
ssh BOT-VPS 'sudo systemctl restart tradex-dashboard-unified'
```

### Redémarrer tous les bots

```bash
ssh BOT-VPS 'sudo systemctl restart tradex-binance tradex-binance-crashbot tradex-momentum tradex-infinity tradex-dashboard-unified'
```

### Arrêter / démarrer un bot

```bash
ssh BOT-VPS 'sudo systemctl stop tradex-infinity'
ssh BOT-VPS 'sudo systemctl start tradex-infinity'
```

### Désactiver un bot (ne redémarre plus au boot)

```bash
ssh BOT-VPS 'sudo systemctl disable tradex-infinity'
```

---

## Déploiement

### Depuis le Mac local

```bash
# Range (Binance)
./deploy/deploy-binance.sh

# CrashBot (Binance)
./deploy/deploy-binance-crashbot.sh

# Infinity (Revolut X)
./deploy/deploy-infinity.sh

# Dashboard
./deploy/deploy-dashboard.sh

# Momentum (Revolut X)
./deploy/deploy.sh
```

### Déploiement manuel rapide (sans script)

```bash
# Sync fichiers + restart
rsync -avz --exclude='.venv' --exclude='__pycache__' --exclude='.git' --exclude='.env' --exclude='data/' . BOT-VPS:/opt/tradex/
ssh BOT-VPS 'sudo systemctl restart tradex-infinity'
```

---

## Git

```bash
# Status
git status --short

# Commit + push tout
git add -A && git commit -m "message" && git push

# Voir les derniers commits
git log --oneline -10
```

---

## Tests locaux

```bash
# Lancer les tests
.venv/bin/pytest tests/ -v

# Compiler un fichier (vérification syntaxe)
.venv/bin/python -m py_compile src/bot_infinity.py

# Dry-run d'un bot
.venv/bin/python -m src.bot_binance --dry-run
.venv/bin/python -m src.bot_binance_crashbot --dry-run
.venv/bin/python -m src.bot_momentum --dry-run
.venv/bin/python -m src.bot_infinity --dry-run

# Dashboard local
.venv/bin/streamlit run dashboard/app_unified.py --server.port 8502
```

---

## Dashboard

```
http://213.199.41.168:8502
```

---

## Fichiers d'état (VPS)

```bash
# Voir l'état persisté de chaque bot
ssh BOT-VPS 'cat /opt/tradex/data/state_binance.json | python3 -m json.tool | head -30'
ssh BOT-VPS 'cat /opt/tradex/data/state_binance_crashbot.json | python3 -m json.tool | head -30'
ssh BOT-VPS 'cat /opt/tradex/data/state_momentum.json | python3 -m json.tool | head -30'
ssh BOT-VPS 'cat /opt/tradex/data/state_infinity.json | python3 -m json.tool | head -30'
```

---

## Diagnostic rapide complet

```bash
# One-liner : statut + dernier heartbeat de chaque bot
ssh BOT-VPS 'echo "══════ STATUT ══════"; for s in tradex-binance tradex-binance-crashbot tradex-momentum tradex-infinity; do printf "%-30s %s\n" "$s" "$(sudo systemctl is-active $s)"; done; echo ""; echo "══════ HEARTBEATS ══════"; for s in tradex-binance tradex-binance-crashbot tradex-momentum tradex-infinity; do echo "── $s ──"; sudo journalctl -u $s -n 100 --no-pager 2>/dev/null | grep "💓" | tail -1; done'
```
