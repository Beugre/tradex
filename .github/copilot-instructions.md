# TradeX – Copilot Instructions

## Aperçu du projet

Écosystème de **4 bots de trading crypto automatisés** déployés sur un VPS Contabo, opérant sur **2 exchanges** :

| Bot | Exchange | Stratégie | Timeframe | Paires |
|-----|----------|-----------|-----------|--------|
| **Trail Range** | Binance (USDC) | Mean-reversion + step-trail OCO | H4 | ~284 paires USDC (auto-discovery) |
| **CrashBot** | Binance (USDC) | Dip-buy (long only) avec step-trail | H4 (polling) | ~284 paires USDC (auto-discovery) |
| **Infinity** | Revolut X (USD) | DCA inversé multi-paires (trailing high → achat → vente paliers) | H4 | BTC, AAVE, XLM, ADA, DOT, LTC |
| **London Breakout** | Revolut X (USD) | Session breakout (range 08-16 UTC → breakout LONG) | H4 | BTC, ETH, SOL, BNB, LINK, ADA, DOT, AVAX |

- **Langage** : Python 3.10+ (VPS), Python 3.12+ (dev local)
- **Notifications** : Telegram Bot API (entrée, SL, TP, clôture, heartbeat)
- **Persistance** : Firebase Firestore (trades, positions, heartbeats, allocation, snapshots)
- **Dashboard** : Streamlit unifié (port 8502) — 5 onglets (Overview, Range, CrashBot, Infinity, London)
- **Déploiement** : VPS Contabo (SSH alias `BOT-VPS`, path `/opt/tradex/`)

## Architecture

```
src/
├── core/                       # Logique métier pure (sans I/O)
│   ├── allocator.py            # Allocation dynamique Crash/Trail basée sur le Profit Factor
│   ├── crashbot_detector.py    # Détection de dip / crash pour CrashBot
│   ├── flow_detector.py        # Analyse de flux pour aide à la décision
│   ├── infinity_engine.py      # Logique DCA inversé (paliers achat/vente, RSI gate, trailing high)
│   ├── indicators.py           # Indicateurs techniques réutilisables (EMA, SMA, ATR, RSI, rolling min/max)
│   ├── models.py               # Modèles de données partagés (dataclass/Pydantic)
│   ├── position_store.py       # Gestion en mémoire des positions ouvertes
│   ├── risk_manager.py         # Money management (% risque, sizing, fiat balance, equity)
│   ├── strategy_mean_rev.py    # Stratégie mean-reversion (Trail Range)
│   ├── strategy_trend.py       # Stratégie tendance / Dow Theory
│   ├── swing_detector.py       # Détection des swings (sommets/creux) sur données OHLC
│   └── trend_engine.py         # Classification HH/HL/LH/LL et état de tendance
├── exchange/
│   ├── binance_client.py       # Wrapper API Binance Spot (USDC pairs, OCO orders)
│   ├── revolut_client.py       # Wrapper API Revolut X (auth Ed25519, maker-only)
│   └── data_provider.py        # Récupération des bougies OHLCV (Revolut X)
├── firebase/
│   ├── client.py               # Connexion Firestore (CRUD générique)
│   └── trade_logger.py         # Logging trades, heartbeats, allocations, snapshots
├── notifications/
│   └── telegram.py             # Envoi d'alertes Telegram
├── bot_binance.py              # Bot Trail Range — Binance (mean-reversion + step-trail OCO)
├── bot_binance_crashbot.py     # Bot CrashBot — Binance (dip-buy, step-trail, long only)
├── bot_infinity.py             # Bot Infinity — Revolut X (DCA inversé multi-paires, maker-only)
├── bot_london.py               # Bot London Breakout — Revolut X (session breakout H4, maker-only)
├── bot.py                      # (legacy) Bot Dow Theory Revolut X
└── config.py                   # Chargement .env (clés API, paramètres de risque)
dashboard/
└── app_unified.py              # Dashboard Streamlit unifié (5 tabs, 2 exchanges)
tests/
├── test_swing_detector.py
├── test_trend_engine.py
├── test_allocator.py
└── test_risk_manager.py
```

## Allocation du capital

### Binance (Trail Range + CrashBot)

Les bots **Trail Range** et **CrashBot** partagent le même capital sur Binance. L'allocation est recalculée **1×/jour** via `allocator.py` basé sur le Profit Factor du Trail Range.

```
┌──────────────────────────────────────────────────────────┐
│ PF 90j du Trail Range   │  Trail Range  │   CrashBot    │
├──────────────────────────────────────────────────────────┤
│ PF < 0.9 OU < 20 trades │     10%       │     90%       │  ← DEFENSIVE
│ 0.9 ≤ PF ≤ 1.1          │     20%       │     80%       │  ← NEUTRAL
│ PF > 1.1                │     40%       │     60%       │  ← AGGRESSIVE
└──────────────────────────────────────────────────────────┘
```

- Le PF est calculé via `compute_profit_factor(pnl_list)` : `sum(gains) / abs(sum(pertes))`
- Les PnL sont récupérés depuis Firebase (`get_trail_range_pnl_list(days=90)`)
- L'allocation est loggée dans Firebase (`allocation/current`) pour le dashboard

### Revolut X (Infinity + London Breakout)

Les bots **Infinity** et **London Breakout** partagent le même compte Revolut X. L'allocation est statique :

| Bot | Part du capital Revolut X |
|-----|---------------------------|
| **Infinity** | 80% (`INF_CAPITAL_PCT=0.80`) |
| **London Breakout** | 20% (`LON_CAPITAL_PCT=0.20`) |

## Bot 1 — Trail Range (`bot_binance.py`)

- **Exchange** : Binance Spot (USDC)
- **Stratégie** : Mean-reversion avec OCO orders natifs + step-trail
- **Paires** : Auto-discovery de toutes les paires USDC en status TRADING
- **Capital** : Portion dynamique via `allocator.py` (10–40% selon PF)
- **Boucle** : Polling continu, analyse à chaque cycle
- **Gestion de position** : OCO orders (TP + SL simultanés), step-trail au-delà du TP

## Bot 2 — CrashBot (`bot_binance_crashbot.py`)

- **Exchange** : Binance Spot (USDC)
- **Stratégie** : Dip-buy (long only) — achète les baisses brutales, sort en step-trail
- **Paires** : Auto-discovery de toutes les paires USDC en status TRADING
- **Capital** : Portion dominante via `allocator.py` (60–90% selon PF Trail)
- **Boucle** : Polling continu, détection de crash via `crashbot_detector.py`
- **Heartbeat** : Toutes les 10 minutes (`CRASHBOT_HEARTBEAT_TELEGRAM_SECONDS=600`)
- **Momentum Sizing** : Le risk% s'ajuste dynamiquement après chaque trade :
  - WIN → `risk *= 1.2` (plafond 10%)
  - LOSS → `risk *= 0.8` (plancher 2%)
  - Base : 5%, persisté dans `crashbot_momentum_state.json`
  - Désactivable via `BINANCE_CRASHBOT_MOMENTUM_SIZING=false`

## Bot 3 — Infinity (`bot_infinity.py`)

- **Exchange** : Revolut X (USD) — Maker 0% fees, Taker 0.09%
- **Stratégie** : DCA inversé multi-paires :
  1. **Trailing high** : N bougies H4 → prix de référence
  2. **Drop entry** : Si prix chute ≥ drop_pct du trailing high → premier achat
  3. **DCA** : 5 paliers d'achat progressifs sous le prix initial
  4. **Vente paliers** : 5 paliers au-dessus du PMP (prix moyen pondéré)
  5. **Breakeven** : SL au PMP après TP1
  6. **Stop-loss** : configurable par paire
- **Paires** : 6 paires walk-forward validées (`INF_TRADING_PAIRS` : BTC, AAVE, XLM, ADA, DOT, LTC)
- **Exécution** : Maker 2 retry → taker fallback (0.09% fee)
- **Capital** : 80% du solde Revolut X (`INF_CAPITAL_PCT=0.80`)

## Bot 4 — London Breakout (`bot_london.py`)

- **Exchange** : Revolut X (USD) — Maker 0% fees, Taker 0.09%
- **Stratégie** : London Session Breakout en 4 phases :
  1. **Session range** (08-16 UTC) : Accumule le high/low des bougies H4 08:00 et 12:00
  2. **Breakout check** : Après 16:00 UTC, si close > session_high
  3. **Filtres** : Volume ≥ 2.0 × MA20, Range session ≥ 1.5%
  4. **Entrée LONG** : SL = entry - 2.0×ATR(14), TP1 = +2% (50%), TP2 = +5% (reste)
  5. **Breakeven** : SL ramené à l'entrée après TP1
- **Paires** : `LON_TRADING_PAIRS` (BTC, ETH, SOL, BNB, LINK, ADA, DOT, AVAX)
- **Exécution** : Maker 2 retry → taker fallback (même logique qu'Infinity)
- **Risk** : 5% par trade (`LON_RISK_PERCENT`), max 1 position simultanée
- **Capital** : 20% du solde Revolut X (`LON_CAPITAL_PCT=0.20`)
- **Cooldown** : 2 bougies H4 (8h) entre deux trades sur la même paire
- **Backtest** : PF 1.98, +530$/an, DD 13.3%, walk-forward stable (PF test 1.57-1.76)

### Money management (`risk_manager.py`)
```python
# Calcul de taille partagé par tous les bots — NE PAS changer sans validation
risk_amount = account_balance * risk_percent   # ex: 500 * 0.05 = 25 USD
sl_distance = abs(entry_price - sl_price)      # en prix
position_size = risk_amount / sl_distance      # en unités de base (ex: ETH)
# Plafond : position_size * entry_price ≤ account_balance * max_position_pct
```

## Conventions de code

- **Séparation stricte I/O / logique** : `src/core/` ne fait AUCUN appel réseau. Les tests de `core/` doivent tourner sans mock d'API.
- **Types** : utiliser des `dataclass` ou `Pydantic BaseModel` pour toutes les structures de données.
- **Enums** pour les états : `AllocationRegime(Enum): DEFENSIVE, NEUTRAL, AGGRESSIVE`, `StrategyType(Enum): TREND, RANGE, CRASHBOT, INFINITY, LONDON`
- **Logging** : module `logging` standard avec le format `[%(asctime)s] %(levelname)s %(name)s: %(message)s`.
- **Config** : toutes les valeurs sensibles et paramètres dans `.env`, chargés via `python-dotenv`. Ne jamais hardcoder de clé API ou de paramètre de risque.
- **Firebase** : Toute persistance passe par `src/firebase/`. Les trades, heartbeats, allocations, et snapshots sont stockés dans Firestore.

## Checklist — Ajout / suppression / modification d'une stratégie

Quand une stratégie (bot) est **ajoutée, supprimée ou modifiée**, **TOUS** les fichiers suivants doivent être mis à jour systématiquement :

| # | Fichier / lieu | Ce qu'il faut modifier |
|---|----------------|------------------------|
| 1 | `src/core/models.py` | Ajouter/retirer la valeur dans `StrategyType(Enum)` |
| 2 | `src/config.py` | Ajouter/retirer les variables d'environnement du bot |
| 3 | `.env` | Ajouter/retirer les clés de config du bot |
| 4 | `STRATEGIE.md` | Mettre à jour TOUTES les sections : table des matières, tableau des bots, section dédiée du bot, exemples concrets, table des fichiers, paramètres, services, commandes, allocation du capital, table de risque |
| 5 | `.github/copilot-instructions.md` | Mettre à jour : aperçu, architecture, section bot, enums, variables d'environnement, services systemd, dashboard |
| 6 | `dashboard/app_unified.py` | Ajouter/retirer l'onglet du bot |
| 7 | `scripts/logs-all.sh` | Ajouter/retirer le pane tmux du bot |
| 8 | `scripts/analyze_profitability.py` | Mettre à jour `EXCHANGE_LABELS` |
| 9 | `deploy/tradex-<bot>.service` | Créer/supprimer le fichier service systemd |
| 10 | `src/notifications/telegram.py` | Ajouter/retirer les templates de notification |
| 11 | VPS `/etc/systemd/system/` | Déployer/supprimer le service, `daemon-reload` |
| 12 | Imports backtest (`backtest/`) | Mettre à jour si des utilitaires sont déplacés |

> **Règle** : ne jamais considérer un ajout/suppression de bot comme terminé tant que TOUS ces fichiers n'ont pas été vérifiés et mis à jour.

## Variables d'environnement (`.env`)

```env
# ── Binance ──
BINANCE_API_KEY=xxx
BINANCE_API_SECRET=xxx

# ── Revolut X ──
REVOLUT_X_API_KEY=xxx                        # Clé API 64 chars obtenue sur exchange.revolut.com
REVOLUT_X_PRIVATE_KEY_PATH=./private.pem     # Clé privée Ed25519 pour signer les requêtes

# ── Telegram ──
TELEGRAM_BOT_TOKEN=xxx
TELEGRAM_CHAT_ID=xxx

# ── Firebase ──
GOOGLE_APPLICATION_CREDENTIALS=./firebase-key.json

# ── Trail Range (bot_binance.py) ──
RISK_PERCENT=0.05
POLLING_INTERVAL_SECONDS=30
MAX_POSITION_PERCENT=0.50

# ── CrashBot (bot_binance_crashbot.py) ──
# (partage le même BINANCE_API_KEY, capital via allocator)
CRASHBOT_HEARTBEAT_TELEGRAM_SECONDS=600
BINANCE_CRASHBOT_MOMENTUM_SIZING=true
BINANCE_CRASHBOT_RISK_BOOST_MULT=1.2
BINANCE_CRASHBOT_RISK_SHRINK_MULT=0.8
BINANCE_CRASHBOT_MIN_RISK_PCT=0.02
BINANCE_CRASHBOT_MAX_RISK_PCT=0.10

# ── Infinity (bot_infinity.py) ──
INF_TRADING_PAIRS=BTC-USD,AAVE-USD,XLM-USD,ADA-USD,DOT-USD,LTC-USD
INF_CAPITAL_PCT=0.80
INF_POLLING_SECONDS=30
INF_HEARTBEAT_SECONDS=600
INF_MAKER_WAIT_SECONDS=60

# ── London Breakout (bot_london.py) ──
LON_TRADING_PAIRS=BTC-USD,ETH-USD,SOL-USD,BNB-USD,LINK-USD,ADA-USD,DOT-USD,AVAX-USD
LON_CAPITAL_PCT=0.20
LON_RISK_PERCENT=0.05
LON_MAX_POSITIONS=1
LON_SESSION_START_HOUR=8
LON_SESSION_END_HOUR=16
LON_SL_ATR_MULT=2.0
LON_TP1_PCT=0.02
LON_TP2_PCT=0.05
LON_VOL_MULT=2.0
LON_MIN_RANGE_PCT=0.015
LON_POLLING_SECONDS=30
LON_HEARTBEAT_SECONDS=600
LON_MAKER_WAIT_SECONDS=60
```

## APIs – Points clés

### Binance Spot (Trail Range + CrashBot)
- **Base URL** : `https://api.binance.com`
- **Auth** : HMAC-SHA256 (API key + secret)
- **Ordres** : OCO orders natifs (TP + SL simultanés), limit, market
- **Symboles** : Format `BASEUSDC` (ex: `BTCUSDC`, `ETHUSDC`)

### Revolut X (Infinity + London Breakout)
- **Base URL** : `https://revx.revolut.com/api/1.0/`
- **Rate limit** : 1000 requêtes/minute
- **Auth** : signature Ed25519 par requête (PAS de token/session) :
  - Header `X-Revx-API-Key` : clé API 64 chars
  - Header `X-Revx-Timestamp` : timestamp Unix ms
  - Header `X-Revx-Signature` : signature Ed25519 base64 de `{timestamp}{METHOD}{path}{query_string}{body}`
- **Ordres** : Uniquement `limit` (pas de stop natif, pas d'OCO)
- **Fees** : Maker 0%, Taker 0.09%
- **Symboles** : Format `BASE-QUOTE` (ex: `BTC-USD`, `ETH-USD`)

| Endpoint | Méthode | Usage |
|----------|---------|-------|
| `/balances` | GET | Solde du compte (available/reserved/total par devise) |
| `/candles/{symbol}` | GET | Bougies OHLCV H4 pour Infinity/London |
| `/tickers` | GET | Prix temps réel (bid/ask/last) |
| `/orders` | POST | Placer un ordre limit (buy/sell) |
| `/orders/active` | GET | Lister les ordres en cours |
| `/orders/{venue_order_id}` | DELETE | Annuler un ordre |
| `/orders/{venue_order_id}` | GET | Détails d'un ordre |
| `/orders/{venue_order_id}/fills` | GET | Fills (exécutions) d'un ordre |

## Services systemd (VPS)

```bash
tradex-binance              # Bot Trail Range
tradex-binance-crashbot     # Bot CrashBot
tradex-infinity             # Bot Infinity (Revolut X)
tradex-london               # Bot London Breakout (Revolut X)
tradex-dashboard-unified    # Dashboard Streamlit (port 8502)
```

## Commandes de développement

```bash
# Installer les dépendances
pip install -r requirements.txt

# Lancer les tests (core uniquement, pas d'API)
pytest tests/ -v

# Lancer un bot en mode dry-run
python -m src.bot_binance --dry-run
python -m src.bot_binance_crashbot --dry-run
python -m src.bot_infinity --dry-run
python -m src.bot_london --dry-run

# Lancer le dashboard en local
streamlit run dashboard/app_unified.py --server.port 8502

# Se connecter au VPS
ssh BOT-VPS

# Voir les logs d'un bot sur le VPS
sudo journalctl -u tradex-binance -f
sudo journalctl -u tradex-binance-crashbot -f
sudo journalctl -u tradex-infinity -f
sudo journalctl -u tradex-london -f

# Redémarrer un service
sudo systemctl restart tradex-binance
```

## Dashboard unifié (`dashboard/app_unified.py`)

Dashboard Streamlit avec 5 onglets :
1. **Overview** : Allocation Binance (gauge CrashBot/Trail), allocation Revolut X (Infinity 80%/London 20%), KPIs globaux, equity cumulée
2. **Trail Range** : Positions ouvertes, trades récents, PnL journalier
3. **CrashBot** : Positions ouvertes, trades récents, statistiques de dip-buy
4. **Infinity** : Cycles par paire, V-curves, paliers achat/vente
5. **London** : Positions Revolut X, sessions détectées, breakouts

L'Overview affiche la répartition dynamique du capital Binance entre CrashBot et Trail Range, et la répartition statique 80/20 du capital Revolut X entre Infinity et London Breakout.

## Notifications Telegram

Chaque notification contient : **paire**, **action**, **prix d'entrée**, **SL**, **taille de position**, et un emoji indicatif.
```
🇬🇧 BUY déclenché – ETH-USD LONDON BREAKOUT
  Entrée: 3500.00 | SL: 3350.00 (4.3%)
  TP1: 3570.00 (+2%) | TP2: 3675.00 (+5%)
  Size: 0.01400000 ETH ($49.00)
  Risque: 5% ($25.00) | ATR: 75.00
```
Alertes envoyées pour : signal d'entrée, fill d'ordre, stop-loss touché, TP1 (partial close + breakeven), TP2, clôture, heartbeat (toutes les 10 min), changement d'allocation.
