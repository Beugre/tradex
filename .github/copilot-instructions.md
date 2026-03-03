# TradeX – Copilot Instructions

## Aperçu du projet

Écosystème de **3 bots de trading crypto automatisés** déployés sur un VPS Contabo, opérant sur **2 exchanges** :

| Bot | Exchange | Stratégie | Timeframe | Paires |
|-----|----------|-----------|-----------|--------|
| **Trail Range** | Binance (USDC) | Mean-reversion + trailing OCO | Multi-timeframe | ~284 paires USDC (auto-discovery) |
| **CrashBot** | Binance (USDC) | Dip-buy (long only) avec step-trail | Continu (polling) | ~284 paires USDC (auto-discovery) |
| **Momentum** | Revolut X (USD) | Momentum Continuation (impulsion M5 → pullback → entrée) | M5 / M15 | ETH, SOL, BNB, XRP, LINK, ADA, LTC |

- **Langage** : Python 3.10+ (VPS), Python 3.12+ (dev local)
- **Notifications** : Telegram Bot API (entrée, SL, clôture, changement de tendance)
- **Persistance** : Firebase Firestore (trades, positions, heartbeats, allocation, snapshots)
- **Dashboard** : Streamlit unifié (port 8502) — 4 onglets (Overview, Range, CrashBot, Momentum)
- **Déploiement** : VPS Contabo (SSH alias `BOT-VPS`, path `/opt/tradex/`)

## Architecture

```
src/
├── core/                       # Logique métier pure (sans I/O)
│   ├── allocator.py            # Allocation dynamique Crash/Trail basée sur le Profit Factor
│   ├── crashbot_detector.py    # Détection de dip / crash pour CrashBot
│   ├── flow_detector.py        # Analyse de flux pour aide à la décision
│   ├── models.py               # Modèles de données partagés (dataclass/Pydantic)
│   ├── momentum_engine.py      # Stratégie Momentum Continuation (impulsion, pullback, entrée)
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
├── bot_binance.py              # Bot Trail Range — Binance (mean-reversion + OCO)
├── bot_binance_crashbot.py     # Bot CrashBot — Binance (dip-buy, step-trail, long only)
├── bot_momentum.py             # Bot Momentum — Revolut X (impulsion M5, maker-only)
├── bot.py                      # (legacy) Bot Dow Theory Revolut X
└── config.py                   # Chargement .env (clés API, paramètres de risque)
dashboard/
└── app_unified.py              # Dashboard Streamlit unifié (4 tabs, 2 exchanges)
tests/
├── test_swing_detector.py
├── test_trend_engine.py
├── test_allocator.py
└── test_risk_manager.py
```

## Allocation dynamique du capital (`allocator.py`)

Les bots **Trail Range** et **CrashBot** partagent le même capital sur Binance. L'allocation est recalculée **1×/jour** par chaque bot au démarrage ou au changement de date UTC.

### Mécanisme

Le **Profit Factor (PF)** du Trail Range sur 90 jours détermine le régime d'allocation :

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
- Chaque bot lit `AllocationResult.trail_balance` ou `.crash_balance` pour dimensionner ses positions

### Le bot Momentum (Revolut X) a son propre capital séparé
Le Momentum bot utilise le solde USD disponible directement sur Revolut X, sans aucun lien avec l'allocator Binance.

## Bot 1 — Trail Range (`bot_binance.py`)

- **Exchange** : Binance Spot (USDC)
- **Stratégie** : Mean-reversion avec OCO orders natifs
- **Paires** : Auto-discovery de toutes les paires USDC en status TRADING
- **Capital** : Portion dynamique via `allocator.py` (10–40% selon PF)
- **Boucle** : Polling continu, analyse à chaque cycle
- **Gestion de position** : OCO orders (take-profit + stop-loss simultanés), trailing via step-trail

## Bot 2 — CrashBot (`bot_binance_crashbot.py`)

- **Exchange** : Binance Spot (USDC)
- **Stratégie** : Dip-buy (long only) — achète les baisses brutales, sort en step-trail
- **Paires** : Auto-discovery de toutes les paires USDC en status TRADING
- **Capital** : Portion dominante via `allocator.py` (60–90% selon PF Trail)
- **Boucle** : Polling continu, détection de crash via `crashbot_detector.py`

## Bot 3 — Momentum Continuation (`bot_momentum.py`)

- **Exchange** : Revolut X (USD) — Maker 0% fees, Taker 0.09%
- **Stratégie** : Momentum Continuation en 3 phases :
  1. **Filtre macro M15** : ATR > MA(ATR), volume > MA(volume) → marché actif
  2. **Impulsion M5** : Bougie avec body ≥ 0.4%, volume ≥ 2× MA20, close dans top 20%, ADX > 15
  3. **Pullback** : Retracement 25–55% du move, RSI 40–65, prix touche EMA20
  4. **Entrée** : Bougie de reprise avec volume > MA10, dans la direction de l'impulsion
- **Paires** : `MC_TRADING_PAIRS` (ETH, SOL, BNB, XRP, LINK, ADA, LTC)
- **Exécution** : Ordres limit maker-only (0% fee), avec fallback taker si le fill ne se fait pas dans `MC_MAKER_WAIT_SECONDS`
- **Risk** : 4% par trade (`MC_RISK_PERCENT`), max 3 positions simultanées (`MC_MAX_POSITIONS`)
- **Capital** : Solde USD disponible sur Revolut X (indépendant de l'allocation Binance)

### Money management (`risk_manager.py`)
```python
# Calcul de taille partagé par tous les bots — NE PAS changer sans validation
risk_amount = account_balance * risk_percent   # ex: 500 * 0.04 = 20 USD
sl_distance = abs(entry_price - sl_price)      # en prix
position_size = risk_amount / sl_distance      # en unités de base (ex: ETH)
# Plafond : position_size * entry_price ≤ account_balance * max_position_pct
```

## Conventions de code

- **Séparation stricte I/O / logique** : `src/core/` ne fait AUCUN appel réseau. Les tests de `core/` doivent tourner sans mock d'API.
- **Types** : utiliser des `dataclass` ou `Pydantic BaseModel` pour toutes les structures de données.
- **Enums** pour les états : `AllocationRegime(Enum): DEFENSIVE, NEUTRAL, AGGRESSIVE`, etc.
- **Logging** : module `logging` standard avec le format `[%(asctime)s] %(levelname)s %(name)s: %(message)s`.
- **Config** : toutes les valeurs sensibles et paramètres dans `.env`, chargés via `python-dotenv`. Ne jamais hardcoder de clé API ou de paramètre de risque.
- **Firebase** : Toute persistance passe par `src/firebase/`. Les trades, heartbeats, allocations, et snapshots sont stockés dans Firestore.

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

# ── CrashBot (bot_binance_crashbot.py) ──
# (partage le même BINANCE_API_KEY, capital via allocator)

# ── Momentum (bot_momentum.py) ──
MC_TRADING_PAIRS=ETH-USD,SOL-USD,BNB-USD,XRP-USD,LINK-USD,ADA-USD,LTC-USD
MC_RISK_PERCENT=0.04
MC_MAX_POSITIONS=3
MC_MAX_POSITION_PCT=0.90
MC_POLLING_SECONDS=30
MC_HEARTBEAT_SECONDS=600
MC_MAKER_WAIT_SECONDS=60
```

## APIs – Points clés

### Binance Spot (Trail Range + CrashBot)
- **Base URL** : `https://api.binance.com`
- **Auth** : HMAC-SHA256 (API key + secret)
- **Ordres** : OCO orders natifs (TP + SL simultanés), limit, market
- **Symboles** : Format `BASEUSDC` (ex: `BTCUSDC`, `ETHUSDC`)

### Revolut X (Momentum)
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
| `/candles/{symbol}` | GET | Bougies OHLCV M5/M15 pour le Momentum |
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
tradex-momentum             # Bot Momentum (Revolut X)
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
python -m src.bot_momentum --dry-run

# Lancer le dashboard en local
streamlit run dashboard/app_unified.py --server.port 8502

# Se connecter au VPS
ssh BOT-VPS

# Voir les logs d'un bot sur le VPS
sudo journalctl -u tradex-binance -f
sudo journalctl -u tradex-binance-crashbot -f
sudo journalctl -u tradex-momentum -f

# Redémarrer un service
sudo systemctl restart tradex-binance
```

## Dashboard unifié (`dashboard/app_unified.py`)

Dashboard Streamlit avec 4 onglets :
1. **Overview** : Allocation Binance (gauge CrashBot/Trail), PF 90j, KPIs globaux, equity cumulée des 3 bots
2. **Trail Range** : Positions ouvertes, trades récents, PnL journalier
3. **CrashBot** : Positions ouvertes, trades récents, statistiques de dip-buy
4. **Momentum** : Positions Revolut X, signaux détectés, equity

L'Overview affiche la répartition dynamique du capital Binance entre CrashBot et Trail Range avec un indicateur visuel (stacked bar). Le bot Momentum (Revolut X) est affiché séparément car son capital est indépendant.

## Notifications Telegram

Chaque notification contient : **paire**, **action**, **prix d'entrée**, **SL**, **taille de position**, et un emoji indicatif.
```
📉 SELL déclenché – BTC-USD
  Entrée: 94500.00 | SL: 96200.00 | Size: 0.003 BTC
  Risque: 4% (20.00 USD)
```
Alertes envoyées pour : signal d'entrée, fill d'ordre, stop-loss touché, changement de tendance, déclenchement zero-risk, clôture de position, changement d'allocation.
