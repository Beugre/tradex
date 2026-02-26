# TradeX – Copilot Instructions

## Aperçu du projet

Robot de trading crypto automatisé basé sur la **Dow Theory** (détection HH/HL/LH/LL), exécutant des ordres via l'**API Revolut X** (Crypto Exchange) et envoyant des notifications via **Telegram Bot API**.

- **Actifs** : Paires crypto à forte liquidité et tendances claires (compatibles Dow Theory) :
  - `BTC-USD` – Bitcoin : tendances macro fortes, swings H4 bien définis
  - `ETH-USD` – Ethereum : corrélé BTC mais avec ses propres structures
  - `SOL-USD` – Solana : volatilité élevée, bons mouvements tendanciels
  - `XRP-USD` – Ripple : phases de range puis breakouts nets
- **Timeframe** : H4 (bougies de 4 heures, intervalle API = `240` minutes)
- **Langage** : Python 3.12+
- **Entrée en position** : ordres limit simulant des stop orders (Buy Stop / Sell Stop) dans le sens de la tendance
- **Déploiement** : VPS (connexion via alias `vps-connexion`)

## Architecture

```
src/
├── core/                  # Logique métier pure (sans I/O)
│   ├── swing_detector.py  # Détection des swings (sommets/creux) sur données OHLC
│   ├── trend_engine.py    # Classification HH/HL/LH/LL et état de tendance
│   ├── order_manager.py   # Calcul du prix d'entrée, SL, taille de position
│   └── risk_manager.py    # Money management (% risque, sizing, zero-risk)
├── exchange/
│   ├── revolut_client.py  # Wrapper API Revolut X (auth Ed25519, signature par requête)
│   └── data_provider.py   # Récupération des bougies H4 (OHLCV) via GET /candles/{symbol}
├── notifications/
│   └── telegram.py        # Envoi d'alertes Telegram (entrée, SL, clôture, changement tendance)
├── bot.py                 # Boucle principale : polling 30s prix + analyse H4 à chaque nouvelle bougie
└── config.py              # Chargement .env (clés API, paramètres de risque)
tests/
├── test_swing_detector.py
├── test_trend_engine.py
├── test_order_manager.py
└── test_risk_manager.py
```

## Algorithme de trading – Règles critiques

### Détection des swings (`swing_detector.py`)
Identifier les **swing highs** et **swing lows** sur les bougies H4 (minimum 3 bougies de confirmation : la bougie pivot doit avoir un high/low plus extrême que ses N voisines de chaque côté).

### Classification de tendance (`trend_engine.py`)
- **Uptrend** : séquence `HH` puis `HL` → le dernier sommet est plus haut ET le dernier creux est plus haut que les précédents.
- **Downtrend** : séquence `LH` puis `LL` → le dernier sommet est plus bas ET le dernier creux est plus bas.
- **Invalidation** : un prix qui casse le dernier `LH` en downtrend ou le dernier `HL` en uptrend invalide la tendance → passer en état `NEUTRAL`, ne plus poser d'ordres.

### Placement des ordres (`order_manager.py`)

**⚠️ Contrainte API Revolut X** : l'API ne supporte que les ordres `limit` (pas de stop order natif). Le bot doit **simuler les stop orders** :
- Le bot surveille le prix (via `GET /tickers` ou `GET /candles/{symbol}`) à chaque cycle.
- Quand le prix atteint le seuil d'entrée → placer un ordre `limit` au prix du marché via `POST /orders`.

| Tendance | Simulation | Seuil de déclenchement | Ordre limit placé | Stop Loss (surveillé par le bot) |
|-----------|------------|------------------------|-------------------|----------------------------------|
| Downtrend | Sell Stop simulé | Prix ≤ dernier `LL` - buffer | Sell limit au marché | Si prix ≥ dernier `LH` + marge → sell pour couper |
| Uptrend | Buy Stop simulé | Prix ≥ dernier `HH` + buffer | Buy limit au marché | Si prix ≤ dernier `HL` - marge → sell pour couper |

La **marge** (buffer) est configurable via `ENTRY_BUFFER_PIPS` et `SL_BUFFER_PIPS` dans `.env`.

### Money management (`risk_manager.py`)
```python
# Pseudo-code du calcul de taille – NE PAS changer cette logique sans validation
risk_amount = account_balance * risk_percent  # ex: 1000 * 0.05 = 50 USD
sl_distance = abs(entry_price - sl_price)     # en prix
position_size = risk_amount / sl_distance     # en unités de base (ex: BTC)
```
- `risk_percent` : configurable, défaut 5% (`RISK_PERCENT=0.05`)
- Toujours vérifier que `position_size` respecte les contraintes min/max de la paire avant de soumettre

### Gestion de position – Zero Risk
1. Quand le prix a parcouru `ZERO_RISK_TRIGGER_PERCENT` (ex: 2%) en faveur du trade :
   - Placer un ordre limit opposé pour verrouiller `ZERO_RISK_LOCK_PERCENT` (ex: 0.5%) de profit
2. Ce trailing s'applique **une seule fois** par trade (flag `is_zero_risk_applied`)
3. Si le trade se clôture en gain → vérifier si la structure de tendance est toujours valide → reposer un ordre si oui

## Conventions de code

- **Séparation stricte I/O / logique** : `src/core/` ne fait AUCUN appel réseau. Les tests de `core/` doivent tourner sans mock d'API.
- **Types** : utiliser des `dataclass` ou `Pydantic BaseModel` pour toutes les structures : `SwingPoint`, `TrendState`, `Order`, `Position`.
- **Enums** pour les états : `TrendDirection(Enum): BULLISH, BEARISH, NEUTRAL` ; `SwingType(Enum): HH, HL, LH, LL`.
- **Logging** : utiliser le module `logging` standard avec le format `[%(asctime)s] %(levelname)s %(name)s: %(message)s`. Logger chaque détection de swing, changement de tendance, placement/modification d'ordre.
- **Config** : toutes les valeurs sensibles et paramètres de trading dans `.env`, chargés via `python-dotenv`. Ne jamais hardcoder de clé API ou de paramètre de risque.

## Variables d'environnement (`.env`)

```env
# Revolut X API
REVOLUT_X_API_KEY=xxx                        # Clé API 64 chars obtenue sur exchange.revolut.com
REVOLUT_X_PRIVATE_KEY_PATH=./private.pem     # Clé privée Ed25519 pour signer les requêtes

# Telegram
TELEGRAM_BOT_TOKEN=xxx
TELEGRAM_CHAT_ID=xxx

# Trading parameters
RISK_PERCENT=0.05
ENTRY_BUFFER_PIPS=5
SL_BUFFER_PIPS=10
ZERO_RISK_TRIGGER_PERCENT=0.02
ZERO_RISK_LOCK_PERCENT=0.005
SWING_LOOKBACK=3

# Assets (format Revolut X : BASE-QUOTE)
TRADING_PAIRS=BTC-USD,ETH-USD,SOL-USD,XRP-USD
TIMEFRAME=H4
POLLING_INTERVAL_SECONDS=30
```

## API Revolut X – Points clés

- **Base URL** : `https://revx.revolut.com/api/1.0/`
- **Doc** : [developer.revolut.com/docs/x-api](https://developer.revolut.com/docs/x-api/revolut-x-crypto-exchange-rest-api)
- **Rate limit** : 1000 requêtes/minute sur tous les endpoints
- **Auth** : signature Ed25519 par requête (PAS de token/session) :
  - Header `X-Revx-API-Key` : clé API 64 chars
  - Header `X-Revx-Timestamp` : timestamp Unix ms
  - Header `X-Revx-Signature` : signature Ed25519 base64 du message construit comme :
    `{timestamp}{METHOD}{path}{query_string}{body}`
  - **Pas de séparateur** entre les champs dans le message à signer
  - Clé privée générée avec : `openssl genpkey -algorithm ed25519 -out private.pem`

### Endpoints utilisés par le bot

| Endpoint | Méthode | Usage |
|----------|---------|-------|
| `/balances` | GET | Solde du compte (available/reserved/total par devise) pour le sizing |
| `/candles/{symbol}` | GET | Bougies OHLCV H4 pour l'analyse des swings |
| `/tickers` | GET | Prix temps réel (bid/ask/mid/last) pour surveiller les seuils d'entrée/SL |
| `/orders` | POST | Placer un ordre limit (buy/sell) |
| `/orders/active` | GET | Lister les ordres en cours |
| `/orders/{venue_order_id}` | DELETE | Annuler un ordre |
| `/orders/{venue_order_id}` | GET | Détails d'un ordre |
| `/orders/{venue_order_id}/fills` | GET | Fills (exécutions) d'un ordre |

### Réponse `GET /balances`
```json
[
  { "currency": "USD", "available": "1000.00", "reserved": "50.00", "total": "1050.00" },
  { "currency": "BTC", "available": "0.005", "reserved": "0.001", "total": "0.006" }
]
```
- Utiliser `available` (pas `total`) pour calculer le `risk_amount` du money management
- Le solde USD sert de base pour `account_balance` dans le calcul de position

### Format des ordres (`POST /orders`)
```json
{
  "client_order_id": "uuid-v4",
  "symbol": "BTC-USD",
  "side": "buy",
  "order_configuration": {
    "limit": {
      "base_size": "0.001",
      "price": "95000.00"
    }
  }
}
```
- **Pas de stop order natif** : le bot simule les stops en surveillant le prix et en plaçant des ordres limit.
- `symbol` : format `BASE-QUOTE` (ex: `BTC-USD`, pas `BTCUSD`)
- `side` : `"buy"` ou `"sell"` (minuscules)

## Commandes de développement

```bash
# Installer les dépendances
pip install -r requirements.txt

# Lancer les tests (core uniquement, pas d'API)
pytest tests/ -v

# Lancer le bot en mode dry-run (log les ordres sans les exécuter)
python -m src.bot --dry-run

# Lancer le bot en production
python -m src.bot

# Déployer / se connecter au VPS
vps-connexion
```

## Boucle principale (`bot.py`) – Deux rythmes

1. **Toutes les 30 secondes** (polling rapide) :
   - `GET /tickers` → vérifier si le prix a atteint un seuil d'entrée ou de SL
   - Si seuil d'entrée atteint → `POST /orders` (limit au marché)
   - Si seuil SL atteint → `POST /orders` (limit opposé pour couper)
   - Si conditions zero-risk remplies → ajuster la protection

2. **À chaque nouvelle bougie H4** (toutes les ~4h, détecté via timestamp des candles) :
   - `GET /candles/{symbol}` → récupérer les dernières bougies
   - Recalculer les swings et la tendance
   - Mettre à jour les seuils d'entrée / SL si la structure a changé
   - `GET /balances` → recalculer la taille de position

## Notifications Telegram

Chaque notification doit contenir : **paire**, **action**, **prix d'entrée**, **SL**, **taille de position**, et un emoji indicatif.
```
📉 SELL déclenché – BTC-USD
  Entrée: 94500.00 | SL: 96200.00 | Size: 0.003 BTC
  Risque: 5% (50.00 USD)
```
Envoyer aussi des alertes pour : changement de tendance, déclenchement d'un seuil, passage en zero-risk, clôture de position.
