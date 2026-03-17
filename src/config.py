"""
Configuration du bot TradeX.
Charge les variables d'environnement depuis .env et expose des objets typés.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()


# ── Revolut X API ──────────────────────────────────────────────────────────────
REVOLUT_X_API_KEY: str = os.getenv("REVOLUT_X_API_KEY", "")
REVOLUT_X_PRIVATE_KEY_PATH: Path = Path(
    os.getenv("REVOLUT_X_PRIVATE_KEY_PATH", "./private.pem")
)
REVOLUT_X_BASE_URL: str = "https://revx.revolut.com/api/1.0"

# ── Binance API ────────────────────────────────────────────────────────────────
BINANCE_API_KEY: str = os.getenv("BINANCE_API_KEY", "")
BINANCE_SECRET_KEY: str = os.getenv("BINANCE_SECRET_KEY", "")
BINANCE_BASE_URL: str = os.getenv("BINANCE_BASE_URL", "https://api.binance.com")

# Paires USDC Binance (format Binance : BASEUSDC sans tiret)
BINANCE_TRADING_PAIRS: list[str] = [
    p.strip() for p in
    os.getenv("BINANCE_TRADING_PAIRS", "").split(",")
    if p.strip()
]
# Si vide, le bot fera un auto-discovery de toutes les paires USDC
BINANCE_AUTO_DISCOVER_PAIRS: bool = os.getenv(
    "BINANCE_AUTO_DISCOVER_PAIRS", "true"
).lower() in ("true", "1", "yes")

# Paramètres spécifiques Binance
BINANCE_RISK_PERCENT_RANGE: float = float(
    os.getenv("BINANCE_RISK_PERCENT_RANGE", "0.02")
)
BINANCE_MAX_SIMULTANEOUS_POSITIONS: int = int(
    os.getenv("BINANCE_MAX_SIMULTANEOUS_POSITIONS", "3")
)
BINANCE_POLLING_INTERVAL_SECONDS: int = int(
    os.getenv("BINANCE_POLLING_INTERVAL_SECONDS", "30")
)
# SL limit offset (pour STOP_LOSS_LIMIT, le limit price est légèrement au-delà du stop)
BINANCE_SL_LIMIT_OFFSET_PCT: float = float(
    os.getenv("BINANCE_SL_LIMIT_OFFSET_PCT", "0.002")
)

# Fees Binance (maker 0.1%, taker 0.1% — avec BNB: maker 0.075%, taker 0.075%)
BINANCE_MAKER_FEE: float = float(os.getenv("BINANCE_MAKER_FEE", "0.001"))
BINANCE_TAKER_FEE: float = float(os.getenv("BINANCE_TAKER_FEE", "0.001"))

# ── Capital alloué (partage du même compte entre bots) ────────────────────────
# Allocation dynamique (True = l'allocator recalcule quotidiennement basé sur le PF Trail Range)
# Le total est calculé automatiquement depuis le solde réel Binance (USDC + positions)
DYNAMIC_ALLOCATION_ENABLED: bool = os.getenv(
    "DYNAMIC_ALLOCATION_ENABLED", "true"
).lower() in ("true", "1", "yes")
# 0 = pas de plafond (100% du USDC dispo). Ex: 500 = max $500 de capital pour le sizing
BINANCE_RANGE_ALLOCATED_BALANCE: float = float(
    os.getenv("BINANCE_RANGE_ALLOCATED_BALANCE", "0")
)

API_SLOW_THRESHOLD_MS: float = float(os.getenv("API_SLOW_THRESHOLD_MS", "5000"))
DATA_STALE_THRESHOLD_SECONDS: int = int(
    os.getenv("DATA_STALE_THRESHOLD_SECONDS", "18000")  # 5h (H4 + 1h marge)
)
SLIPPAGE_WARNING_PCT: float = float(os.getenv("SLIPPAGE_WARNING_PCT", "0.005"))
DD_WARNING_PCT: float = float(os.getenv("DD_WARNING_PCT", "-0.05"))

# ── Binance CrashBot (Dip Buy) ────────────────────────────────────────────────
BINANCE_CRASHBOT_ALLOCATED_BALANCE: float = float(
    os.getenv("BINANCE_CRASHBOT_ALLOCATED_BALANCE", "0")
)
BINANCE_CRASHBOT_RISK_PERCENT: float = float(
    os.getenv("BINANCE_CRASHBOT_RISK_PERCENT", "0.05")
)
BINANCE_CRASHBOT_MAX_POSITIONS: int = int(
    os.getenv("BINANCE_CRASHBOT_MAX_POSITIONS", "5")
)
BINANCE_CRASHBOT_POLLING_SECONDS: int = int(
    os.getenv("BINANCE_CRASHBOT_POLLING_SECONDS", "30")
)
BINANCE_CRASHBOT_PAIRS: list[str] = [
    p.strip() for p in
    os.getenv("BINANCE_CRASHBOT_PAIRS", "").split(",")
    if p.strip()
]
# Si vide et auto-discover=true, le bot scannera TOUTES les paires USDC
BINANCE_CRASHBOT_AUTO_DISCOVER_PAIRS: bool = os.getenv(
    "BINANCE_CRASHBOT_AUTO_DISCOVER_PAIRS", "true"
).lower() in ("true", "1", "yes")
# Crash detection params
BINANCE_CRASHBOT_DROP_THRESHOLD: float = float(
    os.getenv("BINANCE_CRASHBOT_DROP_THRESHOLD", "0.20")
)
BINANCE_CRASHBOT_LOOKBACK_BARS: int = int(
    os.getenv("BINANCE_CRASHBOT_LOOKBACK_BARS", "12")
)
BINANCE_CRASHBOT_TP_PCT: float = float(
    os.getenv("BINANCE_CRASHBOT_TP_PCT", "0.08")
)
BINANCE_CRASHBOT_SL_PCT: float = float(
    os.getenv("BINANCE_CRASHBOT_SL_PCT", "0.02")
)
BINANCE_CRASHBOT_ATR_SL_MULT: float = float(
    os.getenv("BINANCE_CRASHBOT_ATR_SL_MULT", "1.5")
)
BINANCE_CRASHBOT_ATR_PERIOD: int = int(
    os.getenv("BINANCE_CRASHBOT_ATR_PERIOD", "14")
)
BINANCE_CRASHBOT_TRAIL_STEP_PCT: float = float(
    os.getenv("BINANCE_CRASHBOT_TRAIL_STEP_PCT", "0.005")
)
BINANCE_CRASHBOT_TRAIL_TRIGGER_BUFFER: float = float(
    os.getenv("BINANCE_CRASHBOT_TRAIL_TRIGGER_BUFFER", "0.0005")
)
BINANCE_CRASHBOT_TRAIL_SL_LOCK_RATIO: float = float(
    os.getenv("BINANCE_CRASHBOT_TRAIL_SL_LOCK_RATIO", "0.80")
)
BINANCE_CRASHBOT_TRAIL_TP_MULT: float = float(
    os.getenv("BINANCE_CRASHBOT_TRAIL_TP_MULT", "1.20")
)
BINANCE_CRASHBOT_COOLDOWN_BARS: int = int(
    os.getenv("BINANCE_CRASHBOT_COOLDOWN_BARS", "6")
)
# Kill-switch mensuel
BINANCE_CRASHBOT_KILL_SWITCH: bool = os.getenv(
    "BINANCE_CRASHBOT_KILL_SWITCH", "true"
).lower() in ("true", "1", "yes")
BINANCE_CRASHBOT_KILL_PCT: float = float(
    os.getenv("BINANCE_CRASHBOT_KILL_PCT", "-0.10")
)
# Momentum Sizing — ajuste le risk% dynamiquement selon W/L du trade précédent
# Après un WIN  → risk *= BOOST (1.2)  → plafond MAX
# Après un LOSS → risk *= SHRINK (0.8) → plancher MIN
BINANCE_CRASHBOT_MOMENTUM_SIZING: bool = os.getenv(
    "BINANCE_CRASHBOT_MOMENTUM_SIZING", "true"
).lower() in ("true", "1", "yes")
BINANCE_CRASHBOT_RISK_BOOST_MULT: float = float(
    os.getenv("BINANCE_CRASHBOT_RISK_BOOST_MULT", "1.2")
)
BINANCE_CRASHBOT_RISK_SHRINK_MULT: float = float(
    os.getenv("BINANCE_CRASHBOT_RISK_SHRINK_MULT", "0.8")
)
BINANCE_CRASHBOT_MIN_RISK_PCT: float = float(
    os.getenv("BINANCE_CRASHBOT_MIN_RISK_PCT", "0.02")
)
BINANCE_CRASHBOT_MAX_RISK_PCT: float = float(
    os.getenv("BINANCE_CRASHBOT_MAX_RISK_PCT", "0.10")
)

# Heartbeat Telegram CrashBot
CRASHBOT_HEARTBEAT_TELEGRAM_SECONDS: int = int(
    os.getenv("CRASHBOT_HEARTBEAT_TELEGRAM_SECONDS", "600")
)

# ── Telegram ───────────────────────────────────────────────────────────────────
TELEGRAM_BOT_TOKEN: str = os.getenv("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID: str = os.getenv("TELEGRAM_CHAT_ID", "")
TELEGRAM_COMMANDS_POLL_SECONDS: int = int(
    os.getenv("TELEGRAM_COMMANDS_POLL_SECONDS", "2")
)

# ── Paramètres de trading ──────────────────────────────────────────────────────
# Risque par stratégie
RISK_PERCENT_TREND: float = float(os.getenv("RISK_PERCENT_TREND", "0.03"))
RISK_PERCENT_RANGE: float = float(os.getenv("RISK_PERCENT_RANGE", "0.02"))
MAX_TOTAL_RISK_PERCENT: float = float(
    os.getenv("MAX_TOTAL_RISK_PERCENT", "0.06")
)
# Rétro-compatibilité : RISK_PERCENT pointe vers TREND
RISK_PERCENT: float = RISK_PERCENT_TREND

ENTRY_BUFFER_PERCENT: float = float(os.getenv("ENTRY_BUFFER_PERCENT", "0.002"))
SL_BUFFER_PERCENT: float = float(os.getenv("SL_BUFFER_PERCENT", "0.003"))
ZERO_RISK_TRIGGER_PERCENT: float = float(
    os.getenv("ZERO_RISK_TRIGGER_PERCENT", "0.02")
)
ZERO_RISK_LOCK_PERCENT: float = float(
    os.getenv("ZERO_RISK_LOCK_PERCENT", "0.005")
)
TRAILING_STOP_PERCENT: float = float(
    os.getenv("TRAILING_STOP_PERCENT", "0.02")
)
RECOVERY_SL_PERCENT: float = float(
    os.getenv("RECOVERY_SL_PERCENT", "0.05")
)
MAX_POSITION_PERCENT: float = float(
    os.getenv("MAX_POSITION_PERCENT", "0.30")
)
MAX_SIMULTANEOUS_POSITIONS: int = int(
    os.getenv("MAX_SIMULTANEOUS_POSITIONS", "3")
)
SWING_LOOKBACK: int = int(os.getenv("SWING_LOOKBACK", "3"))

# ── Paramètres Mean-Reversion (Range) ──────────────────────────────────────────
RANGE_ENTRY_BUFFER_PERCENT: float = float(
    os.getenv("RANGE_ENTRY_BUFFER_PERCENT", "0.003")
)
RANGE_SL_BUFFER_PERCENT: float = float(
    os.getenv("RANGE_SL_BUFFER_PERCENT", "0.008")
)
RANGE_WIDTH_MIN: float = float(os.getenv("RANGE_WIDTH_MIN", "0.03"))
RANGE_COOLDOWN_BARS: int = int(os.getenv("RANGE_COOLDOWN_BARS", "3"))
# TP ratio : 0.5 = mid, 0.75 = 3/4 du range (optimisé via backtest)
RANGE_TP_RATIO: float = float(os.getenv("RANGE_TP_RATIO", "0.75"))

# Trail@TP : désactivé par défaut (backtest montre que le trail nuit au mean-reversion)
RANGE_TRAIL_ENABLED: bool = os.getenv(
    "RANGE_TRAIL_ENABLED", "false"
).lower() in ("true", "1", "yes")

# Step-trail : paliers discrets basés sur le range (compatible mean-reversion)
# Quand prix atteint TP (75%), on ne ferme pas : on décale SL/TP par paliers
# Step 1 : SL → ratio_sl du range, TP → ratio_tp du range
# Step N : SL et TP décalent de +step_size chacun
RANGE_STEP_TRAIL_ENABLED: bool = os.getenv(
    "RANGE_STEP_TRAIL_ENABLED", "false"
).lower() in ("true", "1", "yes")
RANGE_STEP_TRAIL_INITIAL_SL_RATIO: float = float(
    os.getenv("RANGE_STEP_TRAIL_INITIAL_SL_RATIO", "0.60")
)
RANGE_STEP_TRAIL_INITIAL_TP_RATIO: float = float(
    os.getenv("RANGE_STEP_TRAIL_INITIAL_TP_RATIO", "0.85")
)
RANGE_STEP_TRAIL_STEP_SIZE: float = float(
    os.getenv("RANGE_STEP_TRAIL_STEP_SIZE", "0.05")
)
# Trail@TP : avant que le TP OCO ne fill, on swap vers un nouvel OCO
# avec SL = TP_actuel × (1 - SL_LOCK) et TP = TP_actuel × (1 + STEP)
BINANCE_RANGE_TRAIL_STEP_PCT: float = float(
    os.getenv("BINANCE_RANGE_TRAIL_STEP_PCT", "0.01")   # +1% par palier
)
BINANCE_RANGE_TRAIL_SL_LOCK_PCT: float = float(
    os.getenv("BINANCE_RANGE_TRAIL_SL_LOCK_PCT", "0.02")  # SL = 0.98 × TP actuel
)
BINANCE_RANGE_TRAIL_SWAP_PCT: float = float(
    os.getenv("BINANCE_RANGE_TRAIL_SWAP_PCT", "0.005")   # swap quand < 0.5% du TP
)

# ── Maker-First Order Execution ────────────────────────────────────────────────
# Place un ordre limit passif (maker 0%) → attend X secondes → si pas rempli,
# annule et place un limit agressif (taker 0.09%)
MAKER_WAIT_SECONDS: int = int(os.getenv("MAKER_WAIT_SECONDS", "30"))

# ── Infinity Bot (bot_infinity.py) ─────────────────────────────────────────────
INF_TRADING_PAIR: str = os.getenv("INF_TRADING_PAIR", "BTC-USD")  # legacy, single pair
INF_TRADING_PAIRS: list[str] = [
    p.strip() for p in os.getenv("INF_TRADING_PAIRS", "BTC-USD,AAVE-USD,XLM-USD").split(",")
]
INF_POLLING_SECONDS: int = int(os.getenv("INF_POLLING_SECONDS", "30"))
INF_HEARTBEAT_SECONDS: int = int(os.getenv("INF_HEARTBEAT_SECONDS", "600"))
INF_MAKER_WAIT_SECONDS: int = int(os.getenv("INF_MAKER_WAIT_SECONDS", "60"))
INF_CAPITAL_PCT: float = float(os.getenv("INF_CAPITAL_PCT", "0.80"))  # 80% du capital Revolut X
INF_CAPITAL_ACTIVE_SLOTS: int = int(os.getenv("INF_CAPITAL_ACTIVE_SLOTS", "2"))  # Capital partagé sur 2 positions max

# Paramètres de stratégie
INF_ENTRY_DROP_PCT: float = float(os.getenv("INF_ENTRY_DROP_PCT", "0.05"))
INF_TRAILING_HIGH_PERIOD: int = int(os.getenv("INF_TRAILING_HIGH_PERIOD", "72"))
INF_STOP_LOSS_PCT: float = float(os.getenv("INF_STOP_LOSS_PCT", "0.15"))
INF_MAX_INVESTED_PCT: float = float(os.getenv("INF_MAX_INVESTED_PCT", "0.70"))
INF_BUY_LEVELS: str = os.getenv("INF_BUY_LEVELS", "-0.05,-0.10,-0.15,-0.20,-0.25")
INF_BUY_PCTS: str = os.getenv("INF_BUY_PCTS", "0.25,0.20,0.15,0.10,0.00")
INF_SELL_LEVELS: str = os.getenv("INF_SELL_LEVELS", "0.008,0.015,0.022,0.030,0.040")
INF_RSI_ENTRY_MAX: float = float(os.getenv("INF_RSI_ENTRY_MAX", "50.0"))
INF_USE_BREAKEVEN: bool = os.getenv("INF_USE_BREAKEVEN", "true").lower() in ("true", "1", "yes")

# ── Stratégies actives ─────────────────────────────────────────────────────────
ENABLE_TREND: bool = os.getenv("ENABLE_TREND", "false").lower() in ("true", "1", "yes")
ENABLE_RANGE: bool = os.getenv("ENABLE_RANGE", "true").lower() in ("true", "1", "yes")

# ── Actifs & timing ───────────────────────────────────────────────────────────
TRADING_PAIRS: list[str] = os.getenv(
    "TRADING_PAIRS", "BTC-USD,ETH-USD,SOL-USD,XRP-USD"
).split(",")
TIMEFRAME: str = os.getenv("TIMEFRAME", "H4")
POLLING_INTERVAL_SECONDS: int = int(
    os.getenv("POLLING_INTERVAL_SECONDS", "30")
)
HEARTBEAT_INTERVAL_SECONDS: int = int(
    os.getenv("HEARTBEAT_INTERVAL_SECONDS", "600")  # 10 minutes
)
LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO").upper()

# ── Firebase ───────────────────────────────────────────────────────────────────
FIREBASE_CREDENTIALS_PATH: Path = Path(
    os.getenv("FIREBASE_CREDENTIALS_PATH", "./firebase-credentials.json")
)
FIREBASE_ENABLED: bool = os.getenv(
    "FIREBASE_ENABLED", "true"
).lower() in ("true", "1", "yes")
FIREBASE_EVENTS_RETENTION_DAYS: int = int(
    os.getenv("FIREBASE_EVENTS_RETENTION_DAYS", "2")
)

# ── London Breakout Bot (bot_london.py) ────────────────────────────────────────
LON_TRADING_PAIRS: list[str] = [
    p.strip() for p in os.getenv(
        "LON_TRADING_PAIRS",
        "BTC-USD,ETH-USD,SOL-USD,BNB-USD,LINK-USD,ADA-USD,DOT-USD,AVAX-USD"
    ).split(",") if p.strip()
]
LON_POLLING_SECONDS: int = int(os.getenv("LON_POLLING_SECONDS", "30"))
LON_HEARTBEAT_SECONDS: int = int(os.getenv("LON_HEARTBEAT_SECONDS", "600"))
LON_MAKER_WAIT_SECONDS: int = int(os.getenv("LON_MAKER_WAIT_SECONDS", "60"))
LON_CAPITAL_PCT: float = float(os.getenv("LON_CAPITAL_PCT", "0.20"))  # 20% du capital Revolut X

# Stratégie London Breakout
LON_SESSION_START_HOUR: int = int(os.getenv("LON_SESSION_START_HOUR", "8"))   # UTC
LON_SESSION_END_HOUR: int = int(os.getenv("LON_SESSION_END_HOUR", "16"))      # UTC
LON_SL_ATR_MULT: float = float(os.getenv("LON_SL_ATR_MULT", "2.0"))
LON_TP1_PCT: float = float(os.getenv("LON_TP1_PCT", "0.02"))                 # +2%
LON_TP2_PCT: float = float(os.getenv("LON_TP2_PCT", "0.05"))                 # +5%
LON_TP1_SHARE: float = float(os.getenv("LON_TP1_SHARE", "0.50"))             # 50% au TP1
LON_VOL_MULT: float = float(os.getenv("LON_VOL_MULT", "2.0"))               # volume ≥ 2×MA20
LON_MIN_RANGE_PCT: float = float(os.getenv("LON_MIN_RANGE_PCT", "0.015"))    # 1.5%
LON_RISK_PERCENT: float = float(os.getenv("LON_RISK_PERCENT", "0.05"))       # 5% par trade
LON_MAX_POSITIONS: int = int(os.getenv("LON_MAX_POSITIONS", "1"))
LON_COOLDOWN_BARS: int = int(os.getenv("LON_COOLDOWN_BARS", "2"))            # 2 bougies H4 (8h)
LON_ATR_PERIOD: int = int(os.getenv("LON_ATR_PERIOD", "14"))
LON_VOL_MA_PERIOD: int = int(os.getenv("LON_VOL_MA_PERIOD", "20"))
LON_BREAKEVEN_AFTER_TP1: bool = os.getenv(
    "LON_BREAKEVEN_AFTER_TP1", "true"
).lower() in ("true", "1", "yes")

# ── Listing Bot (bot_binance_listing.py) ─────────────────────────────────────
# Fallback statique si DYNAMIC_ALLOCATION_ENABLED=false
LISTING_ALLOCATED_BALANCE: float = float(
    os.getenv("LISTING_ALLOCATED_BALANCE", "500")
)
# Part fixe du listing dans l'allocator (défaut 30%)
LISTING_CAPITAL_PCT: float = float(
    os.getenv("LISTING_CAPITAL_PCT", "0.30")
)
LISTING_MAX_SLOTS: int = int(
    os.getenv("LISTING_MAX_SLOTS", "3")
)
LISTING_MAX_ALLOC_USD: float = float(
    os.getenv("LISTING_MAX_ALLOC_USD", "5000")
)
LISTING_SL_INIT_PCT: float = float(
    os.getenv("LISTING_SL_INIT_PCT", "0.08")
)
LISTING_TP_INIT_PCT: float = float(
    os.getenv("LISTING_TP_INIT_PCT", "0.30")
)
LISTING_TP_NEAR_RATIO: float = float(
    os.getenv("LISTING_TP_NEAR_RATIO", "0.98")
)
LISTING_SL2_TP1_MULT: float = float(
    os.getenv("LISTING_SL2_TP1_MULT", "0.769")
)
LISTING_TP2_TP1_MULT: float = float(
    os.getenv("LISTING_TP2_TP1_MULT", "1.538")
)
LISTING_MOMENTUM_PCT: float = float(
    os.getenv("LISTING_MOMENTUM_PCT", "0.30")
)
LISTING_MOMENTUM_WINDOW_MIN: int = int(
    os.getenv("LISTING_MOMENTUM_WINDOW_MIN", "1")
)
LISTING_HORIZON_DAYS: int = int(
    os.getenv("LISTING_HORIZON_DAYS", "7")
)
LISTING_POLL_INTERVAL_SECONDS: int = int(
    os.getenv("LISTING_POLL_INTERVAL_SECONDS", "10")
)
LISTING_EXCHANGEINFO_CACHE_SECONDS: int = int(
    os.getenv("LISTING_EXCHANGEINFO_CACHE_SECONDS", "30")
)
LISTING_HEARTBEAT_SECONDS: int = int(
    os.getenv("LISTING_HEARTBEAT_SECONDS", "600")
)

# ── DCA Bot (bot_dca.py) ──────────────────────────────────────────────────────
# Budget dynamique : pourcentages du solde Revolut X (calculé au démarrage)
DCA_CAPITAL_PCT: float = float(os.getenv("DCA_CAPITAL_PCT", "1.0"))     # Part du solde Revolut X allouée au DCA
DCA_ACTIVE_PCT: float = float(os.getenv("DCA_ACTIVE_PCT", "0.80"))     # 80% du capital DCA → achats quotidiens
DCA_CRASH_PCT: float = float(os.getenv("DCA_CRASH_PCT", "0.20"))       # 20% du capital DCA → crash reserve

# Montant de base quotidien ($12, multiplié selon le bracket RSI)
DCA_BASE_DAILY_AMOUNT: float = float(os.getenv("DCA_BASE_DAILY_AMOUNT", "12.0"))

# Allocation BTC/ETH
DCA_BTC_ALLOC: float = float(os.getenv("DCA_BTC_ALLOC", "0.80"))  # 80% BTC
DCA_ETH_ALLOC: float = float(os.getenv("DCA_ETH_ALLOC", "0.20"))  # 20% ETH

# RSI thresholds (daily BTC)
DCA_RSI_OVERBOUGHT: float = float(os.getenv("DCA_RSI_OVERBOUGHT", "70.0"))
DCA_RSI_WARM: float = float(os.getenv("DCA_RSI_WARM", "55.0"))
DCA_RSI_NEUTRAL_LOW: float = float(os.getenv("DCA_RSI_NEUTRAL_LOW", "45.0"))

# Crash reserve levels (drop_pct, amount_usd)
DCA_CRASH_DROP_1: float = float(os.getenv("DCA_CRASH_DROP_1", "0.15"))   # -15%
DCA_CRASH_AMOUNT_1: float = float(os.getenv("DCA_CRASH_AMOUNT_1", "150"))
DCA_CRASH_DROP_2: float = float(os.getenv("DCA_CRASH_DROP_2", "0.25"))   # -25%
DCA_CRASH_AMOUNT_2: float = float(os.getenv("DCA_CRASH_AMOUNT_2", "250"))
DCA_CRASH_DROP_3: float = float(os.getenv("DCA_CRASH_DROP_3", "0.35"))   # -35%
DCA_CRASH_AMOUNT_3: float = float(os.getenv("DCA_CRASH_AMOUNT_3", "350"))
DCA_CRASH_LOOKBACK_DAYS: int = int(os.getenv("DCA_CRASH_LOOKBACK_DAYS", "90"))

# MVRV deep value
DCA_MVRV_ENABLED: bool = os.getenv("DCA_MVRV_ENABLED", "true").lower() in ("true", "1", "yes")
DCA_MVRV_THRESHOLD: float = float(os.getenv("DCA_MVRV_THRESHOLD", "1.0"))
DCA_MVRV_MULTIPLIER: float = float(os.getenv("DCA_MVRV_MULTIPLIER", "5.0"))

# Crash reserve → 100% BTC
DCA_CRASH_BTC_ONLY: bool = os.getenv("DCA_CRASH_BTC_ONLY", "true").lower() in ("true", "1", "yes")

# Timing
DCA_EXECUTION_HOUR_UTC: int = int(os.getenv("DCA_EXECUTION_HOUR_UTC", "10"))
DCA_POLLING_SECONDS: int = int(os.getenv("DCA_POLLING_SECONDS", "60"))
DCA_HEARTBEAT_SECONDS: int = int(os.getenv("DCA_HEARTBEAT_SECONDS", "3600"))
DCA_MAKER_WAIT_SECONDS: int = int(os.getenv("DCA_MAKER_WAIT_SECONDS", "60"))
