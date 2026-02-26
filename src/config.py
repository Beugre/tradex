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
# 0 = pas de plafond (100% du USDC dispo). Ex: 500 = max $500 de capital pour le sizing
BINANCE_RANGE_ALLOCATED_BALANCE: float = float(
    os.getenv("BINANCE_RANGE_ALLOCATED_BALANCE", "0")
)

# ── Binance Breakout Bot ───────────────────────────────────────────────────────
BINANCE_BREAKOUT_ALLOCATED_BALANCE: float = float(
    os.getenv("BINANCE_BREAKOUT_ALLOCATED_BALANCE", "0")
)
BINANCE_BREAKOUT_RISK_PERCENT: float = float(
    os.getenv("BINANCE_BREAKOUT_RISK_PERCENT", "0.02")
)
BINANCE_BREAKOUT_MAX_POSITIONS: int = int(
    os.getenv("BINANCE_BREAKOUT_MAX_POSITIONS", "3")
)
BINANCE_BREAKOUT_POLLING_SECONDS: int = int(
    os.getenv("BINANCE_BREAKOUT_POLLING_SECONDS", "30")
)
# Paires Breakout (si vide → auto-discovery des USDC pairs)
BINANCE_BREAKOUT_PAIRS: list[str] = [
    p.strip() for p in
    os.getenv("BINANCE_BREAKOUT_PAIRS", "").split(",")
    if p.strip()
]
# Breakout detector params
BINANCE_BREAKOUT_BB_PERIOD: int = int(os.getenv("BINANCE_BREAKOUT_BB_PERIOD", "20"))
BINANCE_BREAKOUT_BB_STD: float = float(os.getenv("BINANCE_BREAKOUT_BB_STD", "2.0"))
BINANCE_BREAKOUT_BB_EXPANSION: float = float(os.getenv("BINANCE_BREAKOUT_BB_EXPANSION", "1.2"))
BINANCE_BREAKOUT_DONCHIAN_PERIOD: int = int(os.getenv("BINANCE_BREAKOUT_DONCHIAN_PERIOD", "20"))
BINANCE_BREAKOUT_ADX_THRESHOLD: float = float(os.getenv("BINANCE_BREAKOUT_ADX_THRESHOLD", "25.0"))
BINANCE_BREAKOUT_VOL_MULT: float = float(os.getenv("BINANCE_BREAKOUT_VOL_MULT", "1.2"))
BINANCE_BREAKOUT_SL_ATR_MULT: float = float(os.getenv("BINANCE_BREAKOUT_SL_ATR_MULT", "1.5"))
BINANCE_BREAKOUT_TRAIL_ATR_MULT: float = float(os.getenv("BINANCE_BREAKOUT_TRAIL_ATR_MULT", "2.0"))
# Adaptive trailing (paliers)
BINANCE_BREAKOUT_ADAPTIVE_TRAIL: bool = os.getenv(
    "BINANCE_BREAKOUT_ADAPTIVE_TRAIL", "true"
).lower() in ("true", "1", "yes")
BINANCE_BREAKOUT_TRAIL_STEP1_PCT: float = float(os.getenv("BINANCE_BREAKOUT_TRAIL_STEP1_PCT", "0.02"))
BINANCE_BREAKOUT_TRAIL_STEP2_PCT: float = float(os.getenv("BINANCE_BREAKOUT_TRAIL_STEP2_PCT", "0.05"))
BINANCE_BREAKOUT_TRAIL_LOCK1_PCT: float = float(os.getenv("BINANCE_BREAKOUT_TRAIL_LOCK1_PCT", "0.002"))
BINANCE_BREAKOUT_TRAIL_LOCK2_PCT: float = float(os.getenv("BINANCE_BREAKOUT_TRAIL_LOCK2_PCT", "0.02"))
# Kill-switch mensuel
BINANCE_BREAKOUT_KILL_SWITCH: bool = os.getenv(
    "BINANCE_BREAKOUT_KILL_SWITCH", "true"
).lower() in ("true", "1", "yes")
BINANCE_BREAKOUT_KILL_PCT: float = float(os.getenv("BINANCE_BREAKOUT_KILL_PCT", "-0.10"))

# ── Monitoring & Alertes (Breakout) ────────────────────────────────────────────
BREAKOUT_HEARTBEAT_TELEGRAM_SECONDS: int = int(
    os.getenv("BREAKOUT_HEARTBEAT_TELEGRAM_SECONDS", "3600")  # Heartbeat Telegram 1x/heure
)
API_SLOW_THRESHOLD_MS: float = float(os.getenv("API_SLOW_THRESHOLD_MS", "5000"))
DATA_STALE_THRESHOLD_SECONDS: int = int(
    os.getenv("DATA_STALE_THRESHOLD_SECONDS", "18000")  # 5h (H4 + 1h marge)
)
SLIPPAGE_WARNING_PCT: float = float(os.getenv("SLIPPAGE_WARNING_PCT", "0.005"))
DD_WARNING_PCT: float = float(os.getenv("DD_WARNING_PCT", "-0.05"))

# ── Telegram ───────────────────────────────────────────────────────────────────
TELEGRAM_BOT_TOKEN: str = os.getenv("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID: str = os.getenv("TELEGRAM_CHAT_ID", "")

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
    os.getenv("RANGE_ENTRY_BUFFER_PERCENT", "0.002")
)
RANGE_SL_BUFFER_PERCENT: float = float(
    os.getenv("RANGE_SL_BUFFER_PERCENT", "0.003")
)
RANGE_WIDTH_MIN: float = float(os.getenv("RANGE_WIDTH_MIN", "0.02"))
RANGE_COOLDOWN_BARS: int = int(os.getenv("RANGE_COOLDOWN_BARS", "3"))

# ── Maker-First Order Execution ────────────────────────────────────────────────
# Place un ordre limit passif (maker 0%) → attend X secondes → si pas rempli,
# annule et place un limit agressif (taker 0.09%)
MAKER_WAIT_SECONDS: int = int(os.getenv("MAKER_WAIT_SECONDS", "30"))

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
    os.getenv("FIREBASE_EVENTS_RETENTION_DAYS", "30")
)
