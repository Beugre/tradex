"""
Bot London Breakout — Revolut X, maker-only, H4.

Session 08-16 UTC : accumule le range des bougies H4 08:00 et 12:00
puis trade le breakout au-dessus du session_high.

Stratégie :
  1. Session range : bougies H4 entre SESSION_START et SESSION_END (08-16 UTC)
  2. Breakout LONG : close > session_high + volume ≥ VOL_MULT × MA20 + range ≥ MIN_RANGE
  3. SL = entry - SL_ATR_MULT × ATR(14)
  4. TP1 = entry × (1 + TP1_PCT) → ferme 50%, breakeven SL
  5. TP2 = entry × (1 + TP2_PCT) → ferme le reste
  6. LONG ONLY (Revolut X = spot)

Capital : 20% du solde Revolut X.
Allocation : INF_CAPITAL_PCT=0.80 pour Infinity, LON_CAPITAL_PCT=0.20 pour London.

Usage :
    python -m src.bot_london              # Production
    python -m src.bot_london --dry-run    # Log les ordres sans les exécuter
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import signal
import sys
import time
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo
from pathlib import Path
from typing import Any, Optional

from src import config
from src.core.models import (
    Balance,
    Candle,
    OrderRequest,
    OrderSide,
    Position,
    PositionStatus,
    StrategyType,
    TickerData,
)
from src.exchange.revolut_client import RevolutXClient
from src.exchange.data_provider import DataProvider
from src.notifications.telegram import TelegramNotifier, DASHBOARD_URL
from src.firebase.trade_logger import (
    log_trade_opened,
    log_trade_closed,
    log_heartbeat as fb_log_heartbeat,
    log_event as fb_log_event,
    log_daily_snapshot as fb_log_daily_snapshot,
    cleanup_old_events as fb_cleanup_events,
)

# ── Logging ────────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=getattr(logging, config.LOG_LEVEL, logging.INFO),
    format="[%(asctime)s] %(levelname)s %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("tradex.london_bot")

logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)


# ── Config London (depuis config.py / .env) ────────────────────────────────────

LON_TRADING_PAIRS: list[str] = config.LON_TRADING_PAIRS
LON_POLLING_SECONDS: int = config.LON_POLLING_SECONDS
LON_HEARTBEAT_SECONDS: int = config.LON_HEARTBEAT_SECONDS
LON_MAKER_WAIT_SECONDS: int = config.LON_MAKER_WAIT_SECONDS
LON_CAPITAL_PCT: float = config.LON_CAPITAL_PCT

# Stratégie
LON_SESSION_START: int = config.LON_SESSION_START_HOUR  # 8 UTC
LON_SESSION_END: int = config.LON_SESSION_END_HOUR      # 16 UTC
LON_SL_ATR_MULT: float = config.LON_SL_ATR_MULT        # 2.0
LON_TP1_PCT: float = config.LON_TP1_PCT                 # 0.02 (+2%)
LON_TP2_PCT: float = config.LON_TP2_PCT                 # 0.05 (+5%)
LON_TP1_SHARE: float = config.LON_TP1_SHARE             # 0.50
LON_VOL_MULT: float = config.LON_VOL_MULT               # 2.0
LON_MIN_RANGE: float = config.LON_MIN_RANGE_PCT          # 0.015 (1.5%)
LON_RISK_PCT: float = config.LON_RISK_PERCENT            # 0.05 (5%)
LON_MAX_POSITIONS: int = config.LON_MAX_POSITIONS         # 1
LON_COOLDOWN_BARS: int = config.LON_COOLDOWN_BARS         # 2
LON_ATR_PERIOD: int = config.LON_ATR_PERIOD               # 14
LON_VOL_MA_PERIOD: int = config.LON_VOL_MA_PERIOD         # 20
LON_BREAKEVEN_AFTER_TP1: bool = config.LON_BREAKEVEN_AFTER_TP1

H4_INTERVAL = 240  # H4 en minutes

MAX_MAKER_RETRIES = 2

# State file
LON_STATE_DIR: str = os.path.join(os.path.dirname(__file__), "..", "data")


def _fmt(price: float) -> str:
    """Formate un prix lisible."""
    if price >= 1000:
        return f"{price:,.4f}"
    elif price >= 1:
        return f"{price:.4f}"
    elif price >= 0.0001:
        return f"{price:.6f}"
    else:
        decimals = 6
        temp = price
        while temp < 0.01 and decimals < 10:
            temp *= 10
            decimals += 1
        return f"{price:.{decimals}f}"


# ─────────────────────────────────────────────────────────────────────
#  ATR / SMA helpers (pure functions, no I/O)
# ─────────────────────────────────────────────────────────────────────

def _atr_series(candles: list[Candle], period: int = 14) -> list[float]:
    """Calcule l'ATR sur une liste de bougies."""
    if len(candles) < 2:
        return [0.0] * len(candles)
    trs: list[float] = [candles[0].high - candles[0].low]
    for i in range(1, len(candles)):
        c = candles[i]
        prev_close = candles[i - 1].close
        tr = max(c.high - c.low, abs(c.high - prev_close), abs(c.low - prev_close))
        trs.append(tr)
    # EMA-like ATR (Wilder)
    atrs: list[float] = []
    for i, tr in enumerate(trs):
        if i < period:
            atrs.append(sum(trs[: i + 1]) / (i + 1))
        else:
            atrs.append((atrs[-1] * (period - 1) + tr) / period)
    return atrs


def _sma(values: list[float], period: int) -> list[float]:
    """Simple moving average."""
    result: list[float] = []
    for i in range(len(values)):
        if i < period - 1:
            result.append(sum(values[: i + 1]) / (i + 1))
        else:
            result.append(sum(values[i - period + 1: i + 1]) / period)
    return result


# ─────────────────────────────────────────────────────────────────────
#  Position data
# ─────────────────────────────────────────────────────────────────────

@dataclass
class LondonPosition:
    """Position gérée par le London Breakout bot."""
    symbol: str
    entry_price: float
    sl_price: float
    tp1_price: float
    tp2_price: float
    initial_size: float          # taille initiale (unités base)
    initial_size_usd: float
    remaining_size: float        # taille restante (après TP1)
    remaining_size_usd: float
    venue_order_id: str
    tp1_hit: bool = False
    breakeven_active: bool = False
    firebase_trade_id: Optional[str] = None
    opened_at: float = 0.0

    def to_dict(self) -> dict:
        return {
            "symbol": self.symbol,
            "entry_price": self.entry_price,
            "sl_price": self.sl_price,
            "tp1_price": self.tp1_price,
            "tp2_price": self.tp2_price,
            "initial_size": self.initial_size,
            "initial_size_usd": self.initial_size_usd,
            "remaining_size": self.remaining_size,
            "remaining_size_usd": self.remaining_size_usd,
            "venue_order_id": self.venue_order_id,
            "tp1_hit": self.tp1_hit,
            "breakeven_active": self.breakeven_active,
            "firebase_trade_id": self.firebase_trade_id,
            "opened_at": self.opened_at,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "LondonPosition":
        return cls(
            symbol=d["symbol"],
            entry_price=d["entry_price"],
            sl_price=d["sl_price"],
            tp1_price=d["tp1_price"],
            tp2_price=d["tp2_price"],
            initial_size=d["initial_size"],
            initial_size_usd=d["initial_size_usd"],
            remaining_size=d["remaining_size"],
            remaining_size_usd=d["remaining_size_usd"],
            venue_order_id=d.get("venue_order_id", "unknown"),
            tp1_hit=d.get("tp1_hit", False),
            breakeven_active=d.get("breakeven_active", False),
            firebase_trade_id=d.get("firebase_trade_id"),
            opened_at=d.get("opened_at", 0.0),
        )


# ─────────────────────────────────────────────────────────────────────
#  Per-pair session state
# ─────────────────────────────────────────────────────────────────────

@dataclass
class SessionState:
    """État de la session London pour une paire."""
    session_high: float = 0.0
    session_low: float = float("inf")
    session_bars: int = 0
    session_complete: bool = False
    breakout_consumed: bool = False   # un seul breakout par session
    cooldown_until: int = 0           # bar_ts until cooldown expires


# ─────────────────────────────────────────────────────────────────────
#  State store (JSON persistence)
# ─────────────────────────────────────────────────────────────────────

class LondonStateStore:
    """Persistance atomique du bot London Breakout."""

    def __init__(self, state_file: str) -> None:
        self._path = Path(state_file).resolve()

    def save(
        self,
        positions: dict[str, LondonPosition],
        session_states: dict[str, SessionState],
        last_candle_ts: dict[str, int],
        candle_buffers: dict[str, list[dict]],
    ) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        state = {
            "positions": {s: p.to_dict() for s, p in positions.items()},
            "session_states": {
                s: {
                    "session_high": ss.session_high,
                    "session_low": ss.session_low,
                    "session_bars": ss.session_bars,
                    "session_complete": ss.session_complete,
                    "breakout_consumed": ss.breakout_consumed,
                    "cooldown_until": ss.cooldown_until,
                }
                for s, ss in session_states.items()
            },
            "last_candle_ts": last_candle_ts,
            "candle_buffers": candle_buffers,
        }
        tmp = self._path.with_suffix(".json.tmp")
        try:
            with open(tmp, "w") as f:
                json.dump(state, f, indent=2)
            tmp.replace(self._path)
        except Exception as e:
            logger.error("❌ Save failed: %s", e)
            if tmp.exists():
                try:
                    tmp.unlink()
                except OSError:
                    pass

    def load(self) -> dict:
        if not self._path.exists():
            logger.info("📂 Pas de state London — démarrage à vide")
            return {}
        try:
            with open(self._path, "r") as f:
                return json.load(f)
        except Exception as e:
            logger.error("❌ Load state London échoué: %s", e)
            return {}


# ─────────────────────────────────────────────────────────────────────
#  Bot principal
# ─────────────────────────────────────────────────────────────────────

class LondonBreakoutBot:
    """Bot London Breakout — Revolut X, maker-only, H4, LONG only."""

    def __init__(self, dry_run: bool = False) -> None:
        self.dry_run = dry_run
        self._running = False

        # Services
        self._client = RevolutXClient(
            api_key=config.REVOLUT_X_API_KEY,
            private_key_path=config.REVOLUT_X_PRIVATE_KEY_PATH,
        )
        self._data = DataProvider(self._client)
        self._telegram = TelegramNotifier(
            bot_token=config.TELEGRAM_BOT_TOKEN,
            chat_id=config.TELEGRAM_CHAT_ID,
        )

        # State
        state_file = os.path.join(LON_STATE_DIR, "state_london.json")
        self._store = LondonStateStore(state_file)
        self._positions: dict[str, LondonPosition] = {}
        self._session_states: dict[str, SessionState] = {
            s: SessionState() for s in LON_TRADING_PAIRS
        }

        # Candle tracking
        self._last_candle_ts: dict[str, int] = {}
        # Keep recent candles for ATR / volume MA computation
        self._candle_buffers: dict[str, list[dict]] = {s: [] for s in LON_TRADING_PAIRS}

        # Heartbeat
        self._last_heartbeat: float = 0.0
        self._tick_count: int = 0

        # Daily cleanup / snapshot
        self._last_cleanup_date: str = ""
        self._last_snapshot_date: str = ""

        # Close failure tracking
        self._close_failures: dict[str, dict] = {}

        if dry_run:
            logger.info("🔧 Mode DRY-RUN — aucun ordre ne sera exécuté")

    # ── Run ────────────────────────────────────────────────────────────────

    def run(self) -> None:
        self._running = True

        logger.info("═" * 60)
        logger.info("🇬🇧 LondonBreakoutBot démarré — Session Breakout H4")
        logger.info("   Paires    : %d | %s", len(LON_TRADING_PAIRS), ", ".join(LON_TRADING_PAIRS))
        logger.info("   Session   : %02d:00-%02d:00 UTC", LON_SESSION_START, LON_SESSION_END)
        logger.info("   Capital   : %.0f%% du solde Revolut X", LON_CAPITAL_PCT * 100)
        logger.info("   Risque    : %.0f%% par trade | Max %d position(s)",
                     LON_RISK_PCT * 100, LON_MAX_POSITIONS)
        logger.info("   SL: %.1f×ATR | TP1: +%.0f%% (50%%) → BE | TP2: +%.0f%%",
                     LON_SL_ATR_MULT, LON_TP1_PCT * 100, LON_TP2_PCT * 100)
        logger.info("   Vol: ≥%.1f×MA%d | Range min: %.1f%%",
                     LON_VOL_MULT, LON_VOL_MA_PERIOD, LON_MIN_RANGE * 100)
        logger.info("   Polling   : %ds | Maker wait: %ds",
                     LON_POLLING_SECONDS, LON_MAKER_WAIT_SECONDS)
        logger.info("   Mode      : LONG-ONLY")
        if self.dry_run:
            logger.info("   ⚠️  DRY-RUN actif")
        logger.info("═" * 60)

        self._initialize()

        try:
            while self._running:
                self._tick()
                time.sleep(LON_POLLING_SECONDS)
        except KeyboardInterrupt:
            logger.info("Arrêt demandé par l'utilisateur")
        finally:
            self._shutdown()

    def stop(self) -> None:
        self._running = False

    # ── Init ───────────────────────────────────────────────────────────────

    def _initialize(self) -> None:
        """Charge l'état persisté + bougies initiales."""
        state = self._store.load()
        if state:
            # Positions
            for sym, d in state.get("positions", {}).items():
                try:
                    self._positions[sym] = LondonPosition.from_dict(d)
                except Exception as e:
                    logger.warning("⚠️ Position %s corrompue: %s", sym, e)

            # Session states
            for sym, d in state.get("session_states", {}).items():
                try:
                    ss = SessionState()
                    ss.session_high = d.get("session_high", 0.0)
                    ss.session_low = d.get("session_low", float("inf"))
                    ss.session_bars = d.get("session_bars", 0)
                    ss.session_complete = d.get("session_complete", False)
                    ss.breakout_consumed = d.get("breakout_consumed", False)
                    ss.cooldown_until = d.get("cooldown_until", 0)
                    self._session_states[sym] = ss
                except Exception as e:
                    logger.warning("⚠️ Session state %s corrompu: %s", sym, e)

            # Last candle timestamps
            self._last_candle_ts = state.get("last_candle_ts", {})

            # Candle buffers
            self._candle_buffers = state.get("candle_buffers", {s: [] for s in LON_TRADING_PAIRS})

            logger.info(
                "📂 State London chargé: %d positions, %d session states",
                len(self._positions), len(self._session_states),
            )

        # Réconcilier les positions
        self._reconcile_positions()

        # Charger les bougies H4 initiales
        logger.info("── Chargement des bougies H4 initiales ──")
        for symbol in LON_TRADING_PAIRS:
            try:
                candles = self._client.get_candles(symbol, interval=H4_INTERVAL)
                candles.sort(key=lambda c: c.timestamp)
                if candles:
                    self._last_candle_ts[symbol] = candles[-1].timestamp
                    # Store candle buffer as dicts for JSON serialization
                    self._candle_buffers[symbol] = [
                        {"timestamp": c.timestamp, "open": c.open, "high": c.high,
                         "low": c.low, "close": c.close, "volume": c.volume}
                        for c in candles[-100:]  # garder 100 dernières
                    ]
                    logger.info("[%s] %d bougies H4 chargées", symbol, len(candles))
                else:
                    logger.warning("[%s] Aucune bougie H4 reçue", symbol)
            except Exception as e:
                logger.error("[%s] ❌ Erreur chargement bougies: %s", symbol, e)

        self._save_state()

        open_count = sum(1 for p in self._positions.values())
        logger.info("── Init terminée | %d positions ouvertes ──", open_count)

    def _reconcile_positions(self) -> None:
        """Vérifie les positions contre les soldes exchange."""
        if not self._positions:
            return

        try:
            balances = self._client.get_balances()
        except Exception as e:
            logger.warning("⚠️ Réconciliation impossible: %s", e)
            return

        balance_map = {b.currency: b for b in balances}
        removed = []

        for sym, pos in self._positions.items():
            base = sym.split("-")[0]
            bal = balance_map.get(base)
            held = (bal.available + bal.reserved) if bal else 0.0

            if held >= pos.remaining_size * 0.90:
                logger.info("[%s] ✅ Position confirmée | %.8f %s", sym, held, base)
            else:
                logger.warning(
                    "[%s] ⚠️ Position locale mais solde %s=%.8f < size=%.8f → retirée",
                    sym, base, held, pos.remaining_size,
                )
                removed.append(sym)

        for sym in removed:
            del self._positions[sym]

        if removed:
            self._save_state()

    # ── Tick ───────────────────────────────────────────────────────────────

    def _tick(self) -> None:
        """Un cycle de polling — itère sur toutes les paires."""
        self._tick_count += 1

        for symbol in LON_TRADING_PAIRS:
            try:
                self._process_symbol(symbol)
            except Exception as e:
                logger.error("[%s] Erreur tick: %s", symbol, e)

        self._maybe_heartbeat()
        self._maybe_daily_tasks()

    def _process_symbol(self, symbol: str) -> None:
        """Traite un symbole : check prix (SL/TP) + check nouvelles bougies."""

        # ── 1. Gestion des positions ouvertes via ticker ──
        pos = self._positions.get(symbol)
        if pos:
            ticker = self._data.get_ticker(symbol)
            if ticker:
                self._manage_position(symbol, pos, ticker)
                if symbol not in self._positions:
                    return  # Position vient de fermer

        # ── 2. Vérifier s'il y a une nouvelle bougie H4 ──
        try:
            candles = self._client.get_candles(symbol, interval=H4_INTERVAL)
            candles.sort(key=lambda c: c.timestamp)
        except Exception as e:
            logger.debug("[%s] Candle fetch failed: %s", symbol, e)
            return

        if not candles:
            return

        latest_ts = candles[-1].timestamp
        prev_ts = self._last_candle_ts.get(symbol, 0)

        if latest_ts <= prev_ts:
            return  # Pas de nouvelle bougie

        self._last_candle_ts[symbol] = latest_ts

        # Mettre à jour le buffer de bougies
        self._candle_buffers[symbol] = [
            {"timestamp": c.timestamp, "open": c.open, "high": c.high,
             "low": c.low, "close": c.close, "volume": c.volume}
            for c in candles[-100:]
        ]

        # ── Nouvelle bougie H4 → traiter session + breakout ──
        latest = candles[-1]
        self._process_new_candle(symbol, latest, candles)
        self._save_state()

    # ── Position management ────────────────────────────────────────────────

    def _manage_position(self, symbol: str, pos: LondonPosition, ticker: TickerData) -> None:
        """Gère SL, TP1, TP2 pour une position ouverte."""
        price = ticker.last_price

        # ── SL check ──
        if price <= pos.sl_price:
            reason = "BREAKEVEN" if pos.breakeven_active and pos.sl_price >= pos.entry_price else "SL"
            logger.info("[%s] 🛑 %s HIT | prix=%s | SL=%s", symbol, reason, _fmt(price), _fmt(pos.sl_price))
            self._close_position(symbol, price, f"{reason} atteint", partial=False)
            return

        # ── TP1 check (50% partial close) ──
        if not pos.tp1_hit and price >= pos.tp1_price:
            logger.info("[%s] 🎯 TP1 HIT | prix=%s | TP1=%s", symbol, _fmt(price), _fmt(pos.tp1_price))
            self._close_tp1(symbol, price)
            return

        # ── TP2 check (close remaining) ──
        if pos.tp1_hit and price >= pos.tp2_price:
            logger.info("[%s] 🎯 TP2 HIT | prix=%s | TP2=%s", symbol, _fmt(price), _fmt(pos.tp2_price))
            self._close_position(symbol, price, "TP2 atteint", partial=False)
            return

    def _close_tp1(self, symbol: str, price: float) -> None:
        """Ferme 50% de la position au TP1 et active breakeven."""
        pos = self._positions.get(symbol)
        if not pos:
            return

        tp1_size = pos.initial_size * LON_TP1_SHARE
        tp1_size_usd = pos.initial_size_usd * LON_TP1_SHARE

        # Exécuter la vente partielle
        success = self._execute_sell(symbol, price, tp1_size)
        if not success:
            return

        # PnL TP1
        pnl_usd = (price - pos.entry_price) * tp1_size
        pnl_pct = (price - pos.entry_price) / pos.entry_price * 100

        logger.info(
            "[%s] 🇬🇧 TP1 | prix=%s | size=%.8f ($%.2f) | PnL=$%+.2f (%+.1f%%)",
            symbol, _fmt(price), tp1_size, tp1_size_usd, pnl_usd, pnl_pct,
        )

        # Update position
        pos.tp1_hit = True
        pos.remaining_size -= tp1_size
        pos.remaining_size_usd -= tp1_size_usd

        # Breakeven
        if LON_BREAKEVEN_AFTER_TP1:
            pos.sl_price = pos.entry_price
            pos.breakeven_active = True
            logger.info("[%s] 🔒 BREAKEVEN activé après TP1 | nouveau SL=%s", symbol, _fmt(pos.entry_price))

        self._save_state()

        # Telegram
        base = symbol.split("-")[0]
        self._telegram._send(
            f"🎯 *TP1 London – {symbol}* 🇬🇧\n"
            f"  Prix: `{_fmt(price)}` | Size: `{tp1_size:.8f} {base}` (`${tp1_size_usd:.2f}`)\n"
            f"  P&L partiel: `{pnl_usd:+.2f} USD` (`{pnl_pct:+.1f}%`)\n"
            f"  🔒 Breakeven activé | Restant: `{pos.remaining_size:.8f} {base}`\n"
            f"[Dashboard]({DASHBOARD_URL})"
        )

        # Firebase : log TP1 partial close
        try:
            fb_log_event(
                event_type="london_tp1",
                data={
                    "symbol": symbol,
                    "price": price,
                    "size": tp1_size,
                    "pnl_usd": pnl_usd,
                    "pnl_pct": pnl_pct,
                },
                symbol=symbol,
            )
        except Exception:
            pass

    def _close_position(self, symbol: str, exit_price: float, reason: str, partial: bool = False) -> None:
        """Ferme une position (totale)."""
        pos = self._positions.get(symbol)
        if not pos:
            return

        # Backoff check
        fail_info = self._close_failures.get(symbol)
        if fail_info:
            if fail_info.get("permanent"):
                return
            if time.time() < fail_info.get("next_retry", 0):
                return

        exit_size = pos.remaining_size

        # Vérifier le solde réel
        if not self.dry_run:
            exit_size = self._check_real_balance(symbol, pos.remaining_size)
            if exit_size <= 0:
                del self._positions[symbol]
                self._save_state()
                return

        # Exécuter la vente
        success = self._execute_sell(symbol, exit_price, exit_size, use_taker=(reason.startswith("SL") or reason.startswith("BREAK")))
        if not success:
            self._handle_close_failure(symbol, "sell failed")
            return

        # Clear failure tracking
        if symbol in self._close_failures:
            del self._close_failures[symbol]

        # PnL
        pnl_gross = (exit_price - pos.entry_price) * exit_size
        pnl_pct = (exit_price - pos.entry_price) / pos.entry_price * 100 if pos.entry_price > 0 else 0
        pnl_emoji = "🟢" if pnl_gross >= 0 else "🔴"

        logger.info(
            "[%s] %s CLOSE | entry=%s | exit=%s | PnL=$%+.2f (%+.1f%%) | %s",
            symbol, pnl_emoji, _fmt(pos.entry_price), _fmt(exit_price),
            pnl_gross, pnl_pct, reason,
        )

        # Telegram
        base = symbol.split("-")[0]
        self._telegram._send(
            f"{pnl_emoji} *Position fermée – {symbol}* 🇬🇧 LONDON\n"
            f"  Raison: {reason}\n"
            f"  Entrée: `{_fmt(pos.entry_price)}` → Sortie: `{_fmt(exit_price)}`\n"
            f"  P&L: `{pnl_gross:+.2f} USD` (`{pnl_pct:+.1f}%`)\n"
            f"[Dashboard]({DASHBOARD_URL})"
        )

        # Firebase
        if pos.firebase_trade_id:
            try:
                fb_position = Position(
                    symbol=symbol,
                    side=OrderSide.BUY,
                    entry_price=pos.entry_price,
                    sl_price=pos.sl_price,
                    size=exit_size,
                    venue_order_id=pos.venue_order_id,
                    status=PositionStatus.CLOSED,
                    strategy=StrategyType.LONDON,
                    tp_price=pos.tp2_price,
                    pnl=pnl_gross,
                )
                log_trade_closed(
                    trade_id=pos.firebase_trade_id,
                    position=fb_position,
                    exit_price=exit_price,
                    reason=reason,
                    fill_type="taker" if reason.startswith("SL") else "maker",
                    equity_after=0.0,
                    actual_exit_size=exit_size,
                )
            except Exception as e:
                logger.warning("🔥 Firebase log_trade_closed échoué: %s", e)

        del self._positions[symbol]
        self._save_state()

    # ── Session tracking + Breakout detection ──────────────────────────────

    def _process_new_candle(self, symbol: str, candle: Candle, all_candles: list[Candle]) -> None:
        """Traite une nouvelle bougie H4 : mise à jour session + détection breakout."""
        ss = self._session_states.get(symbol)
        if ss is None:
            ss = SessionState()
            self._session_states[symbol] = ss

        hour = datetime.fromtimestamp(candle.timestamp / 1000, tz=timezone.utc).hour

        # H4 bars: 00:00, 04:00, 08:00, 12:00, 16:00, 20:00
        # Session London: 08:00 and 12:00 are IN session
        in_session = LON_SESSION_START <= hour < LON_SESSION_END

        if in_session:
            if ss.session_bars == 0:
                # Nouvelle session → reset
                ss.session_high = candle.high
                ss.session_low = candle.low
                ss.breakout_consumed = False
                ss.session_complete = False
            else:
                ss.session_high = max(ss.session_high, candle.high)
                ss.session_low = min(ss.session_low, candle.low)
            ss.session_bars += 1

            logger.debug(
                "[%s] 🕐 Session bar %d | H=%s | L=%s",
                symbol, ss.session_bars, _fmt(ss.session_high), _fmt(ss.session_low),
            )
        else:
            # Hors session
            if ss.session_bars > 0 and not ss.session_complete:
                ss.session_complete = True
                range_pct = (ss.session_high - ss.session_low) / ss.session_low * 100 if ss.session_low > 0 else 0
                logger.info(
                    "[%s] 🇬🇧 Session terminée | H=%s | L=%s | Range=%.2f%%",
                    symbol, _fmt(ss.session_high), _fmt(ss.session_low), range_pct,
                )

            # Reset session_bars when a new day's session will start
            # (Bougies 00:00 et 04:00 → reset pour la prochaine session de 08:00)
            if hour < LON_SESSION_START:
                ss.session_bars = 0

            # Check breakout conditions
            if ss.session_complete and not ss.breakout_consumed:
                self._check_breakout(symbol, candle, all_candles, ss)

    def _check_breakout(
        self,
        symbol: str,
        candle: Candle,
        all_candles: list[Candle],
        ss: SessionState,
    ) -> None:
        """Vérifie les conditions de breakout LONG au-dessus du session high."""

        # Guard: déjà une position sur ce symbole
        if symbol in self._positions:
            return

        # Guard: max positions
        n_open = len(self._positions)
        if n_open >= LON_MAX_POSITIONS:
            return

        # Guard: cooldown
        if candle.timestamp <= ss.cooldown_until:
            return

        # ── Range minimum (1.5%) ──
        if ss.session_low <= 0:
            return
        range_pct = (ss.session_high - ss.session_low) / ss.session_low
        if range_pct < LON_MIN_RANGE:
            logger.debug("[%s] Range %.2f%% < min %.1f%% — skip", symbol, range_pct * 100, LON_MIN_RANGE * 100)
            return

        # ── Breakout: close > session_high ──
        if candle.close <= ss.session_high:
            return

        # ── Volume filter ──
        candles_for_vol = [
            Candle(
                timestamp=d["timestamp"], open=d["open"], high=d["high"],
                low=d["low"], close=d["close"], volume=d["volume"],
            )
            for d in self._candle_buffers.get(symbol, [])
        ]
        if len(candles_for_vol) < LON_VOL_MA_PERIOD + 1:
            candles_for_vol = all_candles

        volumes = [c.volume for c in candles_for_vol]
        vol_ma = _sma(volumes, LON_VOL_MA_PERIOD)
        current_vol = candle.volume
        current_vol_ma = vol_ma[-1] if vol_ma else 0

        if LON_VOL_MULT > 0 and current_vol_ma > 0:
            if current_vol < LON_VOL_MULT * current_vol_ma:
                logger.debug(
                    "[%s] Vol %.0f < %.1f × MA %.0f — skip",
                    symbol, current_vol, LON_VOL_MULT, current_vol_ma,
                )
                return

        # ── ATR pour SL ──
        if len(candles_for_vol) < LON_ATR_PERIOD + 1:
            candles_for_vol = all_candles
        atr_vals = _atr_series(candles_for_vol, LON_ATR_PERIOD)
        atr_val = atr_vals[-1] if atr_vals else 0

        if atr_val <= 0:
            logger.warning("[%s] ATR=0, skip breakout", symbol)
            return

        # ── Toutes les conditions remplies → ENTRY ──
        entry_price = candle.close
        sl_price = entry_price - LON_SL_ATR_MULT * atr_val
        tp1_price = entry_price * (1 + LON_TP1_PCT)
        tp2_price = entry_price * (1 + LON_TP2_PCT)

        sl_dist = entry_price - sl_price
        if sl_dist <= 0:
            return

        logger.info(
            "[%s] 🇬🇧 BREAKOUT SIGNAL | close=%s > high=%s | range=%.2f%% | vol=%.0f (%.1fx MA)",
            symbol, _fmt(entry_price), _fmt(ss.session_high),
            range_pct * 100, current_vol,
            current_vol / current_vol_ma if current_vol_ma > 0 else 0,
        )

        # Mark breakout consumed + cooldown
        ss.breakout_consumed = True
        ss.cooldown_until = candle.timestamp + LON_COOLDOWN_BARS * 4 * 3600 * 1000  # H4 bars in ms

        # Execute entry
        self._execute_entry(symbol, entry_price, sl_price, tp1_price, tp2_price, atr_val, range_pct)

    # ── Entry execution ────────────────────────────────────────────────────

    def _execute_entry(
        self,
        symbol: str,
        entry_price: float,
        sl_price: float,
        tp1_price: float,
        tp2_price: float,
        atr_val: float,
        range_pct: float,
    ) -> None:
        """Exécute un signal d'entrée London Breakout."""

        # ── Sizing ──
        try:
            balances = self._data.get_balances()
            quote = symbol.split("-")[1]
            usd_bal = next((b for b in balances if b.currency == quote), None)
            available = usd_bal.available if usd_bal else 0.0
        except Exception as e:
            logger.error("[%s] ❌ Impossible de récupérer le solde: %s", symbol, e)
            return

        if available <= 0:
            logger.info("[%s] ⏭️ Pas de USD disponible — skip", symbol)
            return

        # Capital alloué = solde × capital_pct
        allocated = available * LON_CAPITAL_PCT
        risk_amount = allocated * LON_RISK_PCT
        sl_dist = entry_price - sl_price

        if sl_dist <= 0:
            return

        size = risk_amount / sl_dist
        size_usd = size * entry_price

        # Cap au capital alloué
        if size_usd > allocated:
            size_usd = allocated
            size = size_usd / entry_price

        if size_usd < 5:
            logger.info("[%s] ⏭️ Taille trop faible ($%.2f) — skip", symbol, size_usd)
            return

        # ── Format order ──
        price_str = self._format_order_price(entry_price)
        size_str = f"{size:.8f}"

        order = OrderRequest(
            symbol=symbol,
            side=OrderSide.BUY,
            base_size=size_str,
            price=price_str,
        )

        # ── Exécution maker 2 retry → taker fallback ──
        venue_order_id = "dry-run"
        fill_type = "dry-run"
        actual_price = entry_price

        if not self.dry_run:
            current_price = entry_price
            filled = False

            for attempt in range(1, MAX_MAKER_RETRIES + 1):
                p_str = self._format_order_price(current_price)
                s_str = f"{size:.8f}"
                o = OrderRequest(symbol=symbol, side=OrderSide.BUY, base_size=s_str, price=p_str)

                try:
                    result = self._place_maker_only_order(o)
                    ft = result.get("fill_type", "unknown")

                    if ft != "no_fill":
                        venue_order_id = result.get("venue_order_id", "unknown")
                        fill_type = ft
                        actual_price = result.get("actual_price", current_price)
                        filled = True
                        logger.info(
                            "[%s] 🇬🇧 ✅ BUY exécuté | @ %s | size=%s | maker (tentative %d)",
                            symbol, p_str, s_str, attempt,
                        )
                        break

                    # Maker no-fill → rafraîchir le prix
                    logger.info(
                        "[%s] 🇬🇧 BUY: maker no-fill (tentative %d/%d)",
                        symbol, attempt, MAX_MAKER_RETRIES,
                    )
                    try:
                        ticker = self._data.get_ticker(symbol)
                        if ticker:
                            current_price = ticker.last_price
                    except Exception:
                        pass

                except Exception as e:
                    logger.error("[%s] 🇬🇧 ❌ BUY maker échoué (tentative %d): %s", symbol, attempt, e)
                    break

            if not filled:
                # Taker fallback
                logger.warning(
                    "[%s] 🇬🇧 BUY: %d makers no-fill → TAKER FALLBACK @ %s",
                    symbol, MAX_MAKER_RETRIES, _fmt(current_price),
                )
                p_str = self._format_order_price(current_price)
                s_str = f"{size:.8f}"
                o = OrderRequest(symbol=symbol, side=OrderSide.BUY, base_size=s_str, price=p_str)

                try:
                    result = self._place_taker_order(o)
                    ft = result.get("fill_type", "unknown")
                    if ft == "no_fill":
                        logger.error("[%s] 🇬🇧 BUY: taker fallback échoué aussi — abandonné", symbol)
                        try:
                            fb_log_event(
                                event_type="london_maker_no_fill",
                                data={"price": current_price, "retries": MAX_MAKER_RETRIES},
                                symbol=symbol,
                            )
                        except Exception:
                            pass
                        return
                    venue_order_id = result.get("venue_order_id", "unknown")
                    fill_type = "taker"
                    actual_price = result.get("actual_price", current_price)
                    filled = True
                    logger.info(
                        "[%s] 🇬🇧 ✅ BUY exécuté | @ %s | size=%s | taker fallback",
                        symbol, p_str, s_str,
                    )
                except Exception as e:
                    logger.error("[%s] 🇬🇧 ❌ BUY taker échoué: %s", symbol, e)
                    self._telegram.notify_error(f"🇬🇧 BUY {symbol} échoué: {e}")
                    return

            if not filled:
                return
        else:
            logger.info(
                "[DRY-RUN] LONDON BUY %s | entry=%s | SL=%s | TP1=%s | TP2=%s | size=%s",
                symbol, _fmt(entry_price), _fmt(sl_price),
                _fmt(tp1_price), _fmt(tp2_price), size_str,
            )

        # ── Enregistrer la position ──
        pos = LondonPosition(
            symbol=symbol,
            entry_price=actual_price,
            sl_price=sl_price,
            tp1_price=tp1_price,
            tp2_price=tp2_price,
            initial_size=size,
            initial_size_usd=size_usd,
            remaining_size=size,
            remaining_size_usd=size_usd,
            venue_order_id=venue_order_id,
            opened_at=time.time(),
        )
        self._positions[symbol] = pos
        self._save_state()

        # ── Telegram ──
        base = symbol.split("-")[0]
        sl_pct = abs(entry_price - sl_price) / entry_price * 100
        self._telegram._send(
            f"🇬🇧 *BUY déclenché – {symbol}* LONDON BREAKOUT\n"
            f"  Entrée: `{_fmt(actual_price)}` | SL: `{_fmt(sl_price)}` ({sl_pct:.1f}%)\n"
            f"  TP1: `{_fmt(tp1_price)}` (+{LON_TP1_PCT*100:.0f}%) | TP2: `{_fmt(tp2_price)}` (+{LON_TP2_PCT*100:.0f}%)\n"
            f"  Size: `{size:.8f} {base}` (`${size_usd:.2f}`)\n"
            f"  Risque: {LON_RISK_PCT*100:.0f}% (`${risk_amount:.2f}`) | ATR: `{_fmt(atr_val)}`\n"
            f"  Range: `{range_pct*100:.2f}%` | Fill: {fill_type}\n"
            f"[Dashboard]({DASHBOARD_URL})"
        )

        # ── Firebase ──
        try:
            fb_position = Position(
                symbol=symbol,
                side=OrderSide.BUY,
                entry_price=actual_price,
                sl_price=sl_price,
                size=size,
                venue_order_id=venue_order_id,
                status=PositionStatus.OPEN,
                strategy=StrategyType.LONDON,
                tp_price=tp2_price,
            )
            fb_id = log_trade_opened(
                position=fb_position,
                fill_type=fill_type,
                maker_wait_seconds=LON_MAKER_WAIT_SECONDS,
                risk_pct=LON_RISK_PCT,
                risk_amount_usd=risk_amount,
                fiat_balance=available,
                current_equity=available,
                portfolio_risk_before=0.0,
                exchange="revolut-london",
            )
            if fb_id:
                pos.firebase_trade_id = fb_id
                self._save_state()
        except Exception as e:
            logger.warning("🔥 Firebase log_trade_opened échoué: %s", e)

    # ── Order execution ────────────────────────────────────────────────────

    def _execute_sell(self, symbol: str, price: float, size: float, use_taker: bool = False) -> bool:
        """Exécute une vente via Revolut X."""
        if self.dry_run:
            logger.info("[DRY-RUN] LONDON SELL %s | prix=%s | size=%.8f", symbol, _fmt(price), size)
            return True

        # Vérifier le solde réel
        actual_size = self._check_real_balance(symbol, size)
        if actual_size <= 0:
            return False

        price_str = self._format_order_price(price)
        size_str = f"{actual_size:.8f}"

        order = OrderRequest(
            symbol=symbol,
            side=OrderSide.SELL,
            base_size=size_str,
            price=price_str,
        )

        try:
            if use_taker:
                result = self._place_taker_order(order)
            else:
                result = self._place_maker_only_order(order)
                if result.get("fill_type") == "no_fill":
                    logger.warning("[%s] 🇬🇧 Maker no-fill → TAKER FALLBACK", symbol)
                    result = self._place_taker_order(order)

            logger.info(
                "[%s] 🇬🇧 ✅ SELL exécuté | @ %s | size=%s | %s",
                symbol, price_str, size_str, result.get("fill_type", "?"),
            )
            return True
        except Exception as e:
            logger.error("[%s] 🇬🇧 ❌ SELL échoué: %s", symbol, e)
            self._telegram.notify_error(f"🇬🇧 SELL {symbol} échoué: {e}")
            return False

    def _check_real_balance(self, symbol: str, expected_size: float) -> float:
        """Vérifie le solde réel et retourne la taille disponible."""
        try:
            base_currency = symbol.split("-")[0]
            balances = self._data.get_balances()
            base_bal = next((b for b in balances if b.currency == base_currency), None)
            real_available = base_bal.available if base_bal else 0.0
            real_total = (base_bal.available + base_bal.reserved) if base_bal else 0.0

            if real_total <= 0:
                logger.warning("[%s] 👻 Position fantôme — solde = 0 → purge", symbol)
                return 0.0

            # Annuler les ordres actifs si besoin
            if real_available < expected_size * 0.90:
                try:
                    active_orders = self._client.get_active_orders([symbol])
                    for ao in active_orders:
                        ao_id = ao.get("venue_order_id") or ao.get("id")
                        if ao_id:
                            try:
                                self._client.cancel_order(ao_id)
                                time.sleep(0.5)
                            except Exception:
                                pass
                except Exception:
                    pass
                # Re-fetch
                balances = self._data.get_balances()
                base_bal = next((b for b in balances if b.currency == base_currency), None)
                real_available = base_bal.available if base_bal else 0.0

            if real_available < expected_size:
                logger.info("[%s] 📐 Taille ajustée: %.8f → %.8f", symbol, expected_size, real_available)
                return real_available

            return expected_size
        except Exception as e:
            logger.warning("[%s] ⚠️ Balance check échoué: %s", symbol, e)
            return expected_size

    def _place_maker_only_order(self, order: OrderRequest) -> dict:
        """Place un ordre limit passif (maker 0%). Si pas rempli → annule."""
        logger.info(
            "💰 MAKER-ONLY | %s %s @ %s (attente %ds)",
            order.side.value.upper(), order.symbol, order.price, LON_MAKER_WAIT_SECONDS,
        )

        try:
            resp = self._client.place_order(order)
        except Exception as e:
            logger.error("💰 MAKER-ONLY | Placement échoué: %s", e)
            raise

        data = resp.get("data", {})
        if isinstance(data, dict):
            venue_order_id = data.get("venue_order_id", "unknown")
        elif isinstance(data, list) and data:
            venue_order_id = data[0].get("venue_order_id", "unknown")
        else:
            venue_order_id = "unknown"

        if venue_order_id == "unknown":
            return {"venue_order_id": venue_order_id, "fill_type": "maker", "actual_price": float(order.price)}

        # Check fill instantané
        initial_state = ""
        if isinstance(data, dict):
            initial_state = (data.get("state") or data.get("status") or "").upper()
        if initial_state == "FILLED":
            logger.info("💰 MAKER-ONLY | ✅ Fill instantané")
            return {"venue_order_id": venue_order_id, "fill_type": "maker", "actual_price": float(order.price)}

        # Attendre
        logger.info("💰 MAKER-ONLY | ⏳ Attente %ds…", LON_MAKER_WAIT_SECONDS)
        time.sleep(LON_MAKER_WAIT_SECONDS)

        # Vérifier
        try:
            order_status = self._client.get_order(venue_order_id)
        except Exception as e:
            logger.warning("💰 MAKER-ONLY | Status check failed: %s → assume filled", e)
            return {"venue_order_id": venue_order_id, "fill_type": "maker", "actual_price": float(order.price)}

        order_data_raw = order_status.get("data", order_status) if isinstance(order_status, dict) else order_status
        if isinstance(order_data_raw, list) and order_data_raw:
            order_data_raw = order_data_raw[0]
        od: dict = order_data_raw if isinstance(order_data_raw, dict) else {}
        status = (od.get("status") or od.get("state") or "").upper()
        filled_size = float(od.get("filled_size", "0") or "0")
        total_size = float(order.base_size)

        if status == "FILLED" or (filled_size > 0 and filled_size >= total_size * 0.99):
            logger.info("💰 MAKER-ONLY | ✅ Rempli — fee 0%%")
            return {"venue_order_id": venue_order_id, "fill_type": "maker", "actual_price": float(order.price)}

        if filled_size > 0:
            logger.info("💰 MAKER-ONLY | ⚡ Fill partiel (%.8f/%.8f)", filled_size, total_size)
            try:
                self._client.cancel_order(venue_order_id)
            except Exception:
                pass
            return {
                "venue_order_id": venue_order_id,
                "fill_type": "partial_maker",
                "actual_price": float(order.price),
                "filled_size": filled_size,
            }

        # Pas rempli → annuler
        logger.info("💰 MAKER-ONLY | ❌ Pas de fill → annulation")
        try:
            self._client.cancel_order(venue_order_id)
        except Exception as e:
            if "409" in str(e) or "conflict" in str(e).lower():
                try:
                    recheck = self._client.get_order(venue_order_id)
                    rd_raw = recheck.get("data", recheck) if isinstance(recheck, dict) else recheck
                    if isinstance(rd_raw, list) and rd_raw:
                        rd_raw = rd_raw[0]
                    rd: dict = rd_raw if isinstance(rd_raw, dict) else {}
                    if (rd.get("status") or rd.get("state") or "").upper() == "FILLED":
                        logger.info("💰 MAKER-ONLY | ✅ Rempli entre-temps!")
                        return {"venue_order_id": venue_order_id, "fill_type": "maker",
                                "actual_price": float(order.price)}
                except Exception:
                    pass

        return {"venue_order_id": venue_order_id, "fill_type": "no_fill", "actual_price": float(order.price)}

    def _place_taker_order(self, order: OrderRequest) -> dict:
        """Place un ordre taker agressif (pour SL / close)."""
        try:
            tickers = self._client.get_tickers(symbols=[order.symbol])
            if tickers:
                t = tickers[0]
                taker_price = t.bid if order.side == OrderSide.SELL else t.ask
            else:
                taker_price = float(order.price)
        except Exception:
            taker_price = float(order.price)

        taker_order = OrderRequest(
            symbol=order.symbol,
            side=order.side,
            base_size=order.base_size,
            price=self._format_order_price(taker_price),
        )

        logger.info(
            "💰 TAKER FALLBACK | %s %s @ %s (fee 0.09%%)",
            order.side.value.upper(), order.symbol, taker_order.price,
        )

        resp = self._client.place_order(taker_order)
        data = resp.get("data", {})
        if isinstance(data, dict):
            vid = data.get("venue_order_id", "unknown")
        elif isinstance(data, list) and data:
            vid = data[0].get("venue_order_id", "unknown")
        else:
            vid = "unknown"

        return {"venue_order_id": vid, "fill_type": "taker", "actual_price": taker_price}

    def _handle_close_failure(self, symbol: str, error: str) -> None:
        """Gère les échecs de clôture avec backoff."""
        fail_info = self._close_failures.get(symbol, {"count": 0})
        fail_info["count"] += 1
        fail_info["last_error"] = error

        PERMANENT_ERRORS = ["INACTIVE", "DELISTED", "SUSPENDED", "not supported", "No CURRENT pocket"]
        if any(kw.lower() in error.lower() for kw in PERMANENT_ERRORS):
            fail_info["permanent"] = True
            fail_info["next_retry"] = float("inf")
            self._close_failures[symbol] = fail_info
            logger.critical("[%s] 🚫 Erreur permanente: %s", symbol, error[:200])
            self._telegram.notify_error(f"🚫 ERREUR PERMANENTE {symbol} (LONDON)\n{error[:200]}")
            return

        tiers = [60, 120, 300, 600, 1800]
        idx = min(fail_info["count"] - 1, len(tiers) - 1)
        cooldown = tiers[idx]
        fail_info["next_retry"] = time.time() + cooldown
        self._close_failures[symbol] = fail_info
        logger.warning("[%s] 🔁 Échec close #%d — retry dans %ds", symbol, fail_info["count"], cooldown)

        if fail_info["count"] == 1 or fail_info["count"] % 5 == 0:
            self._telegram.notify_error(
                f"⚠️ Close London {symbol} échouée (×{fail_info['count']})\nRetry dans {cooldown}s\n{error[:200]}"
            )

    # ── Heartbeat ──────────────────────────────────────────────────────────

    def _maybe_heartbeat(self) -> None:
        now = time.time()
        if now - self._last_heartbeat < LON_HEARTBEAT_SECONDS:
            return
        self._last_heartbeat = now

        # Equity scope London (USD + actifs des paires London uniquement)
        total_equity = self._get_london_scoped_equity()

        allocated = total_equity * LON_CAPITAL_PCT

        # Countdown prochaine bougie H4
        now_utc = datetime.now(timezone.utc)
        h4_hour = ((now_utc.hour // 4) + 1) * 4
        if h4_hour >= 24:
            next_h4 = now_utc.replace(hour=0, minute=0, second=0, microsecond=0) + timedelta(days=1)
        else:
            next_h4 = now_utc.replace(hour=h4_hour, minute=0, second=0, microsecond=0)
        countdown_min = max(0, int((next_h4 - now_utc).total_seconds() / 60))
        countdown_hm = f"{countdown_min}min" if countdown_min < 60 else f"{countdown_min // 60}h{countdown_min % 60:02d}"
        next_h4_paris = next_h4.astimezone(ZoneInfo("Europe/Paris"))
        countdown_str = f"{countdown_hm} ({next_h4_paris.strftime('%Hh%M')} Paris - {next_h4.strftime('%Hh%M')} UTC)"

        # Session status
        current_hour = now_utc.hour
        if LON_SESSION_START <= current_hour < LON_SESSION_END:
            session_status = "🟢 EN COURS (range building)"
        elif current_hour >= LON_SESSION_END or current_hour < LON_SESSION_START:
            session_status = "🔵 BREAKOUT WINDOW"
        else:
            session_status = "⚪ HORS SESSION"

        # Positions
        pos_lines: list[str] = []
        for sym, pos in self._positions.items():
            try:
                ticker = self._data.get_ticker(sym)
                if not ticker:
                    continue
                price = ticker.last_price
                pnl = (price - pos.entry_price) * pos.remaining_size
                pnl_pct = (price - pos.entry_price) / pos.entry_price * 100
                sl_dist = (price - pos.sl_price) / price * 100
                base = sym.split("-")[0]
                icon = "🟢" if pnl >= 0 else "🔴"
                tp_tag = "TP1 ✅" if pos.tp1_hit else "waiting TP1"
                be_tag = " 🔒BE" if pos.breakeven_active else ""

                logger.info(
                    "   %s %s | entry=%s → %s | P&L %+.2f%% ($%+.2f) | SL=%s | %s%s",
                    icon, sym, _fmt(pos.entry_price), _fmt(price),
                    pnl_pct, pnl, _fmt(pos.sl_price), tp_tag, be_tag,
                )
                pos_lines.append(
                    f"  {icon} `{sym}` @ `{_fmt(pos.entry_price)}` → `{_fmt(price)}` | "
                    f"P&L `{pnl_pct:+.1f}%` (`{pnl:+.2f}$`) | SL `{_fmt(pos.sl_price)}`{be_tag}"
                )
            except Exception:
                pass

        # Session states summary (readable)
        breakouts_ready = 0
        near_breakout_alerts: list[tuple[str, float, float, float]] = []
        completed_sessions = 0
        consumed_sessions = 0
        cooldown_sessions = 0
        below_breakout = 0
        for sym in LON_TRADING_PAIRS:
            ss = self._session_states.get(sym)
            if ss and ss.session_bars > 0:
                if ss.session_complete:
                    completed_sessions += 1

                    if ss.breakout_consumed:
                        consumed_sessions += 1
                        continue

                    if now_utc.timestamp() * 1000 <= ss.cooldown_until:
                        cooldown_sessions += 1
                        continue

                    breakouts_ready += 1

                    if ss.session_high > 0:
                        try:
                            ticker = self._data.get_ticker(sym)
                            if ticker and ticker.last_price > 0:
                                dist_pct = (ticker.last_price - ss.session_high) / ss.session_high * 100
                                if -1.0 <= dist_pct <= 1.0:
                                    near_breakout_alerts.append((sym, dist_pct, ticker.last_price, ss.session_high))
                                elif dist_pct < -1.0:
                                    below_breakout += 1
                        except Exception:
                            pass

        # Unrealized PnL total
        total_unrealized = 0.0
        for sym, pos in self._positions.items():
            try:
                ticker = self._data.get_ticker(sym)
                if ticker:
                    total_unrealized += (ticker.last_price - pos.entry_price) * pos.remaining_size
            except Exception:
                pass

        logger.info(
            "💓 LONDON Alive | tick=%d | pos=%d/%d | equity=$%.2f | alloc=$%.2f\n"
            "   Session: %s | ⏳ %s",
            self._tick_count, len(self._positions), LON_MAX_POSITIONS,
            total_equity, allocated,
            session_status, countdown_str,
        )

        # Telegram
        try:
            # System status
            if total_unrealized < -allocated * 0.05:
                sys_emoji, sys_label = "🔴", "risk mode"
            elif total_unrealized < 0 and len(self._positions) > 0:
                sys_emoji, sys_label = "🟡", "watching"
            else:
                sys_emoji, sys_label = "🟢", "stable"

            last_update = datetime.now(timezone.utc).strftime("%H:%M UTC")
            unr_emoji = "🟢" if total_unrealized >= 0 else "🔴"

            tg_lines = [
                f"{sys_emoji} *LONDON BREAKOUT* 🇬🇧 ({len(LON_TRADING_PAIRS)} paires) — {sys_label}",
                f"  💰 Equity scope: `${total_equity:,.0f}` | Alloué (20%): `${allocated:,.0f}`",
                f"  Session: {session_status}",
                f"  📊 Pos: `{len(self._positions)}/{LON_MAX_POSITIONS}` | Sessions complètes: `{completed_sessions}` | Candidats: `{breakouts_ready}`",
            ]
            if self._positions:
                tg_lines.append(f"  {unr_emoji} PnL open: `${total_unrealized:+.2f}`")
            tg_lines.append(
                f"  🔎 Filtres: close>H_session | vol>={LON_VOL_MULT:.1f}×MA{LON_VOL_MA_PERIOD} | range>={LON_MIN_RANGE*100:.1f}%"
            )

            if near_breakout_alerts:
                tg_lines.append("\n  *Proches breakout (±1%)*:")
                for sym, dist_pct, price, high in sorted(near_breakout_alerts, key=lambda x: abs(x[1]))[:4]:
                    tg_lines.append(
                        f"  ⚠️ `{sym}` prix `{_fmt(price)}` | H_session `{_fmt(high)}` | écart `{dist_pct:+.1f}%`"
                    )

            if not self._positions and breakouts_ready == 0:
                tg_lines.append(
                    f"  ℹ️ Aucun setup immédiat (consumed={consumed_sessions}, cooldown={cooldown_sessions}, sous breakout={below_breakout})"
                )

            tg_lines.append(f"  ⏳ Prochaine H4: `{countdown_str}`")
            if pos_lines:
                tg_lines.append("")
                tg_lines.extend(pos_lines)
            tg_lines.append(f"\n  🕐 `{last_update}`")
            tg_lines.append(f"[Dashboard]({DASHBOARD_URL})")
            self._telegram.send_raw("\n".join(tg_lines))
        except Exception:
            logger.warning("Telegram heartbeat failed", exc_info=True)

        # Firebase heartbeat
        try:
            fb_log_heartbeat(
                open_positions=len(self._positions),
                total_equity=allocated,
                total_risk_pct=0.0,
                pairs_count=len(LON_TRADING_PAIRS),
                exchange="revolut-london",
            )
        except Exception:
            pass

    def _get_london_scoped_equity(self) -> float:
        """Equity scope London: USD + actifs des paires London uniquement."""
        try:
            balances = self._client.get_balances()
            tracked_bases = {symbol.split("-")[0] for symbol in LON_TRADING_PAIRS}

            total = 0.0
            for balance in balances:
                if balance.total <= 0:
                    continue

                if balance.currency == "USD":
                    total += balance.total
                    continue

                if balance.currency in tracked_bases:
                    try:
                        ticker = self._data.get_ticker(f"{balance.currency}-USD")
                        if ticker:
                            total += balance.total * ticker.last_price
                    except Exception:
                        continue

            return total
        except Exception:
            return 0.0

    # ── Daily tasks ────────────────────────────────────────────────────────

    def _maybe_daily_tasks(self) -> None:
        """Cleanup + snapshot — 1×/jour UTC."""
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")

        if today != self._last_cleanup_date:
            self._last_cleanup_date = today
            try:
                fb_cleanup_events()
                logger.info("🧹 Cleanup events Firebase")
            except Exception:
                pass

        if today != self._last_snapshot_date:
            self._last_snapshot_date = today
            try:
                total_equity = self._get_london_scoped_equity()

                positions = []
                for sym, pos in self._positions.items():
                    positions.append({
                        "symbol": sym,
                        "entry_price": pos.entry_price,
                        "sl_price": pos.sl_price,
                        "tp1_hit": pos.tp1_hit,
                        "remaining_size": pos.remaining_size,
                    })

                fb_log_daily_snapshot(
                    equity=total_equity * LON_CAPITAL_PCT,
                    positions=positions,
                    daily_pnl=0.0,
                    trades_today=0,
                    exchange="revolut-london",
                )
                logger.info("📸 Daily snapshot Firebase loggé")
            except Exception:
                pass

    # ── Helpers ────────────────────────────────────────────────────────────

    def _save_state(self) -> None:
        self._store.save(
            self._positions,
            self._session_states,
            self._last_candle_ts,
            self._candle_buffers,
        )

    @staticmethod
    def _format_order_price(price: float) -> str:
        if price >= 1000:
            return f"{price:.2f}"
        elif price >= 1:
            return f"{price:.4f}"
        elif price >= 0.01:
            return f"{price:.6f}"
        else:
            return f"{price:.8f}"

    def _shutdown(self) -> None:
        logger.info("🛑 Arrêt LondonBreakoutBot...")
        self._save_state()
        logger.info("💾 État final sauvegardé")
        self._client.close()
        self._telegram.close()
        logger.info("LondonBreakoutBot arrêté proprement")


# ── Point d'entrée ─────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="TradeX – London Breakout Bot")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Mode simulation (log les ordres sans les exécuter)",
    )
    args = parser.parse_args()

    bot = LondonBreakoutBot(dry_run=args.dry_run)
    signal.signal(signal.SIGTERM, lambda *_: bot.stop())
    bot.run()


if __name__ == "__main__":
    main()
