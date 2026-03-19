"""
Bot Breakout Momentum — Revolut X, maker-only, 15m.

Stratégie :
  1. Breakout du high(12) sur bougies 15m
  2. Filtres : ATR expansion + volume spike + ATR min
  3. SL = entry − 0.8×ATR | TP = entry + 2.0×ATR
  4. Trailing stop : activation +0.3×ATR, distance 0.2×ATR
  5. LONG ONLY (Revolut X = spot)

Capital : budget fixe isolé (BRK_ALLOCATED_BALANCE, défaut $100).
Walk-forward validé 3/3 OOS, PF OOS 13.04 (ULTRATRAIL, strong, maker).

Usage :
    python -m src.bot_breakout              # Production
    python -m src.bot_breakout --dry-run    # Log les ordres sans les exécuter
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import signal
import sys
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Optional
from zoneinfo import ZoneInfo

from src import config
from src.core.breakout_engine import (
    BreakoutSignal,
    TrailingResult,
    compute_atr,
    compute_sma,
    detect_breakout,
    rolling_high,
    update_trailing_stop,
)
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
from src.runtime_overrides import (
    get_heartbeat_override_seconds,
    get_pending_runtime_actions,
    mark_runtime_action_status,
)

# ── Logging ────────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=getattr(logging, config.LOG_LEVEL, logging.INFO),
    format="[%(asctime)s] %(levelname)s %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("tradex.breakout_bot")

logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)


# ── Config Breakout (depuis config.py / .env) ─────────────────────────────────

BRK_TRADING_PAIRS: list[str] = config.BRK_TRADING_PAIRS
BRK_ALLOCATED_BALANCE: float = config.BRK_ALLOCATED_BALANCE
BRK_RISK_PCT: float = config.BRK_RISK_PERCENT
BRK_MAX_POSITIONS: int = config.BRK_MAX_POSITIONS
BRK_POLLING_SECONDS: int = config.BRK_POLLING_SECONDS
BRK_HEARTBEAT_SECONDS: int = config.BRK_HEARTBEAT_SECONDS
BRK_MAKER_WAIT_SECONDS: int = config.BRK_MAKER_WAIT_SECONDS

# Stratégie
BRK_LOOKBACK: int = config.BRK_LOOKBACK
BRK_ATR_PERIOD: int = config.BRK_ATR_PERIOD
BRK_VOL_MA_PERIOD: int = config.BRK_VOL_MA_PERIOD
BRK_TP_ATR_MULT: float = config.BRK_TP_ATR_MULT
BRK_SL_ATR_MULT: float = config.BRK_SL_ATR_MULT
BRK_TRAIL_ACTIVATION_ATR: float = config.BRK_TRAIL_ACTIVATION_ATR
BRK_TRAIL_DISTANCE_ATR: float = config.BRK_TRAIL_DISTANCE_ATR
BRK_ATR_EXPANSION_LOOKBACK: int = config.BRK_ATR_EXPANSION_LOOKBACK
BRK_ATR_EXPANSION_RATIO: float = config.BRK_ATR_EXPANSION_RATIO
BRK_VOLUME_SPIKE_MULT: float = config.BRK_VOLUME_SPIKE_MULT
BRK_MIN_ATR_PCT: float = config.BRK_MIN_ATR_PCT
BRK_COOLDOWN_BARS: int = config.BRK_COOLDOWN_BARS
BRK_MAX_CONSECUTIVE_LOSSES: int = config.BRK_MAX_CONSECUTIVE_LOSSES
BRK_COOLDOWN_BARS_AFTER_TILT: int = config.BRK_COOLDOWN_BARS_AFTER_TILT
BRK_CANDLE_INTERVAL: int = config.BRK_CANDLE_INTERVAL  # 15 minutes

MAX_MAKER_RETRIES = 2

# State file
BRK_STATE_DIR: str = os.path.join(os.path.dirname(__file__), "..", "data")


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
#  Position data
# ─────────────────────────────────────────────────────────────────────

@dataclass
class BreakoutPosition:
    """Position gérée par le Breakout Momentum bot."""

    symbol: str
    entry_price: float
    sl_price: float
    tp_price: float
    size: float                    # unités base
    size_usd: float
    venue_order_id: str
    atr_at_entry: float
    trailing_activation: float     # prix d'activation du trailing
    trailing_distance: float       # distance en prix
    peak_price: float = 0.0        # plus haut atteint
    trailing_active: bool = False
    firebase_trade_id: Optional[str] = None
    opened_at: float = 0.0

    def to_dict(self) -> dict:
        return {
            "symbol": self.symbol,
            "entry_price": self.entry_price,
            "sl_price": self.sl_price,
            "tp_price": self.tp_price,
            "size": self.size,
            "size_usd": self.size_usd,
            "venue_order_id": self.venue_order_id,
            "atr_at_entry": self.atr_at_entry,
            "trailing_activation": self.trailing_activation,
            "trailing_distance": self.trailing_distance,
            "peak_price": self.peak_price,
            "trailing_active": self.trailing_active,
            "firebase_trade_id": self.firebase_trade_id,
            "opened_at": self.opened_at,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "BreakoutPosition":
        return cls(
            symbol=d["symbol"],
            entry_price=d["entry_price"],
            sl_price=d["sl_price"],
            tp_price=d["tp_price"],
            size=d["size"],
            size_usd=d["size_usd"],
            venue_order_id=d.get("venue_order_id", "unknown"),
            atr_at_entry=d.get("atr_at_entry", 0.0),
            trailing_activation=d.get("trailing_activation", 0.0),
            trailing_distance=d.get("trailing_distance", 0.0),
            peak_price=d.get("peak_price", 0.0),
            trailing_active=d.get("trailing_active", False),
            firebase_trade_id=d.get("firebase_trade_id"),
            opened_at=d.get("opened_at", 0.0),
        )


# ─────────────────────────────────────────────────────────────────────
#  State store (JSON persistence)
# ─────────────────────────────────────────────────────────────────────

class BreakoutStateStore:
    """Persistance atomique du bot Breakout Momentum."""

    def __init__(self, state_file: str) -> None:
        self._path = Path(state_file).resolve()

    def save(
        self,
        positions: dict[str, BreakoutPosition],
        last_candle_ts: dict[str, int],
        candle_buffers: dict[str, list[dict]],
        cooldowns: dict[str, int],
        consecutive_losses: dict[str, int],
        tilt_cooldowns: dict[str, int],
    ) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        state = {
            "positions": {s: p.to_dict() for s, p in positions.items()},
            "last_candle_ts": last_candle_ts,
            "candle_buffers": candle_buffers,
            "cooldowns": cooldowns,
            "consecutive_losses": consecutive_losses,
            "tilt_cooldowns": tilt_cooldowns,
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
            logger.info("📂 Pas de state Breakout — démarrage à vide")
            return {}
        try:
            with open(self._path, "r") as f:
                return json.load(f)
        except Exception as e:
            logger.error("❌ Load state Breakout échoué: %s", e)
            return {}


# ─────────────────────────────────────────────────────────────────────
#  Bot principal
# ─────────────────────────────────────────────────────────────────────

class BreakoutMomentumBot:
    """Bot Breakout Momentum — Revolut X, maker-only, 15m, LONG only."""

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
            silent=self.dry_run,
        )

        # State
        state_file = os.path.join(BRK_STATE_DIR, "state_breakout.json")
        self._store = BreakoutStateStore(state_file)
        self._positions: dict[str, BreakoutPosition] = {}

        # Candle tracking
        self._last_candle_ts: dict[str, int] = {}
        self._candle_buffers: dict[str, list[dict]] = {s: [] for s in BRK_TRADING_PAIRS}

        # Cooldowns & anti-tilt
        self._cooldowns: dict[str, int] = {}        # symbol → candle_ts until cooldown expires
        self._consecutive_losses: dict[str, int] = {}  # symbol → consecutive loss count
        self._tilt_cooldowns: dict[str, int] = {}    # symbol → candle_ts until tilt cooldown expires

        # Heartbeat
        self._last_heartbeat: float = 0.0
        self._heartbeat_seconds: int = BRK_HEARTBEAT_SECONDS
        self._tick_count: int = 0

        # Daily cleanup / snapshot
        self._last_cleanup_date: str = ""
        self._last_snapshot_date: str = ""

        # Close failure tracking
        self._close_failures: dict[str, dict] = {}

        # Stats
        self._total_trades: int = 0
        self._total_wins: int = 0

        if dry_run:
            logger.info("🔧 Mode DRY-RUN — aucun ordre ne sera exécuté")

    # ── Run ────────────────────────────────────────────────────────────────

    def run(self) -> None:
        self._running = True

        logger.info("═" * 60)
        logger.info("⚡ BreakoutMomentumBot démarré — 15m ULTRATRAIL")
        logger.info("   Paires    : %d | %s", len(BRK_TRADING_PAIRS), ", ".join(BRK_TRADING_PAIRS))
        logger.info("   Budget    : $%.0f (fixe, isolé)", BRK_ALLOCATED_BALANCE)
        logger.info("   Risque    : %.0f%% par trade | Max %d position(s)",
                     BRK_RISK_PCT * 100, BRK_MAX_POSITIONS)
        logger.info("   Breakout  : high(%d) | ATR(%d) exp ×%.2f",
                     BRK_LOOKBACK, BRK_ATR_PERIOD, BRK_ATR_EXPANSION_RATIO)
        logger.info("   TP: %.1f×ATR | SL: %.1f×ATR",
                     BRK_TP_ATR_MULT, BRK_SL_ATR_MULT)
        logger.info("   Trail     : activ %.1f×ATR | dist %.1f×ATR",
                     BRK_TRAIL_ACTIVATION_ATR, BRK_TRAIL_DISTANCE_ATR)
        logger.info("   Polling   : %ds | Candles: %dm | Maker wait: %ds",
                     BRK_POLLING_SECONDS, BRK_CANDLE_INTERVAL, BRK_MAKER_WAIT_SECONDS)
        logger.info("   Cooldown  : %d bars | Tilt: %d losses → %d bars",
                     BRK_COOLDOWN_BARS, BRK_MAX_CONSECUTIVE_LOSSES, BRK_COOLDOWN_BARS_AFTER_TILT)
        logger.info("   Mode      : LONG-ONLY")
        if self.dry_run:
            logger.info("   ⚠️  DRY-RUN actif")
        logger.info("═" * 60)

        self._initialize()

        try:
            while self._running:
                self._tick()
                time.sleep(BRK_POLLING_SECONDS)
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
                    self._positions[sym] = BreakoutPosition.from_dict(d)
                except Exception as e:
                    logger.warning("⚠️ Position %s corrompue: %s", sym, e)

            # Last candle timestamps
            self._last_candle_ts = state.get("last_candle_ts", {})

            # Candle buffers
            self._candle_buffers = state.get("candle_buffers", {s: [] for s in BRK_TRADING_PAIRS})

            # Cooldowns
            self._cooldowns = state.get("cooldowns", {})
            self._consecutive_losses = state.get("consecutive_losses", {})
            self._tilt_cooldowns = state.get("tilt_cooldowns", {})

            logger.info(
                "📂 State Breakout chargé: %d positions",
                len(self._positions),
            )

        # Réconcilier les positions
        self._reconcile_positions()

        # Charger les bougies 15m initiales
        logger.info("── Chargement des bougies 15m initiales ──")
        for symbol in BRK_TRADING_PAIRS:
            try:
                candles = self._client.get_candles(symbol, interval=BRK_CANDLE_INTERVAL)
                candles.sort(key=lambda c: c.timestamp)
                if candles:
                    self._last_candle_ts[symbol] = candles[-1].timestamp
                    self._candle_buffers[symbol] = [
                        {"timestamp": c.timestamp, "open": c.open, "high": c.high,
                         "low": c.low, "close": c.close, "volume": c.volume}
                        for c in candles[-200:]  # garder 200 dernières (15m → ~50h)
                    ]
                    logger.info("[%s] %d bougies 15m chargées", symbol, len(candles))
                else:
                    logger.warning("[%s] Aucune bougie 15m reçue", symbol)
            except Exception as e:
                logger.error("[%s] ❌ Erreur chargement bougies: %s", symbol, e)

        self._save_state()

        open_count = len(self._positions)
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

            if held >= pos.size * 0.90:
                logger.info("[%s] ✅ Position confirmée | %.8f %s", sym, held, base)
            else:
                logger.warning(
                    "[%s] ⚠️ Position locale mais solde %s=%.8f < size=%.8f → retirée",
                    sym, base, held, pos.size,
                )
                removed.append(sym)

        for sym in removed:
            del self._positions[sym]

        if removed:
            self._save_state()

    # ── Tick ───────────────────────────────────────────────────────────────

    def _tick(self) -> None:
        """Un cycle de polling — itère sur toutes les paires."""
        self._apply_runtime_actions()
        self._tick_count += 1

        for symbol in BRK_TRADING_PAIRS:
            try:
                self._process_symbol(symbol)
            except Exception as e:
                logger.error("[%s] Erreur tick: %s", symbol, e)

        self._maybe_heartbeat()
        self._maybe_daily_tasks()

    def _apply_runtime_actions(self) -> None:
        """Applique les actions runtime (Telegram commands)."""
        actions = get_pending_runtime_actions("breakout")
        if not actions:
            return

        for action in actions:
            action_id = str(action.get("_id", ""))
            kind = str(action.get("action", "")).lower().strip()
            symbol = str(action.get("symbol", "")).upper().strip()
            value = action.get("value")

            try:
                if kind == "close":
                    targets = list(self._positions.keys()) if symbol == "ALL" else [symbol]
                    closed = 0
                    for sym in targets:
                        pos = self._positions.get(sym)
                        if not pos:
                            continue
                        ticker = self._data.get_ticker(sym)
                        if not ticker:
                            continue
                        self._close_position(sym, ticker.last_price, "Manual close (Telegram)")
                        closed += 1
                    mark_runtime_action_status(action_id, "done", f"manual close ({closed})")
                    if not self.dry_run:
                        try:
                            fb_log_event("MANUAL_ACTION", {
                                "action": "close", "symbol": symbol, "count": closed,
                            }, exchange="revolut-breakout")
                        except Exception:
                            pass
                    continue

                if kind == "set_sl":
                    pos = self._positions.get(symbol)
                    if not pos:
                        mark_runtime_action_status(action_id, "failed", "position introuvable")
                        continue
                    try:
                        price = float(str(value))
                    except Exception:
                        mark_runtime_action_status(action_id, "failed", "price invalide")
                        continue
                    if price <= 0:
                        mark_runtime_action_status(action_id, "failed", "price invalide")
                        continue
                    ticker = self._data.get_ticker(symbol)
                    if not ticker:
                        mark_runtime_action_status(action_id, "failed", "ticker indisponible")
                        continue
                    if price >= ticker.last_price:
                        mark_runtime_action_status(action_id, "failed", "SL doit être sous le prix")
                        continue
                    pos.sl_price = price
                    self._save_state()
                    mark_runtime_action_status(action_id, "done", "set_sl appliqué")
                    continue

                mark_runtime_action_status(action_id, "failed", "action inconnue")
            except Exception as e:
                mark_runtime_action_status(action_id, "failed", f"erreur: {e}")

    def _process_symbol(self, symbol: str) -> None:
        """Traite un symbole : check prix (SL/TP/trailing) + check nouvelles bougies."""

        # ── 1. Gestion des positions ouvertes via ticker ──
        pos = self._positions.get(symbol)
        if pos:
            ticker = self._data.get_ticker(symbol)
            if ticker:
                self._manage_position(symbol, pos, ticker)
                if symbol not in self._positions:
                    return  # Position fermée

        # ── 2. Vérifier s'il y a une nouvelle bougie 15m ──
        try:
            candles = self._client.get_candles(symbol, interval=BRK_CANDLE_INTERVAL)
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

        # Buffer de bougies (200 dernières)
        self._candle_buffers[symbol] = [
            {"timestamp": c.timestamp, "open": c.open, "high": c.high,
             "low": c.low, "close": c.close, "volume": c.volume}
            for c in candles[-200:]
        ]

        # ── Nouvelle bougie 15m → check breakout signal ──
        self._check_breakout_signal(symbol, candles)
        self._save_state()

    # ── Position management ────────────────────────────────────────────────

    def _manage_position(self, symbol: str, pos: BreakoutPosition, ticker: TickerData) -> None:
        """Gère SL, TP, et trailing stop pour une position ouverte."""
        price = ticker.last_price

        # ── SL check ──
        if price <= pos.sl_price:
            reason = "TRAILING SL" if pos.trailing_active else "SL"
            logger.info("[%s] 🛑 %s HIT | prix=%s | SL=%s", symbol, reason, _fmt(price), _fmt(pos.sl_price))
            self._close_position(symbol, price, f"{reason} atteint")
            return

        # ── TP check ──
        if price >= pos.tp_price:
            logger.info("[%s] 🎯 TP HIT | prix=%s | TP=%s", symbol, _fmt(price), _fmt(pos.tp_price))
            self._close_position(symbol, price, "TP atteint")
            return

        # ── Trailing stop update ──
        result = update_trailing_stop(
            current_price=price,
            entry_price=pos.entry_price,
            current_sl=pos.sl_price,
            peak_price=pos.peak_price,
            trailing_activation=pos.trailing_activation,
            trailing_distance=pos.trailing_distance,
        )

        if result.new_sl > pos.sl_price:
            old_sl = pos.sl_price
            pos.sl_price = result.new_sl
            pos.peak_price = result.peak_price

            if not pos.trailing_active and result.trailing_active:
                pos.trailing_active = True
                logger.info(
                    "[%s] 🔒 TRAILING activé | prix=%s ≥ activation=%s | SL: %s → %s",
                    symbol, _fmt(price), _fmt(pos.trailing_activation),
                    _fmt(old_sl), _fmt(result.new_sl),
                )
                # Telegram trailing activation
                self._telegram._send(
                    f"🔒 *Trailing activé – {symbol}* ⚡ BREAKOUT\n"
                    f"  Prix: `{_fmt(price)}` ≥ Activation: `{_fmt(pos.trailing_activation)}`\n"
                    f"  SL: `{_fmt(old_sl)}` → `{_fmt(result.new_sl)}`\n"
                    f"  Entrée: `{_fmt(pos.entry_price)}` | P&L: `{(price - pos.entry_price) / pos.entry_price * 100:+.2f}%`\n"
                    f"[Dashboard]({DASHBOARD_URL})"
                )
            else:
                logger.debug(
                    "[%s] 📈 Trail update | peak=%s | SL: %s → %s",
                    symbol, _fmt(result.peak_price), _fmt(old_sl), _fmt(result.new_sl),
                )

            self._save_state()
        elif result.peak_price > pos.peak_price:
            pos.peak_price = result.peak_price

    def _close_position(self, symbol: str, exit_price: float, reason: str) -> None:
        """Ferme une position."""
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

        exit_size = pos.size

        # Vérifier le solde réel
        if not self.dry_run:
            exit_size = self._check_real_balance(symbol, pos.size)
            if exit_size <= 0:
                del self._positions[symbol]
                self._save_state()
                return

        # Exécuter la vente (taker pour SL, maker sinon)
        use_taker = reason.startswith("SL") or reason.startswith("TRAILING")
        success = self._execute_sell(symbol, exit_price, exit_size, use_taker=use_taker)
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

        # Stats
        self._total_trades += 1
        if pnl_gross >= 0:
            self._total_wins += 1
            self._consecutive_losses[symbol] = 0
        else:
            losses = self._consecutive_losses.get(symbol, 0) + 1
            self._consecutive_losses[symbol] = losses
            if losses >= BRK_MAX_CONSECUTIVE_LOSSES:
                # Anti-tilt cooldown
                latest_ts = self._last_candle_ts.get(symbol, 0)
                self._tilt_cooldowns[symbol] = latest_ts + BRK_COOLDOWN_BARS_AFTER_TILT * BRK_CANDLE_INTERVAL * 60 * 1000
                logger.warning(
                    "[%s] ⚠️ TILT — %d pertes consécutives → cooldown %d bars",
                    symbol, losses, BRK_COOLDOWN_BARS_AFTER_TILT,
                )

        logger.info(
            "[%s] %s CLOSE | entry=%s | exit=%s | PnL=$%+.2f (%+.1f%%) | %s%s",
            symbol, pnl_emoji, _fmt(pos.entry_price), _fmt(exit_price),
            pnl_gross, pnl_pct, reason,
            " 🔒TRAIL" if pos.trailing_active else "",
        )

        # Cooldown après trade
        latest_ts = self._last_candle_ts.get(symbol, 0)
        self._cooldowns[symbol] = latest_ts + BRK_COOLDOWN_BARS * BRK_CANDLE_INTERVAL * 60 * 1000

        # Telegram
        base = symbol.split("-")[0]
        trail_tag = " (trailing)" if pos.trailing_active else ""
        self._telegram._send(
            f"{pnl_emoji} *Position fermée – {symbol}* ⚡ BREAKOUT\n"
            f"  Raison: {reason}{trail_tag}\n"
            f"  Entrée: `{_fmt(pos.entry_price)}` → Sortie: `{_fmt(exit_price)}`\n"
            f"  P&L: `{pnl_gross:+.2f} USD` (`{pnl_pct:+.1f}%`)\n"
            f"  ATR: `{_fmt(pos.atr_at_entry)}` | Peak: `{_fmt(pos.peak_price)}`\n"
            f"[Dashboard]({DASHBOARD_URL})"
        )

        # Firebase
        if not self.dry_run and pos.firebase_trade_id:
            try:
                fb_position = Position(
                    symbol=symbol,
                    side=OrderSide.BUY,
                    entry_price=pos.entry_price,
                    sl_price=pos.sl_price,
                    size=exit_size,
                    venue_order_id=pos.venue_order_id,
                    status=PositionStatus.CLOSED,
                    strategy=StrategyType.BREAKOUT,
                    tp_price=pos.tp_price,
                    pnl=pnl_gross,
                )
                log_trade_closed(
                    trade_id=pos.firebase_trade_id,
                    position=fb_position,
                    exit_price=exit_price,
                    reason=reason,
                    fill_type="taker" if use_taker else "maker",
                    equity_after=0.0,
                    actual_exit_size=exit_size,
                )
            except Exception as e:
                logger.warning("🔥 Firebase log_trade_closed échoué: %s", e)

        del self._positions[symbol]
        self._save_state()

    # ── Breakout signal detection ──────────────────────────────────────────

    def _check_breakout_signal(self, symbol: str, candles: list[Candle]) -> None:
        """Vérifie les conditions de breakout sur la dernière bougie 15m."""

        # Guard: déjà une position sur ce symbole
        if symbol in self._positions:
            return

        # Guard: max positions
        if len(self._positions) >= BRK_MAX_POSITIONS:
            return

        # Guard: cooldown normal
        latest_ts = candles[-1].timestamp if candles else 0
        if latest_ts <= self._cooldowns.get(symbol, 0):
            return

        # Guard: tilt cooldown
        if latest_ts <= self._tilt_cooldowns.get(symbol, 0):
            return

        # Reconstruct Candle objects from buffer for consistent analysis
        buf = self._candle_buffers.get(symbol, [])
        if len(buf) < 50:
            analysis_candles = candles
        else:
            analysis_candles = [
                Candle(
                    timestamp=d["timestamp"], open=d["open"], high=d["high"],
                    low=d["low"], close=d["close"], volume=d["volume"],
                )
                for d in buf
            ]

        # Detect breakout using pure engine
        signal = detect_breakout(
            analysis_candles,
            lookback=BRK_LOOKBACK,
            atr_period=BRK_ATR_PERIOD,
            vol_ma_period=BRK_VOL_MA_PERIOD,
            tp_atr_mult=BRK_TP_ATR_MULT,
            sl_atr_mult=BRK_SL_ATR_MULT,
            trail_activation_atr=BRK_TRAIL_ACTIVATION_ATR,
            trail_distance_atr=BRK_TRAIL_DISTANCE_ATR,
            atr_expansion_lookback=BRK_ATR_EXPANSION_LOOKBACK,
            atr_expansion_ratio=BRK_ATR_EXPANSION_RATIO,
            volume_spike_mult=BRK_VOLUME_SPIKE_MULT,
            min_atr_pct=BRK_MIN_ATR_PCT,
        )

        if signal is None:
            return

        vol_ratio = signal.volume / signal.volume_ma if signal.volume_ma > 0 else 0
        logger.info(
            "[%s] ⚡ BREAKOUT SIGNAL | close=%s > high(%d)=%s | vol=%.0f (%.1fx MA) | ATR=%s",
            symbol, _fmt(signal.entry_price), BRK_LOOKBACK, _fmt(signal.recent_high),
            signal.volume, vol_ratio, _fmt(signal.atr_value),
        )

        # Execute entry
        self._execute_entry(symbol, signal)

    # ── Entry execution ────────────────────────────────────────────────────

    def _execute_entry(self, symbol: str, signal: BreakoutSignal) -> None:
        """Exécute un signal d'entrée Breakout Momentum."""

        # ── Sizing (budget fixe isolé) ──
        allocated = BRK_ALLOCATED_BALANCE
        risk_amount = allocated * BRK_RISK_PCT
        sl_dist = signal.entry_price - signal.sl_price

        if sl_dist <= 0:
            return

        size = risk_amount / sl_dist
        size_usd = size * signal.entry_price

        # Cap au budget alloué
        if size_usd > allocated:
            size_usd = allocated
            size = size_usd / signal.entry_price

        if size_usd < 1:
            logger.info("[%s] ⏭️ Taille trop faible ($%.2f) — skip", symbol, size_usd)
            return

        # Vérifier le solde USD disponible
        try:
            balances = self._data.get_balances()
            quote = symbol.split("-")[1]
            usd_bal = next((b for b in balances if b.currency == quote), None)
            available = usd_bal.available if usd_bal else 0.0
        except Exception as e:
            logger.error("[%s] ❌ Impossible de récupérer le solde: %s", symbol, e)
            return

        if available < size_usd:
            logger.info("[%s] ⏭️ Solde USD insuffisant ($%.2f < $%.2f) — skip", symbol, available, size_usd)
            return

        entry_price = signal.entry_price

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
                            "[%s] ⚡ ✅ BUY exécuté | @ %s | size=%s | %s (tentative %d)",
                            symbol, p_str, s_str, ft, attempt,
                        )
                        break

                    logger.info(
                        "[%s] ⚡ BUY: maker no-fill (tentative %d/%d)",
                        symbol, attempt, MAX_MAKER_RETRIES,
                    )
                    try:
                        ticker = self._data.get_ticker(symbol)
                        if ticker:
                            current_price = ticker.last_price
                    except Exception:
                        pass
                except Exception as e:
                    logger.error("[%s] ⚡ ❌ BUY maker échoué (tentative %d): %s", symbol, attempt, e)
                    break

            if not filled:
                # Taker fallback
                logger.warning(
                    "[%s] ⚡ BUY: %d makers no-fill → TAKER FALLBACK",
                    symbol, MAX_MAKER_RETRIES,
                )
                p_str = self._format_order_price(current_price)
                s_str = f"{size:.8f}"
                o = OrderRequest(symbol=symbol, side=OrderSide.BUY, base_size=s_str, price=p_str)

                try:
                    result = self._place_taker_order(o)
                    ft = result.get("fill_type", "unknown")
                    if ft == "no_fill":
                        logger.error("[%s] ⚡ BUY: taker fallback échoué — abandonné", symbol)
                        return
                    venue_order_id = result.get("venue_order_id", "unknown")
                    fill_type = "taker"
                    actual_price = result.get("actual_price", current_price)
                    filled = True
                except Exception as e:
                    logger.error("[%s] ⚡ ❌ BUY taker échoué: %s", symbol, e)
                    self._telegram.notify_error(f"⚡ BUY {symbol} échoué: {e}")
                    return

            if not filled:
                return
        else:
            logger.info(
                "[DRY-RUN] BREAKOUT BUY %s | entry=%s | SL=%s | TP=%s | trail_act=%s | size=%s ($%.2f)",
                symbol, _fmt(entry_price), _fmt(signal.sl_price),
                _fmt(signal.tp_price), _fmt(signal.trailing_activation),
                f"{size:.8f}", size_usd,
            )

        # ── Enregistrer la position ──
        pos = BreakoutPosition(
            symbol=symbol,
            entry_price=actual_price,
            sl_price=signal.sl_price,
            tp_price=signal.tp_price,
            size=size,
            size_usd=size_usd,
            venue_order_id=venue_order_id,
            atr_at_entry=signal.atr_value,
            trailing_activation=signal.trailing_activation,
            trailing_distance=signal.trailing_distance,
            peak_price=actual_price,
            opened_at=time.time(),
        )
        self._positions[symbol] = pos
        self._save_state()

        # ── Telegram ──
        base = symbol.split("-")[0]
        sl_pct = abs(actual_price - signal.sl_price) / actual_price * 100
        tp_pct = (signal.tp_price - actual_price) / actual_price * 100
        trail_act_pct = (signal.trailing_activation - actual_price) / actual_price * 100
        vol_ratio = signal.volume / signal.volume_ma if signal.volume_ma > 0 else 0

        self._telegram._send(
            f"⚡ *BUY déclenché – {symbol}* BREAKOUT MOMENTUM\n"
            f"  Entrée: `{_fmt(actual_price)}` | SL: `{_fmt(signal.sl_price)}` ({sl_pct:.1f}%)\n"
            f"  TP: `{_fmt(signal.tp_price)}` (+{tp_pct:.1f}%)\n"
            f"  Trail: activ `{_fmt(signal.trailing_activation)}` (+{trail_act_pct:.2f}%) | dist `{_fmt(signal.trailing_distance)}`\n"
            f"  Size: `{size:.8f} {base}` (`${size_usd:.2f}`)\n"
            f"  Risque: {BRK_RISK_PCT*100:.0f}% (`${risk_amount:.2f}`) | ATR: `{_fmt(signal.atr_value)}`\n"
            f"  Vol: `{signal.volume:.0f}` ({vol_ratio:.1f}× MA) | High({BRK_LOOKBACK}): `{_fmt(signal.recent_high)}`\n"
            f"  Fill: {fill_type} | Budget: `${BRK_ALLOCATED_BALANCE:.0f}`\n"
            f"[Dashboard]({DASHBOARD_URL})"
        )

        # ── Firebase ──
        try:
            fb_position = Position(
                symbol=symbol,
                side=OrderSide.BUY,
                entry_price=actual_price,
                sl_price=signal.sl_price,
                size=size,
                venue_order_id=venue_order_id,
                status=PositionStatus.OPEN,
                strategy=StrategyType.BREAKOUT,
                tp_price=signal.tp_price,
            )
            if not self.dry_run:
                fb_id = log_trade_opened(
                    position=fb_position,
                    fill_type=fill_type,
                    maker_wait_seconds=BRK_MAKER_WAIT_SECONDS,
                    risk_pct=BRK_RISK_PCT,
                    risk_amount_usd=risk_amount,
                    fiat_balance=available,
                    current_equity=BRK_ALLOCATED_BALANCE,
                    portfolio_risk_before=0.0,
                    exchange="revolut-breakout",
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
            logger.info("[DRY-RUN] BREAKOUT SELL %s | prix=%s | size=%.8f", symbol, _fmt(price), size)
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
                    logger.warning("[%s] ⚡ Maker no-fill → TAKER FALLBACK", symbol)
                    result = self._place_taker_order(order)

            fill_type = result.get("fill_type", "unknown")

            if fill_type == "no_fill":
                logger.error("[%s] ⚡ SELL taker fallback échoué — abandonné", symbol)
                return False

            logger.info(
                "[%s] ⚡ ✅ SELL exécuté | @ %s | size=%s | %s",
                symbol, price_str, size_str, fill_type,
            )
            return True
        except Exception as e:
            logger.error("[%s] ⚡ ❌ SELL échoué: %s", symbol, e)
            self._telegram.notify_error(f"⚡ SELL {symbol} échoué: {e}")
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
            order.side.value.upper(), order.symbol, order.price, BRK_MAKER_WAIT_SECONDS,
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
        logger.info("💰 MAKER-ONLY | ⏳ Attente %ds…", BRK_MAKER_WAIT_SECONDS)
        time.sleep(BRK_MAKER_WAIT_SECONDS)

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
            self._telegram.notify_error(f"🚫 ERREUR PERMANENTE {symbol} (BREAKOUT)\n{error[:200]}")
            return

        tiers = [60, 120, 300, 600, 1800]
        idx = min(fail_info["count"] - 1, len(tiers) - 1)
        cooldown = tiers[idx]
        fail_info["next_retry"] = time.time() + cooldown
        self._close_failures[symbol] = fail_info
        logger.warning("[%s] 🔁 Échec close #%d — retry dans %ds", symbol, fail_info["count"], cooldown)

        if fail_info["count"] == 1 or fail_info["count"] % 5 == 0:
            self._telegram.notify_error(
                f"⚠️ Close Breakout {symbol} échouée (×{fail_info['count']})\nRetry dans {cooldown}s\n{error[:200]}"
            )

    # ── Heartbeat ──────────────────────────────────────────────────────────

    def _maybe_heartbeat(self) -> None:
        now = time.time()
        heartbeat_seconds = get_heartbeat_override_seconds("breakout", BRK_HEARTBEAT_SECONDS)
        if heartbeat_seconds != self._heartbeat_seconds:
            self._heartbeat_seconds = heartbeat_seconds
            logger.info("💓 [BREAKOUT] Heartbeat runtime override: %ss", self._heartbeat_seconds)

        if now - self._last_heartbeat < self._heartbeat_seconds:
            return
        self._last_heartbeat = now

        # Countdown prochaine bougie 15m
        now_utc = datetime.now(timezone.utc)
        minutes = now_utc.minute
        next_15m_min = ((minutes // 15) + 1) * 15
        if next_15m_min >= 60:
            next_candle = now_utc.replace(minute=0, second=0, microsecond=0) + timedelta(hours=1)
        else:
            next_candle = now_utc.replace(minute=next_15m_min, second=0, microsecond=0)
        countdown_sec = max(0, int((next_candle - now_utc).total_seconds()))
        countdown_str = f"{countdown_sec // 60}m{countdown_sec % 60:02d}s"

        # Positions
        pos_lines: list[str] = []
        total_unrealized = 0.0
        for sym, pos in self._positions.items():
            try:
                ticker = self._data.get_ticker(sym)
                if not ticker:
                    continue
                price = ticker.last_price
                pnl = (price - pos.entry_price) * pos.size
                pnl_pct = (price - pos.entry_price) / pos.entry_price * 100
                total_unrealized += pnl
                base = sym.split("-")[0]
                icon = "🟢" if pnl >= 0 else "🔴"
                trail_tag = " 🔒TRAIL" if pos.trailing_active else ""
                sl_dist_pct = (price - pos.sl_price) / price * 100

                pos_lines.append(
                    f"  {icon} `{sym}` @ `{_fmt(pos.entry_price)}` → `{_fmt(price)}`\n"
                    f"    P&L `{pnl_pct:+.1f}%` (`{pnl:+.2f}$`) | SL `{_fmt(pos.sl_price)}` ({sl_dist_pct:.1f}%){trail_tag}"
                )
            except Exception:
                pass

        wr = self._total_wins / self._total_trades * 100 if self._total_trades > 0 else 0

        logger.info(
            "💓 BREAKOUT Alive | tick=%d | pos=%d/%d | budget=$%.0f | trades=%d (WR %.0f%%)",
            self._tick_count, len(self._positions), BRK_MAX_POSITIONS,
            BRK_ALLOCATED_BALANCE, self._total_trades, wr,
        )

        # Telegram
        try:
            if total_unrealized < -BRK_ALLOCATED_BALANCE * 0.05:
                sys_emoji, sys_label = "🔴", "risk mode"
            elif total_unrealized < 0 and self._positions:
                sys_emoji, sys_label = "🟡", "watching"
            else:
                sys_emoji, sys_label = "🟢", "stable"

            last_update = now_utc.strftime("%H:%M UTC")
            unr_emoji = "🟢" if total_unrealized >= 0 else "🔴"

            tg_lines = [
                f"{sys_emoji} *BREAKOUT MOMENTUM* ⚡ ({len(BRK_TRADING_PAIRS)} paires) — {sys_label}",
                f"  💰 Budget: `${BRK_ALLOCATED_BALANCE:,.0f}` (isolé)",
                f"  📊 Pos: `{len(self._positions)}/{BRK_MAX_POSITIONS}` | Trades: `{self._total_trades}` (WR `{wr:.0f}%`)",
                f"  ⚙️ high({BRK_LOOKBACK}) | TP {BRK_TP_ATR_MULT}×ATR | SL {BRK_SL_ATR_MULT}×ATR | trail {BRK_TRAIL_ACTIVATION_ATR}/{BRK_TRAIL_DISTANCE_ATR}×ATR",
            ]

            if self._positions:
                tg_lines.append(f"  {unr_emoji} PnL open: `${total_unrealized:+.2f}`")

            if pos_lines:
                tg_lines.append("")
                tg_lines.extend(pos_lines)

            tg_lines.append(f"\n  ⏳ Prochaine 15m: `{countdown_str}` ({next_candle.strftime('%H:%M')} UTC)")
            tg_lines.append(f"  🕐 `{last_update}`")
            tg_lines.append(f"[Dashboard]({DASHBOARD_URL})")
            self._telegram.send_raw("\n".join(tg_lines))
        except Exception:
            logger.warning("Telegram heartbeat failed", exc_info=True)

        # Firebase heartbeat
        if not self.dry_run:
            try:
                fb_log_heartbeat(
                    open_positions=len(self._positions),
                    total_equity=BRK_ALLOCATED_BALANCE,
                    total_risk_pct=0.0,
                    pairs_count=len(BRK_TRADING_PAIRS),
                    exchange="revolut-breakout",
                )
            except Exception:
                pass

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
                positions = []
                for sym, pos in self._positions.items():
                    positions.append({
                        "symbol": sym,
                        "entry_price": pos.entry_price,
                        "sl_price": pos.sl_price,
                        "tp_price": pos.tp_price,
                        "size": pos.size,
                        "trailing_active": pos.trailing_active,
                    })

                fb_log_daily_snapshot(
                    equity=BRK_ALLOCATED_BALANCE,
                    positions=positions,
                    daily_pnl=0.0,
                    trades_today=0,
                    exchange="revolut-breakout",
                    dry_run=self.dry_run,
                )
                logger.info("📸 Daily snapshot Firebase loggé")
            except Exception:
                pass

    # ── Helpers ────────────────────────────────────────────────────────────

    def _save_state(self) -> None:
        self._store.save(
            self._positions,
            self._last_candle_ts,
            self._candle_buffers,
            self._cooldowns,
            self._consecutive_losses,
            self._tilt_cooldowns,
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
        logger.info("🛑 Arrêt BreakoutMomentumBot...")
        self._save_state()
        logger.info("💾 État final sauvegardé")
        self._client.close()
        self._telegram.close()
        logger.info("BreakoutMomentumBot arrêté proprement")


# ── Point d'entrée ─────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="TradeX – Breakout Momentum Bot")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Mode simulation (log les ordres sans les exécuter)",
    )
    args = parser.parse_args()

    bot = BreakoutMomentumBot(dry_run=args.dry_run)
    signal.signal(signal.SIGTERM, lambda *_: bot.stop())
    bot.run()


if __name__ == "__main__":
    main()
