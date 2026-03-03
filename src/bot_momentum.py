"""
Bot Intraday Momentum Continuation — Revolut X, maker-only.

Timeframe M5, long-only, 7 paires.
Polling toutes les 30s (prix), détection M5 à chaque nouvelle bougie.

Usage :
    python -m src.bot_momentum              # Production
    python -m src.bot_momentum --dry-run    # Log les ordres sans les exécuter
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
from src.core.momentum_engine import (
    MCConfig,
    MCEntrySignal,
    MCSignalPhase,
    MCSignalState,
    MomentumEngine,
)
from src.core.risk_manager import calculate_position_size, get_fiat_balance, get_total_equity
from src.exchange.revolut_client import RevolutXClient
from src.exchange.data_provider import DataProvider
from src.notifications.telegram import TelegramNotifier
from src.firebase.trade_logger import (
    log_trade_opened,
    log_trade_closed,
    log_heartbeat as fb_log_heartbeat,
    log_event as fb_log_event,
    cleanup_old_events as fb_cleanup_events,
)

# ── Logging ────────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=getattr(logging, config.LOG_LEVEL, logging.INFO),
    format="[%(asctime)s] %(levelname)s %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("tradex.momentum_bot")

logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)


# ── Config Momentum (depuis .env) ─────────────────────────────────────────────

MC_TRADING_PAIRS: list[str] = [
    p.strip()
    for p in os.getenv("MC_TRADING_PAIRS", "ETH-USD,SOL-USD,BNB-USD,XRP-USD,LINK-USD,ADA-USD,LTC-USD").split(",")
    if p.strip()
]
MC_RISK_PERCENT: float = float(os.getenv("MC_RISK_PERCENT", "0.04"))
MC_MAX_POSITIONS: int = int(os.getenv("MC_MAX_POSITIONS", "3"))
MC_MAX_POSITION_PCT: float = float(os.getenv("MC_MAX_POSITION_PCT", "0.90"))
MC_POLLING_SECONDS: int = int(os.getenv("MC_POLLING_SECONDS", "30"))
MC_HEARTBEAT_SECONDS: int = int(os.getenv("MC_HEARTBEAT_SECONDS", "600"))
MC_MAKER_WAIT_SECONDS: int = int(os.getenv("MC_MAKER_WAIT_SECONDS", "60"))
MC_CANDLE_INTERVAL: int = 5  # M5 en minutes

# State file dédié (séparé du bot range)
MC_STATE_FILE: str = os.getenv(
    "MC_STATE_FILE",
    os.path.join(os.path.dirname(__file__), "..", "data", "state_momentum.json"),
)


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


# ── State store (simplifié pour le momentum bot) ──────────────────────────────

@dataclass
class MCPosition:
    """Position gérée par le bot momentum."""
    symbol: str
    side: str                       # "LONG"
    entry_price: float
    sl_price: float
    tp_price: float
    size: float                     # unités de base (ex: 0.05 ETH)
    size_usd: float                 # notionnel à l'entrée
    venue_order_id: str
    status: str = "OPEN"            # OPEN, TRAILING, CLOSED
    trailing_active: bool = False
    peak_price: float = 0.0
    pnl: Optional[float] = None
    firebase_trade_id: Optional[str] = None
    opened_at: float = 0.0         # timestamp

    def to_dict(self) -> dict:
        return {
            "symbol": self.symbol,
            "side": self.side,
            "entry_price": self.entry_price,
            "sl_price": self.sl_price,
            "tp_price": self.tp_price,
            "size": self.size,
            "size_usd": self.size_usd,
            "venue_order_id": self.venue_order_id,
            "status": self.status,
            "trailing_active": self.trailing_active,
            "peak_price": self.peak_price,
            "pnl": self.pnl,
            "firebase_trade_id": self.firebase_trade_id,
            "opened_at": self.opened_at,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "MCPosition":
        return cls(
            symbol=d["symbol"],
            side=d.get("side", "LONG"),
            entry_price=d["entry_price"],
            sl_price=d["sl_price"],
            tp_price=d["tp_price"],
            size=d["size"],
            size_usd=d.get("size_usd", 0.0),
            venue_order_id=d["venue_order_id"],
            status=d.get("status", "OPEN"),
            trailing_active=d.get("trailing_active", False),
            peak_price=d.get("peak_price", 0.0),
            pnl=d.get("pnl"),
            firebase_trade_id=d.get("firebase_trade_id"),
            opened_at=d.get("opened_at", 0.0),
        )


class MCStateStore:
    """Persistance atomique des positions + engine states."""

    def __init__(self, state_file: str = MC_STATE_FILE) -> None:
        self._path = Path(state_file).resolve()

    def save(
        self,
        positions: dict[str, MCPosition],
        signal_states: dict[str, MCSignalState],
    ) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        state = {
            "positions": {s: p.to_dict() for s, p in positions.items()},
            "signal_states": {
                s: {
                    "phase": st.phase.value,
                    "impulse_high": st.impulse_high,
                    "impulse_low": st.impulse_low,
                    "impulse_close": st.impulse_close,
                    "impulse_bar_idx": st.impulse_bar_idx,
                    "pullback_low": st.pullback_low,
                    "pullback_bars": st.pullback_bars,
                    "cooldown_until": st.cooldown_until,
                }
                for s, st in signal_states.items()
            },
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

    def load(self) -> tuple[dict[str, MCPosition], dict[str, MCSignalState]]:
        positions: dict[str, MCPosition] = {}
        signal_states: dict[str, MCSignalState] = {}
        if not self._path.exists():
            logger.info("📂 Pas de state momentum — démarrage à vide")
            return positions, signal_states
        try:
            with open(self._path, "r") as f:
                state = json.load(f)
            for sym, d in state.get("positions", {}).items():
                try:
                    positions[sym] = MCPosition.from_dict(d)
                except Exception as e:
                    logger.warning("⚠️ Position %s corrompue: %s", sym, e)
            for sym, d in state.get("signal_states", {}).items():
                try:
                    st = MCSignalState()
                    st.phase = MCSignalPhase(d.get("phase", "IDLE"))
                    st.impulse_high = d.get("impulse_high", 0.0)
                    st.impulse_low = d.get("impulse_low", 0.0)
                    st.impulse_close = d.get("impulse_close", 0.0)
                    st.impulse_bar_idx = d.get("impulse_bar_idx", 0)
                    st.pullback_low = d.get("pullback_low", 0.0)
                    st.pullback_bars = d.get("pullback_bars", 0)
                    st.cooldown_until = d.get("cooldown_until", 0)
                    signal_states[sym] = st
                except Exception as e:
                    logger.warning("⚠️ Signal state %s corrompu: %s", sym, e)
            logger.info(
                "📂 State momentum chargé: %d positions, %d signal states",
                len(positions), len(signal_states),
            )
        except Exception as e:
            logger.error("❌ Load state momentum échoué: %s", e)
        return positions, signal_states


# ── Bot principal ──────────────────────────────────────────────────────────────


class MomentumBot:
    """Bot Intraday Momentum Continuation — Revolut X maker-only."""

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

        # Engine
        self._engine = MomentumEngine(MCConfig(
            risk_per_trade=MC_RISK_PERCENT,
            max_positions=MC_MAX_POSITIONS,
            max_position_pct=MC_MAX_POSITION_PCT,
        ))

        # State
        self._store = MCStateStore()
        self._positions: dict[str, MCPosition] = {}

        # Candle tracking
        self._last_candle_ts: dict[str, int] = {}  # symbol → last M5 candle timestamp

        # Heartbeat
        self._last_heartbeat: float = 0.0
        self._cycle_count: int = 0

        # Daily cleanup
        self._last_cleanup_date: str = ""

        # Close failure tracking (backoff)
        self._close_failures: dict[str, dict] = {}

        if dry_run:
            logger.info("🔧 Mode DRY-RUN — aucun ordre ne sera exécuté")

    # ── Run ────────────────────────────────────────────────────────────────────

    def run(self) -> None:
        self._running = True
        logger.info("═" * 60)
        logger.info("🚀 MomentumBot démarré — Intraday Momentum Continuation")
        logger.info("   Paires     : %d | %s", len(MC_TRADING_PAIRS), ", ".join(MC_TRADING_PAIRS))
        logger.info("   Risque     : %.0f%% par trade | Max %d positions | Cap %.0f%%/pos",
                     MC_RISK_PERCENT * 100, MC_MAX_POSITIONS, MC_MAX_POSITION_PCT * 100)
        logger.info("   Polling    : %ds | Candle M%d", MC_POLLING_SECONDS, MC_CANDLE_INTERVAL)
        logger.info("   Exécution  : MAKER-ONLY (0%% fee) | attente %ds", MC_MAKER_WAIT_SECONDS)
        logger.info("   Mode       : LONG-ONLY")
        if self.dry_run:
            logger.info("   ⚠️  DRY-RUN actif")
        logger.info("═" * 60)

        self._initialize()

        try:
            while self._running:
                self._tick()
                time.sleep(MC_POLLING_SECONDS)
        except KeyboardInterrupt:
            logger.info("Arrêt demandé par l'utilisateur")
        finally:
            self._shutdown()

    def stop(self) -> None:
        self._running = False

    # ── Init ───────────────────────────────────────────────────────────────────

    def _initialize(self) -> None:
        """Charge l'état + bougies initiales."""
        # Charger l'état persisté
        loaded_pos, loaded_states = self._store.load()
        self._positions = loaded_pos

        # Restaurer les signal states dans l'engine
        for sym, st in loaded_states.items():
            self._engine._states[sym] = st

        # Réconcilier les positions avec l'exchange
        self._reconcile_positions()

        # Charger les bougies M5 initiales pour toutes les paires
        logger.info("── Chargement des bougies M5 initiales ──")
        for symbol in MC_TRADING_PAIRS:
            try:
                candles = self._client.get_candles(symbol, interval=MC_CANDLE_INTERVAL)
                candles.sort(key=lambda c: c.timestamp)
                if candles:
                    self._last_candle_ts[symbol] = candles[-1].timestamp
                    self._engine.update_candles(symbol, candles)
                    logger.info("[%s] %d bougies M5 chargées", symbol, len(candles))
                else:
                    logger.warning("[%s] Aucune bougie M5 reçue", symbol)
            except Exception as e:
                logger.error("[%s] ❌ Erreur chargement bougies: %s", symbol, e)

        open_count = sum(1 for p in self._positions.values() if p.status in ("OPEN", "TRAILING"))
        logger.info("── Init terminée | %d positions ouvertes ──", open_count)

    def _reconcile_positions(self) -> None:
        """Vérifie les positions contre les soldes exchange."""
        active = {s: p for s, p in self._positions.items() if p.status in ("OPEN", "TRAILING")}
        if not active:
            return

        try:
            balances = self._client.get_balances()
        except Exception as e:
            logger.warning("⚠️ Réconciliation impossible: %s — utilisation état local", e)
            return

        balance_map = {b.currency: b for b in balances}
        removed = []

        for sym, pos in active.items():
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

    # ── Tick ───────────────────────────────────────────────────────────────────

    def _tick(self) -> None:
        """Un cycle de polling."""
        self._cycle_count += 1

        for symbol in MC_TRADING_PAIRS:
            try:
                self._process_symbol(symbol)
            except Exception as e:
                logger.error("[%s] Erreur tick: %s", symbol, e)

        self._maybe_heartbeat()
        self._maybe_daily_cleanup()

    def _maybe_daily_cleanup(self) -> None:
        """Cleanup des events Firebase — 1×/jour UTC."""
        from datetime import datetime as dt, timezone as tz
        today = dt.now(tz.utc).strftime("%Y-%m-%d")
        if today != self._last_cleanup_date:
            self._last_cleanup_date = today
            try:
                fb_cleanup_events()
            except Exception:
                pass

    def _process_symbol(self, symbol: str) -> None:
        """Traite un symbole : check prix (SL/TP/trail) + check nouvelles bougies."""

        # ── 1. Gestion des positions ouvertes (SL/TP/Trail via ticker) ──
        pos = self._positions.get(symbol)
        if pos and pos.status in ("OPEN", "TRAILING"):
            ticker = self._data.get_ticker(symbol)
            if ticker:
                self._manage_position(symbol, pos, ticker)
                if pos.status not in ("OPEN", "TRAILING"):
                    return  # Position vient de fermer

        # ── 2. Vérifier s'il y a une nouvelle bougie M5 ──
        try:
            candles = self._client.get_candles(symbol, interval=MC_CANDLE_INTERVAL)
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

        # ── Nouvelle bougie M5 ──
        # Mettre à jour tout le buffer de bougies dans l'engine
        self._engine.update_candles(symbol, candles)

        # Pas de signal si on a déjà une position sur ce symbole
        has_position = pos is not None and pos.status in ("OPEN", "TRAILING")
        n_open = sum(1 for p in self._positions.values() if p.status in ("OPEN", "TRAILING"))

        if has_position:
            return

        # Traiter le signal via l'engine
        new_candle = candles[-1]
        entry_signal = self._engine.process_new_candle(
            symbol=symbol,
            candle=new_candle,
            n_open_positions=n_open,
            has_position=has_position,
        )

        if entry_signal:
            self._execute_entry(entry_signal)

    # ── Position management ────────────────────────────────────────────────────

    def _manage_position(self, symbol: str, pos: MCPosition, ticker: TickerData) -> None:
        """Gère SL, TP, trailing stop pour une position ouverte."""
        price = ticker.last_price

        # ── SL check ──
        if self._engine.check_sl_hit(price, pos.sl_price):
            logger.info("[%s] 🛑 SL HIT | prix=%s | SL=%s", symbol, _fmt(price), _fmt(pos.sl_price))
            self._close_position(symbol, price, "SL atteint")
            return

        # ── TP check ──
        if self._engine.check_tp_hit(price, pos.tp_price):
            logger.info("[%s] 🎯 TP HIT | prix=%s | TP=%s", symbol, _fmt(price), _fmt(pos.tp_price))
            self._close_position(symbol, price, "TP atteint")
            return

        # ── Trailing stop ──
        new_sl, new_peak, trail_active = self._engine.check_trailing_stop(
            entry_price=pos.entry_price,
            current_price=price,
            current_sl=pos.sl_price,
            peak_price=pos.peak_price or pos.entry_price,
            trailing_active=pos.trailing_active,
        )

        changed = False
        if trail_active and not pos.trailing_active:
            pos.trailing_active = True
            pos.status = "TRAILING"
            logger.info("[%s] 📈 TRAILING activé | prix=%s | newSL=%s", symbol, _fmt(price), _fmt(new_sl))
            self._telegram._send(
                f"📈 *Trailing activé – {symbol}* ⚡\n"
                f"  Prix: `{_fmt(price)}` | Nouveau SL: `{_fmt(new_sl)}`\n"
                f"  Entrée: `{_fmt(pos.entry_price)}`"
            )
            changed = True

        if new_sl > pos.sl_price:
            pos.sl_price = new_sl
            changed = True

        if new_peak > (pos.peak_price or 0):
            pos.peak_price = new_peak
            changed = True

        if changed:
            self._save_state()

    # ── Entry execution ────────────────────────────────────────────────────────

    def _execute_entry(self, signal: MCEntrySignal) -> None:
        """Exécute un signal d'entrée via l'API Revolut X (maker-only)."""
        symbol = signal.symbol

        # ── Guard: position existante ──
        existing = self._positions.get(symbol)
        if existing and existing.status in ("OPEN", "TRAILING"):
            logger.debug("[%s] Position déjà ouverte, skip", symbol)
            return

        # ── Guard: max positions ──
        n_open = sum(1 for p in self._positions.values() if p.status in ("OPEN", "TRAILING"))
        if n_open >= MC_MAX_POSITIONS:
            logger.debug("[%s] Max positions atteint (%d/%d)", symbol, n_open, MC_MAX_POSITIONS)
            return

        # ── Sizing ──
        try:
            balances = self._data.get_balances()
            fiat_balance, fiat_currency = get_fiat_balance(balances)
        except Exception as e:
            logger.error("[%s] ❌ Impossible de récupérer le solde: %s", symbol, e)
            return

        # Besoin de fiat pour BUY
        quote_currency = symbol.split("-")[1]
        quote_balance = next((b for b in balances if b.currency == quote_currency), None)
        available_quote = quote_balance.available if quote_balance else 0.0
        if available_quote <= 0:
            logger.info("[%s] ⏭️ Pas de %s disponible — skip", symbol, quote_currency)
            return

        # Utiliser le solde de la devise de cotation
        sizing_balance = available_quote

        size = calculate_position_size(
            account_balance=sizing_balance,
            risk_percent=MC_RISK_PERCENT,
            entry_price=signal.entry_price,
            sl_price=signal.sl_price,
            max_position_percent=MC_MAX_POSITION_PCT,
        )

        if size <= 0:
            logger.warning("[%s] Taille de position invalide (0)", symbol)
            return

        cost = size * signal.entry_price
        if cost > sizing_balance:
            size = sizing_balance / signal.entry_price
            logger.info("[%s] Taille ajustée au solde: %.8f", symbol, size)

        # ── Format price avec assez de décimales ──
        price_str = self._format_order_price(signal.entry_price)
        size_str = f"{size:.8f}"

        order = OrderRequest(
            symbol=symbol,
            side=OrderSide.BUY,
            base_size=size_str,
            price=price_str,
        )

        # ── Exécution maker-only ──
        venue_order_id = "dry-run"
        fill_type = "dry-run"

        if not self.dry_run:
            try:
                result = self._place_maker_only_order(order)
                venue_order_id = result.get("venue_order_id", "unknown")
                fill_type = result.get("fill_type", "unknown")

                if fill_type == "no_fill":
                    logger.info(
                        "[%s] ⏭️ Maker-only: pas de fill après %ds — signal abandonné",
                        symbol, MC_MAKER_WAIT_SECONDS,
                    )
                    # 🔥 Firebase log événement
                    try:
                        fb_log_event(
                            event_type="maker_no_fill",
                            data={"entry_price": signal.entry_price, "wait_s": MC_MAKER_WAIT_SECONDS},
                            symbol=symbol,
                        )
                    except Exception:
                        pass
                    return

                logger.info(
                    "[%s] ✅ Ordre MAKER exécuté | %s @ %s | size=%s",
                    symbol, fill_type, price_str, size_str,
                )
            except Exception as e:
                logger.error("[%s] ❌ Échec placement maker-only: %s", symbol, e)
                self._telegram.notify_error(f"Ordre momentum {symbol} échoué: {e}")
                return
        else:
            logger.info(
                "[DRY-RUN] MOMENTUM BUY %s | entry=%s | SL=%s | TP=%s | size=%s",
                symbol, _fmt(signal.entry_price), _fmt(signal.sl_price),
                _fmt(signal.tp_price), size_str,
            )

        # ── Enregistrer la position ──
        pos = MCPosition(
            symbol=symbol,
            side="LONG",
            entry_price=signal.entry_price,
            sl_price=signal.sl_price,
            tp_price=signal.tp_price,
            size=size,
            size_usd=cost,
            venue_order_id=venue_order_id,
            status="OPEN",
            peak_price=signal.entry_price,
            opened_at=time.time(),
        )
        self._positions[symbol] = pos
        self._save_state()

        # ── Notification Telegram ──
        risk_amount = sizing_balance * MC_RISK_PERCENT
        sl_pct = abs(signal.entry_price - signal.sl_price) / signal.entry_price * 100
        tp_pct = abs(signal.tp_price - signal.entry_price) / signal.entry_price * 100
        base = symbol.split("-")[0]

        self._telegram._send(
            f"⚡ *BUY déclenché – {symbol}* 🚀 MOMENTUM\n"
            f"  Entrée: `{_fmt(signal.entry_price)}` | SL: `{_fmt(signal.sl_price)}` ({sl_pct:.1f}%)\n"
            f"  TP: `{_fmt(signal.tp_price)}` ({tp_pct:.1f}%) | Size: `{size:.8f} {base}`\n"
            f"  Risque: {MC_RISK_PERCENT*100:.0f}% ({risk_amount:.2f} USD)\n"
            f"  Impulse: body={signal.impulse_body_pct*100:.2f}% vol={signal.impulse_vol_ratio:.1f}x\n"
            f"  Retrace: {signal.retrace_pct*100:.1f}%"
        )

        # ── Firebase ──
        try:
            # Adapter au format Position pour Firebase
            fb_position = Position(
                symbol=symbol,
                side=OrderSide.BUY,
                entry_price=signal.entry_price,
                sl_price=signal.sl_price,
                size=size,
                venue_order_id=venue_order_id,
                status=PositionStatus.OPEN,
                strategy=StrategyType.MOMENTUM,
                tp_price=signal.tp_price,
            )
            fb_id = log_trade_opened(
                position=fb_position,
                fill_type=fill_type,
                maker_wait_seconds=MC_MAKER_WAIT_SECONDS,
                risk_pct=MC_RISK_PERCENT,
                risk_amount_usd=risk_amount,
                fiat_balance=sizing_balance,
                current_equity=sizing_balance,  # approximation
                portfolio_risk_before=0.0,
            )
            if fb_id:
                pos.firebase_trade_id = fb_id
                self._save_state()
        except Exception as e:
            logger.warning("🔥 Firebase log_trade_opened échoué: %s", e)

    # ── Close position ─────────────────────────────────────────────────────────

    def _close_position(self, symbol: str, exit_price: float, reason: str) -> None:
        """Ferme une position via un ordre SELL maker-only."""
        pos = self._positions.get(symbol)
        if pos is None:
            return

        # ── Backoff check ──
        fail_info = self._close_failures.get(symbol)
        if fail_info:
            if fail_info.get("permanent"):
                return
            if time.time() < fail_info.get("next_retry", 0):
                return

        # ── Vérifier le solde réel ──
        exit_size = pos.size
        if not self.dry_run:
            try:
                base_currency = symbol.split("-")[0]
                balances = self._data.get_balances()
                base_bal = next((b for b in balances if b.currency == base_currency), None)
                real_available = base_bal.available if base_bal else 0.0
                real_total = (base_bal.available + base_bal.reserved) if base_bal else 0.0

                if real_total <= 0:
                    logger.warning("[%s] 👻 Position fantôme — solde = 0 → purge", symbol)
                    del self._positions[symbol]
                    self._save_state()
                    return

                # Annuler les ordres actifs pour libérer le reserved
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

                # Re-fetch balances après cancel
                if real_available < pos.size * 0.90:
                    balances = self._data.get_balances()
                    base_bal = next((b for b in balances if b.currency == base_currency), None)
                    real_available = base_bal.available if base_bal else 0.0

                if real_available < pos.size:
                    exit_size = real_available
                    logger.info("[%s] 📐 Taille ajustée: %.8f → %.8f", symbol, pos.size, exit_size)
            except Exception as e:
                logger.warning("[%s] ⚠️ Balance check échoué: %s", symbol, e)

        # ── Construire l'ordre SELL ──
        price_str = self._format_order_price(exit_price)
        size_str = f"{exit_size:.8f}"

        exit_order = OrderRequest(
            symbol=symbol,
            side=OrderSide.SELL,
            base_size=size_str,
            price=price_str,
        )

        fill_type = "dry-run"
        actual_exit_price = exit_price

        if not self.dry_run:
            try:
                result = self._place_maker_only_order(exit_order)
                fill_type = result.get("fill_type", "unknown")
                actual_exit_price = result.get("actual_price", exit_price)

                if fill_type == "no_fill":
                    # Pour les clôtures SL/TP, on retente avec un taker pour ne pas rater la sortie
                    logger.warning(
                        "[%s] ⚠️ Maker-only close: pas de fill — TAKER FALLBACK pour SL/TP",
                        symbol,
                    )
                    try:
                        result = self._place_taker_order(exit_order)
                        fill_type = result.get("fill_type", "taker")
                        actual_exit_price = result.get("actual_price", exit_price)
                    except Exception as e2:
                        logger.error("[%s] ❌ Taker fallback échoué aussi: %s", symbol, e2)
                        self._handle_close_failure(symbol, str(e2))
                        return

            except Exception as e:
                logger.error("[%s] ❌ Échec clôture: %s", symbol, e)
                self._handle_close_failure(symbol, str(e))
                return
        else:
            logger.info("[DRY-RUN] CLOSE %s | prix=%s | reason=%s", symbol, _fmt(exit_price), reason)

        # ── Succès ──
        if symbol in self._close_failures:
            del self._close_failures[symbol]

        # PnL
        pnl_gross = (actual_exit_price - pos.entry_price) * exit_size
        pnl_pct = pnl_gross / pos.size_usd * 100 if pos.size_usd > 0 else 0
        pnl_emoji = "🟢" if pnl_gross >= 0 else "🔴"

        logger.info(
            "[%s] %s CLOSE | entry=%s | exit=%s | PnL=$%+.2f (%+.1f%%) | %s | %s",
            symbol, pnl_emoji, _fmt(pos.entry_price), _fmt(actual_exit_price),
            pnl_gross, pnl_pct, reason, fill_type.upper(),
        )

        pos.status = "CLOSED"
        pos.pnl = pnl_gross
        self._save_state()

        # Engine record
        self._engine.record_trade_result(pnl_gross)

        # Telegram
        self._telegram._send(
            f"{pnl_emoji} *Position fermée – {symbol}* ⚡ MOMENTUM\n"
            f"  Raison: {reason}\n"
            f"  Entrée: `{_fmt(pos.entry_price)}` → Sortie: `{_fmt(actual_exit_price)}`\n"
            f"  P&L: `{pnl_gross:+.2f} USD` ({pnl_pct:+.1f}%)"
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
                    strategy=StrategyType.MOMENTUM,
                    tp_price=pos.tp_price,
                    pnl=pnl_gross,
                )
                log_trade_closed(
                    trade_id=pos.firebase_trade_id,
                    position=fb_position,
                    exit_price=actual_exit_price,
                    reason=reason,
                    fill_type=fill_type,
                    equity_after=0.0,
                    actual_exit_size=exit_size,
                )
            except Exception as e:
                logger.warning("🔥 Firebase log_trade_closed échoué: %s", e)

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
            logger.critical("[%s] 🚫 Erreur permanente — intervention manuelle requise: %s", symbol, error[:200])
            self._telegram.notify_error(f"🚫 ERREUR PERMANENTE {symbol} (MOMENTUM)\nAction: vendre manuellement\n{error[:200]}")
            return

        tiers = [60, 120, 300, 600, 1800]
        idx = min(fail_info["count"] - 1, len(tiers) - 1)
        cooldown = tiers[idx]
        fail_info["next_retry"] = time.time() + cooldown
        self._close_failures[symbol] = fail_info
        logger.warning("[%s] 🔁 Échec close #%d — retry dans %ds", symbol, fail_info["count"], cooldown)

        if fail_info["count"] == 1 or fail_info["count"] % 5 == 0:
            self._telegram.notify_error(
                f"⚠️ Close momentum {symbol} échouée (×{fail_info['count']})\nRetry dans {cooldown}s\n{error[:200]}"
            )

    # ── Maker-only order execution ─────────────────────────────────────────────

    def _place_maker_only_order(self, order: OrderRequest) -> dict:
        """Place un ordre limit passif (maker 0%). Si pas rempli → annule (PAS de fallback taker).

        Pour les ENTRY: on abandonne le trade si pas de fill maker.
        Pour les CLOSE (SL/TP): l'appelant gère le fallback taker.
        """
        logger.info(
            "💰 MAKER-ONLY | %s %s @ %s (attente %ds)",
            order.side.value.upper(), order.symbol, order.price, MC_MAKER_WAIT_SECONDS,
        )

        # 1. Placer l'ordre passif
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

        # 2. Attendre
        logger.info("💰 MAKER-ONLY | ⏳ Attente %ds…", MC_MAKER_WAIT_SECONDS)
        time.sleep(MC_MAKER_WAIT_SECONDS)

        # 3. Vérifier
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

        # FILLED
        if status == "FILLED" or (filled_size > 0 and filled_size >= total_size * 0.99):
            logger.info("💰 MAKER-ONLY | ✅ Rempli — fee 0%%")
            return {"venue_order_id": venue_order_id, "fill_type": "maker", "actual_price": float(order.price)}

        # Partial fill
        if filled_size > 0:
            logger.info("💰 MAKER-ONLY | ⚡ Fill partiel (%.8f/%.8f) — cancel reste", filled_size, total_size)
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
            # 409 = peut-être rempli entre-temps
            if "409" in str(e) or "conflict" in str(e).lower():
                try:
                    recheck = self._client.get_order(venue_order_id)
                    rd_raw = recheck.get("data", recheck) if isinstance(recheck, dict) else recheck
                    if isinstance(rd_raw, list) and rd_raw:
                        rd_raw = rd_raw[0]
                    rd: dict = rd_raw if isinstance(rd_raw, dict) else {}
                    if (rd.get("status") or rd.get("state") or "").upper() == "FILLED":
                        logger.info("💰 MAKER-ONLY | ✅ Rempli entre-temps!")
                        return {"venue_order_id": venue_order_id, "fill_type": "maker", "actual_price": float(order.price)}
                except Exception:
                    pass

        return {"venue_order_id": venue_order_id, "fill_type": "no_fill", "actual_price": float(order.price)}

    def _place_taker_order(self, order: OrderRequest) -> dict:
        """Place un ordre taker agressif (pour les clôtures SL/TP uniquement)."""
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

    # ── Heartbeat ──────────────────────────────────────────────────────────────

    def _maybe_heartbeat(self) -> None:
        now = time.time()
        if now - self._last_heartbeat < MC_HEARTBEAT_SECONDS:
            return
        self._last_heartbeat = now

        open_positions = [p for p in self._positions.values() if p.status in ("OPEN", "TRAILING")]

        # Equity
        total_equity = 0.0
        try:
            balances = self._data.get_balances()
            fiat_balance, _ = get_fiat_balance(balances)
            fiat_set = {"USD", "EUR", "GBP"}
            crypto_tickers = []
            for b in balances:
                if b.total > 0 and b.currency not in fiat_set:
                    try:
                        t = self._data.get_ticker(f"{b.currency}-USD")
                        if t:
                            crypto_tickers.append(t)
                    except Exception:
                        pass
            total_equity = get_total_equity(balances, crypto_tickers)
        except Exception:
            pass

        # Signal states
        active_signals = sum(
            1 for st in self._engine._states.values()
            if st.phase != MCSignalPhase.IDLE
        )

        logger.info(
            "💓 MOMENTUM Alive | cycle=%d | positions=%d/%d | signals=%d | equity=$%.2f",
            self._cycle_count, len(open_positions), MC_MAX_POSITIONS,
            active_signals, total_equity,
        )

        # Détail positions
        pos_lines: list[str] = []
        for pos in open_positions:
            try:
                ticker = self._data.get_ticker(pos.symbol)
                if not ticker:
                    continue
                price = ticker.last_price
                pnl = (price - pos.entry_price) * pos.size
                pnl_pct = (price - pos.entry_price) / pos.entry_price * 100
                sl_dist = (price - pos.sl_price) / price * 100
                tp_dist = (pos.tp_price - price) / price * 100
                icon = "🟢" if pnl >= 0 else "🔴"
                trail_tag = "TRAIL" if pos.trailing_active else "OPEN"

                logger.info(
                    "   %s %s %s | %s @ %s → %s | P&L %+.2f%% ($%+.2f) | SL %s (%+.1f%%) | TP %s (%+.1f%%)",
                    icon, trail_tag, pos.symbol,
                    pos.side, _fmt(pos.entry_price), _fmt(price),
                    pnl_pct, pnl, _fmt(pos.sl_price), -sl_dist, _fmt(pos.tp_price), tp_dist,
                )
                pos_lines.append(
                    f"  {icon} {trail_tag} `{pos.symbol}` @ `{_fmt(pos.entry_price)}` → `{_fmt(price)}` | "
                    f"P&L `{pnl_pct:+.1f}%` (`{pnl:+.2f}$`) | SL `{_fmt(pos.sl_price)}`"
                )
            except Exception:
                pass

        # Telegram heartbeat
        try:
            tg_lines = [
                f"💓 *MOMENTUM Alive*",
                f"  Cycle: `{self._cycle_count}` | Pos: `{len(open_positions)}/{MC_MAX_POSITIONS}`",
                f"  Signaux actifs: `{active_signals}` | Equity: `${total_equity:,.2f}`",
            ]
            if pos_lines:
                tg_lines.append("")
                tg_lines.extend(pos_lines)
            self._telegram._send("\n".join(tg_lines))
        except Exception:
            logger.warning("Telegram heartbeat failed", exc_info=True)

        # Firebase heartbeat
        try:
            fb_log_heartbeat(
                open_positions=len(open_positions),
                total_equity=total_equity,
                total_risk_pct=0.0,
                pairs_count=len(MC_TRADING_PAIRS),
            )
        except Exception:
            pass

    # ── Helpers ────────────────────────────────────────────────────────────────

    def _save_state(self) -> None:
        self._store.save(self._positions, self._engine._states)

    @staticmethod
    def _format_order_price(price: float) -> str:
        """Formate un prix pour l'API (assez de décimales)."""
        if price >= 1000:
            return f"{price:.2f}"
        elif price >= 1:
            return f"{price:.4f}"
        elif price >= 0.01:
            return f"{price:.6f}"
        else:
            return f"{price:.8f}"

    def _shutdown(self) -> None:
        logger.info("🛑 Arrêt MomentumBot...")
        self._save_state()
        logger.info("💾 État final sauvegardé")
        self._client.close()
        self._telegram.close()
        logger.info("MomentumBot arrêté proprement")


# ── Point d'entrée ─────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="TradeX – Momentum Continuation Bot")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Mode simulation (log les ordres sans les exécuter)",
    )
    args = parser.parse_args()

    bot = MomentumBot(dry_run=args.dry_run)
    signal.signal(signal.SIGTERM, lambda *_: bot.stop())
    bot.run()


if __name__ == "__main__":
    main()
