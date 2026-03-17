"""
Bot Infinity — DCA inversé multi-paires sur Revolut X, maker-only.

Paires : BTC-USD, AAVE-USD, XLM-USD (configs walk-forward validées)
Timeframe H4, polling toutes les 30s, gestion sur bougie H4.

Stratégie par paire :
  1. Calcul trailing high (N bougies H4, configurable)
  2. Si prix chute ≥ drop_pct du trailing high → premier achat
  3. DCA : 5 paliers d'achat (configurables par paire)
  4. Vente : 5 paliers progressifs au-dessus du PMP
  5. Breakeven stop après TP1
  6. Stop-loss configurable par paire

Capital : pool global (INF_CAPITAL_PCT du solde Revolut X), réparti dynamiquement
entre paires actives pour éviter le capital bloqué sur les paires inactives.

Usage :
    python -m src.bot_infinity              # Production
    python -m src.bot_infinity --dry-run    # Log les ordres sans les exécuter
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
from src.core.infinity_engine import (
    InfinityConfig,
    InfinityCycle,
    InfinityPhase,
    InfinityExitReason,
    InfinityBuyLevel,
    InfinitySellLevel,
    rsi_series,
    sma_series,
    check_first_entry,
    compute_buy_size,
    check_buy_conditions,
    check_sell_conditions,
    check_stop_loss,
    check_override_sell,
)
from src.exchange.revolut_client import RevolutXClient
from src.exchange.data_provider import DataProvider
from src.notifications.telegram import TelegramNotifier
from src.firebase.trade_logger import (
    log_trade_opened,
    log_trade_closed,
    log_heartbeat as fb_log_heartbeat,
    log_event as fb_log_event,
    log_daily_snapshot as fb_log_daily_snapshot,
    cleanup_old_events as fb_cleanup_events,
)
from src.firebase.client import add_document as fb_add_document
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
logger = logging.getLogger("tradex.infinity_bot")

logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)


# ── Config Infinity ────────────────────────────────────────────────────────────

INF_POLLING_SECONDS: int = config.INF_POLLING_SECONDS
INF_HEARTBEAT_SECONDS: int = config.INF_HEARTBEAT_SECONDS
INF_MAKER_WAIT_SECONDS: int = config.INF_MAKER_WAIT_SECONDS
INF_CAPITAL_PCT: float = config.INF_CAPITAL_PCT
INF_CAPITAL_ACTIVE_SLOTS: int = max(1, config.INF_CAPITAL_ACTIVE_SLOTS)

# ── Per-pair validated configs ────────────────────────────────────────────────
# Walk-forward validated: train 2020→2024, test 2024→2026
# BTC  : defaults (déjà live) — backtest 6y: PF 4.07, +912$/333$
# AAVE : OPT_C (sell×2, no BE) — backtest 6y: PF 5.84, +3975$/333$
# XLM  : OPT_A (sell×1.25, BE TP2) — backtest 6y: PF 11.18, +1125$/333$
# ADA  : scanner grid (6y) — Score 183, PF 3.02, +93%/333$, DD 10.3%
# DOT  : scanner grid (6y) — Score 168, PF 3.02, +86%/333$, DD 12.5%
# LTC  : scanner grid (6y) — Score 176, PF 3.57, +57%/333$, DD 7.5%

PAIR_CONFIGS: dict[str, InfinityConfig] = {
    "BTC-USD": InfinityConfig(
        trailing_high_period=72,         # 12 jours
        entry_drop_pct=0.05,             # -5%
        buy_levels=(-0.05, -0.10, -0.15, -0.20, -0.25),
        buy_pcts=(0.25, 0.20, 0.15, 0.10, 0.00),
        sell_levels=(0.008, 0.015, 0.022, 0.030, 0.040),
        stop_loss_pct=0.15,
        max_invested_pct=0.70,
        first_entry_rsi_max=50.0,
        use_breakeven_stop=True,
        scale_with_equity=True,
        rsi_sell_min=0.0,
        maker_fee=0.0,
        taker_fee=0.0009,
    ),
    "AAVE-USD": InfinityConfig(
        trailing_high_period=48,         # 8 jours
        entry_drop_pct=0.12,             # -12%
        buy_levels=(-0.12, -0.20, -0.28, -0.35, -0.42),
        buy_pcts=(0.25, 0.20, 0.15, 0.10, 0.00),
        sell_levels=(0.040, 0.080, 0.120, 0.160, 0.240),  # OPT_C: ×2 vs ancien
        stop_loss_pct=0.25,
        max_invested_pct=0.70,
        first_entry_rsi_max=50.0,
        use_breakeven_stop=False,        # OPT_C: pas de BE
        scale_with_equity=True,
        rsi_sell_min=0.0,
        maker_fee=0.0,
        taker_fee=0.0009,
    ),
    "XLM-USD": InfinityConfig(
        trailing_high_period=48,         # 8 jours
        entry_drop_pct=0.12,             # -12%
        buy_levels=(-0.12, -0.20, -0.28, -0.35, -0.42),
        buy_pcts=(0.25, 0.20, 0.15, 0.10, 0.00),
        sell_levels=(0.010, 0.01875, 0.0275, 0.0375, 0.050),  # OPT_A: ×1.25 vs ancien
        stop_loss_pct=0.25,
        max_invested_pct=0.70,
        first_entry_rsi_max=50.0,
        use_breakeven_stop=True,
        breakeven_after_level=1,         # OPT_A: BE après TP2
        scale_with_equity=True,
        rsi_sell_min=0.0,
        maker_fee=0.0,
        taker_fee=0.0009,
    ),
    "ADA-USD": InfinityConfig(
        trailing_high_period=72,         # 12 jours
        entry_drop_pct=0.15,             # -15%
        buy_levels=(-0.15, -0.22, -0.30, -0.38, -0.45),
        buy_pcts=(0.25, 0.20, 0.15, 0.10, 0.00),
        sell_levels=(0.010, 0.020, 0.030, 0.045, 0.060),  # s2
        stop_loss_pct=0.20,
        max_invested_pct=0.70,
        first_entry_rsi_max=50.0,
        use_breakeven_stop=True,
        scale_with_equity=True,
        rsi_sell_min=0.0,
        maker_fee=0.0,
        taker_fee=0.0009,
    ),
    "DOT-USD": InfinityConfig(
        trailing_high_period=48,         # 8 jours
        entry_drop_pct=0.15,             # -15%
        buy_levels=(-0.15, -0.22, -0.30, -0.38, -0.45),
        buy_pcts=(0.25, 0.20, 0.15, 0.10, 0.00),
        sell_levels=(0.020, 0.040, 0.060, 0.080, 0.120),  # s4
        stop_loss_pct=0.20,
        max_invested_pct=0.70,
        first_entry_rsi_max=50.0,
        use_breakeven_stop=True,
        scale_with_equity=True,
        rsi_sell_min=0.0,
        maker_fee=0.0,
        taker_fee=0.0009,
    ),
    "LTC-USD": InfinityConfig(
        trailing_high_period=48,         # 8 jours
        entry_drop_pct=0.15,             # -15%
        buy_levels=(-0.15, -0.22, -0.30, -0.38, -0.45),
        buy_pcts=(0.25, 0.20, 0.15, 0.10, 0.00),
        sell_levels=(0.010, 0.020, 0.030, 0.045, 0.060),  # s2
        stop_loss_pct=0.15,
        max_invested_pct=0.70,
        first_entry_rsi_max=50.0,
        use_breakeven_stop=True,
        scale_with_equity=True,
        rsi_sell_min=0.0,
        maker_fee=0.0,
        taker_fee=0.0009,
    ),
}

# Trading pairs (from .env or default to all validated pairs)
INF_TRADING_PAIRS: list[str] = [
    p.strip()
    for p in os.getenv("INF_TRADING_PAIRS", ",".join(PAIR_CONFIGS.keys())).split(",")
]

H4_INTERVAL = 240  # H4 en minutes


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


# ── Live Cycle State ──────────────────────────────────────────────────────────


@dataclass
class InfLiveCycle:
    """État d'un cycle DCA live (persisté en JSON)."""
    phase: str = "WAITING"              # WAITING, ACCUMULATING, DISTRIBUTING
    reference_price: float = 0.0        # Prix de référence (trailing high au start)

    # Buys effectués
    buys: list = field(default_factory=list)  # [{level, price, size, cost, ts}]
    total_size: float = 0.0
    total_cost: float = 0.0
    pmp: float = 0.0

    # Sells effectués
    sells: list = field(default_factory=list)  # [{level, price, size, proceeds, ts}]
    size_remaining: float = 0.0
    total_proceeds: float = 0.0
    sell_levels_hit: list = field(default_factory=list)  # indices

    # State
    breakeven_active: bool = False
    cycle_start_ts: float = 0.0

    # Firebase tracking
    firebase_trade_ids: list = field(default_factory=list)

    def recalc_pmp(self) -> None:
        if self.total_size > 0:
            self.pmp = self.total_cost / self.total_size

    def to_dict(self) -> dict:
        return {
            "phase": self.phase,
            "reference_price": self.reference_price,
            "buys": self.buys,
            "total_size": self.total_size,
            "total_cost": self.total_cost,
            "pmp": self.pmp,
            "sells": self.sells,
            "size_remaining": self.size_remaining,
            "total_proceeds": self.total_proceeds,
            "sell_levels_hit": self.sell_levels_hit,
            "breakeven_active": self.breakeven_active,
            "cycle_start_ts": self.cycle_start_ts,
            "firebase_trade_ids": self.firebase_trade_ids,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "InfLiveCycle":
        c = cls()
        c.phase = d.get("phase", "WAITING")
        c.reference_price = d.get("reference_price", 0.0)
        c.buys = d.get("buys", [])
        c.total_size = d.get("total_size", 0.0)
        c.total_cost = d.get("total_cost", 0.0)
        c.pmp = d.get("pmp", 0.0)
        c.sells = d.get("sells", [])
        c.size_remaining = d.get("size_remaining", 0.0)
        c.total_proceeds = d.get("total_proceeds", 0.0)
        c.sell_levels_hit = d.get("sell_levels_hit", [])
        c.breakeven_active = d.get("breakeven_active", False)
        c.cycle_start_ts = d.get("cycle_start_ts", 0.0)
        c.firebase_trade_ids = d.get("firebase_trade_ids", [])
        return c


# ── Per-pair context ───────────────────────────────────────────────────────────


@dataclass
class PairContext:
    """All state for one trading pair."""
    symbol: str
    config: InfinityConfig
    cycle: InfLiveCycle
    store: "InfinityStateStore"
    candle_highs: list[float] = field(default_factory=list)
    last_candle_ts: int = 0
    cycle_count: int = 0
    consecutive_stops: int = 0
    last_eval: dict = field(default_factory=dict)


# ── State store ────────────────────────────────────────────────────────────────


class InfinityStateStore:
    """Persistance atomique du cycle + candle tracking pour une paire."""

    def __init__(self, state_file: str) -> None:
        self._path = Path(state_file).resolve()

    def save(
        self,
        cycle: InfLiveCycle,
        candle_highs: list[float],
        last_candle_ts: int,
        cycle_count: int,
        consecutive_stops: int,
    ) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        state = {
            "cycle": cycle.to_dict(),
            "candle_highs": candle_highs[-200:],  # garder 200 dernières
            "last_candle_ts": last_candle_ts,
            "cycle_count": cycle_count,
            "consecutive_stops": consecutive_stops,
        }
        tmp = self._path.with_suffix(".json.tmp")
        try:
            with open(tmp, "w") as f:
                json.dump(state, f, indent=2)
            tmp.replace(self._path)
        except Exception as e:
            logger.error("❌ Save failed (%s): %s", self._path.name, e)
            if tmp.exists():
                try:
                    tmp.unlink()
                except OSError:
                    pass

    def load(self) -> dict:
        if not self._path.exists():
            logger.info("📂 Pas de state — démarrage à vide (%s)", self._path.name)
            return {}
        try:
            with open(self._path, "r") as f:
                return json.load(f)
        except Exception as e:
            logger.error("❌ Load state échoué (%s): %s", self._path.name, e)
            return {}


# ── Bot principal ──────────────────────────────────────────────────────────────


class InfinityBot:
    """Bot DCA inversé multi-paires — Revolut X, maker-only, H4."""

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

        # Build per-pair contexts
        self._pairs: dict[str, PairContext] = {}
        data_dir = os.path.join(os.path.dirname(__file__), "..", "data")
        for symbol in INF_TRADING_PAIRS:
            cfg = PAIR_CONFIGS.get(symbol)
            if not cfg:
                logger.warning("⚠️ Pas de config validée pour %s — skip", symbol)
                continue
            safe_name = symbol.replace("-", "_")
            state_file = os.path.join(data_dir, f"state_infinity_{safe_name}.json")
            self._pairs[symbol] = PairContext(
                symbol=symbol,
                config=cfg,
                cycle=InfLiveCycle(),
                store=InfinityStateStore(state_file),
            )

        # Migrate old single-pair state file → BTC-USD per-pair file
        old_state = os.path.join(data_dir, "state_infinity.json")
        new_btc = os.path.join(data_dir, "state_infinity_BTC_USD.json")
        if os.path.exists(old_state) and not os.path.exists(new_btc) and "BTC-USD" in self._pairs:
            try:
                import shutil
                shutil.copy2(old_state, new_btc)
                logger.info("📦 State migré: state_infinity.json → state_infinity_BTC_USD.json")
            except Exception as e:
                logger.warning("⚠️ Migration state échouée: %s", e)

        # Heartbeat
        self._last_heartbeat: float = 0.0
        self._heartbeat_seconds: int = INF_HEARTBEAT_SECONDS
        self._tick_count: int = 0

        # Daily cleanup / snapshot
        self._last_cleanup_date: str = ""
        self._last_snapshot_date: str = ""

        # Close failure tracking
        self._close_failures: dict[str, dict] = {}

        if dry_run:
            logger.info("🔧 Mode DRY-RUN — aucun ordre ne sera exécuté")

    # ── Run ────────────────────────────────────────────────────────────────────

    def run(self) -> None:
        self._running = True

        logger.info("═" * 60)
        logger.info("♾️  InfinityBot démarré — DCA inversé multi-paires")
        logger.info("   Paires    : %s", ", ".join(self._pairs.keys()))
        logger.info("   Capital   : %.0f%% du solde Revolut X (pool dynamique)", INF_CAPITAL_PCT * 100)
        logger.info("   Polling   : %ds | Maker wait: %ds",
                     INF_POLLING_SECONDS, INF_MAKER_WAIT_SECONDS)
        for symbol, ctx in self._pairs.items():
            cfg = ctx.config
            buy_lvls = ", ".join(f"{x*100:.0f}%" for x in cfg.buy_levels)
            sell_lvls = ", ".join(f"+{x*100:.1f}%" for x in cfg.sell_levels)
            logger.info("   ── %s ──", symbol)
            logger.info("      Trail: %d bars (%dj) | Drop: %.0f%% | SL: %.0f%%",
                         cfg.trailing_high_period,
                         cfg.trailing_high_period * 4 // 24,
                         cfg.entry_drop_pct * 100, cfg.stop_loss_pct * 100)
            logger.info("      Buy : %s", buy_lvls)
            logger.info("      Sell: %s", sell_lvls)
        if self.dry_run:
            logger.info("   ⚠️  DRY-RUN actif")
        logger.info("═" * 60)

        self._initialize()

        try:
            while self._running:
                self._tick()
                time.sleep(INF_POLLING_SECONDS)
        except KeyboardInterrupt:
            logger.info("Arrêt demandé par l'utilisateur")
        finally:
            self._shutdown()

    def stop(self) -> None:
        self._running = False

    # ── Init ───────────────────────────────────────────────────────────────────

    def _initialize(self) -> None:
        """Charge l'état persisté + bougies initiales pour chaque paire."""
        for symbol, ctx in self._pairs.items():
            self._initialize_pair(ctx)
        logger.info("── Init terminée | %d paires actives ──", len(self._pairs))

    def _initialize_pair(self, ctx: PairContext) -> None:
        """Initialise une paire : charge state + bougies H4."""
        symbol = ctx.symbol

        # Charger l'état persisté
        state = ctx.store.load()
        if state:
            cycle_data = state.get("cycle")
            if cycle_data:
                ctx.cycle = InfLiveCycle.from_dict(cycle_data)
            ctx.candle_highs = state.get("candle_highs", [])
            ctx.last_candle_ts = state.get("last_candle_ts", 0)
            ctx.cycle_count = state.get("cycle_count", 0)
            ctx.consecutive_stops = state.get("consecutive_stops", 0)
            logger.info(
                "[%s] 📂 State chargé: phase=%s, cycle=%d, buys=%d, sells=%d, stops_consec=%d",
                symbol, ctx.cycle.phase, ctx.cycle_count,
                len(ctx.cycle.buys), len(ctx.cycle.sells),
                ctx.consecutive_stops,
            )

        # Charger les bougies H4 initiales
        try:
            candles = self._client.get_candles(symbol, interval=H4_INTERVAL)
            candles.sort(key=lambda c: c.timestamp)
            if candles:
                ctx.candle_highs = [c.high for c in candles]
                ctx.last_candle_ts = candles[-1].timestamp
                logger.info(
                    "[%s] %d bougies H4 chargées | dernier high=%s",
                    symbol, len(candles), _fmt(candles[-1].high),
                )
            else:
                logger.warning("[%s] Aucune bougie H4 reçue", symbol)
        except Exception as e:
            logger.error("[%s] ❌ Erreur chargement bougies: %s", symbol, e)

        # Réconciliation si cycle actif
        if ctx.cycle.phase != "WAITING" and ctx.cycle.size_remaining > 0:
            self._reconcile_position(ctx)

    def _reconcile_position(self, ctx: PairContext) -> None:
        """Vérifie le solde base currency contre le cycle actif."""
        try:
            balances = self._client.get_balances()
            base_currency = ctx.symbol.split("-")[0]
            base_bal = next((b for b in balances if b.currency == base_currency), None)
            held = (base_bal.available + base_bal.reserved) if base_bal else 0.0

            if held >= ctx.cycle.size_remaining * 0.90:
                logger.info(
                    "[%s] ✅ Position confirmée | %.8f %s (attendu %.8f)",
                    ctx.symbol, held, base_currency, ctx.cycle.size_remaining,
                )
            else:
                logger.warning(
                    "[%s] ⚠️ Solde insuffisant: %.8f %s, attendu: %.8f",
                    ctx.symbol, held, base_currency, ctx.cycle.size_remaining,
                )
        except Exception as e:
            logger.warning("[%s] ⚠️ Réconciliation impossible: %s", ctx.symbol, e)

    # ── Tick ───────────────────────────────────────────────────────────────────

    def _tick(self) -> None:
        """Un cycle de polling — itère sur toutes les paires."""
        self._apply_runtime_actions()
        self._tick_count += 1

        for symbol, ctx in self._pairs.items():
            try:
                self._tick_pair(ctx)
            except Exception as e:
                logger.error("[%s] Erreur tick: %s", symbol, e, exc_info=True)

        self._maybe_heartbeat()
        self._maybe_daily_tasks()

    def _apply_runtime_actions(self) -> None:
        actions = get_pending_runtime_actions("infinity")
        if not actions:
            return

        for action in actions:
            action_id = str(action.get("_id", ""))
            kind = str(action.get("action", "")).lower().strip()
            symbol = str(action.get("symbol", "")).upper().strip()

            try:
                if kind == "close":
                    targets = list(self._pairs.keys()) if symbol == "ALL" else [symbol]
                    closed = 0
                    for sym in targets:
                        ctx = self._pairs.get(sym)
                        if not ctx:
                            continue
                        if ctx.cycle.phase == "WAITING" or ctx.cycle.size_remaining <= 0:
                            continue
                        ticker = self._data.get_ticker(sym)
                        if not ticker:
                            continue
                        self._close_cycle(ctx, ticker.last_price, InfinityExitReason.OVERRIDE_SELL)
                        closed += 1

                    mark_runtime_action_status(action_id, "done", f"manual close appliqué ({closed})")
                    try:
                        fb_log_event(
                            "MANUAL_ACTION",
                            {"action": "close", "symbol": symbol, "count": closed},
                            exchange="revolut-infinity",
                        )
                    except Exception:
                        pass
                    continue

                if kind in ("set_sl", "set_tp"):
                    mark_runtime_action_status(action_id, "failed", "non supporté pour infinity")
                    continue

                mark_runtime_action_status(action_id, "failed", "action inconnue")
            except Exception as e:
                mark_runtime_action_status(action_id, "failed", f"erreur: {e}")

    def _tick_pair(self, ctx: PairContext) -> None:
        """Tick pour une paire spécifique."""
        # 1. Récupérer le prix actuel
        ticker = self._data.get_ticker(ctx.symbol)
        if not ticker:
            return
        price = ticker.last_price

        # 2. Vérifier les nouvelles bougies H4
        new_candle = self._check_new_candle(ctx)

        # 3. Gérer le cycle actif
        if ctx.cycle.phase == "WAITING":
            if new_candle:
                self._try_first_entry(ctx, price)
        elif ctx.cycle.phase in ("ACCUMULATING", "DISTRIBUTING"):
            self._manage_cycle(ctx, price, new_candle)

    def _check_new_candle(self, ctx: PairContext) -> bool:
        """Vérifie s'il y a une nouvelle bougie H4. Retourne True si oui."""
        try:
            candles = self._client.get_candles(ctx.symbol, interval=H4_INTERVAL)
            candles.sort(key=lambda c: c.timestamp)
        except Exception as e:
            logger.debug("[%s] Candle fetch failed: %s", ctx.symbol, e)
            return False

        if not candles:
            return False

        latest_ts = candles[-1].timestamp
        if latest_ts <= ctx.last_candle_ts:
            return False

        # Nouvelle bougie H4
        ctx.last_candle_ts = latest_ts
        ctx.candle_highs = [c.high for c in candles]
        self._save_state(ctx)

        logger.debug(
            "[%s] Nouvelle bougie H4 | close=%s | high=%s | vol=%.2f",
            ctx.symbol, _fmt(candles[-1].close), _fmt(candles[-1].high), candles[-1].volume,
        )
        return True

    # ── Trailing high ──────────────────────────────────────────────────────────

    def _get_trailing_high(self, ctx: PairContext) -> float:
        """Calcule le trailing high sur les N dernières bougies H4."""
        period = ctx.config.trailing_high_period
        if not ctx.candle_highs:
            return 0.0
        window = ctx.candle_highs[-period:] if len(ctx.candle_highs) >= period else ctx.candle_highs
        return max(window)

    # ── RSI calculation ────────────────────────────────────────────────────────

    def _get_current_rsi(self, ctx: PairContext) -> float:
        """Calcule le RSI courant à partir des closes H4."""
        try:
            candles = self._client.get_candles(ctx.symbol, interval=H4_INTERVAL)
            candles.sort(key=lambda c: c.timestamp)
            if len(candles) < ctx.config.rsi_period + 1:
                return 50.0
            closes = [c.close for c in candles]
            rsi_vals = rsi_series(closes, ctx.config.rsi_period)
            return rsi_vals[-1]
        except Exception:
            return 50.0

    # ── Volume MA ──────────────────────────────────────────────────────────────

    def _get_volume_data(self, ctx: PairContext) -> tuple[float, float]:
        """Retourne (volume_courant, volume_ma20)."""
        try:
            candles = self._client.get_candles(ctx.symbol, interval=H4_INTERVAL)
            candles.sort(key=lambda c: c.timestamp)
            if not candles:
                return 0.0, 0.0
            volumes = [c.volume for c in candles]
            ma_vals = sma_series(volumes, ctx.config.volume_ma_len)
            return volumes[-1], ma_vals[-1]
        except Exception:
            return 0.0, 0.0

    # ── Allocated capital ──────────────────────────────────────────────────────

    def _is_pair_active(self, ctx: PairContext) -> bool:
        """Une paire est active si un cycle est en cours avec position restante."""
        return ctx.cycle.phase != "WAITING" and ctx.cycle.size_remaining > 0

    def _get_active_pairs_count(self) -> int:
        """Nombre de paires Infinity actuellement actives."""
        return sum(1 for pair_ctx in self._pairs.values() if self._is_pair_active(pair_ctx))

    def _get_dynamic_pair_capital_pct(self, ctx: PairContext) -> float:
        """Part du pool allouée à une paire avec logique de slots fixes.

        - Le pool Infinity est partagé sur `INF_CAPITAL_ACTIVE_SLOTS` positions max.
        - Chaque slot reçoit une part fixe du pool (ex: 2 slots => 50/50).
        - Si tous les slots sont occupés, les nouvelles paires ont 0 allocation.
        """
        active_count = self._get_active_pairs_count()
        slot_pct = INF_CAPITAL_PCT / INF_CAPITAL_ACTIVE_SLOTS

        if self._is_pair_active(ctx):
            return slot_pct

        if active_count >= INF_CAPITAL_ACTIVE_SLOTS:
            return 0.0

        return slot_pct

    def _get_allocated_balance(self, ctx: PairContext) -> float:
        """Retourne le capital alloué à cette paire (part dynamique du pool global)."""
        try:
            balances = self._client.get_balances()
            quote = ctx.symbol.split("-")[1]  # USD
            usd_bal = next((b for b in balances if b.currency == quote), None)
            available = usd_bal.available if usd_bal else 0.0
            capital_pct = self._get_dynamic_pair_capital_pct(ctx)

            # Si cycle actif, l'equity = cash dispo + valeur position
            if ctx.cycle.phase != "WAITING" and ctx.cycle.size_remaining > 0:
                ticker = self._data.get_ticker(ctx.symbol)
                position_value = ctx.cycle.size_remaining * ticker.last_price if ticker else 0
                total = available + position_value
            else:
                total = available

            return total * capital_pct
        except Exception as e:
            logger.warning("[%s] ⚠️ Impossible de calculer le solde alloué: %s", ctx.symbol, e)
            return 0.0

    def _get_cash_available(self, ctx: PairContext) -> float:
        """Retourne le cash USD disponible × part dynamique de la paire."""
        try:
            balances = self._client.get_balances()
            quote = ctx.symbol.split("-")[1]
            usd_bal = next((b for b in balances if b.currency == quote), None)
            available = usd_bal.available if usd_bal else 0.0
            capital_pct = self._get_dynamic_pair_capital_pct(ctx)
            return available * capital_pct
        except Exception:
            return 0.0

    def _get_scoped_equity(self) -> float:
        """Equity scope Infinity: USD + valorisation des positions bot ouvertes.

        Exclut les avoirs externes/staking, même si le ticker appartient à l'univers
        Infinity.
        """
        try:
            balances = self._client.get_balances()
            usd_total = next((b.total for b in balances if b.currency == "USD"), 0.0)

            positions_value = 0.0
            for symbol, ctx in self._pairs.items():
                cycle = ctx.cycle
                if cycle.phase == "WAITING" or cycle.size_remaining <= 0:
                    continue
                try:
                    ticker = self._data.get_ticker(symbol)
                    if ticker:
                        positions_value += cycle.size_remaining * ticker.last_price
                except Exception:
                    continue

            return usd_total + positions_value
        except Exception:
            return 0.0

    # ── First entry ────────────────────────────────────────────────────────────

    def _try_first_entry(self, ctx: PairContext, price: float) -> None:
        """Tente le premier achat d'un nouveau cycle pour la paire."""
        cfg = ctx.config

        # Safety: max consecutive stops
        if ctx.consecutive_stops >= cfg.max_consecutive_stops:
            logger.warning(
                "[%s] ♾️ %d stops consécutifs → pause (max=%d)",
                ctx.symbol, ctx.consecutive_stops, cfg.max_consecutive_stops,
            )
            return

        active_count = self._get_active_pairs_count()
        if active_count >= INF_CAPITAL_ACTIVE_SLOTS:
            logger.info(
                "[%s] ♾️ Slots capital pleins (%d/%d) — skip entry",
                ctx.symbol, active_count, INF_CAPITAL_ACTIVE_SLOTS,
            )
            return

        trailing_high = self._get_trailing_high(ctx)
        if trailing_high <= 0:
            return

        rsi = self._get_current_rsi(ctx)
        volume, volume_ma = self._get_volume_data(ctx)

        # Evaluate each condition individually for diagnostics
        drop = (trailing_high - price) / trailing_high if trailing_high > 0 else 0
        drop_ok = drop >= cfg.entry_drop_pct
        rsi_ok = rsi <= cfg.first_entry_rsi_max
        vol_ok = True
        if cfg.require_volume_entry:
            vol_ok = volume_ma > 0 and volume > volume_ma

        # Build reason string
        reasons = []
        if not drop_ok:
            reasons.append(f"drop {drop*100:.1f}% < {cfg.entry_drop_pct*100:.1f}%")
        if not rsi_ok:
            reasons.append(f"RSI {rsi:.1f} > {cfg.first_entry_rsi_max:.0f}")
        if not vol_ok:
            reasons.append(f"vol {volume:.0f} <= MA {volume_ma:.0f}")

        all_ok = drop_ok and rsi_ok and vol_ok

        # Store evaluation for heartbeat display
        ctx.last_eval = {
            "ts": datetime.now(timezone.utc).isoformat()[:16],
            "price": price,
            "drop_pct": drop * 100,
            "drop_ok": drop_ok,
            "rsi": rsi,
            "rsi_ok": rsi_ok,
            "volume": volume,
            "volume_ma": volume_ma,
            "vol_ok": vol_ok,
            "result": "ENTRY" if all_ok else "SKIP",
            "reason": ", ".join(reasons) if reasons else "all conditions met",
        }

        # Log evaluation result on each new candle
        drop_icon = "✅" if drop_ok else "❌"
        rsi_icon = "✅" if rsi_ok else "❌"
        vol_icon = "✅" if vol_ok else "⬜" if not cfg.require_volume_entry else ("✅" if vol_ok else "❌")
        logger.info(
            "[%s] ♾️ 🕯️ ÉVALUATION H4 | prix=%s | résultat=%s\n"
            "   %s Drop: %.1f%% (seuil: %.1f%%)\n"
            "   %s RSI: %.1f (max: %.0f)\n"
            "   %s Volume: %.0f vs MA %.0f",
            ctx.symbol, _fmt(price), ctx.last_eval["result"],
            drop_icon, drop * 100, cfg.entry_drop_pct * 100,
            rsi_icon, rsi, cfg.first_entry_rsi_max,
            vol_icon, volume, volume_ma,
        )

        if not all_ok:
            return

        drop_pct = (trailing_high - price) / trailing_high * 100
        logger.info(
            "[%s] ♾️ ENTRY SIGNAL | prix=%s | trail_high=%s | drop=%.1f%% | RSI=%.1f",
            ctx.symbol, _fmt(price), _fmt(trailing_high), drop_pct, rsi,
        )

        # Calculer le montant du premier achat
        allocated = self._get_allocated_balance(ctx)
        if allocated <= 0:
            logger.warning("[%s] ♾️ Pas de capital alloué — skip", ctx.symbol)
            return

        # Premier palier : buy_pcts[0] de l'equity allouée
        target_amount = allocated * cfg.buy_pcts[0]

        buy_amount = compute_buy_size(
            rsi=rsi,
            rsi_full=cfg.rsi_full_buy,
            rsi_half=cfg.rsi_half_buy,
            target_amount=target_amount,
            cash_available=self._get_cash_available(ctx),
            max_invested=allocated * cfg.max_invested_pct,
            already_invested=0.0,
        )

        if buy_amount <= 10:  # min $10
            logger.info("[%s] ♾️ Montant trop faible ($%.2f) — skip", ctx.symbol, buy_amount)
            return

        # Exécuter l'achat
        size = buy_amount / price
        success = self._execute_buy(ctx, price, size, buy_amount, level=0)

        if success:
            # Démarrer un nouveau cycle
            ctx.cycle = InfLiveCycle(
                phase="ACCUMULATING",
                reference_price=trailing_high,
                buys=[{"level": 0, "price": price, "size": size, "cost": buy_amount, "ts": time.time()}],
                total_size=size,
                total_cost=buy_amount,
                pmp=price,
                size_remaining=size,
                cycle_start_ts=time.time(),
            )
            ctx.cycle_count += 1
            ctx.consecutive_stops = 0
            self._save_state(ctx)

            logger.info(
                "[%s] ♾️ CYCLE #%d démarré | ref=%s | buy L1 @ %s | size=%.8f ($%.2f)",
                ctx.symbol, ctx.cycle_count, _fmt(trailing_high), _fmt(price), size, buy_amount,
            )

            # Telegram
            self._telegram.notify_infinity_buy(
                symbol=ctx.symbol,
                level=0,
                price=price,
                size=size,
                cost_usd=buy_amount,
                pmp=price,
                total_invested=buy_amount,
                equity=allocated,
            )

            # Firebase
            self._log_buy_firebase(ctx, price, size, buy_amount, allocated)

    # ── Manage active cycle ────────────────────────────────────────────────────

    def _manage_cycle(self, ctx: PairContext, price: float, new_candle: bool) -> None:
        """Gère un cycle actif : check SL, sells, additional buys."""
        cycle = ctx.cycle
        cfg = ctx.config

        # ── 1. Stop-loss check (toujours, pas que sur nouvelle bougie) ──
        if cycle.pmp > 0 and check_stop_loss(price, cycle.pmp, cfg.stop_loss_pct):
            logger.info(
                "[%s] ♾️ 🛑 STOP LOSS | prix=%s | PMP=%s | SL=%s",
                ctx.symbol, _fmt(price), _fmt(cycle.pmp),
                _fmt(cycle.pmp * (1 - cfg.stop_loss_pct)),
            )
            self._close_cycle(ctx, price, InfinityExitReason.STOP_LOSS)
            return

        # ── 2. Breakeven stop check ──
        if cycle.breakeven_active and cycle.pmp > 0 and price <= cycle.pmp:
            logger.info(
                "[%s] ♾️ 🔒 BREAKEVEN STOP | prix=%s | PMP=%s",
                ctx.symbol, _fmt(price), _fmt(cycle.pmp),
            )
            self._close_cycle(ctx, price, InfinityExitReason.STOP_LOSS)
            return

        # ── 3. Override sell (+20% du PMP) ──
        if check_override_sell(price, cycle.pmp, cfg.override_sell_pct):
            logger.info(
                "[%s] ♾️ 🚀 OVERRIDE SELL | prix=%s | PMP=%s | +%.1f%%",
                ctx.symbol, _fmt(price), _fmt(cycle.pmp),
                (price - cycle.pmp) / cycle.pmp * 100,
            )
            self._close_cycle(ctx, price, InfinityExitReason.OVERRIDE_SELL)
            return

        # ── 4. Check sell levels (paliers de vente) ──
        if cycle.size_remaining > 0:
            for i, sell_pct in enumerate(cfg.sell_levels):
                if i in cycle.sell_levels_hit:
                    continue

                if check_sell_conditions(
                    close=price,
                    pmp=cycle.pmp,
                    sell_level_pct=sell_pct,
                    rsi=50.0,  # RSI sell gate désactivé
                    rsi_sell_min=cfg.rsi_sell_min,
                ):
                    # Vendre 20% du total (ou le reste pour le dernier)
                    sell_pct_size = cfg.sell_pcts[i] if i < len(cfg.sell_pcts) else 0.2
                    sell_size = cycle.total_size * sell_pct_size

                    # Dernier palier → vendre tout le reste
                    if i == len(cfg.sell_levels) - 1 or sell_size > cycle.size_remaining:
                        sell_size = cycle.size_remaining

                    if sell_size <= 0:
                        continue

                    proceeds = sell_size * price
                    success = self._execute_sell(ctx, price, sell_size, proceeds, level=i)

                    if success:
                        cycle.sells.append({
                            "level": i, "price": price,
                            "size": sell_size, "proceeds": proceeds,
                            "ts": time.time(),
                        })
                        cycle.size_remaining -= sell_size
                        cycle.total_proceeds += proceeds
                        cycle.sell_levels_hit.append(i)

                        gain_pct = (price - cycle.pmp) / cycle.pmp * 100
                        logger.info(
                            "[%s] ♾️ TP%d HIT | prix=%s | PMP=%s | +%.2f%% | sold=%.8f ($%.2f) | remaining=%.8f",
                            ctx.symbol, i + 1, _fmt(price), _fmt(cycle.pmp),
                            gain_pct, sell_size, proceeds, cycle.size_remaining,
                        )

                        # Breakeven après le premier TP
                        if (cfg.use_breakeven_stop
                                and i >= cfg.breakeven_after_level
                                and not cycle.breakeven_active):
                            cycle.breakeven_active = True
                            logger.info("[%s] ♾️ 🔒 BREAKEVEN activé après TP%d", ctx.symbol, i + 1)

                        self._save_state(ctx)

                        # Telegram
                        self._telegram.notify_infinity_sell(
                            symbol=ctx.symbol,
                            level=i,
                            price=price,
                            size=sell_size,
                            proceeds_usd=proceeds,
                            pmp=cycle.pmp,
                            pnl_pct=gain_pct,
                            remaining_size=cycle.size_remaining,
                        )

                        # Cycle complet ?
                        if cycle.size_remaining <= 0:
                            self._complete_cycle(ctx)
                            return

                    break  # un seul palier par tick

        # ── 5. Check additional buys (DCA, seulement sur nouvelle bougie) ──
        if new_candle and cycle.phase == "ACCUMULATING":
            self._try_additional_buy(ctx, price)

    # ── Additional DCA buy ─────────────────────────────────────────────────────

    def _try_additional_buy(self, ctx: PairContext, price: float) -> None:
        """Tente un achat DCA additionnel si un palier est atteint."""
        cycle = ctx.cycle
        cfg = ctx.config
        filled_levels = {b["level"] for b in cycle.buys}

        for i, drop_pct in enumerate(cfg.buy_levels):
            if i in filled_levels:
                continue
            if i == 0:
                continue  # Premier achat déjà fait

            # Vérifier si le prix est sous le palier
            target_price = cycle.reference_price * (1 + drop_pct)  # drop_pct is negative
            if price > target_price:
                continue

            # RSI + volume check
            rsi = self._get_current_rsi(ctx)
            volume, volume_ma = self._get_volume_data(ctx)

            if not check_buy_conditions(
                close=price,
                pmp=cycle.pmp,
                rsi=rsi,
                rsi_half=cfg.rsi_half_buy,
                volume=volume,
                volume_ma=volume_ma,
            ):
                continue

            # Calculer le montant
            allocated = self._get_allocated_balance(ctx)
            pct = cfg.buy_pcts[i] if i < len(cfg.buy_pcts) else 0.10
            target_amount = allocated * pct if pct > 0 else (allocated - cycle.total_cost)

            buy_amount = compute_buy_size(
                rsi=rsi,
                rsi_full=cfg.rsi_full_buy,
                rsi_half=cfg.rsi_half_buy,
                target_amount=target_amount,
                cash_available=self._get_cash_available(ctx),
                max_invested=allocated * cfg.max_invested_pct,
                already_invested=cycle.total_cost,
            )

            if buy_amount <= 10:
                continue

            size = buy_amount / price
            success = self._execute_buy(ctx, price, size, buy_amount, level=i)

            if success:
                cycle.buys.append({
                    "level": i, "price": price,
                    "size": size, "cost": buy_amount,
                    "ts": time.time(),
                })
                cycle.total_size += size
                cycle.total_cost += buy_amount
                cycle.size_remaining += size
                cycle.recalc_pmp()
                self._save_state(ctx)

                logger.info(
                    "[%s] ♾️ BUY L%d | prix=%s | size=%.8f ($%.2f) | PMP=%s | total_inv=$%.2f",
                    ctx.symbol, i + 1, _fmt(price), size, buy_amount,
                    _fmt(cycle.pmp), cycle.total_cost,
                )

                # Telegram
                self._telegram.notify_infinity_buy(
                    symbol=ctx.symbol,
                    level=i,
                    price=price,
                    size=size,
                    cost_usd=buy_amount,
                    pmp=cycle.pmp,
                    total_invested=cycle.total_cost,
                    equity=allocated,
                )

                # Firebase
                self._log_buy_firebase(ctx, price, size, buy_amount, allocated)

            break  # un seul achat par tick

    # ── Cycle completion ───────────────────────────────────────────────────────

    def _complete_cycle(self, ctx: PairContext) -> None:
        """Cycle terminé : tous les TP atteints."""
        cycle = ctx.cycle
        pnl_usd = cycle.total_proceeds - cycle.total_cost
        pnl_pct = pnl_usd / cycle.total_cost * 100 if cycle.total_cost > 0 else 0

        logger.info(
            "[%s] ♾️ ✅ CYCLE #%d COMPLET | PMP=%s | Inv=$%.2f → Rec=$%.2f | PnL=$%+.2f (%+.1f%%)",
            ctx.symbol, ctx.cycle_count, _fmt(cycle.pmp), cycle.total_cost,
            cycle.total_proceeds, pnl_usd, pnl_pct,
        )

        self._telegram.notify_infinity_cycle_complete(
            symbol=ctx.symbol,
            pmp=cycle.pmp,
            total_cost=cycle.total_cost,
            total_proceeds=cycle.total_proceeds,
            pnl_usd=pnl_usd,
            pnl_pct=pnl_pct,
            n_buys=len(cycle.buys),
            n_sells=len(cycle.sells),
        )

        # Firebase : close all trade docs
        self._log_cycle_close_firebase(ctx, cycle.pmp, pnl_usd, "TP_COMPLETE")

        # Reset cycle
        ctx.cycle = InfLiveCycle()
        self._save_state(ctx)

    def _close_cycle(self, ctx: PairContext, price: float, reason: InfinityExitReason) -> None:
        """Ferme le cycle entier : vend tout le restant."""
        cycle = ctx.cycle
        if cycle.size_remaining <= 0:
            ctx.cycle = InfLiveCycle()
            self._save_state(ctx)
            return

        # Sell tout le restant
        proceeds = cycle.size_remaining * price
        is_stop = reason == InfinityExitReason.STOP_LOSS

        success = self._execute_sell(
            ctx, price, cycle.size_remaining, proceeds,
            level=-1, use_taker=is_stop,
        )

        if not success:
            logger.error("[%s] ♾️ ❌ Échec clôture cycle — retry au prochain tick", ctx.symbol)
            return

        cycle.total_proceeds += proceeds
        pnl_usd = cycle.total_proceeds - cycle.total_cost
        pnl_pct = pnl_usd / cycle.total_cost * 100 if cycle.total_cost > 0 else 0

        logger.info(
            "[%s] ♾️ %s CYCLE #%d | %s | PMP=%s | PnL=$%+.2f (%+.1f%%)",
            ctx.symbol,
            "🛑" if is_stop else "🚀",
            ctx.cycle_count, reason.value, _fmt(cycle.pmp), pnl_usd, pnl_pct,
        )

        if is_stop:
            ctx.consecutive_stops += 1
            self._telegram.notify_infinity_stop(
                symbol=ctx.symbol,
                price=price,
                pmp=cycle.pmp,
                total_cost=cycle.total_cost,
                proceeds=cycle.total_proceeds,
                pnl_usd=pnl_usd,
                pnl_pct=pnl_pct,
            )
        else:
            ctx.consecutive_stops = 0
            self._telegram.notify_infinity_cycle_complete(
                symbol=ctx.symbol,
                pmp=cycle.pmp,
                total_cost=cycle.total_cost,
                total_proceeds=cycle.total_proceeds,
                pnl_usd=pnl_usd,
                pnl_pct=pnl_pct,
                n_buys=len(cycle.buys),
                n_sells=len(cycle.sells),
            )

        # Firebase
        self._log_cycle_close_firebase(ctx, price, pnl_usd, reason.value)

        # Reset
        ctx.cycle = InfLiveCycle()
        self._save_state(ctx)

    # ── Order execution ────────────────────────────────────────────────────────

    def _execute_buy(self, ctx: PairContext, price: float, size: float, cost_usd: float, level: int) -> bool:
        """Exécute un achat via Revolut X.

        Stratégie d'exécution :
        1. Maker (0% fee) avec le prix initial → attente 60s
        2. Si no-fill : rafraîchit le prix, retry maker → attente 60s
        3. Si toujours no-fill : taker fallback (0.09% fee) à prix rafraîchi
        """
        if self.dry_run:
            logger.info(
                "[DRY-RUN] INFINITY BUY L%d | %s @ %s | size=%.8f ($%.2f)",
                level + 1, ctx.symbol, _fmt(price), size, cost_usd,
            )
            return True

        MAX_MAKER_RETRIES = 2

        current_price = price
        for attempt in range(1, MAX_MAKER_RETRIES + 1):
            price_str = self._format_order_price(current_price)
            size_str = f"{cost_usd / current_price:.8f}"

            order = OrderRequest(
                symbol=ctx.symbol,
                side=OrderSide.BUY,
                base_size=size_str,
                price=price_str,
            )

            try:
                result = self._place_maker_only_order(order)
                fill_type = result.get("fill_type", "unknown")

                if fill_type != "no_fill":
                    logger.info(
                        "[%s] ♾️ ✅ BUY L%d exécuté | @ %s | size=%s | maker (tentative %d)",
                        ctx.symbol, level + 1, price_str, size_str, attempt,
                    )
                    return True

                # Maker no-fill → rafraîchir le prix pour la suite
                logger.info(
                    "[%s] ♾️ BUY L%d: maker no-fill (tentative %d/%d)",
                    ctx.symbol, level + 1, attempt, MAX_MAKER_RETRIES,
                )
                try:
                    ticker = self._data.get_ticker(ctx.symbol)
                    if ticker:
                        current_price = ticker.last_price
                except Exception:
                    pass

            except Exception as e:
                logger.error("[%s] ♾️ ❌ BUY L%d maker échoué (tentative %d): %s",
                             ctx.symbol, level + 1, attempt, e)
                break

        # ── Taker fallback (prix rafraîchi) ──
        logger.warning(
            "[%s] ♾️ BUY L%d: %d makers no-fill → TAKER FALLBACK @ %s",
            ctx.symbol, level + 1, MAX_MAKER_RETRIES, _fmt(current_price),
        )
        price_str = self._format_order_price(current_price)
        size_str = f"{cost_usd / current_price:.8f}"
        order = OrderRequest(
            symbol=ctx.symbol,
            side=OrderSide.BUY,
            base_size=size_str,
            price=price_str,
        )

        try:
            result = self._place_taker_order(order)
            fill_type = result.get("fill_type", "unknown")

            if fill_type == "no_fill":
                logger.error(
                    "[%s] ♾️ BUY L%d: taker fallback échoué aussi — abandonné", ctx.symbol, level + 1,
                )
                try:
                    fb_log_event(
                        event_type="infinity_maker_no_fill",
                        data={"level": level, "price": current_price, "side": "BUY", "retries": MAX_MAKER_RETRIES},
                        symbol=ctx.symbol,
                    )
                except Exception:
                    pass
                return False

            logger.info(
                "[%s] ♾️ ✅ BUY L%d exécuté | @ %s | size=%s | taker fallback",
                ctx.symbol, level + 1, price_str, size_str,
            )
            return True

        except Exception as e:
            logger.error("[%s] ♾️ ❌ BUY L%d taker échoué: %s", ctx.symbol, level + 1, e)
            self._telegram.notify_error(f"♾️ BUY L{level + 1} {ctx.symbol} échoué: {e}")
            return False

    def _execute_sell(
        self,
        ctx: PairContext,
        price: float,
        size: float,
        proceeds: float,
        level: int,
        use_taker: bool = False,
    ) -> bool:
        """Exécute une vente via Revolut X."""
        if self.dry_run:
            label = f"TP{level + 1}" if level >= 0 else "CLOSE"
            logger.info(
                "[DRY-RUN] INFINITY SELL %s | %s @ %s | size=%.8f ($%.2f)",
                label, ctx.symbol, _fmt(price), size, proceeds,
            )
            return True

        # Vérifier le solde réel
        try:
            base_currency = ctx.symbol.split("-")[0]
            balances = self._client.get_balances()
            base_bal = next((b for b in balances if b.currency == base_currency), None)
            real_available = base_bal.available if base_bal else 0.0

            # Annuler les ordres actifs si besoin
            if real_available < size * 0.90:
                try:
                    active_orders = self._client.get_active_orders([ctx.symbol])
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
                balances = self._client.get_balances()
                base_bal = next((b for b in balances if b.currency == base_currency), None)
                real_available = base_bal.available if base_bal else 0.0

            if real_available < size:
                size = real_available
                logger.info("[%s] ♾️ 📐 Taille ajustée: %.8f", ctx.symbol, size)

            if size <= 0:
                logger.warning("[%s] ♾️ ⚠️ Aucun token disponible pour la vente", ctx.symbol)
                return False

        except Exception as e:
            logger.warning("[%s] ♾️ ⚠️ Balance check échoué: %s", ctx.symbol, e)

        price_str = self._format_order_price(price)
        size_str = f"{size:.8f}"

        order = OrderRequest(
            symbol=ctx.symbol,
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
                    # Fallback taker pour les ventes (ne pas rater la sortie)
                    logger.warning("[%s] ♾️ Maker no-fill → TAKER FALLBACK", ctx.symbol)
                    result = self._place_taker_order(order)

            fill_type = result.get("fill_type", "unknown")
            logger.info(
                "[%s] ♾️ ✅ SELL exécuté | @ %s | size=%s | %s",
                ctx.symbol, price_str, size_str, fill_type,
            )
            return True

        except Exception as e:
            logger.error("[%s] ♾️ ❌ SELL échoué: %s", ctx.symbol, e)
            self._telegram.notify_error(f"♾️ SELL {ctx.symbol} échoué: {e}")
            return False

    # ── Maker-only order execution (pair-agnostic via OrderRequest) ────────────

    def _place_maker_only_order(self, order: OrderRequest) -> dict:
        """Place un ordre limit passif (maker 0%). Si pas rempli → annule."""
        logger.info(
            "💰 MAKER-ONLY | %s %s @ %s (attente %ds)",
            order.side.value.upper(), order.symbol, order.price, INF_MAKER_WAIT_SECONDS,
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
        logger.info("💰 MAKER-ONLY | ⏳ Attente %ds…", INF_MAKER_WAIT_SECONDS)
        time.sleep(INF_MAKER_WAIT_SECONDS)

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
        """Place un ordre taker agressif (pour SL / close forcée)."""
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

    # ── Firebase logging ───────────────────────────────────────────────────────

    def _log_buy_firebase(self, ctx: PairContext, price: float, size: float, cost: float, equity: float) -> None:
        """Log un achat dans Firebase."""
        try:
            fb_position = Position(
                symbol=ctx.symbol,
                side=OrderSide.BUY,
                entry_price=price,
                sl_price=ctx.cycle.pmp * (1 - ctx.config.stop_loss_pct) if ctx.cycle.pmp > 0 else 0,
                size=size,
                venue_order_id=str(uuid.uuid4()),
                status=PositionStatus.OPEN,
                strategy=StrategyType.INFINITY,
                tp_price=ctx.cycle.pmp * (1 + ctx.config.sell_levels[0]) if ctx.cycle.pmp > 0 else 0,
            )
            fb_id = log_trade_opened(
                position=fb_position,
                fill_type="maker",
                maker_wait_seconds=INF_MAKER_WAIT_SECONDS,
                risk_pct=ctx.config.stop_loss_pct,
                risk_amount_usd=cost * ctx.config.stop_loss_pct,
                fiat_balance=equity,
                current_equity=equity,
                portfolio_risk_before=0.0,
                exchange="revolut-infinity",
                dry_run=self.dry_run,
            )
            if fb_id:
                ctx.cycle.firebase_trade_ids.append(fb_id)
                self._save_state(ctx)
        except Exception as e:
            logger.warning("[%s] 🔥 Firebase log_trade_opened échoué: %s", ctx.symbol, e)

    def _log_cycle_close_firebase(self, ctx: PairContext, exit_price: float, pnl_usd: float, reason: str) -> None:
        """Close tous les trade docs Firebase du cycle."""
        equity = self._get_allocated_balance(ctx)
        for fb_id in ctx.cycle.firebase_trade_ids:
            try:
                fb_position = Position(
                    symbol=ctx.symbol,
                    side=OrderSide.BUY,
                    entry_price=ctx.cycle.pmp,
                    sl_price=0,
                    size=ctx.cycle.total_size,
                    venue_order_id="cycle-close",
                    status=PositionStatus.CLOSED,
                    strategy=StrategyType.INFINITY,
                    pnl=pnl_usd,
                )
                log_trade_closed(
                    trade_id=fb_id,
                    position=fb_position,
                    exit_price=exit_price,
                    reason=reason,
                    fill_type="maker",
                    equity_after=equity,
                )
            except Exception as e:
                logger.warning("[%s] 🔥 Firebase log_trade_closed échoué: %s", ctx.symbol, e)

    # ── Heartbeat ──────────────────────────────────────────────────────────────

    def _maybe_heartbeat(self) -> None:
        now = time.time()
        heartbeat_seconds = get_heartbeat_override_seconds("infinity", INF_HEARTBEAT_SECONDS)
        if heartbeat_seconds != self._heartbeat_seconds:
            self._heartbeat_seconds = heartbeat_seconds
            logger.info("💓 [INFINITY] Heartbeat runtime override: %ss", self._heartbeat_seconds)

        if now - self._last_heartbeat < self._heartbeat_seconds:
            return
        self._last_heartbeat = now

        # Equity scope Infinity (exclut assets non Infinity, ex: staking)
        total_equity = self._get_scoped_equity()
        infinity_pool = total_equity * INF_CAPITAL_PCT

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

        # Per-pair status
        active_pairs = 0
        total_exposure = 0.0
        total_unrealized = 0.0
        tg_active_lines: list[str] = []
        near_targets: list[tuple[str, float, float, float]] = []  # (symbol, ecart_pct, price, target)

        for symbol, ctx in self._pairs.items():
            cycle = ctx.cycle
            cfg = ctx.config
            allocated = self._get_allocated_balance(ctx)

            trailing_high = self._get_trailing_high(ctx)

            # Prix actuel
            current_price = 0.0
            try:
                ticker = self._data.get_ticker(symbol)
                if ticker:
                    current_price = ticker.last_price
            except Exception:
                pass

            target_entry = trailing_high * (1 - cfg.entry_drop_pct)
            ecart = current_price - target_entry
            ecart_pct = (ecart / target_entry * 100) if target_entry > 0 else 0

            # PnL latent
            pnl_latent = 0.0
            pnl_latent_pct = 0.0
            if cycle.phase != "WAITING" and cycle.size_remaining > 0 and current_price > 0:
                active_pairs += 1
                current_value = cycle.size_remaining * current_price + cycle.total_proceeds
                pnl_latent = current_value - cycle.total_cost
                pnl_latent_pct = pnl_latent / cycle.total_cost * 100 if cycle.total_cost > 0 else 0
                total_unrealized += pnl_latent
                total_exposure += cycle.size_remaining * current_price

            # Console log per pair
            if cycle.phase == "WAITING":
                logger.info(
                    "   [%s] phase=%s | alloc=$%.0f | trail=%s | prix=%s | cible=%s | écart=%.1f%%",
                    symbol, cycle.phase, allocated,
                    _fmt(trailing_high), _fmt(current_price), _fmt(target_entry), ecart_pct,
                )
            else:
                # Prochain palier d'achat
                n_buys = len(cycle.buys)
                next_buy_str = "—"
                if n_buys < len(cfg.buy_levels):
                    next_buy_price = cycle.reference_price * (1 + cfg.buy_levels[n_buys])
                    next_buy_str = f"L{n_buys + 1}@{_fmt(next_buy_price)}"
                # Prochain palier de vente
                n_sells = len(cycle.sells)
                next_tp_str = "—"
                if n_sells < len(cfg.sell_levels) and cycle.pmp > 0:
                    next_tp_price = cycle.pmp * (1 + cfg.sell_levels[n_sells])
                    next_tp_str = f"TP{n_sells + 1}@{_fmt(next_tp_price)}"

                logger.info(
                    "   [%s] phase=%s | alloc=$%.0f | prix=%s | PMP=%s | next_buy=%s | next_tp=%s",
                    symbol, cycle.phase, allocated,
                    _fmt(current_price), _fmt(cycle.pmp), next_buy_str, next_tp_str,
                )
                logger.info(
                    "      inv=$%.2f | size=%.8f | buys=%d | sells=%d | PnL=$%+.2f (%+.1f%%)",
                    cycle.total_cost, cycle.size_remaining,
                    len(cycle.buys), len(cycle.sells), pnl_latent, pnl_latent_pct,
                )

            # Log last eval
            if ctx.last_eval:
                ev = ctx.last_eval
                drop_icon = "✅" if ev.get("drop_ok") else "❌"
                rsi_icon = "✅" if ev.get("rsi_ok") else "❌"
                logger.info(
                    "      🔎 Dernière éval (%s): %s | %s Drop: %.1f%% | %s RSI: %.1f",
                    ev.get("ts", "?"), ev.get("result", "?"),
                    drop_icon, ev.get("drop_pct", 0),
                    rsi_icon, ev.get("rsi", 0),
                )

            # Sync cycle to Firebase per pair
            try:
                self._sync_cycle_firebase(ctx, current_price=current_price)
            except Exception:
                logger.debug("[%s] Firebase cycle sync failed", symbol, exc_info=True)

            # Build Telegram sections: actives + near target only
            if cycle.phase != "WAITING":
                pnl_emoji = "🟢" if pnl_latent >= 0 else "🔴"
                be_tag = " 🔒BE" if cycle.breakeven_active else ""
                n_sells = len(cycle.sells)
                next_tp_str = "—"
                if n_sells < len(cfg.sell_levels) and cycle.pmp > 0:
                    nt_price = cycle.pmp * (1 + cfg.sell_levels[n_sells])
                    next_tp_str = f"TP{n_sells + 1}@{_fmt(nt_price)}"
                tg_active_lines.append(
                    f"  {pnl_emoji} `{symbol}` prix `{_fmt(current_price)}` | PMP `{_fmt(cycle.pmp)}` | "
                    f"Latent `{pnl_latent:+.2f}$` (`{pnl_latent_pct:+.1f}%`) | {next_tp_str}{be_tag}"
                )
            elif abs(ecart_pct) <= 1.0:
                near_targets.append((symbol, ecart_pct, current_price, target_entry))

        # Console summary
        logger.info(
            "💓 INFINITY Alive | tick=%d | paires=%d/%d actives | equity=$%.2f | alloc=$%.2f\n"
            "   ⏳ Prochaine H4: %s",
            self._tick_count, active_pairs, len(self._pairs),
            total_equity, infinity_pool,
            countdown_str,
        )

        # Telegram multi-pair heartbeat
        try:
            # System status
            if total_unrealized < -infinity_pool * 0.05:
                sys_emoji, sys_label = "🔴", "risk mode"
            elif total_unrealized < 0:
                sys_emoji, sys_label = "🟡", "watching"
            else:
                sys_emoji, sys_label = "🟢", "stable"

            unr_emoji = "🟢" if total_unrealized >= 0 else "🔴"
            last_update = now_utc.strftime("%H:%M UTC")

            slot_pct = INF_CAPITAL_PCT / INF_CAPITAL_ACTIVE_SLOTS * 100
            slots_free = max(0, INF_CAPITAL_ACTIVE_SLOTS - active_pairs)

            tg_lines = [
                f"{sys_emoji} *INFINITY* ♾️ ({len(self._pairs)} paires) — {sys_label}",
                f"  💰 Equity scope: `${total_equity:,.0f}` | Pool ({INF_CAPITAL_PCT * 100:.0f}%): `${infinity_pool:,.0f}`",
                f"  🧩 Allocation slots: actives=`{active_pairs}/{INF_CAPITAL_ACTIVE_SLOTS}` | part/slot=`{slot_pct:.1f}%` | libres=`{slots_free}`",
                f"  📦 Exposition active: `${total_exposure:,.0f}`",
                f"  {unr_emoji} PnL open: `${total_unrealized:+.2f}`",
                f"  ⏳ Prochaine H4: `{countdown_str}`",
            ]

            if tg_active_lines:
                tg_lines.append("\n  *Paires actives*:")
                tg_lines.extend(tg_active_lines[:6])

            if near_targets:
                tg_lines.append("\n  *Proches cible entrée (±1%)*:")
                for symbol, ecart_pct, current_price, target_entry in sorted(near_targets, key=lambda x: abs(x[1]))[:5]:
                    tg_lines.append(
                        f"  🎯 `{symbol}` prix `{_fmt(current_price)}` | cible `{_fmt(target_entry)}` | écart `{ecart_pct:+.1f}%`"
                    )

            from src.notifications.telegram import DASHBOARD_URL
            tg_lines.append(f"\n  🕐 `{last_update}`")
            tg_lines.append(f"[Dashboard]({DASHBOARD_URL})")
            self._telegram.send_raw("\n".join(tg_lines))
        except Exception:
            logger.warning("Telegram heartbeat failed", exc_info=True)

        # Firebase heartbeat
        try:
            fb_log_heartbeat(
                open_positions=active_pairs,
                total_equity=infinity_pool,
                total_risk_pct=0.0,
                pairs_count=len(self._pairs),
                exchange="revolut-infinity",
                dry_run=self.dry_run,
            )
        except Exception:
            pass

    # ── Helpers ────────────────────────────────────────────────────────────────

    def _maybe_daily_tasks(self) -> None:
        """Cleanup des events Firebase + daily snapshot — 1×/jour UTC."""
        from datetime import datetime as dt, timezone as tz
        today = dt.now(tz.utc).strftime("%Y-%m-%d")

        if today != self._last_cleanup_date:
            self._last_cleanup_date = today
            try:
                fb_cleanup_events()
                logger.info("🧹 Cleanup events Firebase (> %dj)", config.FIREBASE_EVENTS_RETENTION_DAYS)
            except Exception:
                pass

        if today != self._last_snapshot_date:
            self._last_snapshot_date = today
            try:
                total_equity = 0.0
                positions = []
                for symbol, ctx in self._pairs.items():
                    allocated = self._get_allocated_balance(ctx)
                    total_equity += allocated
                    cycle = ctx.cycle
                    if cycle.phase != "WAITING":
                        positions.append({
                            "symbol": symbol,
                            "phase": cycle.phase,
                            "pmp": cycle.pmp,
                            "size": cycle.size_remaining,
                            "invested": cycle.total_cost,
                            "buys": len(cycle.buys),
                            "sells": len(cycle.sells),
                        })
                fb_log_daily_snapshot(
                    equity=total_equity,
                    positions=positions,
                    daily_pnl=0.0,
                    trades_today=0,
                    exchange="revolut-infinity",
                    dry_run=self.dry_run,
                )
                logger.info("📸 Daily snapshot Firebase loggé")
            except Exception:
                pass

    def _save_state(self, ctx: PairContext, sync_firebase: bool = True) -> None:
        ctx.store.save(
            cycle=ctx.cycle,
            candle_highs=ctx.candle_highs,
            last_candle_ts=ctx.last_candle_ts,
            cycle_count=ctx.cycle_count,
            consecutive_stops=ctx.consecutive_stops,
        )
        if sync_firebase:
            self._sync_cycle_firebase(ctx)

    def _sync_cycle_firebase(self, ctx: PairContext, current_price: float | None = None) -> None:
        """Sync le cycle courant dans Firebase (infinity_cycles/{symbol}).

        Per-pair Firebase documents for the dashboard V-curve.
        """
        try:
            cycle = ctx.cycle
            cfg = ctx.config
            trailing_high = self._get_trailing_high(ctx)
            if current_price is None:
                current_price = 0.0
                try:
                    ticker = self._data.get_ticker(ctx.symbol)
                    if ticker:
                        current_price = ticker.last_price
                except Exception:
                    pass

            doc = {
                # Cycle state
                "phase": cycle.phase,
                "reference_price": cycle.reference_price,
                "pmp": cycle.pmp,
                "total_size": cycle.total_size,
                "total_cost": cycle.total_cost,
                "size_remaining": cycle.size_remaining,
                "total_proceeds": cycle.total_proceeds,
                "breakeven_active": cycle.breakeven_active,
                "cycle_start_ts": cycle.cycle_start_ts,
                # Config (pour le dashboard)
                "buy_levels": list(cfg.buy_levels),
                "buy_pcts": list(cfg.buy_pcts),
                "sell_levels": list(cfg.sell_levels),
                "stop_loss_pct": cfg.stop_loss_pct,
                "entry_drop_pct": cfg.entry_drop_pct,
                # Buys / Sells détaillés
                "buys": cycle.buys,
                "sells": cycle.sells,
                "sell_levels_hit": cycle.sell_levels_hit,
                # Contexte marché
                "trailing_high": trailing_high,
                "current_price": current_price,
                "target_entry": trailing_high * (1 - cfg.entry_drop_pct),
                # Meta
                "symbol": ctx.symbol,
                "cycle_count": ctx.cycle_count,
                "consecutive_stops": ctx.consecutive_stops,
                "updated_at": datetime.now(timezone.utc).isoformat(),
            }
            # Per-pair doc: infinity_cycles/BTC-USD, infinity_cycles/AAVE-USD, etc.
            fb_add_document("infinity_cycles", doc, doc_id=ctx.symbol)
        except Exception as e:
            logger.debug("[%s] Firebase cycle sync failed: %s", ctx.symbol, e)

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
        logger.info("🛑 Arrêt InfinityBot...")
        for symbol, ctx in self._pairs.items():
            self._save_state(ctx, sync_firebase=False)
        logger.info("💾 État final sauvegardé (%d paires)", len(self._pairs))
        self._client.close()
        self._telegram.close()
        logger.info("InfinityBot arrêté proprement")


# ── Point d'entrée ─────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="TradeX – Infinity Bot (DCA inversé multi-paires)")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Mode simulation (log les ordres sans les exécuter)",
    )
    args = parser.parse_args()

    bot = InfinityBot(dry_run=args.dry_run)
    signal.signal(signal.SIGTERM, lambda *_: bot.stop())
    bot.run()


if __name__ == "__main__":
    main()
