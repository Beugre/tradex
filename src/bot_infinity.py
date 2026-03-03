"""
Bot Infinity — DCA inversé sur BTC, Revolut X, maker-only.

Timeframe H4, BTC-USD uniquement.
Polling toutes les 30s (prix), gestion sur bougie H4.

Stratégie :
  1. Calcul trailing high (72 H4 = 12 jours)
  2. Si prix chute ≥ 5% du trailing high → premier achat
  3. DCA : 5 paliers (-5% à -25% du prix de référence)
  4. Vente : 5 paliers (+0.8% à +4% du PMP)
  5. Breakeven stop après TP1
  6. Stop-loss : -15% du PMP

Capital : 65% du solde Revolut X (35% réservé au Momentum bot).

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

# ── Logging ────────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=getattr(logging, config.LOG_LEVEL, logging.INFO),
    format="[%(asctime)s] %(levelname)s %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("tradex.infinity_bot")

logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)


# ── Config Infinity (depuis .env) ──────────────────────────────────────────────

INF_SYMBOL: str = config.INF_TRADING_PAIR
INF_POLLING_SECONDS: int = config.INF_POLLING_SECONDS
INF_HEARTBEAT_SECONDS: int = config.INF_HEARTBEAT_SECONDS
INF_MAKER_WAIT_SECONDS: int = config.INF_MAKER_WAIT_SECONDS
INF_CAPITAL_PCT: float = config.INF_CAPITAL_PCT

# Parse buy/sell levels from config
_buy_levels_raw = [float(x) for x in config.INF_BUY_LEVELS.split(",")]
_buy_pcts_raw = [float(x) for x in config.INF_BUY_PCTS.split(",")]
_sell_levels_raw = [float(x) for x in config.INF_SELL_LEVELS.split(",")]

INF_CONFIG = InfinityConfig(
    trailing_high_period=config.INF_TRAILING_HIGH_PERIOD,
    entry_drop_pct=config.INF_ENTRY_DROP_PCT,
    buy_levels=tuple(_buy_levels_raw),
    buy_pcts=tuple(_buy_pcts_raw),
    sell_levels=tuple(_sell_levels_raw),
    stop_loss_pct=config.INF_STOP_LOSS_PCT,
    max_invested_pct=config.INF_MAX_INVESTED_PCT,
    first_entry_rsi_max=config.INF_RSI_ENTRY_MAX,
    use_breakeven_stop=config.INF_USE_BREAKEVEN,
    scale_with_equity=True,
    rsi_sell_min=0.0,  # Pas de RSI gate sur les ventes (clé de perf)
    maker_fee=0.0,
    taker_fee=0.0009,
)

H4_INTERVAL = 240  # H4 en minutes

# State file dédié
INF_STATE_FILE: str = os.getenv(
    "INF_STATE_FILE",
    os.path.join(os.path.dirname(__file__), "..", "data", "state_infinity.json"),
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


# ── State store ────────────────────────────────────────────────────────────────


class InfinityStateStore:
    """Persistance atomique du cycle + candle tracking pour le bot Infinity."""

    def __init__(self, state_file: str = INF_STATE_FILE) -> None:
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
            logger.error("❌ Save failed: %s", e)
            if tmp.exists():
                try:
                    tmp.unlink()
                except OSError:
                    pass

    def load(self) -> dict:
        if not self._path.exists():
            logger.info("📂 Pas de state infinity — démarrage à vide")
            return {}
        try:
            with open(self._path, "r") as f:
                return json.load(f)
        except Exception as e:
            logger.error("❌ Load state infinity échoué: %s", e)
            return {}


# ── Bot principal ──────────────────────────────────────────────────────────────


class InfinityBot:
    """Bot DCA inversé sur BTC — Revolut X, maker-only, H4."""

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

        # Config
        self._cfg = INF_CONFIG

        # State
        self._store = InfinityStateStore()
        self._cycle = InfLiveCycle()
        self._candle_highs: list[float] = []  # H4 highs pour trailing
        self._last_candle_ts: int = 0
        self._cycle_count: int = 0
        self._consecutive_stops: int = 0

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

    # ── Run ────────────────────────────────────────────────────────────────────

    def run(self) -> None:
        self._running = True
        buy_lvls = ", ".join(f"{x*100:.0f}%" for x in self._cfg.buy_levels)
        sell_lvls = ", ".join(f"+{x*100:.1f}%" for x in self._cfg.sell_levels)
        buy_pcts = ", ".join(f"{x*100:.0f}%" for x in self._cfg.buy_pcts)

        logger.info("═" * 60)
        logger.info("♾️  InfinityBot démarré — DCA inversé BTC")
        logger.info("   Paire      : %s", INF_SYMBOL)
        logger.info("   Capital    : %.0f%% du solde Revolut X", INF_CAPITAL_PCT * 100)
        logger.info("   Timeframe  : H4 | Trail high: %d bars (%d jours)",
                     self._cfg.trailing_high_period,
                     self._cfg.trailing_high_period * 4 // 24)
        logger.info("   Entry drop : %.1f%% | SL: %.1f%%",
                     self._cfg.entry_drop_pct * 100, self._cfg.stop_loss_pct * 100)
        logger.info("   Buy levels : %s", buy_lvls)
        logger.info("   Buy sizing : %s (equity %%)", buy_pcts)
        logger.info("   Sell levels: %s", sell_lvls)
        logger.info("   Breakeven  : %s (après TP%d)",
                     "ON" if self._cfg.use_breakeven_stop else "OFF",
                     self._cfg.breakeven_after_level + 1)
        logger.info("   Polling    : %ds | Maker wait: %ds",
                     INF_POLLING_SECONDS, INF_MAKER_WAIT_SECONDS)
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
        """Charge l'état persisté + bougies initiales."""
        # Charger l'état
        state = self._store.load()
        if state:
            cycle_data = state.get("cycle")
            if cycle_data:
                self._cycle = InfLiveCycle.from_dict(cycle_data)
            self._candle_highs = state.get("candle_highs", [])
            self._last_candle_ts = state.get("last_candle_ts", 0)
            self._cycle_count = state.get("cycle_count", 0)
            self._consecutive_stops = state.get("consecutive_stops", 0)
            logger.info(
                "📂 State chargé: phase=%s, cycle=%d, buys=%d, sells=%d, stops_consec=%d",
                self._cycle.phase, self._cycle_count,
                len(self._cycle.buys), len(self._cycle.sells),
                self._consecutive_stops,
            )

        # Charger les bougies H4 initiales
        logger.info("── Chargement des bougies H4 initiales ──")
        try:
            candles = self._client.get_candles(INF_SYMBOL, interval=H4_INTERVAL)
            candles.sort(key=lambda c: c.timestamp)
            if candles:
                # Construire le tableau des highs
                self._candle_highs = [c.high for c in candles]
                self._last_candle_ts = candles[-1].timestamp
                logger.info(
                    "[%s] %d bougies H4 chargées | dernier high=%.2f",
                    INF_SYMBOL, len(candles), candles[-1].high,
                )
            else:
                logger.warning("[%s] Aucune bougie H4 reçue", INF_SYMBOL)
        except Exception as e:
            logger.error("[%s] ❌ Erreur chargement bougies: %s", INF_SYMBOL, e)

        # Réconciliation : vérifier que le BTC est bien là si on a un cycle actif
        if self._cycle.phase != "WAITING" and self._cycle.size_remaining > 0:
            self._reconcile_position()

        logger.info("── Init terminée | phase=%s ──", self._cycle.phase)

    def _reconcile_position(self) -> None:
        """Vérifie le solde BTC contre le cycle actif."""
        try:
            balances = self._client.get_balances()
            base_currency = INF_SYMBOL.split("-")[0]
            base_bal = next((b for b in balances if b.currency == base_currency), None)
            held = (base_bal.available + base_bal.reserved) if base_bal else 0.0

            if held >= self._cycle.size_remaining * 0.90:
                logger.info(
                    "[%s] ✅ Position confirmée | %.8f %s (attendu %.8f)",
                    INF_SYMBOL, held, base_currency, self._cycle.size_remaining,
                )
            else:
                logger.warning(
                    "[%s] ⚠️ BTC insuffisant: solde=%.8f, attendu=%.8f",
                    INF_SYMBOL, held, self._cycle.size_remaining,
                )
        except Exception as e:
            logger.warning("⚠️ Réconciliation impossible: %s", e)

    # ── Tick ───────────────────────────────────────────────────────────────────

    def _tick(self) -> None:
        """Un cycle de polling."""
        self._tick_count += 1

        try:
            # 1. Récupérer le prix actuel
            ticker = self._data.get_ticker(INF_SYMBOL)
            if not ticker:
                return
            price = ticker.last_price

            # 2. Vérifier les nouvelles bougies H4
            new_candle = self._check_new_candle()

            # 3. Gérer le cycle actif
            if self._cycle.phase == "WAITING":
                if new_candle:
                    self._try_first_entry(price)
            elif self._cycle.phase in ("ACCUMULATING", "DISTRIBUTING"):
                self._manage_cycle(price, new_candle)

        except Exception as e:
            logger.error("[%s] Erreur tick: %s", INF_SYMBOL, e, exc_info=True)

        self._maybe_heartbeat()
        self._maybe_daily_tasks()

    def _check_new_candle(self) -> bool:
        """Vérifie s'il y a une nouvelle bougie H4. Retourne True si oui."""
        try:
            candles = self._client.get_candles(INF_SYMBOL, interval=H4_INTERVAL)
            candles.sort(key=lambda c: c.timestamp)
        except Exception as e:
            logger.debug("[%s] Candle fetch failed: %s", INF_SYMBOL, e)
            return False

        if not candles:
            return False

        latest_ts = candles[-1].timestamp
        if latest_ts <= self._last_candle_ts:
            return False

        # Nouvelle bougie H4
        self._last_candle_ts = latest_ts
        self._candle_highs = [c.high for c in candles]
        self._save_state()

        logger.debug(
            "[%s] Nouvelle bougie H4 | close=%.2f | high=%.2f | vol=%.2f",
            INF_SYMBOL, candles[-1].close, candles[-1].high, candles[-1].volume,
        )
        return True

    # ── Trailing high ──────────────────────────────────────────────────────────

    def _get_trailing_high(self) -> float:
        """Calcule le trailing high sur les N dernières bougies H4."""
        period = self._cfg.trailing_high_period
        if not self._candle_highs:
            return 0.0
        window = self._candle_highs[-period:] if len(self._candle_highs) >= period else self._candle_highs
        return max(window)

    # ── RSI calculation ────────────────────────────────────────────────────────

    def _get_current_rsi(self) -> float:
        """Calcule le RSI courant à partir des closes H4."""
        try:
            candles = self._client.get_candles(INF_SYMBOL, interval=H4_INTERVAL)
            candles.sort(key=lambda c: c.timestamp)
            if len(candles) < self._cfg.rsi_period + 1:
                return 50.0
            closes = [c.close for c in candles]
            rsi_vals = rsi_series(closes, self._cfg.rsi_period)
            return rsi_vals[-1]
        except Exception:
            return 50.0

    # ── Volume MA ──────────────────────────────────────────────────────────────

    def _get_volume_data(self) -> tuple[float, float]:
        """Retourne (volume_courant, volume_ma20)."""
        try:
            candles = self._client.get_candles(INF_SYMBOL, interval=H4_INTERVAL)
            candles.sort(key=lambda c: c.timestamp)
            if not candles:
                return 0.0, 0.0
            volumes = [c.volume for c in candles]
            ma_vals = sma_series(volumes, self._cfg.volume_ma_len)
            return volumes[-1], ma_vals[-1]
        except Exception:
            return 0.0, 0.0

    # ── Allocated capital ──────────────────────────────────────────────────────

    def _get_allocated_balance(self) -> float:
        """Retourne le capital alloué à l'Infinity Bot (65% du solde USD)."""
        try:
            balances = self._client.get_balances()
            quote = INF_SYMBOL.split("-")[1]  # USD
            usd_bal = next((b for b in balances if b.currency == quote), None)
            available = usd_bal.available if usd_bal else 0.0

            # Si on a un cycle actif, l'equity = cash dispo + valeur BTC
            if self._cycle.phase != "WAITING" and self._cycle.size_remaining > 0:
                ticker = self._data.get_ticker(INF_SYMBOL)
                btc_value = self._cycle.size_remaining * ticker.last_price if ticker else 0
                total = available + btc_value
            else:
                total = available

            return total * INF_CAPITAL_PCT
        except Exception as e:
            logger.warning("⚠️ Impossible de calculer le solde alloué: %s", e)
            return 0.0

    def _get_cash_available(self) -> float:
        """Retourne le cash USD disponible × INF_CAPITAL_PCT."""
        try:
            balances = self._client.get_balances()
            quote = INF_SYMBOL.split("-")[1]
            usd_bal = next((b for b in balances if b.currency == quote), None)
            available = usd_bal.available if usd_bal else 0.0
            return available * INF_CAPITAL_PCT
        except Exception:
            return 0.0

    # ── First entry ────────────────────────────────────────────────────────────

    def _try_first_entry(self, price: float) -> None:
        """Tente le premier achat d'un nouveau cycle."""
        # Safety: max consecutive stops
        if self._consecutive_stops >= self._cfg.max_consecutive_stops:
            logger.warning(
                "♾️ %d stops consécutifs → pause (max=%d)",
                self._consecutive_stops, self._cfg.max_consecutive_stops,
            )
            return

        trailing_high = self._get_trailing_high()
        if trailing_high <= 0:
            return

        rsi = self._get_current_rsi()
        volume, volume_ma = self._get_volume_data()

        # Check conditions
        ok = check_first_entry(
            close=price,
            trailing_high=trailing_high,
            entry_drop_pct=self._cfg.entry_drop_pct,
            rsi=rsi,
            rsi_max=self._cfg.first_entry_rsi_max,
            volume=volume,
            volume_ma=volume_ma,
            require_volume=self._cfg.require_volume_entry,
        )

        if not ok:
            return

        drop_pct = (trailing_high - price) / trailing_high * 100
        logger.info(
            "♾️ ENTRY SIGNAL | prix=%.2f | trail_high=%.2f | drop=%.1f%% | RSI=%.1f",
            price, trailing_high, drop_pct, rsi,
        )

        # Calculer le montant du premier achat
        allocated = self._get_allocated_balance()
        if allocated <= 0:
            logger.warning("♾️ Pas de capital alloué — skip")
            return

        # Premier palier : buy_pcts[0] de l'equity allouée
        target_amount = allocated * self._cfg.buy_pcts[0]

        buy_amount = compute_buy_size(
            rsi=rsi,
            rsi_full=self._cfg.rsi_full_buy,
            rsi_half=self._cfg.rsi_half_buy,
            target_amount=target_amount,
            cash_available=self._get_cash_available(),
            max_invested=allocated * self._cfg.max_invested_pct,
            already_invested=0.0,
        )

        if buy_amount <= 10:  # min $10
            logger.info("♾️ Montant trop faible ($%.2f) — skip", buy_amount)
            return

        # Exécuter l'achat
        size = buy_amount / price
        success = self._execute_buy(price, size, buy_amount, level=0)

        if success:
            # Démarrer un nouveau cycle
            self._cycle = InfLiveCycle(
                phase="ACCUMULATING",
                reference_price=trailing_high,
                buys=[{"level": 0, "price": price, "size": size, "cost": buy_amount, "ts": time.time()}],
                total_size=size,
                total_cost=buy_amount,
                pmp=price,
                size_remaining=size,
                cycle_start_ts=time.time(),
            )
            self._cycle_count += 1
            self._consecutive_stops = 0
            self._save_state()

            logger.info(
                "♾️ CYCLE #%d démarré | ref=%.2f | buy L1 @ %.2f | size=%.8f ($%.2f)",
                self._cycle_count, trailing_high, price, size, buy_amount,
            )

            # Telegram
            self._telegram.notify_infinity_buy(
                symbol=INF_SYMBOL,
                level=0,
                price=price,
                size=size,
                cost_usd=buy_amount,
                pmp=price,
                total_invested=buy_amount,
                equity=allocated,
            )

            # Firebase
            self._log_buy_firebase(price, size, buy_amount, allocated)

    # ── Manage active cycle ────────────────────────────────────────────────────

    def _manage_cycle(self, price: float, new_candle: bool) -> None:
        """Gère un cycle actif : check SL, sells, additional buys."""
        cycle = self._cycle
        cfg = self._cfg

        # ── 1. Stop-loss check (toujours, pas que sur nouvelle bougie) ──
        if cycle.pmp > 0 and check_stop_loss(price, cycle.pmp, cfg.stop_loss_pct):
            logger.info(
                "♾️ 🛑 STOP LOSS | prix=%.2f | PMP=%.2f | SL=%.2f",
                price, cycle.pmp, cycle.pmp * (1 - cfg.stop_loss_pct),
            )
            self._close_cycle(price, InfinityExitReason.STOP_LOSS)
            return

        # ── 2. Breakeven stop check ──
        if cycle.breakeven_active and cycle.pmp > 0 and price <= cycle.pmp:
            logger.info(
                "♾️ 🔒 BREAKEVEN STOP | prix=%.2f | PMP=%.2f",
                price, cycle.pmp,
            )
            self._close_cycle(price, InfinityExitReason.STOP_LOSS)
            return

        # ── 3. Override sell (+20% du PMP) ──
        if check_override_sell(price, cycle.pmp, cfg.override_sell_pct):
            logger.info(
                "♾️ 🚀 OVERRIDE SELL | prix=%.2f | PMP=%.2f | +%.1f%%",
                price, cycle.pmp,
                (price - cycle.pmp) / cycle.pmp * 100,
            )
            self._close_cycle(price, InfinityExitReason.OVERRIDE_SELL)
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
                    success = self._execute_sell(price, sell_size, proceeds, level=i)

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
                            "♾️ TP%d HIT | prix=%.2f | PMP=%.2f | +%.2f%% | sold=%.8f ($%.2f) | remaining=%.8f",
                            i + 1, price, cycle.pmp, gain_pct, sell_size, proceeds, cycle.size_remaining,
                        )

                        # Breakeven après le premier TP
                        if (cfg.use_breakeven_stop
                                and i >= cfg.breakeven_after_level
                                and not cycle.breakeven_active):
                            cycle.breakeven_active = True
                            logger.info("♾️ 🔒 BREAKEVEN activé après TP%d", i + 1)

                        self._save_state()

                        # Telegram
                        self._telegram.notify_infinity_sell(
                            symbol=INF_SYMBOL,
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
                            self._complete_cycle()
                            return

                    break  # un seul palier par tick

        # ── 5. Check additional buys (DCA, seulement sur nouvelle bougie) ──
        if new_candle and cycle.phase == "ACCUMULATING":
            self._try_additional_buy(price)

    # ── Additional DCA buy ─────────────────────────────────────────────────────

    def _try_additional_buy(self, price: float) -> None:
        """Tente un achat DCA additionnel si un palier est atteint."""
        cycle = self._cycle
        cfg = self._cfg
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
            rsi = self._get_current_rsi()
            volume, volume_ma = self._get_volume_data()

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
            allocated = self._get_allocated_balance()
            pct = cfg.buy_pcts[i] if i < len(cfg.buy_pcts) else 0.10
            target_amount = allocated * pct if pct > 0 else (allocated - cycle.total_cost)

            buy_amount = compute_buy_size(
                rsi=rsi,
                rsi_full=cfg.rsi_full_buy,
                rsi_half=cfg.rsi_half_buy,
                target_amount=target_amount,
                cash_available=self._get_cash_available(),
                max_invested=allocated * cfg.max_invested_pct,
                already_invested=cycle.total_cost,
            )

            if buy_amount <= 10:
                continue

            size = buy_amount / price
            success = self._execute_buy(price, size, buy_amount, level=i)

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
                self._save_state()

                logger.info(
                    "♾️ BUY L%d | prix=%.2f | size=%.8f ($%.2f) | PMP=%.2f | total_inv=$%.2f",
                    i + 1, price, size, buy_amount, cycle.pmp, cycle.total_cost,
                )

                # Telegram
                self._telegram.notify_infinity_buy(
                    symbol=INF_SYMBOL,
                    level=i,
                    price=price,
                    size=size,
                    cost_usd=buy_amount,
                    pmp=cycle.pmp,
                    total_invested=cycle.total_cost,
                    equity=allocated,
                )

                # Firebase
                self._log_buy_firebase(price, size, buy_amount, allocated)

            break  # un seul achat par tick

    # ── Cycle completion ───────────────────────────────────────────────────────

    def _complete_cycle(self) -> None:
        """Cycle terminé : tous les TP atteints."""
        cycle = self._cycle
        pnl_usd = cycle.total_proceeds - cycle.total_cost
        pnl_pct = pnl_usd / cycle.total_cost * 100 if cycle.total_cost > 0 else 0

        logger.info(
            "♾️ ✅ CYCLE #%d COMPLET | PMP=%.2f | Inv=$%.2f → Rec=$%.2f | PnL=$%+.2f (%+.1f%%)",
            self._cycle_count, cycle.pmp, cycle.total_cost,
            cycle.total_proceeds, pnl_usd, pnl_pct,
        )

        self._telegram.notify_infinity_cycle_complete(
            symbol=INF_SYMBOL,
            pmp=cycle.pmp,
            total_cost=cycle.total_cost,
            total_proceeds=cycle.total_proceeds,
            pnl_usd=pnl_usd,
            pnl_pct=pnl_pct,
            n_buys=len(cycle.buys),
            n_sells=len(cycle.sells),
        )

        # Firebase : close all trade docs
        self._log_cycle_close_firebase(cycle.pmp, pnl_usd, "TP_COMPLETE")

        # Reset cycle
        self._cycle = InfLiveCycle()
        self._save_state()

    def _close_cycle(self, price: float, reason: InfinityExitReason) -> None:
        """Ferme le cycle entier : vend tout le BTC restant."""
        cycle = self._cycle
        if cycle.size_remaining <= 0:
            self._cycle = InfLiveCycle()
            self._save_state()
            return

        # Sell tout le restant
        proceeds = cycle.size_remaining * price
        is_stop = reason == InfinityExitReason.STOP_LOSS
        fill_type = "taker" if is_stop else "maker"

        success = self._execute_sell(
            price, cycle.size_remaining, proceeds,
            level=-1, use_taker=is_stop,
        )

        if not success:
            logger.error("♾️ ❌ Échec clôture cycle — retry au prochain tick")
            return

        cycle.total_proceeds += proceeds
        pnl_usd = cycle.total_proceeds - cycle.total_cost
        pnl_pct = pnl_usd / cycle.total_cost * 100 if cycle.total_cost > 0 else 0

        logger.info(
            "♾️ %s CYCLE #%d | %s | PMP=%.2f | PnL=$%+.2f (%+.1f%%)",
            "🛑" if is_stop else "🚀",
            self._cycle_count, reason.value, cycle.pmp, pnl_usd, pnl_pct,
        )

        if is_stop:
            self._consecutive_stops += 1
            self._telegram.notify_infinity_stop(
                symbol=INF_SYMBOL,
                price=price,
                pmp=cycle.pmp,
                total_cost=cycle.total_cost,
                proceeds=cycle.total_proceeds,
                pnl_usd=pnl_usd,
                pnl_pct=pnl_pct,
            )
        else:
            self._consecutive_stops = 0
            self._telegram.notify_infinity_cycle_complete(
                symbol=INF_SYMBOL,
                pmp=cycle.pmp,
                total_cost=cycle.total_cost,
                total_proceeds=cycle.total_proceeds,
                pnl_usd=pnl_usd,
                pnl_pct=pnl_pct,
                n_buys=len(cycle.buys),
                n_sells=len(cycle.sells),
            )

        # Firebase
        self._log_cycle_close_firebase(price, pnl_usd, reason.value)

        # Reset
        self._cycle = InfLiveCycle()
        self._save_state()

    # ── Order execution ────────────────────────────────────────────────────────

    def _execute_buy(self, price: float, size: float, cost_usd: float, level: int) -> bool:
        """Exécute un achat BTC via Revolut X (maker-only)."""
        if self.dry_run:
            logger.info(
                "[DRY-RUN] INFINITY BUY L%d | %s @ %s | size=%.8f ($%.2f)",
                level + 1, INF_SYMBOL, _fmt(price), size, cost_usd,
            )
            return True

        price_str = self._format_order_price(price)
        size_str = f"{size:.8f}"

        order = OrderRequest(
            symbol=INF_SYMBOL,
            side=OrderSide.BUY,
            base_size=size_str,
            price=price_str,
        )

        try:
            result = self._place_maker_only_order(order)
            fill_type = result.get("fill_type", "unknown")

            if fill_type == "no_fill":
                logger.info(
                    "♾️ BUY L%d: pas de fill maker après %ds — abandonné",
                    level + 1, INF_MAKER_WAIT_SECONDS,
                )
                try:
                    fb_log_event(
                        event_type="infinity_maker_no_fill",
                        data={"level": level, "price": price, "side": "BUY"},
                        symbol=INF_SYMBOL,
                    )
                except Exception:
                    pass
                return False

            logger.info(
                "♾️ ✅ BUY L%d exécuté | %s @ %s | size=%s | %s",
                level + 1, INF_SYMBOL, price_str, size_str, fill_type,
            )
            return True

        except Exception as e:
            logger.error("♾️ ❌ BUY L%d échoué: %s", level + 1, e)
            self._telegram.notify_error(f"♾️ BUY L{level + 1} {INF_SYMBOL} échoué: {e}")
            return False

    def _execute_sell(
        self,
        price: float,
        size: float,
        proceeds: float,
        level: int,
        use_taker: bool = False,
    ) -> bool:
        """Exécute une vente BTC via Revolut X."""
        if self.dry_run:
            label = f"TP{level + 1}" if level >= 0 else "CLOSE"
            logger.info(
                "[DRY-RUN] INFINITY SELL %s | %s @ %s | size=%.8f ($%.2f)",
                label, INF_SYMBOL, _fmt(price), size, proceeds,
            )
            return True

        # Vérifier le solde réel
        try:
            base_currency = INF_SYMBOL.split("-")[0]
            balances = self._client.get_balances()
            base_bal = next((b for b in balances if b.currency == base_currency), None)
            real_available = base_bal.available if base_bal else 0.0

            # Annuler les ordres actifs si besoin
            if real_available < size * 0.90:
                try:
                    active_orders = self._client.get_active_orders([INF_SYMBOL])
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
                logger.info("♾️ 📐 Taille ajustée: %.8f", size)

            if size <= 0:
                logger.warning("♾️ ⚠️ Aucun BTC disponible pour la vente")
                return False

        except Exception as e:
            logger.warning("♾️ ⚠️ Balance check échoué: %s", e)

        price_str = self._format_order_price(price)
        size_str = f"{size:.8f}"

        order = OrderRequest(
            symbol=INF_SYMBOL,
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
                    logger.warning("♾️ Maker no-fill → TAKER FALLBACK")
                    result = self._place_taker_order(order)

            fill_type = result.get("fill_type", "unknown")
            logger.info(
                "♾️ ✅ SELL exécuté | %s @ %s | size=%s | %s",
                INF_SYMBOL, price_str, size_str, fill_type,
            )
            return True

        except Exception as e:
            logger.error("♾️ ❌ SELL échoué: %s", e)
            self._telegram.notify_error(f"♾️ SELL {INF_SYMBOL} échoué: {e}")
            return False

    # ── Maker-only order execution ─────────────────────────────────────────────

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

    def _log_buy_firebase(self, price: float, size: float, cost: float, equity: float) -> None:
        """Log un achat dans Firebase."""
        try:
            fb_position = Position(
                symbol=INF_SYMBOL,
                side=OrderSide.BUY,
                entry_price=price,
                sl_price=self._cycle.pmp * (1 - self._cfg.stop_loss_pct) if self._cycle.pmp > 0 else 0,
                size=size,
                venue_order_id=str(uuid.uuid4()),
                status=PositionStatus.OPEN,
                strategy=StrategyType.INFINITY,
                tp_price=self._cycle.pmp * (1 + self._cfg.sell_levels[0]) if self._cycle.pmp > 0 else 0,
            )
            fb_id = log_trade_opened(
                position=fb_position,
                fill_type="maker",
                maker_wait_seconds=INF_MAKER_WAIT_SECONDS,
                risk_pct=self._cfg.stop_loss_pct,
                risk_amount_usd=cost * self._cfg.stop_loss_pct,
                fiat_balance=equity,
                current_equity=equity,
                portfolio_risk_before=0.0,
                exchange="revolut-infinity",
            )
            if fb_id:
                self._cycle.firebase_trade_ids.append(fb_id)
                self._save_state()
        except Exception as e:
            logger.warning("🔥 Firebase log_trade_opened échoué: %s", e)

    def _log_cycle_close_firebase(self, exit_price: float, pnl_usd: float, reason: str) -> None:
        """Close tous les trade docs Firebase du cycle."""
        equity = self._get_allocated_balance()
        for fb_id in self._cycle.firebase_trade_ids:
            try:
                fb_position = Position(
                    symbol=INF_SYMBOL,
                    side=OrderSide.BUY,
                    entry_price=self._cycle.pmp,
                    sl_price=0,
                    size=self._cycle.total_size,
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
                logger.warning("🔥 Firebase log_trade_closed échoué: %s", e)

    # ── Heartbeat ──────────────────────────────────────────────────────────────

    def _maybe_heartbeat(self) -> None:
        now = time.time()
        if now - self._last_heartbeat < INF_HEARTBEAT_SECONDS:
            return
        self._last_heartbeat = now

        cycle = self._cycle
        allocated = self._get_allocated_balance()

        # Prix actuel
        current_price = 0.0
        try:
            ticker = self._data.get_ticker(INF_SYMBOL)
            if ticker:
                current_price = ticker.last_price
        except Exception:
            pass

        trailing_high = self._get_trailing_high()

        # PnL latent
        pnl_latent = 0.0
        pnl_latent_pct = 0.0
        if cycle.phase != "WAITING" and cycle.size_remaining > 0 and current_price > 0:
            current_value = cycle.size_remaining * current_price + cycle.total_proceeds
            pnl_latent = current_value - cycle.total_cost
            pnl_latent_pct = pnl_latent / cycle.total_cost * 100 if cycle.total_cost > 0 else 0

        # Equity totale sur Revolut X
        total_equity = 0.0
        try:
            balances = self._client.get_balances()
            fiat_set = {"USD", "EUR", "GBP"}
            for b in balances:
                if b.total > 0:
                    if b.currency in fiat_set:
                        total_equity += b.total
                    else:
                        try:
                            t = self._data.get_ticker(f"{b.currency}-USD")
                            if t:
                                total_equity += b.total * t.last_price
                        except Exception:
                            pass
        except Exception:
            pass

        target_entry = trailing_high * (1 - INF_CONFIG.entry_drop_pct)
        logger.info(
            "💓 INFINITY Alive | tick=%d | cycle=%d | phase=%s | equity=$%.2f "
            "| alloc=$%.2f | trail_high=%.2f | cible=%.2f | price=%.2f",
            self._tick_count, self._cycle_count, cycle.phase,
            total_equity, allocated, trailing_high, target_entry, current_price,
        )

        if cycle.phase != "WAITING":
            logger.info(
                "   PMP=%.2f | invested=$%.2f | BTC=%.8f | buys=%d | sells=%d | PnL latent=$%+.2f (%+.1f%%)",
                cycle.pmp, cycle.total_cost, cycle.size_remaining,
                len(cycle.buys), len(cycle.sells), pnl_latent, pnl_latent_pct,
            )

        # Telegram heartbeat
        try:
            target_price = trailing_high * (1 - INF_CONFIG.entry_drop_pct)
            self._telegram.notify_infinity_heartbeat(
                equity=total_equity,
                allocated_equity=allocated,
                phase=cycle.phase,
                pmp=cycle.pmp,
                total_invested=cycle.total_cost,
                size_btc=cycle.size_remaining,
                buys_filled=len(cycle.buys),
                sells_filled=len(cycle.sells),
                pnl_latent_usd=pnl_latent,
                pnl_latent_pct=pnl_latent_pct,
                trailing_high=trailing_high,
                current_price=current_price,
                target_price=target_price,
                breakeven_active=cycle.breakeven_active,
                cycle_count=self._cycle_count,
            )
        except Exception:
            logger.warning("Telegram heartbeat failed", exc_info=True)

        # Firebase heartbeat
        try:
            fb_log_heartbeat(
                open_positions=1 if cycle.phase != "WAITING" else 0,
                total_equity=total_equity,
                total_risk_pct=0.0,
                pairs_count=1,
                exchange="revolut-infinity",
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
                equity = self._get_allocated_balance()
                positions = []
                cycle = self._cycle
                if cycle.phase != "WAITING":
                    positions.append({
                        "symbol": INF_SYMBOL,
                        "phase": cycle.phase,
                        "pmp": cycle.pmp,
                        "size": cycle.size_remaining,
                        "invested": cycle.total_cost,
                        "buys": len(cycle.buys),
                        "sells": len(cycle.sells),
                    })
                fb_log_daily_snapshot(
                    equity=equity,
                    positions=positions,
                    daily_pnl=0.0,
                    trades_today=0,
                    exchange="revolut-infinity",
                )
                logger.info("📸 Daily snapshot Firebase loggé")
            except Exception:
                pass

    def _save_state(self) -> None:
        self._store.save(
            cycle=self._cycle,
            candle_highs=self._candle_highs,
            last_candle_ts=self._last_candle_ts,
            cycle_count=self._cycle_count,
            consecutive_stops=self._consecutive_stops,
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
        logger.info("🛑 Arrêt InfinityBot...")
        self._save_state()
        logger.info("💾 État final sauvegardé")
        self._client.close()
        self._telegram.close()
        logger.info("InfinityBot arrêté proprement")


# ── Point d'entrée ─────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="TradeX – Infinity Bot (DCA inversé BTC)")
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
