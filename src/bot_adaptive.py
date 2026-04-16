"""
Bot Adaptive Bull — Binance Spot, USDC, taker (MARKET orders), 15m.

Stratégie : Bull Trend Following validée walk-forward 3/3
  - Régime BULL uniquement (score EMA/ADX/RSI sur 1H)
  - Entrée : pullback EMA50 15m + RSI 50-65 + slope positif + bougie haussière
  - SL -1.5% (close-only via polling) | Trailing -2.5% du peak | TP +8%
  - Pyramiding +15% sur position gagnante (1× par trade)
  - 6 paires : BTCUSDC, ETHUSDC, SOLUSDC, XRPUSDC, AVAXUSDC, NEARUSDC

Performance backtest 6 ans : $1 000 → $5 051 (+405%), CAGR +31%,
  WR 34.1%, PF 1.18, DD max -20.8%, walk-forward OOS PF 1.14.

Capital : budget fixe isolé (ADT_ALLOCATED_BALANCE) — n'interfère pas
  avec l'allocateur Trail Range / CrashBot / Listing.

Usage :
    python -m src.bot_adaptive              # Production
    python -m src.bot_adaptive --dry-run    # Dry-run (pas d'ordres)
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
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from src import config
from src.core.adaptive_engine import (
    Regime,
    BullSignal,
    RegimeDebug,
    EntryDebug,
    check_bull_entry,
    check_bull_entry_debug,
    check_pyramid_entry,
    detect_regime,
    detect_regime_debug,
    is_trend_broken,
    update_adaptive_trailing,
)
from src.core.models import (
    Balance,
    Candle,
    OrderSide,
    Position,
    PositionStatus,
    StrategyType,
)
from src.exchange.binance_client import BinanceClient
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
    get_pending_runtime_actions,
    mark_runtime_action_status,
)

# ── Logging ────────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=getattr(logging, config.LOG_LEVEL, logging.INFO),
    format="[%(asctime)s] %(levelname)s %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("tradex.adaptive_bot")

logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)


# ── Config (depuis config.py / .env) ──────────────────────────────────────────

ADT_TRADING_PAIRS: list[str]   = config.ADT_TRADING_PAIRS
ADT_ALLOCATED_BALANCE: float   = config.ADT_ALLOCATED_BALANCE
ADT_MAX_POSITIONS: int         = config.ADT_MAX_POSITIONS
ADT_POLLING_SECONDS: int       = config.ADT_POLLING_SECONDS
ADT_HEARTBEAT_SECONDS: int     = config.ADT_HEARTBEAT_SECONDS

ADT_BULL_ALLOC_PCT: float      = config.ADT_BULL_ALLOC_PCT
ADT_BULL_SL_PCT: float         = config.ADT_BULL_SL_PCT
ADT_BULL_TRAIL_PCT: float      = config.ADT_BULL_TRAIL_PCT
ADT_BULL_TP_PCT: float         = config.ADT_BULL_TP_PCT
ADT_BULL_PYRAMID_ALLOC: float  = config.ADT_BULL_PYRAMID_ALLOC
ADT_DAILY_DD_MAX: float        = config.ADT_DAILY_DD_MAX
ADT_COOLDOWN_BARS: int         = config.ADT_COOLDOWN_BARS
ADT_LOG_CANDLE: bool           = config.ADT_LOG_CANDLE

EXCHANGE_NAME = "binance-adaptive"
ADT_STATE_DIR = os.path.join(os.path.dirname(__file__), "..", "data")

# ── Ranking des paires par espérance de gain (backtest walk-forward)
# Plus le rang est bas, plus la paire est prioritaire quand plusieurs signaux
# se déclenchent simultanément sur le même tick.
PAIR_PRIORITY: dict[str, int] = {
    "SOLUSDC":  1,  # +125%  — meilleure paire Big5
    "BTCUSDC":  2,  # +41%
    "AVAXUSDC": 3,  # +42%  — meilleure candidate
    "NEARUSDC": 4,  # +35%
    "XRPUSDC":  5,  # +26%
    "ETHUSDC":  6,  # +24%
}


# ── Helpers ────────────────────────────────────────────────────────────────────

def _fmt(price: float) -> str:
    if price >= 1000:
        return f"{price:,.4f}"
    if price >= 1:
        return f"{price:.4f}"
    if price >= 0.0001:
        return f"{price:.6f}"
    decimals = 6
    temp = price
    while temp < 0.01 and decimals < 10:
        temp *= 10
        decimals += 1
    return f"{price:.{decimals}f}"


def _base_asset(symbol: str) -> str:
    """Extrait le base asset d'un symbole Binance (ex: BTCUSDC → BTC)."""
    if symbol.endswith("USDC"):
        return symbol[:-4]
    if symbol.endswith("USDT"):
        return symbol[:-4]
    if "-" in symbol:
        return symbol.split("-")[0]
    return symbol


def _floor_to_step(qty: float, step: float) -> float:
    """Arrondit qty au multiple inférieur de step (LOT_SIZE Binance)."""
    if step <= 0:
        return qty
    precision = max(0, round(-math.log10(step)))
    floored = math.floor(qty / step) * step
    return round(floored, precision)


def _format_qty_for_symbol(client: BinanceClient, symbol: str, qty: float) -> str:
    """Formate une quantité en respectant les filtres LOT_SIZE Binance."""
    try:
        filters = client.get_symbol_filters(symbol)
        lot = filters.get("LOT_SIZE", {})
        step = float(lot.get("stepSize", "0.00000001"))
        min_qty = float(lot.get("minQty", "0"))
        floored = _floor_to_step(qty, step)
        if floored < min_qty:
            return "0"
        # Déterminer le nombre de décimales
        prec = max(0, round(-math.log10(step))) if step > 0 else 8
        return f"{floored:.{prec}f}"
    except Exception as e:
        logger.debug("LOT_SIZE fallback pour %s: %s", symbol, e)
        return f"{qty:.8f}"


def _check_min_notional(client: BinanceClient, symbol: str, qty: float, price: float) -> bool:
    """Vérifie que la valeur notionnelle respecte le MIN_NOTIONAL Binance."""
    try:
        filters = client.get_symbol_filters(symbol)
        mn = filters.get("MIN_NOTIONAL", {})
        min_notional = float(mn.get("minNotional", "0"))
        return qty * price >= min_notional
    except Exception:
        return True   # Si échec de récupération, on laisse passer


def _format_price_for_symbol(client: BinanceClient, symbol: str, price: float) -> str:
    """Formate un prix en respectant le PRICE_FILTER (tickSize) de Binance."""
    try:
        filters  = client.get_symbol_filters(symbol)
        pf       = filters.get("PRICE_FILTER", {})
        tick     = float(pf.get("tickSize", "0.01"))
        if tick <= 0:
            tick = 0.01
        prec     = max(0, round(-math.log10(tick))) if tick < 1 else 0
        floored  = math.floor(price / tick) * tick
        return f"{round(floored, prec):.{prec}f}"
    except Exception:
        if price >= 1000:
            return f"{price:.2f}"
        if price >= 1:
            return f"{price:.4f}"
        return f"{price:.8f}"


# ═══════════════════════════════════════════════════════════════════════════
# Position dataclass
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class AdaptivePosition:
    """Position gérée par le bot Adaptive Bull."""

    symbol: str
    entry_price: float
    sl_price: float            # mis à jour par le trailing
    tp_price: float
    size: float                # base asset (ex: BTC)
    cost_usdc: float           # USDC déboursés (déduit du virtual_balance)
    peak_price: float
    entry_order_id: str
    firebase_trade_id: Optional[str]
    opened_at: float
    pyramided: bool = False    # True = déjà pyramidé (1× par trade)
    oco_order_list_id: Optional[int] = None       # OCO de sécurité sur l'exchange
    oco_list_client_id: Optional[str] = None      # listClientOrderId de l'OCO

    def to_dict(self) -> dict:
        return {
            "symbol": self.symbol,
            "entry_price": self.entry_price,
            "sl_price": self.sl_price,
            "tp_price": self.tp_price,
            "size": self.size,
            "cost_usdc": self.cost_usdc,
            "peak_price": self.peak_price,
            "entry_order_id": self.entry_order_id,
            "firebase_trade_id": self.firebase_trade_id,
            "opened_at": self.opened_at,
            "pyramided": self.pyramided,
            "oco_order_list_id": self.oco_order_list_id,
            "oco_list_client_id": self.oco_list_client_id,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "AdaptivePosition":
        return cls(
            symbol=d["symbol"],
            entry_price=d["entry_price"],
            sl_price=d["sl_price"],
            tp_price=d["tp_price"],
            size=d["size"],
            cost_usdc=d.get("cost_usdc", d.get("size_usd", 0.0)),
            peak_price=d.get("peak_price", d["entry_price"]),
            entry_order_id=d.get("entry_order_id", "unknown"),
            firebase_trade_id=d.get("firebase_trade_id"),
            opened_at=d.get("opened_at", 0.0),
            pyramided=d.get("pyramided", False),
            oco_order_list_id=d.get("oco_order_list_id"),
            oco_list_client_id=d.get("oco_list_client_id"),
        )


# ═══════════════════════════════════════════════════════════════════════════
# State persistence
# ═══════════════════════════════════════════════════════════════════════════

class AdaptiveStateStore:
    """Persistance atomique du bot Adaptive Bull."""

    def __init__(self, state_file: str) -> None:
        self._path = Path(state_file).resolve()

    def save(
        self,
        positions: dict[str, AdaptivePosition],
        virtual_balance: float,
        total_pnl: float,
        last_candle_ts_15m: dict[str, int],
        last_candle_ts_1h: dict[str, int],
        cooldowns: dict[str, int],
        daily_dd_ref: float,
        last_day: str,
    ) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        state = {
            "positions": {s: p.to_dict() for s, p in positions.items()},
            "virtual_balance": virtual_balance,
            "total_pnl": total_pnl,
            "last_candle_ts_15m": last_candle_ts_15m,
            "last_candle_ts_1h": last_candle_ts_1h,
            "cooldowns": cooldowns,
            "daily_dd_ref": daily_dd_ref,
            "last_day": last_day,
        }
        tmp = self._path.with_suffix(".json.tmp")
        try:
            with open(tmp, "w") as f:
                json.dump(state, f, indent=2)
            tmp.replace(self._path)
        except Exception as e:
            logger.error("❌ Save state failed: %s", e)
            if tmp.exists():
                try:
                    tmp.unlink()
                except OSError:
                    pass

    def load(self) -> dict:
        if not self._path.exists():
            logger.info("📂 Pas de state Adaptive — démarrage à vide")
            return {}
        try:
            with open(self._path, "r") as f:
                return json.load(f)
        except Exception as e:
            logger.error("❌ Load state Adaptive échoué: %s", e)
            return {}


# ═══════════════════════════════════════════════════════════════════════════
# Bot principal
# ═══════════════════════════════════════════════════════════════════════════

class AdaptiveBullBot:
    """Bot Adaptive Bull — Binance Spot, USDC, MARKET orders, 15m, LONG only."""

    def __init__(self, dry_run: bool = False) -> None:
        self.dry_run = dry_run
        self._running = False

        # Exchange
        self._client = BinanceClient(
            api_key=config.BINANCE_API_KEY,
            secret_key=config.BINANCE_SECRET_KEY,
            base_url=config.BINANCE_BASE_URL,
        )

        # Notifications
        self._telegram = TelegramNotifier(
            bot_token=config.TELEGRAM_BOT_TOKEN,
            chat_id=config.TELEGRAM_CHAT_ID,
            silent=self.dry_run,
        )

        # State
        state_file = os.path.join(ADT_STATE_DIR, "state_adaptive.json")
        self._store = AdaptiveStateStore(state_file)
        self._positions: dict[str, AdaptivePosition] = {}

        # Capital virtuel — initialisé à 0, synchronisé sur le solde USDC réel au démarrage
        self._virtual_balance: float = 0.0

        # Candle tracking
        self._last_candle_ts_15m: dict[str, int] = {}
        self._last_candle_ts_1h:  dict[str, int] = {}
        self._candle_buf_15m: dict[str, list[Candle]] = {s: [] for s in ADT_TRADING_PAIRS}
        self._candle_buf_1h:  dict[str, list[Candle]] = {s: [] for s in ADT_TRADING_PAIRS}

        # Cooldowns post-perte
        self._cooldowns: dict[str, int] = {}   # symbol → bar_ts jusqu'à expiration

        # File de signaux en attente (collectés pendant _process_symbol, exécutés
        # après tri par priorité à la fin du tick)
        self._pending_signals: list[tuple[int, str, BullSignal]] = []

        # Daily circuit-breaker
        self._daily_dd_ref: float = 0.0
        self._last_day: str = ""
        self._last_cleanup_day: str = ""

        # Heartbeat
        self._last_heartbeat: float = 0.0
        self._last_balance_sync: float = 0.0   # dernière synchro solde Binance
        self._tick_count: int = 0

        # Stats
        self._total_trades: int = 0
        self._total_wins: int = 0
        self._total_pnl: float = 0.0

        if dry_run:
            logger.info("🔧 Mode DRY-RUN — aucun ordre ne sera exécuté")

    # ── Run ────────────────────────────────────────────────────────────────

    def run(self) -> None:
        self._running = True
        self._setup_signal_handlers()

        budget_label = f"${ADT_ALLOCATED_BALANCE:.0f} (cap)" if ADT_ALLOCATED_BALANCE > 0 else "100% solde USDC Binance"
        logger.info("═" * 60)
        logger.info("📈 AdaptiveBullBot démarré — 15m BULL TREND FOLLOWING")
        logger.info("   Paires    : %d | %s", len(ADT_TRADING_PAIRS), ", ".join(ADT_TRADING_PAIRS))
        logger.info("   Budget    : %s", budget_label)
        logger.info("   Alloc     : %.0f%%/trade | Max %d position(s)",
                     ADT_BULL_ALLOC_PCT * 100, ADT_MAX_POSITIONS)
        logger.info("   SL: -%.1f%% | Trail: -%.1f%% | TP: +%.1f%%",
                     ADT_BULL_SL_PCT * 100, ADT_BULL_TRAIL_PCT * 100, ADT_BULL_TP_PCT * 100)
        logger.info("   Pyramid   : %.0f%% | DD max jour: %.0f%%",
                     ADT_BULL_PYRAMID_ALLOC * 100, ADT_DAILY_DD_MAX * 100)
        logger.info("   Cooldown  : %d barres 15m | Polling: %ds",
                     ADT_COOLDOWN_BARS, ADT_POLLING_SECONDS)
        logger.info("   Mode      : BULL-ONLY (USDC paires Binance)")
        if self.dry_run:
            logger.info("   ⚠️  DRY-RUN actif")
        logger.info("═" * 60)

        self._initialize()

        try:
            while self._running:
                self._tick()
                time.sleep(ADT_POLLING_SECONDS)
        except KeyboardInterrupt:
            logger.info("Arrêt demandé par l'utilisateur")
        finally:
            self._shutdown()

    def stop(self) -> None:
        self._running = False

    def _setup_signal_handlers(self) -> None:
        signal.signal(signal.SIGTERM, lambda *_: self.stop())
        signal.signal(signal.SIGINT, lambda *_: self.stop())

    # ── Init ───────────────────────────────────────────────────────────────

    def _initialize(self) -> None:
        state = self._store.load()
        if state:
            for sym, d in state.get("positions", {}).items():
                try:
                    self._positions[sym] = AdaptivePosition.from_dict(d)
                except Exception as e:
                    logger.warning("⚠️ Position %s corrompue ignorée: %s", sym, e)

            self._virtual_balance    = state.get("virtual_balance", 0.0)
            self._total_pnl          = state.get("total_pnl", 0.0)
            self._last_candle_ts_15m = state.get("last_candle_ts_15m", {})
            self._last_candle_ts_1h  = state.get("last_candle_ts_1h", {})
            self._cooldowns          = state.get("cooldowns", {})
            self._daily_dd_ref       = state.get("daily_dd_ref", 0.0)
            self._last_day           = state.get("last_day", "")

            logger.info("📂 State chargé: %d positions | vBalance=%.2f",
                        len(self._positions), self._virtual_balance)

        # Toujours synchroniser le virtual_balance sur le solde USDC réel Binance
        # (le USDC des positions ouvertes est déjà converti en crypto — il n'est
        #  plus dans le compte USDC, donc available USDC = capital libre réel)
        self._sync_balance_from_exchange()

        # Réconcilier les positions avec les soldes réels
        self._reconcile_positions()

        # Pré-charger les bougies
        logger.info("── Chargement des bougies initiales ──")
        for symbol in ADT_TRADING_PAIRS:
            self._load_initial_candles(symbol)

        self._save_state()
        logger.info("── Init terminée | %d positions ouvertes ──", len(self._positions))

    def _maybe_sync_balance(self, interval_seconds: float = 300.0) -> None:
        """Synchro périodique (défaut 5 min) du virtual_balance sur le vrai solde USDC Binance.
        Permet de prendre en compte automatiquement les dépôts sans toucher au state.
        """
        if self.dry_run:
            return
        now = time.time()
        if now - self._last_balance_sync < interval_seconds:
            return
        self._sync_balance_from_exchange()

    def _sync_balance_from_exchange(self) -> None:
        """Synchronise le virtual_balance sur le solde USDC réel de Binance.

        - ADT_ALLOCATED_BALANCE = 0  → utilise tout le solde USDC disponible
        - ADT_ALLOCATED_BALANCE > 0  → cap au montant configuré
        """
        try:
            balances = self._client.get_balances()
            usdc_bal = next((b for b in balances if b.currency == "USDC"), None)
            available = usdc_bal.available if usdc_bal else 0.0

            if ADT_ALLOCATED_BALANCE > 0:
                budget = min(available, ADT_ALLOCATED_BALANCE)
                logger.info(
                    "💰 Budget USDC : %.2f (disponible=%.2f, cap=%.0f USDC)",
                    budget, available, ADT_ALLOCATED_BALANCE,
                )
            else:
                budget = available
                logger.info(
                    "💰 Budget USDC : %.2f (solde réel complet Binance)", budget
                )

            self._virtual_balance = budget
            self._last_balance_sync = time.time()
            if self._daily_dd_ref <= 0:
                self._daily_dd_ref = budget
        except Exception as e:
            fallback = ADT_ALLOCATED_BALANCE if ADT_ALLOCATED_BALANCE > 0 else 100.0
            logger.warning(
                "⚠️ Impossible de récupérer le solde USDC — fallback %.0f : %s",
                fallback, e,
            )
            self._virtual_balance = fallback
            self._last_balance_sync = time.time()
            if self._daily_dd_ref <= 0:
                self._daily_dd_ref = fallback

    def _load_initial_candles(self, symbol: str) -> None:
        """Charge 300 bougies 15m + 250 bougies 1H au démarrage."""
        try:
            c15 = self._client.get_candles(symbol, interval=15, limit=300)
            c15 = sorted(c15, key=lambda c: c.timestamp)
            if c15:
                self._candle_buf_15m[symbol] = c15
                self._last_candle_ts_15m[symbol] = c15[-1].timestamp
                logger.info("[%s] %d bougies 15m chargées", symbol, len(c15))
        except Exception as e:
            logger.error("[%s] ❌ Chargement 15m échoué: %s", symbol, e)

        try:
            c1h = self._client.get_candles(symbol, interval=60, limit=250)
            c1h = sorted(c1h, key=lambda c: c.timestamp)
            if c1h:
                self._candle_buf_1h[symbol] = c1h
                self._last_candle_ts_1h[symbol] = c1h[-1].timestamp
                logger.info("[%s] %d bougies 1H chargées", symbol, len(c1h))
        except Exception as e:
            logger.error("[%s] ❌ Chargement 1H échoué: %s", symbol, e)

    def _reconcile_positions(self) -> None:
        """Vérifie les positions locales contre les soldes Binance."""
        if not self._positions:
            return
        try:
            balances = self._client.get_balances()
            balance_map = {b.currency: b for b in balances}
        except Exception as e:
            logger.warning("⚠️ Réconciliation impossible: %s", e)
            return

        removed = []
        for sym, pos in self._positions.items():
            base = _base_asset(sym)
            bal = balance_map.get(base)
            held = (bal.available + bal.reserved) if bal else 0.0
            if held >= pos.size * 0.85:
                logger.info("[%s] ✅ Position confirmée | %.8f %s", sym, held, base)
            else:
                logger.warning(
                    "[%s] ⚠️ Solde %s=%.8f < size=%.8f → position retirée",
                    sym, base, held, pos.size,
                )
                removed.append(sym)

        for sym in removed:
            # Récupérer le capital investi dans le virtual_balance
            pos = self._positions[sym]
            self._virtual_balance += pos.cost_usdc
            del self._positions[sym]

        if removed:
            self._save_state()

    # ── Tick ───────────────────────────────────────────────────────────────

    def _tick(self) -> None:
        self._tick_count += 1
        self._apply_runtime_actions()
        self._update_daily_ref()

        # Phase 1 : traitement de chaque symbole (positions existantes + collecte signaux)
        self._pending_signals.clear()
        for symbol in ADT_TRADING_PAIRS:
            try:
                self._process_symbol(symbol)
            except Exception as e:
                logger.error("[%s] Erreur tick: %s", symbol, e)

        # Phase 2 : exécution des signaux par ordre de priorité
        if self._pending_signals:
            self._pending_signals.sort(key=lambda x: x[0])
            for rank, symbol, signal in self._pending_signals:
                if len(self._positions) >= ADT_MAX_POSITIONS:
                    skipped = [s for _, s, _ in self._pending_signals
                               if s not in self._positions]
                    if skipped:
                        logger.debug(
                            "⏸️  Signaux ignorés (max_positions=%d atteint) : %s",
                            ADT_MAX_POSITIONS, ", ".join(skipped),
                        )
                    break
                self._execute_entry(symbol, signal)

        self._maybe_sync_balance()
        self._maybe_heartbeat()
        self._maybe_daily_tasks()

    def _update_daily_ref(self) -> None:
        """Met à jour la référence du circuit-breaker DD journalier."""
        now_day = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        if now_day != self._last_day:
            self._last_day = now_day
            equity = self._compute_equity()
            self._daily_dd_ref = equity
            self._save_state()

    def _compute_equity(self) -> float:
        """Equity courante = virtual_balance + valeur approximative des positions ouvertes."""
        eq = self._virtual_balance
        for pos in self._positions.values():
            eq += pos.cost_usdc   # approximation basée sur le coût d'entrée
        return eq

    def _daily_dd_ok(self) -> bool:
        """Retourne True si le DD journalier est dans les limites."""
        if self._daily_dd_ref <= 0:
            return True
        equity = self._compute_equity()
        dd = (equity - self._daily_dd_ref) / self._daily_dd_ref
        return dd > -ADT_DAILY_DD_MAX

    # ── Process symbol ─────────────────────────────────────────────────────

    def _process_symbol(self, symbol: str) -> None:
        """Traite un symbole : polling SL/TP/trail + check nouvelle bougie."""

        # ── 1. Gestion de la position ouverte (via ticker live) ──
        pos = self._positions.get(symbol)
        if pos:
            try:
                price = self._client.get_ticker_price(symbol)
                if price > 0:
                    self._manage_position(symbol, pos, price)
                    if symbol not in self._positions:
                        return
            except Exception as e:
                logger.debug("[%s] Ticker échoué: %s", symbol, e)

        # ── 2. Vérifier s'il y a une nouvelle bougie 15m ──
        try:
            candles_15m = self._client.get_candles(symbol, interval=15, limit=300)
            candles_15m = sorted(candles_15m, key=lambda c: c.timestamp)
        except Exception as e:
            logger.debug("[%s] Fetch 15m échoué: %s", symbol, e)
            return

        if not candles_15m:
            return

        latest_ts_15m = candles_15m[-1].timestamp
        prev_ts_15m   = self._last_candle_ts_15m.get(symbol, 0)

        if latest_ts_15m <= prev_ts_15m:
            return  # Pas de nouvelle bougie 15m

        # Nouvelle bougie 15m
        self._last_candle_ts_15m[symbol] = latest_ts_15m
        self._candle_buf_15m[symbol] = candles_15m[-300:]

        # Vérifier également si nouvelle bougie 1H
        try:
            candles_1h = self._client.get_candles(symbol, interval=60, limit=250)
            candles_1h = sorted(candles_1h, key=lambda c: c.timestamp)
            if candles_1h:
                self._last_candle_ts_1h[symbol] = candles_1h[-1].timestamp
                self._candle_buf_1h[symbol] = candles_1h[-250:]
        except Exception as e:
            logger.debug("[%s] Fetch 1H échoué: %s", symbol, e)
            # Utiliser le buffer existant

        # Log diagnostique par bougie
        self._log_candle_state(symbol)

        # ── 3. Check rupture de tendance sur position ouverte ──
        pos_after = self._positions.get(symbol)
        if pos_after:
            buf_15m = self._candle_buf_15m.get(symbol, [])
            if len(buf_15m) >= 210 and is_trend_broken(buf_15m):
                logger.info("[%s] 📉 TREND_BREAK (EMA50 < EMA200 15m) → sortie", symbol)
                try:
                    price = self._client.get_ticker_price(symbol)
                except Exception:
                    price = pos_after.entry_price
                self._close_position(symbol, price, "TREND_BREAK")
                self._save_state()
                return

        # ── 4. Check pyramiding sur position ouverte ──
        pos_after = self._positions.get(symbol)
        if pos_after and not pos_after.pyramided:
            buf_1h  = self._candle_buf_1h.get(symbol, [])
            regime  = detect_regime(buf_1h) if len(buf_1h) >= 220 else Regime.UNKNOWN
            buf_15m = self._candle_buf_15m.get(symbol, [])
            if check_pyramid_entry(buf_15m, regime, pos_after.entry_price):
                try:
                    price = self._client.get_ticker_price(symbol)
                    if price > 0:
                        self._execute_pyramid(symbol, pos_after, price)
                except Exception as e:
                    logger.debug("[%s] Pyramiding échoué: %s", symbol, e)

        # ── 5. Check nouveau signal d'entrée ──
        if symbol not in self._positions:
            self._check_entry_signal(symbol)

        self._save_state()

    # ── Position management ────────────────────────────────────────────────

    def _manage_position(
        self,
        symbol: str,
        pos: AdaptivePosition,
        price: float,
    ) -> None:
        """Gère SL (initial + trailing) et TP sur chaque tick de prix."""

        # Vérifier si l'OCO de sécurité a été déclenché sur l'exchange
        if self._check_oco_triggered(symbol, pos):
            return  # position déjà fermée par l'OCO

        # Trailing stop update
        new_sl, new_peak = update_adaptive_trailing(
            current_price=price,
            peak_price=pos.peak_price,
            current_sl=pos.sl_price,
            bull_trail_pct=ADT_BULL_TRAIL_PCT,
        )
        sl_updated = new_sl > pos.sl_price or new_peak > pos.peak_price
        old_sl = pos.sl_price
        was_below_entry = old_sl < pos.entry_price
        pos.peak_price = new_peak
        if new_sl > pos.sl_price:
            pos.sl_price = new_sl
            locked_pnl = (pos.sl_price - pos.entry_price) * pos.size
            locked_pct = (pos.sl_price - pos.entry_price) / pos.entry_price * 100
            lock_emoji = "🔒" if locked_pnl >= 0 else "⚠️"
            logger.info(
                "[%s] %s Trail SL: %s → %s | peak=%s | verrouillé: %+.2f USDC (%+.1f%%)",
                symbol, lock_emoji, _fmt(old_sl), _fmt(pos.sl_price),
                _fmt(pos.peak_price), locked_pnl, locked_pct,
            )
            # Notification Telegram lors du passage en territoire positif (SL > entry)
            now_above_entry = pos.sl_price > pos.entry_price
            if was_below_entry and now_above_entry:
                self._telegram._send(
                    f"🔒 *Trail SL en positif — {symbol}* 📈 ADAPTIVE BULL\n"
                    f"  Entrée: `{_fmt(pos.entry_price)}` | Peak: `{_fmt(pos.peak_price)}`\n"
                    f"  SL verrouillé: `{_fmt(pos.sl_price)}` → gain min garanti: `{locked_pnl:+.2f} USDC` (`{locked_pct:+.1f}%`)\n"
                    f"[Dashboard]({DASHBOARD_URL})"
                )

        # SL check (initial ou trailing)
        if price <= pos.sl_price:
            reason = "SL" if pos.peak_price <= pos.entry_price * 1.01 else "TRAILING_SL"
            logger.info(
                "[%s] 🛑 %s HIT | prix=%s | SL=%s | peak=%s",
                symbol, reason, _fmt(price), _fmt(pos.sl_price), _fmt(pos.peak_price),
            )
            self._close_position(symbol, price, reason)
            return

        # TP check
        if price >= pos.tp_price:
            logger.info(
                "[%s] 🎯 TP HIT | prix=%s | TP=%s",
                symbol, _fmt(price), _fmt(pos.tp_price),
            )
            self._close_position(symbol, price, "TP")
            return

        if sl_updated:
            self._save_state()

    def _close_position(
        self,
        symbol: str,
        exit_price: float,
        reason: str,
    ) -> None:
        """Ferme une position : MARKET SELL + Firebase + Telegram."""
        pos = self._positions.get(symbol)
        if not pos:
            return

        exit_size = pos.size

        # Raisons OCO_TP / OCO_SL : l'exchange a déjà vendu — skip le MARKET SELL
        oco_already_filled = reason.startswith("OCO_")

        if not self.dry_run and not oco_already_filled:
            # Annuler l'OCO avant de placer le MARKET SELL
            self._cancel_oco_for_position(symbol, pos)

            # Vérifier le solde réel disponible
            exit_size = self._get_real_balance(symbol, pos.size)
            if exit_size <= 0:
                logger.warning("[%s] Solde réel = 0 — position retirée localement", symbol)
                self._virtual_balance += pos.cost_usdc
                del self._positions[symbol]
                self._save_state()
                return

            qty_str = _format_qty_for_symbol(self._client, symbol, exit_size)
            if qty_str == "0" or float(qty_str) <= 0:
                logger.warning("[%s] Quantité trop faible pour le SELL — skip", symbol)
                return

            order_id = str(uuid.uuid4())[:8]
            try:
                resp = self._client.place_market_order(
                    symbol=symbol,
                    side="SELL",
                    quantity=qty_str,
                    client_order_id=order_id,
                )
                # Récupérer le prix de fill réel
                fills = resp.get("fills", [])
                if fills:
                    avg = sum(float(f["price"]) * float(f["qty"]) for f in fills)
                    total_qty = sum(float(f["qty"]) for f in fills)
                    exit_price = avg / total_qty if total_qty > 0 else exit_price
                exit_size = float(resp.get("executedQty", exit_size))
                logger.info(
                    "[%s] ✅ SELL MARKET exécuté | qty=%s | prix=%s",
                    symbol, qty_str, _fmt(exit_price),
                )
            except Exception as e:
                logger.error("[%s] ❌ SELL MARKET échoué: %s", symbol, e)
                self._telegram.notify_error(f"⚠️ Adaptive SELL {symbol} échoué: {e}")
                return
        elif self.dry_run:
            logger.info(
                "[DRY-RUN] SELL %s | exit=%s | reason=%s",
                symbol, _fmt(exit_price), reason,
            )
        else:
            # OCO déjà fillé sur l'exchange — pas de MARKET SELL
            logger.info(
                "[%s] 🔗 Fermeture OCO confirmée | exit=%s | reason=%s",
                symbol, _fmt(exit_price), reason,
            )

        # PnL
        fee_rate = config.BINANCE_TAKER_FEE
        proceeds = exit_size * exit_price * (1.0 - fee_rate)
        pnl = proceeds - pos.cost_usdc
        pnl_pct = pnl / pos.cost_usdc * 100 if pos.cost_usdc > 0 else 0.0
        pnl_emoji = "🟢" if pnl >= 0 else "🔴"

        # Stats
        self._total_trades += 1
        is_win = pnl >= 0
        if is_win:
            self._total_wins += 1
        self._total_pnl += pnl

        # Cooldown post-perte
        if not is_win:
            bar_ts = self._last_candle_ts_15m.get(symbol, 0)
            self._cooldowns[symbol] = bar_ts + ADT_COOLDOWN_BARS * 15 * 60 * 1000

        # Récupérer le capital dans le virtual_balance
        self._virtual_balance += proceeds

        logger.info(
            "[%s] %s CLOSE | entry=%s → exit=%s | PnL=$%+.2f (%+.1f%%) | %s",
            symbol, pnl_emoji, _fmt(pos.entry_price), _fmt(exit_price),
            pnl, pnl_pct, reason,
        )

        # Telegram
        base = _base_asset(symbol)
        self._telegram._send(
            f"{pnl_emoji} *Position fermée – {symbol}* 📈 ADAPTIVE BULL\n"
            f"  Raison: {reason}\n"
            f"  Entrée: `{_fmt(pos.entry_price)}` → Sortie: `{_fmt(exit_price)}`\n"
            f"  P&L: `{pnl:+.2f} USDC` (`{pnl_pct:+.1f}%`)\n"
            f"  Peak: `{_fmt(pos.peak_price)}` | Size: `{exit_size:.8f} {base}`\n"
            f"[Dashboard]({DASHBOARD_URL})"
        )

        # Firebase
        if not self.dry_run and pos.firebase_trade_id:
            try:
                fb_pos = Position(
                    symbol=symbol,
                    side=OrderSide.BUY,
                    entry_price=pos.entry_price,
                    sl_price=pos.sl_price,
                    size=exit_size,
                    venue_order_id=pos.entry_order_id,
                    status=PositionStatus.CLOSED,
                    strategy=StrategyType.ADAPTIVE,
                    tp_price=pos.tp_price,
                    pnl=pnl,
                )
                log_trade_closed(
                    trade_id=pos.firebase_trade_id,
                    position=fb_pos,
                    exit_price=exit_price,
                    reason=reason,
                    fill_type="taker",
                    equity_after=self._virtual_balance,
                    actual_exit_size=exit_size,
                )
            except Exception as e:
                logger.warning("🔥 Firebase log_trade_closed échoué: %s", e)

        del self._positions[symbol]
        self._save_state()

    # ── OCO safety net ────────────────────────────────────────────────────

    def _place_oco_for_position(self, symbol: str, pos: AdaptivePosition) -> None:
        """
        Pose un OCO SELL (SL + TP) sur l'exchange après un BUY MARKET.
        Sert de filet de sécurité si le bot crashe — la position sera
        fermée automatiquement par Binance sans intervention du bot.
        """
        if self.dry_run or pos.size <= 0:
            return
        try:
            qty_str    = _format_qty_for_symbol(self._client, symbol, pos.size)
            sl_stop    = _format_price_for_symbol(self._client, symbol, pos.sl_price)
            sl_limit   = _format_price_for_symbol(
                self._client, symbol, pos.sl_price * 0.995  # 0.5% slip tolléré
            )
            tp_str     = _format_price_for_symbol(self._client, symbol, pos.tp_price)
            list_cid   = f"adt-{symbol[:3]}-{str(uuid.uuid4())[:6]}"
            resp       = self._client.place_oco_order(
                symbol=symbol,
                side="SELL",
                quantity=qty_str,
                tp_price=tp_str,
                sl_stop_price=sl_stop,
                sl_limit_price=sl_limit,
                list_client_order_id=list_cid,
            )
            pos.oco_order_list_id = resp.get("orderListId")
            pos.oco_list_client_id = list_cid
            logger.info(
                "[%s] 🔗 OCO posé | SL=%s | TP=%s | listId=%s",
                symbol, sl_stop, tp_str, pos.oco_order_list_id,
            )
        except Exception as e:
            logger.warning(
                "[%s] ⚠️ OCO placement échoué (polling SL/TP actif): %s", symbol, e
            )

    def _cancel_oco_for_position(self, symbol: str, pos: AdaptivePosition) -> None:
        """Annule l'OCO de sécurité avant de placer un MARKET SELL."""
        if self.dry_run or not pos.oco_order_list_id:
            return
        try:
            self._client.cancel_order_list(
                symbol=symbol,
                order_list_id=pos.oco_order_list_id,
            )
            logger.info("[%s] 🗑️ OCO annulé (listId=%s)", symbol, pos.oco_order_list_id)
        except Exception as e:
            # L'OCO peut déjà être terminé (SL/TP déjà déclenché) — pas critique
            logger.debug("[%s] OCO cancel (peut être déjà terminé): %s", symbol, e)
        finally:
            pos.oco_order_list_id  = None
            pos.oco_list_client_id = None

    def _check_oco_triggered(self, symbol: str, pos: AdaptivePosition) -> bool:
        """
        Vérifie si l'OCO a été déclenché sur l'exchange (hors polling du bot).
        Si oui, réconcilie la position localement et retourne True.
        """
        if self.dry_run or not pos.oco_order_list_id:
            return False
        try:
            resp        = self._client.get_order_list(pos.oco_order_list_id)
            list_status = resp.get("listOrderStatus", "")
            if list_status != "ALL_DONE":
                return False

            # L'OCO est terminé — identifier quelle jambe a été exécutée
            orders = resp.get("orders", [])
            exit_price = 0.0
            reason     = "OCO_TRIGGERED"

            for order_ref in orders:
                try:
                    order = self._client.get_order(
                        symbol=symbol,
                        order_id=int(order_ref["orderId"]),
                    )
                    if order.get("status") != "FILLED":
                        continue
                    order_type = order.get("type", "")
                    filled_qty  = float(order.get("executedQty", 0))
                    if filled_qty <= 0:
                        continue
                    # Prix fill moyen
                    cum_quote  = float(order.get("cummulativeQuoteQty", 0))
                    fill_price = cum_quote / filled_qty if filled_qty > 0 else 0.0
                    if fill_price <= 0:
                        fill_price = float(order.get("price", pos.entry_price))
                    exit_price = fill_price
                    # Identifier TP vs SL
                    if "TAKE_PROFIT" in order_type or "LIMIT_MAKER" in order_type:
                        reason = "OCO_TP"
                    else:
                        reason = "OCO_SL"
                    break
                except Exception:
                    continue

            if exit_price <= 0:
                exit_price = pos.sl_price if reason == "OCO_SL" else pos.tp_price

            logger.info(
                "[%s] 🔗 OCO déclenché sur l'exchange | reason=%s | prix=~%s",
                symbol, reason, _fmt(exit_price),
            )
            # L'OCO a déjà vendu → on NE remet PAS de MARKET SELL
            # On passe reason spéciale pour court-circuiter le SELL dans _close_position
            pos.oco_order_list_id  = None
            pos.oco_list_client_id = None
            self._close_position(symbol, exit_price, reason)
            return True

        except Exception as e:
            logger.debug("[%s] Check OCO échoué: %s", symbol, e)
            return False

    def _get_real_balance(self, symbol: str, expected_size: float) -> float:
        """Retourne le solde réellement disponible pour le base asset."""
        try:
            balances  = self._client.get_balances()
            base      = _base_asset(symbol)
            bal       = next((b for b in balances if b.currency == base), None)
            available = bal.available if bal else 0.0
            return min(available, expected_size * 1.01)   # léger buffer pour fees
        except Exception as e:
            logger.debug("[%s] get_real_balance échoué: %s", symbol, e)
            return expected_size

    # ── Entry logic ─────────────────────────────────────────────────────────

    def _check_entry_signal(self, symbol: str) -> None:
        """Vérifie les conditions d'entrée BULL sur la dernière bougie 15m."""

        # Guards globaux
        if symbol in self._positions:
            return
        if len(self._positions) >= ADT_MAX_POSITIONS:
            return
        if self._virtual_balance < 10.0:
            return
        if not self._daily_dd_ok():
            return

        # Guard: cooldown post-perte
        bar_ts = self._last_candle_ts_15m.get(symbol, 0)
        if bar_ts <= self._cooldowns.get(symbol, 0):
            return

        # Régime 1H
        buf_1h = self._candle_buf_1h.get(symbol, [])
        if len(buf_1h) < 220:
            return
        regime = detect_regime(buf_1h)

        if regime != Regime.BULL:
            logger.debug("[%s] Régime=%s — pas de signal", symbol, regime.value)
            return

        # Signal d'entrée 15m
        buf_15m = self._candle_buf_15m.get(symbol, [])
        signal = check_bull_entry(
            buf_15m,
            regime,
            bull_rsi_min=50.0,
            bull_rsi_max=65.0,
            bull_sl_pct=ADT_BULL_SL_PCT,
            bull_tp_pct=ADT_BULL_TP_PCT,
        )

        if signal is None:
            return

        rank = PAIR_PRIORITY.get(symbol, 99)
        logger.info(
            "[%s] 📈 BULL SIGNAL | close=%s | EMA50=%s | EMA200=%s | RSI=%.1f | priorité=%d",
            symbol, _fmt(signal.entry_price), _fmt(signal.ema50), _fmt(signal.ema200), signal.rsi, rank,
        )
        self._pending_signals.append((rank, symbol, signal))

    def _execute_entry(self, symbol: str, signal: BullSignal) -> None:
        """Exécute un ordre d'entrée BUY MARKET."""

        # Sizing : fraction du virtual_balance
        cost = self._virtual_balance * ADT_BULL_ALLOC_PCT
        fee_rate = config.BINANCE_TAKER_FEE
        cost_after_fee = cost * (1.0 - fee_rate)
        size = cost_after_fee / signal.entry_price

        # Vérifier min notional
        if cost < 5.0:
            logger.info("[%s] ⏭️ Budget trop faible ($%.2f) — skip", symbol, cost)
            return

        # Utiliser le solde USDC réel comme source de vérité (dépôts inclus)
        if not self.dry_run:
            try:
                balances   = self._client.get_balances()
                usdc_bal   = next((b for b in balances if b.currency == "USDC"), None)
                available  = usdc_bal.available if usdc_bal else 0.0
                # Mettre à jour le virtual_balance si le solde réel est plus élevé
                # (ex: dépôt depuis l'extérieur)
                if available > self._virtual_balance + 1.0:
                    logger.info(
                        "[%s] 💰 Solde USDC sync : %.2f → %.2f",
                        symbol, self._virtual_balance, available,
                    )
                    self._virtual_balance = available
                    self._last_balance_sync = time.time()
                    cost = self._virtual_balance * ADT_BULL_ALLOC_PCT
                    cost_after_fee = cost * (1.0 - fee_rate)
                    size = cost_after_fee / signal.entry_price
            except Exception as e:
                logger.error("[%s] Impossible de récupérer le solde USDC: %s", symbol, e)
                return

            if available < cost * 0.99:
                logger.info(
                    "[%s] ⏭️ Solde USDC insuffisant ($%.2f < $%.2f) — skip",
                    symbol, available, cost,
                )
                return

            if not _check_min_notional(self._client, symbol, size, signal.entry_price):
                logger.info("[%s] ⏭️ Notional trop faible — skip", symbol)
                return

        actual_entry  = signal.entry_price
        order_id      = "dry-run"

        if not self.dry_run:
            qty_str = _format_qty_for_symbol(self._client, symbol, size)
            if qty_str == "0" or float(qty_str) <= 0:
                logger.info("[%s] ⏭️ Quantité trop faible — skip", symbol)
                return

            try:
                resp = self._client.place_market_order(
                    symbol=symbol,
                    side="BUY",
                    quantity=qty_str,
                    client_order_id=str(uuid.uuid4())[:8],
                )
                order_id = str(resp.get("orderId", "unknown"))
                fills = resp.get("fills", [])
                if fills:
                    avg = sum(float(f["price"]) * float(f["qty"]) for f in fills)
                    total_qty = sum(float(f["qty"]) for f in fills)
                    actual_entry = avg / total_qty if total_qty > 0 else actual_entry
                size = float(resp.get("executedQty", size))
                logger.info(
                    "[%s] 📈 ✅ BUY MARKET exécuté | qty=%s @ %s",
                    symbol, qty_str, _fmt(actual_entry),
                )
            except Exception as e:
                logger.error("[%s] ❌ BUY MARKET échoué: %s", symbol, e)
                self._telegram.notify_error(f"⚠️ Adaptive BUY {symbol} échoué: {e}")
                return
        else:
            logger.info(
                "[DRY-RUN] BUY %s | entry=%s | SL=%s | TP=%s | size=%.8f ($%.2f)",
                symbol, _fmt(signal.entry_price), _fmt(signal.sl_price),
                _fmt(signal.tp_price), size, cost,
            )

        # Recalculer SL/TP basés sur le prix fill réel
        real_sl = actual_entry * (1.0 - ADT_BULL_SL_PCT)
        real_tp = actual_entry * (1.0 + ADT_BULL_TP_PCT)
        real_cost = size * actual_entry * (1.0 + fee_rate)

        # Créer la position
        pos = AdaptivePosition(
            symbol=symbol,
            entry_price=actual_entry,
            sl_price=real_sl,
            tp_price=real_tp,
            size=size,
            cost_usdc=real_cost,
            peak_price=actual_entry,
            entry_order_id=order_id,
            firebase_trade_id=None,
            opened_at=time.time(),
        )
        self._positions[symbol] = pos
        self._virtual_balance  -= real_cost

        # Poser l'OCO de sécurité sur l'exchange (filet si le bot crashe)
        self._place_oco_for_position(symbol, pos)

        # Telegram
        base   = _base_asset(symbol)
        sl_pct = ADT_BULL_SL_PCT * 100
        tp_pct = ADT_BULL_TP_PCT * 100

        self._telegram._send(
            f"📈 *BUY déclenché – {symbol}* ADAPTIVE BULL\n"
            f"  Entrée: `{_fmt(actual_entry)}` | SL: `{_fmt(real_sl)}` (-{sl_pct:.1f}%)\n"
            f"  TP: `{_fmt(real_tp)}` (+{tp_pct:.1f}%) | Trail: -{ADT_BULL_TRAIL_PCT*100:.1f}% du peak\n"
            f"  Size: `{size:.8f} {base}` (`${real_cost:.2f} USDC`)\n"
            f"  RSI: {signal.rsi:.1f} | EMA50: `{_fmt(signal.ema50)}`\n"
            f"  Budget restant: `${self._virtual_balance:.2f}` / `${ADT_ALLOCATED_BALANCE:.0f}`\n"
            f"[Dashboard]({DASHBOARD_URL})"
        )

        # Firebase
        if not self.dry_run:
            try:
                fb_pos = Position(
                    symbol=symbol,
                    side=OrderSide.BUY,
                    entry_price=actual_entry,
                    sl_price=real_sl,
                    size=size,
                    venue_order_id=order_id,
                    status=PositionStatus.OPEN,
                    strategy=StrategyType.ADAPTIVE,
                    tp_price=real_tp,
                )
                trade_id = log_trade_opened(
                    position=fb_pos,
                    fill_type="taker",
                    maker_wait_seconds=0,
                    risk_pct=ADT_BULL_ALLOC_PCT,
                    risk_amount_usd=real_cost,
                    fiat_balance=self._virtual_balance,
                    current_equity=self._compute_equity(),
                    portfolio_risk_before=0.0,
                    exchange=EXCHANGE_NAME,
                    dry_run=self.dry_run,
                )
                pos.firebase_trade_id = trade_id
            except Exception as e:
                logger.warning("🔥 Firebase log_trade_opened échoué: %s", e)

        self._save_state()

    # ── Pyramiding ─────────────────────────────────────────────────────────

    def _execute_pyramid(
        self,
        symbol: str,
        pos: AdaptivePosition,
        current_price: float,
    ) -> None:
        """Ajoute sur une position gagnante (15% du virtual_balance restant)."""
        extra_cost = self._virtual_balance * ADT_BULL_PYRAMID_ALLOC
        if extra_cost < 5.0:
            return

        fee_rate = config.BINANCE_TAKER_FEE
        extra_size = extra_cost * (1.0 - fee_rate) / current_price

        if not self.dry_run:
            if not _check_min_notional(self._client, symbol, extra_size, current_price):
                return

            qty_str = _format_qty_for_symbol(self._client, symbol, extra_size)
            if qty_str == "0" or float(qty_str) <= 0:
                return

            try:
                resp = self._client.place_market_order(
                    symbol=symbol, side="BUY", quantity=qty_str,
                    client_order_id=str(uuid.uuid4())[:8],
                )
                fills = resp.get("fills", [])
                if fills:
                    avg = sum(float(f["price"]) * float(f["qty"]) for f in fills)
                    total_qty = sum(float(f["qty"]) for f in fills)
                    current_price = avg / total_qty if total_qty > 0 else current_price
                extra_size = float(resp.get("executedQty", extra_size))
            except Exception as e:
                logger.error("[%s] ❌ PYRAMID BUY échoué: %s", symbol, e)
                return
        else:
            logger.info(
                "[DRY-RUN] PYRAMID %s | +%.8f @ %s | extra=$%.2f",
                symbol, extra_size, _fmt(current_price), extra_cost,
            )

        extra_real_cost = extra_size * current_price * (1.0 + fee_rate)

        # Mettre à jour la position (moyenne pondérée)
        total_size = pos.size + extra_size
        total_cost = pos.cost_usdc + extra_real_cost
        pos.entry_price = total_cost / total_size
        pos.size        = total_size
        pos.cost_usdc   = total_cost
        pos.pyramided   = True
        # Recalculer TP basé sur le nouveau prix moyen
        pos.tp_price = pos.entry_price * (1.0 + ADT_BULL_TP_PCT)

        self._virtual_balance -= extra_real_cost

        # Remplacer l'OCO avec la nouvelle quantité totale et le nouveau TP
        self._cancel_oco_for_position(symbol, pos)
        self._place_oco_for_position(symbol, pos)

        base = _base_asset(symbol)
        logger.info(
            "[%s] 📊 PYRAMID | +%.8f @ %s | avg_entry=%s | total=%.8f %s",
            symbol, extra_size, _fmt(current_price), _fmt(pos.entry_price), total_size, base,
        )

        self._telegram._send(
            f"📊 *Pyramiding – {symbol}* ADAPTIVE BULL\n"
            f"  +`{extra_size:.8f} {base}` @ `{_fmt(current_price)}`\n"
            f"  Avg entry: `{_fmt(pos.entry_price)}` | Total: `{total_size:.8f} {base}`\n"
            f"  SL: `{_fmt(pos.sl_price)}` | TP: `{_fmt(pos.tp_price)}`\n"
            f"[Dashboard]({DASHBOARD_URL})"
        )

        self._save_state()

    # ── Heartbeat ──────────────────────────────────────────────────────────

    def _maybe_heartbeat(self) -> None:
        now = time.time()
        if now - self._last_heartbeat < ADT_HEARTBEAT_SECONDS:
            return
        self._last_heartbeat = now

        now_utc = datetime.now(timezone.utc)
        equity  = self._compute_equity()
        open_pos = len(self._positions)
        wr = (self._total_wins / self._total_trades * 100) if self._total_trades > 0 else 0.0

        # Régimes courants
        regime_summary = []
        for sym in ADT_TRADING_PAIRS:
            buf = self._candle_buf_1h.get(sym, [])
            r   = detect_regime(buf) if len(buf) >= 220 else Regime.UNKNOWN
            regime_summary.append(f"{_base_asset(sym)}:{r.value[:1]}")

        pos_lines = []
        total_latent_pnl = 0.0
        for sym, pos in self._positions.items():
            try:
                price = self._client.get_ticker_price(sym)
            except Exception:
                price = pos.entry_price
            pnl_pct = (price - pos.entry_price) / pos.entry_price * 100 if pos.entry_price > 0 else 0
            pnl_usdc = (price - pos.entry_price) * pos.size
            total_latent_pnl += pnl_usdc
            emoji = "🟢" if pnl_pct >= 0 else "🔴"
            # Gain verrouillé par le trailing SL
            locked_pnl = (pos.sl_price - pos.entry_price) * pos.size
            locked_pct = (pos.sl_price - pos.entry_price) / pos.entry_price * 100 if pos.entry_price > 0 else 0
            if locked_pnl > 0:
                lock_str = f" | 🔒 min `{locked_pnl:+.2f}$` (`{locked_pct:+.1f}%`)"
            elif locked_pnl < -0.01:
                lock_str = f" | ⚠️ SL `{_fmt(pos.sl_price)}`"
            else:
                lock_str = f" | SL `{_fmt(pos.sl_price)}`"
            pos_lines.append(
                f"  {emoji} `{sym}` `{pnl_pct:+.1f}%` (`{pnl_usdc:+.2f}$`) | peak `{_fmt(pos.peak_price)}`{lock_str}"
                + (" 🔶pyramided" if pos.pyramided else "")
            )

        pnl_closed_emoji = "🟢" if self._total_pnl >= 0 else "🔴"
        pnl_latent_emoji = "🟢" if total_latent_pnl >= 0 else "🔴"

        lines = [
            f"💓 *ADAPTIVE BULL Heartbeat* 📈",
            f"  Budget: `${equity:.2f}` / `${ADT_ALLOCATED_BALANCE:.0f}`",
            f"  Pos: `{open_pos}/{ADT_MAX_POSITIONS}` | Trades: {self._total_trades} | WR: `{wr:.0f}%`",
            f"  {pnl_closed_emoji} PnL clôturé: `{self._total_pnl:+.2f} USDC`",
        ]
        if open_pos > 0:
            lines.append(f"  {pnl_latent_emoji} PnL latent: `{total_latent_pnl:+.2f} USDC`")
        lines.append(f"  Régimes: {' '.join(regime_summary)}")
        if pos_lines:
            lines.append("")
            lines.extend(pos_lines)
        lines.append(f"  🕐 `{now_utc.strftime('%H:%M UTC')}`")
        lines.append(f"[Dashboard]({DASHBOARD_URL})")

        self._telegram._send("\n".join(lines))

        if not self.dry_run:
            try:
                fb_log_heartbeat(
                    exchange=EXCHANGE_NAME,
                    open_positions=open_pos,
                    total_equity=equity,
                    total_risk_pct=0.0,
                    pairs_count=len(ADT_TRADING_PAIRS),
                )
            except Exception:
                pass

    # ── Logging diagnostique par bougie ──────────────────────────────────────

    _REGIME_EMOJI: dict[str, str] = {
        "BULL": "🟢", "BEAR": "🔴", "RANGE": "🟡",
        "STAGNATION": "⚪", "UNKNOWN": "❓",
    }

    def _log_candle_state(self, symbol: str) -> None:
        """
        Log lisible à chaque nouvelle bougie 15m clôturée.
        Affiche régime 1H (score + indicateurs) + conditions signal ou état de position.
        Désactivable via ADT_LOG_CANDLE=false dans .env.
        """
        if not ADT_LOG_CANDLE:
            return

        buf_1h  = self._candle_buf_1h.get(symbol, [])
        buf_15m = self._candle_buf_15m.get(symbol, [])
        if len(buf_1h) < 220 or len(buf_15m) < 250:
            return

        ts  = self._last_candle_ts_15m.get(symbol, 0)
        dt  = datetime.fromtimestamp(ts / 1000, tz=timezone.utc).strftime("%H:%M") if ts else "--:--"
        rdebug = detect_regime_debug(buf_1h)
        price  = buf_15m[-1].close
        rem    = self._REGIME_EMOJI.get(rdebug.regime.value, "❓")
        ck     = lambda ok: "✅" if ok else "❌"

        pos = self._positions.get(symbol)

        if pos:
            pnl_pct = (price / pos.entry_price - 1) * 100
            pnl_usd = pnl_pct / 100 * pos.cost_usdc
            sign    = "+" if pnl_pct >= 0 else ""
            trend_ok = not is_trend_broken(buf_15m) if len(buf_15m) >= 210 else True
            logger.info(
                "[%s] 🕯️ %s UTC  %s %s %d/5  ADX=%.1f  RSI(1H)=%.1f\n"
                "       💼 entr=%s  prix=%s  (%s%.2f%% / %s$%.2f)\n"
                "       peak=%s  🔒SL=%s  🎯TP=%s  trend:%s",
                symbol, dt, rem, rdebug.regime.value, rdebug.bull_score,
                rdebug.adx, rdebug.rsi_1h,
                _fmt(pos.entry_price), _fmt(price),
                sign, pnl_pct, sign, abs(pnl_usd),
                _fmt(pos.peak_price), _fmt(pos.sl_price), _fmt(pos.tp_price),
                "✅" if trend_ok else "⚠️ BRISÉ",
            )
            return

        if rdebug.regime != Regime.BULL:
            logger.info(
                "[%s] 🕯️ %s UTC  %s %s %d/5  ADX=%.1f RSI(1H)=%.1f  ⏭️ skip",
                symbol, dt, rem, rdebug.regime.value, rdebug.bull_score,
                rdebug.adx, rdebug.rsi_1h,
            )
            return

        # Régime BULL → détailler les conditions d'entrée
        in_cooldown = ts <= self._cooldowns.get(symbol, 0)
        max_pos_ok  = len(self._positions) < ADT_MAX_POSITIONS
        budget_ok   = self._virtual_balance >= 10.0
        dd_ok       = self._daily_dd_ok()

        edebug = check_bull_entry_debug(
            buf_15m, rdebug.regime,
            bull_rsi_min=50.0, bull_rsi_max=65.0,
            bull_sl_pct=ADT_BULL_SL_PCT, bull_tp_pct=ADT_BULL_TP_PCT,
        )

        logger.info(
            "[%s] 🕯️ %s UTC  %s BULL %d/5  ADX=%.1f RSI(1H)=%.1f  prix=%s EMA50=%s EMA200=%s",
            symbol, dt, rem, rdebug.bull_score, rdebug.adx, rdebug.rsi_1h,
            _fmt(price), _fmt(edebug.ema50), _fmt(edebug.ema200),
        )

        if in_cooldown:
            extra = "⏳ cooldown"
        elif not max_pos_ok:
            extra = f"⏳ max pos ({len(self._positions)}/{ADT_MAX_POSITIONS})"
        elif not budget_ok:
            extra = f"⏳ budget (${self._virtual_balance:.2f})"
        elif not dd_ok:
            extra = "⛔ DD journalier"
        elif edebug.signal:
            extra = "🚀 SIGNAL → entrée !"
        else:
            first_fail = next(
                (name for name, ok in [
                    ("GoldenCross",  edebug.cond_golden_cross),
                    ("Prix>EMA50",   edebug.cond_price_ema50),
                    ("RSI zone",     edebug.cond_rsi_range),
                    ("RSI↑3",        edebug.cond_rsi_rising),
                    ("Slope",        edebug.cond_slope),
                    ("Pullback",     edebug.cond_pullback),
                    ("Bougie",       edebug.cond_bull_candle),
                ] if not ok
                ), "?",
            )
            extra = f"❌ bloqué → {first_fail}"

        logger.info(
            "[%s]   %s CrossEMA  %s Prix>50  %s RSI=%.0f(50-65)  %s RSI↑  %s Slope(%+.3f%%)  %s Pullback  %s Bougie  →  %s",
            symbol,
            ck(edebug.cond_golden_cross),
            ck(edebug.cond_price_ema50),
            ck(edebug.cond_rsi_range), edebug.rsi,
            ck(edebug.cond_rsi_rising),
            ck(edebug.cond_slope), edebug.slope_pct * 100,
            ck(edebug.cond_pullback),
            ck(edebug.cond_bull_candle),
            extra,
        )

    # ── Daily tasks ────────────────────────────────────────────────────────

    def _maybe_daily_tasks(self) -> None:
        if self.dry_run:
            return
        now_day = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        if now_day == self._last_cleanup_day:
            return
        self._last_cleanup_day = now_day
        try:
            fb_cleanup_events()
        except Exception:
            pass

    # ── Runtime actions (Telegram commands) ───────────────────────────────

    def _apply_runtime_actions(self) -> None:
        actions = get_pending_runtime_actions("adaptive")
        if not actions:
            return

        for action in actions:
            action_id = str(action.get("_id", ""))
            kind      = str(action.get("action", "")).lower().strip()
            symbol    = str(action.get("symbol", "")).upper().strip()
            value     = action.get("value")

            try:
                if kind == "close":
                    targets = list(self._positions.keys()) if symbol == "ALL" else [symbol]
                    closed = 0
                    for sym in targets:
                        if sym not in self._positions:
                            continue
                        try:
                            price = self._client.get_ticker_price(sym)
                        except Exception:
                            price = self._positions[sym].entry_price
                        self._close_position(sym, price, "Manual close (Telegram)")
                        closed += 1
                    mark_runtime_action_status(action_id, "done", f"manual close ({closed})")
                    continue

                if kind == "set_sl":
                    pos = self._positions.get(symbol)
                    if not pos:
                        mark_runtime_action_status(action_id, "failed", "position introuvable")
                        continue
                    try:
                        new_sl = float(str(value))
                    except Exception:
                        mark_runtime_action_status(action_id, "failed", "price invalide")
                        continue
                    if new_sl <= 0:
                        mark_runtime_action_status(action_id, "failed", "price invalide")
                        continue
                    try:
                        price = self._client.get_ticker_price(symbol)
                    except Exception:
                        price = pos.entry_price
                    if new_sl >= price:
                        mark_runtime_action_status(action_id, "failed", "SL doit être sous le prix")
                        continue
                    pos.sl_price = new_sl
                    self._save_state()
                    mark_runtime_action_status(action_id, "done", "set_sl appliqué")
                    continue

                mark_runtime_action_status(action_id, "failed", "action inconnue")
            except Exception as e:
                mark_runtime_action_status(action_id, "failed", f"erreur: {e}")

    # ── Persistence ────────────────────────────────────────────────────────

    def _save_state(self) -> None:
        try:
            self._store.save(
                positions=self._positions,
                virtual_balance=self._virtual_balance,
                total_pnl=self._total_pnl,
                last_candle_ts_15m=self._last_candle_ts_15m,
                last_candle_ts_1h=self._last_candle_ts_1h,
                cooldowns=self._cooldowns,
                daily_dd_ref=self._daily_dd_ref,
                last_day=self._last_day,
            )
        except Exception as e:
            logger.error("❌ _save_state échoué: %s", e)

    # ── Shutdown ───────────────────────────────────────────────────────────

    def _shutdown(self) -> None:
        logger.info("📴 AdaptiveBullBot arrêté | %d positions ouvertes", len(self._positions))
        self._save_state()


# ═══════════════════════════════════════════════════════════════════════════
# Entrypoint
# ═══════════════════════════════════════════════════════════════════════════

def main() -> None:
    parser = argparse.ArgumentParser(description="AdaptiveBullBot — Binance Spot, USDC, 15m")
    parser.add_argument("--dry-run", action="store_true", help="Mode dry-run (pas d'ordres réels)")
    args = parser.parse_args()

    bot = AdaptiveBullBot(dry_run=args.dry_run)
    bot.run()


if __name__ == "__main__":
    main()
