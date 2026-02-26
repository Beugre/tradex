"""
Boucle principale du bot TradeX — Binance Breakout Volatility Expansion.

Stratégie complémentaire au bot RANGE : gagne quand le range casse.
LONG ONLY — Adaptive Trailing Stop — Kill-Switch mensuel.

Architecture :
  - Même infrastructure que bot_binance.py (BinanceClient, Firebase, Telegram)
  - État séparé (state_binance_breakout.json)
  - Capital alloué séparé (BINANCE_BREAKOUT_ALLOCATED_BALANCE)
  - Détection via breakout_detector.py (BB + Donchian + ADX + Volume)
  - Exit via trailing stop adaptatif (pas d'OCO — le trail bouge dynamiquement)

Usage :
    python -m src.bot_binance_breakout              # Production
    python -m src.bot_binance_breakout --dry-run    # Simulation
"""

from __future__ import annotations

import argparse
import logging
import os
import signal
import sys
import time
from datetime import datetime, timedelta, timezone
from typing import Optional

from src import config
from src.core.models import (
    Balance,
    Candle,
    OrderSide,
    Position,
    PositionStatus,
    StrategyType,
    TickerData,
)
from src.core.breakout_detector import (
    BreakoutConfig,
    BreakoutDirection,
    BreakoutSignal,
    atr as compute_atr,
    detect_breakout_signals,
    diagnose_last_candle,
)
from src.core.position_store import PositionStore
from src.core.risk_manager import (
    calculate_position_size,
    check_total_risk_exposure,
    get_total_equity,
)
from src.exchange.binance_client import BinanceClient
from src.exchange.binance_data_provider import BinanceDataProvider
from src.notifications.telegram import TelegramNotifier
from src.firebase.trade_logger import (
    log_trade_opened,
    log_trade_closed,
    log_trailing_sl_update as fb_log_trailing_sl,
    log_event as fb_log_event,
    log_heartbeat as fb_log_heartbeat,
    log_daily_snapshot as fb_log_daily_snapshot,
    cleanup_old_events as fb_cleanup_events,
)


# ── Logging ────────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=getattr(logging, config.LOG_LEVEL, logging.INFO),
    format="[%(asctime)s] %(levelname)s %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("tradex.bot.binance.breakout")

logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)

EXCHANGE_NAME = "binance-breakout"

_STATE_FILE = os.environ.get(
    "TRADEX_BINANCE_BREAKOUT_STATE_FILE",
    os.path.join(os.path.dirname(__file__), "..", "data", "state_binance_breakout.json"),
)


def _fmt(price: float) -> str:
    """Formate un prix de façon lisible."""
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


def _pct(a: float, b: float) -> str:
    if b == 0:
        return "N/A"
    return f"{((a - b) / b) * 100:+.2f}%"


# ── 20 paires Breakout par défaut (identiques au backtest) ─────────────────────
DEFAULT_BREAKOUT_PAIRS = [
    "BTCUSDC", "ETHUSDC", "SOLUSDC", "XRPUSDC", "LINKUSDC",
    "ADAUSDC", "AVAXUSDC", "DOGEUSDC", "DOTUSDC", "ATOMUSDC",
    "UNIUSDC", "NEARUSDC", "LTCUSDC", "ETCUSDC", "FILUSDC",
    "AAVEUSDC", "INJUSDC", "SUIUSDC", "HBARUSDC", "BNBUSDC",
]


class TradeXBinanceBreakoutBot:
    """Bot Breakout Volatility Expansion pour Binance — LONG ONLY."""

    def __init__(self, dry_run: bool = False) -> None:
        self.dry_run = dry_run
        self._running = False

        # Services Binance (même API keys que le bot RANGE)
        self._client = BinanceClient(
            api_key=config.BINANCE_API_KEY,
            secret_key=config.BINANCE_SECRET_KEY,
            base_url=config.BINANCE_BASE_URL,
        )
        self._data = BinanceDataProvider(self._client)
        self._telegram = TelegramNotifier(
            bot_token=config.TELEGRAM_BOT_TOKEN,
            chat_id=config.TELEGRAM_CHAT_ID,
        )

        # Persistance séparée
        self._store = PositionStore(state_file=_STATE_FILE)

        # Paires
        self._trading_pairs: list[str] = []

        # État par paire
        self._positions: dict[str, Position] = {}
        self._candles_cache: dict[str, list[Candle]] = {}
        self._last_signal: dict[str, Optional[BreakoutSignal]] = {}  # dernier signal émis par paire
        self._atr_cache: dict[str, float] = {}  # ATR courant par paire (pour trailing)

        # Adaptive trailing state par paire
        self._peak_prices: dict[str, float] = {}  # plus haut atteint depuis l'entrée
        self._trail_sl: dict[str, float] = {}  # trailing SL actuel

        # Kill-switch mensuel
        self._month_start_equity: float = 0.0
        self._current_month: str = ""
        self._kill_switch_active: bool = False

        # Heartbeat
        self._last_heartbeat_time: float = 0.0
        self._cycle_count: int = 0
        self._last_h4_ts: int = 0  # timestamp de la dernière bougie H4 traitée

        # Firebase cleanup + daily snapshot
        self._last_cleanup_date: str = ""
        self._last_snapshot_date: str = ""

        # ── Métriques & monitoring ──
        self._daily_pnl: float = 0.0
        self._daily_trades: int = 0
        self._daily_date: str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        self._signals_detected: int = 0
        self._signals_rejected: int = 0
        self._api_latencies: list[float] = []
        self._next_h4_close_ts: float = 0.0
        self._last_telegram_heartbeat: float = 0.0
        self._dd_warning_sent: bool = False
        self._trail_last_palier: dict[str, int] = {}  # 0=initial, 1=step1, 2=step2

        # Breakout config
        self._breakout_cfg = BreakoutConfig(
            bb_period=config.BINANCE_BREAKOUT_BB_PERIOD,
            bb_std=config.BINANCE_BREAKOUT_BB_STD,
            bb_width_expansion=config.BINANCE_BREAKOUT_BB_EXPANSION,
            donchian_period=config.BINANCE_BREAKOUT_DONCHIAN_PERIOD,
            adx_threshold=config.BINANCE_BREAKOUT_ADX_THRESHOLD,
            vol_multiplier=config.BINANCE_BREAKOUT_VOL_MULT,
            sl_atr_mult=config.BINANCE_BREAKOUT_SL_ATR_MULT,
            allow_short=False,  # LONG ONLY
        )

        if dry_run:
            logger.info("🔧 Mode DRY-RUN activé — aucun ordre ne sera exécuté")

    # ═══════════════════════════════════════════════════════════════════════════
    # LIFECYCLE
    # ═══════════════════════════════════════════════════════════════════════════

    def run(self) -> None:
        """Lance la boucle principale."""
        self._running = True
        self._discover_pairs()
        self._load_state()
        self._init_kill_switch()

        logger.info("═" * 60)
        logger.info("🚀 TradeX BINANCE BREAKOUT démarré — Long Only")
        logger.info("   Paires     : %d", len(self._trading_pairs))
        logger.info(
            "   Stratégie  : BB(%d,%.1f) + Donchian(%d) + ADX(>%.0f) + Vol(>%.1fx)",
            self._breakout_cfg.bb_period, self._breakout_cfg.bb_std,
            self._breakout_cfg.donchian_period,
            self._breakout_cfg.adx_threshold,
            self._breakout_cfg.vol_multiplier,
        )
        logger.info(
            "   Risk       : %.0f%% par trade | Max positions: %d",
            config.BINANCE_BREAKOUT_RISK_PERCENT * 100,
            config.BINANCE_BREAKOUT_MAX_POSITIONS,
        )
        allocated = config.BINANCE_BREAKOUT_ALLOCATED_BALANCE
        logger.info(
            "   Capital    : %s | SL: %.1f×ATR | Trail: adaptive=%s",
            f"${allocated:.0f} alloué" if allocated > 0 else "100% du USDC dispo",
            config.BINANCE_BREAKOUT_SL_ATR_MULT,
            "ON" if config.BINANCE_BREAKOUT_ADAPTIVE_TRAIL else "OFF",
        )
        logger.info(
            "   Kill-switch: %s (seuil: %.0f%%)",
            "ON" if config.BINANCE_BREAKOUT_KILL_SWITCH else "OFF",
            config.BINANCE_BREAKOUT_KILL_PCT * 100,
        )
        logger.info("═" * 60)

        # Initialisation : charger les bougies + détecter les signaux existants
        self._initialize_candles()
        self._next_h4_close_ts = self._compute_next_h4_close()

        try:
            while self._running:
                self._tick()
                time.sleep(config.BINANCE_BREAKOUT_POLLING_SECONDS)
        except KeyboardInterrupt:
            logger.info("Arrêt demandé par l'utilisateur")
        finally:
            self._shutdown()

    def stop(self) -> None:
        self._running = False

    # ═══════════════════════════════════════════════════════════════════════════
    # PAIR DISCOVERY
    # ═══════════════════════════════════════════════════════════════════════════

    def _discover_pairs(self) -> None:
        """Résout les paires Breakout."""
        if config.BINANCE_BREAKOUT_PAIRS:
            self._trading_pairs = config.BINANCE_BREAKOUT_PAIRS
        else:
            # Utiliser les 20 paires par défaut (vérifier disponibilité)
            try:
                all_usdc = set(self._client.get_all_usdc_pairs())
                self._trading_pairs = [p for p in DEFAULT_BREAKOUT_PAIRS if p in all_usdc]
            except Exception as e:
                logger.error("Erreur discovery: %s — fallback 20 paires", e)
                self._trading_pairs = DEFAULT_BREAKOUT_PAIRS
        logger.info("Paires Breakout: %d → %s", len(self._trading_pairs), ", ".join(self._trading_pairs))

    def _load_state(self) -> None:
        """Charge les positions depuis le JSON + réconcilie avec Binance."""
        loaded_positions, _ = self._store.load()
        active = {
            sym: pos for sym, pos in loaded_positions.items()
            if pos.status in (PositionStatus.OPEN, PositionStatus.ZERO_RISK)
        }
        if not active:
            logger.info("📂 Aucune position Breakout active")
            return

        logger.info("📂 %d position(s) Breakout à réconcilier", len(active))
        try:
            balances = self._client.get_balances()
        except Exception as e:
            logger.error("❌ API inaccessible pour réconciliation: %s", e)
            for sym, pos in active.items():
                self._positions[sym] = pos
            return

        balance_map = {b.currency: b for b in balances}
        for sym, pos in active.items():
            base_currency = sym.replace("USDC", "")
            base_bal = balance_map.get(base_currency)
            held = (base_bal.available + base_bal.reserved) if base_bal else 0.0
            if held >= pos.size * 0.90:
                self._positions[sym] = pos
                # Restaurer le trailing state
                self._peak_prices[sym] = pos.peak_price or pos.entry_price
                self._trail_sl[sym] = pos.zero_risk_sl or pos.sl_price
                logger.info("[%s] ✅ Position BREAKOUT confirmée | size=%.8f", sym, pos.size)
            else:
                logger.warning("[%s] Position Breakout locale mais solde insuffisant → retirée", sym)

    def _init_kill_switch(self) -> None:
        """Initialise l'equity du début de mois pour le kill-switch."""
        now = datetime.now(timezone.utc)
        self._current_month = now.strftime("%Y-%m")
        try:
            balances = self._client.get_balances()
            self._month_start_equity = self._calculate_equity(balances)
        except Exception:
            self._month_start_equity = 0
        logger.info("Kill-switch: equity début mois = $%.2f", self._month_start_equity)

    # ═══════════════════════════════════════════════════════════════════════════
    # CANDLE INITIALIZATION
    # ═══════════════════════════════════════════════════════════════════════════

    def _initialize_candles(self) -> None:
        """Charge les bougies H4 initiales et détecte les signaux existants."""
        logger.info("── Chargement bougies H4 pour %d paires... ──", len(self._trading_pairs))
        for symbol in self._trading_pairs:
            try:
                candles = self._data.get_h4_candles(symbol)
                self._candles_cache[symbol] = candles
                if candles:
                    self._last_h4_ts = max(self._last_h4_ts, candles[-1].timestamp)

                    # Calculer ATR courant
                    atr_vals = compute_atr(candles, self._breakout_cfg.atr_period)
                    last_atr = None
                    for v in reversed(atr_vals):
                        if v is not None:
                            last_atr = v
                            break
                    if last_atr:
                        self._atr_cache[symbol] = last_atr

                    # Détecter les signaux sur les bougies existantes
                    # Ne garder que le signal de la DERNIÈRE bougie (pas d'ancien signal)
                    signals = detect_breakout_signals(candles, self._breakout_cfg)
                    if signals:
                        latest = signals[-1]
                        if latest.candle_index == len(candles) - 1:
                            self._last_signal[symbol] = latest
                            logger.info(
                                "[%s] 🔔 Signal BREAKOUT détecté à l'init | %s | entry=%s | SL=%s",
                                symbol, latest.direction.value,
                                _fmt(latest.entry_price), _fmt(latest.sl_price),
                            )
                        else:
                            self._last_signal[symbol] = None

                logger.debug("[%s] %d bougies chargées, ATR=%.4f",
                             symbol, len(candles), self._atr_cache.get(symbol, 0))
            except Exception as e:
                logger.error("[%s] Erreur chargement bougies: %s", symbol, e)

        open_count = sum(1 for p in self._positions.values()
                         if p.status in (PositionStatus.OPEN, PositionStatus.ZERO_RISK))
        logger.info("── Init terminée | %d positions ouvertes ──", open_count)

    # ═══════════════════════════════════════════════════════════════════════════
    # TICK (boucle rapide — toutes les 30s)
    # ═══════════════════════════════════════════════════════════════════════════

    def _tick(self) -> None:
        """Un cycle de polling."""
        self._check_new_h4_candle()
        self._check_kill_switch_month_reset()

        for symbol in self._trading_pairs:
            try:
                self._process_symbol(symbol)
            except Exception as e:
                logger.error("[%s] Erreur tick: %s", symbol, e)

        self._cycle_count += 1
        self._maybe_heartbeat()

    def _check_new_h4_candle(self) -> None:
        """Détecte une nouvelle bougie H4 — optimisé : skip si pas encore l'heure."""
        now = time.time()

        # Optimisation : ne vérifier que si on a dépassé le prochain close H4 (+60s buffer)
        if self._next_h4_close_ts > 0 and now < self._next_h4_close_ts + 60:
            return

        if not self._trading_pairs:
            return
        sample = self._trading_pairs[0]
        try:
            t0 = time.time()
            # Fetch seulement 2 bougies (pas 200) pour vérifier le timestamp
            candles = self._data.get_h4_candles(sample, limit=2)
            self._record_api_latency(time.time() - t0)

            if candles and candles[-1].timestamp > self._last_h4_ts:
                self._last_h4_ts = candles[-1].timestamp
                self._next_h4_close_ts = self._compute_next_h4_close()
                logger.info("═" * 40)
                logger.info("🕐 Nouvelle bougie H4 — recalcul des signaux Breakout")
                self._refresh_all_signals()
                self._firebase_daily_cleanup()
                self._maybe_daily_snapshot()
                self._reset_daily_if_needed()
            else:
                # Pas encore de nouvelle bougie → recalculer le prochain close
                self._next_h4_close_ts = self._compute_next_h4_close()
        except Exception as e:
            logger.debug("Erreur check H4: %s", e)

    def _refresh_all_signals(self) -> None:
        """Recharge bougies + détecte les signaux pour toutes les paires."""
        signals_count = 0
        near_miss_count = 0

        for symbol in self._trading_pairs:
            try:
                t0 = time.time()
                candles = self._data.get_h4_candles(symbol)
                self._record_api_latency(time.time() - t0)
                self._candles_cache[symbol] = candles

                # ATR
                atr_vals = compute_atr(candles, self._breakout_cfg.atr_period)
                for v in reversed(atr_vals):
                    if v is not None:
                        self._atr_cache[symbol] = v
                        break

                # Signaux
                signals = detect_breakout_signals(candles, self._breakout_cfg)
                if signals:
                    latest = signals[-1]
                    # Signal = sur la DERNIÈRE bougie seulement (pas d'ancien signal)
                    if latest.candle_index == len(candles) - 1:
                        self._last_signal[symbol] = latest
                        signals_count += 1
                        self._signals_detected += 1
                        logger.info(
                            "[%s] 🔔 Signal BREAKOUT %s | entry=%s | SL=%s | ADX=%.1f | BBw=%.3f | Vol=%.1fx",
                            symbol, latest.direction.value,
                            _fmt(latest.entry_price), _fmt(latest.sl_price),
                            latest.adx, latest.bb_width, latest.volume_ratio,
                        )
                    else:
                        self._last_signal[symbol] = None
                else:
                    self._last_signal[symbol] = None

                    # Diagnostic near-miss : quels filtres bloquent ?
                    if candles:
                        diag = diagnose_last_candle(candles, self._breakout_cfg)
                        passed = diag.get("filters_passed", 0)
                        total = diag.get("filters_total", 4)
                        if passed >= 3:  # Near-miss : 3/4 filtres passés
                            near_miss_count += 1
                            self._signals_rejected += 1
                            reasons = diag.get("reasons", [])
                            logger.info(
                                "[%s] ❗ Near-miss breakout (%d/%d) | %s",
                                symbol, passed, total,
                                " | ".join(reasons) if reasons else "?",
                            )
                            self._telegram.notify_signal_rejected(
                                symbol=symbol,
                                reasons=reasons,
                                filters_passed=passed,
                                filters_total=total,
                            )
            except Exception as e:
                logger.error("[%s] Erreur refresh signal: %s", symbol, e)

        logger.info(
            "📊 Scan H4 terminé | %d signal(s) détecté(s) | %d near-miss | %d paires analysées",
            signals_count, near_miss_count, len(self._trading_pairs),
        )

    def _check_kill_switch_month_reset(self) -> None:
        """Reset le kill-switch en début de mois."""
        now = datetime.now(timezone.utc)
        month_key = now.strftime("%Y-%m")
        if month_key != self._current_month:
            self._current_month = month_key
            self._kill_switch_active = False
            try:
                balances = self._client.get_balances()
                self._month_start_equity = self._calculate_equity(balances)
            except Exception:
                pass
            logger.info("📅 Nouveau mois %s | Equity reset = $%.2f | Kill-switch: OFF",
                        month_key, self._month_start_equity)

    def _process_symbol(self, symbol: str) -> None:
        """Traite un symbole : gérer trailing ou chercher une entrée."""
        position = self._positions.get(symbol)

        if position and position.status in (PositionStatus.OPEN, PositionStatus.ZERO_RISK):
            # Gérer la position ouverte (trailing stop)
            self._manage_trailing(symbol, position)
        else:
            # Pas de position → chercher un signal Breakout
            self._seek_breakout_entry(symbol)

    # ═══════════════════════════════════════════════════════════════════════════
    # ENTRÉE EN POSITION
    # ═══════════════════════════════════════════════════════════════════════════

    def _seek_breakout_entry(self, symbol: str) -> None:
        """Si un signal Breakout existe sur la dernière bougie → ouvrir."""
        sig = self._last_signal.get(symbol)
        if sig is None:
            return

        # LONG ONLY
        if sig.direction != BreakoutDirection.LONG:
            return

        # Kill-switch actif → pas de nouvelles positions
        if self._kill_switch_active:
            logger.debug("[%s] Signal ignoré : kill-switch actif", symbol)
            return

        # Déjà une position ?
        pos = self._positions.get(symbol)
        if pos and pos.status in (PositionStatus.OPEN, PositionStatus.ZERO_RISK):
            return

        # Max positions
        open_count = sum(
            1 for p in self._positions.values()
            if p.status in (PositionStatus.OPEN, PositionStatus.ZERO_RISK)
        )
        if open_count >= config.BINANCE_BREAKOUT_MAX_POSITIONS:
            logger.info(
                "[%s] ❌ Signal BREAKOUT ignoré : max positions atteint (%d/%d)",
                symbol, open_count, config.BINANCE_BREAKOUT_MAX_POSITIONS,
            )
            return

        self._open_breakout_position(symbol, sig)

    def _open_breakout_position(self, symbol: str, sig: BreakoutSignal) -> None:
        """Ouvre une position Breakout via MARKET order."""
        # Balances
        balances = self._client.get_balances()
        usdc_balance = next((b for b in balances if b.currency == "USDC"), None)
        available_usdc = usdc_balance.available if usdc_balance else 0.0

        if available_usdc <= 0:
            return

        # Capital alloué (plafond virtuel)
        allocated = config.BINANCE_BREAKOUT_ALLOCATED_BALANCE
        fiat_balance = min(allocated, available_usdc) if allocated > 0 else available_usdc

        # Risk check global
        open_pos_list = [
            p for p in self._positions.values()
            if p.status in (PositionStatus.OPEN, PositionStatus.ZERO_RISK)
        ]
        current_risk = check_total_risk_exposure(
            open_pos_list, fiat_balance, config.MAX_TOTAL_RISK_PERCENT,
        )
        if current_risk >= config.MAX_TOTAL_RISK_PERCENT:
            logger.debug("[%s] Risque global max atteint (%.1f%%), skip", symbol, current_risk * 100)
            return

        # Sizing
        risk_pct = config.BINANCE_BREAKOUT_RISK_PERCENT
        ticker = self._data.get_ticker(symbol)
        if ticker is None:
            return
        current_price = ticker.last_price

        # Guard: le prix doit être AU-DESSUS du SL pour un LONG
        # (le marché peut avoir bougé entre le signal et maintenant)
        if current_price <= sig.sl_price:
            logger.warning(
                "[%s] ⚠️ Prix actuel %s ≤ SL signal %s — signal périmé, skip",
                symbol, _fmt(current_price), _fmt(sig.sl_price),
            )
            self._last_signal[symbol] = None
            return

        # Guard: marge minimale entre prix et SL (au moins 0.3%)
        sl_distance_pct = (current_price - sig.sl_price) / current_price
        if sl_distance_pct < 0.003:
            logger.warning(
                "[%s] ⚠️ SL trop proche (%+.2f%%) — skip",
                symbol, sl_distance_pct * 100,
            )
            self._last_signal[symbol] = None
            return

        size = calculate_position_size(
            account_balance=fiat_balance,
            risk_percent=risk_pct,
            entry_price=current_price,
            sl_price=sig.sl_price,
            max_position_percent=config.MAX_POSITION_PERCENT,
        )
        if size <= 0:
            return

        # Format + checks Binance
        quantity_str = self._client.format_quantity(symbol, size)
        quantity = float(quantity_str)

        if not self._client.check_min_notional(symbol, quantity, current_price):
            logger.warning("[%s] Notionnel insuffisant, skip", symbol)
            return

        estimated_cost = quantity * current_price * 1.002
        if estimated_cost > available_usdc:
            logger.warning("[%s] Coût %.2f > USDC dispo %.2f — skip", symbol, estimated_cost, available_usdc)
            return

        # MARKET ORDER
        venue_order_id = "dry-run"
        fill_price = current_price

        if not self.dry_run:
            try:
                result = self._client.place_market_order(
                    symbol=symbol,
                    side="BUY",
                    quantity=quantity_str,
                )
                venue_order_id = str(result.get("orderId", "unknown"))

                cq = float(result.get("cummulativeQuoteQty", 0))
                eq = float(result.get("executedQty", 0))
                if cq > 0 and eq > 0:
                    fill_price = cq / eq
                    quantity = eq

                # Déduire frais payés en base asset
                base_currency = symbol.replace("USDC", "")
                total_commission = 0.0
                for fill in result.get("fills", []):
                    if fill.get("commissionAsset") == base_currency:
                        total_commission += float(fill.get("commission", 0))
                if total_commission > 0:
                    quantity -= total_commission

                logger.info("[%s] ✅ MARKET BUY fill @ %s (qty=%.8f)", symbol, _fmt(fill_price), quantity)

                # Vérification slippage
                slippage_pct = abs(fill_price - current_price) / current_price
                if slippage_pct > config.SLIPPAGE_WARNING_PCT:
                    logger.warning(
                        "[%s] ⚠️ Slippage %.2f%% (expected=%s, fill=%s)",
                        symbol, slippage_pct * 100, _fmt(current_price), _fmt(fill_price),
                    )
                    self._telegram.notify_warning(
                        f"Slippage {symbol}",
                        f"Slippage {slippage_pct*100:.2f}% (expected={_fmt(current_price)}, fill={_fmt(fill_price)})",
                    )
            except Exception as e:
                logger.error("[%s] ❌ MARKET BUY échoué: %s", symbol, e)
                self._telegram.notify_error(f"Breakout {symbol} BUY échoué: {e}")
                return
        else:
            logger.info("[DRY-RUN] MARKET BUY %s qty=%s @ %s", symbol, quantity_str, _fmt(fill_price))

        # Créer la position
        position = Position(
            symbol=symbol,
            side=OrderSide.BUY,
            entry_price=fill_price,
            sl_price=sig.sl_price,
            size=quantity,
            venue_order_id=venue_order_id,
            status=PositionStatus.OPEN,
            strategy=StrategyType.BREAKOUT,
            peak_price=fill_price,
        )
        self._positions[symbol] = position
        self._peak_prices[symbol] = fill_price
        self._trail_sl[symbol] = sig.sl_price
        self._save_state()

        # Consommer le signal (ne pas re-entrer sur la même bougie)
        self._last_signal[symbol] = None

        # Notifications riches
        risk_amount = fiat_balance * risk_pct
        sl_dist_pct = abs(fill_price - sig.sl_price) / fill_price * 100
        size_usd = quantity * fill_price

        self._telegram.notify_breakout_entry(
            symbol=symbol,
            entry_price=fill_price,
            sl_price=sig.sl_price,
            size=quantity,
            size_usd=size_usd,
            risk_pct=risk_pct,
            risk_usd=risk_amount,
            sl_distance_pct=sl_dist_pct,
            adx=sig.adx,
            bb_width=sig.bb_width,
            volume_ratio=sig.volume_ratio,
        )

        logger.info(
            "[%s] 📈 BREAKOUT LONG | entry=%s | SL=%s (%.1f%%) | size=%.8f | risk=$%.2f",
            symbol, _fmt(fill_price), _fmt(sig.sl_price), sl_dist_pct, quantity, risk_amount,
        )

        # Firebase
        try:
            equity = self._calculate_equity(balances)
            portfolio_risk = check_total_risk_exposure(
                [p for p in self._positions.values()
                 if p.status in (PositionStatus.OPEN, PositionStatus.ZERO_RISK)],
                fiat_balance, 1.0,
            ) if fiat_balance > 0 else 0.0

            fb_id = log_trade_opened(
                position=position,
                fill_type="taker",
                maker_wait_seconds=0,
                risk_pct=risk_pct,
                risk_amount_usd=risk_amount,
                fiat_balance=fiat_balance,
                current_equity=equity,
                portfolio_risk_before=portfolio_risk,
                exchange=EXCHANGE_NAME,
            )
            if fb_id:
                position.firebase_trade_id = fb_id
                self._save_state()
        except Exception as e:
            logger.warning("🔥 Firebase log échoué: %s", e)

    # ═══════════════════════════════════════════════════════════════════════════
    # TRAILING STOP ADAPTATIF
    # ═══════════════════════════════════════════════════════════════════════════

    def _manage_trailing(self, symbol: str, position: Position) -> None:
        """Gère le trailing stop adaptatif par paliers (comme le backtest)."""
        ticker = self._data.get_ticker(symbol)
        if ticker is None:
            return

        price = ticker.last_price
        current_sl = self._trail_sl.get(symbol, position.sl_price)
        peak = self._peak_prices.get(symbol, position.entry_price)

        # Vérifier SL touché
        if price <= current_sl:
            reason = "Trailing SL" if current_sl != position.sl_price else "SL initial"
            pnl_pct = (price - position.entry_price) / position.entry_price * 100
            logger.warning(
                "[%s] 🛑 %s | prix=%s ≤ SL=%s | PnL≈%+.1f%%",
                symbol, reason, _fmt(price), _fmt(current_sl), pnl_pct,
            )
            self._close_breakout_position(symbol, price, reason)
            return

        # Mettre à jour le peak
        if price > peak:
            peak = price
            self._peak_prices[symbol] = peak

        # Adaptive trailing stop (3 paliers — identique au backtest gagnant)
        entry = position.entry_price
        gain_pct = (peak - entry) / entry if entry > 0 else 0

        if config.BINANCE_BREAKOUT_ADAPTIVE_TRAIL:
            new_sl = current_sl

            step1 = config.BINANCE_BREAKOUT_TRAIL_STEP1_PCT  # 2%
            step2 = config.BINANCE_BREAKOUT_TRAIL_STEP2_PCT  # 5%
            lock1 = config.BINANCE_BREAKOUT_TRAIL_LOCK1_PCT  # 0.2%
            lock2 = config.BINANCE_BREAKOUT_TRAIL_LOCK2_PCT  # 2%

            current_palier = 0
            if gain_pct >= step2:
                current_palier = 2
                # Palier 2 : verrouiller lock2% de profit
                candidate = entry * (1 + lock2)
                new_sl = max(new_sl, candidate)
                # + trailing classique serré
                trail_atr = self._atr_cache.get(symbol, 0)
                if trail_atr > 0:
                    atr_trail_sl = peak - config.BINANCE_BREAKOUT_TRAIL_ATR_MULT * trail_atr
                    new_sl = max(new_sl, atr_trail_sl)
            elif gain_pct >= step1:
                current_palier = 1
                # Palier 1 : verrouiller lock1% de profit (quasi breakeven)
                candidate = entry * (1 + lock1)
                new_sl = max(new_sl, candidate)

            if new_sl > current_sl:
                self._trail_sl[symbol] = new_sl
                # Mettre à jour le SL dans la position pour persistance
                position.zero_risk_sl = new_sl
                position.peak_price = peak
                self._save_state()

                # Déterminer le label du palier
                palier_labels = {0: "", 1: "Palier 1 (≈BE)", 2: "Palier 2 (+trailing)"}
                palier_label = palier_labels.get(current_palier, "")

                logger.info(
                    "[%s] 📊 Trail SL ↑ %s → %s | gain=%.1f%% | peak=%s | %s",
                    symbol, _fmt(current_sl), _fmt(new_sl), gain_pct * 100, _fmt(peak),
                    palier_label or "initial",
                )

                # Notification Telegram : uniquement sur changement de palier
                last_palier = self._trail_last_palier.get(symbol, 0)
                if current_palier > last_palier:
                    self._trail_last_palier[symbol] = current_palier
                    self._telegram.notify_breakout_trail(
                        symbol=symbol,
                        entry_price=entry,
                        old_sl=current_sl,
                        new_sl=new_sl,
                        peak=peak,
                        gain_pct=gain_pct * 100,
                        palier=palier_label,
                    )

                # Firebase : mettre à jour SL + peak_price
                if position.firebase_trade_id:
                    try:
                        fb_log_trailing_sl(position.firebase_trade_id, new_sl)
                        from src.firebase.client import update_document
                        update_document("trades", position.firebase_trade_id, {
                            "peak_price": peak,
                            "trail_gain_pct": round(gain_pct, 6),
                        })
                    except Exception as e:
                        logger.debug("Firebase trail update échoué: %s", e)
        else:
            # Trailing classique basé sur ATR
            trail_atr = self._atr_cache.get(symbol, 0)
            if trail_atr > 0 and peak > entry:
                new_sl = peak - config.BINANCE_BREAKOUT_TRAIL_ATR_MULT * trail_atr
                if new_sl > current_sl:
                    self._trail_sl[symbol] = new_sl
                    position.zero_risk_sl = new_sl
                    position.peak_price = peak
                    self._save_state()
                    # Firebase
                    if position.firebase_trade_id:
                        try:
                            fb_log_trailing_sl(position.firebase_trade_id, new_sl)
                        except Exception:
                            pass

        # Kill-switch check + DD warning
        if config.BINANCE_BREAKOUT_KILL_SWITCH and self._month_start_equity > 0:
            try:
                balances = self._client.get_balances()
                equity = self._calculate_equity(balances)
                month_return = (equity - self._month_start_equity) / self._month_start_equity

                # DD Warning (avant kill-switch)
                if month_return <= config.DD_WARNING_PCT and not self._dd_warning_sent:
                    self._dd_warning_sent = True
                    logger.warning(
                        "⚠️ DD Warning | equity=$%.2f | month=%.1f%% (seuil warning: %.1f%%, kill: %.1f%%)",
                        equity, month_return * 100,
                        config.DD_WARNING_PCT * 100, config.BINANCE_BREAKOUT_KILL_PCT * 100,
                    )
                    self._telegram.notify_warning(
                        f"Drawdown {month_return*100:.1f}%",
                        f"Equity: ${equity:,.0f} | DD mois: {month_return*100:+.1f}% "
                        f"(kill-switch à {config.BINANCE_BREAKOUT_KILL_PCT*100:.0f}%)",
                    )

                if month_return <= config.BINANCE_BREAKOUT_KILL_PCT:
                    if not self._kill_switch_active:
                        self._kill_switch_active = True
                        logger.warning(
                            "🚨 KILL-SWITCH activé | equity=$%.2f | month=%.1f%% ≤ %.1f%%",
                            equity, month_return * 100, config.BINANCE_BREAKOUT_KILL_PCT * 100,
                        )
                        self._telegram.notify_error(
                            f"🚨 BREAKOUT Kill-Switch activé | Perf mois: {month_return:.1%} "
                            f"| Fermeture de toutes les positions"
                        )
                        # Firebase event
                        try:
                            fb_log_event("KILL_SWITCH", {
                                "equity": equity,
                                "month_start_equity": self._month_start_equity,
                                "month_return_pct": round(month_return * 100, 2),
                                "threshold_pct": config.BINANCE_BREAKOUT_KILL_PCT * 100,
                            }, exchange=EXCHANGE_NAME)
                        except Exception:
                            pass
                        # Fermer toutes les positions
                        self._close_all_positions("Kill-switch mensuel")
            except Exception as e:
                logger.debug("Kill-switch equity check échoué: %s", e)

    # ═══════════════════════════════════════════════════════════════════════════
    # FERMETURE DE POSITION
    # ═══════════════════════════════════════════════════════════════════════════

    def _close_breakout_position(self, symbol: str, exit_price: float, reason: str) -> None:
        """Ferme une position Breakout via MARKET SELL."""
        position = self._positions.get(symbol)
        if not position:
            return

        # Quantité = solde réel disponible
        close_qty = position.size
        if not self.dry_run:
            base_currency = symbol.replace("USDC", "")
            balances = self._client.get_balances()
            base_bal = next((b for b in balances if b.currency == base_currency), None)
            if base_bal and base_bal.available < close_qty:
                close_qty = base_bal.available

        qty_str = self._client.format_quantity(symbol, close_qty)
        actual_price = exit_price

        if not self.dry_run:
            try:
                result = self._client.place_market_order(
                    symbol=symbol,
                    side="SELL",
                    quantity=qty_str,
                )
                cq = float(result.get("cummulativeQuoteQty", 0))
                eq = float(result.get("executedQty", 0))
                if cq > 0 and eq > 0:
                    actual_price = cq / eq
            except Exception as e:
                logger.error("[%s] ❌ Close MARKET échoué: %s", symbol, e)
                self._telegram.notify_error(f"Breakout close {symbol} échoué: {e}")
                return
        else:
            logger.info("[DRY-RUN] MARKET SELL %s qty=%s", symbol, qty_str)

        self._finalize_close(symbol, actual_price, reason)

    def _close_all_positions(self, reason: str) -> None:
        """Ferme toutes les positions ouvertes (kill-switch)."""
        for symbol in list(self._positions.keys()):
            pos = self._positions.get(symbol)
            if pos and pos.status in (PositionStatus.OPEN, PositionStatus.ZERO_RISK):
                ticker = self._data.get_ticker(symbol)
                if ticker:
                    self._close_breakout_position(symbol, ticker.last_price, reason)

    def _finalize_close(self, symbol: str, exit_price: float, reason: str) -> None:
        """Finalise la clôture : PnL, nettoyage, notifications, Firebase."""
        position = self._positions.get(symbol)
        if not position:
            return

        # PnL
        pnl_gross = (exit_price - position.entry_price) * position.size
        notional = position.size * position.entry_price
        fee_rate = config.BINANCE_TAKER_FEE
        fees = notional * fee_rate + position.size * exit_price * fee_rate
        pnl_net = pnl_gross - fees
        pnl_pct = pnl_net / notional if notional > 0 else 0

        pnl_emoji = "🟢" if pnl_net >= 0 else "🔴"
        logger.info(
            "[%s] %s CLOSE BREAKOUT | %s | PnL=$%+.4f (%+.2f%%) | fees=$%.4f",
            symbol, pnl_emoji, reason, pnl_net, pnl_pct * 100, fees,
        )

        position.status = PositionStatus.CLOSED
        position.pnl = pnl_net

        # Daily PnL tracking
        self._daily_pnl += pnl_net
        self._daily_trades += 1

        # Cleanup trailing state
        self._peak_prices.pop(symbol, None)
        self._trail_sl.pop(symbol, None)
        self._last_signal.pop(symbol, None)
        self._trail_last_palier.pop(symbol, None)
        self._save_state()

        # Telegram
        self._telegram.notify_sl_hit(position, exit_price)

        # Firebase
        equity_after = 0.0
        try:
            balances = self._client.get_balances()
            equity_after = self._calculate_equity(balances)
        except Exception:
            pass

        if position.firebase_trade_id:
            try:
                log_trade_closed(
                    trade_id=position.firebase_trade_id,
                    position=position,
                    exit_price=exit_price,
                    reason=reason,
                    fill_type="taker",
                    equity_after=equity_after,
                )
            except Exception as e:
                logger.warning("🔥 Firebase log_trade_closed échoué: %s", e)

    # ═══════════════════════════════════════════════════════════════════════════
    # HELPERS
    # ═══════════════════════════════════════════════════════════════════════════

    def _calculate_equity(self, balances: Optional[list[Balance]] = None) -> float:
        """Equity totale du compte Binance."""
        if balances is None:
            balances = self._client.get_balances()

        fiat_set = {"USD", "EUR", "GBP", "USDC", "USDT", "BUSD"}
        crypto_tickers = []
        for b in balances:
            if b.total > 0 and b.currency not in fiat_set:
                try:
                    price = self._client.get_ticker_price(f"{b.currency}USDC")
                    if price > 0:
                        crypto_tickers.append(TickerData(
                            symbol=f"{b.currency}-USD",
                            bid=price, ask=price, mid=price, last_price=price,
                        ))
                except Exception:
                    pass

        adjusted_balances = []
        for b in balances:
            if b.currency == "USDC":
                adjusted_balances.append(Balance(
                    currency="USD", available=b.available,
                    reserved=b.reserved, total=b.total,
                ))
            else:
                adjusted_balances.append(b)

        return get_total_equity(adjusted_balances, crypto_tickers)

    def _save_state(self) -> None:
        self._store.save(self._positions, {})

    def _maybe_heartbeat(self) -> None:
        """Heartbeat étendu — log + Telegram périodique (niveau fund)."""
        now = time.time()
        if now - self._last_heartbeat_time < config.HEARTBEAT_INTERVAL_SECONDS:
            return
        self._last_heartbeat_time = now

        open_pos = [
            p for p in self._positions.values()
            if p.status in (PositionStatus.OPEN, PositionStatus.ZERO_RISK)
        ]

        try:
            t0 = time.time()
            balances = self._client.get_balances()
            self._record_api_latency(time.time() - t0)
            equity = self._calculate_equity(balances)
        except Exception:
            equity = 0
            balances = []

        # Equity allouée pour ce bot
        allocated = config.BINANCE_BREAKOUT_ALLOCATED_BALANCE
        allocated_equity = min(allocated, equity) if allocated > 0 else equity

        # Drawdown mensuel
        dd_pct = 0.0
        if self._month_start_equity > 0:
            dd_pct = (equity - self._month_start_equity) / self._month_start_equity * 100

        # Exposition courante (somme notionnelle des positions / capital alloué)
        exposure_notional = 0.0
        positions_detail = []
        for pos in open_pos:
            ticker = self._data.get_ticker(pos.symbol)
            price = ticker.last_price if ticker else pos.entry_price
            notional = pos.size * price
            exposure_notional += notional
            sl = self._trail_sl.get(pos.symbol, pos.sl_price)
            peak = self._peak_prices.get(pos.symbol, pos.entry_price)
            gain = (price - pos.entry_price) / pos.entry_price * 100 if pos.entry_price > 0 else 0
            positions_detail.append({
                "symbol": pos.symbol,
                "entry": pos.entry_price,
                "sl": sl,
                "peak": peak,
                "gain_pct": gain,
                "notional": notional,
            })
            logger.info(
                "  [%s] LONG @ %s | SL=%s | peak=%s | now=%s (%+.1f%%)",
                pos.symbol, _fmt(pos.entry_price), _fmt(sl), _fmt(peak), _fmt(price), gain,
            )

        exposure_pct = (exposure_notional / allocated_equity * 100) if allocated_equity > 0 else 0

        # Daily PnL en %
        daily_pnl_pct = (self._daily_pnl / allocated_equity * 100) if allocated_equity > 0 else 0

        # API latency moyenne (valeurs déjà en ms)
        avg_latency = (sum(self._api_latencies) / len(self._api_latencies)) if self._api_latencies else 0

        # Vérification data stale
        self._check_data_freshness()

        logger.info(
            "💓 BREAKOUT H4 | Equity: $%.0f (alloué: $%.0f) | DD: %+.1f%% | "
            "Expo: %.0f%% | Pos: %d/%d | PnL jour: %+.2f$ (%+.1f%%) | Kill: %s | "
            "Signaux: %d📡 %d❌ | API: %.0fms | cycle #%d",
            equity, allocated_equity, dd_pct,
            exposure_pct, len(open_pos), config.BINANCE_BREAKOUT_MAX_POSITIONS,
            self._daily_pnl, daily_pnl_pct,
            "🔴 ON" if self._kill_switch_active else "🟢 OFF",
            self._signals_detected, self._signals_rejected,
            avg_latency, self._cycle_count,
        )

        # Heartbeat Telegram (moins fréquent que le log)
        if now - self._last_telegram_heartbeat >= config.BREAKOUT_HEARTBEAT_TELEGRAM_SECONDS:
            self._last_telegram_heartbeat = now
            self._telegram.notify_breakout_heartbeat(
                equity=equity,
                allocated_equity=allocated_equity,
                drawdown_pct=dd_pct,
                exposure_pct=exposure_pct,
                open_positions=len(open_pos),
                max_positions=config.BINANCE_BREAKOUT_MAX_POSITIONS,
                daily_pnl=self._daily_pnl,
                daily_pnl_pct=daily_pnl_pct,
                kill_switch=self._kill_switch_active,
                positions_detail=positions_detail,
                signals_detected=self._signals_detected,
                signals_rejected=self._signals_rejected,
                avg_api_latency_ms=avg_latency,
            )

        # Firebase heartbeat
        try:
            fb_log_heartbeat(
                open_positions=len(open_pos),
                total_equity=equity,
                total_risk_pct=exposure_pct / 100 if exposure_pct > 0 else 0,
                pairs_count=len(self._trading_pairs),
                exchange=EXCHANGE_NAME,
            )
        except Exception:
            pass

    def _firebase_daily_cleanup(self) -> None:
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        if today == self._last_cleanup_date:
            return
        self._last_cleanup_date = today
        try:
            fb_cleanup_events()
        except Exception:
            pass

    def _maybe_daily_snapshot(self) -> None:
        """Log un snapshot quotidien dans Firebase (1x/jour)."""
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        if today == self._last_snapshot_date:
            return
        self._last_snapshot_date = today

        try:
            balances = self._client.get_balances()
            equity = self._calculate_equity(balances)
        except Exception:
            return

        open_pos = [
            p for p in self._positions.values()
            if p.status in (PositionStatus.OPEN, PositionStatus.ZERO_RISK)
        ]
        positions_data = []
        for p in open_pos:
            positions_data.append({
                "symbol": p.symbol,
                "side": p.side.value,
                "entry_price": p.entry_price,
                "sl_price": self._trail_sl.get(p.symbol, p.sl_price),
                "peak_price": self._peak_prices.get(p.symbol, p.entry_price),
                "size": p.size,
                "strategy": p.strategy.value,
            })

        # P&L du jour (trades fermés aujourd'hui)
        daily_pnl = sum(
            p.pnl or 0 for p in self._positions.values()
            if p.status == PositionStatus.CLOSED and p.pnl is not None
        )
        trades_today = sum(
            1 for p in self._positions.values()
            if p.status == PositionStatus.CLOSED
        )

        try:
            fb_log_daily_snapshot(
                equity=equity,
                positions=positions_data,
                daily_pnl=daily_pnl,
                trades_today=trades_today,
                exchange=EXCHANGE_NAME,
            )
            logger.info("📸 Snapshot quotidien → equity=$%.2f | %d positions", equity, len(open_pos))
        except Exception as e:
            logger.warning("Firebase snapshot échoué: %s", e)

    def _shutdown(self) -> None:
        logger.info("🛑 Arrêt de TradeX Binance Breakout...")
        self._save_state()
        self._client.close()
        self._telegram.close()
        logger.info("TradeX Binance Breakout arrêté proprement")

    # ═══════════════════════════════════════════════════════════════════════════
    # HELPERS — Monitoring & Alertes
    # ═══════════════════════════════════════════════════════════════════════════

    def _compute_next_h4_close(self) -> float:
        """Calcule le timestamp du prochain close H4 (UTC 0/4/8/12/16/20).

        Évite des appels API inutiles entre les fermetures de bougies.
        """
        now = datetime.now(timezone.utc)
        next_close_hour = ((now.hour // 4) + 1) * 4
        close_dt = now.replace(hour=0, minute=0, second=0, microsecond=0)
        close_dt += timedelta(hours=next_close_hour)
        return close_dt.timestamp()

    def _record_api_latency(self, elapsed_seconds: float) -> None:
        """Enregistre la latence d'un appel API et alerte si lent."""
        elapsed_ms = elapsed_seconds * 1000
        self._api_latencies.append(elapsed_ms)
        # Garder les 100 dernières mesures
        if len(self._api_latencies) > 100:
            self._api_latencies = self._api_latencies[-50:]
        # Alerte si API lente
        if elapsed_ms > config.API_SLOW_THRESHOLD_MS:
            logger.warning(
                "⚠️ API lente : %.0fms (seuil: %.0fms)",
                elapsed_ms, config.API_SLOW_THRESHOLD_MS,
            )
            self._telegram.notify_warning(
                "API lente",
                f"Appel en {elapsed_ms:.0f}ms (seuil: {config.API_SLOW_THRESHOLD_MS:.0f}ms)",
            )

    def _check_data_freshness(self) -> None:
        """Vérifie que les données de marché ne sont pas stale."""
        if self._last_h4_ts <= 0:
            return
        # Timestamp de la dernière bougie en secondes (Binance renvoie ms)
        last_candle_s = self._last_h4_ts / 1000 if self._last_h4_ts > 1e12 else self._last_h4_ts
        age_seconds = time.time() - last_candle_s
        if age_seconds > config.DATA_STALE_THRESHOLD_SECONDS:
            hours = age_seconds / 3600
            logger.warning(
                "⚠️ Data stale : dernière bougie il y a %.1fh (seuil: %.0fh)",
                hours, config.DATA_STALE_THRESHOLD_SECONDS / 3600,
            )
            self._telegram.notify_warning(
                "Data stale",
                f"Dernière bougie H4 reçue il y a {hours:.1f}h "
                f"(seuil: {config.DATA_STALE_THRESHOLD_SECONDS / 3600:.0f}h)",
            )

    def _reset_daily_if_needed(self) -> None:
        """Reset les métriques daily si le jour a changé."""
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        if today != self._daily_date:
            logger.info(
                "📅 Nouveau jour %s | PnL veille: $%+.2f | Trades: %d | Signaux: %d/%d",
                today, self._daily_pnl, self._daily_trades,
                self._signals_detected, self._signals_rejected,
            )
            self._daily_pnl = 0.0
            self._daily_trades = 0
            self._daily_date = today
            self._signals_detected = 0
            self._signals_rejected = 0
            self._dd_warning_sent = False


# ── Point d'entrée ─────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="TradeX Binance – Breakout Bot")
    parser.add_argument("--dry-run", action="store_true", help="Mode simulation")
    args = parser.parse_args()

    bot = TradeXBinanceBreakoutBot(dry_run=args.dry_run)
    signal.signal(signal.SIGTERM, lambda *_: bot.stop())
    bot.run()


if __name__ == "__main__":
    main()
