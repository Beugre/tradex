"""
Boucle principale du bot TradeX — Binance.

Version parallèle pour Binance avec paires USDC et ordres OCO natifs.
Stratégie : Range Only (même logique core que le bot Revolut).

Différences clés vs bot.py (Revolut) :
  - Ordres MARKET à l'entrée (pas de maker-first, fees Binance ~0.1%)
  - OCO natif pour SL + TP (exit gérée par l'exchange, pas de polling SL/TP)
  - Paires USDC (format BTCUSDC, pas BTC-USD)
  - Le bot poll seulement pour : nouvelles bougies H4, mise à jour tendance,
    vérification d'exécution des OCO

Usage :
    python -m src.bot_binance              # Production
    python -m src.bot_binance --dry-run    # Log les ordres sans les exécuter
"""

from __future__ import annotations

import argparse
import logging
import signal
import sys
import time
from typing import Optional

from src import config
from src.core.models import (
    Balance,
    OrderSide,
    Position,
    PositionStatus,
    RangeState,
    StrategyType,
    SwingLevel,
    TickerData,
    TrendDirection,
    TrendState,
)
from src.core.strategy_mean_rev import (
    activate_cooldown,
    build_range_from_trend,
    check_range_entry_signal,
    check_range_sl_hit,
    check_range_tp_hit,
    is_in_cooldown,
)
from src.core.position_store import PositionStore
from src.core.risk_manager import (
    calculate_position_size,
    check_total_risk_exposure,
    get_fiat_balance,
    get_total_equity,
)
from src.core.swing_detector import detect_swings
from src.core.trend_engine import check_trend_invalidation, determine_trend
from src.exchange.binance_client import BinanceClient
from src.exchange.binance_data_provider import BinanceDataProvider
from src.notifications.telegram import TelegramNotifier
from src.firebase.trade_logger import (
    log_trade_opened,
    log_trade_closed,
    log_zero_risk_applied,
    log_trailing_sl_update,
    log_trend_change as fb_log_trend_change,
    log_heartbeat as fb_log_heartbeat,
    log_event as fb_log_event,
    log_close_failure as fb_log_close_failure,
    clear_close_failure as fb_clear_close_failure,
    cleanup_old_events as fb_cleanup_events,
)

# ── Logging ────────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=getattr(logging, config.LOG_LEVEL, logging.INFO),
    format="[%(asctime)s] %(levelname)s %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("tradex.bot.binance")

logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)

# ── Identifiant exchange pour Firebase ─────────────────────────────────────────
EXCHANGE_NAME = "binance"

# ── State file séparé ─────────────────────────────────────────────────────────
import os
_STATE_FILE = os.environ.get(
    "TRADEX_BINANCE_STATE_FILE",
    os.path.join(os.path.dirname(__file__), "..", "data", "state_binance.json"),
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


def _pct_dist(current: float, target: float) -> str:
    if current == 0:
        return "N/A"
    dist = ((target - current) / current) * 100
    return f"{dist:+.2f}%"


def _cooldown_str(rs: RangeState) -> str:
    if not is_in_cooldown(rs):
        return ""
    remaining_s = rs.cooldown_until - int(time.time())
    bar_s = 4 * 3600
    remaining_bars = max(1, -(-remaining_s // bar_s))
    return f" ⏳ COOLDOWN ({remaining_bars}/{config.RANGE_COOLDOWN_BARS} bars)"


_TREND_EMOJI = {
    TrendDirection.BULLISH: "🟢 BULLISH",
    TrendDirection.BEARISH: "🔴 BEARISH",
    TrendDirection.NEUTRAL: "⚪ NEUTRAL",
}


class TradeXBinanceBot:
    """Bot de trading TradeX pour Binance — paires USDC avec OCO natifs."""

    def __init__(self, dry_run: bool = False) -> None:
        self.dry_run = dry_run
        self._running = False

        # Services Binance
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

        # Persistance séparée (state_binance.json)
        self._store = PositionStore(state_file=_STATE_FILE)

        # Trading pairs (auto-discovery ou config)
        self._trading_pairs: list[str] = []

        # État par paire
        self._trends: dict[str, TrendState] = {}
        self._ranges: dict[str, RangeState] = {}
        self._positions: dict[str, Position] = {}

        # OCO tracking : symbol → {"order_list_id": int, "tp_order_id": int, "sl_order_id": int}
        self._oco_orders: dict[str, dict] = {}

        # Heartbeat
        self._last_heartbeat_time: float = 0.0
        self._cycle_count: int = 0

        # Log on-change
        self._prev_state: dict[str, dict] = {}
        self._prev_ignored: dict[str, str] = {}

        # Signal persistence
        self._signal_persistence: dict[str, dict] = {}

        # Close failures
        self._close_failures: dict[str, dict] = {}

        # Firebase cleanup
        self._last_cleanup_date: str = ""

        if dry_run:
            logger.info("🔧 Mode DRY-RUN activé — aucun ordre ne sera exécuté")

    # ═══════════════════════════════════════════════════════════════════════════
    # LIFECYCLE
    # ═══════════════════════════════════════════════════════════════════════════

    def run(self) -> None:
        """Lance la boucle principale."""
        self._running = True

        # ── Découverte des paires USDC ──
        self._discover_pairs()

        logger.info("═" * 60)
        logger.info("🚀 TradeX BINANCE démarré — Range Only (USDC)")
        logger.info(
            "   Paires     : %d paires USDC",
            len(self._trading_pairs),
        )
        logger.info(
            "   🔄 RANGE   : risque %.0f%% | Largeur min: %.0f%% | Cooldown: %d bougies",
            config.BINANCE_RISK_PERCENT_RANGE * 100,
            config.RANGE_WIDTH_MIN * 100,
            config.RANGE_COOLDOWN_BARS,
        )
        logger.info(
            "   Ordres     : MARKET à l'entrée + OCO natif (SL+TP)",
        )
        logger.info(
            "   Fees       : maker %.2f%% | taker %.2f%%",
            config.BINANCE_MAKER_FEE * 100,
            config.BINANCE_TAKER_FEE * 100,
        )
        logger.info(
            "   Polling    : %ds | Max positions: %d",
            config.BINANCE_POLLING_INTERVAL_SECONDS,
            config.BINANCE_MAX_SIMULTANEOUS_POSITIONS,
        )
        logger.info("═" * 60)

        # Initialiser les tendances
        self._initialize()

        try:
            while self._running:
                self._tick()
                time.sleep(config.BINANCE_POLLING_INTERVAL_SECONDS)
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
        """Découvre les paires USDC disponibles sur Binance."""
        if config.BINANCE_TRADING_PAIRS:
            self._trading_pairs = config.BINANCE_TRADING_PAIRS
            logger.info("Paires configurées: %s", ", ".join(self._trading_pairs))
            return

        if config.BINANCE_AUTO_DISCOVER_PAIRS:
            try:
                all_usdc = self._client.get_all_usdc_pairs()
                self._trading_pairs = all_usdc
                logger.info(
                    "Auto-discovery: %d paires USDC trouvées: %s",
                    len(all_usdc),
                    ", ".join(all_usdc[:20]) + ("..." if len(all_usdc) > 20 else ""),
                )
            except Exception as e:
                logger.error("Auto-discovery échouée: %s", e)
                # Fallback minimal
                self._trading_pairs = ["BTCUSDC", "ETHUSDC", "SOLUSDC"]
        else:
            self._trading_pairs = ["BTCUSDC", "ETHUSDC", "SOLUSDC"]

    # ═══════════════════════════════════════════════════════════════════════════
    # INITIALIZATION
    # ═══════════════════════════════════════════════════════════════════════════

    def _initialize(self) -> None:
        """Charge les bougies initiales et calcule les tendances."""
        self._reconcile_positions()
        self._firebase_daily_cleanup()

        logger.info("── Initialisation des %d paires USDC... ──", len(self._trading_pairs))
        for symbol in self._trading_pairs:
            try:
                self._update_trend(symbol)
            except Exception as e:
                logger.error("[%s] ❌ Erreur d'initialisation: %s", symbol, e)

        open_count = sum(
            1 for p in self._positions.values()
            if p.status in (PositionStatus.OPEN, PositionStatus.ZERO_RISK)
        )
        logger.info("── Initialisation terminée ──")
        logger.info(
            "   Positions ouvertes: %d/%d | NEUTRAL: %d",
            open_count,
            config.BINANCE_MAX_SIMULTANEOUS_POSITIONS,
            sum(1 for t in self._trends.values() if t.direction == TrendDirection.NEUTRAL),
        )

    def _reconcile_positions(self) -> None:
        """Réconcilie l'état local avec Binance (balances + ordres actifs)."""
        loaded_positions, loaded_ranges = self._store.load()
        self._ranges = loaded_ranges

        active_local = {
            sym: pos for sym, pos in loaded_positions.items()
            if pos.status in (PositionStatus.OPEN, PositionStatus.ZERO_RISK)
        }

        if not active_local:
            logger.info("📂 Aucune position locale active")
        else:
            logger.info("📂 %d position(s) locale(s) à réconcilier", len(active_local))

        # Récupérer balances et ordres actifs
        try:
            balances = self._client.get_balances()
        except Exception as e:
            logger.error("❌ API Binance inaccessible: %s", e)
            for sym, pos in active_local.items():
                self._positions[sym] = pos
            return

        balance_map = {b.currency: b for b in balances}

        # Récupérer les OCOs actifs
        try:
            active_oco_lists = self._client.get_active_order_lists()
            # Map par symbol → dernière OCO list
            oco_by_symbol: dict[str, dict] = {}
            for oco in active_oco_lists:
                sym = oco.get("symbol", "")
                oco_by_symbol[sym] = oco
        except Exception as e:
            logger.warning("Impossible de récupérer les OCOs actifs: %s", e)
            oco_by_symbol = {}

        reconciled = 0
        for sym, pos in active_local.items():
            base_currency = sym.replace("USDC", "")  # ex: "BTC" de "BTCUSDC"
            base_bal = balance_map.get(base_currency)
            held = (base_bal.available + base_bal.reserved) if base_bal else 0.0

            if pos.side == OrderSide.BUY:
                if held >= pos.size * 0.90:
                    self._positions[sym] = pos
                    logger.info(
                        "[%s] ✅ Position BUY confirmée | %s=%.8f",
                        sym, base_currency, held,
                    )
                    # Restaurer l'OCO tracking si actif
                    if sym in oco_by_symbol:
                        oco = oco_by_symbol[sym]
                        self._oco_orders[sym] = {
                            "order_list_id": oco.get("orderListId"),
                        }
                        logger.info("[%s] OCO actif restauré (listId=%s)", sym, oco.get("orderListId"))
                    reconciled += 1
                else:
                    logger.warning(
                        "[%s] Position BUY locale mais %s=%.8f < size=%.8f → retirée",
                        sym, base_currency, held, pos.size,
                    )
            elif pos.side == OrderSide.SELL:
                if held < pos.size * 0.10:
                    self._positions[sym] = pos
                    reconciled += 1
                else:
                    logger.warning("[%s] Position SELL locale mais solde élevé → retirée", sym)

        logger.info("── Réconciliation: %d confirmées ──", reconciled)
        self._save_state()

    def _save_state(self) -> None:
        self._store.save(self._positions, self._ranges)

    # ═══════════════════════════════════════════════════════════════════════════
    # TICK (boucle rapide)
    # ═══════════════════════════════════════════════════════════════════════════

    def _tick(self) -> None:
        """Un cycle de polling."""
        self._check_new_h4_candle()
        self._check_oco_fills()

        self._cycle_signals = 0
        self._cycle_executed = 0

        for symbol in self._trading_pairs:
            try:
                self._process_symbol(symbol)
            except Exception as e:
                logger.error("[%s] Erreur dans le tick: %s", symbol, e)

        self._cycle_count += 1
        self._maybe_heartbeat()

    def _check_oco_fills(self) -> None:
        """Vérifie si des OCO ont été exécutés (TP ou SL atteint)."""
        filled_symbols = []

        for symbol, oco_info in list(self._oco_orders.items()):
            order_list_id = oco_info.get("order_list_id")
            if order_list_id is None:
                continue

            try:
                oco_status = self._client.get_order_list(order_list_id)
            except Exception as e:
                logger.debug("[%s] Impossible de vérifier l'OCO %s: %s", symbol, order_list_id, e)
                continue

            list_status = oco_status.get("listOrderStatus", "")

            if list_status == "ALL_DONE":
                # OCO terminé → l'un des deux ordres a été FILLED
                position = self._positions.get(symbol)
                if not position:
                    filled_symbols.append(symbol)
                    continue

                # Trouver quel ordre a été fill
                order_reports = oco_status.get("orderReports", [])
                if not order_reports:
                    # Fallback: regarder les ordres individuels
                    orders = oco_status.get("orders", [])
                    for o in orders:
                        try:
                            detail = self._client.get_order(symbol, order_id=o.get("orderId"))
                            order_reports.append(detail)
                        except Exception:
                            pass

                filled_order = None
                for report in order_reports:
                    if report.get("status") == "FILLED":
                        filled_order = report
                        break

                if filled_order:
                    filled_price = float(filled_order.get("price", 0))
                    filled_type = filled_order.get("type", "")

                    # Déterminer si c'est le TP ou le SL
                    is_tp = filled_type in ("LIMIT_MAKER", "TAKE_PROFIT", "TAKE_PROFIT_LIMIT")
                    is_sl = filled_type in ("STOP_LOSS", "STOP_LOSS_LIMIT")

                    # Calculer le prix moyen réel à partir des fills si disponibles
                    cq = float(filled_order.get("cummulativeQuoteQty", 0))
                    eq = float(filled_order.get("executedQty", 0))
                    if cq > 0 and eq > 0:
                        filled_price = cq / eq

                    if is_tp:
                        reason = "TP Range atteint (OCO)"
                        logger.info(
                            "[%s] 🎯 OCO TP FILLED | prix=%.6f",
                            symbol, filled_price,
                        )
                    elif is_sl:
                        reason = "SL atteint (OCO breakout)"
                        logger.warning(
                            "[%s] 🛑 OCO SL FILLED | prix=%.6f",
                            symbol, filled_price,
                        )
                        # Activer le cooldown
                        rs = self._ranges.get(symbol)
                        if rs:
                            activate_cooldown(rs, config.RANGE_COOLDOWN_BARS)
                            self._save_state()
                    else:
                        reason = f"OCO exécuté ({filled_type})"
                        filled_price = filled_price or position.entry_price

                    self._finalize_close(symbol, filled_price, reason, "maker" if is_tp else "taker")
                else:
                    logger.warning("[%s] OCO ALL_DONE mais pas de fill trouvé", symbol)

                filled_symbols.append(symbol)

            elif list_status in ("REJECT", "EXPIRED"):
                logger.warning("[%s] OCO %s → statut %s", symbol, order_list_id, list_status)
                filled_symbols.append(symbol)

        for sym in filled_symbols:
            self._oco_orders.pop(sym, None)

    def _check_new_h4_candle(self) -> None:
        """Détecte une nouvelle bougie H4 et recalcule les tendances."""
        # Échantillonner une paire pour détecter le changement H4
        if not self._trading_pairs:
            return
        sample = self._trading_pairs[0]
        try:
            candles = self._data.get_h4_candles(sample)
            if self._data.has_new_candle(sample, candles):
                logger.info("═" * 40)
                logger.info("🕐 Nouvelle bougie H4 — mise à jour de toutes les paires")
                for symbol in self._trading_pairs:
                    try:
                        self._update_trend(symbol)
                    except Exception as e:
                        logger.error("[%s] Erreur mise à jour tendance: %s", symbol, e)
                self._firebase_daily_cleanup()
        except Exception as e:
            logger.debug("Erreur check H4: %s", e)

    def _process_symbol(self, symbol: str) -> None:
        """Traite un symbole : prix → décision → action."""
        # Skip si un OCO est actif (l'exchange gère le SL/TP)
        if symbol in self._oco_orders:
            return

        ticker = self._data.get_ticker(symbol)
        if ticker is None:
            return

        trend = self._trends.get(symbol)
        position = self._positions.get(symbol)

        if trend is None:
            return

        # Vérifier l'invalidation de tendance
        old_direction = trend.direction
        trend = check_trend_invalidation(trend, ticker.last_price)
        if trend.direction != old_direction:
            self._trends[symbol] = trend
            logger.warning(
                "[%s] ⚠️ INVALIDATION | %s → %s | prix=%s",
                symbol, old_direction.value, trend.direction.value, _fmt(ticker.last_price),
            )
            self._telegram.notify_trend_change(trend, old_direction)
            try:
                fb_log_trend_change(symbol, old_direction, trend.direction, exchange=EXCHANGE_NAME)
            except Exception:
                pass

            # Sortie forcée si position RANGE et tendance confirmée
            if (
                position
                and position.strategy == StrategyType.RANGE
                and position.status in (PositionStatus.OPEN, PositionStatus.ZERO_RISK)
                and trend.direction in (TrendDirection.BULLISH, TrendDirection.BEARISH)
            ):
                logger.warning("[%s] RANGE FORCED EXIT | tendance %s", symbol, trend.direction.value)
                self._close_position_market(symbol, ticker.last_price, "Tendance confirmée (sortie forcée)")
                return

            if trend.direction == TrendDirection.NEUTRAL:
                self._update_range(symbol, trend)
                self._save_state()
            elif symbol in self._ranges:
                del self._ranges[symbol]
                self._save_state()

        # Position ouverte sans OCO → vérifier SL/TP puis tenter OCO
        if position and position.status in (PositionStatus.OPEN, PositionStatus.ZERO_RISK):
            if symbol not in self._oco_orders:
                price = ticker.last_price

                # Vérifier si TP/SL déjà franchis (conditions OCO invalides)
                # Pour SELL OCO (fermer un BUY) : TP doit être > prix, SL < prix
                # Pour BUY OCO (fermer un SELL) : SL doit être > prix, TP < prix
                oco_impossible = False
                if position.side == OrderSide.BUY:
                    if position.tp_price and price >= position.tp_price:
                        logger.info("[%s] 🎯 TP dépassé (prix=%s ≥ TP=%s) → close",
                                    symbol, _fmt(price), _fmt(position.tp_price))
                        self._close_position_market(symbol, price, "TP déjà dépassé")
                        return
                    if price <= position.sl_price:
                        logger.warning("[%s] 🛑 SL dépassé (prix=%s ≤ SL=%s) → close",
                                       symbol, _fmt(price), _fmt(position.sl_price))
                        rs = self._ranges.get(symbol)
                        if rs:
                            activate_cooldown(rs, config.RANGE_COOLDOWN_BARS)
                            self._save_state()
                        self._close_position_market(symbol, price, "SL déjà dépassé")
                        return
                else:  # SELL position
                    if position.tp_price and price <= position.tp_price:
                        logger.info("[%s] 🎯 TP dépassé (prix=%s ≤ TP=%s) → close",
                                    symbol, _fmt(price), _fmt(position.tp_price))
                        self._close_position_market(symbol, price, "TP déjà dépassé")
                        return
                    if price >= position.sl_price:
                        logger.warning("[%s] 🛑 SL dépassé (prix=%s ≥ SL=%s) → close",
                                       symbol, _fmt(price), _fmt(position.sl_price))
                        rs = self._ranges.get(symbol)
                        if rs:
                            activate_cooldown(rs, config.RANGE_COOLDOWN_BARS)
                            self._save_state()
                        self._close_position_market(symbol, price, "SL déjà dépassé")
                        return

                # Prix entre SL et TP → tenter l'OCO
                if not self.dry_run and position.tp_price:
                    if self._place_oco(symbol, position):
                        return  # OCO placé avec succès
                # Fallback : gérer manuellement en attendant
                self._manage_range_position_manual(symbol, position, ticker)
            return

        # Pas de position → chercher un signal RANGE
        if trend.direction == TrendDirection.NEUTRAL:
            self._seek_range_entry(symbol, trend, ticker)

    # ═══════════════════════════════════════════════════════════════════════════
    # GESTION POSITION (fallback manuel si OCO pas placé)
    # ═══════════════════════════════════════════════════════════════════════════

    def _manage_range_position_manual(
        self, symbol: str, position: Position, ticker
    ) -> None:
        """Gère SL/TP manuellement — fallback si OCO pas encore actif."""
        price = ticker.last_price

        if check_range_tp_hit(position, ticker):
            logger.info("[%s] 🎯 TP HIT (manual) | prix=%s", symbol, _fmt(price))
            self._close_position_market(symbol, price, "TP Range atteint (manual)")
            return

        if check_range_sl_hit(position, ticker, config.RANGE_SL_BUFFER_PERCENT):
            logger.warning("[%s] 🛑 SL HIT (manual) | prix=%s", symbol, _fmt(price))
            rs = self._ranges.get(symbol)
            if rs:
                activate_cooldown(rs, config.RANGE_COOLDOWN_BARS)
                self._save_state()
            self._close_position_market(symbol, price, "SL atteint (manual breakout)")
            return

    # ═══════════════════════════════════════════════════════════════════════════
    # ENTRÉE EN POSITION
    # ═══════════════════════════════════════════════════════════════════════════

    def _seek_range_entry(self, symbol: str, trend: TrendState, ticker) -> None:
        """Cherche un signal d'entrée Range sur Binance."""
        pos = self._positions.get(symbol)
        if pos and pos.status in (PositionStatus.OPEN, PositionStatus.ZERO_RISK):
            return

        rs = self._ranges.get(symbol)
        if rs is None:
            self._update_range(symbol, trend)
            rs = self._ranges.get(symbol)
            if rs is None:
                return

        signal = check_range_entry_signal(rs, ticker, config.RANGE_ENTRY_BUFFER_PERCENT)
        if not signal:
            return

        # Binance Spot : pas de vente à découvert — on ne prend que les BUY
        if signal["side"] == OrderSide.SELL:
            return

        signal_key = f"RANGE_{signal['side'].value.upper()}"
        prev_info = self._signal_persistence.get(symbol, {})
        if prev_info.get("key") == signal_key:
            prev_info["count"] += 1
        else:
            self._signal_persistence[symbol] = {"key": signal_key, "count": 1}
            logger.info(
                "[%s] 🔄 RANGE %s | Entry=%s | SL=%s | TP=%s",
                symbol, signal["side"].value.upper(),
                _fmt(signal["entry_price"]), _fmt(signal["sl_price"]), _fmt(signal["tp_price"]),
            )

        self._open_position(symbol, signal, ticker.last_price)

    def _open_position(
        self,
        symbol: str,
        signal: dict,
        current_price: float,
    ) -> None:
        """Ouvre une position via MARKET + place un OCO (SL+TP)."""
        self._cycle_signals += 1

        # Guards
        existing = self._positions.get(symbol)
        if existing and existing.status in (PositionStatus.OPEN, PositionStatus.ZERO_RISK):
            return

        open_count = sum(
            1 for p in self._positions.values()
            if p.status in (PositionStatus.OPEN, PositionStatus.ZERO_RISK)
        )
        if open_count >= config.BINANCE_MAX_SIMULTANEOUS_POSITIONS:
            return

        # Balances
        balances = self._client.get_balances()
        usdc_balance = next(
            (b for b in balances if b.currency == "USDC"), None
        )
        available_usdc = usdc_balance.available if usdc_balance else 0.0

        if signal["side"] == OrderSide.BUY:
            if available_usdc <= 0:
                return
            # Capital alloué (plafond virtuel pour cohabiter avec le bot Breakout)
            allocated = config.BINANCE_RANGE_ALLOCATED_BALANCE
            fiat_balance = min(allocated, available_usdc) if allocated > 0 else available_usdc
        else:
            # Binance Spot : SELL nécessite de détenir l'actif base
            base_currency = symbol.replace("USDC", "")
            base_bal = next((b for b in balances if b.currency == base_currency), None)
            if not base_bal or base_bal.available <= 0:
                return
            # Vérifier qu'on a assez de base pour couvrir le sizing prévu
            fiat_balance = base_bal.available * current_price
            if fiat_balance < 10:  # Moins de 10 USD de base → dust, ignorer
                logger.debug("[%s] SELL ignoré — solde base insuffisant (%.4f %s ≈ %.2f USD)",
                             symbol, base_bal.available, base_currency, fiat_balance)
                return

        # Risk check
        open_pos_list = [
            p for p in self._positions.values()
            if p.status in (PositionStatus.OPEN, PositionStatus.ZERO_RISK)
        ]
        current_risk = check_total_risk_exposure(
            open_pos_list, fiat_balance, config.MAX_TOTAL_RISK_PERCENT,
        )
        if current_risk >= config.MAX_TOTAL_RISK_PERCENT:
            return

        risk_pct = config.BINANCE_RISK_PERCENT_RANGE

        # Calculate position size
        size = calculate_position_size(
            account_balance=fiat_balance,
            risk_percent=risk_pct,
            entry_price=signal["entry_price"],
            sl_price=signal["sl_price"],
            max_position_percent=config.MAX_POSITION_PERCENT,
        )
        if size <= 0:
            return

        # Format according to Binance filters
        quantity_str = self._client.format_quantity(symbol, size)
        quantity = float(quantity_str)

        if not self._client.check_min_notional(symbol, quantity, signal["entry_price"]):
            logger.warning("[%s] Notionnel insuffisant, skip", symbol)
            return

        # Vérification coût réel pour BUY : qty × prix + marge frais ≤ USDC dispo
        if signal["side"] == OrderSide.BUY:
            estimated_cost = quantity * current_price * 1.002  # 0.2% marge frais/slippage
            if estimated_cost > available_usdc:
                logger.warning(
                    "[%s] Coût estimé %.2f > USDC dispo %.2f — skip",
                    symbol, estimated_cost, available_usdc,
                )
                return

        # ── 1. MARKET ORDER for entry ──
        venue_order_id = "dry-run"
        fill_price = signal["entry_price"]

        if not self.dry_run:
            try:
                result = self._client.place_market_order(
                    symbol=symbol,
                    side=signal["side"].value.upper(),
                    quantity=quantity_str,
                )
                venue_order_id = str(result.get("orderId", "unknown"))

                # Calculer le prix moyen de fill
                cq = float(result.get("cummulativeQuoteQty", 0))
                eq = float(result.get("executedQty", 0))
                if cq > 0 and eq > 0:
                    fill_price = cq / eq
                    quantity = eq  # Quantité brute remplie

                # Déduire les frais payés en base asset (sinon l'OCO échoue)
                base_currency = symbol.replace("USDC", "")
                total_commission_base = 0.0
                for fill in result.get("fills", []):
                    if fill.get("commissionAsset") == base_currency:
                        total_commission_base += float(fill.get("commission", 0))
                if total_commission_base > 0:
                    quantity -= total_commission_base
                    logger.info(
                        "[%s] Frais déduits: %.8f %s → qty nette=%.8f",
                        symbol, total_commission_base, base_currency, quantity,
                    )

                logger.info(
                    "[%s] ✅ MARKET %s fill @ %s (qty=%.8f)",
                    symbol, signal["side"].value.upper(), _fmt(fill_price), quantity,
                )
            except Exception as e:
                logger.error("[%s] ❌ MARKET order échoué: %s", symbol, e)
                self._telegram.notify_error(f"Ordre {symbol} MARKET échoué: {e}")
                return
        else:
            logger.info("[DRY-RUN] MARKET %s %s qty=%s", signal["side"].value.upper(), symbol, quantity_str)

        # Créer la position
        tp_price = signal.get("tp_price")
        position = Position(
            symbol=symbol,
            side=signal["side"],
            entry_price=fill_price,
            sl_price=signal["sl_price"],
            size=quantity,
            venue_order_id=venue_order_id,
            status=PositionStatus.OPEN,
            strategy=StrategyType.RANGE,
            tp_price=tp_price,
        )
        self._positions[symbol] = position
        self._save_state()
        self._cycle_executed += 1

        # ── 2. OCO ORDER for SL + TP ──
        if not self.dry_run and tp_price:
            self._place_oco(symbol, position)

        # Notifications
        risk_amount = fiat_balance * risk_pct
        self._telegram.notify_entry(
            symbol=symbol,
            side=signal["side"],
            entry_price=fill_price,
            sl_price=signal["sl_price"],
            size=quantity,
            risk_percent=risk_pct,
            risk_amount=risk_amount,
            strategy=StrategyType.RANGE,
            tp_price=tp_price,
        )

        # 🔥 Firebase
        try:
            current_equity = self._calculate_equity(balances)
            portfolio_risk = check_total_risk_exposure(
                [p for p in self._positions.values()
                 if p.status in (PositionStatus.OPEN, PositionStatus.ZERO_RISK)],
                fiat_balance, 1.0,
            ) if fiat_balance > 0 else 0.0

            fb_id = log_trade_opened(
                position=position,
                fill_type="taker",  # MARKET = taker
                maker_wait_seconds=0,
                risk_pct=risk_pct,
                risk_amount_usd=risk_amount,
                fiat_balance=fiat_balance,
                current_equity=current_equity,
                portfolio_risk_before=portfolio_risk,
                exchange=EXCHANGE_NAME,
            )
            if fb_id:
                position.firebase_trade_id = fb_id
                self._save_state()
        except Exception as e:
            logger.warning("🔥 Firebase log_trade_opened échoué: %s", e)

    def _place_oco(self, symbol: str, position: Position) -> bool:
        """Place un OCO (SL + TP) pour protéger une position ouverte."""
        if not position.tp_price:
            logger.warning("[%s] Pas de TP → OCO impossible", symbol)
            return False

        # Side de l'OCO = opposé à la position
        oco_side = "SELL" if position.side == OrderSide.BUY else "BUY"

        # Formater les prix
        tp_price_str = self._client.format_price(symbol, position.tp_price)
        sl_stop_str = self._client.format_price(symbol, position.sl_price)

        # SL limit = stop price ± offset (pour que le STOP_LOSS_LIMIT se remplisse)
        offset = config.BINANCE_SL_LIMIT_OFFSET_PCT
        if oco_side == "SELL":
            sl_limit = position.sl_price * (1 - offset)  # Vendre un peu en-dessous du stop
        else:
            sl_limit = position.sl_price * (1 + offset)  # Acheter un peu au-dessus du stop
        sl_limit_str = self._client.format_price(symbol, sl_limit)

        # Quantité = taille de la position, ajustée au solde réel disponible
        base_currency = symbol.replace("USDC", "")
        balances = self._client.get_balances()
        base_bal = next((b for b in balances if b.currency == base_currency), None)
        oco_qty = position.size
        if base_bal and base_bal.available < oco_qty:
            logger.info(
                "[%s] OCO qty ajustée: %.8f → %.8f (solde réel %s)",
                symbol, oco_qty, base_bal.available, base_currency,
            )
            oco_qty = base_bal.available
        qty_str = self._client.format_quantity(symbol, oco_qty)

        try:
            result = self._client.place_oco_order(
                symbol=symbol,
                side=oco_side,
                quantity=qty_str,
                tp_price=tp_price_str,
                sl_stop_price=sl_stop_str,
                sl_limit_price=sl_limit_str,
            )

            order_list_id = result.get("orderListId")
            self._oco_orders[symbol] = {
                "order_list_id": order_list_id,
            }

            logger.info(
                "[%s] 🎯 OCO placé | TP=%s | SL_stop=%s SL_limit=%s | listId=%s",
                symbol, tp_price_str, sl_stop_str, sl_limit_str, order_list_id,
            )
            return True

        except Exception as e:
            logger.error("[%s] ❌ OCO placement échoué: %s", symbol, e)
            self._telegram.notify_error(f"OCO {symbol} échoué: {e}")
            return False

    # ═══════════════════════════════════════════════════════════════════════════
    # FERMETURE DE POSITION
    # ═══════════════════════════════════════════════════════════════════════════

    def _close_position_market(
        self, symbol: str, exit_price: float, reason: str
    ) -> None:
        """Ferme une position par ordre MARKET (annule l'OCO si actif)."""
        position = self._positions.get(symbol)
        if not position:
            return

        # Annuler l'OCO actif s'il y en a un
        oco_info = self._oco_orders.get(symbol)
        if oco_info and not self.dry_run:
            try:
                self._client.cancel_order_list(
                    symbol=symbol,
                    order_list_id=oco_info.get("order_list_id"),
                )
                logger.info("[%s] OCO annulé avant close market", symbol)
            except Exception as e:
                logger.warning("[%s] Cancel OCO échoué: %s", symbol, e)
            self._oco_orders.pop(symbol, None)

        # Placer l'ordre MARKET opposé — utiliser le solde réel pour les SELL
        exit_side = "SELL" if position.side == OrderSide.BUY else "BUY"
        close_qty = position.size
        if exit_side == "SELL" and not self.dry_run:
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
                    side=exit_side,
                    quantity=qty_str,
                )
                cq = float(result.get("cummulativeQuoteQty", 0))
                eq = float(result.get("executedQty", 0))
                if cq > 0 and eq > 0:
                    actual_price = cq / eq
            except Exception as e:
                logger.error("[%s] ❌ Close MARKET échoué: %s", symbol, e)
                # Backoff
                fail_info = self._close_failures.get(symbol, {"count": 0})
                fail_info["count"] += 1
                fail_info["last_error"] = str(e)
                backoff = [60, 120, 300, 600, 1800]
                idx = min(fail_info["count"] - 1, len(backoff) - 1)
                fail_info["next_retry"] = time.time() + backoff[idx]
                self._close_failures[symbol] = fail_info
                return
        else:
            logger.info("[DRY-RUN] Close MARKET %s %s qty=%s", exit_side, symbol, qty_str)

        self._finalize_close(symbol, actual_price, reason, "taker", close_qty)

    def _finalize_close(
        self, symbol: str, exit_price: float, reason: str, fill_type: str,
        actual_exit_size: Optional[float] = None,
    ) -> None:
        """Finalise la clôture d'une position (PnL, Firebase, notifications)."""
        position = self._positions.get(symbol)
        if not position:
            return

        # Reset close failures
        if symbol in self._close_failures:
            del self._close_failures[symbol]
            if position.firebase_trade_id:
                try:
                    fb_clear_close_failure(position.firebase_trade_id)
                except Exception:
                    pass

        # Taille réellement vendue (après ajustement au solde réel)
        exit_size = actual_exit_size if actual_exit_size is not None else position.size

        # PnL — calculé sur la taille réellement vendue
        if position.side == OrderSide.BUY:
            pnl_gross = (exit_price - position.entry_price) * exit_size
        else:
            pnl_gross = (position.entry_price - exit_price) * exit_size

        notional = exit_size * position.entry_price
        fee_rate = config.BINANCE_TAKER_FEE if fill_type == "taker" else config.BINANCE_MAKER_FEE
        fees = notional * fee_rate + exit_size * exit_price * fee_rate
        pnl_net = pnl_gross - fees
        pnl_pct = pnl_net / notional if notional > 0 else 0

        pnl_emoji = "🟢" if pnl_net >= 0 else "🔴"
        logger.info(
            "[%s] %s CLOSE | %s | PnL=$%+.4f (%+.2f%%) | fees=$%.4f | size=%.8f (orig=%.8f)",
            symbol, pnl_emoji, reason, pnl_net, pnl_pct * 100, fees,
            exit_size, position.size,
        )

        position.status = PositionStatus.CLOSED
        position.pnl = pnl_net
        self._oco_orders.pop(symbol, None)
        self._save_state()

        # Telegram
        self._telegram.notify_sl_hit(position, exit_price)

        # Equity
        equity_after = 0.0
        try:
            balances = self._client.get_balances()
            equity_after = self._calculate_equity(balances)
        except Exception:
            pass

        # Firebase
        if position.firebase_trade_id:
            try:
                log_trade_closed(
                    trade_id=position.firebase_trade_id,
                    position=position,
                    exit_price=exit_price,
                    reason=reason,
                    fill_type=fill_type,
                    equity_after=equity_after,
                    actual_exit_size=exit_size,
                )
            except Exception as e:
                logger.warning("🔥 Firebase log_trade_closed échoué: %s", e)

    # ═══════════════════════════════════════════════════════════════════════════
    # HELPERS
    # ═══════════════════════════════════════════════════════════════════════════

    def _calculate_equity(self, balances: Optional[list[Balance]] = None) -> float:
        """Calcule l'equity totale du compte Binance."""
        if balances is None:
            balances = self._client.get_balances()

        # Construire des TickerData pour get_total_equity
        fiat_set = {"USD", "EUR", "GBP", "USDC", "USDT", "BUSD"}
        crypto_tickers = []

        for b in balances:
            if b.total > 0 and b.currency not in fiat_set:
                try:
                    price = self._client.get_ticker_price(f"{b.currency}USDC")
                    if price > 0:
                        crypto_tickers.append(TickerData(
                            symbol=f"{b.currency}-USD",  # Format attendu par get_total_equity
                            bid=price, ask=price, mid=price, last_price=price,
                        ))
                except Exception:
                    pass

        # get_total_equity cherche "USD" → on doit adapter pour USDC
        # Créer un balance "USD" synthétique à partir de USDC
        adjusted_balances = []
        for b in balances:
            if b.currency == "USDC":
                adjusted_balances.append(Balance(
                    currency="USD",
                    available=b.available,
                    reserved=b.reserved,
                    total=b.total,
                ))
            else:
                adjusted_balances.append(b)

        return get_total_equity(adjusted_balances, crypto_tickers)

    def _update_trend(self, symbol: str) -> None:
        """Recalcule la tendance à partir des bougies H4."""
        candles = self._data.get_h4_candles(symbol)
        if len(candles) < (2 * config.SWING_LOOKBACK + 1):
            logger.warning("[%s] Pas assez de bougies (%d)", symbol, len(candles))
            return

        swings = detect_swings(candles, lookback=config.SWING_LOOKBACK)
        if len(swings) < 4:
            self._trends[symbol] = TrendState(symbol=symbol, direction=TrendDirection.NEUTRAL)
            return

        old_trend = self._trends.get(symbol)
        old_direction = old_trend.direction if old_trend else TrendDirection.NEUTRAL

        new_trend = determine_trend(swings, symbol)
        self._trends[symbol] = new_trend

        if new_trend.direction != old_direction:
            logger.warning("[%s] 🔄 %s → %s", symbol, old_direction.value, new_trend.direction.value)
            try:
                fb_log_trend_change(symbol, old_direction, new_trend.direction, exchange=EXCHANGE_NAME)
            except Exception:
                pass

        if new_trend.direction == TrendDirection.NEUTRAL:
            self._update_range(symbol, new_trend)
            rs = self._ranges.get(symbol)
            if rs and rs.is_valid:
                logger.info(
                    "[%s] ⚪ NEUTRAL | RANGE w=%.1f%% H=%s L=%s%s",
                    symbol, rs.range_width_pct * 100,
                    _fmt(rs.range_high), _fmt(rs.range_low),
                    _cooldown_str(rs),
                )
        elif symbol in self._ranges:
            del self._ranges[symbol]

    def _update_range(self, symbol: str, trend: TrendState) -> None:
        old_rs = self._ranges.get(symbol)
        rs = build_range_from_trend(trend, config.RANGE_WIDTH_MIN)
        if rs is not None:
            if old_rs and old_rs.cooldown_until > 0:
                rs.cooldown_until = old_rs.cooldown_until
            self._ranges[symbol] = rs

    def _maybe_heartbeat(self) -> None:
        now = time.time()
        if now - self._last_heartbeat_time < config.HEARTBEAT_INTERVAL_SECONDS:
            return
        self._last_heartbeat_time = now

        open_pos = [
            p for p in self._positions.values()
            if p.status in (PositionStatus.OPEN, PositionStatus.ZERO_RISK)
        ]
        try:
            balances = self._client.get_balances()
            equity = self._calculate_equity(balances)
        except Exception:
            equity = 0
            balances = []

        logger.info(
            "💓 BINANCE | %d/%d positions | equity=$%.2f | cycle #%d",
            len(open_pos), config.BINANCE_MAX_SIMULTANEOUS_POSITIONS,
            equity, self._cycle_count,
        )

        try:
            fb_log_heartbeat(
                open_positions=len(open_pos),
                total_equity=equity,
                total_risk_pct=0,
                pairs_count=len(self._trading_pairs),
            )
        except Exception:
            pass

    def _firebase_daily_cleanup(self) -> None:
        from datetime import datetime, timezone
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        if today == self._last_cleanup_date:
            return
        self._last_cleanup_date = today
        try:
            fb_cleanup_events()
        except Exception:
            pass

    def _shutdown(self) -> None:
        logger.info("🛑 Arrêt de TradeX Binance...")
        self._save_state()
        self._client.close()
        self._telegram.close()
        logger.info("TradeX Binance arrêté proprement")


# ── Point d'entrée ─────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="TradeX Binance – Bot USDC")
    parser.add_argument("--dry-run", action="store_true", help="Mode simulation")
    args = parser.parse_args()

    bot = TradeXBinanceBot(dry_run=args.dry_run)
    signal.signal(signal.SIGTERM, lambda *_: bot.stop())
    bot.run()


if __name__ == "__main__":
    main()
