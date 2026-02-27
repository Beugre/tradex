"""
Boucle principale du bot TradeX.

Deux rythmes :
1. Toutes les 30 secondes : polling prix → vérifier seuils d'entrée / SL / zero-risk
2. À chaque nouvelle bougie H4 : recalculer swings et tendance

Usage :
    python -m src.bot              # Production
    python -m src.bot --dry-run    # Log les ordres sans les exécuter
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
    OrderSide,
    Position,
    PositionStatus,
    RangeState,
    StrategyType,
    SwingLevel,
    TrendDirection,
    TrendState,
)
from src.core.strategy_trend import (
    build_entry_order,
    build_exit_order,
    check_entry_signal,
    check_sl_hit,
    process_trailing_stop,
    process_zero_risk,
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
from src.exchange.data_provider import DataProvider
from src.exchange.revolut_client import RevolutXClient
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
logger = logging.getLogger("tradex.bot")

# Réduire le bruit des librairies HTTP (1 ligne par requête toutes les 30s × 6 paires)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)


def _fmt(price: float) -> str:
    """Formate un prix de façon lisible selon sa grandeur (min 4 décimales)."""
    if price >= 1000:
        return f"{price:,.4f}"
    elif price >= 1:
        return f"{price:.4f}"
    elif price >= 0.0001:
        return f"{price:.6f}"
    else:
        # Sub-cent tokens (PEPE, SHIB, BONK, FLOKI…)
        decimals = 6
        temp = price
        while temp < 0.01 and decimals < 10:
            temp *= 10
            decimals += 1
        return f"{price:.{decimals}f}"


def _pct_dist(current: float, target: float) -> str:
    """Retourne la distance en % entre current et target, avec signe."""
    if current == 0:
        return "N/A"
    dist = ((target - current) / current) * 100
    return f"{dist:+.2f}%"


def _cooldown_str(rs: RangeState) -> str:
    """Retourne un label de cooldown avec le nombre de barres restantes, ou ''."""
    if not is_in_cooldown(rs):
        return ""
    remaining_s = rs.cooldown_until - int(time.time())
    bar_s = 4 * 3600  # H4
    remaining_bars = max(1, -(-remaining_s // bar_s))  # ceil division
    return f" ⏳ COOLDOWN ({remaining_bars}/{config.RANGE_COOLDOWN_BARS} bars)"


# Emojis de tendance
_TREND_EMOJI = {
    TrendDirection.BULLISH: "🟢 BULLISH",
    TrendDirection.BEARISH: "🔴 BEARISH",
    TrendDirection.NEUTRAL: "⚪ NEUTRAL",
}


class TradeXBot:
    """Bot de trading TradeX — boucle principale."""

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

        # Persistance
        self._store = PositionStore()

        # État par paire
        self._trends: dict[str, TrendState] = {}
        self._ranges: dict[str, RangeState] = {}
        self._positions: dict[str, Position] = {}

        # Heartbeat (unique timer périodique)
        self._last_heartbeat_time: float = 0.0
        self._cycle_count: int = 0

        # Tracker d'état précédent pour log-on-change (évite le bruit)
        self._prev_state: dict[str, dict] = {}
        self._prev_ignored: dict[str, str] = {}  # symbol → dernière raison d'ignore

        # Compteur de signaux persistants (symbol → {"key": str, "count": int})
        self._signal_persistence: dict[str, dict] = {}

        # ── Gestion des échecs de clôture (évite le spam API) ──
        # symbol → {"count": int, "next_retry": float (timestamp), "last_error": str}
        self._close_failures: dict[str, dict] = {}

        # Firebase cleanup (1 fois par jour)
        self._last_cleanup_date: str = ""

        if dry_run:
            logger.info("🔧 Mode DRY-RUN activé — aucun ordre ne sera exécuté")

    def run(self) -> None:
        """Lance la boucle principale."""
        self._running = True
        logger.info("═" * 60)
        logger.info("🚀 TradeX démarré — %s",
            ("Dual Strategy (Trend + Range)" if config.ENABLE_TREND and config.ENABLE_RANGE
             else "Range Only" if config.ENABLE_RANGE
             else "Trend Only" if config.ENABLE_TREND
             else "AUCUNE stratégie active!"),
        )
        logger.info(
            "   Paires     : %d paires | %s",
            len(config.TRADING_PAIRS),
            ", ".join(config.TRADING_PAIRS),
        )
        logger.info(
            "   📊 TREND   : risque %.0f%% | Max/pos: %.0f%% | Max simultanées: %d",
            config.RISK_PERCENT_TREND * 100,
            config.MAX_POSITION_PERCENT * 100,
            config.MAX_SIMULTANEOUS_POSITIONS,
        )
        logger.info(
            "   🔄 RANGE   : risque %.0f%% | Largeur min: %.0f%% | Cooldown: %d bougies",
            config.RISK_PERCENT_RANGE * 100,
            config.RANGE_WIDTH_MIN * 100,
            config.RANGE_COOLDOWN_BARS,
        )
        logger.info(
            "   🛡️ Global   : risque total max %.0f%%",
            config.MAX_TOTAL_RISK_PERCENT * 100,
        )
        logger.info(
            "   Buffers    : entrée %.1f%% | SL %.1f%%",
            config.ENTRY_BUFFER_PERCENT * 100,
            config.SL_BUFFER_PERCENT * 100,
        )
        logger.info(
            "   Protection : zero-risk à +%.0f%% (lock %.1f%%) | trailing %.0f%%",
            config.ZERO_RISK_TRIGGER_PERCENT * 100,
            config.ZERO_RISK_LOCK_PERCENT * 100,
            config.TRAILING_STOP_PERCENT * 100,
        )
        logger.info(
            "   Polling    : %ds | Swing lookback: %d bougies",
            config.POLLING_INTERVAL_SECONDS,
            config.SWING_LOOKBACK,
        )
        logger.info(
            "   Maker-First: attente %ds (maker 0%% → fallback taker 0.09%%)",
            config.MAKER_WAIT_SECONDS,
        )
        logger.info(
            "   Stratégies : TREND=%s | RANGE=%s",
            "✅" if config.ENABLE_TREND else "❌",
            "✅" if config.ENABLE_RANGE else "❌",
        )
        logger.info("═" * 60)

        # Initialiser les tendances au démarrage
        self._initialize()

        try:
            while self._running:
                self._tick()
                time.sleep(config.POLLING_INTERVAL_SECONDS)
        except KeyboardInterrupt:
            logger.info("Arrêt demandé par l'utilisateur")
        finally:
            self._shutdown()

    def stop(self) -> None:
        """Arrête le bot proprement."""
        self._running = False

    # ── Initialisation ─────────────────────────────────────────────────────────

    def _initialize(self) -> None:
        """Charge les bougies initiales et calcule les tendances."""
        # ── Réconcilier l'état local avec l'exchange (source de vérité) ──
        self._reconcile_positions()

        # ── 🧹 Firebase : nettoyage des vieux events au démarrage ──
        self._firebase_daily_cleanup()

        logger.info("── Initialisation des %d paires... ──", len(config.TRADING_PAIRS))
        for symbol in config.TRADING_PAIRS:
            try:
                self._update_trend(symbol)
            except Exception as e:
                logger.error("[%s] ❌ Erreur d'initialisation: %s", symbol, e)
                self._telegram.notify_error(
                    f"Initialisation {symbol} échouée: {e}"
                )
        # Résumé
        open_count = sum(
            1 for p in self._positions.values()
            if p.status in (PositionStatus.OPEN, PositionStatus.ZERO_RISK)
        )
        logger.info("── Initialisation terminée ──")
        logger.info(
            "   Positions ouvertes: %d/%d | Paires en BULLISH: %d | BEARISH: %d | NEUTRAL: %d",
            open_count,
            config.MAX_SIMULTANEOUS_POSITIONS,
            sum(1 for t in self._trends.values() if t.direction == TrendDirection.BULLISH),
            sum(1 for t in self._trends.values() if t.direction == TrendDirection.BEARISH),
            sum(1 for t in self._trends.values() if t.direction == TrendDirection.NEUTRAL),
        )

    def _reconcile_positions(self) -> None:
        """Réconcilie l'état local (state.json) avec l'exchange Revolut X.

        Source de vérité = exchange (balances réelles).
        Le state.json fournit les métadonnées (SL, prix d'entrée, stratégie).
        """
        # 1. Charger l'état local (positions + ranges)
        loaded_positions, loaded_ranges = self._store.load()
        self._ranges = loaded_ranges

        active_local = {
            sym: pos for sym, pos in loaded_positions.items()
            if pos.status in (PositionStatus.OPEN, PositionStatus.ZERO_RISK)
        }

        if not active_local:
            logger.info("📂 Aucune position locale active — vérification des soldes exchange")
        else:
            logger.info(
                "📂 %d position(s) locale(s) active(s) à réconcilier",
                len(active_local),
            )

        # 2. Récupérer l'état réel de l'exchange
        try:
            balances = self._client.get_balances()
        except Exception as e:
            logger.error(
                "❌ Impossible de contacter l'exchange pour la réconciliation: %s", e,
            )
            logger.warning("⚠️ Repli sur l'état local uniquement")
            for sym, pos in active_local.items():
                self._positions[sym] = pos
                logger.info(
                    "[%s] ♻️ Position restaurée (local, non vérifiée): %s %s @ %s",
                    sym, pos.side.value.upper(), pos.strategy.value,
                    _fmt(pos.entry_price),
                )
            self._telegram.notify_error(
                f"Réconciliation impossible (API down): {e} — état local utilisé"
            )
            return

        # Map des balances par devise
        balance_map = {b.currency: b for b in balances}

        reconciled = 0
        removed = 0

        # 3. Vérifier chaque position locale OPEN/ZERO_RISK
        for sym, pos in active_local.items():
            base_currency = sym.split("-")[0]  # ex: "BTC" de "BTC-USD"
            base_bal = balance_map.get(base_currency)
            held = (base_bal.available + base_bal.reserved) if base_bal else 0.0

            if pos.side == OrderSide.BUY:
                # BUY spot = on devrait détenir le base asset
                if held >= pos.size * 0.90:  # Tolérance 10% pour frais
                    self._positions[sym] = pos
                    logger.info(
                        "[%s] ✅ Position BUY confirmée | solde %s=%.8f ≥ size=%.8f",
                        sym, base_currency, held, pos.size,
                    )
                    reconciled += 1
                else:
                    logger.warning(
                        "[%s] ⚠️ Position BUY locale mais solde %s=%.8f < size=%.8f → retirée",
                        sym, base_currency, held, pos.size,
                    )
                    removed += 1

            elif pos.side == OrderSide.SELL:
                # SELL spot = on a vendu le crypto → solde base devrait être faible
                if held < pos.size * 0.10:
                    self._positions[sym] = pos
                    logger.info(
                        "[%s] ✅ Position SELL confirmée | solde %s=%.8f faible (vendu: %.8f)",
                        sym, base_currency, held, pos.size,
                    )
                    reconciled += 1
                else:
                    logger.warning(
                        "[%s] ⚠️ Position SELL locale mais solde %s=%.8f élevé → retirée (rachetée ?)",
                        sym, base_currency, held,
                    )
                    removed += 1

        # 4. Reconstruire les positions orphelines (crypto détenu sans position trackée)
        orphans: list[str] = []
        recovered: list[Position] = []
        for currency, bal in balance_map.items():
            if currency in ("USD", "EUR", "USDT", "USDC"):
                continue
            if bal.total <= 0:
                continue
            symbol = f"{currency}-USD"
            if symbol in config.TRADING_PAIRS and symbol not in self._positions:
                # Tenter de reconstruire la position à partir du prix actuel
                try:
                    ticker = self._data.get_ticker(symbol)
                    if ticker is None:
                        orphans.append(f"{currency}: {bal.total:.8f} (prix indisponible)")
                        logger.warning(
                            "[%s] 🔍 Solde orphelin: %.8f %s — pas de prix pour reconstruire",
                            symbol, bal.total, currency,
                        )
                        continue

                    current_price = ticker.last_price

                    # Déterminer la stratégie selon le mode actif et le range disponible
                    recovery_strategy = StrategyType.TREND
                    tp_price = None
                    range_state = self._ranges.get(symbol)

                    if config.ENABLE_RANGE and range_state and range_state.is_valid:
                        recovery_strategy = StrategyType.RANGE
                        tp_price = range_state.range_mid
                        # SL = borne opposée du range + buffer
                        sl_price = range_state.range_low * (1 - config.RANGE_ENTRY_BUFFER_PERCENT)
                        logger.info(
                            "[%s] 🔄 Reconstruction RANGE | TP=%s (mid) | SL=%s (low - buffer)",
                            symbol, _fmt(tp_price), _fmt(sl_price),
                        )
                    elif not config.ENABLE_TREND and config.ENABLE_RANGE:
                        # Mode Range Only mais pas de range dispo → SL défensif
                        recovery_strategy = StrategyType.RANGE
                        sl_price = current_price * (1 - config.RECOVERY_SL_PERCENT)
                        logger.warning(
                            "[%s] 🔄 Reconstruction RANGE (sans range actif) | SL=%s (-%d%%)",
                            symbol, _fmt(sl_price), int(config.RECOVERY_SL_PERCENT * 100),
                        )
                    else:
                        sl_price = current_price * (1 - config.RECOVERY_SL_PERCENT)

                    position = Position(
                        symbol=symbol,
                        side=OrderSide.BUY,
                        entry_price=current_price,
                        sl_price=sl_price,
                        size=bal.total,
                        venue_order_id="recovered",
                        status=PositionStatus.OPEN,
                        strategy=recovery_strategy,
                        tp_price=tp_price,
                    )
                    self._positions[symbol] = position
                    recovered.append(position)
                    logger.warning(
                        "[%s] 🔄 Position RECONSTRUITE %s | %.8f %s @ %s | SL=%s | TP=%s",
                        symbol, recovery_strategy.value,
                        bal.total, currency, _fmt(current_price),
                        _fmt(sl_price),
                        _fmt(tp_price) if tp_price else "N/A",
                    )
                except Exception as e:
                    orphans.append(f"{currency}: {bal.total:.8f} (erreur: {e})")
                    logger.error(
                        "[%s] ❌ Impossible de reconstruire: %s", symbol, e,
                    )

        # 5. Résumé + notification
        logger.info(
            "── Réconciliation: %d confirmées, %d retirées, %d reconstruites, %d orphelins ──",
            reconciled, removed, len(recovered), len(orphans),
        )

        self._save_state()
        self._telegram.notify_reconciliation(reconciled, removed, orphans, recovered)

    def _save_state(self) -> None:
        """Persiste les positions et ranges sur disque."""
        self._store.save(self._positions, self._ranges)

    # ── Tick principal (toutes les 30s) ────────────────────────────────────────

    def _tick(self) -> None:
        """Un cycle de polling : vérifier prix, seuils, SL, zero-risk."""
        # ── Header de cycle ───────────────────────────────────────────
        now_str = time.strftime("%H:%M:%S")
        logger.debug("══════════ CYCLE %s ══════════", now_str)

        # ── Check si nouvelle bougie H4 (basé sur le temps) ───────────
        self._check_new_h4_candle()

        # Compteurs pour le résumé cycle
        self._cycle_signals = 0
        self._cycle_executed = 0

        for symbol in config.TRADING_PAIRS:
            try:
                self._process_symbol(symbol)
            except Exception as e:
                logger.error("[%s] Erreur dans le tick: %s", symbol, e)

        # ── Compteur de cycles + heartbeat ─────────────────────────────
        self._cycle_count += 1
        self._maybe_heartbeat()

    def _maybe_heartbeat(self) -> None:
        """Heartbeat minimal toutes les HEARTBEAT_INTERVAL_SECONDS.

        Une seule ligne de vie + détail positions si ouvertes.
        """
        now = time.time()
        if now - self._last_heartbeat_time < config.HEARTBEAT_INTERVAL_SECONDS:
            return
        self._last_heartbeat_time = now

        open_positions = [
            p for p in self._positions.values()
            if p.status in (PositionStatus.OPEN, PositionStatus.ZERO_RISK)
        ]

        # Calculer le risque total + equity réelle
        total_risk_pct = 0.0
        total_equity = 0.0
        try:
            balances = self._data.get_balances()
            fiat_balance, _ = get_fiat_balance(balances)
            if open_positions and fiat_balance > 0:
                total_risk_pct = check_total_risk_exposure(
                    open_positions, fiat_balance, 1.0,
                ) * 100
            # Equity réelle = USD + toutes les cryptos valorisées
            # Récupérer un ticker par crypto en balance (1 appel API par crypto)
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

        # Map compacte des tendances : BTC:BEAR SOL:BEAR ...
        trend_tags = []
        for sym in config.TRADING_PAIRS:
            t = self._trends.get(sym)
            if t is None:
                continue
            short = sym.split("-")[0]
            d = {TrendDirection.BULLISH: "BULL", TrendDirection.BEARISH: "BEAR", TrendDirection.NEUTRAL: "NEUT"}
            trend_tags.append(f"{short}:{d.get(t.direction, '?')}")

        logger.info(
            "💓 Alive | cycle=%d | positions=%d/%d | risk=%.1f%% | equity=$%.2f | %s",
            self._cycle_count,
            len(open_positions),
            config.MAX_SIMULTANEOUS_POSITIONS,
            total_risk_pct,
            total_equity,
            " ".join(trend_tags),
        )

        # 🔥 Firebase heartbeat
        try:
            fb_log_heartbeat(
                open_positions=len(open_positions),
                total_equity=total_equity,
                total_risk_pct=total_risk_pct / 100,
                pairs_count=len(config.TRADING_PAIRS),
            )
        except Exception:
            pass

        # Détail des positions ouvertes (seulement s'il y en a)
        for pos in open_positions:
            try:
                ticker = self._data.get_ticker(pos.symbol)
                if ticker is None:
                    continue
                price = ticker.last_price
            except Exception:
                continue

            if pos.side == OrderSide.BUY:
                pnl_pct = ((price - pos.entry_price) / pos.entry_price) * 100
                pnl_usd = (price - pos.entry_price) * pos.size
            else:
                pnl_pct = ((pos.entry_price - price) / pos.entry_price) * 100
                pnl_usd = (pos.entry_price - price) * pos.size

            raw_sl = pos.zero_risk_sl if pos.zero_risk_sl else pos.sl_price
            status_tag = "ZR" if pos.status == PositionStatus.ZERO_RISK else "OPEN"
            strat_tag = "T" if pos.strategy == StrategyType.TREND else "R"
            pnl_icon = "🟢" if pnl_pct >= 0 else "🔴"

            # SL effectif (avec buffer) — c'est le vrai seuil de déclenchement
            if raw_sl and raw_sl > 0:
                sl_buf = (
                    config.RANGE_SL_BUFFER_PERCENT
                    if pos.strategy == StrategyType.RANGE
                    else config.SL_BUFFER_PERCENT
                )
                if pos.side == OrderSide.BUY:
                    effective_sl = raw_sl * (1 - sl_buf)
                else:
                    effective_sl = raw_sl * (1 + sl_buf)
            else:
                effective_sl = raw_sl

            # Distance au SL effectif et TP
            sl_dist = ""
            tp_dist = ""
            if effective_sl and effective_sl > 0:
                if pos.side == OrderSide.BUY:
                    sl_pct = (price - effective_sl) / price * 100
                else:
                    sl_pct = (effective_sl - price) / price * 100
                sl_dist = f" ({sl_pct:+.1f}%)"
            if pos.tp_price and pos.tp_price > 0:
                if pos.side == OrderSide.BUY:
                    tp_pct = (pos.tp_price - price) / price * 100
                else:
                    tp_pct = (price - pos.tp_price) / price * 100
                tp_dist = f" | TP {_fmt(pos.tp_price)} ({tp_pct:+.1f}%)"

            logger.info(
                "   %s %s %s %s | %s @ %s → %s | P&L %+.2f%% (%+.4f$) | SL %s%s%s",
                pnl_icon, strat_tag, status_tag,
                pos.side.value.upper(),
                pos.symbol, _fmt(pos.entry_price), _fmt(price),
                pnl_pct, pnl_usd, _fmt(effective_sl), sl_dist, tp_dist,
            )

            # ── Alerte si clôture bloquée ──
            fail_info = self._close_failures.get(pos.symbol)
            if fail_info:
                if fail_info.get("permanent"):
                    logger.warning(
                        "   🚫 %s CLOSE PERMANENTE — intervention manuelle requise | %s",
                        pos.symbol, fail_info.get("last_error", "?")[:80],
                    )
                else:
                    remaining = max(0, int(fail_info["next_retry"] - time.time()))
                    logger.warning(
                        "   ⚠️ %s CLOSE BLOQUÉE (×%d) — retry dans %ds | %s",
                        pos.symbol, fail_info["count"], remaining,
                        fail_info.get("last_error", "?")[:80],
                    )

    def _check_new_h4_candle(self) -> None:
        """Vérifie si une nouvelle bougie H4 est disponible (toutes les 4h)."""
        import math
        now = time.time()
        current_h4_slot = math.floor(now / (4 * 3600))

        if not hasattr(self, "_last_h4_slot"):
            self._last_h4_slot = current_h4_slot
            return  # Déjà initialisé au démarrage

        if current_h4_slot > self._last_h4_slot:
            self._last_h4_slot = current_h4_slot
            logger.info("═" * 60)
            logger.info("🕐 NOUVELLE BOUGIE H4 — recalcul des tendances")
            logger.info("═" * 60)
            for symbol in config.TRADING_PAIRS:
                try:
                    self._update_trend(symbol)
                except Exception as e:
                    logger.error("[%s] Erreur de mise à jour tendance: %s", symbol, e)

            # 🧹 Firebase cleanup quotidien (1 fois par jour, piggyback sur le cycle H4)
            self._firebase_daily_cleanup()

    def _process_symbol(self, symbol: str) -> None:
        """Traite un symbole : prix → décision → action (fast loop, tickers seulement)."""
        ticker = self._data.get_ticker(symbol)
        if ticker is None:
            return

        trend = self._trends.get(symbol)
        position = self._positions.get(symbol)

        if trend is None:
            return

        # ── Log structuré (on-change) ─────────────────────────────────
        self._log_symbol_if_changed(symbol, ticker.last_price, trend, position)

        # ── Vérifier l'invalidation de tendance ──────────────────────────
        old_direction = trend.direction
        trend = check_trend_invalidation(trend, ticker.last_price)
        if trend.direction != old_direction:
            self._trends[symbol] = trend
            logger.info(self._separator(symbol))
            logger.warning(
                "[%s] ⚠️ INVALIDATION | %s → %s | prix=%s",
                symbol,
                old_direction.value,
                trend.direction.value,
                _fmt(ticker.last_price),
            )
            self._telegram.notify_trend_change(trend, old_direction)
            # 🔥 Firebase
            try:
                fb_log_trend_change(symbol, old_direction, trend.direction)
            except Exception:
                pass

            # ── Sortie forcée si position RANGE et tendance confirmée ──
            if (
                position
                and position.strategy == StrategyType.RANGE
                and position.status in (PositionStatus.OPEN, PositionStatus.ZERO_RISK)
                and trend.direction in (TrendDirection.BULLISH, TrendDirection.BEARISH)
            ):
                logger.warning(
                    "[%s] STRAT=RANGE | FORCED EXIT | tendance %s confirmée",
                    symbol,
                    trend.direction.value,
                )
                self._close_position(symbol, ticker.last_price, "Tendance confirmée (sortie forcée RANGE)", forced=True)
                return

            # ── Mettre à jour le range si passage en NEUTRAL ──
            if trend.direction == TrendDirection.NEUTRAL:
                self._update_range(symbol, trend)
                self._save_state()
            else:
                # Plus en NEUTRAL → invalider le range
                if symbol in self._ranges:
                    del self._ranges[symbol]
                    self._save_state()

        # ── Dispatch selon la stratégie ──────────────────────────────────
        if position and position.status in (
            PositionStatus.OPEN,
            PositionStatus.ZERO_RISK,
        ):
            # ── Position ouverte : gérer selon sa stratégie ──
            if position.strategy == StrategyType.RANGE:
                self._manage_range_position(symbol, position, ticker)
            else:
                self._manage_trend_position(symbol, position, ticker)
        else:
            # ── Pas de position : chercher un signal ──
            if config.ENABLE_TREND and trend.direction in (TrendDirection.BULLISH, TrendDirection.BEARISH):
                self._seek_trend_entry(symbol, trend, ticker)
            elif config.ENABLE_RANGE and trend.direction == TrendDirection.NEUTRAL:
                self._seek_range_entry(symbol, trend, ticker)

    # ── Gestion position TREND ─────────────────────────────────────────────────

    def _manage_trend_position(
        self, symbol: str, position: Position, ticker
    ) -> None:
        """Gère SL, zero-risk, trailing pour une position TREND."""
        raw_sl = position.zero_risk_sl if position.zero_risk_sl else position.sl_price
        # SL effectif (avec buffer) pour le debug
        if raw_sl and raw_sl > 0:
            if position.side == OrderSide.BUY:
                eff_sl = raw_sl * (1 - config.SL_BUFFER_PERCENT)
            else:
                eff_sl = raw_sl * (1 + config.SL_BUFFER_PERCENT)
        else:
            eff_sl = raw_sl
        logger.debug(
            "[%s] TREND CHECK | prix=%s | SL_eff=%s | ZR=%s",
            symbol, _fmt(ticker.last_price), _fmt(eff_sl),
            "oui" if position.is_zero_risk_applied else "non",
        )
        if check_sl_hit(position, ticker, config.SL_BUFFER_PERCENT):
            logger.info(self._separator(symbol))
            logger.warning(
                "[%s] STRAT=TREND | 🛑 SL HIT | prix=%s | SL=%s",
                symbol, _fmt(ticker.last_price), _fmt(position.sl_price),
            )
            self._close_position(symbol, ticker.last_price, "SL atteint (TREND)")
            return

        new_sl = process_zero_risk(
            position, ticker.last_price,
            config.ZERO_RISK_TRIGGER_PERCENT,
            config.ZERO_RISK_LOCK_PERCENT,
        )
        if new_sl is not None:
            logger.info(self._separator(symbol))
            logger.info(
                "[%s] STRAT=TREND | 🔒 ZERO-RISK | nouveau SL=%s",
                symbol, _fmt(new_sl),
            )
            self._telegram.notify_zero_risk(position, new_sl)
            self._save_state()
            # 🔥 Firebase
            if position.firebase_trade_id:
                try:
                    log_zero_risk_applied(position.firebase_trade_id, new_sl)
                except Exception:
                    pass

        trailing_sl = process_trailing_stop(
            position, ticker.last_price, config.TRAILING_STOP_PERCENT,
        )
        if trailing_sl is not None:
            logger.info(self._separator(symbol))
            logger.info(
                "[%s] STRAT=TREND | 📈 TRAILING | nouveau SL=%s",
                symbol, _fmt(trailing_sl),
            )
            self._telegram.notify_trailing_stop(position, trailing_sl)
            self._save_state()
            # 🔥 Firebase
            if position.firebase_trade_id:
                try:
                    log_trailing_sl_update(position.firebase_trade_id, trailing_sl)
                except Exception:
                    pass

    # ── Gestion position RANGE ─────────────────────────────────────────────────

    def _manage_range_position(
        self, symbol: str, position: Position, ticker
    ) -> None:
        """Gère TP et SL pour une position RANGE."""
        price = ticker.last_price

        # SL effectif = raw SL - buffer (c'est le vrai seuil de déclenchement)
        raw_sl = position.sl_price
        if raw_sl and raw_sl > 0:
            if position.side == OrderSide.BUY:
                eff_sl = raw_sl * (1 - config.RANGE_SL_BUFFER_PERCENT)
            else:
                eff_sl = raw_sl * (1 + config.RANGE_SL_BUFFER_PERCENT)
        else:
            eff_sl = raw_sl
        sl_label = _fmt(eff_sl) if eff_sl else "N/A"
        tp_label = _fmt(position.tp_price) if position.tp_price else "N/A"
        logger.debug(
            "[%s] RANGE CHECK | prix=%s | SL_eff=%s | TP=%s",
            symbol, _fmt(price), sl_label, tp_label,
        )

        # TP au milieu du range
        if check_range_tp_hit(position, ticker):
            logger.info(self._separator(symbol))
            logger.info(
                "[%s] STRAT=RANGE | 🎯 TP HIT | prix=%s | TP=%s",
                symbol, _fmt(ticker.last_price), _fmt(position.tp_price or 0),
            )
            self._close_position(symbol, ticker.last_price, "TP Range atteint", is_range_tp=True)
            return

        # SL = breakout du range
        if check_range_sl_hit(position, ticker, config.RANGE_SL_BUFFER_PERCENT):
            logger.info(self._separator(symbol))
            logger.warning(
                "[%s] STRAT=RANGE | 🛑 SL HIT (breakout) | prix=%s",
                symbol, _fmt(ticker.last_price),
            )
            # Activer le cooldown
            rs = self._ranges.get(symbol)
            if rs:
                activate_cooldown(rs, config.RANGE_COOLDOWN_BARS)
                self._save_state()
            self._close_position(symbol, ticker.last_price, "SL atteint (RANGE breakout)")
            return

    # ── Recherche d'entrée TREND ───────────────────────────────────────────────

    def _seek_trend_entry(self, symbol: str, trend: TrendState, ticker) -> None:
        """Cherche un signal d'entrée Trend Following."""
        signal = check_entry_signal(trend, ticker, config.ENTRY_BUFFER_PERCENT)
        signal_key = f"TREND_{signal['side'].value.upper()}" if signal else None

        if signal:
            # Compteur de persistance
            prev_info = self._signal_persistence.get(symbol, {})
            if prev_info.get("key") == signal_key:
                prev_info["count"] += 1
                # Silence total après la 1ère occurrence — on log uniquement en DEBUG
                logger.debug(
                    "[%s] TREND SIGNAL %s persisting (×%d)",
                    symbol, signal["side"].value.upper(), prev_info["count"],
                )
            else:
                # Nouveau signal → reset compteur, log immédiat
                self._signal_persistence[symbol] = {"key": signal_key, "count": 1}
                logger.info(self._separator(symbol))
                logger.info(
                    "[%s] 🎯 TREND %s | Entry=%s | SL=%s",
                    symbol,
                    signal["side"].value.upper(),
                    _fmt(signal["entry_price"]),
                    _fmt(signal["sl_price"]),
                )

            self._open_position(symbol, signal, ticker.last_price, StrategyType.TREND)
        else:
            # Signal disparu → reset compteur, log seulement si c'était long
            if symbol in self._signal_persistence and self._signal_persistence[symbol].get("key", "").startswith("TREND_"):
                old_info = self._signal_persistence.pop(symbol)
                if old_info["count"] > 5:
                    logger.info(
                        "[%s] 🎯 TREND signal OFF après %d cycles",
                        symbol, old_info["count"],
                    )

    # ── Recherche d'entrée RANGE ───────────────────────────────────────────────

    def _seek_range_entry(self, symbol: str, trend: TrendState, ticker) -> None:
        """Cherche un signal d'entrée Mean-Reversion Range."""
        # Vérifier qu'il n'y a pas déjà une position TREND sur ce symbole
        pos = self._positions.get(symbol)
        if pos and pos.status in (PositionStatus.OPEN, PositionStatus.ZERO_RISK):
            return  # Conflit : déjà une position ouverte

        rs = self._ranges.get(symbol)
        if rs is None:
            # Essayer de construire le range
            self._update_range(symbol, trend)
            rs = self._ranges.get(symbol)
            if rs is None:
                return

        signal = check_range_entry_signal(rs, ticker, config.RANGE_ENTRY_BUFFER_PERCENT)
        signal_key = f"RANGE_{signal['side'].value.upper()}" if signal else None

        if signal:
            # Compteur de persistance
            prev_info = self._signal_persistence.get(symbol, {})
            if prev_info.get("key") == signal_key:
                prev_info["count"] += 1
                logger.debug(
                    "[%s] RANGE SIGNAL %s persisting (×%d)",
                    symbol, signal["side"].value.upper(), prev_info["count"],
                )
            else:
                self._signal_persistence[symbol] = {"key": signal_key, "count": 1}
                logger.info(self._separator(symbol))
                logger.info(
                    "[%s] 🔄 RANGE %s | Entry=%s | SL=%s | TP=%s",
                    symbol,
                    signal["side"].value.upper(),
                    _fmt(signal["entry_price"]),
                    _fmt(signal["sl_price"]),
                    _fmt(signal["tp_price"]),
                )

            self._open_position(symbol, signal, ticker.last_price, StrategyType.RANGE)
        else:
            # Signal disparu → reset compteur, log seulement si c'était long
            if symbol in self._signal_persistence and self._signal_persistence[symbol].get("key", "").startswith("RANGE_"):
                old_info = self._signal_persistence.pop(symbol)
                if old_info["count"] > 5:
                    logger.info(
                        "[%s] 🔄 RANGE signal OFF après %d cycles",
                        symbol, old_info["count"],
                    )

    # ── Helper: mise à jour du range ───────────────────────────────────────────

    def _update_range(self, symbol: str, trend: TrendState) -> None:
        """Met à jour ou crée le RangeState pour un symbole NEUTRAL."""
        old_rs = self._ranges.get(symbol)
        rs = build_range_from_trend(trend, config.RANGE_WIDTH_MIN)
        if rs is not None:
            # Conserver le cooldown de l'ancien range si les bornes sont similaires
            if old_rs and old_rs.cooldown_until > 0:
                rs.cooldown_until = old_rs.cooldown_until
            self._ranges[symbol] = rs
        elif old_rs is None:
            # Pas de range possible et aucun existant
            pass

    # ── Logging intelligent (on-change) ─────────────────────────────────────────

    @staticmethod
    def _separator(symbol: str) -> str:
        """Séparateur visuel pour un symbole."""
        pad = max(0, 20 - len(symbol))
        half = pad // 2
        return f"{'─' * (8 + half)} {symbol} {'─' * (8 + pad - half)}"

    def _log_ignored_once(self, symbol: str, reason_key: str, message: str) -> None:
        """Log un signal ignoré une seule fois tant que la raison reste la même."""
        prev = self._prev_ignored.get(symbol)
        if prev == reason_key:
            return  # Déjà logué, silence
        self._prev_ignored[symbol] = reason_key
        logger.info("[%s] ⏭️ %s — état enregistré", symbol, message)

    def _build_current_state(self, symbol: str, price: float, trend: TrendState) -> dict:
        """Construit un snapshot de l'état courant pour comparaison."""
        rs = self._ranges.get(symbol)
        return {
            "direction": trend.direction.value,
            "entry_level": round(trend.entry_level or 0, 6),
            "sl_level": round(trend.sl_level or 0, 6),
            "neutral_reason": trend.neutral_reason,
            "range_key": f"{rs.range_high:.4f}-{rs.range_low:.4f}" if rs else None,
            "cooldown": is_in_cooldown(rs) if rs else False,
        }

    def _log_symbol_if_changed(
        self,
        symbol: str,
        price: float,
        trend: TrendState,
        position: Optional[Position],
    ) -> None:
        """Log uniquement si l'état du symbole a changé (direction, swings, range)."""
        current = self._build_current_state(symbol, price, trend)
        prev = self._prev_state.get(symbol)

        # Détecter si quelque chose a changé
        changed = prev is None or current != prev
        if not changed:
            return  # Silence total — rien n'a changé

        self._prev_state[symbol] = current

        # ── Log changement (toujours visible) ──
        logger.info(self._separator(symbol))

        if trend.direction in (TrendDirection.BULLISH, TrendDirection.BEARISH):
            high_type = (
                trend.last_high.swing_type.value
                if trend.last_high and trend.last_high.swing_type
                else "?"
            )
            low_type = (
                trend.last_low.swing_type.value
                if trend.last_low and trend.last_low.swing_type
                else "?"
            )

            # Niveaux clés
            entry_lvl = trend.entry_level
            sl_lvl = trend.sl_level
            if entry_lvl and sl_lvl:
                if trend.direction == TrendDirection.BULLISH:
                    entry_thresh = entry_lvl * (1 + config.ENTRY_BUFFER_PERCENT)
                    logger.info(
                        "[%s] 🟢 BULLISH %s/%s | 🎯 BUY ≥ %s (%s) | 🛑 SL %s",
                        symbol, high_type, low_type,
                        _fmt(entry_thresh), _pct_dist(price, entry_thresh),
                        _fmt(sl_lvl),
                    )
                else:
                    entry_thresh = entry_lvl * (1 - config.ENTRY_BUFFER_PERCENT)
                    logger.info(
                        "[%s] 🔴 BEARISH %s/%s | 🎯 SELL ≤ %s (%s) | 🛑 SL %s [SPOT: watch only]",
                        symbol, high_type, low_type,
                        _fmt(entry_thresh), _pct_dist(price, entry_thresh),
                        _fmt(sl_lvl),
                    )
            else:
                logger.info(
                    "[%s] %s | %s/%s confirmed",
                    symbol,
                    _TREND_EMOJI.get(trend.direction, trend.direction.value),
                    high_type, low_type,
                )

        elif trend.direction == TrendDirection.NEUTRAL:
            reason = trend.neutral_reason or "raison inconnue"
            rs = self._ranges.get(symbol)
            if rs and rs.is_valid:
                cooldown_str = _cooldown_str(rs)
                buy_zone = rs.range_low * (1 + config.RANGE_ENTRY_BUFFER_PERCENT)
                sell_zone = rs.range_high * (1 - config.RANGE_ENTRY_BUFFER_PERCENT)
                mid = (rs.range_high + rs.range_low) / 2
                logger.info(
                    "[%s] ⚪ NEUTRAL → RANGE | width=%.1f%% | H=%s L=%s%s",
                    symbol,
                    rs.range_width_pct * 100,
                    _fmt(rs.range_high),
                    _fmt(rs.range_low),
                    cooldown_str,
                )
                logger.info(
                    "[%s]    💰 Price=%s | BUY zone ≤ %s (%s) | SELL zone ≥ %s (%s) | mid=%s",
                    symbol,
                    _fmt(price),
                    _fmt(buy_zone), _pct_dist(price, buy_zone),
                    _fmt(sell_zone), _pct_dist(price, sell_zone),
                    _fmt(mid),
                )
            else:
                logger.info(
                    "[%s] ⚪ NEUTRAL | %s — pas de range",
                    symbol, reason,
                )

    # ── Actions ────────────────────────────────────────────────────────────────

    def _open_position(
        self,
        symbol: str,
        signal: dict,
        current_price: float,
        strategy: StrategyType = StrategyType.TREND,
    ) -> None:
        """Ouvre une position (place un ordre limit)."""
        strat_label = "📊 TREND" if strategy == StrategyType.TREND else "🔄 RANGE"

        self._cycle_signals += 1

        # ── GUARD: Ne jamais ouvrir deux fois sur le même symbole ──
        existing = self._positions.get(symbol)
        if existing and existing.status in (PositionStatus.OPEN, PositionStatus.ZERO_RISK):
            logger.debug("[%s] Position déjà ouverte, skip %s", symbol, strat_label)
            return

        # Vérifier le nombre max de positions simultanées
        open_positions = sum(
            1 for p in self._positions.values()
            if p.status in (PositionStatus.OPEN, PositionStatus.ZERO_RISK)
        )
        if open_positions >= config.MAX_SIMULTANEOUS_POSITIONS:
            self._log_ignored_once(symbol, "MAX_POS",
                f"Signal {strat_label} ignoré — {open_positions}/{config.MAX_SIMULTANEOUS_POSITIONS} positions")
            return

        # Calculer la taille
        balances = self._data.get_balances()
        fiat_balance, fiat_currency = get_fiat_balance(balances)

        # ── Vérifier le risque total ──
        open_pos_list = [
            p for p in self._positions.values()
            if p.status in (PositionStatus.OPEN, PositionStatus.ZERO_RISK)
        ]
        current_risk = check_total_risk_exposure(
            open_pos_list, fiat_balance, config.MAX_TOTAL_RISK_PERCENT,
        )
        if current_risk >= config.MAX_TOTAL_RISK_PERCENT:
            self._log_ignored_once(symbol, "MAX_RISK",
                f"Signal {strat_label} ignoré — risque {current_risk*100:.1f}% ≥ plafond {config.MAX_TOTAL_RISK_PERCENT*100:.0f}%")
            return

        # ── Risque adapté à la stratégie ──
        risk_pct = (
            config.RISK_PERCENT_TREND
            if strategy == StrategyType.TREND
            else config.RISK_PERCENT_RANGE
        )

        # ── Vérification spot : on ne peut SELL que si on possède l'actif ──
        if signal["side"] == OrderSide.SELL:
            base_currency = symbol.split("-")[0]  # ex: "BTC" de "BTC-USD"
            base_balance = next(
                (b for b in balances if b.currency == base_currency), None
            )
            available = base_balance.available if base_balance else 0.0
            if available <= 0:
                self._log_ignored_once(symbol, "NO_SHORT",
                    f"SELL {strat_label} ignoré — pas de {base_currency} (spot, short impossible)")
                return
            logger.debug(
                "[%s] Solde %s disponible: %.8f",
                symbol, base_currency, available,
            )

        # ── Vérification spot BUY : besoin de la devise de cotation ──
        if signal["side"] == OrderSide.BUY:
            quote_currency = symbol.split("-")[1]  # ex: "USD" de "BTC-USD"
            quote_balance = next(
                (b for b in balances if b.currency == quote_currency), None
            )
            available_quote = quote_balance.available if quote_balance else 0.0
            if available_quote <= 0:
                self._log_ignored_once(symbol, "NO_FIAT",
                    f"BUY {strat_label} ignoré — pas de {quote_currency} disponible")
                return
            logger.debug(
                "[%s] Solde %s disponible: %.2f",
                symbol, quote_currency, available_quote,
            )
            # Sizing basé sur le solde réel de la devise de cotation (pas l'EUR converti)
            fiat_balance = available_quote

        size = calculate_position_size(
            account_balance=fiat_balance,
            risk_percent=risk_pct,
            entry_price=signal["entry_price"],
            sl_price=signal["sl_price"],
            max_position_percent=config.MAX_POSITION_PERCENT,
        )
        if size <= 0:
            return

        # Construire l'ordre
        order = build_entry_order(
            symbol=symbol,
            side=signal["side"],
            entry_price=signal["entry_price"],
            position_size=size,
        )

        # Exécuter (ou simuler)
        venue_order_id = "dry-run"
        fill_type = "dry-run"
        if not self.dry_run:
            # 💰 Maker-First : limit passif (0%) → attente → fallback taker (0.09%)
            try:
                result = self._client.place_maker_first_order(
                    order, wait_seconds=config.MAKER_WAIT_SECONDS,
                )
                venue_order_id = result.get("venue_order_id", "unknown")
                fill_type = result.get("fill_type", "unknown")
                logger.info(
                    "[%s] 💰 Ordre %s exécuté en %s (fee: %s)",
                    symbol, strat_label, fill_type,
                    "0%%" if fill_type == "maker" else "0.09%%",
                )
            except Exception as e:
                logger.error("[%s] Échec du placement d'ordre %s: %s", symbol, strat_label, e)
                self._telegram.notify_error(f"Ordre {symbol} ({strat_label}) échoué: {e}")
                return
        else:
            logger.info("[DRY-RUN] Ordre %s simulé: %s", strat_label, order.to_api_payload())

        # Enregistrer la position
        tp_price = signal.get("tp_price") if strategy == StrategyType.RANGE else None
        position = Position(
            symbol=symbol,
            side=signal["side"],
            entry_price=signal["entry_price"],
            sl_price=signal["sl_price"],
            size=size,
            venue_order_id=venue_order_id,
            status=PositionStatus.OPEN,
            strategy=strategy,
            tp_price=tp_price,
        )
        self._positions[symbol] = position
        self._save_state()
        self._cycle_executed += 1
        # Effacer l'état "ignoré" pour ce symbole (il a été traité)
        self._prev_ignored.pop(symbol, None)

        # Notifier
        risk_amount = fiat_balance * risk_pct
        self._telegram.notify_entry(
            symbol=symbol,
            side=signal["side"],
            entry_price=signal["entry_price"],
            sl_price=signal["sl_price"],
            size=size,
            risk_percent=risk_pct,
            risk_amount=risk_amount,
            strategy=strategy,
            tp_price=tp_price,
        )

        # 🔥 Firebase — log ouverture
        try:
            open_pos_list = [
                p for p in self._positions.values()
                if p.status in (PositionStatus.OPEN, PositionStatus.ZERO_RISK)
            ]
            portfolio_risk = check_total_risk_exposure(
                open_pos_list, fiat_balance, 1.0,
            ) if fiat_balance > 0 else 0.0
            # Equity réelle = USD + cryptos valorisées
            try:
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
                current_equity = get_total_equity(balances, crypto_tickers)
            except Exception:
                current_equity = fiat_balance  # fallback
            fb_id = log_trade_opened(
                position=position,
                fill_type=fill_type,
                maker_wait_seconds=config.MAKER_WAIT_SECONDS,
                risk_pct=risk_pct,
                risk_amount_usd=risk_amount,
                fiat_balance=fiat_balance,
                current_equity=current_equity,
                portfolio_risk_before=portfolio_risk,
            )
            if fb_id:
                position.firebase_trade_id = fb_id
                self._save_state()
        except Exception as e:
            logger.warning("🔥 Firebase log_trade_opened échoué: %s", e)

    def _close_position(
        self,
        symbol: str,
        exit_price: float,
        reason: str,
        forced: bool = False,
        is_range_tp: bool = False,
    ) -> None:
        """Ferme une position (place un ordre limit opposé)."""
        position = self._positions.get(symbol)
        if position is None:
            return

        # ── Vérifier le cooldown d'échec de clôture ──────────────────
        fail_info = self._close_failures.get(symbol)
        if fail_info:
            # Erreur permanente → ne plus réessayer
            if fail_info.get("permanent"):
                logger.debug(
                    "[%s] 🚫 Close bloquée définitivement — intervention manuelle requise",
                    symbol,
                )
                return
            # Cooldown temporaire
            if time.time() < fail_info["next_retry"]:
                remaining = int(fail_info["next_retry"] - time.time())
                logger.debug(
                    "[%s] ⏳ Close en cooldown (échec #%d) — retry dans %ds",
                    symbol, fail_info["count"], remaining,
                )
                return
            # Cooldown expiré → on retente la clôture
            logger.info(
                "[%s] 🔄 Cooldown expiré (échec #%d) — nouvelle tentative de clôture",
                symbol, fail_info["count"],
            )

        # ── Ajuster la taille au solde réel (évite "Insufficient balance") ──
        exit_size = position.size
        if position.side == OrderSide.BUY:
            # On va SELL le crypto → vérifier le solde réel disponible
            try:
                base_currency = symbol.split("-")[0]
                balances = self._data.get_balances()
                base_bal = next(
                    (b for b in balances if b.currency == base_currency), None
                )
                real_available = base_bal.available if base_bal else 0.0

                if real_available <= 0:
                    # ── Position fantôme : solde réel = 0 → purger ──
                    logger.warning(
                        "[%s] 👻 Position FANTÔME détectée — solde %s = 0 sur l'exchange. "
                        "Suppression de la position locale (entry=%.8f, size=%.8f)",
                        symbol, base_currency, position.entry_price, position.size,
                    )
                    self._telegram.notify_error(
                        f"👻 Position fantôme supprimée : {symbol}\n"
                        f"Entry: {position.entry_price} | Size: {position.size}\n"
                        f"Solde réel {base_currency} = 0 → position purgée"
                    )
                    del self._positions[symbol]
                    if symbol in self._close_failures:
                        del self._close_failures[symbol]
                    self._save_state()
                    return

                if real_available < position.size:
                    old_size = position.size
                    exit_size = real_available
                    logger.info(
                        "[%s] 📐 Ajustement taille sortie : %.8f → %.8f (solde réel %s)",
                        symbol, old_size, exit_size, base_currency,
                    )
            except Exception as e:
                logger.warning("[%s] ⚠️ Impossible de vérifier le solde pour ajustement: %s", symbol, e)

        # Temporairement remplacer la taille pour build_exit_order
        original_size = position.size
        position.size = exit_size
        exit_order = build_exit_order(position, exit_price)
        position.size = original_size  # restaurer pour les calculs PnL

        fill_type = "dry-run"
        actual_exit_price = exit_price  # fallback = prix demandé

        if not self.dry_run:
            # 💰 Maker-First pour la clôture aussi
            try:
                result = self._client.place_maker_first_order(
                    exit_order, wait_seconds=config.MAKER_WAIT_SECONDS,
                )
                fill_type = result.get("fill_type", "unknown")
                actual_exit_price = result.get("actual_price", exit_price)
            except Exception as e:
                error_str = str(e)
                logger.error("[%s] Échec de la clôture: %s", symbol, e)

                # ── Détecter les erreurs permanentes (pas de retry utile) ──
                PERMANENT_ERRORS = ["INACTIVE", "DELISTED", "SUSPENDED", "not supported", "No CURRENT pocket"]
                is_permanent = any(kw.lower() in error_str.lower() for kw in PERMANENT_ERRORS)

                if is_permanent:
                    logger.critical(
                        "[%s] 🚫 ERREUR PERMANENTE — arrêt des retries. Intervention manuelle requise: %s",
                        symbol, error_str[:300],
                    )
                    # Marquer comme permanent dans close_failures
                    fail_info = self._close_failures.get(symbol, {"count": 0})
                    fail_info["count"] += 1
                    fail_info["last_error"] = error_str
                    fail_info["permanent"] = True
                    fail_info["next_retry"] = float("inf")  # Ne plus réessayer
                    self._close_failures[symbol] = fail_info

                    # Alerte Telegram immédiate
                    self._telegram.notify_error(
                        f"🚫 ERREUR PERMANENTE {symbol}\n"
                        f"Clôture impossible via API.\n"
                        f"Action: vendre manuellement sur Revolut X\n"
                        f"Erreur: {error_str[:300]}"
                    )

                    # 🔥 Firebase — logger l'erreur permanente
                    try:
                        fb_log_close_failure(
                            symbol=symbol,
                            attempt=fail_info["count"],
                            error=f"PERMANENT: {error_str[:300]}",
                            next_retry_seconds=-1,
                            trade_id=position.firebase_trade_id,
                        )
                    except Exception:
                        pass
                    return

                # ── Backoff exponentiel pour éviter le spam API ──
                fail_info = self._close_failures.get(symbol, {"count": 0})
                fail_info["count"] += 1
                fail_info["last_error"] = error_str

                # Paliers de cooldown : 60s, 120s, 300s, 600s, 1800s (max 30min)
                backoff_tiers = [60, 120, 300, 600, 1800]
                tier_idx = min(fail_info["count"] - 1, len(backoff_tiers) - 1)
                cooldown_s = backoff_tiers[tier_idx]
                fail_info["next_retry"] = time.time() + cooldown_s
                self._close_failures[symbol] = fail_info

                logger.warning(
                    "[%s] 🔁 Échec clôture #%d — prochain retry dans %ds",
                    symbol, fail_info["count"], cooldown_s,
                )

                # 🔥 Firebase — logger chaque échec
                try:
                    fb_log_close_failure(
                        symbol=symbol,
                        attempt=fail_info["count"],
                        error=error_str,
                        next_retry_seconds=cooldown_s,
                        trade_id=position.firebase_trade_id,
                    )
                except Exception:
                    pass

                # Alerte Telegram uniquement au 1er échec et ensuite toutes les 5 tentatives
                if fail_info["count"] == 1 or fail_info["count"] % 5 == 0:
                    self._telegram.notify_error(
                        f"⚠️ Clôture {symbol} échouée (×{fail_info['count']})\n"
                        f"Retry dans {cooldown_s}s\n"
                        f"Erreur: {error_str[:200]}"
                    )
                return
        else:
            logger.info(
                "[DRY-RUN] Clôture simulée: %s", exit_order.to_api_payload()
            )

        # ── Clôture réussie → reset du compteur d'échecs ──
        if symbol in self._close_failures:
            logger.info(
                "[%s] ✅ Clôture réussie après %d échec(s) précédent(s)",
                symbol, self._close_failures[symbol]["count"],
            )
            del self._close_failures[symbol]
            # 🔥 Firebase — clear le flag close_blocked
            if position.firebase_trade_id:
                try:
                    fb_clear_close_failure(position.firebase_trade_id)
                except Exception:
                    pass

        # ── LOG STRUCTURÉ POST-CLÔTURE ──────────────────────────────────
        notional_entry = position.size * position.entry_price
        notional_exit = position.size * actual_exit_price

        # PnL brut (avant fees)
        if position.side == OrderSide.BUY:
            pnl_gross = (actual_exit_price - position.entry_price) * position.size
        else:
            pnl_gross = (position.entry_price - actual_exit_price) * position.size
        pnl_pct = pnl_gross / notional_entry if notional_entry > 0 else 0.0

        # Fees estimées
        fee_entry_pct = 0.0 if position.venue_order_id == "recovered" else 0.0  # maker à l'entrée
        fee_exit_pct = 0.0 if fill_type == "maker" else 0.0009  # taker = 0.09%
        fee_entry_usd = notional_entry * fee_entry_pct
        fee_exit_usd = notional_exit * fee_exit_pct
        fees_total_usd = fee_entry_usd + fee_exit_usd

        # PnL net (après fees)
        pnl_net = pnl_gross - fees_total_usd
        pnl_net_pct = pnl_net / notional_entry if notional_entry > 0 else 0.0

        # Slippage (prix demandé vs prix réel d'exécution)
        slippage_pct = abs(actual_exit_price - exit_price) / exit_price if exit_price > 0 else 0.0

        # Equity après clôture (USD + toutes les cryptos valorisées)
        equity_after = 0.0
        if not self.dry_run:
            try:
                balances = self._data.get_balances()
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
                equity_after = get_total_equity(balances, crypto_tickers)
            except Exception:
                # Fallback sur le cash USD seul
                try:
                    fiat_balance, _ = get_fiat_balance(balances)
                    equity_after = fiat_balance
                except Exception:
                    pass

        # ── LOG PRO ──
        strat_label = position.strategy.value if position.strategy else "?"
        exit_label = reason.upper().replace(" ", "_")
        maker_info = f"MAKER (0%)" if fill_type == "maker" else f"TAKER (0.09%)"
        if fill_type == "partial_maker":
            maker_info = "PARTIAL_MAKER"
        pnl_emoji = "🟢" if pnl_net >= 0 else "🔴"

        logger.info(self._separator(symbol))
        logger.info(
            "[%s] %s CLOSE | STRAT=%s | EXIT=%s",
            symbol, pnl_emoji, strat_label, exit_label,
        )
        logger.info(
            "[%s]   Entry=%.6f | Exit=%.6f (demandé=%.6f)",
            symbol, position.entry_price, actual_exit_price, exit_price,
        )
        logger.info(
            "[%s]   Size=%.4f | Notional=$%.2f",
            symbol, position.size, notional_exit,
        )
        logger.info(
            "[%s]   Execution=%s | Slippage=%.4f%%",
            symbol, maker_info, slippage_pct * 100,
        )
        logger.info(
            "[%s]   Fees: entry=$%.4f + exit=$%.4f = $%.4f total",
            symbol, fee_entry_usd, fee_exit_usd, fees_total_usd,
        )
        logger.info(
            "[%s]   PnL GROSS=$%+.4f (%+.2f%%) | NET=$%+.4f (%+.2f%%)",
            symbol, pnl_gross, pnl_pct * 100, pnl_net, pnl_net_pct * 100,
        )
        if equity_after > 0:
            logger.info(
                "[%s]   Equity=$%.2f",
                symbol, equity_after,
            )

        position.status = PositionStatus.CLOSED
        position.pnl = pnl_net
        self._save_state()

        # Notification adaptée
        if forced:
            self._telegram.notify_forced_exit(position, actual_exit_price, reason)
        elif is_range_tp:
            self._telegram.notify_range_tp_hit(position, actual_exit_price)
        else:
            self._telegram.notify_sl_hit(position, actual_exit_price)

        # 🔥 Firebase — log clôture
        if position.firebase_trade_id:
            try:
                log_trade_closed(
                    trade_id=position.firebase_trade_id,
                    position=position,
                    exit_price=actual_exit_price,
                    reason=reason,
                    fill_type=fill_type,
                    equity_after=equity_after,
                )
            except Exception as e:
                logger.warning("🔥 Firebase log_trade_closed échoué: %s", e)

    def _update_trend(self, symbol: str) -> None:
        """Recalcule la tendance à partir des bougies H4."""
        candles = self._data.get_h4_candles(symbol)
        if len(candles) < (2 * config.SWING_LOOKBACK + 1):
            logger.warning(
                "[%s] Pas assez de bougies (%d) pour l'analyse",
                symbol,
                len(candles),
            )
            return

        swings = detect_swings(candles, lookback=config.SWING_LOOKBACK)
        swing_highs = [s for s in swings if s.level == SwingLevel.HIGH]
        swing_lows = [s for s in swings if s.level == SwingLevel.LOW]

        if len(swings) < 4:
            logger.warning(
                "[%s] Pas assez de swings (%d) pour classer la tendance",
                symbol,
                len(swings),
            )
            self._trends[symbol] = TrendState(
                symbol=symbol, direction=TrendDirection.NEUTRAL
            )
            return

        old_trend = self._trends.get(symbol)
        old_direction = old_trend.direction if old_trend else TrendDirection.NEUTRAL

        new_trend = determine_trend(swings, symbol)
        self._trends[symbol] = new_trend

        # ── Log H4 compact : 1 ligne par paire ──
        trend_label = _TREND_EMOJI.get(new_trend.direction, new_trend.direction.value)

        # Swings récents (DEBUG)
        recent = swings[-4:] if len(swings) >= 4 else swings
        swing_chain = " → ".join(
            f"{s.swing_type.value if s.swing_type else s.level.value} {_fmt(s.price)}"
            for s in recent
        )
        logger.debug(
            "[%s] H4 | %d bougies → %d swings | %s",
            symbol, len(candles), len(swings), swing_chain,
        )

        # ── Résumé condensé (1 ligne INFO) ──
        if new_trend.direction in (TrendDirection.BULLISH, TrendDirection.BEARISH):
            high_type = (
                new_trend.last_high.swing_type.value
                if new_trend.last_high and new_trend.last_high.swing_type
                else "?"
            )
            low_type = (
                new_trend.last_low.swing_type.value
                if new_trend.last_low and new_trend.last_low.swing_type
                else "?"
            )
            if new_trend.entry_level and new_trend.sl_level:
                if new_trend.direction == TrendDirection.BULLISH:
                    thresh = new_trend.entry_level * (1 + config.ENTRY_BUFFER_PERCENT)
                    logger.info(
                        "[%s] 🟢 BULLISH %s/%s | BUY ≥ %s | SL %s",
                        symbol, high_type, low_type,
                        _fmt(thresh), _fmt(new_trend.sl_level),
                    )
                else:
                    thresh = new_trend.entry_level * (1 - config.ENTRY_BUFFER_PERCENT)
                    logger.info(
                        "[%s] 🔴 BEARISH %s/%s | SELL ≤ %s | SL %s [SPOT: watch only]",
                        symbol, high_type, low_type,
                        _fmt(thresh), _fmt(new_trend.sl_level),
                    )
            else:
                logger.info("[%s] %s | %s/%s", symbol, trend_label, high_type, low_type)
        else:
            reason = new_trend.neutral_reason or "inconnue"
            logger.info("[%s] ⚪ NEUTRAL | %s", symbol, reason)

        # ── Changement de tendance (WARNING = toujours visible) ──
        if new_trend.direction != old_direction:
            logger.warning(
                "[%s] 🔄 CHANGEMENT: %s → %s",
                symbol, old_direction.value, new_trend.direction.value,
            )
            self._telegram.notify_trend_change(new_trend, old_direction)
            # 🔥 Firebase
            try:
                fb_log_trend_change(symbol, old_direction, new_trend.direction)
            except Exception:
                pass

        # ── Mise à jour du range pour Mean-Reversion ──
        if new_trend.direction == TrendDirection.NEUTRAL:
            self._update_range(symbol, new_trend)
            rs = self._ranges.get(symbol)
            if rs and rs.is_valid:
                cooldown_str = _cooldown_str(rs)
                buy_zone = rs.range_low * (1 + config.RANGE_ENTRY_BUFFER_PERCENT)
                sell_zone = rs.range_high * (1 - config.RANGE_ENTRY_BUFFER_PERCENT)
                mid = (rs.range_high + rs.range_low) / 2
                logger.info(
                    "[%s]    → RANGE width=%.1f%% | H=%s L=%s%s",
                    symbol,
                    rs.range_width_pct * 100,
                    _fmt(rs.range_high),
                    _fmt(rs.range_low),
                    cooldown_str,
                )
                last_close = candles[-1].close if candles else 0
                logger.info(
                    "[%s]    💰 Price≈%s | BUY zone ≤ %s | SELL zone ≥ %s | mid=%s",
                    symbol,
                    _fmt(last_close),
                    _fmt(buy_zone),
                    _fmt(sell_zone),
                    _fmt(mid),
                )
        elif symbol in self._ranges:
            del self._ranges[symbol]

    # ── Firebase maintenance ─────────────────────────────────────────────────

    def _firebase_daily_cleanup(self) -> None:
        """Supprime les vieux events Firebase (1 fois par jour max)."""
        from datetime import datetime, timezone
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        if today == self._last_cleanup_date:
            return  # Déjà fait aujourd'hui
        self._last_cleanup_date = today
        try:
            fb_cleanup_events()
        except Exception as e:
            logger.debug("🧹 Firebase cleanup échoué: %s", e)

    # ── Shutdown ───────────────────────────────────────────────────────────────

    def _shutdown(self) -> None:
        """Nettoyage à l'arrêt du bot."""
        logger.info("🛑 Arrêt de TradeX...")
        self._save_state()
        logger.info("💾 État final sauvegardé")
        self._client.close()
        self._telegram.close()
        logger.info("TradeX arrêté proprement")


# ── Point d'entrée ─────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(description="TradeX – Bot de trading crypto")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Mode simulation (log les ordres sans les exécuter)",
    )
    args = parser.parse_args()

    bot = TradeXBot(dry_run=args.dry_run)

    # Arrêt propre sur SIGTERM (utile sur VPS)
    signal.signal(signal.SIGTERM, lambda *_: bot.stop())

    bot.run()


if __name__ == "__main__":
    main()
