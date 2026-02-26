"""
Moteur de backtest : simule la boucle du bot TradeX bar par bar.

Réutilise directement les modules de src/core/ pour la logique de trading.
Seule la gestion du temps (cooldowns, etc.) est adaptée pour le replay.
"""

from __future__ import annotations

import logging
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

from src.core.models import (
    Candle,
    OrderSide,
    Position,
    PositionStatus,
    RangeState,
    StrategyType,
    TrendDirection,
    TrendState,
)
from src.core.swing_detector import detect_swings
from src.core.trend_engine import determine_trend
from src.core.strategy_mean_rev import build_range_from_trend
from src.core.risk_manager import (
    calculate_position_size,
    should_apply_zero_risk,
    calculate_zero_risk_sl,
    update_trailing_stop,
)

logger = logging.getLogger(__name__)


# ── Structures de résultat ─────────────────────────────────────────────────────


@dataclass
class Trade:
    """Un trade terminé (ouverture → fermeture)."""
    symbol: str
    strategy: StrategyType
    side: OrderSide
    entry_price: float
    exit_price: float
    size: float
    entry_time: int          # timestamp ms
    exit_time: int           # timestamp ms
    pnl_usd: float
    pnl_pct: float
    exit_reason: str         # SL, TRAILING_SL, RANGE_SL, RANGE_TP, END


@dataclass
class EquityPoint:
    timestamp: int
    equity: float


@dataclass
class BacktestResult:
    trades: list[Trade]
    equity_curve: list[EquityPoint]
    initial_balance: float
    final_equity: float
    start_date: datetime
    end_date: datetime
    pairs: list[str]


@dataclass
class BacktestConfig:
    """Paramètres du backtest — miroir du .env de prod."""
    initial_balance: float = 1000.0
    # Trend
    risk_percent_trend: float = 0.03
    entry_buffer_pct: float = 0.002
    sl_buffer_pct: float = 0.003
    zero_risk_trigger_pct: float = 0.02
    zero_risk_lock_pct: float = 0.005
    trailing_stop_pct: float = 0.02
    max_position_pct: float = 0.20
    max_simultaneous_positions: int = 3
    swing_lookback: int = 3
    # Range
    risk_percent_range: float = 0.02
    range_width_min: float = 0.02
    range_entry_buffer_pct: float = 0.002
    range_sl_buffer_pct: float = 0.003
    range_cooldown_bars: int = 3
    # Global
    max_total_risk_pct: float = 0.06
    fee_pct: float = 0.00075        # 0.075 % par côté
    slippage_pct: float = 0.001     # 0.1 % slippage adverse à l'entrée
    compound: bool = False          # False = sizing fixe sur capital initial
    candle_window: int = 100        # fenêtre glissante pour swings

    # ── Filtre EMA200 D1 ──
    use_ema_filter: bool = False     # Activer le filtre BTC EMA200 D1
    ema_period: int = 200            # Période de l'EMA
    # Mode DEFENSIVE (BTC sous EMA200) : risque réduit
    risk_percent_trend_defensive: float = 0.015   # ~moitié du NORMAL
    risk_percent_range_defensive: float = 0.01    # ~moitié du NORMAL

    # ── TREND pullback (D1→H4) ──
    use_d1_pullback: bool = False    # True = pullback D1, False = breakout H4 (ancien)
    pullback_zone_pct: float = 0.01  # 1% — zone autour de HL pour entrer en pullback
    d1_candle_window: int = 60       # fenêtre glissante pour swings D1
    d1_swing_lookback: int = 3       # lookback pour swings D1

    # ── Activation des stratégies ──
    enable_trend: bool = True
    enable_range: bool = True
    allow_short: bool = False        # True = autoriser SELL (short) pour TREND


# ── Moteur ─────────────────────────────────────────────────────────────────────


class BacktestEngine:
    """Simule TradeX bar-par-bar sur données historiques."""

    def __init__(
        self,
        candles_by_symbol: dict[str, list[Candle]],
        config: BacktestConfig,
        btc_d1_candles: Optional[list[Candle]] = None,
        d1_candles_by_symbol: Optional[dict[str, list[Candle]]] = None,
    ) -> None:
        self.candles = candles_by_symbol
        self.cfg = config
        self.pairs = list(candles_by_symbol.keys())

        # État financier
        self.cash: float = config.initial_balance
        self.positions: list[Position] = []
        self.closed_trades: list[Trade] = []
        self.equity_curve: list[EquityPoint] = []

        # État par paire (H4 — utilisé par RANGE)
        self.trends: dict[str, Optional[TrendState]] = {p: None for p in self.pairs}
        self.ranges: dict[str, Optional[RangeState]] = {p: None for p in self.pairs}
        self.cooldown_until: dict[str, int] = {p: 0 for p in self.pairs}
        self.last_close: dict[str, float] = {}

        # État D1 par paire (utilisé par TREND pullback)
        self.d1_trends: dict[str, Optional[TrendState]] = {p: None for p in self.pairs}
        self._last_d1_ts_seen: dict[str, int] = {p: 0 for p in self.pairs}

        # Signaux en attente (calculés à bar N, vérifiés à bar N+1)
        self._pend_trend: dict[str, Optional[dict]] = {p: None for p in self.pairs}
        self._pend_range: dict[str, Optional[dict]] = {p: None for p in self.pairs}

        # Tracking entry time
        self._entry_ts: dict[str, int] = {}   # venue_order_id → ts

        # ── D1 candles (structure trend + EMA) ─────────────────────────────
        self.d1_candles: dict[str, list[Candle]] = d1_candles_by_symbol or {}

        # ── Filtre EMA200 D1 ───────────────────────────────────────────────
        self._ema_mode: dict[int, bool] = {}  # ts_d1 → is_normal
        self._current_mode_normal: bool = True
        if config.use_ema_filter:
            # Préférer BTC D1 depuis d1_candles_by_symbol, sinon fallback
            btc_d1 = None
            if d1_candles_by_symbol and "BTC-USD" in d1_candles_by_symbol:
                btc_d1 = d1_candles_by_symbol["BTC-USD"]
            elif btc_d1_candles:
                btc_d1 = btc_d1_candles
            if btc_d1:
                self._build_ema_lookup(btc_d1)

        # Index rapide H4 : {(symbol, ts): Candle}
        self._idx: dict[tuple[str, int], Candle] = {}
        for sym, clist in candles_by_symbol.items():
            for c in clist:
                self._idx[(sym, c.timestamp)] = c

    # ── EMA200 D1 ──────────────────────────────────────────────────────────

    def _build_ema_lookup(self, d1_candles: list[Candle]) -> None:
        """Pré-calcule l'EMA200 sur BTC D1 et stocke le mode par jour."""
        period = self.cfg.ema_period
        if len(d1_candles) < period:
            logger.warning(
                "⚠️ Seulement %d bougies D1 (< EMA%d) — filtre désactivé",
                len(d1_candles), period,
            )
            return

        closes = [c.close for c in d1_candles]

        # SMA initiale sur les `period` premières bougies
        sma = sum(closes[:period]) / period
        k = 2.0 / (period + 1)

        ema = sma
        for i in range(period, len(closes)):
            ema = closes[i] * k + ema * (1 - k)
            ts = d1_candles[i].timestamp
            self._ema_mode[ts] = closes[i] > ema

        # Convertir en liste triée pour lookup rapide par bisect
        self._ema_ts_sorted = sorted(self._ema_mode.keys())
        n_normal = sum(1 for v in self._ema_mode.values() if v)
        n_def = len(self._ema_mode) - n_normal
        logger.info(
            "📊 EMA%d D1 : %d jours NORMAL, %d jours DEFENSIVE",
            period, n_normal, n_def,
        )

    def _update_market_mode(self, ts_h4: int) -> None:
        """Met à jour le mode NORMAL/DEFENSIVE basé sur la dernière EMA D1.

        Pour chaque bougie H4, on cherche la dernière bougie D1
        clôturée AVANT ce timestamp (pas de lookahead).
        """
        if not self._ema_mode:
            return

        import bisect
        idx = bisect.bisect_right(self._ema_ts_sorted, ts_h4) - 1
        if idx >= 0:
            last_d1_ts = self._ema_ts_sorted[idx]
            self._current_mode_normal = self._ema_mode[last_d1_ts]

    # ── Point d'entrée ─────────────────────────────────────────────────────

    def run(self) -> BacktestResult:
        # Couper les logs verbeux des modules core pendant le replay
        for mod in (
            "src.core.swing_detector",
            "src.core.trend_engine",
            "src.core.strategy_mean_rev",
            "src.core.risk_manager",
        ):
            logging.getLogger(mod).setLevel(logging.WARNING)

        timeline = self._build_timeline()
        total = len(timeline)
        logger.info(
            "📊 Backtest : %d barres H4, %d paires, capital $%.0f",
            total, len(self.pairs), self.cfg.initial_balance,
        )

        for i, ts in enumerate(timeline):
            # Mettre à jour les last_close
            for sym in self.pairs:
                c = self._idx.get((sym, ts))
                if c:
                    self.last_close[sym] = c.close

            # ── Mettre à jour le mode macro (EMA200 D1) ──
            if self.cfg.use_ema_filter:
                self._update_market_mode(ts)

            # ORDRE CRITIQUE pour éviter le lookahead :
            # 1) Gérer les positions existantes (SL / trailing / TP)
            # 2) Vérifier les entrées sur signaux générés à la bougie PRÉCÉDENTE
            # 3) Analyser la bougie courante → signaux pour la bougie SUIVANTE
            self._manage_positions(ts)
            self._check_pending_entries(ts)
            self._update_h4_analysis(ts)  # RANGE signals (H4)
            if self.cfg.use_d1_pullback:
                self._update_d1_analysis(ts)  # TREND pullback signals (D1)
            self._record_equity(ts)

            # Progression
            if (i + 1) % 500 == 0 or i == total - 1:
                eq = self.equity_curve[-1].equity
                print(
                    f"\r   ⏳ {i+1}/{total} ({100*(i+1)/total:.0f}%) "
                    f"| Equity: ${eq:,.2f} | Trades: {len(self.closed_trades)}",
                    end="", flush=True,
                )
        print()  # newline

        # Fermer les positions ouvertes au dernier close
        self._close_remaining(timeline[-1])
        if self.positions:
            # recalculer equity après fermeture
            self._record_equity(timeline[-1])

        final_eq = self.equity_curve[-1].equity if self.equity_curve else self.cash
        return BacktestResult(
            trades=self.closed_trades,
            equity_curve=self.equity_curve,
            initial_balance=self.cfg.initial_balance,
            final_equity=final_eq,
            start_date=datetime.fromtimestamp(timeline[0] / 1000, tz=timezone.utc),
            end_date=datetime.fromtimestamp(timeline[-1] / 1000, tz=timezone.utc),
            pairs=self.pairs,
        )

    # ── Timeline ───────────────────────────────────────────────────────────

    def _build_timeline(self) -> list[int]:
        ts_set: set[int] = set()
        for clist in self.candles.values():
            for c in clist:
                ts_set.add(c.timestamp)
        return sorted(ts_set)

    def _visible(self, symbol: str, up_to_ts: int) -> list[Candle]:
        """Fenêtre glissante de bougies pour le calcul des swings."""
        clist = self.candles[symbol]
        vis = [c for c in clist if c.timestamp <= up_to_ts]
        return vis[-self.cfg.candle_window:]

    # ── Gestion des positions ouvertes ─────────────────────────────────────

    def _eff_sl(self, pos: Position) -> float:
        if pos.is_zero_risk_applied and pos.zero_risk_sl is not None:
            return pos.zero_risk_sl
        return pos.sl_price

    def _manage_positions(self, ts: int) -> None:
        to_close: list[tuple[Position, float, str]] = []

        for pos in self.positions:
            c = self._idx.get((pos.symbol, ts))
            if c is None:
                continue

            if pos.strategy == StrategyType.TREND:
                sl = self._eff_sl(pos)
                hit_sl = (
                    (pos.side == OrderSide.BUY and c.low <= sl)
                    or (pos.side == OrderSide.SELL and c.high >= sl)
                )

                # ── FIX #1 : SL prioritaire (worst-case) ──
                # Si le SL est touché dans cette bougie, on sort au SL.
                # Même si le high a aussi touché un trailing favorable.
                if hit_sl:
                    reason = "TRAILING_SL" if pos.is_zero_risk_applied else "SL"
                    to_close.append((pos, sl, reason))
                    continue

                # ── FIX #2 : Zero-risk & trailing sur CLOSE, pas HIGH ──
                # On ne connaît le close qu'en fin de bougie → pas de lookahead.
                if not pos.is_zero_risk_applied:
                    if should_apply_zero_risk(pos, c.close, self.cfg.zero_risk_trigger_pct):
                        pos.zero_risk_sl = calculate_zero_risk_sl(pos, self.cfg.zero_risk_lock_pct)
                        pos.is_zero_risk_applied = True
                        pos.status = PositionStatus.ZERO_RISK

                if pos.is_zero_risk_applied:
                    update_trailing_stop(pos, c.close, self.cfg.trailing_stop_pct)

            elif pos.strategy == StrategyType.RANGE:
                sl_eff = pos.sl_price * (1 - self.cfg.range_sl_buffer_pct)
                hit_sl = (pos.side == OrderSide.BUY and c.low <= sl_eff)
                hit_tp = (pos.tp_price and pos.side == OrderSide.BUY
                          and c.high >= pos.tp_price)

                # ── FIX #1 : SL+TP même bougie → SL gagne (worst-case) ──
                if hit_sl:
                    to_close.append((pos, pos.sl_price, "RANGE_SL"))
                    self.cooldown_until[pos.symbol] = (
                        ts + self.cfg.range_cooldown_bars * 4 * 3600 * 1000
                    )
                    continue

                if hit_tp:
                    to_close.append((pos, pos.tp_price, "RANGE_TP"))
                    continue

        for pos, exit_px, reason in to_close:
            self._close_position(pos, exit_px, ts, reason)

    # ── Signaux en attente → entrée ────────────────────────────────────────

    def _check_pending_entries(self, ts: int) -> None:
        if len(self.positions) >= self.cfg.max_simultaneous_positions:
            return

        for sym in self.pairs:
            if len(self.positions) >= self.cfg.max_simultaneous_positions:
                break
            # Une seule position par paire
            if any(p.symbol == sym for p in self.positions):
                continue

            c = self._idx.get((sym, ts))
            if c is None:
                continue

            # ── FIX #3 : on vérifie que le signal date d'AVANT cette bougie ──
            # Le signal est généré par _update_analysis() à la fin de la
            # bougie précédente. L'entrée se fait à l'OPEN de la bougie
            # courante si le seuil est déjà franchi à l'ouverture, sinon skip.
            # On n'utilise PAS high/low (= lookahead intra-candle).

            # Trend entry (BUY + SELL si allow_short)
            if self.cfg.enable_trend:
                # En mode DEFENSIVE (EMA filter), TREND est désactivé
                trend_allowed = True
                if self.cfg.use_ema_filter and not self._current_mode_normal:
                    trend_allowed = False

                sig = self._pend_trend.get(sym)
                if trend_allowed and sig:
                    entered = False

                    if sig["side"] == OrderSide.BUY:
                        if sig.get("mode") == "pullback":
                            # PULLBACK D1 LONG : prix descend vers HL
                            if sig["sl_price"] < c.open <= sig["pullback_ceiling"]:
                                entry_px = c.open * (1 + self.cfg.slippage_pct)
                                if sig["sl_price"] >= entry_px:
                                    pass  # invalide
                                else:
                                    self._open_position(
                                        sym, StrategyType.TREND, OrderSide.BUY,
                                        c.open, sig["sl_price"], None, ts,
                                    )
                                    entered = True
                        else:
                            # BREAKOUT H4 LONG
                            if c.open >= sig.get("entry_threshold", 0):
                                entry_px = c.open * (1 + self.cfg.slippage_pct)
                                if sig["sl_price"] >= entry_px:
                                    pass  # invalide
                                else:
                                    self._open_position(
                                        sym, StrategyType.TREND, OrderSide.BUY,
                                        c.open, sig["sl_price"], None, ts,
                                    )
                                    entered = True

                    elif sig["side"] == OrderSide.SELL and self.cfg.allow_short:
                        if sig.get("mode") == "pullback":
                            # PULLBACK D1 SHORT : prix remonte vers LH
                            if sig["pullback_floor"] <= c.open < sig["sl_price"]:
                                entry_px = c.open * (1 - self.cfg.slippage_pct)
                                if sig["sl_price"] <= entry_px:
                                    pass  # invalide
                                else:
                                    self._open_position(
                                        sym, StrategyType.TREND, OrderSide.SELL,
                                        c.open, sig["sl_price"], None, ts,
                                    )
                                    entered = True
                        else:
                            # BREAKOUT H4 SHORT
                            if c.open <= sig.get("entry_threshold", float("inf")):
                                entry_px = c.open * (1 - self.cfg.slippage_pct)
                                if sig["sl_price"] <= entry_px:
                                    pass  # invalide
                                else:
                                    self._open_position(
                                        sym, StrategyType.TREND, OrderSide.SELL,
                                        c.open, sig["sl_price"], None, ts,
                                    )
                                    entered = True

                    if entered:
                        continue

            # Range entry (BUY au bas du range — spot)
            if self.cfg.enable_range:
                sig = self._pend_range.get(sym)
                if sig and sig["side"] == OrderSide.BUY:
                    if ts < self.cooldown_until.get(sym, 0):
                        continue
                    if c.open <= sig["buy_zone"]:
                        entry_px = c.open * (1 + self.cfg.slippage_pct)
                        # FIX: rejeter si SL >= entry (open a gappé sous le SL)
                        # Un BUY avec SL au-dessus du prix d'entrée = non-sens
                        if sig["sl_price"] >= entry_px:
                            continue
                        self._open_position(
                            sym, StrategyType.RANGE, OrderSide.BUY,
                            c.open, sig["sl_price"], sig["tp_price"], ts,
                        )

    # ── Analyse technique H4 (RANGE) ────────────────────────────────────────

    def _update_h4_analysis(self, ts: int) -> None:
        """Analyse H4 : produit les signaux RANGE + signaux TREND breakout (si pas pullback D1)."""
        for sym in self.pairs:
            vis = self._visible(sym, ts)
            if len(vis) < 2 * self.cfg.swing_lookback + 1:
                continue

            swings = detect_swings(vis, self.cfg.swing_lookback)
            if not swings:
                continue

            trend = determine_trend(swings, sym)
            self.trends[sym] = trend

            # Si D1 pullback est actif, les signaux TREND viennent de D1, pas H4
            if not self.cfg.use_d1_pullback:
                self._gen_trend_signal_breakout(sym, trend)

            self._gen_range_signal(sym, trend)

    # Ancien mode : breakout H4 (compatibilité)
    def _gen_trend_signal_breakout(self, sym: str, trend: TrendState) -> None:
        if trend.direction == TrendDirection.BULLISH:
            if trend.entry_level is not None and trend.sl_level is not None:
                buf = trend.entry_level * self.cfg.entry_buffer_pct
                sl_buf = trend.sl_level * self.cfg.entry_buffer_pct
                self._pend_trend[sym] = {
                    "side": OrderSide.BUY,
                    "entry_threshold": trend.entry_level + buf,
                    "sl_price": trend.sl_level - sl_buf,
                    "mode": "breakout",
                }
                return
        if trend.direction == TrendDirection.BEARISH and self.cfg.allow_short:
            if trend.entry_level is not None and trend.sl_level is not None:
                buf = trend.entry_level * self.cfg.entry_buffer_pct
                sl_buf = trend.sl_level * self.cfg.entry_buffer_pct
                self._pend_trend[sym] = {
                    "side": OrderSide.SELL,
                    "entry_threshold": trend.entry_level - buf,  # sous LL
                    "sl_price": trend.sl_level + sl_buf,          # au-dessus LH
                    "mode": "breakout",
                }
                return
        self._pend_trend[sym] = None

    # ── Analyse technique D1 (TREND pullback) ──────────────────────────────

    def _visible_d1(self, symbol: str, up_to_h4_ts: int) -> list[Candle]:
        """Bougies D1 clôturées AVANT le timestamp H4 (pas de lookahead).

        Une bougie D1 ouvre à T et clôture à T + 86400000 ms.
        Elle n'est visible que si T + 86400000 <= up_to_h4_ts.
        """
        _D1_MS = 86_400_000
        clist = self.d1_candles.get(symbol, [])
        vis = [c for c in clist if c.timestamp + _D1_MS <= up_to_h4_ts]
        return vis[-self.cfg.d1_candle_window:]

    def _update_d1_analysis(self, ts: int) -> None:
        """Analyse D1 : structure trend (HH+HL) → signal pullback.

        N'est recalculée que quand une nouvelle bougie D1 devient visible.
        """
        _D1_MS = 86_400_000
        for sym in self.pairs:
            d1_vis = self._visible_d1(sym, ts)
            if len(d1_vis) < 2 * self.cfg.d1_swing_lookback + 1:
                continue

            # Optimisation : ne recalculer que si nouvelle bougie D1
            latest_d1_ts = d1_vis[-1].timestamp
            if latest_d1_ts <= self._last_d1_ts_seen.get(sym, 0):
                continue
            self._last_d1_ts_seen[sym] = latest_d1_ts

            swings = detect_swings(d1_vis, self.cfg.d1_swing_lookback)
            if not swings:
                continue

            d1_trend = determine_trend(swings, sym)
            self.d1_trends[sym] = d1_trend
            self._gen_trend_signal_pullback(sym, d1_trend)

    def _gen_trend_signal_pullback(self, sym: str, d1_trend: TrendState) -> None:
        """Génère un signal TREND pullback basé sur la structure D1.

        LONG  : prix H4 redescend vers le dernier HL D1 (pullback).
        SHORT : prix H4 remonte vers le dernier LH D1 (pullback).
        Pas de TP fixe : trailing stop.
        """
        if d1_trend.direction == TrendDirection.BULLISH:
            hl = d1_trend.sl_level  # last_low.price = dernier HL
            hh = d1_trend.entry_level  # last_high.price = dernier HH
            if hl is not None and hh is not None:
                pullback_ceiling = hl * (1 + self.cfg.pullback_zone_pct)
                sl_price = hl * (1 - self.cfg.sl_buffer_pct)
                self._pend_trend[sym] = {
                    "side": OrderSide.BUY,
                    "pullback_ceiling": pullback_ceiling,
                    "sl_price": sl_price,
                    "hl_level": hl,
                    "hh_level": hh,
                    "mode": "pullback",
                }
                return
        if d1_trend.direction == TrendDirection.BEARISH and self.cfg.allow_short:
            lh = d1_trend.sl_level    # last_high.price = dernier LH
            ll = d1_trend.entry_level  # last_low.price = dernier LL
            if lh is not None and ll is not None:
                pullback_floor = lh * (1 - self.cfg.pullback_zone_pct)
                sl_price = lh * (1 + self.cfg.sl_buffer_pct)
                self._pend_trend[sym] = {
                    "side": OrderSide.SELL,
                    "pullback_floor": pullback_floor,
                    "sl_price": sl_price,
                    "lh_level": lh,
                    "ll_level": ll,
                    "mode": "pullback",
                }
                return
        self._pend_trend[sym] = None

    def _gen_range_signal(self, sym: str, trend: TrendState) -> None:
        if trend.direction != TrendDirection.NEUTRAL:
            self.ranges[sym] = None
            self._pend_range[sym] = None
            return

        rs = build_range_from_trend(trend, self.cfg.range_width_min)
        if rs is None:
            self._pend_range[sym] = None
            return

        self.ranges[sym] = rs
        buy_zone = rs.range_low * (1 + self.cfg.range_entry_buffer_pct)
        sl_price = rs.range_low * (1 - self.cfg.range_entry_buffer_pct)
        self._pend_range[sym] = {
            "side": OrderSide.BUY,
            "buy_zone": buy_zone,
            "sl_price": sl_price,
            "tp_price": rs.range_mid,
        }

    # ── Ouverture / Fermeture ──────────────────────────────────────────────

    def _open_position(
        self,
        symbol: str,
        strategy: StrategyType,
        side: OrderSide,
        entry_price: float,
        sl_price: float,
        tp_price: Optional[float],
        ts: int,
    ) -> None:
        # Slippage adverse à l'entrée
        if side == OrderSide.BUY:
            entry_price = entry_price * (1 + self.cfg.slippage_pct)
        else:
            entry_price = entry_price * (1 - self.cfg.slippage_pct)

        # Sélection du risque : NORMAL vs DEFENSIVE (EMA filter)
        is_normal = self._current_mode_normal
        if strategy == StrategyType.TREND:
            risk_pct = (
                self.cfg.risk_percent_trend
                if is_normal or not self.cfg.use_ema_filter
                else self.cfg.risk_percent_trend_defensive
            )
        else:
            risk_pct = (
                self.cfg.risk_percent_range
                if is_normal or not self.cfg.use_ema_filter
                else self.cfg.risk_percent_range_defensive
            )
        # Sizing : capital fixe (réaliste) ou equity courante (compound)
        sizing_balance = self.cash if self.cfg.compound else self.cfg.initial_balance
        size = calculate_position_size(
            account_balance=sizing_balance,
            risk_percent=risk_pct,
            entry_price=entry_price,
            sl_price=sl_price,
            max_position_percent=self.cfg.max_position_pct,
        )
        if size <= 0:
            return

        cost = size * entry_price * (1 + self.cfg.fee_pct)
        if cost > self.cash:
            size = self.cash / (entry_price * (1 + self.cfg.fee_pct))
            cost = size * entry_price * (1 + self.cfg.fee_pct)
        if size <= 0 or cost > self.cash:
            return

        self.cash -= cost
        oid = f"bt-{len(self.closed_trades) + len(self.positions)}"
        pos = Position(
            symbol=symbol,
            side=side,
            entry_price=entry_price,
            sl_price=sl_price,
            size=size,
            venue_order_id=oid,
            status=PositionStatus.OPEN,
            strategy=strategy,
            tp_price=tp_price,
        )
        self.positions.append(pos)
        self._entry_ts[oid] = ts

    def _close_position(
        self, pos: Position, exit_price: float, ts: int, reason: str,
    ) -> None:
        if pos.side == OrderSide.BUY:
            revenue = pos.size * exit_price * (1 - self.cfg.fee_pct)
            pnl = pos.size * (
                exit_price * (1 - self.cfg.fee_pct)
                - pos.entry_price * (1 + self.cfg.fee_pct)
            )
        else:
            # SHORT : marge restituée + PnL - frais de sortie
            margin = pos.size * pos.entry_price * (1 + self.cfg.fee_pct)
            trade_pnl = pos.size * (pos.entry_price - exit_price)
            exit_fee = pos.size * exit_price * self.cfg.fee_pct
            revenue = margin + trade_pnl - exit_fee
            pnl = trade_pnl - exit_fee
        self.cash += revenue
        pnl_pct = pnl / (pos.size * pos.entry_price) if pos.entry_price else 0

        self.closed_trades.append(
            Trade(
                symbol=pos.symbol,
                strategy=pos.strategy,
                side=pos.side,
                entry_price=pos.entry_price,
                exit_price=exit_price,
                size=pos.size,
                entry_time=self._entry_ts.pop(pos.venue_order_id, 0),
                exit_time=ts,
                pnl_usd=pnl,
                pnl_pct=pnl_pct,
                exit_reason=reason,
            )
        )
        self.positions.remove(pos)

    def _close_remaining(self, last_ts: int) -> None:
        for pos in list(self.positions):
            px = self.last_close.get(pos.symbol, pos.entry_price)
            self._close_position(pos, px, last_ts, "END")

    # ── Equity ─────────────────────────────────────────────────────────────

    def _record_equity(self, ts: int) -> None:
        unrealized = 0.0
        for p in self.positions:
            last = self.last_close.get(p.symbol, p.entry_price)
            if p.side == OrderSide.BUY:
                unrealized += p.size * last
            else:
                # SHORT : marge bloquée + PnL non réalisé
                unrealized += p.size * (2 * p.entry_price - last)
        self.equity_curve.append(EquityPoint(ts, self.cash + unrealized))
