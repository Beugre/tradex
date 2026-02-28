#!/usr/bin/env python
"""
Backtest comparatif : 5 variantes RANGE
  Classique      : TP fixe au range_mid
  Trail@TP       : trailing à partir du range_mid (baseline déjà testé)
  A) Cond. Trail : trailing seulement si ADX↑ + Volume↑ + BB width↑, sinon ferme au mid
  B) Partial 70/30: 70% fermé au mid, 30% laissé courir avec trailing
  C) Time Exit   : trailing activé au mid, mais coupe après N bougies sans nouveau step
"""

from __future__ import annotations

import argparse
import logging
import math
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

from dotenv import load_dotenv
load_dotenv()

from src.core.models import (
    Candle, OrderSide, RangeState, StrategyType,
    TrendDirection, TrendState,
)
from src.core.swing_detector import detect_swings
from src.core.trend_engine import determine_trend
from src.core.strategy_mean_rev import build_range_from_trend
from src.core.risk_manager import calculate_position_size
from backtest.data_loader import download_all_pairs
from backtest.simulator import (
    BacktestConfig, BacktestEngine, BacktestResult, Trade, EquityPoint,
)
from backtest.metrics import compute_metrics

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("range_abc")


# ═══════════════════ Indicators ═══════════════════════════════════════════════

def sma(values: list[float], period: int) -> list[float]:
    result: list[float] = []
    for i in range(len(values)):
        if i < period - 1:
            result.append(float("nan"))
        else:
            result.append(sum(values[i - period + 1:i + 1]) / period)
    return result


def ema(values: list[float], period: int) -> list[float]:
    result: list[float] = []
    k = 2.0 / (period + 1)
    for i, v in enumerate(values):
        if i == 0:
            result.append(v)
        else:
            result.append(v * k + result[-1] * (1 - k))
    return result


def atr_series(candles: list[Candle], period: int = 14) -> list[float]:
    trs: list[float] = []
    for i, c in enumerate(candles):
        if i == 0:
            trs.append(c.high - c.low)
        else:
            prev_c = candles[i - 1].close
            trs.append(max(c.high - c.low, abs(c.high - prev_c), abs(c.low - prev_c)))
    # Wilder smoothing
    result: list[float] = []
    for i in range(len(trs)):
        if i < period - 1:
            result.append(float("nan"))
        elif i == period - 1:
            result.append(sum(trs[:period]) / period)
        else:
            result.append((result[-1] * (period - 1) + trs[i]) / period)
    return result


def adx_series(candles: list[Candle], period: int = 14) -> list[float]:
    """ADX (Wilder). Returns list aligned with candles."""
    if len(candles) < 2:
        return [float("nan")] * len(candles)

    plus_dm: list[float] = [0.0]
    minus_dm: list[float] = [0.0]
    tr_list: list[float] = [candles[0].high - candles[0].low]

    for i in range(1, len(candles)):
        c, p = candles[i], candles[i - 1]
        up = c.high - p.high
        down = p.low - c.low
        plus_dm.append(up if (up > down and up > 0) else 0.0)
        minus_dm.append(down if (down > up and down > 0) else 0.0)
        tr_list.append(max(c.high - c.low, abs(c.high - p.close), abs(c.low - p.close)))

    def wilder_smooth(data: list[float], n: int) -> list[float]:
        res: list[float] = []
        for i in range(len(data)):
            if i < n - 1:
                res.append(float("nan"))
            elif i == n - 1:
                res.append(sum(data[:n]) / n)
            else:
                res.append((res[-1] * (n - 1) + data[i]) / n)
        return res

    sm_tr = wilder_smooth(tr_list, period)
    sm_plus = wilder_smooth(plus_dm, period)
    sm_minus = wilder_smooth(minus_dm, period)

    dx: list[float] = []
    for i in range(len(candles)):
        if math.isnan(sm_tr[i]) or sm_tr[i] == 0:
            dx.append(float("nan"))
        else:
            di_p = 100 * sm_plus[i] / sm_tr[i]
            di_m = 100 * sm_minus[i] / sm_tr[i]
            s = di_p + di_m
            dx.append(100 * abs(di_p - di_m) / s if s > 0 else 0.0)

    adx = wilder_smooth([d if not math.isnan(d) else 0.0 for d in dx], period)
    return adx


def bollinger_width(candles: list[Candle], period: int = 20) -> list[float]:
    """Bollinger Band width = (upper - lower) / mid, normalized."""
    closes = [c.close for c in candles]
    result: list[float] = []
    for i in range(len(closes)):
        if i < period - 1:
            result.append(float("nan"))
        else:
            window = closes[i - period + 1:i + 1]
            m = sum(window) / period
            std = (sum((x - m) ** 2 for x in window) / period) ** 0.5
            upper = m + 2 * std
            lower = m - 2 * std
            result.append((upper - lower) / m if m > 0 else 0.0)
    return result


def volume_sma(candles: list[Candle], period: int = 20) -> list[float]:
    vols = [c.volume for c in candles]
    return sma(vols, period)


# ═══════════════════ Config & Models ═════════════════════════════════════════

@dataclass
class VariantConfig:
    initial_balance: float = 1000.0
    step_pct: float = 0.01
    risk_percent: float = 0.02
    max_position_pct: float = 0.30
    max_simultaneous_positions: int = 3
    fee_pct: float = 0.00075
    slippage_pct: float = 0.001
    swing_lookback: int = 3
    candle_window: int = 100
    range_width_min: float = 0.02
    range_entry_buffer_pct: float = 0.002
    range_sl_buffer_pct: float = 0.003
    range_cooldown_bars: int = 3
    compound: bool = False

    # Option A
    adx_period: int = 14
    bb_period: int = 20
    vol_period: int = 20
    cond_lookback: int = 3  # compare current vs N bars ago

    # Option B
    partial_close_pct: float = 0.70  # 70% fermé au mid

    # Option C
    time_exit_bars: int = 6  # fermer après N bougies sans nouveau step


@dataclass
class VTrade:
    symbol: str
    side: OrderSide
    entry_price: float
    exit_price: float
    size: float
    entry_time: int
    exit_time: int
    pnl_usd: float
    pnl_pct: float
    exit_reason: str
    steps_reached: int = 0
    bars_held: int = 0
    tp_level: float = 0.0


@dataclass
class VPosition:
    symbol: str
    side: OrderSide
    entry_price: float
    sl_price: float
    tp_level: float
    current_sl: float
    size: float
    entry_time: int
    trailing_active: bool = False
    steps_completed: int = 0
    highest_since_entry: float = 0.0
    bars_count: int = 0
    bars_since_last_step: int = 0  # Option C
    partial_closed: bool = False    # Option B
    original_size: float = 0.0     # Option B: taille initiale


# ═══════════════════ Engine Générique ════════════════════════════════════════

class RangeVariantEngine:
    """
    Moteur paramétrable pour les 4 variantes :
      - trail_all   : trailing systématique au TP (= test précédent)
      - cond_trail  : Option A — trail seulement si conditions remplies
      - partial     : Option B — 70% close au TP, 30% trail
      - time_exit   : Option C — trail + time-based exit
    """

    VARIANTS = ("trail_all", "cond_trail", "partial", "time_exit")

    def __init__(self, candles_by_symbol: dict[str, list[Candle]],
                 config: VariantConfig, variant: str, label: str = "") -> None:
        assert variant in self.VARIANTS, f"Unknown variant: {variant}"
        self.variant = variant
        self.label = label or variant
        self.candles = candles_by_symbol
        self.cfg = config
        self.pairs = list(candles_by_symbol.keys())

        self.cash = config.initial_balance
        self.positions: list[VPosition] = []
        self.closed_trades: list[VTrade] = []
        self.equity_curve: list[EquityPoint] = []

        self.trends: dict[str, Optional[TrendState]] = {p: None for p in self.pairs}
        self.ranges: dict[str, Optional[RangeState]] = {p: None for p in self.pairs}
        self.cooldown_until: dict[str, int] = {p: 0 for p in self.pairs}
        self._pending: dict[str, Optional[dict]] = {p: None for p in self.pairs}
        self.last_close: dict[str, float] = {}

        self._idx: dict[tuple[str, int], Candle] = {}
        self._sorted: dict[str, list[Candle]] = {}
        for sym, clist in candles_by_symbol.items():
            self._sorted[sym] = sorted(clist, key=lambda c: c.timestamp)
            for c in clist:
                self._idx[(sym, c.timestamp)] = c

        # Pré-calculer les indicateurs pour Option A
        self._adx: dict[str, list[float]] = {}
        self._bb_width: dict[str, list[float]] = {}
        self._vol_sma: dict[str, list[float]] = {}
        if variant == "cond_trail":
            for sym in self.pairs:
                clist = self._sorted[sym]
                self._adx[sym] = adx_series(clist, config.adx_period)
                self._bb_width[sym] = bollinger_width(clist, config.bb_period)
                self._vol_sma[sym] = volume_sma(clist, config.vol_period)

        # Index ts → position dans sorted list
        self._ts_pos: dict[str, dict[int, int]] = {}
        for sym in self.pairs:
            self._ts_pos[sym] = {
                c.timestamp: i for i, c in enumerate(self._sorted[sym])
            }

    def run(self) -> BacktestResult:
        for mod in ("src.core.swing_detector", "src.core.trend_engine",
                     "src.core.strategy_mean_rev"):
            logging.getLogger(mod).setLevel(logging.WARNING)

        timeline = self._build_timeline()
        total = len(timeline)

        for i, ts in enumerate(timeline):
            for sym in self.pairs:
                c = self._idx.get((sym, ts))
                if c:
                    self.last_close[sym] = c.close

            self._manage_exits(ts)
            self._execute_pending(ts)
            self._analyze(ts)
            self._record_equity(ts)

            if (i + 1) % 500 == 0 or i == total - 1:
                eq = self.equity_curve[-1].equity
                print(f"\r   ⏳ [{self.label}] {i+1}/{total} ({100*(i+1)/total:.0f}%) "
                      f"| Eq: ${eq:,.0f} | Trades: {len(self.closed_trades)}",
                      end="", flush=True)
        print()

        self._close_remaining(timeline[-1])
        final_eq = self.equity_curve[-1].equity if self.equity_curve else self.cash

        trades = [
            Trade(symbol=t.symbol, strategy=StrategyType.RANGE, side=t.side,
                  entry_price=t.entry_price, exit_price=t.exit_price, size=t.size,
                  entry_time=t.entry_time, exit_time=t.exit_time,
                  pnl_usd=t.pnl_usd, pnl_pct=t.pnl_pct, exit_reason=t.exit_reason)
            for t in self.closed_trades
        ]
        return BacktestResult(
            trades=trades, equity_curve=self.equity_curve,
            initial_balance=self.cfg.initial_balance, final_equity=final_eq,
            start_date=datetime.fromtimestamp(timeline[0] / 1000, tz=timezone.utc),
            end_date=datetime.fromtimestamp(timeline[-1] / 1000, tz=timezone.utc),
            pairs=self.pairs,
        )

    def _build_timeline(self) -> list[int]:
        ts_set: set[int] = set()
        for clist in self.candles.values():
            for c in clist:
                ts_set.add(c.timestamp)
        return sorted(ts_set)

    def _visible(self, symbol: str, up_to_ts: int) -> list[Candle]:
        clist = self.candles[symbol]
        vis = [c for c in clist if c.timestamp <= up_to_ts]
        return vis[-self.cfg.candle_window:]

    # ── Conditions Option A ──────────────────────────────────────────────

    def _should_trail_cond(self, symbol: str, ts: int) -> bool:
        """Option A : ADX en hausse + Volume > SMA + BB width en expansion."""
        idx = self._ts_pos.get(symbol, {}).get(ts)
        if idx is None:
            return False
        lb = self.cfg.cond_lookback

        adx = self._adx.get(symbol, [])
        bb = self._bb_width.get(symbol, [])
        vol_s = self._vol_sma.get(symbol, [])

        if idx < lb or idx >= len(adx):
            return False

        # ADX en hausse sur les N dernières bougies
        adx_now = adx[idx]
        adx_prev = adx[idx - lb]
        if math.isnan(adx_now) or math.isnan(adx_prev):
            return False
        adx_rising = adx_now > adx_prev

        # BB width en expansion
        bb_now = bb[idx] if idx < len(bb) else float("nan")
        bb_prev = bb[idx - lb] if (idx - lb) < len(bb) else float("nan")
        if math.isnan(bb_now) or math.isnan(bb_prev):
            return False
        bb_expanding = bb_now > bb_prev

        # Volume > SMA (au moins 1 des dernières N bougies)
        clist = self._sorted[symbol]
        vol_above = False
        for j in range(max(0, idx - lb + 1), idx + 1):
            if j < len(vol_s) and not math.isnan(vol_s[j]):
                if clist[j].volume > vol_s[j]:
                    vol_above = True
                    break

        # Au moins 2 conditions sur 3
        score = sum([adx_rising, bb_expanding, vol_above])
        return score >= 2

    # ── Exits ──────────────────────────────────────────────────────────────

    def _manage_exits(self, ts: int) -> None:
        to_close: list[tuple[VPosition, float, str, float]] = []  # pos, px, reason, size_to_close

        for pos in self.positions:
            c = self._idx.get((pos.symbol, ts))
            if c is None:
                continue

            pos.bars_count += 1
            if c.high > pos.highest_since_entry:
                pos.highest_since_entry = c.high

            # ── 1) SL — PRIORITAIRE ──
            if c.low <= pos.current_sl:
                reason = "TP_TRAIL" if pos.trailing_active else "RANGE_SL"
                to_close.append((pos, pos.current_sl, reason, pos.size))
                if not pos.trailing_active:
                    self.cooldown_until[pos.symbol] = (
                        ts + self.cfg.range_cooldown_bars * 4 * 3600 * 1000
                    )
                continue

            # ── 2) Prix atteint TP (range_mid) ──
            if not pos.trailing_active and c.high >= pos.tp_level:
                if self.variant == "trail_all":
                    # Trail systématique
                    pos.trailing_active = True
                    pos.current_sl = pos.tp_level
                    pos.steps_completed = 1
                    pos.bars_since_last_step = 0

                elif self.variant == "cond_trail":
                    # Option A — Trail conditionnel
                    if self._should_trail_cond(pos.symbol, ts):
                        pos.trailing_active = True
                        pos.current_sl = pos.tp_level
                        pos.steps_completed = 1
                        pos.bars_since_last_step = 0
                    else:
                        # Conditions non remplies → fermer au TP classique
                        to_close.append((pos, pos.tp_level, "RANGE_TP", pos.size))
                        continue

                elif self.variant == "partial":
                    # Option B — Partial close 70/30
                    close_size = pos.size * self.cfg.partial_close_pct
                    remain_size = pos.size - close_size
                    # Fermer 70%
                    to_close.append((pos, pos.tp_level, "PARTIAL_TP", close_size))
                    # Le reste passe en trailing
                    pos.size = remain_size
                    pos.original_size = pos.size  # pour tracking
                    pos.trailing_active = True
                    pos.current_sl = pos.tp_level
                    pos.steps_completed = 1
                    pos.partial_closed = True
                    pos.bars_since_last_step = 0

                elif self.variant == "time_exit":
                    # Option C — Trail + time-based exit
                    pos.trailing_active = True
                    pos.current_sl = pos.tp_level
                    pos.steps_completed = 1
                    pos.bars_since_last_step = 0

            # ── 3) Step trailing au-delà du TP ──
            if pos.trailing_active:
                pos.bars_since_last_step += 1
                old_steps = pos.steps_completed

                next_step = pos.steps_completed + 1
                next_target = pos.tp_level * (1 + self.cfg.step_pct * (next_step - 1))

                while c.high >= next_target:
                    pos.steps_completed = next_step
                    pos.current_sl = pos.tp_level * (
                        1 + self.cfg.step_pct * (next_step - 1)
                    )
                    pos.bars_since_last_step = 0  # reset timer
                    next_step = pos.steps_completed + 1
                    next_target = pos.tp_level * (1 + self.cfg.step_pct * (next_step - 1))

                # Option C — Time exit : trop longtemps sans progrès → fermer
                if self.variant == "time_exit":
                    if pos.bars_since_last_step >= self.cfg.time_exit_bars:
                        to_close.append((pos, c.close, "TIME_EXIT", pos.size))
                        continue

        for pos, exit_px, reason, close_size in to_close:
            if reason == "PARTIAL_TP":
                self._partial_close(pos, exit_px, ts, reason, close_size)
            else:
                self._close_position(pos, exit_px, ts, reason)

    # ── Pending entries ────────────────────────────────────────────────────

    def _execute_pending(self, ts: int) -> None:
        if len(self.positions) >= self.cfg.max_simultaneous_positions:
            return

        for sym in self.pairs:
            if len(self.positions) >= self.cfg.max_simultaneous_positions:
                break
            if any(p.symbol == sym for p in self.positions):
                continue

            sig = self._pending.get(sym)
            if sig is None:
                continue

            c = self._idx.get((sym, ts))
            if c is None:
                continue

            if ts < self.cooldown_until.get(sym, 0):
                continue
            if c.open > sig["buy_zone"]:
                continue

            entry_price = c.open * (1 + self.cfg.slippage_pct)
            if sig["sl_price"] >= entry_price:
                continue
            if sig["tp_price"] <= entry_price:
                continue

            sizing_balance = self.cash if self.cfg.compound else self.cfg.initial_balance
            size = calculate_position_size(
                account_balance=sizing_balance, risk_percent=self.cfg.risk_percent,
                entry_price=entry_price, sl_price=sig["sl_price"],
                max_position_percent=self.cfg.max_position_pct,
            )
            if size <= 0:
                continue

            cost = size * entry_price * (1 + self.cfg.fee_pct)
            if cost > self.cash:
                size = self.cash / (entry_price * (1 + self.cfg.fee_pct))
                cost = size * entry_price * (1 + self.cfg.fee_pct)
            if size <= 0 or cost > self.cash:
                continue

            self.cash -= cost
            self.positions.append(VPosition(
                symbol=sym, side=OrderSide.BUY,
                entry_price=entry_price, sl_price=sig["sl_price"],
                tp_level=sig["tp_price"], current_sl=sig["sl_price"],
                size=size, entry_time=ts,
                highest_since_entry=entry_price,
                original_size=size,
            ))
            self._pending[sym] = None

    # ── Analyse Dow ────────────────────────────────────────────────────────

    def _analyze(self, ts: int) -> None:
        for sym in self.pairs:
            vis = self._visible(sym, ts)
            if len(vis) < 2 * self.cfg.swing_lookback + 1:
                continue

            swings = detect_swings(vis, self.cfg.swing_lookback)
            if not swings:
                continue
            trend = determine_trend(swings, sym)
            self.trends[sym] = trend

            if trend.direction != TrendDirection.NEUTRAL:
                self.ranges[sym] = None
                self._pending[sym] = None
                for pos in list(self.positions):
                    if pos.symbol == sym:
                        px = self.last_close.get(sym, pos.entry_price)
                        self._close_position(pos, px, ts, "TREND_BREAK")
                continue

            rs = build_range_from_trend(trend, self.cfg.range_width_min)
            if rs is None:
                self._pending[sym] = None
                continue
            self.ranges[sym] = rs

            buy_zone = rs.range_low * (1 + self.cfg.range_entry_buffer_pct)
            sl_price = rs.range_low * (1 - self.cfg.range_sl_buffer_pct)

            self._pending[sym] = {
                "side": OrderSide.BUY, "buy_zone": buy_zone,
                "sl_price": sl_price, "tp_price": rs.range_mid,
            }

    # ── Close / Partial / Equity ───────────────────────────────────────────

    def _partial_close(self, pos: VPosition, exit_price: float,
                       ts: int, reason: str, close_size: float) -> None:
        """Fermer une partie de la position (Option B)."""
        revenue = close_size * exit_price * (1 - self.cfg.fee_pct)
        pnl = close_size * (
            exit_price * (1 - self.cfg.fee_pct)
            - pos.entry_price * (1 + self.cfg.fee_pct)
        )
        self.cash += revenue
        pnl_pct = pnl / (close_size * pos.entry_price) if pos.entry_price else 0

        self.closed_trades.append(VTrade(
            symbol=pos.symbol, side=pos.side,
            entry_price=pos.entry_price, exit_price=exit_price,
            size=close_size, entry_time=pos.entry_time, exit_time=ts,
            pnl_usd=pnl, pnl_pct=pnl_pct, exit_reason=reason,
            steps_reached=pos.steps_completed, bars_held=pos.bars_count,
            tp_level=pos.tp_level,
        ))
        # Position reste ouverte avec size réduite

    def _close_position(self, pos: VPosition, exit_price: float,
                        ts: int, reason: str) -> None:
        revenue = pos.size * exit_price * (1 - self.cfg.fee_pct)
        pnl = pos.size * (
            exit_price * (1 - self.cfg.fee_pct)
            - pos.entry_price * (1 + self.cfg.fee_pct)
        )
        self.cash += revenue
        pnl_pct = pnl / (pos.size * pos.entry_price) if pos.entry_price else 0

        self.closed_trades.append(VTrade(
            symbol=pos.symbol, side=pos.side,
            entry_price=pos.entry_price, exit_price=exit_price,
            size=pos.size, entry_time=pos.entry_time, exit_time=ts,
            pnl_usd=pnl, pnl_pct=pnl_pct, exit_reason=reason,
            steps_reached=pos.steps_completed, bars_held=pos.bars_count,
            tp_level=pos.tp_level,
        ))
        if pos in self.positions:
            self.positions.remove(pos)

    def _close_remaining(self, last_ts: int) -> None:
        for pos in list(self.positions):
            px = self.last_close.get(pos.symbol, pos.entry_price)
            self._close_position(pos, px, last_ts, "END")

    def _record_equity(self, ts: int) -> None:
        unrealized = sum(
            p.size * self.last_close.get(p.symbol, p.entry_price)
            for p in self.positions
        )
        self.equity_curve.append(EquityPoint(ts, self.cash + unrealized))


# ═══════════════════ Rapport ════════════════════════════════════════════════

def print_multi_comparison(results: dict[str, tuple[BacktestResult, dict, list[VTrade]]],
                           m_classic: dict, res_classic: BacktestResult) -> None:
    sep = "═" * 100
    print(f"\n{sep}")
    print(f"  🔬 COMPARAISON MULTI-VARIANTES RANGE")
    print(f"  📅 {res_classic.start_date:%b %Y} → {res_classic.end_date:%b %Y}")
    print(f"  🪙 {len(res_classic.pairs)} paires | Capital: ${res_classic.initial_balance:,.0f}")
    print(sep)

    labels = ["CLASSIQUE"] + list(results.keys())
    metrics_list = [
        ("Return", "total_return", "{:+.1%}"),
        ("CAGR", "cagr", "{:+.1%}"),
        ("Max Drawdown", "max_drawdown", "{:.1%}"),
        ("Sharpe", "sharpe", "{:.2f}"),
        ("Sortino", "sortino", "{:.2f}"),
        ("Win Rate", "win_rate", "{:.1%}"),
        ("Profit Factor", "profit_factor", "{:.2f}"),
        ("Trades", "n_trades", "{:d}"),
        ("PnL moyen ($)", "avg_pnl_usd", "{:+.2f}"),
        ("Capital final", "final_equity", "${:,.2f}"),
    ]

    # Header
    hdr = f"  {'Métrique':<20s} │ {'CLASSIQUE':>12s}"
    for lbl in results.keys():
        hdr += f" │ {lbl:>12s}"
    print(f"\n{hdr}")
    print("  " + "─" * (22 + 15 * (len(results) + 1)))

    all_m = [m_classic] + [r[1] for r in results.values()]

    for metric_label, key, fmt in metrics_list:
        row = f"  {metric_label:<20s}"
        vals = []
        for m in all_m:
            v = m.get(key, 0)
            vals.append(v)
            s = fmt.format(v) if isinstance(v, (int, float)) else str(v)
            row += f" │ {s:>12s}"

        # Find best
        if key == "max_drawdown":
            best_idx = max(range(len(vals)), key=lambda i: vals[i])
        elif key == "n_trades":
            best_idx = -1  # no "best"
        else:
            best_idx = max(range(len(vals)), key=lambda i: vals[i])

        if best_idx >= 0:
            row += f"  ← {labels[best_idx]}"
        print(row)

    # Détails par variante
    for name, (res, m, trades) in results.items():
        if not trades:
            continue
        print(f"\n  📊 {name} — Détail sorties")
        print("  " + "─" * 80)

        exit_map: dict[str, list[VTrade]] = {}
        for t in trades:
            exit_map.setdefault(t.exit_reason, []).append(t)

        for reason, tlist in sorted(exit_map.items()):
            n = len(tlist)
            wins = sum(1 for t in tlist if t.pnl_usd > 0)
            pnl = sum(t.pnl_usd for t in tlist)
            wr = wins / n if n else 0
            avg_steps = sum(t.steps_reached for t in tlist) / n if n else 0
            print(f"    {reason:16s} : {n:4d} trades | WR {wr:4.0%} | "
                  f"PnL ${pnl:+8.2f} | Steps {avg_steps:.1f}")

        # Trades qui ont atteint le TP
        activated = [t for t in trades if t.steps_reached >= 1]
        if activated:
            act_pnl = sum(t.pnl_usd for t in activated)
            classic_equiv = sum(
                t.size * (t.tp_level * (1 - 0.00075) - t.entry_price * (1 + 0.00075))
                for t in activated
            )
            diff = act_pnl - classic_equiv
            emoji = "✅" if diff > 0 else "❌"
            print(f"    Trailing activé : {len(activated)} trades | "
                  f"PnL ${act_pnl:+.2f} vs classic TP ${classic_equiv:+.2f} "
                  f"(Δ ${diff:+.2f} {emoji})")

    print(f"\n{sep}\n")


# ═══════════════════ Main ═══════════════════════════════════════════════════

PAIRS_20 = [
    "BTC-USD", "ETH-USD", "SOL-USD", "XRP-USD", "LINK-USD",
    "ADA-USD", "DOT-USD", "AVAX-USD", "DOGE-USD", "ATOM-USD",
    "UNI-USD", "NEAR-USD", "ALGO-USD", "LTC-USD", "ETC-USD",
    "FIL-USD", "AAVE-USD", "INJ-USD", "SAND-USD", "MANA-USD",
]


def main() -> None:
    parser = argparse.ArgumentParser(description="Range A/B/C multi-variant test")
    parser.add_argument("--balance", type=float, default=1000.0)
    parser.add_argument("--start", default="2022-02-20")
    parser.add_argument("--end", default="2026-02-20")
    parser.add_argument("--pairs", type=int, default=20)
    parser.add_argument("--step-pct", type=float, default=0.01)
    parser.add_argument("--time-bars", type=int, default=6)
    parser.add_argument("--partial-pct", type=float, default=0.70)
    parser.add_argument("--no-show", action="store_true")
    args = parser.parse_args()

    start = datetime.strptime(args.start, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    end = datetime.strptime(args.end, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    pairs = PAIRS_20[:args.pairs]

    print(f"\n{'═'*100}")
    print(f"  🔬 RANGE MULTI-VARIANTES A/B/C — Comparaison complète")
    print(f"  📅 {start:%Y-%m-%d} → {end:%Y-%m-%d} | {len(pairs)} paires | ${args.balance:,.0f}")
    print(f"  Step: {args.step_pct:.0%} | Partial: {args.partial_pct:.0%} | Time exit: {args.time_bars} bougies")
    print(f"{'═'*100}\n")

    logger.info("📥 Téléchargement des données…")
    candles = download_all_pairs(pairs, start, end, interval="4h")

    # ── 0) Classique ──
    print("\n" + "─" * 60)
    print("  🏛️  RANGE CLASSIQUE (TP = range_mid)")
    print("─" * 60)

    from src import config
    cfg_classic = BacktestConfig(
        initial_balance=args.balance, risk_percent_range=0.02,
        entry_buffer_pct=getattr(config, "ENTRY_BUFFER_PERCENT", 0.002),
        sl_buffer_pct=getattr(config, "SL_BUFFER_PERCENT", 0.003),
        zero_risk_trigger_pct=getattr(config, "ZERO_RISK_TRIGGER_PERCENT", 0.02),
        zero_risk_lock_pct=getattr(config, "ZERO_RISK_LOCK_PERCENT", 0.005),
        trailing_stop_pct=getattr(config, "TRAILING_STOP_PERCENT", 0.02),
        max_position_pct=0.30, max_simultaneous_positions=3,
        swing_lookback=getattr(config, "SWING_LOOKBACK", 3),
        range_width_min=getattr(config, "RANGE_WIDTH_MIN", 0.02),
        range_entry_buffer_pct=getattr(config, "RANGE_ENTRY_BUFFER_PERCENT", 0.002),
        range_sl_buffer_pct=getattr(config, "RANGE_SL_BUFFER_PERCENT", 0.003),
        range_cooldown_bars=getattr(config, "RANGE_COOLDOWN_BARS", 3),
        enable_trend=False, enable_range=True,
    )
    engine_classic = BacktestEngine(candles, cfg_classic)
    result_classic = engine_classic.run()
    m_classic = compute_metrics(result_classic)

    # ── Config commune pour toutes les variantes ──
    base_cfg = VariantConfig(
        initial_balance=args.balance, step_pct=args.step_pct,
        range_entry_buffer_pct=getattr(config, "RANGE_ENTRY_BUFFER_PERCENT", 0.002),
        range_sl_buffer_pct=getattr(config, "RANGE_SL_BUFFER_PERCENT", 0.003),
        range_cooldown_bars=getattr(config, "RANGE_COOLDOWN_BARS", 3),
        swing_lookback=getattr(config, "SWING_LOOKBACK", 3),
        range_width_min=getattr(config, "RANGE_WIDTH_MIN", 0.02),
        time_exit_bars=args.time_bars,
        partial_close_pct=args.partial_pct,
    )

    variants = [
        ("Trail@TP", "trail_all"),
        ("A:Cond", "cond_trail"),
        ("B:Part70/30", "partial"),
        ("C:TimeExit", "time_exit"),
    ]

    results: dict[str, tuple[BacktestResult, dict, list[VTrade]]] = {}

    for label, var_name in variants:
        print(f"\n{'─'*60}")
        print(f"  🔧 {label}")
        print("─" * 60)
        engine = RangeVariantEngine(candles, base_cfg, var_name, label)
        res = engine.run()
        m = compute_metrics(res)
        results[label] = (res, m, engine.closed_trades)

    # ── Comparaison ──
    print_multi_comparison(results, m_classic, result_classic)


if __name__ == "__main__":
    main()
