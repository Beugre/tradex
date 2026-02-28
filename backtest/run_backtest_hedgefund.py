#!/usr/bin/env python
"""
Backtest "Scalable Hedge Fund" — Architecture complète.

Compare :
  1) 100% Classic (baseline)
  2) 60/40 simple (sans risk overlay)
  3) 60/40 Hedge Fund (avec risk engine complet)

Risk Engine :
  🔒 Equity-Based Scaling   — réduit taille selon le drawdown courant
  📊 Volatility Targeting    — normalise la taille pour vol cible 25% annualisée
  🛑 Kill Switch             — pause après 4 pertes consécutives ou perte extrême
  🔗 Correlation Monitor     — réduit exposition si classic/trail trop corrélés
  📈 Monte Carlo             — simulation worst case en fin de rapport
"""

from __future__ import annotations

import argparse
import logging
import math
import random
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
logger = logging.getLogger("hedge_fund")


# ═══════════════════ Config ═════════════════════════════════════════════════

@dataclass
class HedgeFundConfig:
    initial_balance: float = 1000.0
    step_pct: float = 0.01          # trailing step size
    risk_percent: float = 0.02
    max_position_pct: float = 0.30
    max_simultaneous_symbols: int = 3
    fee_pct: float = 0.00075
    slippage_pct: float = 0.001
    swing_lookback: int = 3
    candle_window: int = 100
    range_width_min: float = 0.02
    range_entry_buffer_pct: float = 0.002
    range_sl_buffer_pct: float = 0.003
    range_cooldown_bars: int = 3
    compound: bool = False

    # 60/40 split
    classic_pct: float = 0.60
    trail_pct: float = 0.40

    # Risk Engine toggle
    risk_overlay: bool = True

    # Equity-Based Scaling
    dd_scale_3: float = 0.80     # DD 3-6% → 80%
    dd_scale_6: float = 0.60     # DD 6-8% → 60%
    dd_scale_8: float = 0.30     # DD >8% → 30%

    # Volatility Targeting
    target_vol_annual: float = 0.25  # 25% annualisée
    vol_lookback: int = 30           # barres H4 pour le calcul
    vol_scale_min: float = 0.50
    vol_scale_max: float = 1.50

    # Kill Switch
    kill_consecutive_losses: int = 4
    kill_extreme_mult: float = 2.0   # si perte > 2× perte moyenne → pause
    kill_pause_bars: int = 6         # 6 × 4h = 24h

    # Correlation
    corr_lookback: int = 30          # 30 derniers trades
    corr_threshold: float = 0.80
    corr_reduction: float = 0.50     # réduire de 50%


# ═══════════════════ Models ═════════════════════════════════════════════════

@dataclass
class HFTrade:
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
    mode: str = "classic"      # "classic" ou "trail"
    steps_reached: int = 0
    bars_held: int = 0
    tp_level: float = 0.0
    risk_scales: str = ""      # debug: scales appliquées


@dataclass
class HFPosition:
    symbol: str
    side: OrderSide
    entry_price: float
    sl_price: float
    tp_level: float
    current_sl: float
    size: float
    entry_time: int
    mode: str = "classic"
    trailing_active: bool = False
    steps_completed: int = 0
    highest_since_entry: float = 0.0
    bars_count: int = 0


# ═══════════════════ Risk Engine ════════════════════════════════════════════

class RiskEngine:
    """Couche de risk management centralisée."""

    def __init__(self, cfg: HedgeFundConfig) -> None:
        self.cfg = cfg
        self.peak_equity: float = cfg.initial_balance
        self.current_equity: float = cfg.initial_balance

        # Kill switch
        self.recent_results: list[float] = []    # PnL des derniers trades
        self.paused_until: int = 0

        # Correlation
        self.classic_pnls: list[float] = []
        self.trail_pnls: list[float] = []

        # Volatility
        self.equity_history: list[float] = []

        # Stats
        self.n_dd_scaled: int = 0
        self.n_vol_scaled: int = 0
        self.n_kill_paused: int = 0
        self.n_corr_reduced: int = 0
        self.scale_history: list[tuple[float, float, float, float]] = []  # dd,vol,corr,total

    def update_equity(self, equity: float) -> None:
        self.current_equity = equity
        if equity > self.peak_equity:
            self.peak_equity = equity
        self.equity_history.append(equity)

    def record_trade(self, trade: HFTrade) -> None:
        self.recent_results.append(trade.pnl_usd)
        if trade.mode == "classic":
            self.classic_pnls.append(trade.pnl_usd)
        else:
            self.trail_pnls.append(trade.pnl_usd)

    def is_paused(self, ts: int) -> bool:
        if not self.cfg.risk_overlay:
            return False
        return ts < self.paused_until

    def check_kill_switch(self, ts: int) -> None:
        if not self.cfg.risk_overlay:
            return

        n = self.cfg.kill_consecutive_losses
        if len(self.recent_results) >= n:
            last_n = self.recent_results[-n:]
            if all(pnl < 0 for pnl in last_n):
                self.paused_until = ts + self.cfg.kill_pause_bars * 4 * 3600 * 1000
                self.n_kill_paused += 1
                return

        # Perte extrême
        if len(self.recent_results) >= 10:
            losses = [p for p in self.recent_results if p < 0]
            if losses:
                avg_loss = sum(losses) / len(losses)
                if self.recent_results[-1] < 0:
                    if self.recent_results[-1] < avg_loss * self.cfg.kill_extreme_mult:
                        self.paused_until = ts + self.cfg.kill_pause_bars * 4 * 3600 * 1000
                        self.n_kill_paused += 1

    def get_position_scale(self) -> float:
        if not self.cfg.risk_overlay:
            return 1.0

        dd_scale = self._dd_scale()
        vol_scale = self._vol_scale()
        corr_scale = self._corr_scale()
        total = dd_scale * vol_scale * corr_scale

        self.scale_history.append((dd_scale, vol_scale, corr_scale, total))

        if dd_scale < 1.0:
            self.n_dd_scaled += 1
        if vol_scale != 1.0:
            self.n_vol_scaled += 1
        if corr_scale < 1.0:
            self.n_corr_reduced += 1

        return max(0.1, min(total, 1.5))

    def _dd_scale(self) -> float:
        if self.peak_equity <= 0:
            return 1.0
        dd = (self.peak_equity - self.current_equity) / self.peak_equity
        if dd < 0.03:
            return 1.0
        elif dd < 0.06:
            return self.cfg.dd_scale_3
        elif dd < 0.08:
            return self.cfg.dd_scale_6
        else:
            return self.cfg.dd_scale_8

    def _vol_scale(self) -> float:
        n = self.cfg.vol_lookback
        if len(self.equity_history) < n + 1:
            return 1.0

        recent = self.equity_history[-n:]
        returns = []
        for i in range(1, len(recent)):
            if recent[i - 1] > 0:
                returns.append(recent[i] / recent[i - 1] - 1)

        if len(returns) < 5:
            return 1.0

        mean_r = sum(returns) / len(returns)
        var = sum((r - mean_r) ** 2 for r in returns) / len(returns)
        h4_vol = var ** 0.5

        # Annualiser : ~1500 barres H4/an
        annual_vol = h4_vol * (1500 ** 0.5)

        if annual_vol <= 0.001:
            return self.cfg.vol_scale_max

        scale = self.cfg.target_vol_annual / annual_vol
        return max(self.cfg.vol_scale_min, min(scale, self.cfg.vol_scale_max))

    def _corr_scale(self) -> float:
        n = self.cfg.corr_lookback
        if len(self.classic_pnls) < n or len(self.trail_pnls) < n:
            return 1.0

        c = self.classic_pnls[-n:]
        t = self.trail_pnls[-n:]

        # Pearson correlation
        mean_c = sum(c) / n
        mean_t = sum(t) / n
        cov = sum((c[i] - mean_c) * (t[i] - mean_t) for i in range(n)) / n
        std_c = (sum((x - mean_c) ** 2 for x in c) / n) ** 0.5
        std_t = (sum((x - mean_t) ** 2 for x in t) / n) ** 0.5

        if std_c < 1e-10 or std_t < 1e-10:
            return 1.0

        corr = cov / (std_c * std_t)

        if corr > self.cfg.corr_threshold:
            return self.cfg.corr_reduction
        return 1.0


# ═══════════════════ Hedge Fund Engine ══════════════════════════════════════

class HedgeFundEngine:
    """Engine 60/40 avec Risk Engine centralisé."""

    def __init__(self, candles_by_symbol: dict[str, list[Candle]],
                 config: HedgeFundConfig, label: str = "HF") -> None:
        self.label = label
        self.candles = candles_by_symbol
        self.cfg = config
        self.pairs = list(candles_by_symbol.keys())

        self.cash = config.initial_balance
        self.positions: list[HFPosition] = []
        self.closed_trades: list[HFTrade] = []
        self.equity_curve: list[EquityPoint] = []

        self.risk = RiskEngine(config)

        self.trends: dict[str, Optional[TrendState]] = {p: None for p in self.pairs}
        self.ranges: dict[str, Optional[RangeState]] = {p: None for p in self.pairs}
        self.cooldown_until: dict[str, int] = {p: 0 for p in self.pairs}
        self._pending: dict[str, Optional[dict]] = {p: None for p in self.pairs}
        self.last_close: dict[str, float] = {}

        self._idx: dict[tuple[str, int], Candle] = {}
        for sym, clist in candles_by_symbol.items():
            for c in clist:
                self._idx[(sym, c.timestamp)] = c

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
            eq = self._record_equity(ts)
            self.risk.update_equity(eq)

            if (i + 1) % 500 == 0 or i == total - 1:
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

    def _active_symbols(self) -> set[str]:
        return {p.symbol for p in self.positions}

    # ── Exits ──────────────────────────────────────────────────────────────

    def _manage_exits(self, ts: int) -> None:
        to_close: list[tuple[HFPosition, float, str]] = []

        for pos in self.positions:
            c = self._idx.get((pos.symbol, ts))
            if c is None:
                continue

            pos.bars_count += 1
            if c.high > pos.highest_since_entry:
                pos.highest_since_entry = c.high

            # SL
            if c.low <= pos.current_sl:
                if pos.mode == "trail" and pos.trailing_active:
                    reason = "TP_TRAIL"
                elif pos.mode == "classic":
                    reason = "RANGE_SL"
                else:
                    reason = "RANGE_SL"
                to_close.append((pos, pos.current_sl, reason))
                if not pos.trailing_active:
                    self.cooldown_until[pos.symbol] = (
                        ts + self.cfg.range_cooldown_bars * 4 * 3600 * 1000
                    )
                continue

            # TP atteint
            if c.high >= pos.tp_level:
                if pos.mode == "classic":
                    # Classic : fermer au TP
                    to_close.append((pos, pos.tp_level, "RANGE_TP"))
                    continue
                elif pos.mode == "trail" and not pos.trailing_active:
                    # Trail : activer trailing
                    pos.trailing_active = True
                    pos.current_sl = pos.tp_level
                    pos.steps_completed = 1

            # Step trailing
            if pos.mode == "trail" and pos.trailing_active:
                next_step = pos.steps_completed + 1
                next_target = pos.tp_level * (1 + self.cfg.step_pct * (next_step - 1))
                while c.high >= next_target:
                    pos.steps_completed = next_step
                    pos.current_sl = pos.tp_level * (
                        1 + self.cfg.step_pct * (next_step - 1)
                    )
                    next_step = pos.steps_completed + 1
                    next_target = pos.tp_level * (1 + self.cfg.step_pct * (next_step - 1))

        for pos, exit_px, reason in to_close:
            self._close_position(pos, exit_px, ts, reason)

    # ── Entries ────────────────────────────────────────────────────────────

    def _execute_pending(self, ts: int) -> None:
        active_syms = self._active_symbols()
        if len(active_syms) >= self.cfg.max_simultaneous_symbols:
            return

        # Kill switch check
        if self.risk.is_paused(ts):
            return

        # Risk scale
        risk_scale = self.risk.get_position_scale()

        for sym in self.pairs:
            if len(self._active_symbols()) >= self.cfg.max_simultaneous_symbols:
                break
            if sym in self._active_symbols():
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
            base_size = calculate_position_size(
                account_balance=sizing_balance, risk_percent=self.cfg.risk_percent,
                entry_price=entry_price, sl_price=sig["sl_price"],
                max_position_percent=self.cfg.max_position_pct,
            )
            if base_size <= 0:
                continue

            # Apply risk scale
            scaled_size = base_size * risk_scale
            scale_info = f"scale={risk_scale:.2f}"

            # Split 60/40
            classic_size = scaled_size * self.cfg.classic_pct
            trail_size = scaled_size * self.cfg.trail_pct

            total_cost = scaled_size * entry_price * (1 + self.cfg.fee_pct)
            if total_cost > self.cash:
                ratio = self.cash / total_cost
                classic_size *= ratio
                trail_size *= ratio
                total_cost = self.cash

            if classic_size <= 0 and trail_size <= 0:
                continue

            self.cash -= total_cost

            # Classic position
            if classic_size > 0:
                self.positions.append(HFPosition(
                    symbol=sym, side=OrderSide.BUY, mode="classic",
                    entry_price=entry_price, sl_price=sig["sl_price"],
                    tp_level=sig["tp_price"], current_sl=sig["sl_price"],
                    size=classic_size, entry_time=ts,
                    highest_since_entry=entry_price,
                ))

            # Trail position
            if trail_size > 0:
                self.positions.append(HFPosition(
                    symbol=sym, side=OrderSide.BUY, mode="trail",
                    entry_price=entry_price, sl_price=sig["sl_price"],
                    tp_level=sig["tp_price"], current_sl=sig["sl_price"],
                    size=trail_size, entry_time=ts,
                    highest_since_entry=entry_price,
                ))

            self._pending[sym] = None

    # ── Analyse ────────────────────────────────────────────────────────────

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

    # ── Close / Equity ─────────────────────────────────────────────────────

    def _close_position(self, pos: HFPosition, exit_price: float,
                        ts: int, reason: str) -> None:
        revenue = pos.size * exit_price * (1 - self.cfg.fee_pct)
        pnl = pos.size * (
            exit_price * (1 - self.cfg.fee_pct)
            - pos.entry_price * (1 + self.cfg.fee_pct)
        )
        self.cash += revenue
        pnl_pct = pnl / (pos.size * pos.entry_price) if pos.entry_price else 0

        trade = HFTrade(
            symbol=pos.symbol, side=pos.side,
            entry_price=pos.entry_price, exit_price=exit_price,
            size=pos.size, entry_time=pos.entry_time, exit_time=ts,
            pnl_usd=pnl, pnl_pct=pnl_pct, exit_reason=reason,
            mode=pos.mode, steps_reached=pos.steps_completed,
            bars_held=pos.bars_count, tp_level=pos.tp_level,
        )
        self.closed_trades.append(trade)
        self.risk.record_trade(trade)
        self.risk.check_kill_switch(ts)

        if pos in self.positions:
            self.positions.remove(pos)

    def _close_remaining(self, last_ts: int) -> None:
        for pos in list(self.positions):
            px = self.last_close.get(pos.symbol, pos.entry_price)
            self._close_position(pos, px, last_ts, "END")

    def _record_equity(self, ts: int) -> float:
        unrealized = sum(
            p.size * self.last_close.get(p.symbol, p.entry_price)
            for p in self.positions
        )
        eq = self.cash + unrealized
        self.equity_curve.append(EquityPoint(ts, eq))
        return eq


# ═══════════════════ Monthly Returns ════════════════════════════════════════

def compute_monthly_returns(equity_curve: list[EquityPoint]) -> list[tuple[str, float]]:
    """Returns list of (YYYY-MM, return_pct)."""
    if not equity_curve:
        return []
    monthly: dict[str, tuple[float, float]] = {}  # month -> (first_eq, last_eq)
    for ep in equity_curve:
        dt = datetime.fromtimestamp(ep.timestamp / 1000, tz=timezone.utc)
        key = f"{dt.year}-{dt.month:02d}"
        if key not in monthly:
            monthly[key] = (ep.equity, ep.equity)
        else:
            monthly[key] = (monthly[key][0], ep.equity)

    result = []
    prev_eq = None
    for month in sorted(monthly.keys()):
        first_eq, last_eq = monthly[month]
        base = prev_eq if prev_eq else first_eq
        ret = (last_eq / base - 1) if base > 0 else 0
        result.append((month, ret))
        prev_eq = last_eq
    return result


# ═══════════════════ Monte Carlo ════════════════════════════════════════════

def monte_carlo_simulation(trades: list[HFTrade], initial_balance: float,
                           n_sims: int = 5000) -> dict:
    """Bootstrap Monte Carlo sur les trades."""
    if not trades:
        return {}

    pnls = [t.pnl_usd for t in trades]
    n_trades = len(pnls)

    final_returns = []
    max_dds = []
    max_streaks = []

    random.seed(42)
    for _ in range(n_sims):
        sample = random.choices(pnls, k=n_trades)
        equity = initial_balance
        peak = equity
        max_dd = 0.0
        streak = 0
        max_s = 0

        for pnl in sample:
            equity += pnl
            if equity > peak:
                peak = equity
            dd = (peak - equity) / peak if peak > 0 else 0
            if dd > max_dd:
                max_dd = dd
            if pnl < 0:
                streak += 1
                if streak > max_s:
                    max_s = streak
            else:
                streak = 0

        final_returns.append(equity / initial_balance - 1)
        max_dds.append(max_dd)
        max_streaks.append(max_s)

    final_returns.sort()
    max_dds.sort()
    max_streaks.sort()

    return {
        "return_5th": final_returns[int(0.05 * n_sims)],
        "return_50th": final_returns[int(0.50 * n_sims)],
        "return_95th": final_returns[int(0.95 * n_sims)],
        "dd_5th": max_dds[int(0.05 * n_sims)],
        "dd_50th": max_dds[int(0.50 * n_sims)],
        "dd_95th": max_dds[int(0.95 * n_sims)],
        "streak_50th": max_streaks[int(0.50 * n_sims)],
        "streak_95th": max_streaks[int(0.95 * n_sims)],
    }


# ═══════════════════ Rapport ════════════════════════════════════════════════

def print_report(
    m_classic: dict, res_classic: BacktestResult,
    m_simple: dict, res_simple: BacktestResult,
    m_hf: dict, res_hf: BacktestResult,
    engine_hf: HedgeFundEngine,
) -> None:
    sep = "═" * 95
    print(f"\n{sep}")
    print(f"  🏦 HEDGE FUND ARCHITECTURE — Comparaison complète")
    print(f"  📅 {res_classic.start_date:%b %Y} → {res_classic.end_date:%b %Y}")
    print(f"  🪙 {len(res_classic.pairs)} paires | Capital: ${res_classic.initial_balance:,.0f}")
    print(sep)

    labels = ["100% Classic", "60/40 Simple", "60/40 HedgeFund"]
    all_m = [m_classic, m_simple, m_hf]

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

    hdr = f"  {'Métrique':<20s}"
    for lbl in labels:
        hdr += f" │ {lbl:>16s}"
    print(f"\n{hdr}")
    print("  " + "─" * (22 + 19 * len(labels)))

    for metric_label, key, fmt in metrics_list:
        row = f"  {metric_label:<20s}"
        vals = []
        for m in all_m:
            v = m.get(key, 0)
            vals.append(v)
            s = fmt.format(v) if isinstance(v, (int, float)) else str(v)
            row += f" │ {s:>16s}"

        if key == "max_drawdown":
            best_idx = max(range(len(vals)), key=lambda i: vals[i])
        elif key == "n_trades":
            best_idx = -1
        else:
            best_idx = max(range(len(vals)), key=lambda i: vals[i])

        if best_idx >= 0:
            row += f"  ← {labels[best_idx]}"
        print(row)

    # Return/MaxDD
    print(f"\n  {'Return/MaxDD':<20s}", end="")
    ratios = []
    for m in all_m:
        r = m.get("total_return", 0)
        dd = abs(m.get("max_drawdown", -1))
        ratio = r / dd if dd > 0 else 0
        ratios.append(ratio)
        print(f" │ {ratio:>16.2f}", end="")
    best_r = max(range(len(ratios)), key=lambda i: ratios[i])
    print(f"  ← {labels[best_r]}")

    # Objectifs Hedge Fund
    print(f"\n  🎯 Objectifs Hedge Fund (CAGR>40%, DD<10%, Sharpe>1.3, Sortino>0.8)")
    print("  " + "─" * 75)
    targets = [
        ("CAGR > 40%", "cagr", lambda v: v > 0.40),
        ("Max DD < 10%", "max_drawdown", lambda v: v > -0.10),
        ("Sharpe > 1.3", "sharpe", lambda v: v > 1.3),
        ("Sortino > 0.8", "sortino", lambda v: v > 0.8),
    ]
    for label, key, check in targets:
        val = m_hf.get(key, 0)
        ok = check(val)
        emoji = "✅" if ok else "❌"
        fmt_val = f"{val:+.1%}" if "return" in key or key in ("cagr", "max_drawdown") else f"{val:.2f}"
        print(f"    {emoji} {label:20s} → {fmt_val}")

    # Pire mois
    monthly_hf = compute_monthly_returns(res_hf.equity_curve)
    if monthly_hf:
        worst_month = min(monthly_hf, key=lambda x: x[1])
        best_month = max(monthly_hf, key=lambda x: x[1])
        neg_months = [m for m in monthly_hf if m[1] < 0]
        bad_months = [m for m in monthly_hf if m[1] < -0.08]
        print(f"\n  📅 Analyse mensuelle HedgeFund")
        print("  " + "─" * 75)
        print(f"    Pire mois          : {worst_month[0]} ({worst_month[1]:+.1%})")
        print(f"    Meilleur mois      : {best_month[0]} ({best_month[1]:+.1%})")
        print(f"    Mois négatifs      : {len(neg_months)} / {len(monthly_hf)}")
        print(f"    Mois < -8%         : {len(bad_months)} {'✅' if len(bad_months) == 0 else '❌'}")

    # Risk Engine stats
    risk = engine_hf.risk
    print(f"\n  🔒 Risk Engine — Statistiques")
    print("  " + "─" * 75)
    print(f"    DD scaling activé  : {risk.n_dd_scaled} fois")
    print(f"    Vol scaling activé : {risk.n_vol_scaled} fois")
    print(f"    Corr réduite       : {risk.n_corr_reduced} fois")
    print(f"    Kill switch activé : {risk.n_kill_paused} fois")

    if risk.scale_history:
        avg_scale = sum(s[3] for s in risk.scale_history) / len(risk.scale_history)
        min_scale = min(s[3] for s in risk.scale_history)
        print(f"    Scale moyen        : {avg_scale:.2f}")
        print(f"    Scale minimum      : {min_scale:.2f}")

    # Mode breakdown
    classic_trades = [t for t in engine_hf.closed_trades if t.mode == "classic"]
    trail_trades = [t for t in engine_hf.closed_trades if t.mode == "trail"]
    print(f"\n  📊 Breakdown par mode")
    print("  " + "─" * 75)
    for mode_name, mode_trades in [("Classic (60%)", classic_trades), ("Trail (40%)", trail_trades)]:
        if not mode_trades:
            continue
        n = len(mode_trades)
        wins = sum(1 for t in mode_trades if t.pnl_usd > 0)
        total_pnl = sum(t.pnl_usd for t in mode_trades)
        wr = wins / n if n else 0
        print(f"    {mode_name:16s} : {n:4d} trades | WR {wr:.0%} | PnL ${total_pnl:+.2f}")

        exit_map: dict[str, int] = {}
        for t in mode_trades:
            exit_map[t.exit_reason] = exit_map.get(t.exit_reason, 0) + 1
        for reason, count in sorted(exit_map.items()):
            print(f"      {reason:16s} : {count}")

    # Monte Carlo
    print(f"\n  🎲 Monte Carlo (5000 simulations)")
    print("  " + "─" * 75)
    mc = monte_carlo_simulation(engine_hf.closed_trades, engine_hf.cfg.initial_balance)
    if mc:
        print(f"    Return  5ème %ile  : {mc['return_5th']:+.1%}")
        print(f"    Return médian      : {mc['return_50th']:+.1%}")
        print(f"    Return 95ème %ile  : {mc['return_95th']:+.1%}")
        print(f"    MaxDD  5ème %ile   : {mc['dd_5th']:.1%}")
        print(f"    MaxDD médian       : {mc['dd_50th']:.1%}")
        print(f"    MaxDD 95ème %ile   : {mc['dd_95th']:.1%}  {'✅' if mc['dd_95th'] < 0.10 else '⚠️'}")
        print(f"    Losing streak méd. : {mc['streak_50th']}")
        print(f"    Losing streak 95%  : {mc['streak_95th']}")

        # Sizing recommendation
        if mc['dd_95th'] > 0:
            safe_risk = 0.02 * (0.10 / mc['dd_95th'])
            print(f"\n    💡 Pour MaxDD 95% < 10% → risk_percent = {safe_risk:.3f}")

    print(f"\n{sep}\n")


# ═══════════════════ Combine equity (pour 60/40 simple) ═════════════════════

def combine_equity(eq_a: list[EquityPoint], bal_a: float,
                   eq_b: list[EquityPoint], bal_b: float) -> list[EquityPoint]:
    ts_a = {e.timestamp: e.equity for e in eq_a}
    ts_b = {e.timestamp: e.equity for e in eq_b}
    all_ts = sorted(set(ts_a.keys()) | set(ts_b.keys()))
    combined = []
    la, lb = bal_a, bal_b
    for ts in all_ts:
        la = ts_a.get(ts, la)
        lb = ts_b.get(ts, lb)
        combined.append(EquityPoint(ts, la + lb))
    return combined


# ═══════════════════ Trail@TP Engine (simplifié) ════════════════════════════

class SimpleTrailEngine:
    """Trail@TP sans risk overlay."""
    def __init__(self, candles: dict[str, list[Candle]], balance: float,
                 step_pct: float, cfg_overrides: dict, label: str = "Trail") -> None:
        self.label = label
        self.candles = candles
        self.pairs = list(candles.keys())
        self.balance = balance
        self.cash = balance
        self.step_pct = step_pct
        self.cfg = cfg_overrides
        self.positions: list[HFPosition] = []
        self.closed_trades: list[HFTrade] = []
        self.equity_curve: list[EquityPoint] = []
        self.trends: dict[str, Optional[TrendState]] = {p: None for p in self.pairs}
        self.ranges: dict[str, Optional[RangeState]] = {p: None for p in self.pairs}
        self.cooldown_until: dict[str, int] = {p: 0 for p in self.pairs}
        self._pending: dict[str, Optional[dict]] = {p: None for p in self.pairs}
        self.last_close: dict[str, float] = {}
        self._idx: dict[tuple[str, int], Candle] = {}
        for sym, clist in candles.items():
            for c in clist:
                self._idx[(sym, c.timestamp)] = c

    def run(self) -> BacktestResult:
        for mod in ("src.core.swing_detector", "src.core.trend_engine",
                     "src.core.strategy_mean_rev"):
            logging.getLogger(mod).setLevel(logging.WARNING)
        timeline = sorted({c.timestamp for cl in self.candles.values() for c in cl})
        total = len(timeline)
        for i, ts in enumerate(timeline):
            for sym in self.pairs:
                c = self._idx.get((sym, ts))
                if c:
                    self.last_close[sym] = c.close
            self._exits(ts)
            self._entries(ts)
            self._analyze(ts)
            unrealized = sum(p.size * self.last_close.get(p.symbol, p.entry_price) for p in self.positions)
            eq = self.cash + unrealized
            self.equity_curve.append(EquityPoint(ts, eq))
            if (i+1) % 500 == 0 or i == total - 1:
                print(f"\r   ⏳ [{self.label}] {i+1}/{total} ({100*(i+1)/total:.0f}%) "
                      f"| Eq: ${eq:,.0f}", end="", flush=True)
        print()
        for pos in list(self.positions):
            self._close(pos, self.last_close.get(pos.symbol, pos.entry_price), timeline[-1], "END")
        trades = [Trade(symbol=t.symbol, strategy=StrategyType.RANGE, side=t.side,
                        entry_price=t.entry_price, exit_price=t.exit_price, size=t.size,
                        entry_time=t.entry_time, exit_time=t.exit_time,
                        pnl_usd=t.pnl_usd, pnl_pct=t.pnl_pct, exit_reason=t.exit_reason)
                  for t in self.closed_trades]
        final_eq = self.equity_curve[-1].equity if self.equity_curve else self.cash
        return BacktestResult(trades=trades, equity_curve=self.equity_curve,
                              initial_balance=self.balance, final_equity=final_eq,
                              start_date=datetime.fromtimestamp(timeline[0]/1000, tz=timezone.utc),
                              end_date=datetime.fromtimestamp(timeline[-1]/1000, tz=timezone.utc),
                              pairs=self.pairs)

    def _exits(self, ts):
        to_close = []
        for pos in self.positions:
            c = self._idx.get((pos.symbol, ts))
            if not c: continue
            pos.bars_count += 1
            if c.high > pos.highest_since_entry: pos.highest_since_entry = c.high
            if c.low <= pos.current_sl:
                reason = "TP_TRAIL" if pos.trailing_active else "RANGE_SL"
                to_close.append((pos, pos.current_sl, reason))
                if not pos.trailing_active:
                    self.cooldown_until[pos.symbol] = ts + self.cfg.get("cooldown_bars", 3)*4*3600*1000
                continue
            if not pos.trailing_active and c.high >= pos.tp_level:
                pos.trailing_active = True
                pos.current_sl = pos.tp_level
                pos.steps_completed = 1
            if pos.trailing_active:
                ns = pos.steps_completed + 1
                nt = pos.tp_level * (1 + self.step_pct * (ns - 1))
                while c.high >= nt:
                    pos.steps_completed = ns
                    pos.current_sl = pos.tp_level * (1 + self.step_pct * (ns - 1))
                    ns = pos.steps_completed + 1
                    nt = pos.tp_level * (1 + self.step_pct * (ns - 1))
        for pos, px, reason in to_close:
            self._close(pos, px, ts, reason)

    def _entries(self, ts):
        if len({p.symbol for p in self.positions}) >= 3: return
        for sym in self.pairs:
            if len({p.symbol for p in self.positions}) >= 3: break
            if any(p.symbol == sym for p in self.positions): continue
            sig = self._pending.get(sym)
            if not sig: continue
            c = self._idx.get((sym, ts))
            if not c: continue
            if ts < self.cooldown_until.get(sym, 0): continue
            if c.open > sig["buy_zone"]: continue
            entry = c.open * 1.001
            if sig["sl_price"] >= entry or sig["tp_price"] <= entry: continue
            size = calculate_position_size(self.balance, 0.02, entry, sig["sl_price"], 0.30)
            if size <= 0: continue
            cost = size * entry * 1.00075
            if cost > self.cash:
                size = self.cash / (entry * 1.00075)
                cost = size * entry * 1.00075
            if size <= 0 or cost > self.cash: continue
            self.cash -= cost
            self.positions.append(HFPosition(symbol=sym, side=OrderSide.BUY, mode="trail",
                entry_price=entry, sl_price=sig["sl_price"], tp_level=sig["tp_price"],
                current_sl=sig["sl_price"], size=size, entry_time=ts, highest_since_entry=entry))
            self._pending[sym] = None

    def _analyze(self, ts):
        for sym in self.pairs:
            vis = [c for c in self.candles[sym] if c.timestamp <= ts][-100:]
            if len(vis) < 7: continue
            swings = detect_swings(vis, 3)
            if not swings: continue
            trend = determine_trend(swings, sym)
            self.trends[sym] = trend
            if trend.direction != TrendDirection.NEUTRAL:
                self.ranges[sym] = None; self._pending[sym] = None
                for pos in list(self.positions):
                    if pos.symbol == sym:
                        self._close(pos, self.last_close.get(sym, pos.entry_price), ts, "TREND_BREAK")
                continue
            rs = build_range_from_trend(trend, 0.02)
            if not rs: self._pending[sym] = None; continue
            self.ranges[sym] = rs
            self._pending[sym] = {"side": OrderSide.BUY,
                "buy_zone": rs.range_low * 1.002, "sl_price": rs.range_low * 0.997, "tp_price": rs.range_mid}

    def _close(self, pos, exit_px, ts, reason):
        revenue = pos.size * exit_px * (1 - 0.00075)
        pnl = pos.size * (exit_px * (1-0.00075) - pos.entry_price * (1+0.00075))
        self.cash += revenue
        pnl_pct = pnl / (pos.size * pos.entry_price) if pos.entry_price else 0
        self.closed_trades.append(HFTrade(symbol=pos.symbol, side=pos.side,
            entry_price=pos.entry_price, exit_price=exit_px, size=pos.size,
            entry_time=pos.entry_time, exit_time=ts, pnl_usd=pnl, pnl_pct=pnl_pct,
            exit_reason=reason, mode="trail", steps_reached=pos.steps_completed,
            bars_held=pos.bars_count, tp_level=pos.tp_level))
        if pos in self.positions: self.positions.remove(pos)


# ═══════════════════ Main ═══════════════════════════════════════════════════

PAIRS_20 = [
    "BTC-USD", "ETH-USD", "SOL-USD", "XRP-USD", "LINK-USD",
    "ADA-USD", "DOT-USD", "AVAX-USD", "DOGE-USD", "ATOM-USD",
    "UNI-USD", "NEAR-USD", "ALGO-USD", "LTC-USD", "ETC-USD",
    "FIL-USD", "AAVE-USD", "INJ-USD", "SAND-USD", "MANA-USD",
]


def main() -> None:
    parser = argparse.ArgumentParser(description="Hedge Fund Architecture backtest")
    parser.add_argument("--balance", type=float, default=1000.0)
    parser.add_argument("--start", default="2022-02-20")
    parser.add_argument("--end", default="2026-02-20")
    parser.add_argument("--pairs", type=int, default=20)
    parser.add_argument("--step-pct", type=float, default=0.01)
    parser.add_argument("--target-vol", type=float, default=0.25)
    parser.add_argument("--no-show", action="store_true")
    args = parser.parse_args()

    start = datetime.strptime(args.start, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    end = datetime.strptime(args.end, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    pairs = PAIRS_20[:args.pairs]

    print(f"\n{'═'*95}")
    print(f"  🏦 HEDGE FUND ARCHITECTURE — Backtest complet")
    print(f"  📅 {start:%Y-%m-%d} → {end:%Y-%m-%d} | {len(pairs)} paires | ${args.balance:,.0f}")
    print(f"  60% Classic / 40% Trail@TP | Risk Engine: DD scaling + Vol target + Kill switch + Corr")
    print(f"{'═'*95}\n")

    logger.info("📥 Téléchargement des données…")
    candles = download_all_pairs(pairs, start, end, interval="4h")
    from src import config

    # ── 1) 100% Classic ──
    print("\n" + "─" * 60)
    print("  🏛️  100% CLASSIQUE")
    print("─" * 60)
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
    res_classic = engine_classic.run()
    m_classic = compute_metrics(res_classic)

    # ── 2) 60/40 Simple (sans risk overlay) ──
    print("\n" + "─" * 60)
    print("  🔀 60/40 SIMPLE (sans risk overlay)")
    print("─" * 60)

    # Classic part (60%)
    cfg_c60 = BacktestConfig(
        initial_balance=args.balance * 0.60, risk_percent_range=0.02,
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
    eng_c60 = BacktestEngine(candles, cfg_c60)
    res_c60 = eng_c60.run()

    # Trail part (40%)
    eng_t40 = SimpleTrailEngine(candles, args.balance * 0.40, args.step_pct,
                                 {"cooldown_bars": 3}, "60/40 Trail part")
    res_t40 = eng_t40.run()

    # Combine
    eq_simple = combine_equity(res_c60.equity_curve, args.balance * 0.60,
                                res_t40.equity_curve, args.balance * 0.40)
    trades_simple = sorted(res_c60.trades + res_t40.trades, key=lambda t: t.entry_time)
    res_simple = BacktestResult(
        trades=trades_simple, equity_curve=eq_simple,
        initial_balance=args.balance,
        final_equity=eq_simple[-1].equity if eq_simple else args.balance,
        start_date=res_c60.start_date, end_date=res_c60.end_date, pairs=pairs,
    )
    m_simple = compute_metrics(res_simple)

    # ── 3) 60/40 Hedge Fund ──
    print("\n" + "─" * 60)
    print("  🏦 60/40 HEDGE FUND (risk overlay complet)")
    print("─" * 60)

    cfg_hf = HedgeFundConfig(
        initial_balance=args.balance, step_pct=args.step_pct,
        range_entry_buffer_pct=getattr(config, "RANGE_ENTRY_BUFFER_PERCENT", 0.002),
        range_sl_buffer_pct=getattr(config, "RANGE_SL_BUFFER_PERCENT", 0.003),
        range_cooldown_bars=getattr(config, "RANGE_COOLDOWN_BARS", 3),
        swing_lookback=getattr(config, "SWING_LOOKBACK", 3),
        range_width_min=getattr(config, "RANGE_WIDTH_MIN", 0.02),
        target_vol_annual=args.target_vol,
        risk_overlay=True,
    )
    engine_hf = HedgeFundEngine(candles, cfg_hf, "HedgeFund")
    res_hf = engine_hf.run()
    m_hf = compute_metrics(res_hf)

    # ── Rapport ──
    print_report(m_classic, res_classic, m_simple, res_simple,
                 m_hf, res_hf, engine_hf)


if __name__ == "__main__":
    main()
