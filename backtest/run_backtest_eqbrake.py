#!/usr/bin/env python
"""
Backtest "Equity Brake" v2 — Architecture correcte.

ARCHITECTURE :
  - Deux moteurs INDÉPENDANTS (Classic 60% + Trail 40%) — comme le Simple
  - Un EquityBrake PARTAGÉ qui surveille l'equity COMBINÉE
  - Quand le DD depuis peak dépasse les seuils → scale les DEUX moteurs

Les moteurs tournent en lockstep bar par bar.
Après chaque barre, l'equity combinée est calculée et le brake mis à jour.
Avant chaque entrée, le moteur consulte le brake pour le scale.

Seuils calibrés sur le DD naturel du 60/40 Simple (~7.1% backtest, ~11% MC):
  Soft  : DD ≥ 9% → taille -30%   (au-dessus du DD normal)
  Hard  : DD ≥ 12% → taille -60%  (scénario extrême)
  Kill  : DD ≥ 15% → pause 48h    (catastrophe / bug)
"""

from __future__ import annotations

import argparse
import logging
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
logger = logging.getLogger("equity_brake")


# ═══════════════════ Equity Brake ═══════════════════════════════════════════

class EquityBrake:
    """Protection par drawdown depuis equity peak.
    Rarement déclenché. Violent quand déclenché. Jamais micro-géré."""

    def __init__(self, initial_balance: float,
                 soft_dd: float = 0.09, soft_reduction: float = 0.70,
                 hard_dd: float = 0.12, hard_reduction: float = 0.40,
                 kill_dd: float = 0.15, kill_pause_hours: int = 48) -> None:
        self.initial_balance = initial_balance
        self.peak_equity: float = initial_balance
        self.current_equity: float = initial_balance
        self.paused_until: int = 0

        self.soft_dd = soft_dd
        self.soft_reduction = soft_reduction
        self.hard_dd = hard_dd
        self.hard_reduction = hard_reduction
        self.kill_dd = kill_dd
        self.kill_pause_hours = kill_pause_hours

        # Stats
        self.n_soft: int = 0
        self.n_hard: int = 0
        self.n_kill: int = 0
        self.n_entries_scaled: int = 0
        self.n_entries_blocked: int = 0
        self.dd_history: list[float] = []

    def update_equity(self, equity: float) -> None:
        self.current_equity = equity
        if equity > self.peak_equity:
            self.peak_equity = equity
        self.dd_history.append(self.current_dd())

    def current_dd(self) -> float:
        if self.peak_equity <= 0:
            return 0.0
        return (self.peak_equity - self.current_equity) / self.peak_equity

    def is_paused(self, ts: int) -> bool:
        return ts < self.paused_until

    def get_scale(self, ts: int) -> float:
        """Retourne le multiplicateur de taille (1.0 = normal)."""
        if self.is_paused(ts):
            return 0.0

        dd = self.current_dd()

        if dd >= self.kill_dd:
            self.paused_until = ts + self.kill_pause_hours * 3600 * 1000
            self.n_kill += 1
            self.n_entries_blocked += 1
            return 0.0

        if dd >= self.hard_dd:
            self.n_hard += 1
            self.n_entries_scaled += 1
            return self.hard_reduction

        if dd >= self.soft_dd:
            self.n_soft += 1
            self.n_entries_scaled += 1
            return self.soft_reduction

        return 1.0

    def summary(self) -> str:
        max_dd = max(self.dd_history) if self.dd_history else 0
        return (
            f"Peak DD: {max_dd:.1%} | "
            f"Entries scaled: {self.n_entries_scaled} | blocked: {self.n_entries_blocked} | "
            f"Triggers — soft: {self.n_soft} hard: {self.n_hard} kill: {self.n_kill}"
        )


# ═══════════════════ Position model ═════════════════════════════════════════

@dataclass
class Position:
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
    scale_applied: float = 1.0


@dataclass
class ClosedTrade:
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
    mode: str = "classic"
    steps_reached: int = 0
    bars_held: int = 0
    scale_applied: float = 1.0


# ═══════════════════ Independent Range Engine ═══════════════════════════════

class IndependentRangeEngine:
    """Moteur Range indépendant (Classic ou Trail).
    Gère son propre cash, positions, cooldowns.
    Consulte un EquityBrake externe pour le scaling."""

    def __init__(self, candles: dict[str, list[Candle]], balance: float,
                 mode: str, step_pct: float = 0.01,
                 brake: Optional[EquityBrake] = None,
                 label: str = "") -> None:
        self.label = label
        self.candles = candles
        self.pairs = list(candles.keys())
        self.initial_balance = balance
        self.cash = balance
        self.mode = mode
        self.step_pct = step_pct
        self.brake = brake

        self.positions: list[Position] = []
        self.closed_trades: list[ClosedTrade] = []
        self.equity_history: list[tuple[int, float]] = []

        self.trends: dict[str, Optional[TrendState]] = {p: None for p in self.pairs}
        self.ranges: dict[str, Optional[RangeState]] = {p: None for p in self.pairs}
        self.cooldown_until: dict[str, int] = {p: 0 for p in self.pairs}
        self._pending: dict[str, Optional[dict]] = {p: None for p in self.pairs}
        self.last_close: dict[str, float] = {}

        self._idx: dict[tuple[str, int], Candle] = {}
        for sym, clist in candles.items():
            for c in clist:
                self._idx[(sym, c.timestamp)] = c

        # Config
        self.risk_percent = 0.02
        self.max_position_pct = 0.30
        self.max_simultaneous = 3
        self.fee_pct = 0.00075
        self.slippage_pct = 0.001
        self.swing_lookback = 3
        self.candle_window = 100
        self.range_width_min = 0.02
        self.entry_buffer_pct = 0.002
        self.sl_buffer_pct = 0.003
        self.cooldown_bars = 3

    def step(self, ts: int) -> float:
        """Process one bar. Returns current equity."""
        for sym in self.pairs:
            c = self._idx.get((sym, ts))
            if c:
                self.last_close[sym] = c.close

        self._manage_exits(ts)
        self._execute_pending(ts)
        self._analyze(ts)
        eq = self._equity()
        self.equity_history.append((ts, eq))
        return eq

    def _equity(self) -> float:
        unrealized = sum(
            p.size * self.last_close.get(p.symbol, p.entry_price)
            for p in self.positions
        )
        return self.cash + unrealized

    def _active_symbols(self) -> set[str]:
        return {p.symbol for p in self.positions}

    # ── Exits ──

    def _manage_exits(self, ts: int) -> None:
        to_close: list[tuple[Position, float, str]] = []

        for pos in self.positions:
            c = self._idx.get((pos.symbol, ts))
            if c is None:
                continue

            pos.bars_count += 1
            if c.high > pos.highest_since_entry:
                pos.highest_since_entry = c.high

            # SL hit
            if c.low <= pos.current_sl:
                if pos.mode == "trail" and pos.trailing_active:
                    reason = "TP_TRAIL"
                else:
                    reason = "RANGE_SL"
                to_close.append((pos, pos.current_sl, reason))
                if not pos.trailing_active:
                    self.cooldown_until[pos.symbol] = (
                        ts + self.cooldown_bars * 4 * 3600 * 1000
                    )
                continue

            # TP hit
            if c.high >= pos.tp_level:
                if pos.mode == "classic":
                    to_close.append((pos, pos.tp_level, "RANGE_TP"))
                    continue
                elif pos.mode == "trail" and not pos.trailing_active:
                    pos.trailing_active = True
                    pos.current_sl = pos.tp_level
                    pos.steps_completed = 1

            # Step trailing (trail mode)
            if pos.mode == "trail" and pos.trailing_active:
                ns = pos.steps_completed + 1
                nt = pos.tp_level * (1 + self.step_pct * (ns - 1))
                while c.high >= nt:
                    pos.steps_completed = ns
                    pos.current_sl = pos.tp_level * (1 + self.step_pct * (ns - 1))
                    ns = pos.steps_completed + 1
                    nt = pos.tp_level * (1 + self.step_pct * (ns - 1))

        for pos, exit_px, reason in to_close:
            self._close_position(pos, exit_px, ts, reason)

    # ── Entries ──

    def _execute_pending(self, ts: int) -> None:
        if len(self._active_symbols()) >= self.max_simultaneous:
            return

        # Check brake
        scale = 1.0
        if self.brake:
            scale = self.brake.get_scale(ts)
            if scale <= 0:
                return

        for sym in self.pairs:
            if len(self._active_symbols()) >= self.max_simultaneous:
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

            entry_price = c.open * (1 + self.slippage_pct)
            if sig["sl_price"] >= entry_price:
                continue
            if sig["tp_price"] <= entry_price:
                continue

            base_size = calculate_position_size(
                account_balance=self.initial_balance,
                risk_percent=self.risk_percent,
                entry_price=entry_price,
                sl_price=sig["sl_price"],
                max_position_percent=self.max_position_pct,
            )
            if base_size <= 0:
                continue

            # Apply brake scale
            size = base_size * scale

            cost = size * entry_price * (1 + self.fee_pct)
            if cost > self.cash:
                size = self.cash / (entry_price * (1 + self.fee_pct))
                cost = size * entry_price * (1 + self.fee_pct)
            if size <= 0 or cost > self.cash:
                continue

            self.cash -= cost
            self.positions.append(Position(
                symbol=sym, side=OrderSide.BUY, mode=self.mode,
                entry_price=entry_price, sl_price=sig["sl_price"],
                tp_level=sig["tp_price"], current_sl=sig["sl_price"],
                size=size, entry_time=ts,
                highest_since_entry=entry_price,
                scale_applied=scale,
            ))
            self._pending[sym] = None

    # ── Analysis ──

    def _analyze(self, ts: int) -> None:
        for sym in self.pairs:
            vis = [c for c in self.candles[sym] if c.timestamp <= ts]
            vis = vis[-self.candle_window:]
            if len(vis) < 2 * self.swing_lookback + 1:
                continue
            swings = detect_swings(vis, self.swing_lookback)
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

            rs = build_range_from_trend(trend, self.range_width_min)
            if rs is None:
                self._pending[sym] = None
                continue
            self.ranges[sym] = rs
            buy_zone = rs.range_low * (1 + self.entry_buffer_pct)
            sl_price = rs.range_low * (1 - self.sl_buffer_pct)
            self._pending[sym] = {
                "side": OrderSide.BUY,
                "buy_zone": buy_zone,
                "sl_price": sl_price,
                "tp_price": rs.range_mid,
            }

    # ── Close ──

    def _close_position(self, pos: Position, exit_price: float,
                        ts: int, reason: str) -> None:
        revenue = pos.size * exit_price * (1 - self.fee_pct)
        pnl = pos.size * (
            exit_price * (1 - self.fee_pct)
            - pos.entry_price * (1 + self.fee_pct)
        )
        self.cash += revenue
        pnl_pct = pnl / (pos.size * pos.entry_price) if pos.entry_price else 0

        self.closed_trades.append(ClosedTrade(
            symbol=pos.symbol, side=pos.side,
            entry_price=pos.entry_price, exit_price=exit_price,
            size=pos.size, entry_time=pos.entry_time, exit_time=ts,
            pnl_usd=pnl, pnl_pct=pnl_pct, exit_reason=reason,
            mode=pos.mode, steps_reached=pos.steps_completed,
            bars_held=pos.bars_count, scale_applied=pos.scale_applied,
        ))
        if pos in self.positions:
            self.positions.remove(pos)

    def close_remaining(self, ts: int) -> None:
        for pos in list(self.positions):
            px = self.last_close.get(pos.symbol, pos.entry_price)
            self._close_position(pos, px, ts, "END")


# ═══════════════════ Orchestrator ═══════════════════════════════════════════

class DualEngineOrchestrator:
    """Fait tourner Classic + Trail en lockstep.
    L'EquityBrake surveille l'equity combinée."""

    def __init__(self, candles: dict[str, list[Candle]],
                 balance: float, step_pct: float,
                 classic_pct: float = 0.60, trail_pct: float = 0.40,
                 soft_dd: float = 0.09, hard_dd: float = 0.12,
                 kill_dd: float = 0.15, use_brake: bool = True) -> None:

        self.balance = balance
        self.classic_pct = classic_pct
        self.trail_pct = trail_pct
        self.candles = candles
        self.pairs = list(candles.keys())

        bal_c = balance * classic_pct
        bal_t = balance * trail_pct

        self.brake: Optional[EquityBrake] = None
        if use_brake:
            self.brake = EquityBrake(
                initial_balance=balance,
                soft_dd=soft_dd, hard_dd=hard_dd, kill_dd=kill_dd,
            )

        self.engine_classic = IndependentRangeEngine(
            candles, bal_c, mode="classic", step_pct=step_pct,
            brake=self.brake, label="Classic",
        )
        self.engine_trail = IndependentRangeEngine(
            candles, bal_t, mode="trail", step_pct=step_pct,
            brake=self.brake, label="Trail",
        )

        self.combined_equity: list[EquityPoint] = []

    def run(self, label: str = "EqBrake") -> BacktestResult:
        for mod in ("src.core.swing_detector", "src.core.trend_engine",
                     "src.core.strategy_mean_rev"):
            logging.getLogger(mod).setLevel(logging.WARNING)

        timeline = self._build_timeline()
        total = len(timeline)

        for i, ts in enumerate(timeline):
            eq_c = self.engine_classic.step(ts)
            eq_t = self.engine_trail.step(ts)
            combined = eq_c + eq_t

            self.combined_equity.append(EquityPoint(ts, combined))

            if self.brake:
                self.brake.update_equity(combined)

            if (i + 1) % 500 == 0 or i == total - 1:
                dd = self.brake.current_dd() if self.brake else 0
                n_trades = (len(self.engine_classic.closed_trades)
                            + len(self.engine_trail.closed_trades))
                print(f"\r   ⏳ [{label}] {i+1}/{total} ({100*(i+1)/total:.0f}%) "
                      f"| Eq: ${combined:,.0f} | DD: {dd:.1%} | Trades: {n_trades}",
                      end="", flush=True)
        print()

        last_ts = timeline[-1]
        self.engine_classic.close_remaining(last_ts)
        self.engine_trail.close_remaining(last_ts)

        all_trades = sorted(
            self.engine_classic.closed_trades + self.engine_trail.closed_trades,
            key=lambda t: t.entry_time,
        )
        trades = [
            Trade(
                symbol=t.symbol, strategy=StrategyType.RANGE, side=t.side,
                entry_price=t.entry_price, exit_price=t.exit_price, size=t.size,
                entry_time=t.entry_time, exit_time=t.exit_time,
                pnl_usd=t.pnl_usd, pnl_pct=t.pnl_pct, exit_reason=t.exit_reason,
            )
            for t in all_trades
        ]
        final_eq = self.combined_equity[-1].equity if self.combined_equity else self.balance

        return BacktestResult(
            trades=trades, equity_curve=self.combined_equity,
            initial_balance=self.balance, final_equity=final_eq,
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


# ═══════════════════ Monthly + Monte Carlo ══════════════════════════════════

def compute_monthly_returns(equity_curve: list[EquityPoint]) -> list[tuple[str, float]]:
    if not equity_curve:
        return []
    monthly: dict[str, tuple[float, float]] = {}
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


def monte_carlo(trades: list[ClosedTrade], initial_balance: float,
                n_sims: int = 5000) -> dict:
    if not trades:
        return {}
    pnls = [t.pnl_usd for t in trades]
    n = len(pnls)
    finals, dds = [], []
    random.seed(42)
    for _ in range(n_sims):
        sample = random.choices(pnls, k=n)
        eq = initial_balance
        peak = eq
        max_dd = 0.0
        for pnl in sample:
            eq += pnl
            if eq > peak:
                peak = eq
            dd = (peak - eq) / peak if peak > 0 else 0
            if dd > max_dd:
                max_dd = dd
        finals.append(eq / initial_balance - 1)
        dds.append(max_dd)
    finals.sort()
    dds.sort()
    p = lambda lst, pct: lst[int(pct * len(lst))]
    return {
        "ret_5": p(finals, 0.05), "ret_50": p(finals, 0.50), "ret_95": p(finals, 0.95),
        "dd_5": p(dds, 0.05), "dd_50": p(dds, 0.50), "dd_95": p(dds, 0.95),
    }


# ═══════════════════ Rapport ════════════════════════════════════════════════

def print_report(
    m_classic: dict, res_classic: BacktestResult,
    m_simple: dict, res_simple: BacktestResult,
    m_brake: dict, res_brake: BacktestResult,
    orchestrator: DualEngineOrchestrator,
) -> None:
    sep = "═" * 100
    brake = orchestrator.brake
    print(f"\n{sep}")
    print(f"  🏦 EQUITY BRAKE v2 — Comparaison complète")
    print(f"  📅 {res_classic.start_date:%b %Y} → {res_classic.end_date:%b %Y}")
    print(f"  🪙 {len(res_classic.pairs)} paires | Capital: ${res_classic.initial_balance:,.0f}")
    if brake:
        print(f"  Brake: soft={brake.soft_dd:.0%} hard={brake.hard_dd:.0%} "
              f"kill={brake.kill_dd:.0%}")
    print(f"  Architecture: 2 moteurs indépendants + brake sur equity combinée")
    print(sep)

    labels = ["100% Classic", "60/40 Simple", "60/40 EqBrake"]
    all_m = [m_classic, m_simple, m_brake]

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

    # Delta vs Simple
    ret_simple = m_simple.get("total_return", 0)
    ret_brake = m_brake.get("total_return", 0)
    dd_simple = m_simple.get("max_drawdown", 0)
    dd_brake = m_brake.get("max_drawdown", 0)
    print(f"\n  📐 Delta EqBrake vs Simple")
    print("  " + "─" * 80)
    print(f"    Return : {ret_brake - ret_simple:+.1%} pts")
    print(f"    MaxDD  : {dd_brake - dd_simple:+.1%} pts (négatif = mieux protégé)")
    print(f"    Sharpe : {m_brake.get('sharpe',0) - m_simple.get('sharpe',0):+.2f}")

    # Equity Brake stats
    if brake:
        print(f"\n  🔒 Equity Brake — Statistiques")
        print("  " + "─" * 80)
        print(f"    {brake.summary()}")
        n_bars = len(brake.dd_history)
        if n_bars:
            pct_normal = sum(1 for d in brake.dd_history if d < brake.soft_dd) / n_bars * 100
            pct_soft = sum(1 for d in brake.dd_history
                          if brake.soft_dd <= d < brake.hard_dd) / n_bars * 100
            pct_hard = sum(1 for d in brake.dd_history
                          if brake.hard_dd <= d < brake.kill_dd) / n_bars * 100
            pct_kill = sum(1 for d in brake.dd_history if d >= brake.kill_dd) / n_bars * 100
            print(f"    Temps normal (<{brake.soft_dd:.0%} DD) : {pct_normal:.1f}%")
            print(f"    Temps soft   ({brake.soft_dd:.0%}-{brake.hard_dd:.0%}) : {pct_soft:.1f}%")
            print(f"    Temps hard   ({brake.hard_dd:.0%}-{brake.kill_dd:.0%}) : {pct_hard:.1f}%")
            print(f"    Temps kill   (≥{brake.kill_dd:.0%})  : {pct_kill:.1f}%")

    # Monthly EqBrake
    monthly = compute_monthly_returns(res_brake.equity_curve)
    if monthly:
        worst = min(monthly, key=lambda x: x[1])
        best_m = max(monthly, key=lambda x: x[1])
        neg = sum(1 for m in monthly if m[1] < 0)
        bad = sum(1 for m in monthly if m[1] < -0.08)
        print(f"\n  📅 Analyse mensuelle EqBrake")
        print("  " + "─" * 80)
        print(f"    Pire mois       : {worst[0]} ({worst[1]:+.1%})")
        print(f"    Meilleur mois   : {best_m[0]} ({best_m[1]:+.1%})")
        print(f"    Mois négatifs   : {neg} / {len(monthly)}")
        print(f"    Mois < -8%      : {bad} {'✅' if bad == 0 else '❌'}")

    # Mode breakdown
    all_ct = orchestrator.engine_classic.closed_trades
    all_tt = orchestrator.engine_trail.closed_trades
    print(f"\n  📊 Breakdown par mode")
    print("  " + "─" * 80)
    for name, tlist in [("Classic (60%)", all_ct), ("Trail (40%)", all_tt)]:
        if not tlist:
            continue
        n = len(tlist)
        wins = sum(1 for t in tlist if t.pnl_usd > 0)
        total_pnl = sum(t.pnl_usd for t in tlist)
        wr = wins / n if n else 0
        scaled = sum(1 for t in tlist if t.scale_applied < 1.0)
        print(f"    {name:16s} : {n:4d} trades | WR {wr:.0%} | PnL ${total_pnl:+.2f}"
              f" | {scaled} scaled by brake")
        exit_map: dict[str, int] = {}
        for t in tlist:
            exit_map[t.exit_reason] = exit_map.get(t.exit_reason, 0) + 1
        for reason, count in sorted(exit_map.items()):
            print(f"      {reason:16s} : {count}")

    # Monte Carlo
    all_trades = orchestrator.engine_classic.closed_trades + orchestrator.engine_trail.closed_trades
    print(f"\n  🎲 Monte Carlo (5000 simulations)")
    print("  " + "─" * 80)
    mc = monte_carlo(all_trades, orchestrator.balance)
    if mc:
        print(f"    Return  5ème %ile  : {mc['ret_5']:+.1%}")
        print(f"    Return médian      : {mc['ret_50']:+.1%}")
        print(f"    Return 95ème %ile  : {mc['ret_95']:+.1%}")
        print(f"    MaxDD  5ème %ile   : {mc['dd_5']:.1%}")
        print(f"    MaxDD médian       : {mc['dd_50']:.1%}")
        print(f"    MaxDD 95ème %ile   : {mc['dd_95']:.1%}  {'✅' if mc['dd_95'] < 0.10 else '⚠️'}")

    print(f"\n{sep}\n")


# ═══════════════════ Main ═══════════════════════════════════════════════════

PAIRS_20 = [
    "BTC-USD", "ETH-USD", "SOL-USD", "XRP-USD", "LINK-USD",
    "ADA-USD", "DOT-USD", "AVAX-USD", "DOGE-USD", "ATOM-USD",
    "UNI-USD", "NEAR-USD", "ALGO-USD", "LTC-USD", "ETC-USD",
    "FIL-USD", "AAVE-USD", "INJ-USD", "SAND-USD", "MANA-USD",
]


def main() -> None:
    parser = argparse.ArgumentParser(description="Equity Brake v2 backtest")
    parser.add_argument("--balance", type=float, default=1000.0)
    parser.add_argument("--start", default="2022-02-20")
    parser.add_argument("--end", default="2026-02-20")
    parser.add_argument("--pairs", type=int, default=20)
    parser.add_argument("--step-pct", type=float, default=0.01)
    parser.add_argument("--soft-dd", type=float, default=0.09)
    parser.add_argument("--hard-dd", type=float, default=0.12)
    parser.add_argument("--kill-dd", type=float, default=0.15)
    parser.add_argument("--no-show", action="store_true")
    args = parser.parse_args()

    start = datetime.strptime(args.start, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    end = datetime.strptime(args.end, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    pairs = PAIRS_20[:args.pairs]

    print(f"\n{'═'*100}")
    print(f"  🔒 EQUITY BRAKE v2 — 2 moteurs indépendants + brake sur equity combinée")
    print(f"  📅 {start:%Y-%m-%d} → {end:%Y-%m-%d} | {len(pairs)} paires | ${args.balance:,.0f}")
    print(f"  Soft: -{args.soft_dd:.0%} → -30% taille | "
          f"Hard: -{args.hard_dd:.0%} → -60% | "
          f"Kill: -{args.kill_dd:.0%} → pause 48h")
    print(f"{'═'*100}\n")

    logger.info("📥 Téléchargement des données…")
    candles = download_all_pairs(pairs, start, end, interval="4h")
    from src import config

    # ────────── 1) 100% Classic (référence) ──────────
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
    res_classic = BacktestEngine(candles, cfg_classic).run()
    m_classic = compute_metrics(res_classic)

    # ────────── 2) 60/40 Simple (sans brake) ──────────
    print("\n" + "─" * 60)
    print("  🔀 60/40 SIMPLE (sans brake)")
    print("─" * 60)
    orch_simple = DualEngineOrchestrator(
        candles, args.balance, args.step_pct,
        classic_pct=0.60, trail_pct=0.40,
        use_brake=False,
    )
    res_simple = orch_simple.run(label="Simple")
    m_simple = compute_metrics(res_simple)

    # ────────── 3) 60/40 Equity Brake ──────────
    print("\n" + "─" * 60)
    print("  🔒 60/40 EQUITY BRAKE")
    print("─" * 60)
    orch_brake = DualEngineOrchestrator(
        candles, args.balance, args.step_pct,
        classic_pct=0.60, trail_pct=0.40,
        soft_dd=args.soft_dd, hard_dd=args.hard_dd, kill_dd=args.kill_dd,
        use_brake=True,
    )
    res_brake = orch_brake.run(label="EqBrake")
    m_brake = compute_metrics(res_brake)

    # ────────── Rapport ──────────
    print_report(
        m_classic, res_classic,
        m_simple, res_simple,
        m_brake, res_brake,
        orch_brake,
    )


if __name__ == "__main__":
    main()
