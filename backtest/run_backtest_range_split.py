#!/usr/bin/env python
"""
Backtest : Split capital entre RANGE Classique et Trail@TP.

Teste des allocations mixtes :
  50/50 : 50% classique + 50% trail@TP
  60/40 : 60% classique + 40% trail@TP

L'idée : combiner le PF élevé du classique avec le return du trail.
"""

from __future__ import annotations

import argparse
import logging
import math
from dataclasses import dataclass
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
logger = logging.getLogger("range_split")


# ═══════════════════ Trail@TP Engine (copié du test précédent) ═══════════════

@dataclass
class TpTrailConfig:
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


@dataclass
class TpTrade:
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
class TpPosition:
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


class RangeTpTrailEngine:
    def __init__(self, candles_by_symbol: dict[str, list[Candle]],
                 config: TpTrailConfig, label: str = "Trail@TP") -> None:
        self.label = label
        self.candles = candles_by_symbol
        self.cfg = config
        self.pairs = list(candles_by_symbol.keys())
        self.cash = config.initial_balance
        self.positions: list[TpPosition] = []
        self.closed_trades: list[TpTrade] = []
        self.equity_curve: list[EquityPoint] = []
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

    def _manage_exits(self, ts: int) -> None:
        to_close: list[tuple[TpPosition, float, str]] = []
        for pos in self.positions:
            c = self._idx.get((pos.symbol, ts))
            if c is None:
                continue
            pos.bars_count += 1
            if c.high > pos.highest_since_entry:
                pos.highest_since_entry = c.high
            if c.low <= pos.current_sl:
                reason = "TP_TRAIL" if pos.trailing_active else "RANGE_SL"
                to_close.append((pos, pos.current_sl, reason))
                if not pos.trailing_active:
                    self.cooldown_until[pos.symbol] = (
                        ts + self.cfg.range_cooldown_bars * 4 * 3600 * 1000
                    )
                continue
            if not pos.trailing_active and c.high >= pos.tp_level:
                pos.trailing_active = True
                pos.current_sl = pos.tp_level
                pos.steps_completed = 1
            if pos.trailing_active:
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
            self.positions.append(TpPosition(
                symbol=sym, side=OrderSide.BUY,
                entry_price=entry_price, sl_price=sig["sl_price"],
                tp_level=sig["tp_price"], current_sl=sig["sl_price"],
                size=size, entry_time=ts,
                highest_since_entry=entry_price,
            ))
            self._pending[sym] = None

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

    def _close_position(self, pos: TpPosition, exit_price: float,
                        ts: int, reason: str) -> None:
        revenue = pos.size * exit_price * (1 - self.cfg.fee_pct)
        pnl = pos.size * (
            exit_price * (1 - self.cfg.fee_pct)
            - pos.entry_price * (1 + self.cfg.fee_pct)
        )
        self.cash += revenue
        pnl_pct = pnl / (pos.size * pos.entry_price) if pos.entry_price else 0
        self.closed_trades.append(TpTrade(
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


# ═══════════════════ Combinaison d'equity curves ════════════════════════════

def combine_equity_curves(
    eq_a: list[EquityPoint], bal_a: float,
    eq_b: list[EquityPoint], bal_b: float,
) -> list[EquityPoint]:
    """Combine deux equity curves en % returns additifs."""
    ts_a = {e.timestamp: e.equity for e in eq_a}
    ts_b = {e.timestamp: e.equity for e in eq_b}
    all_ts = sorted(set(ts_a.keys()) | set(ts_b.keys()))

    combined: list[EquityPoint] = []
    last_a = bal_a
    last_b = bal_b
    for ts in all_ts:
        ea = ts_a.get(ts, last_a)
        eb = ts_b.get(ts, last_b)
        last_a = ea
        last_b = eb
        combined.append(EquityPoint(ts, ea + eb))
    return combined


def combine_trades(trades_a: list[Trade], trades_b: list[Trade]) -> list[Trade]:
    """Merge deux listes de trades."""
    return sorted(trades_a + trades_b, key=lambda t: t.entry_time)


# ═══════════════════ Rapport ════════════════════════════════════════════════

def print_split_comparison(
    splits: dict[str, tuple[dict, BacktestResult]],
    m_classic_full: dict, res_classic_full: BacktestResult,
    m_trail_full: dict, res_trail_full: BacktestResult,
) -> None:
    sep = "═" * 110
    print(f"\n{sep}")
    print(f"  🔬 COMPARAISON SPLIT CAPITAL : Classique vs Trail@TP vs Mix")
    print(f"  📅 {res_classic_full.start_date:%b %Y} → {res_classic_full.end_date:%b %Y}")
    print(f"  🪙 {len(res_classic_full.pairs)} paires | Capital total: ${res_classic_full.initial_balance:,.0f}")
    print(sep)

    labels = ["100% Classic", "100% Trail"]
    all_m = [m_classic_full, m_trail_full]
    for name, (m, _) in splits.items():
        labels.append(name)
        all_m.append(m)

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
    hdr = f"  {'Métrique':<18s}"
    for lbl in labels:
        hdr += f" │ {lbl:>14s}"
    print(f"\n{hdr}")
    print("  " + "─" * (20 + 17 * len(labels)))

    for metric_label, key, fmt in metrics_list:
        row = f"  {metric_label:<18s}"
        vals = []
        for m in all_m:
            v = m.get(key, 0)
            vals.append(v)
            s = fmt.format(v) if isinstance(v, (int, float)) else str(v)
            row += f" │ {s:>14s}"

        # Best indicator
        if key == "max_drawdown":
            best_idx = max(range(len(vals)), key=lambda i: vals[i])
        elif key == "n_trades":
            best_idx = -1
        else:
            best_idx = max(range(len(vals)), key=lambda i: vals[i])

        if best_idx >= 0:
            row += f"  ← {labels[best_idx]}"
        print(row)

    # Bonus: Return/DD ratio
    print(f"\n  {'Return/MaxDD':<18s}", end="")
    for m in all_m:
        ret = m.get("total_return", 0)
        dd = abs(m.get("max_drawdown", -1))
        ratio = ret / dd if dd > 0 else 0
        print(f" │ {ratio:>14.2f}", end="")
    ratios = []
    for m in all_m:
        ret = m.get("total_return", 0)
        dd = abs(m.get("max_drawdown", -1))
        ratios.append(ret / dd if dd > 0 else 0)
    best_ratio = max(range(len(ratios)), key=lambda i: ratios[i])
    print(f"  ← {labels[best_ratio]}")

    print(f"\n{sep}\n")


# ═══════════════════ Main ═══════════════════════════════════════════════════

PAIRS_20 = [
    "BTC-USD", "ETH-USD", "SOL-USD", "XRP-USD", "LINK-USD",
    "ADA-USD", "DOT-USD", "AVAX-USD", "DOGE-USD", "ATOM-USD",
    "UNI-USD", "NEAR-USD", "ALGO-USD", "LTC-USD", "ETC-USD",
    "FIL-USD", "AAVE-USD", "INJ-USD", "SAND-USD", "MANA-USD",
]


def main() -> None:
    parser = argparse.ArgumentParser(description="Split capital Classic / Trail@TP")
    parser.add_argument("--balance", type=float, default=1000.0)
    parser.add_argument("--start", default="2022-02-20")
    parser.add_argument("--end", default="2026-02-20")
    parser.add_argument("--pairs", type=int, default=20)
    parser.add_argument("--step-pct", type=float, default=0.01)
    parser.add_argument("--no-show", action="store_true")
    args = parser.parse_args()

    start = datetime.strptime(args.start, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    end = datetime.strptime(args.end, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    pairs = PAIRS_20[:args.pairs]
    total_balance = args.balance

    print(f"\n{'═'*110}")
    print(f"  🔬 SPLIT CAPITAL — Classique vs Trail@TP vs Mix 50/50 vs Mix 60/40")
    print(f"  📅 {start:%Y-%m-%d} → {end:%Y-%m-%d} | {len(pairs)} paires | ${total_balance:,.0f}")
    print(f"{'═'*110}\n")

    logger.info("📥 Téléchargement des données…")
    candles = download_all_pairs(pairs, start, end, interval="4h")

    from src import config

    splits_to_test = [
        ("50C/50T", 0.50, 0.50),
        ("60C/40T", 0.60, 0.40),
    ]

    # ── 1) 100% Classique (full balance) ──
    print("\n" + "─" * 60)
    print(f"  🏛️  100% CLASSIQUE — ${total_balance:,.0f}")
    print("─" * 60)

    cfg_classic_full = BacktestConfig(
        initial_balance=total_balance, risk_percent_range=0.02,
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
    engine_full_classic = BacktestEngine(candles, cfg_classic_full)
    res_classic_full = engine_full_classic.run()
    m_classic_full = compute_metrics(res_classic_full)

    # ── 2) 100% Trail@TP (full balance) ──
    print("\n" + "─" * 60)
    print(f"  🚀 100% Trail@TP — ${total_balance:,.0f}")
    print("─" * 60)

    cfg_trail_full = TpTrailConfig(
        initial_balance=total_balance, step_pct=args.step_pct,
        range_entry_buffer_pct=getattr(config, "RANGE_ENTRY_BUFFER_PERCENT", 0.002),
        range_sl_buffer_pct=getattr(config, "RANGE_SL_BUFFER_PERCENT", 0.003),
        range_cooldown_bars=getattr(config, "RANGE_COOLDOWN_BARS", 3),
        swing_lookback=getattr(config, "SWING_LOOKBACK", 3),
        range_width_min=getattr(config, "RANGE_WIDTH_MIN", 0.02),
    )
    engine_full_trail = RangeTpTrailEngine(candles, cfg_trail_full, "100% Trail")
    res_trail_full = engine_full_trail.run()
    m_trail_full = compute_metrics(res_trail_full)

    # ── 3) Splits ──
    split_results: dict[str, tuple[dict, BacktestResult]] = {}

    for split_name, pct_classic, pct_trail in splits_to_test:
        bal_classic = total_balance * pct_classic
        bal_trail = total_balance * pct_trail

        print(f"\n{'─'*60}")
        print(f"  🔀 {split_name} — Classic ${bal_classic:,.0f} + Trail ${bal_trail:,.0f}")
        print("─" * 60)

        # Classic part
        cfg_c = BacktestConfig(
            initial_balance=bal_classic, risk_percent_range=0.02,
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
        eng_c = BacktestEngine(candles, cfg_c)
        res_c = eng_c.run()

        # Trail part
        cfg_t = TpTrailConfig(
            initial_balance=bal_trail, step_pct=args.step_pct,
            range_entry_buffer_pct=getattr(config, "RANGE_ENTRY_BUFFER_PERCENT", 0.002),
            range_sl_buffer_pct=getattr(config, "RANGE_SL_BUFFER_PERCENT", 0.003),
            range_cooldown_bars=getattr(config, "RANGE_COOLDOWN_BARS", 3),
            swing_lookback=getattr(config, "SWING_LOOKBACK", 3),
            range_width_min=getattr(config, "RANGE_WIDTH_MIN", 0.02),
        )
        eng_t = RangeTpTrailEngine(candles, cfg_t, f"{split_name} Trail")
        res_t = eng_t.run()

        # Combine
        combined_eq = combine_equity_curves(
            res_c.equity_curve, bal_classic,
            res_t.equity_curve, bal_trail,
        )
        combined_trades = combine_trades(res_c.trades, res_t.trades)
        final_eq = combined_eq[-1].equity if combined_eq else total_balance

        combined_result = BacktestResult(
            trades=combined_trades,
            equity_curve=combined_eq,
            initial_balance=total_balance,
            final_equity=final_eq,
            start_date=res_c.start_date,
            end_date=res_c.end_date,
            pairs=pairs,
        )
        m_combined = compute_metrics(combined_result)
        split_results[split_name] = (m_combined, combined_result)

        print(f"   Classic part: ${res_c.final_equity:,.2f} ({res_c.final_equity/bal_classic - 1:+.1%})")
        print(f"   Trail part:   ${res_t.final_equity:,.2f} ({res_t.final_equity/bal_trail - 1:+.1%})")
        print(f"   Combined:     ${final_eq:,.2f} ({final_eq/total_balance - 1:+.1%})")

    # ── Comparaison finale ──
    print_split_comparison(
        split_results,
        m_classic_full, res_classic_full,
        m_trail_full, res_trail_full,
    )


if __name__ == "__main__":
    main()
