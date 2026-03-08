#!/usr/bin/env python3
"""
Backtest : Range Grid Strategy

Exploite les marchés latéraux avec un grid de niveaux d'achat/vente.

Règles :
  - Détection range : ATR14 < ATR50 × 0.8
  - Grid : N niveaux espacés de grid_step %
  - Buy aux niveaux bas, sell aux niveaux hauts
  - SL si le prix sort du range (break below grid - margin)

Usage :
    python -m backtest.run_backtest_grid --years 6 --balance 500 --variants
"""

from __future__ import annotations

import argparse
import logging
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from backtest.data_loader import download_candles
from src.core.models import Candle
from src.core.indicators import sma, atr_series

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(message)s")
logger = logging.getLogger(__name__)

OUTPUT_DIR = Path(__file__).parent / "output"
OUTPUT_DIR.mkdir(exist_ok=True)

PAIRS = ["BTC-USD", "ETH-USD", "SOL-USD", "BNB-USD", "LINK-USD"]
MAKER_FEE = 0.0
TAKER_FEE = 0.0009


@dataclass
class GridConfig:
    # ── Range detection ──
    atr_fast: int = 14
    atr_slow: int = 50
    range_ratio: float = 0.8          # ATR14 < ATR50 × ratio → range

    # ── Grid params ──
    grid_step_pct: float = 0.015       # 1.5% entre chaque niveau
    grid_levels: int = 6               # nombre de niveaux
    grid_recalc_bars: int = 30         # recalculer la grille tous les N bars

    # ── Risk ──
    risk_per_level: float = 0.008      # 0.8% equity par niveau
    max_exposure_pct: float = 0.50
    sl_margin_pct: float = 0.02        # SL si break sous lowest_grid - 2%

    # ── Take profit ──
    tp_grid_levels: int = 2            # sell quand le prix monte de N niveaux


@dataclass
class Trade:
    symbol: str; side: str; entry_price: float; exit_price: float
    size: float; size_usd: float; entry_time: int; exit_time: int
    pnl_usd: float; pnl_pct: float; exit_reason: str; hold_bars: int = 0


@dataclass
class BacktestResult:
    label: str; trades: list[Trade]; equity_curve: list[float]
    initial_balance: float; final_equity: float; config_desc: dict


@dataclass
class _GridOrder:
    symbol: str
    level_idx: int
    entry_price: float
    size: float
    size_usd: float
    entry_bar: int
    entry_ts: int
    target_price: float  # sell target (entry + N grid steps)


@dataclass
class _GridState:
    is_range: bool = False
    grid_center: float = 0.0
    grid_levels: list[float] = field(default_factory=list)
    last_recalc: int = 0
    filled_levels: set = field(default_factory=set)


def simulate_grid(
    all_candles: dict[str, list[Candle]],
    cfg: GridConfig,
    initial_balance: float = 500.0,
) -> BacktestResult:
    balance = initial_balance
    orders: list[_GridOrder] = []
    closed: list[Trade] = []
    equity_curve: list[float] = [balance]

    # ATR
    all_atr_fast: dict[str, list[float]] = {}
    all_atr_slow: dict[str, list[float]] = {}
    for sym, c in all_candles.items():
        all_atr_fast[sym] = atr_series(c, cfg.atr_fast)
        all_atr_slow[sym] = atr_series(c, cfg.atr_slow)

    grid_states: dict[str, _GridState] = {s: _GridState() for s in all_candles}
    min_len = min(len(c) for c in all_candles.values())
    start_bar = cfg.atr_slow + 10

    for bar_idx in range(start_bar, min_len):

        # ── Manage existing grid orders ──
        for order in orders[:]:
            c = all_candles[order.symbol][bar_idx]
            hold = bar_idx - order.entry_bar
            gs = grid_states[order.symbol]

            # SL: price breaks below grid
            if gs.grid_levels:
                lowest = min(gs.grid_levels)
                sl_price = lowest * (1 - cfg.sl_margin_pct)
                if c.low <= sl_price:
                    pnl_pct = (sl_price - order.entry_price) / order.entry_price
                    pnl_usd = order.size_usd * pnl_pct - order.size_usd * TAKER_FEE
                    balance += order.size_usd + pnl_usd
                    closed.append(Trade(
                        symbol=order.symbol, side="LONG",
                        entry_price=order.entry_price, exit_price=sl_price,
                        size=order.size, size_usd=order.size_usd,
                        entry_time=order.entry_ts, exit_time=c.timestamp,
                        pnl_usd=pnl_usd, pnl_pct=pnl_pct * 100,
                        exit_reason="SL", hold_bars=hold,
                    ))
                    orders.remove(order)
                    continue

            # TP: price reaches target
            if c.high >= order.target_price:
                pnl_pct = (order.target_price - order.entry_price) / order.entry_price
                pnl_usd = order.size_usd * pnl_pct - order.size_usd * MAKER_FEE
                balance += order.size_usd + pnl_usd
                closed.append(Trade(
                    symbol=order.symbol, side="LONG",
                    entry_price=order.entry_price, exit_price=order.target_price,
                    size=order.size, size_usd=order.size_usd,
                    entry_time=order.entry_ts, exit_time=c.timestamp,
                    pnl_usd=pnl_usd, pnl_pct=pnl_pct * 100,
                    exit_reason="TP_GRID", hold_bars=hold,
                ))
                orders.remove(order)
                if order.level_idx in gs.filled_levels:
                    gs.filled_levels.discard(order.level_idx)
                continue

            # Exit if range broken (ATR expanded)
            af = all_atr_fast[order.symbol][bar_idx] if bar_idx < len(all_atr_fast[order.symbol]) else 0
            asl = all_atr_slow[order.symbol][bar_idx] if bar_idx < len(all_atr_slow[order.symbol]) else 0
            if asl > 0 and af > asl * 1.2:  # range broken
                pnl_pct = (c.close - order.entry_price) / order.entry_price
                pnl_usd = order.size_usd * pnl_pct - order.size_usd * TAKER_FEE
                balance += order.size_usd + pnl_usd
                closed.append(Trade(
                    symbol=order.symbol, side="LONG",
                    entry_price=order.entry_price, exit_price=c.close,
                    size=order.size, size_usd=order.size_usd,
                    entry_time=order.entry_ts, exit_time=c.timestamp,
                    pnl_usd=pnl_usd, pnl_pct=pnl_pct * 100,
                    exit_reason="RANGE_BREAK", hold_bars=hold,
                ))
                orders.remove(order)
                if order.level_idx in gs.filled_levels:
                    gs.filled_levels.discard(order.level_idx)
                continue

        # ── Grid management per symbol ──
        for symbol in all_candles:
            c = all_candles[symbol][bar_idx]
            gs = grid_states[symbol]

            af = all_atr_fast[symbol][bar_idx] if bar_idx < len(all_atr_fast[symbol]) else 0
            asl = all_atr_slow[symbol][bar_idx] if bar_idx < len(all_atr_slow[symbol]) else 0

            if asl <= 0:
                continue

            was_range = gs.is_range
            gs.is_range = af < asl * cfg.range_ratio

            # Recalculate grid periodically
            if gs.is_range and (bar_idx - gs.last_recalc >= cfg.grid_recalc_bars or not gs.grid_levels):
                gs.grid_center = c.close
                gs.grid_levels = []
                half = cfg.grid_levels // 2
                for i in range(-half, half + 1):
                    level = gs.grid_center * (1 + i * cfg.grid_step_pct)
                    gs.grid_levels.append(level)
                gs.filled_levels = set()
                gs.last_recalc = bar_idx

            # Fill grid levels
            if gs.is_range and gs.grid_levels and balance > 10:
                sym_orders = [o for o in orders if o.symbol == symbol]
                total_exp = sum(o.size_usd for o in orders)
                equity = balance + total_exp
                max_exp = equity * cfg.max_exposure_pct

                for i, level in enumerate(gs.grid_levels):
                    if i in gs.filled_levels:
                        continue
                    if level >= c.close:  # Only buy at/below current price
                        continue
                    if c.low <= level:  # Price touched this level
                        if total_exp >= max_exp:
                            break
                        size_usd = equity * cfg.risk_per_level
                        remaining = max_exp - total_exp
                        if size_usd > remaining:
                            size_usd = remaining
                        if size_usd < 5:
                            continue
                        size = size_usd / level if level > 0 else 0
                        target = level * (1 + cfg.tp_grid_levels * cfg.grid_step_pct)

                        balance -= size_usd
                        total_exp += size_usd
                        orders.append(_GridOrder(
                            symbol=symbol, level_idx=i,
                            entry_price=level, size=size, size_usd=size_usd,
                            entry_bar=bar_idx, entry_ts=c.timestamp,
                            target_price=target,
                        ))
                        gs.filled_levels.add(i)

        # Equity
        pv = sum(
            o.size * all_candles[o.symbol][min(bar_idx, len(all_candles[o.symbol]) - 1)].close
            for o in orders
        )
        equity_curve.append(balance + pv)

    # Close remaining
    for o in orders:
        last = all_candles[o.symbol][min_len - 1]
        pnl_pct = (last.close - o.entry_price) / o.entry_price
        pnl_usd = o.size_usd * pnl_pct - o.size_usd * TAKER_FEE
        balance += o.size_usd + pnl_usd
        closed.append(Trade(
            symbol=o.symbol, side="LONG",
            entry_price=o.entry_price, exit_price=last.close,
            size=o.size, size_usd=o.size_usd,
            entry_time=o.entry_ts, exit_time=last.timestamp,
            pnl_usd=pnl_usd, pnl_pct=pnl_pct * 100,
            exit_reason="END", hold_bars=min_len - 1 - o.entry_bar,
        ))

    return BacktestResult(
        label="", trades=closed, equity_curve=equity_curve,
        initial_balance=initial_balance,
        final_equity=equity_curve[-1] if equity_curve else initial_balance,
        config_desc={},
    )


def _define_variants(run_variants: bool) -> list[tuple[str, GridConfig, dict, list[str] | None]]:
    configs = []
    configs.append(("GRID_BASE", GridConfig(), {"step": "1.5%", "levels": 6}, None))
    configs.append(("GRID_3P", GridConfig(), {"pairs": "3"}, ["BTC-USD", "ETH-USD", "SOL-USD"]))

    if run_variants:
        configs.append(("GRID_1PCT", GridConfig(grid_step_pct=0.01), {"step": "1%"}, None))
        configs.append(("GRID_2PCT", GridConfig(grid_step_pct=0.02), {"step": "2%"}, None))
        configs.append(("GRID_3PCT", GridConfig(grid_step_pct=0.03), {"step": "3%"}, None))
        configs.append(("GRID_8LVL", GridConfig(grid_levels=8), {"levels": "8"}, None))
        configs.append(("GRID_4LVL", GridConfig(grid_levels=4), {"levels": "4"}, None))
        configs.append(("GRID_TP3", GridConfig(tp_grid_levels=3), {"tp": "3 levels"}, None))
        configs.append(("GRID_TP1", GridConfig(tp_grid_levels=1), {"tp": "1 level"}, None))
        configs.append(("GRID_LOOSE", GridConfig(range_ratio=0.9), {"range": "ATR<0.9"}, None))
        configs.append(("GRID_TIGHT", GridConfig(range_ratio=0.7), {"range": "ATR<0.7"}, None))
        configs.append(("GRID_R15", GridConfig(risk_per_level=0.015), {"risk": "1.5%/level"}, None))

    return configs


# ── Reporting (reuse pattern) ─────────────────────────────────

def compute_kpis(r: BacktestResult) -> dict:
    t = r.trades
    if not t:
        return {"label": r.label, "trades": 0, "win_rate": 0, "pf": 0,
                "pnl": 0, "avg_pnl": 0, "max_dd": 0, "final": r.initial_balance, "rr": 0, "avg_hold": 0}
    w = [x for x in t if x.pnl_usd > 0]; l = [x for x in t if x.pnl_usd <= 0]
    tg = sum(x.pnl_usd for x in w) if w else 0; tl = abs(sum(x.pnl_usd for x in l)) if l else 0.001
    pk = r.equity_curve[0]; md = 0
    for eq in r.equity_curve:
        if eq > pk: pk = eq
        dd = (pk - eq) / pk * 100 if pk > 0 else 0
        if dd > md: md = dd
    aw = tg / len(w) if w else 0; al = tl / len(l) if l else 0.001
    return {"label": r.label, "trades": len(t), "win_rate": len(w)/len(t)*100,
            "pf": tg/tl, "pnl": r.final_equity-r.initial_balance,
            "avg_pnl": sum(x.pnl_usd for x in t)/len(t),
            "max_dd": md, "final": r.final_equity, "rr": aw/al, "avg_hold": sum(x.hold_bars for x in t)/len(t)}


def print_table(kl, title):
    print(f"\n{'='*120}\n  {title}\n{'='*120}")
    print(f"{'Config':<22} {'Trades':>6} {'WR%':>7} {'PF':>7} {'PnL $':>9} {'Avg PnL':>8} {'Max DD%':>8} {'Final $':>9} {'R:R':>5} {'Hold':>6}")
    print("-"*120)
    for k in kl:
        print(f"{k['label']:<22} {k['trades']:>6} {k['win_rate']:>6.1f}% {k['pf']:>7.2f} "
              f"{k['pnl']:>+9.2f}$ {k['avg_pnl']:>+7.2f}$ {k['max_dd']:>7.1f}% "
              f"{k['final']:>9.2f}$ {k['rr']:>5.2f} {k['avg_hold']:>5.1f}b")
    print("-"*120)
    if kl:
        b=max(kl,key=lambda k:k["pf"]); print(f"  Meilleur PF : {b['label']} (PF {b['pf']:.2f}, PnL {b['pnl']:+.2f}$)")


def print_exits(results):
    print(f"\n{'='*70}\n  Répartition des sorties\n{'='*70}")
    for r in results:
        if not r.trades: continue
        ct = Counter(t.exit_reason for t in r.trades)
        print(f"\n  {r.label}:")
        for reason,cnt in ct.most_common():
            pct=cnt/len(r.trades)*100;avg=sum(t.pnl_usd for t in r.trades if t.exit_reason==reason)/cnt
            print(f"    {reason:<14}: {cnt:>5} ({pct:>5.1f}%)  avg: {avg:>+.2f}$")


def plot_eq(results, title, fn):
    fig, ax = plt.subplots(figsize=(14, 6))
    for r in results:
        if r.trades: ax.plot(r.equity_curve, label=r.label, lw=1)
    ax.axhline(y=results[0].initial_balance, color="grey", ls="--", alpha=.5)
    ax.set_title(title);ax.legend(fontsize=7);ax.grid(True,alpha=.3)
    p=OUTPUT_DIR/fn;fig.savefig(p,dpi=150,bbox_inches="tight");plt.close(fig)
    print(f"\n  Chart : {p}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--years", type=int, default=6)
    ap.add_argument("--balance", type=float, default=500)
    ap.add_argument("--variants", action="store_true")
    args = ap.parse_args()

    end = datetime.now(timezone.utc); start = end - timedelta(days=args.years*365)

    print(f"\n{'═'*70}\n  Range Grid — H4 ({args.years} ans)\n{'═'*70}")
    all_candles = {}
    for pair in PAIRS:
        all_candles[pair] = download_candles(pair, start, end, interval="4h")
        logger.info("  %s: %d bougies", pair, len(all_candles[pair]))

    configs = _define_variants(args.variants)
    results, kpis_list = [], []
    for label, cfg, desc, po in configs:
        cd = {k: v for k, v in all_candles.items() if k in po} if po else all_candles
        if not cd: continue
        r = simulate_grid(cd, cfg, args.balance); r.label = label; r.config_desc = desc
        results.append(r); kpis_list.append(compute_kpis(r))

    print_table(kpis_list, f"Range Grid — H4 ({args.years} ans, ${args.balance:.0f})")
    print_exits(results)
    plot_eq(results, "Range Grid (H4)", "grid_equity.png")

    valid = [k for k in kpis_list if k["trades"] >= 10]
    if valid:
        b = max(valid, key=lambda k: k["pf"])
        tag = "✅ PROMETTEUR" if b["pf"] >= 1.5 else ("⚠️  MARGINAL" if b["pf"] >= 1.0 else "❌ NON RENTABLE")
        print(f"\n{'═'*70}\n  {tag} : {b['label']} — PF {b['pf']:.2f}, PnL {b['pnl']:+.2f}$, {b['trades']} trades\n{'═'*70}\n")


if __name__ == "__main__":
    main()
