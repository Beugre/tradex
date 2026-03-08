#!/usr/bin/env python3
"""
Asian Range Breakout — Deep Exploration

Analyses :
  1. Per-pair : PF & PnL de chaque paire individuellement
  2. Grid search : optimisation combinée (sl, tp, vol, range_min)
  3. Walk-forward : train 4 ans → test 2 ans (robustesse)
  4. Short breakout : test des shorts sous asian_low
  5. Session hours : différentes fenêtres horaires (London, US…)

Usage :
    python -m backtest.run_asian_deep --years 6 --balance 500
    python -m backtest.run_asian_deep --years 6 --balance 500 --mode perpair
    python -m backtest.run_asian_deep --years 6 --balance 500 --mode grid
    python -m backtest.run_asian_deep --years 6 --balance 500 --mode walkforward
    python -m backtest.run_asian_deep --years 6 --balance 500 --mode all
"""

from __future__ import annotations

import argparse
import itertools
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

ALL_PAIRS = ["BTC-USD", "ETH-USD", "SOL-USD", "BNB-USD", "LINK-USD",
             "ADA-USD", "DOT-USD", "AVAX-USD", "XRP-USD", "LTC-USD",
             "AAVE-USD", "UNI-USD", "DOGE-USD", "NEAR-USD", "ATOM-USD"]

MAKER_FEE = 0.0
TAKER_FEE = 0.0009


# ─────────────────────────────────────────────────────────────────────
#  Config
# ─────────────────────────────────────────────────────────────────────

@dataclass
class AsianConfig:
    asian_start_hour: int = 0
    asian_end_hour: int = 8
    vol_ma_period: int = 20
    vol_mult: float = 1.5
    min_range_pct: float = 0.005
    atr_period: int = 14
    sl_mode: str = "atr"          # "atr" | "asian_low"
    sl_atr_mult: float = 1.5
    tp1_pct: float = 0.02
    tp2_pct: float = 0.04
    tp1_share: float = 0.50
    tp2_share: float = 0.50
    breakeven_after_tp1: bool = True
    risk_per_trade: float = 0.015
    max_positions: int = 4
    max_exposure_pct: float = 0.50
    cooldown_bars: int = 2         # H4 default

    # ── SHORT ──
    allow_short: bool = False


# ─────────────────────────────────────────────────────────────────────
#  Data classes
# ─────────────────────────────────────────────────────────────────────

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
class _OpenPos:
    symbol: str; side: str   # "LONG" or "SHORT"
    entry_price: float; sl_price: float
    initial_size: float; initial_size_usd: float
    remaining_size: float; remaining_size_usd: float
    entry_bar: int; entry_ts: int
    tp1_price: float = 0.0; tp2_price: float = 0.0
    tp1_hit: bool = False


@dataclass
class _PairState:
    cooldown_until: int = 0
    asian_high: float = 0.0
    asian_low: float = float("inf")
    session_bars: int = 0
    session_complete: bool = False
    breakout_consumed_long: bool = False
    breakout_consumed_short: bool = False


# ─────────────────────────────────────────────────────────────────────
#  Simulator  (supports LONG + SHORT)
# ─────────────────────────────────────────────────────────────────────

def _bar_hour_utc(candle: Candle) -> int:
    return datetime.fromtimestamp(candle.timestamp / 1000, tz=timezone.utc).hour


def simulate_asian(
    all_candles: dict[str, list[Candle]],
    cfg: AsianConfig,
    initial_balance: float = 500.0,
    tf: str = "4h",
    start_bar: int | None = None,
    end_bar: int | None = None,
) -> BacktestResult:
    balance = initial_balance
    positions: list[_OpenPos] = []
    closed: list[Trade] = []
    equity_curve: list[float] = [balance]

    all_atr: dict[str, list[float]] = {}
    all_vol_ma: dict[str, list[float]] = {}
    for sym, c in all_candles.items():
        all_atr[sym] = atr_series(c, cfg.atr_period)
        all_vol_ma[sym] = sma([x.volume for x in c], cfg.vol_ma_period)

    states: dict[str, _PairState] = {s: _PairState() for s in all_candles}
    min_len = min(len(c) for c in all_candles.values())
    sb = start_bar if start_bar is not None else max(cfg.atr_period + 5, cfg.vol_ma_period + 5)
    eb = end_bar if end_bar is not None else min_len

    for bar_idx in range(sb, eb):

        # ── Manage open positions ──
        for pos in positions[:]:
            c = all_candles[pos.symbol][bar_idx]
            hold = bar_idx - pos.entry_bar

            if pos.side == "LONG":
                # SL hit
                if c.low <= pos.sl_price:
                    ep = pos.sl_price
                    pnl_pct = (ep - pos.entry_price) / pos.entry_price
                    pnl_usd = pos.remaining_size_usd * pnl_pct - pos.remaining_size_usd * TAKER_FEE
                    balance += pos.remaining_size_usd + pnl_usd
                    reason = "BE" if pos.tp1_hit and pos.sl_price >= pos.entry_price else "SL"
                    closed.append(Trade(pos.symbol, "LONG", pos.entry_price, ep,
                        pos.remaining_size, pos.remaining_size_usd,
                        pos.entry_ts, c.timestamp, pnl_usd, pnl_pct * 100, reason, hold))
                    positions.remove(pos); continue

                # TP1
                if not pos.tp1_hit and c.high >= pos.tp1_price:
                    cs = pos.initial_size * cfg.tp1_share
                    cu = pos.initial_size_usd * cfg.tp1_share
                    pp = (pos.tp1_price - pos.entry_price) / pos.entry_price
                    pu = cu * pp - cu * MAKER_FEE
                    balance += cu + pu
                    closed.append(Trade(pos.symbol, "LONG", pos.entry_price, pos.tp1_price,
                        cs, cu, pos.entry_ts, c.timestamp, pu, pp * 100, "TP1", hold))
                    pos.tp1_hit = True; pos.remaining_size -= cs; pos.remaining_size_usd -= cu
                    if cfg.breakeven_after_tp1:
                        pos.sl_price = pos.entry_price

                # TP2
                if pos.tp1_hit and c.high >= pos.tp2_price:
                    pp = (pos.tp2_price - pos.entry_price) / pos.entry_price
                    pu = pos.remaining_size_usd * pp - pos.remaining_size_usd * MAKER_FEE
                    balance += pos.remaining_size_usd + pu
                    closed.append(Trade(pos.symbol, "LONG", pos.entry_price, pos.tp2_price,
                        pos.remaining_size, pos.remaining_size_usd,
                        pos.entry_ts, c.timestamp, pu, pp * 100, "TP2", hold))
                    positions.remove(pos); continue

            else:  # SHORT
                # SL hit (price goes UP)
                if c.high >= pos.sl_price:
                    ep = pos.sl_price
                    pnl_pct = (pos.entry_price - ep) / pos.entry_price
                    pnl_usd = pos.remaining_size_usd * pnl_pct - pos.remaining_size_usd * TAKER_FEE
                    balance += pos.remaining_size_usd + pnl_usd
                    reason = "BE" if pos.tp1_hit and pos.sl_price <= pos.entry_price else "SL"
                    closed.append(Trade(pos.symbol, "SHORT", pos.entry_price, ep,
                        pos.remaining_size, pos.remaining_size_usd,
                        pos.entry_ts, c.timestamp, pnl_usd, pnl_pct * 100, reason, hold))
                    positions.remove(pos); continue

                # TP1 (price goes DOWN)
                if not pos.tp1_hit and c.low <= pos.tp1_price:
                    cs = pos.initial_size * cfg.tp1_share
                    cu = pos.initial_size_usd * cfg.tp1_share
                    pp = (pos.entry_price - pos.tp1_price) / pos.entry_price
                    pu = cu * pp - cu * MAKER_FEE
                    balance += cu + pu
                    closed.append(Trade(pos.symbol, "SHORT", pos.entry_price, pos.tp1_price,
                        cs, cu, pos.entry_ts, c.timestamp, pu, pp * 100, "TP1", hold))
                    pos.tp1_hit = True; pos.remaining_size -= cs; pos.remaining_size_usd -= cu
                    if cfg.breakeven_after_tp1:
                        pos.sl_price = pos.entry_price

                # TP2 (price goes DOWN)
                if pos.tp1_hit and c.low <= pos.tp2_price:
                    pp = (pos.entry_price - pos.tp2_price) / pos.entry_price
                    pu = pos.remaining_size_usd * pp - pos.remaining_size_usd * MAKER_FEE
                    balance += pos.remaining_size_usd + pu
                    closed.append(Trade(pos.symbol, "SHORT", pos.entry_price, pos.tp2_price,
                        pos.remaining_size, pos.remaining_size_usd,
                        pos.entry_ts, c.timestamp, pu, pp * 100, "TP2", hold))
                    positions.remove(pos); continue

            if pos.remaining_size_usd < 1:
                positions.remove(pos)

        # ── Track session & detect breakouts ──
        for symbol in all_candles:
            c = all_candles[symbol][bar_idx]
            st = states[symbol]
            hour = _bar_hour_utc(c)

            # H4: 00:00 and 04:00 are "Asian"
            in_asian = hour < cfg.asian_end_hour if tf == "4h" else (cfg.asian_start_hour <= hour < cfg.asian_end_hour)

            if in_asian:
                if st.session_bars == 0:
                    st.asian_high = c.high
                    st.asian_low = c.low
                    st.breakout_consumed_long = False
                    st.breakout_consumed_short = False
                else:
                    st.asian_high = max(st.asian_high, c.high)
                    st.asian_low = min(st.asian_low, c.low)
                st.session_bars += 1
                st.session_complete = False
            else:
                if st.session_bars > 0 and not st.session_complete:
                    st.session_complete = True

                if hour >= cfg.asian_end_hour and st.session_complete:

                    # ── Common checks ──
                    already_in = any(p.symbol == symbol for p in positions)
                    on_cooldown = bar_idx < st.cooldown_until
                    full = len(positions) >= cfg.max_positions
                    broke = balance <= 10

                    range_pct = (st.asian_high - st.asian_low) / st.asian_low if st.asian_low > 0 else 0
                    too_small = range_pct < cfg.min_range_pct

                    vol_ma = all_vol_ma[symbol][bar_idx] if bar_idx < len(all_vol_ma[symbol]) else 0
                    atr_val = all_atr[symbol][bar_idx] if bar_idx < len(all_atr[symbol]) else 0

                    # ── LONG breakout ──
                    if (not st.breakout_consumed_long and not already_in and not on_cooldown
                            and not full and not broke and not too_small
                            and c.close > st.asian_high
                            and (cfg.vol_mult <= 0 or (vol_ma > 0 and c.volume >= cfg.vol_mult * vol_ma))):

                        entry_price = c.close
                        if cfg.sl_mode == "asian_low":
                            sl_price = st.asian_low * 0.998
                        else:
                            sl_price = entry_price - cfg.sl_atr_mult * atr_val if atr_val > 0 else st.asian_low

                        sl_dist = entry_price - sl_price
                        if sl_dist > 0:
                            total_exp = sum(p.remaining_size_usd for p in positions)
                            equity = balance + total_exp
                            max_exp = equity * cfg.max_exposure_pct
                            risk_amount = equity * cfg.risk_per_trade
                            size = risk_amount / sl_dist
                            size_usd = size * entry_price
                            rem = max_exp - total_exp
                            if size_usd > rem:
                                size_usd = rem; size = size_usd / entry_price if entry_price > 0 else 0
                            if size_usd >= 5:
                                balance -= size_usd
                                positions.append(_OpenPos(
                                    symbol=symbol, side="LONG",
                                    entry_price=entry_price, sl_price=sl_price,
                                    initial_size=size, initial_size_usd=size_usd,
                                    remaining_size=size, remaining_size_usd=size_usd,
                                    entry_bar=bar_idx, entry_ts=c.timestamp,
                                    tp1_price=entry_price * (1 + cfg.tp1_pct),
                                    tp2_price=entry_price * (1 + cfg.tp2_pct),
                                ))
                                st.cooldown_until = bar_idx + cfg.cooldown_bars
                                st.breakout_consumed_long = True

                    # ── SHORT breakout ──
                    if (cfg.allow_short and not st.breakout_consumed_short
                            and not already_in and not on_cooldown
                            and not full and not broke and not too_small
                            and c.close < st.asian_low
                            and (cfg.vol_mult <= 0 or (vol_ma > 0 and c.volume >= cfg.vol_mult * vol_ma))):

                        entry_price = c.close
                        if cfg.sl_mode == "asian_low":
                            sl_price = st.asian_high * 1.002
                        else:
                            sl_price = entry_price + cfg.sl_atr_mult * atr_val if atr_val > 0 else st.asian_high

                        sl_dist = sl_price - entry_price
                        if sl_dist > 0:
                            total_exp = sum(p.remaining_size_usd for p in positions)
                            equity = balance + total_exp
                            max_exp = equity * cfg.max_exposure_pct
                            risk_amount = equity * cfg.risk_per_trade
                            size = risk_amount / sl_dist
                            size_usd = size * entry_price
                            rem = max_exp - total_exp
                            if size_usd > rem:
                                size_usd = rem; size = size_usd / entry_price if entry_price > 0 else 0
                            if size_usd >= 5:
                                balance -= size_usd
                                positions.append(_OpenPos(
                                    symbol=symbol, side="SHORT",
                                    entry_price=entry_price, sl_price=sl_price,
                                    initial_size=size, initial_size_usd=size_usd,
                                    remaining_size=size, remaining_size_usd=size_usd,
                                    entry_bar=bar_idx, entry_ts=c.timestamp,
                                    tp1_price=entry_price * (1 - cfg.tp1_pct),
                                    tp2_price=entry_price * (1 - cfg.tp2_pct),
                                ))
                                st.cooldown_until = bar_idx + cfg.cooldown_bars
                                st.breakout_consumed_short = True

                # Reset session at end of day
                if hour >= 20 or (tf == "4h" and hour >= 16):
                    st.session_bars = 0

        pv = sum(
            p.remaining_size * all_candles[p.symbol][min(bar_idx, len(all_candles[p.symbol]) - 1)].close
            for p in positions
        )
        equity_curve.append(balance + pv)

    # ── Close remaining positions ──
    for pos in positions:
        last = all_candles[pos.symbol][min(eb - 1, len(all_candles[pos.symbol]) - 1)]
        if pos.side == "LONG":
            pp = (last.close - pos.entry_price) / pos.entry_price
        else:
            pp = (pos.entry_price - last.close) / pos.entry_price
        pu = pos.remaining_size_usd * pp - pos.remaining_size_usd * TAKER_FEE
        balance += pos.remaining_size_usd + pu
        closed.append(Trade(pos.symbol, pos.side, pos.entry_price, last.close,
            pos.remaining_size, pos.remaining_size_usd,
            pos.entry_ts, last.timestamp, pu, pp * 100, "END", eb - 1 - pos.entry_bar))

    return BacktestResult("", closed, equity_curve, initial_balance,
        equity_curve[-1] if equity_curve else initial_balance, {})


# ─────────────────────────────────────────────────────────────────────
#  KPI helpers
# ─────────────────────────────────────────────────────────────────────

def compute_kpis(r: BacktestResult) -> dict:
    t = r.trades
    if not t:
        return {"label": r.label, "trades": 0, "win_rate": 0, "pf": 0, "pnl": 0,
                "avg_pnl": 0, "max_dd": 0, "final": r.initial_balance, "rr": 0, "avg_hold": 0}
    w = [x for x in t if x.pnl_usd > 0]
    l = [x for x in t if x.pnl_usd <= 0]
    tg = sum(x.pnl_usd for x in w) if w else 0
    tl = abs(sum(x.pnl_usd for x in l)) if l else 0.001
    pk = r.equity_curve[0]; md = 0
    for eq in r.equity_curve:
        if eq > pk: pk = eq
        dd = (pk - eq) / pk * 100 if pk > 0 else 0
        if dd > md: md = dd
    aw = tg / len(w) if w else 0
    al = tl / len(l) if l else 0.001
    return {
        "label": r.label, "trades": len(t), "win_rate": len(w) / len(t) * 100,
        "pf": tg / tl, "pnl": r.final_equity - r.initial_balance,
        "avg_pnl": sum(x.pnl_usd for x in t) / len(t), "max_dd": md,
        "final": r.final_equity, "rr": aw / al, "avg_hold": sum(x.hold_bars for x in t) / len(t),
    }


def print_table(kl, title):
    print(f"\n{'=' * 120}\n  {title}\n{'=' * 120}")
    print(f"{'Config':<30} {'Trades':>6} {'WR%':>7} {'PF':>7} {'PnL $':>9} "
          f"{'Avg PnL':>8} {'Max DD%':>8} {'Final $':>9} {'R:R':>5} {'Hold':>6}")
    print("-" * 120)
    for k in kl:
        tag = "✅" if k["pf"] >= 1.3 else ("⚠️ " if k["pf"] >= 1.0 else "❌")
        print(f"{tag} {k['label']:<28} {k['trades']:>6} {k['win_rate']:>6.1f}% {k['pf']:>7.2f} "
              f"{k['pnl']:>+9.2f}$ {k['avg_pnl']:>+7.2f}$ {k['max_dd']:>7.1f}% "
              f"{k['final']:>9.2f}$ {k['rr']:>5.2f} {k['avg_hold']:>5.1f}b")
    print("-" * 120)
    valid = [k for k in kl if k["trades"] >= 10]
    if valid:
        b = max(valid, key=lambda k: k["pf"])
        print(f"  🏆 Meilleur PF : {b['label']} (PF {b['pf']:.2f}, PnL {b['pnl']:+.2f}$, {b['trades']} trades)")


def print_exits(results):
    print(f"\n{'=' * 70}\n  Répartition des sorties\n{'=' * 70}")
    for r in results:
        if not r.trades: continue
        ct = Counter(t.exit_reason for t in r.trades)
        print(f"\n  {r.label}:")
        for reason, cnt in ct.most_common():
            pct = cnt / len(r.trades) * 100
            avg = sum(t.pnl_usd for t in r.trades if t.exit_reason == reason) / cnt
            print(f"    {reason:<14}: {cnt:>5} ({pct:>5.1f}%)  avg: {avg:>+.2f}$")


def plot_eq(results, title, fn):
    fig, ax = plt.subplots(figsize=(14, 6))
    for r in results:
        if r.trades: ax.plot(r.equity_curve, label=r.label, lw=1)
    ax.axhline(y=results[0].initial_balance, color="grey", ls="--", alpha=0.5)
    ax.set_title(title); ax.legend(fontsize=7); ax.grid(True, alpha=0.3)
    p = OUTPUT_DIR / fn; fig.savefig(p, dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"\n  📊 Chart : {p}")


# ═════════════════════════════════════════════════════════════════════
#  MODE 1 — Per-Pair Analysis
# ═════════════════════════════════════════════════════════════════════

def run_perpair(all_candles: dict[str, list[Candle]], balance: float, tf: str):
    """Run Asian Breakout on each pair individually + combined."""
    print(f"\n{'═' * 80}")
    print(f"  📋 ANALYSE PER-PAIR — Asian Breakout {tf.upper()}")
    print(f"{'═' * 80}")

    # Best base config from previous run
    cfg_base = AsianConfig(sl_atr_mult=2.0, cooldown_bars=2)
    kpis_list = []
    results = []

    # Each pair solo
    for pair in sorted(all_candles.keys()):
        single = {pair: all_candles[pair]}
        cfg_solo = AsianConfig(sl_atr_mult=2.0, cooldown_bars=2, max_positions=1)
        r = simulate_asian(single, cfg_solo, balance, tf)
        r.label = f"{pair}"
        results.append(r)
        kpis_list.append(compute_kpis(r))

    # All pairs combined
    r_all = simulate_asian(all_candles, cfg_base, balance, tf)
    r_all.label = "ALL_COMBINED"
    results.append(r_all)
    kpis_list.append(compute_kpis(r_all))

    print_table(kpis_list, f"Per-Pair — Asian Breakout {tf.upper()}")
    print_exits(results)

    # Per-pair trade breakdown from combined run
    print(f"\n{'=' * 80}")
    print(f"  📊 Détail par paire (run combiné)")
    print(f"{'=' * 80}")
    print(f"{'Paire':<12} {'Trades':>6} {'WR%':>7} {'PF':>7} {'PnL $':>9} {'Avg':>8}")
    print("-" * 60)
    pair_trades: dict[str, list[Trade]] = {}
    for t in r_all.trades:
        pair_trades.setdefault(t.symbol, []).append(t)
    for pair in sorted(pair_trades.keys()):
        ts = pair_trades[pair]
        w = [t for t in ts if t.pnl_usd > 0]
        tg = sum(t.pnl_usd for t in w) if w else 0
        tl_val = abs(sum(t.pnl_usd for t in ts if t.pnl_usd <= 0)) or 0.001
        wr = len(w) / len(ts) * 100 if ts else 0
        pf = tg / tl_val
        pnl = sum(t.pnl_usd for t in ts)
        avg = pnl / len(ts) if ts else 0
        tag = "✅" if pf >= 1.3 else ("⚠️ " if pf >= 1.0 else "❌")
        print(f"{tag} {pair:<10} {len(ts):>6} {wr:>6.1f}% {pf:>7.2f} {pnl:>+9.2f}$ {avg:>+7.2f}$")
    print("-" * 60)

    plot_eq(results, f"Asian Breakout Per-Pair {tf.upper()}", f"asian_perpair_{tf}.png")
    return kpis_list


# ═════════════════════════════════════════════════════════════════════
#  MODE 2 — Grid Search (params optimisation)
# ═════════════════════════════════════════════════════════════════════

def run_grid(all_candles: dict[str, list[Candle]], balance: float, tf: str):
    """Grid search over key parameters."""
    print(f"\n{'═' * 80}")
    print(f"  🔍 GRID SEARCH — Asian Breakout {tf.upper()}")
    print(f"{'═' * 80}")

    sl_mults = [1.0, 1.5, 2.0, 2.5, 3.0]
    tp1_pcts = [0.015, 0.02, 0.03, 0.04]
    tp2_pcts = [0.03, 0.04, 0.05, 0.06, 0.08]
    vol_mults = [0.0, 1.0, 1.5, 2.0]
    min_ranges = [0.003, 0.005, 0.008, 0.01]

    # Phase 1: SL × TP grid (vol=1.5, min_range=0.005)
    print("\n─── Phase 1 : SL_ATR_MULT × TP1/TP2 ───")
    kpis_p1 = []
    for sl_m in sl_mults:
        for tp1, tp2 in [(0.015, 0.03), (0.02, 0.04), (0.03, 0.05), (0.03, 0.06), (0.04, 0.08)]:
            if tp2 <= tp1: continue
            label = f"SL{sl_m}_TP{int(tp1*100)}/{int(tp2*100)}"
            cfg = AsianConfig(sl_atr_mult=sl_m, tp1_pct=tp1, tp2_pct=tp2, cooldown_bars=2)
            r = simulate_asian(all_candles, cfg, balance, tf)
            r.label = label
            kpis_p1.append(compute_kpis(r))
    print_table(sorted(kpis_p1, key=lambda k: -k["pf"])[:15],
                f"Grid Phase 1 (SL×TP) — Top 15 par PF — {tf.upper()}")

    # Phase 2: Volume filter × min range (using best SL/TP from Phase 1)
    best_p1 = max([k for k in kpis_p1 if k["trades"] >= 20], key=lambda k: k["pf"], default=None)
    if best_p1:
        # Parse best SL/TP
        parts = best_p1["label"].split("_")
        best_sl = float(parts[0].replace("SL", ""))
        tp_parts = parts[1].replace("TP", "").split("/")
        best_tp1 = int(tp_parts[0]) / 100
        best_tp2 = int(tp_parts[1]) / 100
    else:
        best_sl, best_tp1, best_tp2 = 2.0, 0.02, 0.04

    print(f"\n─── Phase 2 : Vol × MinRange (SL={best_sl}, TP={best_tp1}/{best_tp2}) ───")
    kpis_p2 = []
    for vm in vol_mults:
        for mr in min_ranges:
            label = f"VOL{vm}_MR{int(mr*1000)}"
            cfg = AsianConfig(sl_atr_mult=best_sl, tp1_pct=best_tp1, tp2_pct=best_tp2,
                              vol_mult=vm, min_range_pct=mr, cooldown_bars=2)
            r = simulate_asian(all_candles, cfg, balance, tf)
            r.label = label
            kpis_p2.append(compute_kpis(r))
    print_table(sorted(kpis_p2, key=lambda k: -k["pf"])[:10],
                f"Grid Phase 2 (Vol×MinRange) — Top 10 par PF — {tf.upper()}")

    # Phase 3: Risk sizing
    best_p2 = max([k for k in kpis_p2 if k["trades"] >= 20], key=lambda k: k["pf"], default=None)
    if best_p2:
        parts = best_p2["label"].split("_")
        best_vm = float(parts[0].replace("VOL", ""))
        best_mr = int(parts[1].replace("MR", "")) / 1000
    else:
        best_vm, best_mr = 1.5, 0.005

    print(f"\n─── Phase 3 : Sizing (SL={best_sl}, TP={best_tp1}/{best_tp2}, VOL={best_vm}, MR={best_mr}) ───")
    kpis_p3 = []
    for risk in [0.01, 0.015, 0.02, 0.025, 0.03, 0.04, 0.05]:
        for max_pos in [2, 3, 4, 5]:
            label = f"R{int(risk*100)}%_MP{max_pos}"
            cfg = AsianConfig(sl_atr_mult=best_sl, tp1_pct=best_tp1, tp2_pct=best_tp2,
                              vol_mult=best_vm, min_range_pct=best_mr,
                              risk_per_trade=risk, max_positions=max_pos,
                              cooldown_bars=2)
            r = simulate_asian(all_candles, cfg, balance, tf)
            r.label = label
            kpis_p3.append(compute_kpis(r))
    print_table(sorted(kpis_p3, key=lambda k: -k["pf"])[:10],
                f"Grid Phase 3 (Sizing) — Top 10 par PF — {tf.upper()}")

    # Final: best overall config
    all_kpis = kpis_p1 + kpis_p2 + kpis_p3
    valid = [k for k in all_kpis if k["trades"] >= 50]
    if valid:
        best = max(valid, key=lambda k: k["pf"])
        print(f"\n{'═' * 80}")
        print(f"  🏆 MEILLEURE CONFIG GRID : {best['label']}")
        print(f"     PF {best['pf']:.2f} | PnL {best['pnl']:+.2f}$ | {best['trades']} trades | DD {best['max_dd']:.1f}%")
        print(f"{'═' * 80}")
    return all_kpis


# ═════════════════════════════════════════════════════════════════════
#  MODE 3 — Walk-Forward Validation
# ═════════════════════════════════════════════════════════════════════

def run_walkforward(all_candles: dict[str, list[Candle]], balance: float, tf: str):
    """Walk-forward: train 4 years, test 2 years. Also 3/3 split."""
    print(f"\n{'═' * 80}")
    print(f"  📈 WALK-FORWARD — Asian Breakout {tf.upper()}")
    print(f"{'═' * 80}")

    min_len = min(len(c) for c in all_candles.values())
    bars_per_year = 365 * 6 if tf == "4h" else 365 * 24  # approximate

    # Split points
    splits = [
        ("4yr_train / 2yr_test", int(min_len * 4 / 6), min_len),
        ("3yr_train / 3yr_test", int(min_len * 3 / 6), min_len),
        ("First_half / Second_half", min_len // 2, min_len),
    ]

    configs_to_test = [
        ("BASE", AsianConfig(cooldown_bars=2)),
        ("SL2", AsianConfig(sl_atr_mult=2.0, cooldown_bars=2)),
        ("SL2_VOL2", AsianConfig(sl_atr_mult=2.0, vol_mult=2.0, cooldown_bars=2)),
        ("SL2_TP35", AsianConfig(sl_atr_mult=2.0, tp1_pct=0.03, tp2_pct=0.05, cooldown_bars=2)),
        ("SL2_AL", AsianConfig(sl_atr_mult=2.0, sl_mode="asian_low", cooldown_bars=2)),
        ("SL25_TP24", AsianConfig(sl_atr_mult=2.5, tp1_pct=0.02, tp2_pct=0.04, cooldown_bars=2)),
        ("SL3_TP36", AsianConfig(sl_atr_mult=3.0, tp1_pct=0.03, tp2_pct=0.06, cooldown_bars=2)),
    ]

    sb = max(20, 30)  # warmup

    for split_name, split_bar, total_bars in splits:
        print(f"\n{'─' * 70}")
        print(f"  🔄 Split : {split_name} (bar {sb}-{split_bar} → {split_bar}-{total_bars})")
        print(f"{'─' * 70}")

        kpis_train = []
        kpis_test = []

        for label, cfg in configs_to_test:
            # Train
            r_train = simulate_asian(all_candles, cfg, balance, tf, start_bar=sb, end_bar=split_bar)
            r_train.label = f"{label}_TRAIN"
            k_train = compute_kpis(r_train)
            kpis_train.append(k_train)

            # Test
            r_test = simulate_asian(all_candles, cfg, balance, tf, start_bar=split_bar, end_bar=total_bars)
            r_test.label = f"{label}_TEST"
            k_test = compute_kpis(r_test)
            kpis_test.append(k_test)

        print_table(kpis_train, f"TRAIN ({split_name.split('/')[0].strip()})")
        print_table(kpis_test, f"TEST ({split_name.split('/')[1].strip()})")

        # Compare train vs test
        print(f"\n  {'Config':<20} {'PF_train':>8} {'PF_test':>8} {'Δ PF':>7} {'Stable?':>8}")
        print(f"  {'-' * 55}")
        for kt, kts in zip(kpis_train, kpis_test):
            lbl = kt["label"].replace("_TRAIN", "")
            pf_tr = kt["pf"]
            pf_te = kts["pf"]
            delta = pf_te - pf_tr
            stable = "✅" if abs(delta) < 0.3 and pf_te >= 1.0 else ("⚠️ " if pf_te >= 0.9 else "❌")
            print(f"  {lbl:<20} {pf_tr:>8.2f} {pf_te:>8.2f} {delta:>+7.2f} {stable:>8}")
        print(f"  {'-' * 55}")


# ═════════════════════════════════════════════════════════════════════
#  MODE 4 — Short Breakout Test
# ═════════════════════════════════════════════════════════════════════

def run_short_test(all_candles: dict[str, list[Candle]], balance: float, tf: str):
    """Test SHORT breakouts (below asian_low) + combined LONG+SHORT."""
    print(f"\n{'═' * 80}")
    print(f"  📉 SHORT BREAKOUT — Asian Breakout {tf.upper()}")
    print(f"{'═' * 80}")

    configs = [
        ("LONG_ONLY", AsianConfig(sl_atr_mult=2.0, cooldown_bars=2, allow_short=False)),
        ("SHORT_ONLY", AsianConfig(sl_atr_mult=2.0, cooldown_bars=2, allow_short=True, vol_mult=0)),
        ("LONG+SHORT_BASE", AsianConfig(sl_atr_mult=2.0, cooldown_bars=2, allow_short=True)),
        ("LONG+SHORT_VOL2", AsianConfig(sl_atr_mult=2.0, vol_mult=2.0, cooldown_bars=2, allow_short=True)),
        ("LONG+SHORT_SL25", AsianConfig(sl_atr_mult=2.5, cooldown_bars=2, allow_short=True)),
    ]

    kpis_list = []
    results = []
    for label, cfg in configs:
        r = simulate_asian(all_candles, cfg, balance, tf)
        r.label = label
        results.append(r)
        kpis_list.append(compute_kpis(r))

    # For SHORT_ONLY, filter to show only shorts
    for r in results:
        if r.label == "SHORT_ONLY":
            short_trades = [t for t in r.trades if t.side == "SHORT"]
            long_trades = [t for t in r.trades if t.side == "LONG"]
            print(f"\n  SHORT_ONLY : {len(short_trades)} shorts, {len(long_trades)} longs")

    print_table(kpis_list, f"Short Breakout Test — {tf.upper()}")
    print_exits(results)

    # Side breakdown for combined configs
    for r in results:
        if "LONG+SHORT" in r.label:
            long_t = [t for t in r.trades if t.side == "LONG"]
            short_t = [t for t in r.trades if t.side == "SHORT"]
            print(f"\n  {r.label} breakdown:")
            for side_name, ts in [("LONG", long_t), ("SHORT", short_t)]:
                if not ts: continue
                w = [t for t in ts if t.pnl_usd > 0]
                tg = sum(t.pnl_usd for t in w) if w else 0
                tl_val = abs(sum(t.pnl_usd for t in ts if t.pnl_usd <= 0)) or 0.001
                pnl = sum(t.pnl_usd for t in ts)
                print(f"    {side_name:<8}: {len(ts)} trades, WR {len(w)/len(ts)*100:.1f}%, "
                      f"PF {tg/tl_val:.2f}, PnL {pnl:+.2f}$")

    plot_eq(results, f"Asian Short Test {tf.upper()}", f"asian_short_{tf}.png")
    return kpis_list


# ═════════════════════════════════════════════════════════════════════
#  MODE 5 — Session Hours Variants
# ═════════════════════════════════════════════════════════════════════

def run_sessions(all_candles: dict[str, list[Candle]], balance: float, tf: str):
    """Test different session definitions (not just Asian 00-08)."""
    print(f"\n{'═' * 80}")
    print(f"  🌍 SESSION VARIANTS — {tf.upper()}")
    print(f"{'═' * 80}")

    sessions = [
        ("Asian_00-08", 0, 8),
        ("Asian_21-05", 21, 5),      # Extended Asian (includes pre-Asian)
        ("London_08-16", 8, 16),
        ("US_13-21", 13, 21),
        ("Night_20-04", 20, 4),       # Overnight session
    ]

    kpis_list = []
    results = []
    for label, start_h, end_h in sessions:
        cfg = AsianConfig(
            asian_start_hour=start_h,
            asian_end_hour=end_h,
            sl_atr_mult=2.0,
            cooldown_bars=2,
        )
        r = simulate_asian(all_candles, cfg, balance, tf)
        r.label = label
        results.append(r)
        kpis_list.append(compute_kpis(r))

    print_table(kpis_list, f"Session Variants — {tf.upper()}")
    print_exits(results)
    plot_eq(results, f"Session Variants {tf.upper()}", f"asian_sessions_{tf}.png")
    return kpis_list


# ═════════════════════════════════════════════════════════════════════
#  MODE 6 — London Deep (Grid + Walk-Forward on London session)
# ═════════════════════════════════════════════════════════════════════

def run_london_deep(all_candles: dict[str, list[Candle]], balance: float, tf: str):
    """Targeted grid search + walk-forward on London session breakout."""
    print(f"\n{'═' * 80}")
    print(f"  🇬🇧 LONDON SESSION DEEP — {tf.upper()}")
    print(f"{'═' * 80}")

    # ── Per-pair on London ──
    print(f"\n{'─' * 70}")
    print(f"  📋 Per-Pair — London Breakout")
    print(f"{'─' * 70}")
    kpis_pp = []
    for pair in sorted(all_candles.keys()):
        single = {pair: all_candles[pair]}
        cfg = AsianConfig(asian_start_hour=8, asian_end_hour=16,
                          sl_atr_mult=2.0, cooldown_bars=2, max_positions=1)
        r = simulate_asian(single, cfg, balance, tf)
        r.label = pair
        kpis_pp.append(compute_kpis(r))
    print_table(kpis_pp, "Per-Pair — London Breakout")

    # ── Grid Phase 1: SL × TP ──
    print(f"\n{'─' * 70}")
    print(f"  🔍 Grid SL × TP — London")
    print(f"{'─' * 70}")
    kpis_g1 = []
    for sl_m in [1.0, 1.5, 2.0, 2.5, 3.0, 3.5]:
        for tp1, tp2 in [(0.015, 0.03), (0.02, 0.04), (0.02, 0.05), (0.03, 0.05),
                         (0.03, 0.06), (0.04, 0.06), (0.04, 0.08), (0.05, 0.10)]:
            label = f"SL{sl_m}_TP{int(tp1*100)}/{int(tp2*100)}"
            cfg = AsianConfig(asian_start_hour=8, asian_end_hour=16,
                              sl_atr_mult=sl_m, tp1_pct=tp1, tp2_pct=tp2, cooldown_bars=2)
            r = simulate_asian(all_candles, cfg, balance, tf)
            r.label = label
            kpis_g1.append(compute_kpis(r))
    print_table(sorted(kpis_g1, key=lambda k: -k["pf"])[:20],
                "Grid SL×TP — London — Top 20")

    # ── Grid Phase 2: Vol × MinRange (best SL/TP) ──
    best_g1 = max([k for k in kpis_g1 if k["trades"] >= 30], key=lambda k: k["pf"], default=None)
    if best_g1:
        parts = best_g1["label"].split("_")
        b_sl = float(parts[0].replace("SL", ""))
        tp_p = parts[1].replace("TP", "").split("/")
        b_tp1, b_tp2 = int(tp_p[0]) / 100, int(tp_p[1]) / 100
    else:
        b_sl, b_tp1, b_tp2 = 2.0, 0.03, 0.06

    print(f"\n{'─' * 70}")
    print(f"  🔍 Grid Vol × MinRange — London (SL={b_sl}, TP={b_tp1}/{b_tp2})")
    print(f"{'─' * 70}")
    kpis_g2 = []
    for vm in [0.0, 0.5, 1.0, 1.5, 2.0, 2.5]:
        for mr in [0.002, 0.003, 0.005, 0.008, 0.01, 0.015]:
            label = f"VOL{vm}_MR{int(mr*1000)}"
            cfg = AsianConfig(asian_start_hour=8, asian_end_hour=16,
                              sl_atr_mult=b_sl, tp1_pct=b_tp1, tp2_pct=b_tp2,
                              vol_mult=vm, min_range_pct=mr, cooldown_bars=2)
            r = simulate_asian(all_candles, cfg, balance, tf)
            r.label = label
            kpis_g2.append(compute_kpis(r))
    print_table(sorted(kpis_g2, key=lambda k: -k["pf"])[:15],
                "Grid Vol×MinRange — London — Top 15")

    # ── Grid Phase 3: Sizing ──
    best_g2 = max([k for k in kpis_g2 if k["trades"] >= 30], key=lambda k: k["pf"], default=None)
    if best_g2:
        parts = best_g2["label"].split("_")
        b_vm = float(parts[0].replace("VOL", ""))
        b_mr = int(parts[1].replace("MR", "")) / 1000
    else:
        b_vm, b_mr = 1.5, 0.005

    print(f"\n{'─' * 70}")
    print(f"  🔍 Grid Sizing — London (SL={b_sl}, TP={b_tp1}/{b_tp2}, VOL={b_vm}, MR={b_mr})")
    print(f"{'─' * 70}")
    kpis_g3 = []
    for risk in [0.01, 0.015, 0.02, 0.025, 0.03, 0.04, 0.05]:
        for mp in [1, 2, 3, 4]:
            label = f"R{int(risk*100)}%_MP{mp}"
            cfg = AsianConfig(asian_start_hour=8, asian_end_hour=16,
                              sl_atr_mult=b_sl, tp1_pct=b_tp1, tp2_pct=b_tp2,
                              vol_mult=b_vm, min_range_pct=b_mr,
                              risk_per_trade=risk, max_positions=mp, cooldown_bars=2)
            r = simulate_asian(all_candles, cfg, balance, tf)
            r.label = label
            kpis_g3.append(compute_kpis(r))
    print_table(sorted(kpis_g3, key=lambda k: -k["pf"])[:10],
                "Grid Sizing — London — Top 10")

    # ── Walk-Forward on best configs ──
    min_len = min(len(c) for c in all_candles.values())
    sb = 30

    wf_configs = [
        ("BASE_LON", AsianConfig(asian_start_hour=8, asian_end_hour=16, cooldown_bars=2)),
        (f"SL{b_sl}_LON", AsianConfig(asian_start_hour=8, asian_end_hour=16,
                                       sl_atr_mult=b_sl, tp1_pct=b_tp1, tp2_pct=b_tp2, cooldown_bars=2)),
        (f"BEST_LON", AsianConfig(asian_start_hour=8, asian_end_hour=16,
                                   sl_atr_mult=b_sl, tp1_pct=b_tp1, tp2_pct=b_tp2,
                                   vol_mult=b_vm, min_range_pct=b_mr, cooldown_bars=2)),
        ("SL3_TP36_LON", AsianConfig(asian_start_hour=8, asian_end_hour=16,
                                      sl_atr_mult=3.0, tp1_pct=0.03, tp2_pct=0.06, cooldown_bars=2)),
        ("SL25_TP35_LON", AsianConfig(asian_start_hour=8, asian_end_hour=16,
                                       sl_atr_mult=2.5, tp1_pct=0.03, tp2_pct=0.05, cooldown_bars=2)),
    ]

    for split_name, split_bar in [("4yr/2yr", int(min_len * 4 / 6)),
                                   ("3yr/3yr", int(min_len * 3 / 6))]:
        print(f"\n{'─' * 70}")
        print(f"  🔄 Walk-Forward London : {split_name}")
        print(f"{'─' * 70}")
        kpis_tr, kpis_te = [], []
        for label, cfg in wf_configs:
            rt = simulate_asian(all_candles, cfg, balance, tf, start_bar=sb, end_bar=split_bar)
            rt.label = f"{label}_TR"
            kpis_tr.append(compute_kpis(rt))
            rte = simulate_asian(all_candles, cfg, balance, tf, start_bar=split_bar, end_bar=min_len)
            rte.label = f"{label}_TE"
            kpis_te.append(compute_kpis(rte))

        print_table(kpis_tr, f"TRAIN ({split_name.split('/')[0]})")
        print_table(kpis_te, f"TEST ({split_name.split('/')[1]})")

        print(f"\n  {'Config':<22} {'PF_train':>8} {'PF_test':>8} {'Δ PF':>7} {'Robust?':>8}")
        print(f"  {'-' * 60}")
        for kt, kte in zip(kpis_tr, kpis_te):
            lbl = kt["label"].replace("_TR", "")
            d = kte["pf"] - kt["pf"]
            tag = "✅" if abs(d) < 0.3 and kte["pf"] >= 1.0 else ("⚠️ " if kte["pf"] >= 0.9 else "❌")
            print(f"  {lbl:<22} {kt['pf']:>8.2f} {kte['pf']:>8.2f} {d:>+7.2f} {tag:>8}")
        print(f"  {'-' * 60}")

    # Summary
    all_kpis = kpis_g1 + kpis_g2 + kpis_g3
    valid = [k for k in all_kpis if k["trades"] >= 50]
    if valid:
        best = max(valid, key=lambda k: k["pf"])
        print(f"\n{'═' * 80}")
        print(f"  🏆 MEILLEURE CONFIG LONDON : {best['label']}")
        print(f"     PF {best['pf']:.2f} | PnL {best['pnl']:+.2f}$ | {best['trades']} trades | DD {best['max_dd']:.1f}%")
        print(f"     Params : SL={b_sl}×ATR, TP={b_tp1}/{b_tp2}, VOL={b_vm}, MR={b_mr}")
        print(f"{'═' * 80}")
    return all_kpis


# ═════════════════════════════════════════════════════════════════════
#  MAIN
# ═════════════════════════════════════════════════════════════════════

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--years", type=int, default=6)
    ap.add_argument("--balance", type=float, default=500)
    ap.add_argument("--mode", default="all",
                    choices=["perpair", "grid", "walkforward", "short", "sessions", "london", "all"])
    ap.add_argument("--pairs", default=None, help="Comma-separated pairs override")
    args = ap.parse_args()

    end = datetime.now(timezone.utc)
    start = end - timedelta(days=args.years * 365)

    pairs = args.pairs.split(",") if args.pairs else ALL_PAIRS[:10]  # Top 10 most liquid
    tf = "4h"  # H4 confirmed as best TF

    print(f"\n{'═' * 80}")
    print(f"  🔬 ASIAN BREAKOUT — Deep Exploration")
    print(f"     {args.years} ans | ${args.balance:.0f} | {tf.upper()} | {len(pairs)} paires")
    print(f"     Paires : {', '.join(pairs)}")
    print(f"{'═' * 80}")

    # Download data
    all_candles: dict[str, list[Candle]] = {}
    for pair in pairs:
        try:
            all_candles[pair] = download_candles(pair, start, end, interval=tf)
            logger.info("  %s: %d bougies", pair, len(all_candles[pair]))
        except Exception as e:
            logger.warning("  %s: SKIP (%s)", pair, e)

    if not all_candles:
        print("❌ Aucune donnée chargée."); return

    modes = [args.mode] if args.mode != "all" else ["perpair", "grid", "walkforward", "short", "sessions"]

    for mode in modes:
        if mode == "perpair":
            run_perpair(all_candles, args.balance, tf)
        elif mode == "grid":
            run_grid(all_candles, args.balance, tf)
        elif mode == "walkforward":
            run_walkforward(all_candles, args.balance, tf)
        elif mode == "short":
            run_short_test(all_candles, args.balance, tf)
        elif mode == "sessions":
            run_sessions(all_candles, args.balance, tf)
        elif mode == "london":
            run_london_deep(all_candles, args.balance, tf)

    print(f"\n{'═' * 80}")
    print(f"  ✅ Exploration terminée")
    print(f"{'═' * 80}\n")


if __name__ == "__main__":
    main()
