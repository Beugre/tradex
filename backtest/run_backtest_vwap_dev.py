#!/usr/bin/env python3
"""
Backtest : VWAP Deviation Strategy

Mean-reversion vers le VWAP quand le prix s'éloigne de ≥ 2σ.

Règles LONG only :
  - price < VWAP - 2σ
  - RSI < 35
  - volume > 1.5× MA(20)
  - SL : 1.5 × ATR
  - TP : VWAP (ou VWAP - 1σ pour TP partiel)

Usage :
    python -m backtest.run_backtest_vwap_dev --years 6 --balance 500 --variants
"""

from __future__ import annotations

import argparse
import logging
import math
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from backtest.data_loader import download_candles
from src.core.models import Candle
from src.core.momentum_engine import ema, sma, atr_series, rsi_series

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(message)s")
logger = logging.getLogger(__name__)

OUTPUT_DIR = Path(__file__).parent / "output"
OUTPUT_DIR.mkdir(exist_ok=True)

PAIRS = ["BTC-USD", "ETH-USD", "SOL-USD", "BNB-USD", "LINK-USD"]
MAKER_FEE = 0.0
TAKER_FEE = 0.0009


# ══════════════════════════════════════════════════════════════════════════════
#  CONFIG
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class VWAPConfig:
    # ── VWAP params ──
    vwap_period: int = 30              # Rolling VWAP lookback (bars)
    sigma_mult: float = 2.0            # Entry at VWAP - N×σ

    # ── RSI ──
    rsi_period: int = 14
    rsi_max: float = 35.0

    # ── Volume ──
    vol_ma_period: int = 20
    vol_mult: float = 1.5

    # ── ATR ──
    atr_period: int = 14

    # ── SL ──
    sl_atr_mult: float = 1.5

    # ── TP ──
    tp_mode: str = "vwap"              # "vwap" = TP at VWAP, "sigma" = TP at VWAP-1σ
    tp_fixed_pct: float = 0.03         # fallback TP if VWAP too close (3%)

    # ── Partial close ──
    partial_close_at_1sigma: bool = True  # Close 50% at VWAP - 1σ
    partial_pct: float = 0.50

    # ── Risk ──
    risk_per_trade: float = 0.015
    max_positions: int = 4
    max_exposure_pct: float = 0.50

    # ── Cooldown ──
    cooldown_bars: int = 3


# ══════════════════════════════════════════════════════════════════════════════
#  DATA STRUCTURES
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class Trade:
    symbol: str
    side: str
    entry_price: float
    exit_price: float
    size: float
    size_usd: float
    entry_time: int
    exit_time: int
    pnl_usd: float
    pnl_pct: float
    exit_reason: str
    hold_bars: int = 0


@dataclass
class BacktestResult:
    label: str
    trades: list[Trade]
    equity_curve: list[float]
    initial_balance: float
    final_equity: float
    config_desc: dict


@dataclass
class _OpenPos:
    symbol: str
    entry_price: float
    sl_price: float
    initial_size: float
    initial_size_usd: float
    remaining_size: float
    remaining_size_usd: float
    entry_bar: int
    entry_ts: int
    tp_vwap: float = 0.0
    tp_1sigma: float = 0.0
    partial_hit: bool = False


@dataclass
class _PairState:
    cooldown_until: int = 0


# ══════════════════════════════════════════════════════════════════════════════
#  VWAP CALCULATION
# ══════════════════════════════════════════════════════════════════════════════

def compute_rolling_vwap(candles: list[Candle], period: int) -> tuple[list[float], list[float]]:
    """Compute rolling VWAP and standard deviation.

    Returns (vwap, sigma) lists, same length as candles.
    """
    n = len(candles)
    vwap_vals = [0.0] * n
    sigma_vals = [0.0] * n

    for i in range(period, n):
        window = candles[i - period + 1:i + 1]
        total_vol = sum(c.volume for c in window)
        if total_vol <= 0:
            vwap_vals[i] = candles[i].close
            sigma_vals[i] = 0
            continue

        # VWAP = Σ(typical_price × volume) / Σ(volume)
        tp_vol_sum = sum(((c.high + c.low + c.close) / 3) * c.volume for c in window)
        vwap = tp_vol_sum / total_vol
        vwap_vals[i] = vwap

        # Standard deviation of typical price from VWAP, weighted by volume
        variance = sum(
            c.volume * (((c.high + c.low + c.close) / 3) - vwap) ** 2
            for c in window
        ) / total_vol
        sigma_vals[i] = math.sqrt(variance) if variance > 0 else 0

    return vwap_vals, sigma_vals


# ══════════════════════════════════════════════════════════════════════════════
#  SIMULATOR
# ══════════════════════════════════════════════════════════════════════════════

def simulate_vwap(
    all_candles: dict[str, list[Candle]],
    cfg: VWAPConfig,
    initial_balance: float = 500.0,
) -> BacktestResult:
    balance = initial_balance
    positions: list[_OpenPos] = []
    closed_trades: list[Trade] = []
    equity_curve: list[float] = [balance]

    # Pre-compute indicators
    all_vwap: dict[str, tuple[list[float], list[float]]] = {}
    all_rsi: dict[str, list[float]] = {}
    all_atr: dict[str, list[float]] = {}
    all_vol_ma: dict[str, list[float]] = {}

    for symbol, candles in all_candles.items():
        all_vwap[symbol] = compute_rolling_vwap(candles, cfg.vwap_period)
        all_rsi[symbol] = rsi_series(candles, cfg.rsi_period)
        all_atr[symbol] = atr_series(candles, cfg.atr_period)
        all_vol_ma[symbol] = sma([c.volume for c in candles], cfg.vol_ma_period)

    states: dict[str, _PairState] = {sym: _PairState() for sym in all_candles}
    min_len = min(len(c) for c in all_candles.values())
    start_bar = max(cfg.vwap_period + 5, cfg.rsi_period + 5, cfg.atr_period + 5)

    for bar_idx in range(start_bar, min_len):

        # ── Manage open positions ──
        for pos in positions[:]:
            c = all_candles[pos.symbol][bar_idx]
            hold = bar_idx - pos.entry_bar

            # SL check
            if c.low <= pos.sl_price:
                pnl_pct = (pos.sl_price - pos.entry_price) / pos.entry_price
                pnl_usd = pos.remaining_size_usd * pnl_pct - pos.remaining_size_usd * TAKER_FEE
                balance += pos.remaining_size_usd + pnl_usd
                closed_trades.append(Trade(
                    symbol=pos.symbol, side="LONG",
                    entry_price=pos.entry_price, exit_price=pos.sl_price,
                    size=pos.remaining_size, size_usd=pos.remaining_size_usd,
                    entry_time=pos.entry_ts, exit_time=c.timestamp,
                    pnl_usd=pnl_usd, pnl_pct=pnl_pct * 100,
                    exit_reason="SL", hold_bars=hold,
                ))
                positions.remove(pos)
                continue

            # Update dynamic VWAP TP
            vwap_val = all_vwap[pos.symbol][0][bar_idx]
            sigma_val = all_vwap[pos.symbol][1][bar_idx]
            if vwap_val > 0:
                pos.tp_vwap = vwap_val
                pos.tp_1sigma = vwap_val - sigma_val if sigma_val > 0 else vwap_val

            # Partial close at VWAP - 1σ
            if cfg.partial_close_at_1sigma and not pos.partial_hit and pos.tp_1sigma > 0:
                if c.high >= pos.tp_1sigma:
                    close_size = pos.initial_size * cfg.partial_pct
                    close_usd = pos.initial_size_usd * cfg.partial_pct
                    pnl_pct = (pos.tp_1sigma - pos.entry_price) / pos.entry_price
                    pnl_usd = close_usd * pnl_pct - close_usd * MAKER_FEE
                    balance += close_usd + pnl_usd
                    closed_trades.append(Trade(
                        symbol=pos.symbol, side="LONG",
                        entry_price=pos.entry_price, exit_price=pos.tp_1sigma,
                        size=close_size, size_usd=close_usd,
                        entry_time=pos.entry_ts, exit_time=c.timestamp,
                        pnl_usd=pnl_usd, pnl_pct=pnl_pct * 100,
                        exit_reason="TP_1S", hold_bars=hold,
                    ))
                    pos.partial_hit = True
                    pos.remaining_size -= close_size
                    pos.remaining_size_usd -= close_usd
                    # Move SL to breakeven
                    pos.sl_price = pos.entry_price

            # Full TP at VWAP
            if pos.tp_vwap > 0 and c.high >= pos.tp_vwap:
                pnl_pct = (pos.tp_vwap - pos.entry_price) / pos.entry_price
                pnl_usd = pos.remaining_size_usd * pnl_pct - pos.remaining_size_usd * MAKER_FEE
                balance += pos.remaining_size_usd + pnl_usd
                closed_trades.append(Trade(
                    symbol=pos.symbol, side="LONG",
                    entry_price=pos.entry_price, exit_price=pos.tp_vwap,
                    size=pos.remaining_size, size_usd=pos.remaining_size_usd,
                    entry_time=pos.entry_ts, exit_time=c.timestamp,
                    pnl_usd=pnl_usd, pnl_pct=pnl_pct * 100,
                    exit_reason="TP_VWAP", hold_bars=hold,
                ))
                positions.remove(pos)
                continue

            if pos.remaining_size_usd < 1:
                positions.remove(pos)

        # ── New entries ──
        if len(positions) < cfg.max_positions and balance > 10:
            total_exp = sum(p.remaining_size_usd for p in positions)
            current_eq = balance + total_exp
            max_exp = current_eq * cfg.max_exposure_pct

            for symbol in all_candles:
                if any(p.symbol == symbol for p in positions):
                    continue
                if len(positions) >= cfg.max_positions:
                    break

                state = states[symbol]
                if bar_idx < state.cooldown_until:
                    continue

                c = all_candles[symbol][bar_idx]
                vwap_val = all_vwap[symbol][0][bar_idx]
                sigma_val = all_vwap[symbol][1][bar_idx]
                rsi_val = all_rsi[symbol][bar_idx] if bar_idx < len(all_rsi[symbol]) else 50
                atr_val = all_atr[symbol][bar_idx] if bar_idx < len(all_atr[symbol]) else 0
                vol_ma = all_vol_ma[symbol][bar_idx] if bar_idx < len(all_vol_ma[symbol]) else 0

                if vwap_val <= 0 or sigma_val <= 0 or atr_val <= 0 or vol_ma <= 0:
                    continue

                # ── Conditions LONG ──
                lower_band = vwap_val - cfg.sigma_mult * sigma_val
                if c.close >= lower_band:
                    continue
                if rsi_val >= cfg.rsi_max:
                    continue
                if c.volume < cfg.vol_mult * vol_ma:
                    continue

                # ── Sizing ──
                entry_price = c.close
                sl_price = entry_price - cfg.sl_atr_mult * atr_val
                sl_dist = entry_price - sl_price
                if sl_dist <= 0:
                    continue

                equity = balance + sum(p.remaining_size_usd for p in positions)
                risk_amount = equity * cfg.risk_per_trade
                size = risk_amount / sl_dist
                size_usd = size * entry_price

                remaining_exp = max_exp - total_exp
                if size_usd > remaining_exp:
                    size_usd = remaining_exp
                    size = size_usd / entry_price if entry_price > 0 else 0

                if size_usd < 5:
                    continue

                tp_vwap = vwap_val
                tp_1sigma = vwap_val - sigma_val

                fee = size_usd * MAKER_FEE
                balance -= size_usd + fee
                total_exp += size_usd

                positions.append(_OpenPos(
                    symbol=symbol, entry_price=entry_price,
                    sl_price=sl_price,
                    initial_size=size, initial_size_usd=size_usd,
                    remaining_size=size, remaining_size_usd=size_usd,
                    entry_bar=bar_idx, entry_ts=c.timestamp,
                    tp_vwap=tp_vwap, tp_1sigma=tp_1sigma,
                ))
                state.cooldown_until = bar_idx + cfg.cooldown_bars

        # Equity
        pos_value = sum(
            p.remaining_size * all_candles[p.symbol][min(bar_idx, len(all_candles[p.symbol]) - 1)].close
            for p in positions
        )
        equity_curve.append(balance + pos_value)

    # Close remaining
    for pos in positions:
        last = all_candles[pos.symbol][min_len - 1]
        pnl_pct = (last.close - pos.entry_price) / pos.entry_price
        pnl_usd = pos.remaining_size_usd * pnl_pct - pos.remaining_size_usd * TAKER_FEE
        balance += pos.remaining_size_usd + pnl_usd
        closed_trades.append(Trade(
            symbol=pos.symbol, side="LONG",
            entry_price=pos.entry_price, exit_price=last.close,
            size=pos.remaining_size, size_usd=pos.remaining_size_usd,
            entry_time=pos.entry_ts, exit_time=last.timestamp,
            pnl_usd=pnl_usd, pnl_pct=pnl_pct * 100,
            exit_reason="END", hold_bars=min_len - 1 - pos.entry_bar,
        ))

    return BacktestResult(
        label="", trades=closed_trades, equity_curve=equity_curve,
        initial_balance=initial_balance,
        final_equity=equity_curve[-1] if equity_curve else initial_balance,
        config_desc={},
    )


# ══════════════════════════════════════════════════════════════════════════════
#  VARIANTS
# ══════════════════════════════════════════════════════════════════════════════

def _define_variants(run_variants: bool, tf: str = "4h") -> list[tuple[str, VWAPConfig, dict, list[str] | None]]:
    configs = []
    configs.append((f"VWAP_BASE_{tf.upper()}", VWAPConfig(), {"entry": "VWAP-2σ, RSI<35"}, None))
    configs.append((f"VWAP_3P_{tf.upper()}", VWAPConfig(), {"pairs": "BTC,ETH,SOL"}, ["BTC-USD", "ETH-USD", "SOL-USD"]))

    if run_variants:
        configs.append((f"VWAP_1.5S_{tf.upper()}", VWAPConfig(sigma_mult=1.5), {"sigma": "1.5"}, None))
        configs.append((f"VWAP_2.5S_{tf.upper()}", VWAPConfig(sigma_mult=2.5), {"sigma": "2.5"}, None))
        configs.append((f"VWAP_3S_{tf.upper()}", VWAPConfig(sigma_mult=3.0), {"sigma": "3.0"}, None))
        configs.append((f"VWAP_RSI40_{tf.upper()}", VWAPConfig(rsi_max=40.0), {"rsi": "<40"}, None))
        configs.append((f"VWAP_RSI30_{tf.upper()}", VWAPConfig(rsi_max=30.0), {"rsi": "<30"}, None))
        configs.append((f"VWAP_SL2_{tf.upper()}", VWAPConfig(sl_atr_mult=2.0), {"sl": "2×ATR"}, None))
        configs.append((f"VWAP_NOVOL_{tf.upper()}", VWAPConfig(vol_mult=0.0), {"vol": "off"}, None))
        configs.append((f"VWAP_P20_{tf.upper()}", VWAPConfig(vwap_period=20), {"period": "20"}, None))
        configs.append((f"VWAP_P50_{tf.upper()}", VWAPConfig(vwap_period=50), {"period": "50"}, None))
        configs.append((f"VWAP_NOPT_{tf.upper()}", VWAPConfig(partial_close_at_1sigma=False), {"partial": "off"}, None))
        configs.append((f"VWAP_LOOSE_{tf.upper()}", VWAPConfig(
            sigma_mult=1.5, rsi_max=40.0, vol_mult=1.0,
        ), {"combo": "loose"}, None))

    return configs


# ══════════════════════════════════════════════════════════════════════════════
#  REPORTING (shared pattern)
# ══════════════════════════════════════════════════════════════════════════════

def compute_kpis(result: BacktestResult) -> dict:
    trades = result.trades
    if not trades:
        return {"label": result.label, "trades": 0, "win_rate": 0, "pf": 0,
                "pnl": 0, "avg_pnl": 0, "max_dd": 0, "final": result.initial_balance, "rr": 0, "avg_hold": 0}
    wins = [t for t in trades if t.pnl_usd > 0]
    losses = [t for t in trades if t.pnl_usd <= 0]
    tg = sum(t.pnl_usd for t in wins) if wins else 0
    tl = abs(sum(t.pnl_usd for t in losses)) if losses else 0.001
    peak = result.equity_curve[0]; max_dd = 0
    for eq in result.equity_curve:
        if eq > peak: peak = eq
        dd = (peak - eq) / peak * 100 if peak > 0 else 0
        if dd > max_dd: max_dd = dd
    aw = tg / len(wins) if wins else 0
    al = tl / len(losses) if losses else 0.001
    return {"label": result.label, "trades": len(trades),
            "win_rate": len(wins) / len(trades) * 100, "pf": tg / tl if tl > 0 else 0,
            "pnl": result.final_equity - result.initial_balance,
            "avg_pnl": sum(t.pnl_usd for t in trades) / len(trades),
            "max_dd": max_dd, "final": result.final_equity, "rr": aw / al if al > 0 else 0,
            "avg_hold": sum(t.hold_bars for t in trades) / len(trades)}


def print_table(kpis_list, title):
    print(f"\n{'='*120}\n  {title}\n{'='*120}")
    print(f"{'Config':<22} {'Trades':>6} {'WR%':>7} {'PF':>7} {'PnL $':>9} {'Avg PnL':>8} {'Max DD%':>8} {'Final $':>9} {'R:R':>5} {'Hold':>6}")
    print("-" * 120)
    for k in kpis_list:
        print(f"{k['label']:<22} {k['trades']:>6} {k['win_rate']:>6.1f}% {k['pf']:>7.2f} "
              f"{k['pnl']:>+9.2f}$ {k['avg_pnl']:>+7.2f}$ {k['max_dd']:>7.1f}% "
              f"{k['final']:>9.2f}$ {k['rr']:>5.2f} {k['avg_hold']:>5.1f}b")
    print("-" * 120)
    if kpis_list:
        best = max(kpis_list, key=lambda k: k["pf"])
        print(f"  Meilleur PF : {best['label']} (PF {best['pf']:.2f}, PnL {best['pnl']:+.2f}$)")


def print_exits(results):
    print(f"\n{'='*70}\n  Répartition des sorties\n{'='*70}")
    for res in results:
        if not res.trades: continue
        counter = Counter(t.exit_reason for t in res.trades)
        print(f"\n  {res.label}:")
        for reason, cnt in counter.most_common():
            pct = cnt / len(res.trades) * 100
            avg = sum(t.pnl_usd for t in res.trades if t.exit_reason == reason) / cnt
            print(f"    {reason:<12}: {cnt:>4} ({pct:>5.1f}%)  avg: {avg:>+.2f}$")


def print_pairs(results):
    for res in results[:3]:
        if not res.trades: continue
        print(f"\n{'='*60}\n  {res.label} — Par paire\n{'='*60}")
        for pair in sorted(set(t.symbol for t in res.trades)):
            pt = [t for t in res.trades if t.symbol == pair]
            w = [t for t in pt if t.pnl_usd > 0]
            wr = len(w) / len(pt) * 100 if pt else 0
            pnl = sum(t.pnl_usd for t in pt)
            print(f"  {pair:<12} {len(pt):>5} trades  WR {wr:>5.1f}%  PnL {pnl:>+8.2f}$")


def plot_eq(results, title, fn):
    fig, ax = plt.subplots(figsize=(14, 6))
    for r in results:
        if r.trades: ax.plot(r.equity_curve, label=r.label, linewidth=1)
    ax.axhline(y=results[0].initial_balance, color="grey", ls="--", alpha=.5)
    ax.set_title(title); ax.set_xlabel("Bar"); ax.set_ylabel("Equity ($)")
    ax.legend(fontsize=7); ax.grid(True, alpha=.3)
    p = OUTPUT_DIR / fn; fig.savefig(p, dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"\n  Chart : {p}")


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--years", type=int, default=6)
    ap.add_argument("--balance", type=float, default=500)
    ap.add_argument("--variants", action="store_true")
    args = ap.parse_args()

    end = datetime.now(timezone.utc)
    start = end - timedelta(days=args.years * 365)

    all_kpis, all_results = [], []

    for tf in ["4h", "1h"]:
        print(f"\n{'═'*70}\n  VWAP Deviation — {tf.upper()} ({args.years} ans)\n{'═'*70}")
        all_candles = {}
        for pair in PAIRS:
            all_candles[pair] = download_candles(pair, start, end, interval=tf)
            logger.info("  %s %s: %d bougies", pair, tf, len(all_candles[pair]))

        configs = _define_variants(args.variants, tf)
        results, kpis_list = [], []
        for label, cfg, desc, po in configs:
            cd = {k: v for k, v in all_candles.items() if k in po} if po else all_candles
            if not cd: continue
            r = simulate_vwap(cd, cfg, args.balance); r.label = label; r.config_desc = desc
            results.append(r); kpis_list.append(compute_kpis(r))

        print_table(kpis_list, f"VWAP Deviation — {tf.upper()}")
        print_exits(results); print_pairs(results)
        all_kpis.extend(kpis_list); all_results.extend(results)

    plot_eq(all_results, "VWAP Deviation (H4 + H1)", "vwap_dev_equity.png")

    valid = [k for k in all_kpis if k["trades"] >= 10]
    if valid:
        best = max(valid, key=lambda k: k["pf"])
        print(f"\n{'═'*70}")
        tag = "✅ PROMETTEUR" if best["pf"] >= 1.5 else ("⚠️  MARGINAL" if best["pf"] >= 1.0 else "❌ NON RENTABLE")
        print(f"  {tag} : {best['label']} — PF {best['pf']:.2f}, PnL {best['pnl']:+.2f}$, {best['trades']} trades")
        print(f"{'═'*70}\n")


if __name__ == "__main__":
    main()
