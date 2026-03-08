#!/usr/bin/env python3
"""
Backtest : Asian Range Breakout (Volatility Expansion)

Calcule le range de la session asiatique (00:00–08:00 UTC),
puis achète le breakout du high avec confirmation volume.

Règles LONG :
  - Asian range = high/low entre 00:00 et 08:00 UTC
  - Breakout : close > asian_high AND volume > avg_volume
  - SL : 1.5 × ATR (ou asian_low)
  - TP : 2-5% ou trailing

Usage :
    python -m backtest.run_backtest_asian --years 6 --balance 500 --variants
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
from src.core.momentum_engine import sma, atr_series

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(message)s")
logger = logging.getLogger(__name__)

OUTPUT_DIR = Path(__file__).parent / "output"
OUTPUT_DIR.mkdir(exist_ok=True)

PAIRS = ["BTC-USD", "ETH-USD", "SOL-USD", "BNB-USD", "LINK-USD"]
MAKER_FEE = 0.0
TAKER_FEE = 0.0009


@dataclass
class AsianConfig:
    # ── Session definition ──
    asian_start_hour: int = 0          # UTC
    asian_end_hour: int = 8            # UTC

    # ── Breakout confirmation ──
    vol_ma_period: int = 20
    vol_mult: float = 1.5
    min_range_pct: float = 0.005       # Range minimum 0.5% pour éviter le bruit

    # ── ATR ──
    atr_period: int = 14

    # ── SL ──
    sl_mode: str = "atr"               # "atr" or "asian_low"
    sl_atr_mult: float = 1.5

    # ── TP ──
    tp1_pct: float = 0.02              # +2%
    tp2_pct: float = 0.04              # +4%
    tp1_share: float = 0.50
    tp2_share: float = 0.50
    breakeven_after_tp1: bool = True

    # ── Risk ──
    risk_per_trade: float = 0.015
    max_positions: int = 4
    max_exposure_pct: float = 0.50

    # ── Cooldown ──
    cooldown_bars: int = 8             # ~8h H1 ou 32h H4


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
    symbol: str; entry_price: float; sl_price: float
    initial_size: float; initial_size_usd: float
    remaining_size: float; remaining_size_usd: float
    entry_bar: int; entry_ts: int
    tp1_price: float = 0.0; tp2_price: float = 0.0
    tp1_hit: bool = False


@dataclass
class _PairState:
    cooldown_until: int = 0
    # Asian session tracking
    asian_high: float = 0.0
    asian_low: float = float("inf")
    session_bars: int = 0
    session_complete: bool = False


def _bar_hour_utc(candle: Candle) -> int:
    """Extract UTC hour from candle timestamp (ms)."""
    return datetime.fromtimestamp(candle.timestamp / 1000, tz=timezone.utc).hour


def simulate_asian(
    all_candles: dict[str, list[Candle]],
    cfg: AsianConfig,
    initial_balance: float = 500.0,
    tf: str = "1h",
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
    start_bar = max(cfg.atr_period + 5, cfg.vol_ma_period + 5)

    for bar_idx in range(start_bar, min_len):

        # ── Manage open positions ──
        for pos in positions[:]:
            c = all_candles[pos.symbol][bar_idx]
            hold = bar_idx - pos.entry_bar

            if c.low <= pos.sl_price:
                ep = pos.sl_price
                pnl_pct = (ep - pos.entry_price) / pos.entry_price
                pnl_usd = pos.remaining_size_usd * pnl_pct - pos.remaining_size_usd * TAKER_FEE
                balance += pos.remaining_size_usd + pnl_usd
                reason = "BE" if pos.tp1_hit and pos.sl_price >= pos.entry_price else "SL"
                closed.append(Trade(pos.symbol, "LONG", pos.entry_price, ep,
                    pos.remaining_size, pos.remaining_size_usd,
                    pos.entry_ts, c.timestamp, pnl_usd, pnl_pct*100, reason, hold))
                positions.remove(pos); continue

            if not pos.tp1_hit and c.high >= pos.tp1_price:
                cs = pos.initial_size * cfg.tp1_share
                cu = pos.initial_size_usd * cfg.tp1_share
                pp = (pos.tp1_price - pos.entry_price) / pos.entry_price
                pu = cu * pp - cu * MAKER_FEE
                balance += cu + pu
                closed.append(Trade(pos.symbol, "LONG", pos.entry_price, pos.tp1_price,
                    cs, cu, pos.entry_ts, c.timestamp, pu, pp*100, "TP1", hold))
                pos.tp1_hit = True; pos.remaining_size -= cs; pos.remaining_size_usd -= cu
                if cfg.breakeven_after_tp1: pos.sl_price = pos.entry_price

            if pos.tp1_hit and c.high >= pos.tp2_price:
                pp = (pos.tp2_price - pos.entry_price) / pos.entry_price
                pu = pos.remaining_size_usd * pp - pos.remaining_size_usd * MAKER_FEE
                balance += pos.remaining_size_usd + pu
                closed.append(Trade(pos.symbol, "LONG", pos.entry_price, pos.tp2_price,
                    pos.remaining_size, pos.remaining_size_usd,
                    pos.entry_ts, c.timestamp, pu, pp*100, "TP2", hold))
                positions.remove(pos); continue

            if pos.remaining_size_usd < 1: positions.remove(pos)

        # ── Track asian session & detect breakouts ──
        for symbol in all_candles:
            c = all_candles[symbol][bar_idx]
            st = states[symbol]
            hour = _bar_hour_utc(c)

            # H1: each bar = 1 hour, track 00-07 UTC as Asian
            # H4: bars at 00:00, 04:00, 08:00 etc
            if tf == "1h":
                in_asian = cfg.asian_start_hour <= hour < cfg.asian_end_hour
            else:  # 4h
                # For H4: 00:00 and 04:00 bars are "Asian"
                in_asian = hour < cfg.asian_end_hour

            if in_asian:
                if st.session_bars == 0:  # New session
                    st.asian_high = c.high
                    st.asian_low = c.low
                else:
                    st.asian_high = max(st.asian_high, c.high)
                    st.asian_low = min(st.asian_low, c.low)
                st.session_bars += 1
                st.session_complete = False
            else:
                if st.session_bars > 0 and not st.session_complete:
                    st.session_complete = True

                # Reset for next session when we're past asian and next day starts
                if hour >= cfg.asian_end_hour and st.session_complete:
                    # Check for breakout
                    if (any(p.symbol == symbol for p in positions) or
                        bar_idx < st.cooldown_until or
                        len(positions) >= cfg.max_positions or
                        balance <= 10):
                        # Reset session at end of day
                        if hour >= 20 or (tf == "4h" and hour >= 16):
                            st.session_bars = 0
                        continue

                    range_pct = (st.asian_high - st.asian_low) / st.asian_low if st.asian_low > 0 else 0
                    if range_pct < cfg.min_range_pct:
                        if hour >= 20 or (tf == "4h" and hour >= 16):
                            st.session_bars = 0
                        continue

                    # LONG breakout: close > asian_high + vol confirmation
                    vol_ma = all_vol_ma[symbol][bar_idx] if bar_idx < len(all_vol_ma[symbol]) else 0
                    atr_val = all_atr[symbol][bar_idx] if bar_idx < len(all_atr[symbol]) else 0

                    if c.close > st.asian_high and vol_ma > 0 and c.volume >= cfg.vol_mult * vol_ma:
                        entry_price = c.close
                        if cfg.sl_mode == "asian_low":
                            sl_price = st.asian_low * 0.998
                        else:
                            sl_price = entry_price - cfg.sl_atr_mult * atr_val if atr_val > 0 else st.asian_low

                        sl_dist = entry_price - sl_price
                        if sl_dist <= 0:
                            if hour >= 20: st.session_bars = 0
                            continue

                        total_exp = sum(p.remaining_size_usd for p in positions)
                        equity = balance + total_exp
                        max_exp = equity * cfg.max_exposure_pct
                        risk_amount = equity * cfg.risk_per_trade
                        size = risk_amount / sl_dist
                        size_usd = size * entry_price
                        rem = max_exp - total_exp
                        if size_usd > rem: size_usd = rem; size = size_usd / entry_price if entry_price > 0 else 0
                        if size_usd < 5:
                            if hour >= 20: st.session_bars = 0
                            continue

                        balance -= size_usd
                        positions.append(_OpenPos(
                            symbol=symbol, entry_price=entry_price, sl_price=sl_price,
                            initial_size=size, initial_size_usd=size_usd,
                            remaining_size=size, remaining_size_usd=size_usd,
                            entry_bar=bar_idx, entry_ts=c.timestamp,
                            tp1_price=entry_price * (1 + cfg.tp1_pct),
                            tp2_price=entry_price * (1 + cfg.tp2_pct),
                        ))
                        st.cooldown_until = bar_idx + cfg.cooldown_bars
                        st.session_complete = False  # consumed

                # Reset session at end of day
                if hour >= 20 or (tf == "4h" and hour >= 16):
                    st.session_bars = 0

        pv = sum(
            p.remaining_size * all_candles[p.symbol][min(bar_idx, len(all_candles[p.symbol])-1)].close
            for p in positions
        )
        equity_curve.append(balance + pv)

    for pos in positions:
        last = all_candles[pos.symbol][min_len - 1]
        pp = (last.close - pos.entry_price) / pos.entry_price
        pu = pos.remaining_size_usd * pp - pos.remaining_size_usd * TAKER_FEE
        balance += pos.remaining_size_usd + pu
        closed.append(Trade(pos.symbol, "LONG", pos.entry_price, last.close,
            pos.remaining_size, pos.remaining_size_usd,
            pos.entry_ts, last.timestamp, pu, pp*100, "END", min_len-1-pos.entry_bar))

    return BacktestResult("", closed, equity_curve, initial_balance,
        equity_curve[-1] if equity_curve else initial_balance, {})


def _define_variants(run_variants: bool, tf: str) -> list[tuple[str, AsianConfig, dict, list[str] | None]]:
    configs = []
    cd = 8 if tf == "1h" else 2  # cooldown adapted
    configs.append((f"ASIAN_BASE_{tf.upper()}", AsianConfig(cooldown_bars=cd), {"base": True}, None))
    configs.append((f"ASIAN_3P_{tf.upper()}", AsianConfig(cooldown_bars=cd), {"pairs": "3"}, ["BTC-USD", "ETH-USD", "SOL-USD"]))

    if run_variants:
        configs.append((f"ASIAN_NOVOL_{tf.upper()}", AsianConfig(vol_mult=0.0, cooldown_bars=cd), {"vol": "off"}, None))
        configs.append((f"ASIAN_VOL2_{tf.upper()}", AsianConfig(vol_mult=2.0, cooldown_bars=cd), {"vol": "2×"}, None))
        configs.append((f"ASIAN_SL_AL_{tf.upper()}", AsianConfig(sl_mode="asian_low", cooldown_bars=cd), {"sl": "asian_low"}, None))
        configs.append((f"ASIAN_SL2_{tf.upper()}", AsianConfig(sl_atr_mult=2.0, cooldown_bars=cd), {"sl": "2×ATR"}, None))
        configs.append((f"ASIAN_TP35_{tf.upper()}", AsianConfig(tp1_pct=0.03, tp2_pct=0.05, cooldown_bars=cd), {"tp": "3/5%"}, None))
        configs.append((f"ASIAN_TP12_{tf.upper()}", AsianConfig(tp1_pct=0.01, tp2_pct=0.02, cooldown_bars=cd), {"tp": "1/2%"}, None))
        configs.append((f"ASIAN_NOBE_{tf.upper()}", AsianConfig(breakeven_after_tp1=False, cooldown_bars=cd), {"be": "off"}, None))
        configs.append((f"ASIAN_R25_{tf.upper()}", AsianConfig(risk_per_trade=0.025, max_exposure_pct=0.6, cooldown_bars=cd), {"risk": "2.5%"}, None))
        configs.append((f"ASIAN_MR1_{tf.upper()}", AsianConfig(min_range_pct=0.01, cooldown_bars=cd), {"min_range": "1%"}, None))
        configs.append((f"ASIAN_MR02_{tf.upper()}", AsianConfig(min_range_pct=0.002, cooldown_bars=cd), {"min_range": "0.2%"}, None))

    return configs


# ── Reporting ─────────────────────────────────────────────────

def compute_kpis(r):
    t=r.trades
    if not t: return {"label":r.label,"trades":0,"win_rate":0,"pf":0,"pnl":0,"avg_pnl":0,"max_dd":0,"final":r.initial_balance,"rr":0,"avg_hold":0}
    w=[x for x in t if x.pnl_usd>0];l=[x for x in t if x.pnl_usd<=0]
    tg=sum(x.pnl_usd for x in w)if w else 0;tl=abs(sum(x.pnl_usd for x in l))if l else .001
    pk=r.equity_curve[0];md=0
    for eq in r.equity_curve:
        if eq>pk:pk=eq
        dd=(pk-eq)/pk*100 if pk>0 else 0
        if dd>md:md=dd
    aw=tg/len(w)if w else 0;al=tl/len(l)if l else .001
    return {"label":r.label,"trades":len(t),"win_rate":len(w)/len(t)*100,"pf":tg/tl,"pnl":r.final_equity-r.initial_balance,
            "avg_pnl":sum(x.pnl_usd for x in t)/len(t),"max_dd":md,"final":r.final_equity,"rr":aw/al,"avg_hold":sum(x.hold_bars for x in t)/len(t)}


def print_table(kl,title):
    print(f"\n{'='*120}\n  {title}\n{'='*120}")
    print(f"{'Config':<22} {'Trades':>6} {'WR%':>7} {'PF':>7} {'PnL $':>9} {'Avg PnL':>8} {'Max DD%':>8} {'Final $':>9} {'R:R':>5} {'Hold':>6}")
    print("-"*120)
    for k in kl:
        print(f"{k['label']:<22} {k['trades']:>6} {k['win_rate']:>6.1f}% {k['pf']:>7.2f} "
              f"{k['pnl']:>+9.2f}$ {k['avg_pnl']:>+7.2f}$ {k['max_dd']:>7.1f}% "
              f"{k['final']:>9.2f}$ {k['rr']:>5.2f} {k['avg_hold']:>5.1f}b")
    print("-"*120)
    if kl:b=max(kl,key=lambda k:k["pf"]);print(f"  Meilleur PF : {b['label']} (PF {b['pf']:.2f}, PnL {b['pnl']:+.2f}$)")


def print_exits(results):
    print(f"\n{'='*70}\n  Répartition des sorties\n{'='*70}")
    for r in results:
        if not r.trades:continue
        ct=Counter(t.exit_reason for t in r.trades)
        print(f"\n  {r.label}:")
        for reason,cnt in ct.most_common():
            pct=cnt/len(r.trades)*100;avg=sum(t.pnl_usd for t in r.trades if t.exit_reason==reason)/cnt
            print(f"    {reason:<14}: {cnt:>5} ({pct:>5.1f}%)  avg: {avg:>+.2f}$")


def plot_eq(results,title,fn):
    fig,ax=plt.subplots(figsize=(14,6))
    for r in results:
        if r.trades:ax.plot(r.equity_curve,label=r.label,lw=1)
    ax.axhline(y=results[0].initial_balance,color="grey",ls="--",alpha=.5)
    ax.set_title(title);ax.legend(fontsize=7);ax.grid(True,alpha=.3)
    p=OUTPUT_DIR/fn;fig.savefig(p,dpi=150,bbox_inches="tight");plt.close(fig)
    print(f"\n  Chart : {p}")


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--years",type=int,default=6)
    ap.add_argument("--balance",type=float,default=500)
    ap.add_argument("--variants",action="store_true")
    args=ap.parse_args()

    end=datetime.now(timezone.utc);start=end-timedelta(days=args.years*365)
    all_kpis,all_results=[],[]

    for tf in ["1h", "4h"]:
        print(f"\n{'═'*70}\n  Asian Range Breakout — {tf.upper()} ({args.years} ans)\n{'═'*70}")
        all_candles={}
        for pair in PAIRS:
            all_candles[pair]=download_candles(pair,start,end,interval=tf)
            logger.info("  %s %s: %d bougies",pair,tf,len(all_candles[pair]))

        configs=_define_variants(args.variants,tf)
        results,kpis_list=[],[]
        for label,cfg,desc,po in configs:
            cd={k:v for k,v in all_candles.items() if k in po}if po else all_candles
            if not cd:continue
            r=simulate_asian(cd,cfg,args.balance,tf);r.label=label;r.config_desc=desc
            results.append(r);kpis_list.append(compute_kpis(r))

        print_table(kpis_list,f"Asian Breakout — {tf.upper()} ({args.years} ans, ${args.balance:.0f})")
        print_exits(results)
        all_kpis.extend(kpis_list);all_results.extend(results)

    plot_eq(all_results,"Asian Range Breakout","asian_equity.png")

    valid=[k for k in all_kpis if k["trades"]>=10]
    if valid:
        b=max(valid,key=lambda k:k["pf"])
        tag="✅ PROMETTEUR"if b["pf"]>=1.5 else("⚠️  MARGINAL"if b["pf"]>=1.0 else"❌ NON RENTABLE")
        print(f"\n{'═'*70}\n  {tag} : {b['label']} — PF {b['pf']:.2f}, PnL {b['pnl']:+.2f}$, {b['trades']} trades\n{'═'*70}\n")
    else:
        print(f"\n{'═'*70}\n  ⚠️  PAS ASSEZ DE TRADES\n{'═'*70}\n")


if __name__=="__main__":
    main()
