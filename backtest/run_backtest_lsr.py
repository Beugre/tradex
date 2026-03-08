#!/usr/bin/env python3
"""
Backtest : LSR (Liquidation Spike Reversal)

Acheter les spikes de liquidation (wick + volume) sur crypto majors.

Règles :
  - Entry : wick_down ≥ 2× body, volume > 2× MA, RSI < 30, close > low + 40% range
  - SL : low - 0.5%
  - TP ladder : TP1 +2% (40%), TP2 +4% (40%), TP3 +6% (20%)
  - Breakeven après TP1
  - Filtre : skip si 24h change < -12% (news crash)
  - Paires : BTC, ETH, SOL, BNB, LINK

Usage :
    python -m backtest.run_backtest_lsr
    python -m backtest.run_backtest_lsr --years 6 --balance 500
    python -m backtest.run_backtest_lsr --years 6 --balance 500 --variants
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
class LSRConfig:
    """Configuration pour Liquidation Spike Reversal."""
    # ── Wick detection ──
    wick_body_ratio: float = 2.0       # wick_down ≥ N × body
    close_above_low_pct: float = 0.40  # close > low + 40% range

    # ── Volume ──
    vol_ma_period: int = 20
    vol_mult: float = 2.0              # volume > N × MA

    # ── RSI ──
    rsi_period: int = 14
    rsi_max: float = 30.0

    # ── Crash filter ──
    crash_lookback: int = 6            # bars pour calculer 24h change (6 H4 ou 24 H1)
    crash_threshold: float = -0.12     # skip si change < -12%

    # ── TP ladder ──
    tp1_pct: float = 0.02              # +2%
    tp2_pct: float = 0.04              # +4%
    tp3_pct: float = 0.06              # +6%
    tp1_share: float = 0.40
    tp2_share: float = 0.40
    tp3_share: float = 0.20

    # ── SL ──
    sl_below_low_pct: float = 0.005    # SL = low - 0.5%
    breakeven_after_tp1: bool = True

    # ── Risk ──
    risk_per_trade: float = 0.015      # 1.5% equity
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
    tp1_hit: bool = False
    tp2_hit: bool = False
    tp1_price: float = 0.0
    tp2_price: float = 0.0
    tp3_price: float = 0.0


@dataclass
class _PairState:
    cooldown_until: int = 0


# ══════════════════════════════════════════════════════════════════════════════
#  INDICATORS
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class LSRIndicators:
    rsi: list[float] = field(default_factory=list)
    vol_ma: list[float] = field(default_factory=list)


def compute_lsr_indicators(candles: list[Candle], cfg: LSRConfig) -> LSRIndicators:
    ind = LSRIndicators()
    ind.rsi = rsi_series(candles, cfg.rsi_period)
    volumes = [c.volume for c in candles]
    ind.vol_ma = sma(volumes, cfg.vol_ma_period)
    return ind


# ══════════════════════════════════════════════════════════════════════════════
#  SIMULATOR
# ══════════════════════════════════════════════════════════════════════════════

def simulate_lsr(
    all_candles: dict[str, list[Candle]],
    cfg: LSRConfig,
    initial_balance: float = 500.0,
) -> BacktestResult:
    balance = initial_balance
    positions: list[_OpenPos] = []
    closed_trades: list[Trade] = []
    equity_curve: list[float] = [balance]

    all_ind: dict[str, LSRIndicators] = {}
    for symbol, candles in all_candles.items():
        all_ind[symbol] = compute_lsr_indicators(candles, cfg)

    states: dict[str, _PairState] = {sym: _PairState() for sym in all_candles}
    min_len = min(len(c) for c in all_candles.values())

    start_bar = max(cfg.vol_ma_period + 5, cfg.rsi_period + 5, cfg.crash_lookback + 5)

    for bar_idx in range(start_bar, min_len):

        # ── Gestion positions ouvertes ──
        for pos in positions[:]:
            c = all_candles[pos.symbol][bar_idx]
            hold = bar_idx - pos.entry_bar

            # Check SL
            if c.low <= pos.sl_price:
                exit_price = pos.sl_price
                pnl_pct = (exit_price - pos.entry_price) / pos.entry_price
                pnl_usd = pos.remaining_size_usd * pnl_pct - pos.remaining_size_usd * TAKER_FEE
                balance += pos.remaining_size_usd + pnl_usd
                reason = "BE" if pos.tp1_hit else "SL"
                closed_trades.append(Trade(
                    symbol=pos.symbol, side="LONG",
                    entry_price=pos.entry_price, exit_price=exit_price,
                    size=pos.remaining_size, size_usd=pos.remaining_size_usd,
                    entry_time=pos.entry_ts, exit_time=c.timestamp,
                    pnl_usd=pnl_usd, pnl_pct=pnl_pct * 100,
                    exit_reason=reason, hold_bars=hold,
                ))
                positions.remove(pos)
                continue

            # TP1
            if not pos.tp1_hit and c.high >= pos.tp1_price:
                close_size = pos.initial_size * cfg.tp1_share
                close_size_usd = pos.initial_size_usd * cfg.tp1_share
                pnl_pct = (pos.tp1_price - pos.entry_price) / pos.entry_price
                pnl_usd = close_size_usd * pnl_pct - close_size_usd * MAKER_FEE
                balance += close_size_usd + pnl_usd
                closed_trades.append(Trade(
                    symbol=pos.symbol, side="LONG",
                    entry_price=pos.entry_price, exit_price=pos.tp1_price,
                    size=close_size, size_usd=close_size_usd,
                    entry_time=pos.entry_ts, exit_time=c.timestamp,
                    pnl_usd=pnl_usd, pnl_pct=pnl_pct * 100,
                    exit_reason="TP1", hold_bars=hold,
                ))
                pos.tp1_hit = True
                pos.remaining_size -= close_size
                pos.remaining_size_usd -= close_size_usd
                if cfg.breakeven_after_tp1:
                    pos.sl_price = pos.entry_price

            # TP2
            if not pos.tp2_hit and pos.tp1_hit and c.high >= pos.tp2_price:
                close_size = pos.initial_size * cfg.tp2_share
                close_size_usd = pos.initial_size_usd * cfg.tp2_share
                pnl_pct = (pos.tp2_price - pos.entry_price) / pos.entry_price
                pnl_usd = close_size_usd * pnl_pct - close_size_usd * MAKER_FEE
                balance += close_size_usd + pnl_usd
                closed_trades.append(Trade(
                    symbol=pos.symbol, side="LONG",
                    entry_price=pos.entry_price, exit_price=pos.tp2_price,
                    size=close_size, size_usd=close_size_usd,
                    entry_time=pos.entry_ts, exit_time=c.timestamp,
                    pnl_usd=pnl_usd, pnl_pct=pnl_pct * 100,
                    exit_reason="TP2", hold_bars=hold,
                ))
                pos.tp2_hit = True
                pos.remaining_size -= close_size
                pos.remaining_size_usd -= close_size_usd

            # TP3
            if pos.tp1_hit and pos.tp2_hit and c.high >= pos.tp3_price:
                close_size = pos.remaining_size
                close_size_usd = pos.remaining_size_usd
                if close_size_usd <= 0:
                    positions.remove(pos)
                    continue
                pnl_pct = (pos.tp3_price - pos.entry_price) / pos.entry_price
                pnl_usd = close_size_usd * pnl_pct - close_size_usd * MAKER_FEE
                balance += close_size_usd + pnl_usd
                closed_trades.append(Trade(
                    symbol=pos.symbol, side="LONG",
                    entry_price=pos.entry_price, exit_price=pos.tp3_price,
                    size=close_size, size_usd=close_size_usd,
                    entry_time=pos.entry_ts, exit_time=c.timestamp,
                    pnl_usd=pnl_usd, pnl_pct=pnl_pct * 100,
                    exit_reason="TP3", hold_bars=hold,
                ))
                positions.remove(pos)
                continue

            if pos.remaining_size_usd < 1:
                positions.remove(pos)

        # ── Détection de nouvelles entrées ──
        if len(positions) < cfg.max_positions and balance > 10:
            total_exposure = sum(p.remaining_size_usd for p in positions)
            current_eq = balance + total_exposure
            max_exposure = current_eq * cfg.max_exposure_pct

            for symbol in all_candles:
                if any(p.symbol == symbol for p in positions):
                    continue
                if len(positions) >= cfg.max_positions:
                    break

                state = states[symbol]
                if bar_idx < state.cooldown_until:
                    continue

                candles = all_candles[symbol]
                c = candles[bar_idx]
                ind = all_ind[symbol]

                # ── 1. Crash filter : skip si 24h change < threshold ──
                past = candles[bar_idx - cfg.crash_lookback]
                if past.close > 0:
                    change_24h = (c.close - past.close) / past.close
                    if change_24h < cfg.crash_threshold:
                        continue

                # ── 2. Wick detection ──
                candle_range = c.high - c.low
                if candle_range <= 0:
                    continue

                body = abs(c.close - c.open)
                wick_down = min(c.open, c.close) - c.low

                # wick_down ≥ ratio × body
                if body > 0:
                    if wick_down < cfg.wick_body_ratio * body:
                        continue
                else:
                    # Doji : wick doit exister
                    if wick_down <= 0:
                        continue

                # close > low + 40% range (rebond dans la bougie)
                if c.close < c.low + cfg.close_above_low_pct * candle_range:
                    continue

                # ── 3. Volume confirmation ──
                vol_ma = ind.vol_ma[bar_idx] if bar_idx < len(ind.vol_ma) else 0
                if vol_ma <= 0:
                    continue
                if c.volume < cfg.vol_mult * vol_ma:
                    continue

                # ── 4. RSI check ──
                rsi_val = ind.rsi[bar_idx] if bar_idx < len(ind.rsi) else 50
                if rsi_val >= cfg.rsi_max:
                    continue

                # ── 5. Sizing ──
                entry_price = c.close
                sl_price = c.low * (1 - cfg.sl_below_low_pct)
                sl_distance = entry_price - sl_price
                if sl_distance <= 0:
                    continue

                equity = balance + sum(p.remaining_size_usd for p in positions)
                risk_amount = equity * cfg.risk_per_trade
                size = risk_amount / sl_distance
                size_usd = size * entry_price

                remaining_exposure = max_exposure - total_exposure
                if size_usd > remaining_exposure:
                    size_usd = remaining_exposure
                    size = size_usd / entry_price if entry_price > 0 else 0

                if size_usd < 5:
                    continue

                # TP prices
                tp1_price = entry_price * (1 + cfg.tp1_pct)
                tp2_price = entry_price * (1 + cfg.tp2_pct)
                tp3_price = entry_price * (1 + cfg.tp3_pct)

                # Execute
                fee = size_usd * MAKER_FEE
                balance -= size_usd + fee
                total_exposure += size_usd

                positions.append(_OpenPos(
                    symbol=symbol, entry_price=entry_price,
                    sl_price=sl_price,
                    initial_size=size, initial_size_usd=size_usd,
                    remaining_size=size, remaining_size_usd=size_usd,
                    entry_bar=bar_idx, entry_ts=c.timestamp,
                    tp1_price=tp1_price, tp2_price=tp2_price, tp3_price=tp3_price,
                ))
                state.cooldown_until = bar_idx + cfg.cooldown_bars

        # ── Equity tracking ──
        pos_value = sum(
            p.remaining_size * all_candles[p.symbol][min(bar_idx, len(all_candles[p.symbol]) - 1)].close
            for p in positions
        )
        equity_curve.append(balance + pos_value)

    # Clôturer positions restantes
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

def _define_variants(run_variants: bool = False, timeframe: str = "4h") -> list[tuple[str, LSRConfig, dict, list[str] | None]]:
    configs = []

    # Adjust crash lookback for timeframe
    crash_lb = 6 if timeframe == "4h" else 24  # 6 H4 = 24h, 24 H1 = 24h

    # ── BASE ──
    configs.append((f"LSR_BASE_{timeframe.upper()}", LSRConfig(
        crash_lookback=crash_lb,
    ), {
        "entry": "Wick≥2×body + Vol>2×MA + RSI<30",
        "sl": "low-0.5%", "tp": "2/4/6% ladder",
        "tf": timeframe,
    }, None))

    # ── 3 paires ──
    configs.append((f"LSR_3P_{timeframe.upper()}", LSRConfig(
        crash_lookback=crash_lb,
    ), {
        "entry": "Wick≥2×body + Vol>2×MA + RSI<30",
        "pairs": "BTC,ETH,SOL",
        "tf": timeframe,
    }, ["BTC-USD", "ETH-USD", "SOL-USD"]))

    if run_variants:
        # ── V1 : RSI < 35 (plus permissif) ──
        configs.append((f"LSR_RSI35_{timeframe.upper()}", LSRConfig(
            rsi_max=35.0,
            crash_lookback=crash_lb,
        ), {"rsi": "<35"}, None))

        # ── V2 : RSI < 25 (plus strict) ──
        configs.append((f"LSR_RSI25_{timeframe.upper()}", LSRConfig(
            rsi_max=25.0,
            crash_lookback=crash_lb,
        ), {"rsi": "<25"}, None))

        # ── V3 : Wick ratio 1.5 (plus permissif) ──
        configs.append((f"LSR_WICK15_{timeframe.upper()}", LSRConfig(
            wick_body_ratio=1.5,
            crash_lookback=crash_lb,
        ), {"wick": "≥1.5×body"}, None))

        # ── V4 : Wick ratio 3 (plus strict) ──
        configs.append((f"LSR_WICK3_{timeframe.upper()}", LSRConfig(
            wick_body_ratio=3.0,
            crash_lookback=crash_lb,
        ), {"wick": "≥3×body"}, None))

        # ── V5 : Volume 1.5× (plus permissif) ──
        configs.append((f"LSR_VOL15_{timeframe.upper()}", LSRConfig(
            vol_mult=1.5,
            crash_lookback=crash_lb,
        ), {"vol": ">1.5×MA"}, None))

        # ── V6 : TP plus large 3/6/10% ──
        configs.append((f"LSR_WIDE_TP_{timeframe.upper()}", LSRConfig(
            tp1_pct=0.03,
            tp2_pct=0.06,
            tp3_pct=0.10,
            crash_lookback=crash_lb,
        ), {"tp": "3/6/10%"}, None))

        # ── V7 : TP serré 1/2/3% ──
        configs.append((f"LSR_TIGHT_TP_{timeframe.upper()}", LSRConfig(
            tp1_pct=0.01,
            tp2_pct=0.02,
            tp3_pct=0.03,
            crash_lookback=crash_lb,
        ), {"tp": "1/2/3%"}, None))

        # ── V8 : Sans breakeven ──
        configs.append((f"LSR_NO_BE_{timeframe.upper()}", LSRConfig(
            breakeven_after_tp1=False,
            crash_lookback=crash_lb,
        ), {"BE": "non"}, None))

        # ── V9 : Sans filtre crash ──
        configs.append((f"LSR_NO_CRASH_{timeframe.upper()}", LSRConfig(
            crash_threshold=-1.0,  # désactivé
            crash_lookback=crash_lb,
        ), {"crash_filter": "off"}, None))

        # ── V10 : Combo permissif (RSI35 + Wick1.5 + Vol1.5) ──
        configs.append((f"LSR_LOOSE_{timeframe.upper()}", LSRConfig(
            rsi_max=35.0,
            wick_body_ratio=1.5,
            vol_mult=1.5,
            crash_lookback=crash_lb,
        ), {"combo": "RSI35+Wick1.5+Vol1.5"}, None))

        # ── V11 : Risk 2.5% ──
        configs.append((f"LSR_RISK25_{timeframe.upper()}", LSRConfig(
            risk_per_trade=0.025,
            max_exposure_pct=0.60,
            crash_lookback=crash_lb,
        ), {"risk": "2.5%"}, None))

    return configs


# ══════════════════════════════════════════════════════════════════════════════
#  REPORTING
# ══════════════════════════════════════════════════════════════════════════════

def compute_kpis(result: BacktestResult) -> dict:
    trades = result.trades
    if not trades:
        return {
            "label": result.label, "trades": 0, "win_rate": 0, "pf": 0,
            "pnl": 0, "avg_pnl": 0, "max_dd": 0, "final": result.initial_balance,
            "rr": 0, "avg_hold": 0, "avg_win_pct": 0, "avg_loss_pct": 0,
        }

    wins = [t for t in trades if t.pnl_usd > 0]
    losses = [t for t in trades if t.pnl_usd <= 0]
    total_gains = sum(t.pnl_usd for t in wins) if wins else 0
    total_losses = abs(sum(t.pnl_usd for t in losses)) if losses else 0.001

    peak = result.equity_curve[0]
    max_dd = 0
    for eq in result.equity_curve:
        if eq > peak:
            peak = eq
        dd = (peak - eq) / peak * 100 if peak > 0 else 0
        if dd > max_dd:
            max_dd = dd

    avg_win = (total_gains / len(wins)) if wins else 0
    avg_loss = (total_losses / len(losses)) if losses else 0.001

    return {
        "label": result.label,
        "trades": len(trades),
        "win_rate": len(wins) / len(trades) * 100 if trades else 0,
        "pf": total_gains / total_losses if total_losses > 0 else 0,
        "pnl": result.final_equity - result.initial_balance,
        "avg_pnl": sum(t.pnl_usd for t in trades) / len(trades) if trades else 0,
        "max_dd": max_dd,
        "final": result.final_equity,
        "rr": avg_win / avg_loss if avg_loss > 0 else 0,
        "avg_hold": sum(t.hold_bars for t in trades) / len(trades) if trades else 0,
        "avg_win_pct": (sum(t.pnl_pct for t in wins) / len(wins)) if wins else 0,
        "avg_loss_pct": (sum(t.pnl_pct for t in losses) / len(losses)) if losses else 0,
    }


def print_table(kpis_list: list[dict], title: str) -> None:
    print(f"\n{'='*115}")
    print(f"  {title}")
    print(f"{'='*115}")
    hdr = f"{'Config':<22} {'Trades':>6} {'WR%':>7} {'PF':>7} {'PnL $':>9} {'Avg PnL':>8} {'Max DD%':>8} {'Final $':>9} {'R:R':>5} {'Avg Hold':>9}"
    print(hdr)
    print("-" * 115)
    for k in kpis_list:
        print(
            f"{k['label']:<22} {k['trades']:>6} {k['win_rate']:>6.1f}% {k['pf']:>7.2f} "
            f"{k['pnl']:>+9.2f}$ {k['avg_pnl']:>+7.2f}$ {k['max_dd']:>7.1f}% "
            f"{k['final']:>9.2f}$ {k['rr']:>5.2f} {k['avg_hold']:>8.1f}b"
        )
    print("-" * 115)
    best = max(kpis_list, key=lambda k: k["pf"])
    print(f"  Meilleur PF : {best['label']} (PF {best['pf']:.2f}, PnL {best['pnl']:+.2f}$, WR {best['win_rate']:.1f}%)")


def print_exit_breakdown(results: list[BacktestResult], prefix: str) -> None:
    print(f"\n{'='*70}")
    print(f"  {prefix} — Répartition des sorties")
    print(f"{'='*70}")
    for res in results:
        if not res.trades:
            continue
        counter = Counter(t.exit_reason for t in res.trades)
        print(f"\n  {res.label}:")
        for reason, count in counter.most_common():
            pct = count / len(res.trades) * 100
            avg = sum(t.pnl_usd for t in res.trades if t.exit_reason == reason) / count
            avg_p = sum(t.pnl_pct for t in res.trades if t.exit_reason == reason) / count
            print(f"    {reason:<16}: {count:>4} ({pct:>5.1f}%)  avg PnL: {avg:>+.2f}$  avg %: {avg_p:>+.2f}%")


def print_per_pair(results: list[BacktestResult]) -> None:
    # Print per-pair stats for the best result only (or first with trades)
    for res in results[:3]:  # Limit to top 3
        if not res.trades:
            continue
        print(f"\n{'='*70}")
        print(f"  {res.label} — Stats par paire")
        print(f"{'='*70}")
        pairs = sorted(set(t.symbol for t in res.trades))
        print(f"  {'Paire':<12} {'Trades':>6} {'WR%':>7} {'PnL $':>9} {'Avg Win%':>9} {'Avg Loss%':>10}")
        print(f"  {'-'*55}")
        for pair in pairs:
            pt = [t for t in res.trades if t.symbol == pair]
            w = [t for t in pt if t.pnl_usd > 0]
            l = [t for t in pt if t.pnl_usd <= 0]
            wr = len(w) / len(pt) * 100 if pt else 0
            pnl = sum(t.pnl_usd for t in pt)
            awp = sum(t.pnl_pct for t in w) / len(w) if w else 0
            alp = sum(t.pnl_pct for t in l) / len(l) if l else 0
            print(f"  {pair:<12} {len(pt):>6} {wr:>6.1f}% {pnl:>+9.2f}$ {awp:>+8.2f}% {alp:>+9.2f}%")


def plot_equity(results: list[BacktestResult], title: str, filename: str) -> None:
    fig, ax = plt.subplots(figsize=(14, 6))
    for res in results:
        if res.trades:
            ax.plot(res.equity_curve, label=res.label, linewidth=1)
    ax.axhline(y=results[0].initial_balance, color="grey", linestyle="--", alpha=0.5)
    ax.set_title(title)
    ax.set_xlabel("Bar")
    ax.set_ylabel("Equity ($)")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)
    path = OUTPUT_DIR / filename
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\n  Chart : {path}")


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    parser = argparse.ArgumentParser(description="Backtest Liquidation Spike Reversal")
    parser.add_argument("--years", type=int, default=6, help="Années de données")
    parser.add_argument("--balance", type=float, default=500, help="Capital initial ($)")
    parser.add_argument("--variants", action="store_true", help="Tester aussi les variantes")
    args = parser.parse_args()

    end = datetime.now(timezone.utc)
    start = end - timedelta(days=args.years * 365)

    # Test on both H4 and H1
    all_kpis: list[dict] = []
    all_results: list[BacktestResult] = []

    for tf in ["4h", "1h"]:
        print(f"\n{'═'*70}")
        print(f"  LSR — Téléchargement {tf.upper()} ({args.years} ans)")
        print(f"  Paires : {', '.join(PAIRS)}")
        print(f"{'═'*70}")

        all_candles: dict[str, list[Candle]] = {}
        for pair in PAIRS:
            logger.info("  %s %s…", pair, tf)
            candles = download_candles(pair, start, end, interval=tf)
            all_candles[pair] = candles
            n_days = len(candles) // (6 if tf == "4h" else 24)
            logger.info("    %s : %d bougies %s (%d jours)", pair, len(candles), tf, n_days)

        configs = _define_variants(run_variants=args.variants, timeframe=tf)
        results: list[BacktestResult] = []
        kpis_list: list[dict] = []

        for label, cfg, desc, pairs_override in configs:
            logger.info("  %s…", label)
            candles_for_run = {k: v for k, v in all_candles.items() if k in pairs_override} if pairs_override else all_candles
            if not candles_for_run:
                continue
            result = simulate_lsr(candles_for_run, cfg, initial_balance=args.balance)
            result.label = label
            result.config_desc = desc
            results.append(result)
            kpis = compute_kpis(result)
            kpis_list.append(kpis)

        n_bars = min(len(c) for c in all_candles.values())
        print_table(kpis_list, f"LSR — {tf.upper()} ({n_bars} bars, {args.years} ans, ${args.balance:.0f})")
        print_exit_breakdown(results, f"LSR {tf.upper()}")
        print_per_pair(results)

        all_kpis.extend(kpis_list)
        all_results.extend(results)

    plot_equity(all_results, "Liquidation Spike Reversal (H4 + H1)", "lsr_equity.png")

    # Résumé global
    valid = [k for k in all_kpis if k["trades"] >= 10]
    if valid:
        best = max(valid, key=lambda k: k["pf"])
        print(f"\n{'═'*70}")
        if best["pf"] >= 1.5:
            print(f"  ✅ PROMETTEUR : {best['label']} — PF {best['pf']:.2f}, WR {best['win_rate']:.1f}%, PnL {best['pnl']:+.2f}$")
            print(f"     DD {best['max_dd']:.1f}% — R:R {best['rr']:.2f} — {best['trades']} trades")
        elif best["pf"] >= 1.0:
            print(f"  ⚠️  MARGINAL : {best['label']} — PF {best['pf']:.2f}, PnL {best['pnl']:+.2f}$, {best['trades']} trades")
        else:
            print(f"  ❌ NON RENTABLE : Meilleur PF = {best['pf']:.2f} ({best['label']}, {best['trades']} trades)")
        print(f"{'═'*70}\n")
    else:
        print(f"\n{'═'*70}")
        print(f"  ⚠️  PAS ASSEZ DE TRADES (< 10) sur toutes les configs")
        print(f"{'═'*70}\n")


if __name__ == "__main__":
    main()
