#!/usr/bin/env python3
"""
Backtest — RSI Dip-Buy (Spot, pas de SL)

Strategie ultra-simple :
  - Achat quand RSI(14) < 35
  - Pas de stop-loss
  - Vente a un TP fixe (+5%, +10%, ou autres variantes)

Paires : BTC, ETH, SOL, XRP, BNB, LINK
Capital : $1,000 par defaut
Timeframe : 4H par defaut

Variantes A/B :
  - TP +5% vs +10% vs +15% vs +20%
  - RSI < 30 vs < 35 vs < 40
  - Avec/sans filtre volume
  - Risk 5% vs 10% vs 20% du capital par position

Usage:
    python3 -m backtest.run_backtest_rsi_dipbuy --compare
    python3 -m backtest.run_backtest_rsi_dipbuy --balance 1000 --years 2
    python3 -m backtest.run_backtest_rsi_dipbuy --interval 1h --years 1
"""

from __future__ import annotations

import argparse
import logging
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from backtest.data_loader import download_candles
from src.core.models import Candle

logging.basicConfig(level=logging.WARNING, format="[%(asctime)s] %(message)s")
logger = logging.getLogger(__name__)

OUTPUT_DIR = Path(__file__).parent / "output"

# ── Univers de paires ─────────────────────────────────────────────────────────
PAIRS_BIG5  = ["BTC-USD", "ETH-USD", "SOL-USD", "BNB-USD", "XRP-USD"]
PAIRS_ALTS5 = ["LINK-USD", "ADA-USD", "AVAX-USD", "DOT-USD", "DOGE-USD"]
PAIRS_ALL10 = PAIRS_BIG5 + PAIRS_ALTS5
PAIRS = PAIRS_ALL10  # par defaut : 10 paires


# ── Config ─────────────────────────────────────────────────────────────────────


@dataclass
class DipBuyConfig:
    """Configuration de la strategie RSI dip-buy."""
    name: str = "RSI_DIP_BUY"

    # ── Filtrage des paires ──
    pairs_filter: list = field(default_factory=list)  # vide = toutes les paires disponibles

    # ── RSI ──
    rsi_period: int = 14
    rsi_entry: float = 35.0         # Achat quand RSI < ce seuil

    # ── TP / SL ──
    tp_pct: float = 0.15            # Take profit en % (+15% par defaut)
    sl_enabled: bool = False        # Pas de stop-loss
    sl_pct: float = 0.0             # Inutilise si sl_enabled=False

    # ── Trailing stop ──
    trailing_enabled: bool = False
    trailing_activation_pct: float = 0.05   # Active quand prix >= entree * 1.05
    trailing_lock_pct: float = 0.02         # Garantit au moins +2% a l'activation
    trailing_distance_pct: float = 0.03     # Trail 3% sous le pic (→ stop ≈ +2% a activation)

    # ── Risk / Sizing ──
    risk_pct: float = 0.20          # 20% du capital par position
    max_simultaneous: int = 10      # max 10 positions, 1 par paire (exposition 100%)

    # ── Filtre volume ──
    volume_filter: bool = False
    volume_mult: float = 1.2        # volume > N * SMA(20)
    volume_ma_period: int = 20

    # ── Cooldown ──
    cooldown_bars: int = 3          # barres minimum entre 2 trades meme paire

    # ── Max duration ──
    max_bars_in_trade: int = 0      # 0 = illimite (pas de force close)

    # ── Fees ──
    entry_fee_pct: float = 0.0      # Maker 0% (Revolut X)
    exit_fee_pct: float = 0.0009    # Taker 0.09% (Revolut X)


# ── Structures ─────────────────────────────────────────────────────────────────


@dataclass
class DipTrade:
    symbol: str
    entry_bar: int
    entry_price: float
    entry_ts: int
    tp_price: float
    size: float
    rsi_at_entry: float = 0.0
    exit_bar: int = 0
    exit_price: float = 0.0
    exit_ts: int = 0
    exit_reason: str = ""
    pnl_usd: float = 0.0
    pnl_pct: float = 0.0
    fees: float = 0.0
    duration_bars: int = 0
    # Trailing state
    trailing_active: bool = False
    max_price_seen: float = 0.0


@dataclass
class EquityPoint:
    ts: int
    equity: float


# ── Indicateurs ────────────────────────────────────────────────────────────────


def rsi_series(closes: list[float], period: int = 14) -> list[float]:
    """RSI via EMA des gains / pertes."""
    n = len(closes)
    rsi = [50.0] * n
    if n < period + 1:
        return rsi
    gains = [0.0] * n
    losses = [0.0] * n
    for i in range(1, n):
        diff = closes[i] - closes[i - 1]
        if diff > 0:
            gains[i] = diff
        else:
            losses[i] = -diff
    avg_gain = sum(gains[1: period + 1]) / period
    avg_loss = sum(losses[1: period + 1]) / period
    for i in range(period, n):
        if i > period:
            avg_gain = (avg_gain * (period - 1) + gains[i]) / period
            avg_loss = (avg_loss * (period - 1) + losses[i]) / period
        if avg_loss == 0:
            rsi[i] = 100.0
        else:
            rs = avg_gain / avg_loss
            rsi[i] = 100.0 - 100.0 / (1 + rs)
    return rsi


def sma_series(values: list[float], period: int) -> list[float]:
    """SMA classique."""
    n = len(values)
    sma = [0.0] * n
    for i in range(n):
        if i < period - 1:
            sma[i] = sum(values[: i + 1]) / (i + 1)
        else:
            sma[i] = sum(values[i - period + 1: i + 1]) / period
    return sma


# ── Simulation ─────────────────────────────────────────────────────────────────


def run_pair(candles: list[Candle], cfg: DipBuyConfig, balance: float,
             symbol: str) -> tuple[list[DipTrade], list[EquityPoint], float]:
    """Simule la strategie dip-buy sur une paire."""
    n = len(candles)
    if n < cfg.rsi_period + 2:
        return [], [], balance

    closes = [c.close for c in candles]
    volumes = [c.volume for c in candles]
    rsi = rsi_series(closes, cfg.rsi_period)
    vol_sma = sma_series(volumes, cfg.volume_ma_period) if cfg.volume_filter else []

    trades: list[DipTrade] = []
    equity_curve: list[EquityPoint] = []
    open_trades: list[DipTrade] = []
    cooldown_until: int = 0

    for i in range(cfg.rsi_period + 1, n):
        c = candles[i]

        # ── Gestion sorties ──
        new_open: list[DipTrade] = []
        for t in open_trades:
            closed = False

            # Mise a jour du pic (pour trailing)
            if cfg.trailing_enabled:
                t.max_price_seen = max(t.max_price_seen, c.high)

            # TP atteint
            if c.high >= t.tp_price:
                t.exit_price = t.tp_price
                t.exit_reason = "TP"
                closed = True

            # Trailing stop (apres activation a +trailing_activation_pct)
            elif cfg.trailing_enabled and t.max_price_seen >= t.entry_price * (1 + cfg.trailing_activation_pct):
                t.trailing_active = True
                trail_stop = max(
                    t.entry_price * (1 + cfg.trailing_lock_pct),
                    t.max_price_seen * (1 - cfg.trailing_distance_pct),
                )
                if c.low <= trail_stop:
                    t.exit_price = trail_stop
                    t.exit_reason = "TRAIL"
                    closed = True

            # SL fixe si actif
            elif cfg.sl_enabled and cfg.sl_pct > 0:
                sl_price = t.entry_price * (1 - cfg.sl_pct)
                if c.low <= sl_price:
                    t.exit_price = sl_price
                    t.exit_reason = "SL"
                    closed = True

            # Max duration
            elif cfg.max_bars_in_trade > 0 and (i - t.entry_bar) >= cfg.max_bars_in_trade:
                t.exit_price = c.close
                t.exit_reason = "TIMEOUT"
                closed = True

            if closed:
                t.exit_bar = i
                t.exit_ts = c.timestamp
                exit_fee = t.size * t.exit_price * cfg.exit_fee_pct
                entry_fee = t.size * t.entry_price * cfg.entry_fee_pct
                t.fees = entry_fee + exit_fee
                t.pnl_usd = (t.exit_price - t.entry_price) * t.size - t.fees
                t.pnl_pct = t.pnl_usd / (t.size * t.entry_price) if t.size * t.entry_price > 0 else 0
                t.duration_bars = t.exit_bar - t.entry_bar
                balance += t.pnl_usd + (t.size * t.entry_price)  # Rendre le capital investi + PnL
                trades.append(t)
            else:
                new_open.append(t)

        open_trades = new_open

        # ── Signal d'entree ──
        if i > cooldown_until and len(open_trades) < cfg.max_simultaneous:
            # Verifier qu'on n'a pas deja une position sur cette paire
            if not any(t.symbol == symbol for t in open_trades):
                if rsi[i] < cfg.rsi_entry:
                    # Filtre volume optionnel
                    if cfg.volume_filter and vol_sma:
                        if volumes[i] < cfg.volume_mult * vol_sma[i]:
                            equity_curve.append(EquityPoint(c.timestamp, balance + _unrealized(open_trades, c.close)))
                            continue

                    # Sizing : % fixe du capital
                    alloc = balance * cfg.risk_pct
                    if alloc < 1.0 or c.close <= 0:
                        equity_curve.append(EquityPoint(c.timestamp, balance + _unrealized(open_trades, c.close)))
                        continue

                    size = alloc / c.close
                    tp_price = c.close * (1 + cfg.tp_pct)

                    trade = DipTrade(
                        symbol=symbol,
                        entry_bar=i,
                        entry_price=c.close,
                        entry_ts=c.timestamp,
                        tp_price=tp_price,
                        size=size,
                        rsi_at_entry=rsi[i],
                    )
                    balance -= alloc  # Retirer le capital investi
                    open_trades.append(trade)
                    cooldown_until = i + cfg.cooldown_bars

        # Equity = cash + unrealized
        equity_curve.append(EquityPoint(c.timestamp, balance + _unrealized(open_trades, c.close)))

    # Force-close les positions ouvertes a la fin
    if open_trades:
        last = candles[-1]
        for t in open_trades:
            t.exit_bar = n - 1
            t.exit_price = last.close
            t.exit_ts = last.timestamp
            t.exit_reason = "END"
            exit_fee = t.size * t.exit_price * cfg.exit_fee_pct
            entry_fee = t.size * t.entry_price * cfg.entry_fee_pct
            t.fees = exit_fee + entry_fee
            t.pnl_usd = (t.exit_price - t.entry_price) * t.size - t.fees
            t.pnl_pct = t.pnl_usd / (t.size * t.entry_price) if t.size * t.entry_price > 0 else 0
            t.duration_bars = t.exit_bar - t.entry_bar
            balance += t.pnl_usd + (t.size * t.entry_price)
            trades.append(t)

    return trades, equity_curve, balance


def _unrealized(open_trades: list[DipTrade], current_price: float) -> float:
    """PnL non-realise des positions ouvertes."""
    return sum((current_price - t.entry_price) * t.size for t in open_trades)


# ── Variantes A/B ──────────────────────────────────────────────────────────────


def get_variants() -> list[DipBuyConfig]:
    """Retourne les variantes a comparer — configuration V2.

    Groupes :
      G1 — Sweep RSI (Big5, TP+15%) : RSI 25/30/35/40/45
      G2 — Sweep TP (Big5, RSI<35) : TP 5/10/15/20%
      G3 — Trailing (Big5, RSI<35) : sans vs avec trailing +5%/+2%
      G4 — Altcoins (Alts5, RSI<35, TP+15%)
      G5 — All 10 paires : meilleurs params sur univers complet
    """
    return [
        # ══ G1 — Sweep RSI sur Big5, TP+15% ══════════════════════════════════════
        DipBuyConfig(name="B5_RSI25_TP15", pairs_filter=PAIRS_BIG5, rsi_entry=25.0, tp_pct=0.15),
        DipBuyConfig(name="B5_RSI30_TP15", pairs_filter=PAIRS_BIG5, rsi_entry=30.0, tp_pct=0.15),
        DipBuyConfig(name="B5_RSI35_TP15", pairs_filter=PAIRS_BIG5, rsi_entry=35.0, tp_pct=0.15),  # REFERENCE
        DipBuyConfig(name="B5_RSI40_TP15", pairs_filter=PAIRS_BIG5, rsi_entry=40.0, tp_pct=0.15),
        DipBuyConfig(name="B5_RSI45_TP15", pairs_filter=PAIRS_BIG5, rsi_entry=45.0, tp_pct=0.15),

        # ══ G2 — Sweep TP sur Big5, RSI<35 ══════════════════════════════════════
        DipBuyConfig(name="B5_RSI35_TP5",  pairs_filter=PAIRS_BIG5, rsi_entry=35.0, tp_pct=0.05),
        DipBuyConfig(name="B5_RSI35_TP10", pairs_filter=PAIRS_BIG5, rsi_entry=35.0, tp_pct=0.10),
        # B5_RSI35_TP15 deja dans G1
        DipBuyConfig(name="B5_RSI35_TP20", pairs_filter=PAIRS_BIG5, rsi_entry=35.0, tp_pct=0.20),

        # ══ G3 — Trailing stop (Big5, activation +5%, lock +2%) ══════════════════
        DipBuyConfig(name="B5_RSI35_TP15_TRAIL",
                     pairs_filter=PAIRS_BIG5, rsi_entry=35.0, tp_pct=0.15,
                     trailing_enabled=True),
        DipBuyConfig(name="B5_RSI30_TP15_TRAIL",
                     pairs_filter=PAIRS_BIG5, rsi_entry=30.0, tp_pct=0.15,
                     trailing_enabled=True),
        DipBuyConfig(name="B5_RSI35_TP10_TRAIL",
                     pairs_filter=PAIRS_BIG5, rsi_entry=35.0, tp_pct=0.10,
                     trailing_enabled=True),

        # ══ G4 — Altcoins seuls (Alts5, RSI<35, TP+15%) ═════════════════════════
        DipBuyConfig(name="A5_RSI30_TP15",  pairs_filter=PAIRS_ALTS5, rsi_entry=30.0, tp_pct=0.15),
        DipBuyConfig(name="A5_RSI35_TP15",  pairs_filter=PAIRS_ALTS5, rsi_entry=35.0, tp_pct=0.15),
        DipBuyConfig(name="A5_RSI35_TP10",  pairs_filter=PAIRS_ALTS5, rsi_entry=35.0, tp_pct=0.10),

        # ══ G5 — All 10 paires (meilleurs params) ════════════════════════════════
        DipBuyConfig(name="ALL10_RSI30_TP15",  rsi_entry=30.0, tp_pct=0.15),
        DipBuyConfig(name="ALL10_RSI35_TP15",  rsi_entry=35.0, tp_pct=0.15),
        DipBuyConfig(name="ALL10_RSI35_TP10",  rsi_entry=35.0, tp_pct=0.10),
        DipBuyConfig(name="ALL10_RSI35_T_TRAIL",
                     rsi_entry=35.0, tp_pct=0.15, trailing_enabled=True),
        DipBuyConfig(name="ALL10_RSI40_TP15",  rsi_entry=40.0, tp_pct=0.15),
    ]


# ── Rapport ────────────────────────────────────────────────────────────────────


def _compute_metrics(trades: list[DipTrade], equity_curve: list[EquityPoint],
                     initial_balance: float, final_balance: float,
                     years: float) -> dict:
    """Calcule les metriques de performance."""
    n = len(trades)
    total_return = (final_balance - initial_balance) / initial_balance if initial_balance > 0 else 0
    cagr = (final_balance / initial_balance) ** (1 / max(years, 0.01)) - 1 if final_balance > 0 else 0

    # Drawdown
    peak = initial_balance
    max_dd = 0.0
    for pt in equity_curve:
        peak = max(peak, pt.equity)
        dd = (pt.equity - peak) / peak if peak > 0 else 0
        max_dd = min(max_dd, dd)

    # Sharpe
    returns = []
    for i in range(1, len(equity_curve)):
        prev = equity_curve[i - 1].equity
        if prev > 0:
            returns.append((equity_curve[i].equity - prev) / prev)
    sharpe = 0.0
    if returns:
        import math
        mean_r = sum(returns) / len(returns)
        std_r = (sum((r - mean_r) ** 2 for r in returns) / len(returns)) ** 0.5
        if std_r > 0:
            sharpe = (mean_r / std_r) * math.sqrt(365 * 6)  # annualise H4

    # Trades stats
    if n:
        wins = [t for t in trades if t.pnl_usd > 0]
        losses = [t for t in trades if t.pnl_usd <= 0]
        win_rate = len(wins) / n
        gross_profit = sum(t.pnl_usd for t in wins) or 0
        gross_loss = abs(sum(t.pnl_usd for t in losses)) or 1e-9
        profit_factor = gross_profit / gross_loss
        avg_pnl = sum(t.pnl_usd for t in trades) / n
        avg_pnl_pct = sum(t.pnl_pct for t in trades) / n
        avg_duration = sum(t.duration_bars for t in trades) / n
        total_fees = sum(t.fees for t in trades)
        best = max(trades, key=lambda t: t.pnl_usd)
        worst = min(trades, key=lambda t: t.pnl_usd)

        # Max consecutive losses
        max_consec = 0
        consec = 0
        for t in trades:
            if t.pnl_usd <= 0:
                consec += 1
                max_consec = max(max_consec, consec)
            else:
                consec = 0
    else:
        win_rate = profit_factor = avg_pnl = avg_pnl_pct = avg_duration = 0
        total_fees = 0
        best = worst = None
        max_consec = 0

    # Par paire
    by_pair: dict[str, dict] = {}
    pair_trades: dict[str, list[DipTrade]] = defaultdict(list)
    for t in trades:
        pair_trades[t.symbol].append(t)
    for sym, pts in sorted(pair_trades.items()):
        w = [t for t in pts if t.pnl_usd > 0]
        by_pair[sym] = {
            "n": len(pts),
            "wr": len(w) / len(pts) if pts else 0,
            "pnl": sum(t.pnl_usd for t in pts),
            "avg_dur": sum(t.duration_bars for t in pts) / len(pts) if pts else 0,
        }

    # Par motif de sortie
    by_exit: dict[str, dict] = {}
    exit_trades: dict[str, list[DipTrade]] = defaultdict(list)
    for t in trades:
        exit_trades[t.exit_reason].append(t)
    for reason, ets in sorted(exit_trades.items()):
        w = [t for t in ets if t.pnl_usd > 0]
        by_exit[reason] = {
            "n": len(ets),
            "wr": len(w) / len(ets) if ets else 0,
            "pnl": sum(t.pnl_usd for t in ets),
        }

    return {
        "total_return": total_return,
        "cagr": cagr,
        "max_drawdown": max_dd,
        "sharpe": sharpe,
        "win_rate": win_rate,
        "profit_factor": profit_factor,
        "n_trades": n,
        "avg_pnl": avg_pnl,
        "avg_pnl_pct": avg_pnl_pct,
        "avg_duration": avg_duration,
        "total_fees": total_fees,
        "max_consec_losses": max_consec,
        "best": best,
        "worst": worst,
        "by_pair": by_pair,
        "by_exit": by_exit,
        "final_balance": final_balance,
    }


def _print_report(cfg: DipBuyConfig, metrics: dict, initial_balance: float,
                  years: float) -> None:
    """Affiche le rapport console."""
    m = metrics
    sep = "=" * 80
    print(f"\n{sep}")
    print(f"  RSI DIP-BUY — {cfg.name}")
    print(f"  Capital: ${initial_balance:,.0f} | RSI < {cfg.rsi_entry:.0f}"
          f" | TP: +{cfg.tp_pct:.0%} | SL: {'OFF' if not cfg.sl_enabled else f'-{cfg.sl_pct:.0%}'}"
          f" | Risk/trade: {cfg.risk_pct:.0%}")
    print(f"  Fees: {cfg.entry_fee_pct:.2%} + {cfg.exit_fee_pct:.2%}")
    print(sep)

    print(f"\n  RESULTATS GLOBAUX")
    print("  " + "-" * 76)
    print(f"  Capital final      : ${m['final_balance']:,.2f} ({m['total_return']:+.1%})")
    print(f"  CAGR               : {m['cagr']:.1%}")
    print(f"  Max Drawdown       : {m['max_drawdown']:.1%}")
    print(f"  Sharpe Ratio       : {m['sharpe']:.2f}")
    print(f"  Win Rate           : {m['win_rate']:.1%} ({int(m['win_rate'] * m['n_trades'])}/{m['n_trades']})")
    print(f"  Profit Factor      : {m['profit_factor']:.2f}")
    print(f"  Trades             : {m['n_trades']}")
    print(f"  PnL moyen          : ${m['avg_pnl']:+.2f} ({m['avg_pnl_pct']:+.3%})")
    print(f"  Duree moy. trade   : {m['avg_duration']:.0f} barres")
    print(f"  Total fees         : ${m['total_fees']:.2f}")
    print(f"  Max pertes consec. : {m['max_consec_losses']}")

    if m["best"]:
        b = m["best"]
        print(f"  Meilleur trade     : ${b.pnl_usd:+.2f} ({b.pnl_pct:+.1%}) {b.symbol}")
    if m["worst"]:
        w = m["worst"]
        print(f"  Pire trade         : ${w.pnl_usd:+.2f} ({w.pnl_pct:+.1%}) {w.symbol}")

    # Par paire
    if m["by_pair"]:
        print(f"\n  PAR PAIRE")
        print("  " + "-" * 76)
        for sym, s in m["by_pair"].items():
            print(f"  {sym:12s} : {s['n']:4d} trades | WR {s['wr']:.0%}"
                  f" | PnL ${s['pnl']:+.2f} | Duree moy {s['avg_dur']:.0f} barres")

    # Par motif de sortie
    if m["by_exit"]:
        print(f"\n  PAR MOTIF DE SORTIE")
        print("  " + "-" * 76)
        for reason, s in m["by_exit"].items():
            print(f"  {reason:12s} : {s['n']:4d} trades | WR {s['wr']:.0%} | PnL ${s['pnl']:+.2f}")

    print(f"\n{sep}\n")


# ── Comparaison ────────────────────────────────────────────────────────────────


def _print_comparison(results: list[tuple[DipBuyConfig, dict]], initial_balance: float) -> None:
    """Affiche le tableau comparatif."""
    sep = "=" * 120
    print(f"\n{sep}")
    print(f"  COMPARAISON RSI DIP-BUY — {len(results)} variantes")
    print(sep)

    header = (f"  {'Variante':22s} | {'RSI<':>4s} | {'TP':>5s} | {'Trail':>5s}"
              f" | {'Trades':>6s} | {'WR':>6s} | {'PF':>5s}"
              f" | {'PnL':>10s} | {'DD':>6s} | {'Sharpe':>7s} | {'Dur moy':>8s}")
    print(header)
    print("  " + "-" * 108)

    for cfg, m in results:
        trail_str = f"+{cfg.trailing_activation_pct:.0%}" if cfg.trailing_enabled else "OFF"
        print(f"  {cfg.name:22s} | {cfg.rsi_entry:4.0f} | +{cfg.tp_pct:.0%}"
              f" | {trail_str:>5s}"
              f" | {m['n_trades']:>6d} | {m['win_rate']:5.1%} | {m['profit_factor']:5.2f}"
              f" | ${m['final_balance'] - initial_balance:>+9.2f}"
              f" | {m['max_drawdown']:>5.1%} | {m['sharpe']:>7.2f}"
              f" | {m['avg_duration']:>6.0f} b")

    print(sep)

    # Comparaisons cles
    by_name = {cfg.name: (cfg, m) for cfg, m in results}
    print(f"\n  COMPARAISONS CLES :")
    print("  " + "-" * 76)

    # G1 — Sweep RSI sur Big5 (TP +15%)
    rsi_g1 = ["B5_RSI25_TP15", "B5_RSI30_TP15", "B5_RSI35_TP15", "B5_RSI40_TP15", "B5_RSI45_TP15"]
    found = [(n, by_name[n]) for n in rsi_g1 if n in by_name]
    if found:
        print(f"\n  G1 — RSI sweep (Big5, TP +15%) :")
        for name, (cfg, m) in found:
            pnl = m['final_balance'] - initial_balance
            print(f"    RSI < {cfg.rsi_entry:2.0f} : PF {m['profit_factor']:.2f} | {m['n_trades']:3d} trades"
                  f" | WR {m['win_rate']:.1%} | PnL ${pnl:+.2f} | Dur {m['avg_duration']:.0f}b")

    # G2 — Sweep TP sur Big5 (RSI<35)
    tp_g2 = ["B5_RSI35_TP5", "B5_RSI35_TP10", "B5_RSI35_TP15", "B5_RSI35_TP20"]
    found = [(n, by_name[n]) for n in tp_g2 if n in by_name]
    if found:
        print(f"\n  G2 — TP sweep (Big5, RSI < 35) :")
        for name, (cfg, m) in found:
            pnl = m['final_balance'] - initial_balance
            print(f"    TP +{cfg.tp_pct:.0%} : PF {m['profit_factor']:.2f} | {m['n_trades']:3d} trades"
                  f" | WR {m['win_rate']:.1%} | PnL ${pnl:+.2f} | Dur {m['avg_duration']:.0f}b")

    # G3 — Trailing vs sans trailing
    trail_g3 = [
        ("B5_RSI35_TP15", "B5_RSI35_TP15_TRAIL"),
        ("B5_RSI30_TP15", "B5_RSI30_TP15_TRAIL"),
        ("B5_RSI35_TP10", "B5_RSI35_TP10_TRAIL"),
    ]
    print(f"\n  G3 — Sans trailing vs avec trailing (+5%/+2%) :")
    for no_t, with_t in trail_g3:
        if no_t in by_name and with_t in by_name:
            _, m_no = by_name[no_t]
            _, m_tr = by_name[with_t]
            pnl_no = m_no['final_balance'] - initial_balance
            pnl_tr = m_tr['final_balance'] - initial_balance
            cfg_no = by_name[no_t][0]
            print(f"    RSI<{cfg_no.rsi_entry:.0f} TP+{cfg_no.tp_pct:.0%}  sans: PF {m_no['profit_factor']:.2f}"
                  f" PnL ${pnl_no:+.2f}  |  avec: PF {m_tr['profit_factor']:.2f} PnL ${pnl_tr:+.2f}")

    # G4 — Altcoins vs Big caps
    alts_g4 = [("B5_RSI35_TP15", "A5_RSI35_TP15"), ("B5_RSI30_TP15", "A5_RSI30_TP15")]
    print(f"\n  G4 — Big5 vs Alts5 :")
    for big, alt in alts_g4:
        if big in by_name and alt in by_name:
            _, m_b = by_name[big]
            _, m_a = by_name[alt]
            pnl_b = m_b['final_balance'] - initial_balance
            pnl_a = m_a['final_balance'] - initial_balance
            cfg_b = by_name[big][0]
            print(f"    RSI<{cfg_b.rsi_entry:.0f} TP+{cfg_b.tp_pct:.0%}")
            print(f"      Big5 : PF {m_b['profit_factor']:.2f} | {m_b['n_trades']} trades | PnL ${pnl_b:+.2f} | DD {m_b['max_drawdown']:.1%}")
            print(f"      Alts5: PF {m_a['profit_factor']:.2f} | {m_a['n_trades']} trades | PnL ${pnl_a:+.2f} | DD {m_a['max_drawdown']:.1%}")

    # G5 — All10
    all10_g5 = ["ALL10_RSI30_TP15", "ALL10_RSI35_TP15", "ALL10_RSI35_TP10",
                "ALL10_RSI35_T_TRAIL", "ALL10_RSI40_TP15"]
    found = [(n, by_name[n]) for n in all10_g5 if n in by_name]
    if found:
        print(f"\n  G5 — All 10 paires :")
        for name, (cfg, m) in found:
            pnl = m['final_balance'] - initial_balance
            trail_s = "TRAIL" if cfg.trailing_enabled else "    "
            print(f"    {name:22s} RSI<{cfg.rsi_entry:.0f} TP+{cfg.tp_pct:.0%} {trail_s}"
                  f" : PF {m['profit_factor']:.2f} | {m['n_trades']:3d} trades"
                  f" | WR {m['win_rate']:.1%} | PnL ${pnl:+.2f} | DD {m['max_drawdown']:.1%}")

    # Best variant
    best_cfg, best_m = max(results, key=lambda x: x[1]["profit_factor"])
    pnl_best = best_m['final_balance'] - initial_balance
    print(f"\n  ★ MEILLEURE VARIANTE : {best_cfg.name}"
          f" | PF {best_m['profit_factor']:.2f} | WR {best_m['win_rate']:.1%}"
          f" | PnL ${pnl_best:+.2f} | DD {best_m['max_drawdown']:.1%}")

    print(f"\n{sep}\n")


# ── Charts ─────────────────────────────────────────────────────────────────────


def _generate_chart(cfg: DipBuyConfig, equity_curve: list[EquityPoint],
                    trades: list[DipTrade], initial_balance: float) -> Path:
    """Genere le graphique equity curve."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUTPUT_DIR / f"dipbuy_{cfg.name}.png"

    if not equity_curve:
        return path

    dates = [datetime.fromtimestamp(pt.ts / 1000, tz=timezone.utc) for pt in equity_curve]
    equities = [pt.equity for pt in equity_curve]

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(dates, equities, linewidth=1.0, color="#2196F3")
    ax.axhline(initial_balance, color="gray", linestyle="--", alpha=0.5)
    ax.set_title(f"RSI Dip-Buy — {cfg.name}")
    ax.set_ylabel("Equity ($)")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=100)
    plt.close(fig)
    logger.warning("Chart saved: %s", path)
    return path


# ── Main ───────────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(description="Backtest RSI Dip-Buy")
    parser.add_argument("--balance", type=float, default=1000.0)
    parser.add_argument("--years", type=float, default=2.0)
    parser.add_argument("--interval", type=str, default="4h",
                        help="Intervalle des bougies (1h, 4h, 1d)")
    parser.add_argument("--compare", action="store_true",
                        help="Comparer toutes les variantes")
    args = parser.parse_args()

    initial_balance = args.balance
    years = args.years
    interval = args.interval

    end = datetime.now(timezone.utc)
    start = end - timedelta(days=int(years * 365.25))

    # Telecharger les donnees (toutes les paires de l'univers global)
    print(f"\n📥 Telechargement {len(PAIRS)} paires ({interval}, {years:.1f} ans)...")
    all_candles: dict[str, list[Candle]] = {}
    for pair in PAIRS:
        candles = download_candles(pair, start, end, interval=interval)
        if candles:
            all_candles[pair] = candles
            print(f"  ✓ {pair}: {len(candles)} bougies")
        else:
            print(f"  ✗ {pair}: aucune donnee")

    if not all_candles:
        print("❌ Aucune donnee disponible.")
        return

    if args.compare:
        variants = get_variants()
        print(f"\n🔬 Comparaison de {len(variants)} variantes...")
        results: list[tuple[DipBuyConfig, dict]] = []

        for cfg in variants:
            all_trades: list[DipTrade] = []
            merged_equity: list[EquityPoint] = []
            balance = initial_balance

            # Simuler sur les paires autorisees par la config
            active_pairs = [p for p in sorted(all_candles.keys())
                            if not cfg.pairs_filter or p in cfg.pairs_filter]
            for pair in active_pairs:
                candles = all_candles[pair]
                trades, eq, balance = run_pair(candles, cfg, balance, pair)
                all_trades.extend(trades)
                merged_equity.extend(eq)

            # Trier equity par timestamp
            merged_equity.sort(key=lambda p: p.ts)

            metrics = _compute_metrics(all_trades, merged_equity, initial_balance,
                                       balance, years)
            results.append((cfg, metrics))
            pnl = balance - initial_balance
            trail_s = "TRAIL" if cfg.trailing_enabled else "     "
            print(f"  {cfg.name:22s} {trail_s} : {len(all_trades):4d} trades"
                  f" | PF {metrics['profit_factor']:.2f} | PnL ${pnl:+.2f}")

        _print_comparison(results, initial_balance)

        # Rapport detaille + chart pour la meilleure variante
        best_cfg, best_m = max(results, key=lambda x: x[1]["profit_factor"])
        _print_report(best_cfg, best_m, initial_balance, years)

        # Charts pour les variantes principales uniquement
        for cfg in variants:
            balance_run = initial_balance
            all_eq: list[EquityPoint] = []
            all_tr: list[DipTrade] = []
            active_pairs = [p for p in sorted(all_candles.keys())
                            if not cfg.pairs_filter or p in cfg.pairs_filter]
            for pair in active_pairs:
                tr, eq, balance_run = run_pair(all_candles[pair], cfg, balance_run, pair)
                all_tr.extend(tr)
                all_eq.extend(eq)
            all_eq.sort(key=lambda p: p.ts)
            _generate_chart(cfg, all_eq, all_tr, initial_balance)

    else:
        # Run unique avec config par defaut
        cfg = DipBuyConfig()
        all_trades: list[DipTrade] = []
        merged_equity: list[EquityPoint] = []
        balance = initial_balance

        for pair in sorted(all_candles.keys()):
            candles = all_candles[pair]
            trades, eq, balance = run_pair(candles, cfg, balance, pair)
            all_trades.extend(trades)
            merged_equity.extend(eq)

        merged_equity.sort(key=lambda p: p.ts)
        metrics = _compute_metrics(all_trades, merged_equity, initial_balance,
                                   balance, years)
        _print_report(cfg, metrics, initial_balance, years)
        _generate_chart(cfg, merged_equity, all_trades, initial_balance)


if __name__ == "__main__":
    main()
