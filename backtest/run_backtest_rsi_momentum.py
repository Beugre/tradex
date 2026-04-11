#!/usr/bin/env python3
"""
Backtest — Event-Driven RSI Momentum (Spot, 1H)

Stratégie en 2 phases :

  Phase 1 — Pré-signal (accumulation détectée) :
    - RSI < 40 sur les N dernières bougies
    - RSI en train de remonter (RSI[i] > RSI[i-1] > RSI[i-2])
    - Prix tient un support (pas de nouveau lower low sur support_bars bougies)
    - Volume croissant (volume > MA volume sur vol_ma_period bougies)
    → Entrée "anticipation" avec 40% de l'allocation

  Phase 2 — Déclencheur (breakout confirmé) :
    - Pre-signal actif
    - Cassure résistance locale (close > max(high) des breakout_bars dernières bougies)
    - RSI > 50
    - Bougie impulsive (close > open ET range > ATR moyen)
    → Entrée "breakout" avec les 60% restants

  Exit :
    - TP1 : +0.8%  (clôture 50% de la position)
    - TP2 : +1.5%  (clôture 50% restants)
    - SL  : -0.7%  (full close)
    - Trailing optionnel après TP1

Paires : BTC-USD, ETH-USD, SOL-USD, BNB-USD, XRP-USD (Big5)
Timeframe : 1H
Capital : $1,000 | Frais : 0.10% maker + 0.10% taker

Usage :
    python3 -m backtest.run_backtest_rsi_momentum
    python3 -m backtest.run_backtest_rsi_momentum --compare
    python3 -m backtest.run_backtest_rsi_momentum --balance 1000 --years 3
    python3 -m backtest.run_backtest_rsi_momentum --start 2023-01-01 --end 2024-01-01
"""

from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path

from backtest.data_loader import download_candles
from src.core.models import Candle

logging.basicConfig(level=logging.WARNING, format="[%(asctime)s] %(message)s")
logger = logging.getLogger(__name__)

PAIRS_BIG5 = ["BTC-USD", "ETH-USD", "SOL-USD", "BNB-USD", "XRP-USD"]
H1_PER_DAY = 24


# ── Config ─────────────────────────────────────────────────────────────────────

@dataclass
class RsiMomConfig:
    name: str

    # --- Pré-signal RSI ---
    rsi_period: int = 14
    rsi_entry_threshold: float = 40.0    # RSI doit être passé sous ce seuil
    rsi_rising_bars: int = 2             # nb de bars consécutifs en hausse
    rsi_breakout_min: float = 50.0       # RSI au moment du breakout

    # --- Support ---
    support_bars: int = 10               # fenêtre pour "pas de lower low"

    # --- Volume ---
    vol_ma_period: int = 10              # MA volume pour filtrer "volume croissant"

    # --- Breakout ---
    breakout_bars: int = 10              # cassure du high des N dernières bougies
    impulse_atr_mult: float = 0.8        # range bougie > mult * ATR moyen
    atr_period: int = 14

    # --- Capital split ---
    pre_signal_pct: float = 0.40         # 40% à l'entrée anticipation
    breakout_pct: float = 0.60           # 60% au breakout

    # --- TP / SL ---
    tp1_pct: float = 0.008               # +0.8%
    tp2_pct: float = 0.015               # +1.5%
    sl_pct: float = 0.007                # -0.7%
    tp1_close_ratio: float = 0.50        # clôture 50% au TP1

    # --- Trailing après TP1 ---
    trailing_enabled: bool = False
    trailing_distance_pct: float = 0.004 # trail 0.4% sous le pic

    # --- Mode entrée ---
    split_entry: bool = True             # True = 40/60, False = full breakout only

    # --- Cooldown ---
    cooldown_bars: int = 3               # bougies H1 de cooldown après clôture

    # --- Risk ---
    risk_pct: float = 0.20               # fraction du capital par slot complet

    # --- Frais ---
    entry_fee_pct: float = 0.001
    exit_fee_pct: float = 0.001


# ── Indicateurs ───────────────────────────────────────────────────────────────

def _rsi(closes: list[float], period: int) -> list[float]:
    """RSI Wilder — retourne une liste de même longueur (NaN=0 sur les premiers)."""
    result = [0.0] * len(closes)
    if len(closes) <= period:
        return result
    gains, losses = [], []
    for i in range(1, period + 1):
        d = closes[i] - closes[i - 1]
        gains.append(max(d, 0.0))
        losses.append(max(-d, 0.0))
    avg_gain = sum(gains) / period
    avg_loss = sum(losses) / period
    if avg_loss == 0:
        result[period] = 100.0
    else:
        rs = avg_gain / avg_loss
        result[period] = 100.0 - 100.0 / (1.0 + rs)
    for i in range(period + 1, len(closes)):
        d = closes[i] - closes[i - 1]
        g = max(d, 0.0)
        l = max(-d, 0.0)
        avg_gain = (avg_gain * (period - 1) + g) / period
        avg_loss = (avg_loss * (period - 1) + l) / period
        if avg_loss == 0:
            result[i] = 100.0
        else:
            rs = avg_gain / avg_loss
            result[i] = 100.0 - 100.0 / (1.0 + rs)
    return result


def _atr(candles: list[Candle], period: int) -> list[float]:
    result = [0.0] * len(candles)
    if len(candles) < 2:
        return result
    trs = [0.0]
    for i in range(1, len(candles)):
        h, l, pc = candles[i].high, candles[i].low, candles[i - 1].close
        trs.append(max(h - l, abs(h - pc), abs(l - pc)))
    if len(trs) < period:
        return result
    atr_val = sum(trs[1: period + 1]) / period
    result[period] = atr_val
    for i in range(period + 1, len(candles)):
        atr_val = (atr_val * (period - 1) + trs[i]) / period
        result[i] = atr_val
    return result


def _vol_ma(volumes: list[float], period: int) -> list[float]:
    result = [0.0] * len(volumes)
    for i in range(period - 1, len(volumes)):
        result[i] = sum(volumes[i - period + 1: i + 1]) / period
    return result


# ── Dataclass résultat ─────────────────────────────────────────────────────────

@dataclass
class TradeResult:
    entry_price: float
    exit_price: float
    pnl_pct: float   # net (frais inclus), sur la portion fermée
    pnl_abs: float   # en dollars
    is_win: bool


# ── Simulation d'une paire ─────────────────────────────────────────────────────

def run_pair(
    candles: list[Candle],
    cfg: RsiMomConfig,
    initial_balance: float,
) -> tuple[float, list[TradeResult], list[float]]:
    """
    Retourne (final_balance, trades, equity_curve).
    equity_curve[i] = valeur du compte après la bougie i.
    """
    if len(candles) < max(cfg.rsi_period, cfg.atr_period, cfg.breakout_bars, cfg.support_bars) + 5:
        return initial_balance, [], [initial_balance] * len(candles)

    closes  = [c.close for c in candles]
    highs   = [c.high  for c in candles]
    lows    = [c.low   for c in candles]
    opens   = [c.open  for c in candles]
    volumes = [c.volume for c in candles]

    rsi_vals = _rsi(closes, cfg.rsi_period)
    atr_vals = _atr(candles, cfg.atr_period)
    vol_ma   = _vol_ma(volumes, cfg.vol_ma_period)

    balance = initial_balance
    equity  = [initial_balance] * len(candles)
    trades: list[TradeResult] = []

    # État de position
    in_pre_signal   = False   # phase 1 active
    pre_entry_price = 0.0
    pre_entry_size  = 0.0     # taille en unités (phase 1)
    pre_entry_cost  = 0.0     # coût en $ (phase 1)

    in_breakout     = False   # phase 2 active (position complète)
    entry_price_full = 0.0    # prix moyen pondéré total
    full_size        = 0.0    # taille totale en unités
    full_cost        = 0.0    # coût total en $
    peak_price       = 0.0
    tp1_done         = False  # TP1 déjà touché

    cooldown = 0

    warmup = max(cfg.rsi_period, cfg.atr_period, cfg.breakout_bars, cfg.support_bars, cfg.vol_ma_period) + 2

    for i in range(warmup, len(candles)):
        c = candles[i]
        rsi = rsi_vals[i]
        atr = atr_vals[i]
        vol = volumes[i]
        vma = vol_ma[i]

        if cooldown > 0:
            cooldown -= 1
            equity[i] = balance + (full_size * c.close - full_cost if in_breakout else
                                   pre_entry_size * c.close - pre_entry_cost if in_pre_signal else 0.0)
            continue

        # ── Gestion de la position ouverte ────────────────────────────────
        if in_breakout:
            current_pnl_pct = (c.close - entry_price_full) / entry_price_full

            # Trailing après TP1
            if cfg.trailing_enabled and tp1_done:
                peak_price = max(peak_price, c.high)
                trail_stop = peak_price * (1 - cfg.trailing_distance_pct)
                if c.close <= trail_stop:
                    # Clôture trailing
                    exit_p = trail_stop
                    fee = full_size * exit_p * cfg.exit_fee_pct
                    pnl = full_size * exit_p - full_cost - fee
                    balance += full_cost + pnl
                    trades.append(TradeResult(entry_price_full, exit_p,
                                              pnl / full_cost, pnl, pnl > 0))
                    in_breakout = False; full_size = 0.0; full_cost = 0.0; tp1_done = False
                    cooldown = cfg.cooldown_bars
                    equity[i] = balance
                    continue

            # TP2
            if current_pnl_pct >= cfg.tp2_pct:
                exit_p = entry_price_full * (1 + cfg.tp2_pct)
                fee = full_size * exit_p * cfg.exit_fee_pct
                pnl = full_size * exit_p - full_cost - fee
                balance += full_cost + pnl
                trades.append(TradeResult(entry_price_full, exit_p,
                                          pnl / full_cost, pnl, pnl > 0))
                in_breakout = False; full_size = 0.0; full_cost = 0.0; tp1_done = False
                cooldown = cfg.cooldown_bars
                equity[i] = balance
                continue

            # TP1 (partiel)
            if not tp1_done and current_pnl_pct >= cfg.tp1_pct:
                close_size = full_size * cfg.tp1_close_ratio
                exit_p = entry_price_full * (1 + cfg.tp1_pct)
                fee = close_size * exit_p * cfg.exit_fee_pct
                pnl = close_size * exit_p - (close_size / full_size) * full_cost - fee
                balance += (close_size / full_size) * full_cost + pnl
                full_cost  *= (1 - cfg.tp1_close_ratio)
                full_size  -= close_size
                tp1_done    = True
                peak_price  = exit_p
                trades.append(TradeResult(entry_price_full, exit_p,
                                          cfg.tp1_pct, pnl, True))
                equity[i] = balance + full_size * c.close - full_cost
                continue

            # SL
            if current_pnl_pct <= -cfg.sl_pct:
                exit_p = entry_price_full * (1 - cfg.sl_pct)
                fee = full_size * exit_p * cfg.exit_fee_pct
                pnl = full_size * exit_p - full_cost - fee
                balance += full_cost + pnl
                trades.append(TradeResult(entry_price_full, exit_p,
                                          pnl / full_cost, pnl, False))
                in_breakout = False; full_size = 0.0; full_cost = 0.0; tp1_done = False
                in_pre_signal = False; pre_entry_size = 0.0; pre_entry_cost = 0.0
                cooldown = cfg.cooldown_bars
                equity[i] = balance
                continue

            peak_price = max(peak_price, c.high)
            equity[i] = balance + full_size * c.close - full_cost
            continue

        # ── Phase 1 : clôture si pré-signal détourne ─────────────────────
        if in_pre_signal:
            if pre_entry_price == 0.0:
                in_pre_signal = False
                equity[i] = balance
                continue
            current_pnl_pct = (c.close - pre_entry_price) / pre_entry_price
            # SL sur la position pré-signal
            if current_pnl_pct <= -cfg.sl_pct:
                exit_p = pre_entry_price * (1 - cfg.sl_pct)
                fee = pre_entry_size * exit_p * cfg.exit_fee_pct
                pnl = pre_entry_size * exit_p - pre_entry_cost - fee
                balance += pre_entry_cost + pnl
                trades.append(TradeResult(pre_entry_price, exit_p,
                                          pnl / pre_entry_cost, pnl, False))
                in_pre_signal = False; pre_entry_size = 0.0; pre_entry_cost = 0.0
                cooldown = cfg.cooldown_bars
                equity[i] = balance
                continue

        # ── Détection pré-signal ──────────────────────────────────────────
        if not in_pre_signal and not in_breakout:
            rsi_was_low = any(rsi_vals[j] < cfg.rsi_entry_threshold
                              for j in range(i - cfg.rsi_period, i))
            rsi_rising  = all(rsi_vals[i - k] > rsi_vals[i - k - 1]
                              for k in range(cfg.rsi_rising_bars))
            support_holds = all(lows[j] >= min(lows[i - cfg.support_bars: i])
                                for j in range(i - cfg.support_bars, i))
            vol_growing = vol > vma if vma > 0 else False

            if rsi_was_low and rsi_rising and support_holds and vol_growing:
                if cfg.split_entry:
                    # Entrée phase 1 (40%)
                    alloc = balance * cfg.risk_pct * cfg.pre_signal_pct
                    fee = alloc * cfg.entry_fee_pct
                    size = (alloc - fee) / c.close
                    pre_entry_price = c.close
                    pre_entry_size  = size
                    pre_entry_cost  = alloc
                    balance        -= alloc
                    in_pre_signal   = True
                # Si split_entry=False, on attend juste le breakout sans entrer
                else:
                    in_pre_signal = True  # flag only, no position yet
                equity[i] = balance + (pre_entry_size * c.close - pre_entry_cost if cfg.split_entry else 0.0)
                continue

        # ── Déclencheur breakout ──────────────────────────────────────────
        if in_pre_signal and not in_breakout:
            # Résistance locale = max des N dernières bougies (hors bougie courante)
            resistance = max(highs[max(0, i - cfg.breakout_bars): i])
            breakout   = c.close > resistance
            rsi_ok     = rsi > cfg.rsi_breakout_min
            impulsive  = (c.close > c.open) and (atr > 0) and ((c.high - c.low) >= cfg.impulse_atr_mult * atr)

            if breakout and rsi_ok and impulsive:
                alloc_brk = balance * cfg.risk_pct * (cfg.breakout_pct if cfg.split_entry else 1.0)
                fee_brk   = alloc_brk * cfg.entry_fee_pct
                size_brk  = (alloc_brk - fee_brk) / c.close
                balance  -= alloc_brk

                if cfg.split_entry and pre_entry_size > 0:
                    # Fusionner phase 1 + phase 2
                    total_cost = pre_entry_cost + alloc_brk
                    total_size = pre_entry_size + size_brk
                    entry_price_full = total_cost / total_size
                    full_size  = total_size
                    full_cost  = total_cost
                else:
                    full_size  = size_brk
                    full_cost  = alloc_brk
                    entry_price_full = c.close

                peak_price    = c.close
                tp1_done      = False
                in_breakout   = True
                in_pre_signal = False
                pre_entry_size = 0.0; pre_entry_cost = 0.0

        equity[i] = balance + (full_size * c.close - full_cost if in_breakout else 0.0)

    # Clôture forcée fin de backtest
    if in_breakout and full_size > 0:
        exit_p = candles[-1].close
        fee    = full_size * exit_p * cfg.exit_fee_pct
        pnl    = full_size * exit_p - full_cost - fee
        balance += full_cost + pnl
        trades.append(TradeResult(entry_price_full, exit_p, pnl / full_cost, pnl, pnl > 0))
    elif in_pre_signal and pre_entry_size > 0:
        exit_p = candles[-1].close
        fee    = pre_entry_size * exit_p * cfg.exit_fee_pct
        pnl    = pre_entry_size * exit_p - pre_entry_cost - fee
        balance += pre_entry_cost + pnl
        trades.append(TradeResult(pre_entry_price, exit_p, pnl / pre_entry_cost, pnl, pnl > 0))

    return balance, trades, equity


# ── Métriques ──────────────────────────────────────────────────────────────────

def compute_metrics(trades: list[TradeResult], equity: list[float], initial_balance: float) -> dict:
    n = len(trades)
    if n == 0:
        return {
            "n_trades": 0, "win_rate": 0.0, "profit_factor": 0.0,
            "final_balance": initial_balance, "max_drawdown": 0.0,
        }
    wins   = [t for t in trades if t.is_win]
    losses = [t for t in trades if not t.is_win]
    gross_profit = sum(t.pnl_abs for t in wins)
    gross_loss   = abs(sum(t.pnl_abs for t in losses))
    pf = gross_profit / gross_loss if gross_loss > 0 else float("inf")

    peak, dd = initial_balance, 0.0
    for v in equity:
        if v > peak:
            peak = v
        if peak > 0:
            dd = min(dd, (v - peak) / peak)

    return {
        "n_trades":      n,
        "win_rate":      len(wins) / n,
        "profit_factor": pf,
        "final_balance": equity[-1] if equity else initial_balance,
        "max_drawdown":  dd,
    }


# ── Run toutes les paires ──────────────────────────────────────────────────────

def _run_variants(
    cfgs: list[RsiMomConfig],
    pairs: list[str],
    start: datetime,
    end: datetime,
    initial_balance: float,
) -> list[tuple[RsiMomConfig, dict]]:
    print(f"\n📥 Téléchargement {len(pairs)} paires (1H, {start:%Y-%m-%d} → {end:%Y-%m-%d})...")
    candles_by_pair: dict[str, list[Candle]] = {}
    for pair in pairs:
        c = download_candles(pair, start, end, interval="1h")
        candles_by_pair[pair] = c
        print(f"  ✓ {pair}: {len(c)} bougies")

    results = []
    for cfg in cfgs:
        all_trades: list[TradeResult] = []
        combined_equity: list[float]  = []
        balance = initial_balance

        for pair in pairs:
            candles = candles_by_pair[pair]
            if not candles:
                continue
            bal, trades, equity = run_pair(candles, cfg, balance / len(pairs))
            all_trades.extend(trades)
            if not combined_equity:
                combined_equity = equity
            else:
                combined_equity = [a + b for a, b in zip(combined_equity, equity)]

        # Recalibrer l'equity sur le capital initial
        if combined_equity:
            scale = initial_balance / sum(
                initial_balance / len(pairs) for _ in pairs
            )
            combined_equity = [v * scale / len(pairs) for v in combined_equity]

        m = compute_metrics(all_trades, combined_equity, initial_balance)
        # Recalcul balance finale agrégée
        m["final_balance"] = sum(
            run_pair(candles_by_pair[p], cfg, initial_balance / len(pairs))[0]
            for p in pairs if candles_by_pair[p]
        )
        results.append((cfg, m))

    return results


# ── Affichage ─────────────────────────────────────────────────────────────────

def _print_results(cfg: RsiMomConfig, m: dict, initial_balance: float) -> None:
    pnl   = m["final_balance"] - initial_balance
    sep   = "=" * 80
    print(f"\n{sep}")
    print(f"  RSI MOMENTUM 1H — {cfg.name}")
    print(f"  Capital: ${initial_balance:,.0f} | RSI seuil: <{cfg.rsi_entry_threshold}"
          f" | Breakout: {cfg.breakout_bars}b | SL: -{cfg.sl_pct:.1%} | TP1: +{cfg.tp1_pct:.1%} | TP2: +{cfg.tp2_pct:.1%}")
    print(f"  Risk/trade: {cfg.risk_pct:.0%} | Split: {cfg.split_entry} | Trail: {cfg.trailing_enabled}")
    print(f"{sep}")
    print(f"\n  RÉSULTATS GLOBAUX")
    print(f"  {'-'*60}")
    print(f"  Capital final      : ${m['final_balance']:,.2f} ({pnl/initial_balance:+.1%})")
    print(f"  Max Drawdown       : {m['max_drawdown']:.1%}")
    print(f"  Win Rate           : {m['win_rate']:.1%} ({int(m['win_rate']*m['n_trades'])}/{m['n_trades']})")
    print(f"  Profit Factor      : {m['profit_factor']:.2f}")
    print(f"  Trades totaux      : {m['n_trades']}")
    print(f"\n{sep}\n")


def _print_comparison(results: list[tuple[RsiMomConfig, dict]], initial_balance: float) -> None:
    sep = "=" * 80
    print(f"\n{sep}")
    print(f"  COMPARAISON RSI MOMENTUM 1H — {len(results)} variantes")
    print(f"  Capital: ${initial_balance:,.0f} | Big5 (BTC ETH SOL BNB XRP)")
    print(f"{sep}\n")

    # Tableau brut
    header = f"  {'NOM':30s} | {'RSI<':>4s} | {'BRK':>4s} | {'RISK':>5s} | {'SPLIT':>5s} | {'T':>5s} | {'WR':>6s} | {'PF':>7s} | {'PnL':>12s} | {'DD':>6s}"
    print(header)
    print("  " + "-" * (len(header) - 2))
    for cfg, m in sorted(results, key=lambda x: x[1]["profit_factor"], reverse=True):
        pnl = m["final_balance"] - initial_balance
        split_str = "Y" if cfg.split_entry else "N"
        print(f"  {cfg.name:30s} | {cfg.rsi_entry_threshold:>4.0f} | {cfg.breakout_bars:>4d}"
              f" | {cfg.risk_pct:>4.0%} | {split_str:>5s} | {m['n_trades']:>5d}"
              f" | {m['win_rate']:>6.1%} | {m['profit_factor']:>7.2f}"
              f" | {pnl:>+12.2f} | {m['max_drawdown']:>6.1%}")

    by_name = {cfg.name: (cfg, m) for cfg, m in results}

    # Groupes
    groups = [
        ("G1 — Référence (full breakout, no split)",
         ["BASE_NOSPLIT", "BASE_NOSPLIT_R10", "BASE_NOSPLIT_TRAIL"]),
        ("G2 — Split 40/60 (pré-signal + breakout)",
         ["BASE_SPLIT", "BASE_SPLIT_R10", "BASE_SPLIT_TRAIL"]),
        ("G3 — RSI seuil élargi (<45)",
         ["RSI45_NOSPLIT", "RSI45_SPLIT"]),
        ("G4 — Breakout fenêtre courte (5b)",
         ["BRK5_NOSPLIT", "BRK5_SPLIT"]),
        ("G5 — Risk 30%",
         ["BASE_NOSPLIT_R30", "BASE_SPLIT_R30"]),
    ]
    for title, names in groups:
        found = [(n, by_name[n]) for n in names if n in by_name]
        if not found:
            continue
        print(f"\n  {title} :\n")
        for name, (cfg, m) in found:
            pnl = m["final_balance"] - initial_balance
            print(f"    {name:30s} | PF {m['profit_factor']:.2f}"
                  f" | WR {m['win_rate']:.1%} | PnL ${pnl:+.2f}"
                  f" | DD {m['max_drawdown']:.1%} | {m['n_trades']}t")

    best_pf_cfg,  best_pf_m  = max(results, key=lambda x: x[1]["profit_factor"])
    best_pnl_cfg, best_pnl_m = max(results, key=lambda x: x[1]["final_balance"])
    print(f"\n  ★ MEILLEUR PF  : {best_pf_cfg.name} | PF {best_pf_m['profit_factor']:.2f}"
          f" | PnL ${best_pf_m['final_balance']-initial_balance:+.2f}"
          f" | DD {best_pf_m['max_drawdown']:.1%}")
    print(f"  ★ MEILLEUR PnL : {best_pnl_cfg.name} | PF {best_pnl_m['profit_factor']:.2f}"
          f" | PnL ${best_pnl_m['final_balance']-initial_balance:+.2f}"
          f" | DD {best_pnl_m['max_drawdown']:.1%}")
    print(f"\n{sep}\n")


# ── Variantes ─────────────────────────────────────────────────────────────────

def get_variants() -> list[RsiMomConfig]:
    # ══ G1 — Référence full breakout (pas de split) ══
    g1 = [
        RsiMomConfig(name="BASE_NOSPLIT",       split_entry=False, risk_pct=0.20),
        RsiMomConfig(name="BASE_NOSPLIT_R10",   split_entry=False, risk_pct=0.10),
        RsiMomConfig(name="BASE_NOSPLIT_TRAIL", split_entry=False, risk_pct=0.20,
                     trailing_enabled=True),
    ]
    # ══ G2 — Split 40/60 ══
    g2 = [
        RsiMomConfig(name="BASE_SPLIT",       split_entry=True, risk_pct=0.20),
        RsiMomConfig(name="BASE_SPLIT_R10",   split_entry=True, risk_pct=0.10),
        RsiMomConfig(name="BASE_SPLIT_TRAIL", split_entry=True, risk_pct=0.20,
                     trailing_enabled=True),
    ]
    # ══ G3 — RSI seuil élargi ══
    g3 = [
        RsiMomConfig(name="RSI45_NOSPLIT", split_entry=False, risk_pct=0.20,
                     rsi_entry_threshold=45.0),
        RsiMomConfig(name="RSI45_SPLIT",   split_entry=True,  risk_pct=0.20,
                     rsi_entry_threshold=45.0),
    ]
    # ══ G4 — Breakout fenêtre courte (5 bougies) ══
    g4 = [
        RsiMomConfig(name="BRK5_NOSPLIT", split_entry=False, risk_pct=0.20,
                     breakout_bars=5),
        RsiMomConfig(name="BRK5_SPLIT",   split_entry=True,  risk_pct=0.20,
                     breakout_bars=5),
    ]
    # ══ G5 — Risk agressif 30% ══
    g5 = [
        RsiMomConfig(name="BASE_NOSPLIT_R30", split_entry=False, risk_pct=0.30),
        RsiMomConfig(name="BASE_SPLIT_R30",   split_entry=True,  risk_pct=0.30),
    ]
    return g1 + g2 + g3 + g4 + g5


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Backtest RSI Momentum 1H")
    parser.add_argument("--balance", type=float, default=1000.0)
    parser.add_argument("--years",   type=int,   default=3)
    parser.add_argument("--compare", action="store_true")
    parser.add_argument("--start",   type=str,   default=None)
    parser.add_argument("--end",     type=str,   default=None)
    args = parser.parse_args()

    now = datetime.now(timezone.utc)
    if args.end:
        end = datetime.strptime(args.end, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    else:
        end = now
    if args.start:
        start = datetime.strptime(args.start, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    else:
        start = end - timedelta(days=365 * args.years)

    if args.compare:
        cfgs    = get_variants()
        results = _run_variants(cfgs, PAIRS_BIG5, start, end, args.balance)
        _print_comparison(results, args.balance)
    else:
        cfg     = RsiMomConfig(name="BASE_NOSPLIT", split_entry=False, risk_pct=0.20)
        results = _run_variants([cfg], PAIRS_BIG5, start, end, args.balance)
        _print_results(cfg, results[0][1], args.balance)


if __name__ == "__main__":
    main()
