#!/usr/bin/env python3
"""
Backtest CrashBot — Analyse d'alternance W/L et allocation adaptative.

Hypothèse : les gains et pertes du CrashBot alternent (W→L→W→L).
Test : comparer 3 modes d'allocation :
  A) BASELINE  — risque fixe (5%)
  B) ADAPTIVE  — après perte → risque × boost_mult, après gain → risque × shrink_mult
  C) INVERSE   — contrôle : après gain → boost, après perte → shrink

Usage:
    python -m backtest.run_backtest_crashbot_adaptive
    python -m backtest.run_backtest_crashbot_adaptive --years 6 --balance 500
    python -m backtest.run_backtest_crashbot_adaptive --pairs BTC-USD,ETH-USD
"""

from __future__ import annotations

import argparse
import logging
import sys
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from backtest.data_loader import download_candles
from src.core.crashbot_detector import (
    CrashConfig,
    CrashSignal,
    atr as compute_atr,
    detect_crash_signals,
    compute_step_trailing,
)
from src.core.models import Candle, StrategyType

logging.basicConfig(level=logging.WARNING, format="[%(asctime)s] %(message)s")
logger = logging.getLogger(__name__)

OUTPUT_DIR = Path(__file__).parent / "output"

# ── Fees Binance ───────────────────────────────────────────────────────────────
MAKER_FEE = 0.001   # 0.1%
TAKER_FEE = 0.001   # 0.1%

# ── Paires par défaut (liquides, volatiles) ────────────────────────────────────
DEFAULT_PAIRS = [
    "BTC-USD", "ETH-USD", "SOL-USD", "XRP-USD", "BNB-USD",
    "DOGE-USD", "ADA-USD", "AVAX-USD", "LINK-USD", "DOT-USD",
    "LTC-USD", "INJ-USD", "AAVE-USD", "NEAR-USD", "UNI-USD",
    "FIL-USD", "ATOM-USD",
]


# ── Structures ─────────────────────────────────────────────────────────────────


@dataclass
class CrashTrade:
    """Un trade CrashBot complet."""
    symbol: str
    entry_bar: int
    entry_price: float
    entry_ts: int
    sl_price: float
    tp_price: float
    exit_bar: int = 0
    exit_price: float = 0.0
    exit_ts: int = 0
    exit_reason: str = ""
    pnl_usd: float = 0.0
    pnl_pct: float = 0.0
    size: float = 0.0
    fees: float = 0.0
    drop_pct: float = 0.0
    trailing_steps: int = 0
    duration_hours: float = 0.0
    risk_pct_used: float = 0.0   # risque effectif utilisé pour ce trade


@dataclass
class EquityPoint:
    ts: int
    equity: float


# ── Simulateur CrashBot ───────────────────────────────────────────────────────


def simulate_crashbot(
    all_candles: dict[str, list[Candle]],
    cfg: CrashConfig,
    initial_balance: float = 500.0,
    base_risk_pct: float = 0.05,
    max_position_pct: float = 0.50,
    max_concurrent: int = 5,
    allocation_mode: str = "baseline",  # "baseline", "adaptive", "inverse"
    boost_mult: float = 1.8,       # après perte (adaptive) ou gain (inverse)
    shrink_mult: float = 0.5,      # après gain (adaptive) ou perte (inverse)
    min_risk_pct: float = 0.02,    # plancher de risque
    max_risk_pct: float = 0.12,    # plafond de risque
) -> tuple[list[CrashTrade], list[EquityPoint]]:
    """
    Simule CrashBot sur toutes les paires en multi-pair.

    allocation_mode:
      - "baseline" : risque fixe = base_risk_pct
      - "adaptive" : après perte → risk × boost_mult, après gain → risk × shrink_mult
      - "inverse"  : après gain → risk × boost_mult, après perte → risk × shrink_mult
    """

    # ── Détection des signaux par paire ────────────────────────────────────
    signals_by_pair: dict[str, list[CrashSignal]] = {}
    for pair, candles in all_candles.items():
        sigs = detect_crash_signals(candles, cfg)
        for s in sigs:
            s.symbol = pair
        signals_by_pair[pair] = sigs

    # ── Merge et tri chronologique ─────────────────────────────────────────
    # On crée un timeline global basé sur la première paire pour l'equity curve
    ref_pair = list(all_candles.keys())[0]
    ref_candles = all_candles[ref_pair]
    n_bars = len(ref_candles)

    # ── État ────────────────────────────────────────────────────────────────
    balance = initial_balance
    trades: list[CrashTrade] = []
    equity_curve: list[EquityPoint] = []

    # Positions ouvertes : {symbol: trade info}
    open_positions: dict[str, dict] = {}
    # Cooldown tracker : {symbol: bar_de_sortie}
    cooldowns: dict[str, int] = {}

    # Allocation adaptative : le prochain risque à utiliser
    current_risk = base_risk_pct
    last_trade_result: Optional[str] = None  # "win" ou "loss"

    # ── Boucle bar par bar ─────────────────────────────────────────────────
    for bar_idx in range(n_bars):
        ts = ref_candles[bar_idx].timestamp

        # 1) Gérer les positions ouvertes
        closed_symbols: list[str] = []
        for sym, pos in list(open_positions.items()):
            candles = all_candles[sym]
            if bar_idx >= len(candles):
                continue
            c = candles[bar_idx]

            # Step trailing
            new_sl, new_tp, new_steps = compute_step_trailing(
                current_price=c.high,  # optimiste : check high
                entry_price=pos["entry_price"],
                current_sl=pos["sl"],
                current_tp=pos["tp"],
                trail_steps=pos["trail_steps"],
                cfg=cfg,
            )
            pos["sl"] = new_sl
            pos["tp"] = new_tp
            pos["trail_steps"] = new_steps

            # Check SL (low touche SL)
            if c.low <= pos["sl"]:
                exit_price = pos["sl"]
                exit_reason = "SL"
                pnl_pct = (exit_price - pos["entry_price"]) / pos["entry_price"]
                fees = pos["size"] * pos["entry_price"] * TAKER_FEE + pos["size"] * exit_price * TAKER_FEE
                pnl_usd = pos["size"] * (exit_price - pos["entry_price"]) - fees
                duration_h = (bar_idx - pos["entry_bar"]) * 4

                trade = CrashTrade(
                    symbol=sym,
                    entry_bar=pos["entry_bar"],
                    entry_price=pos["entry_price"],
                    entry_ts=pos["entry_ts"],
                    sl_price=pos["original_sl"],
                    tp_price=pos["original_tp"],
                    exit_bar=bar_idx,
                    exit_price=exit_price,
                    exit_ts=ts,
                    exit_reason=exit_reason,
                    pnl_usd=pnl_usd,
                    pnl_pct=pnl_pct,
                    size=pos["size"],
                    fees=fees,
                    drop_pct=pos["drop_pct"],
                    trailing_steps=pos["trail_steps"],
                    duration_hours=duration_h,
                    risk_pct_used=pos["risk_pct_used"],
                )
                trades.append(trade)
                balance += pnl_usd
                closed_symbols.append(sym)
                cooldowns[sym] = bar_idx
                continue

            # Check TP (high touche TP) — avec step trailing, TP est dynamique
            # Le step trailing gère l'extension du TP ; on ferme si high >= TP
            # et que le trailing n'étend plus (quand le prix retombe sous le trigger)
            # En pratique, la sortie TP se fait quand le prix redescend vers le SL
            # après un step trail. On ne ferme pas au TP directement car
            # compute_step_trailing déplace le SL/TP.

        for sym in closed_symbols:
            del open_positions[sym]

            # Mise à jour de l'allocation adaptative
            last_trade = trades[-1]
            if allocation_mode == "adaptive":
                if last_trade.pnl_usd <= 0:
                    # Perte → augmenter le risque (hypothèse : le suivant sera gagnant)
                    current_risk = min(current_risk * boost_mult, max_risk_pct)
                    last_trade_result = "loss"
                else:
                    # Gain → réduire le risque
                    current_risk = max(current_risk * shrink_mult, min_risk_pct)
                    last_trade_result = "win"
            elif allocation_mode == "inverse":
                if last_trade.pnl_usd > 0:
                    # Gain → augmenter le risque (stratégie inverse)
                    current_risk = min(current_risk * boost_mult, max_risk_pct)
                    last_trade_result = "win"
                else:
                    # Perte → réduire le risque
                    current_risk = max(current_risk * shrink_mult, min_risk_pct)
                    last_trade_result = "loss"
            # baseline : rien à faire, current_risk reste fixe

        # 2) Chercher de nouveaux signaux
        if len(open_positions) < max_concurrent and balance > 10:
            for pair, sigs in signals_by_pair.items():
                if pair in open_positions:
                    continue
                # Cooldown check
                if pair in cooldowns and (bar_idx - cooldowns[pair]) < cfg.cooldown_bars:
                    continue

                # Chercher un signal pour ce bar
                for sig in sigs:
                    if sig.candle_index != bar_idx:
                        continue

                    # Entry
                    entry_price = sig.entry_price
                    sl_price = sig.sl_price

                    if entry_price <= sl_price or entry_price <= 0:
                        continue

                    sl_distance = entry_price - sl_price
                    if sl_distance <= 0:
                        continue

                    # Sizing avec le risque courant (adaptatif ou fixe)
                    effective_risk = current_risk if allocation_mode != "baseline" else base_risk_pct
                    risk_amount = balance * effective_risk
                    size = risk_amount / sl_distance

                    # Plafond
                    max_cost = balance * max_position_pct
                    if size * entry_price > max_cost:
                        size = max_cost / entry_price

                    if size * entry_price < 5:  # min $5
                        continue

                    open_positions[pair] = {
                        "entry_bar": bar_idx,
                        "entry_price": entry_price,
                        "entry_ts": ts,
                        "sl": sl_price,
                        "tp": sig.tp_price,
                        "original_sl": sl_price,
                        "original_tp": sig.tp_price,
                        "size": size,
                        "trail_steps": 0,
                        "drop_pct": sig.drop_pct,
                        "risk_pct_used": effective_risk,
                    }

                    if len(open_positions) >= max_concurrent:
                        break

        # 3) Equity curve
        unrealized = 0.0
        for sym, pos in open_positions.items():
            candles = all_candles[sym]
            if bar_idx < len(candles):
                current_px = candles[bar_idx].close
                unrealized += pos["size"] * (current_px - pos["entry_price"])

        equity_curve.append(EquityPoint(ts=ts, equity=balance + unrealized))

    # ── Fermer les positions restantes ─────────────────────────────────────
    for sym, pos in open_positions.items():
        candles = all_candles[sym]
        last_bar = min(n_bars - 1, len(candles) - 1)
        exit_price = candles[last_bar].close
        pnl_pct = (exit_price - pos["entry_price"]) / pos["entry_price"]
        fees = pos["size"] * pos["entry_price"] * TAKER_FEE + pos["size"] * exit_price * TAKER_FEE
        pnl_usd = pos["size"] * (exit_price - pos["entry_price"]) - fees

        trade = CrashTrade(
            symbol=sym,
            entry_bar=pos["entry_bar"],
            entry_price=pos["entry_price"],
            entry_ts=pos["entry_ts"],
            sl_price=pos["original_sl"],
            tp_price=pos["original_tp"],
            exit_bar=last_bar,
            exit_price=exit_price,
            exit_ts=candles[last_bar].timestamp,
            exit_reason="END",
            pnl_usd=pnl_usd,
            pnl_pct=pnl_pct,
            size=pos["size"],
            fees=fees,
            drop_pct=pos["drop_pct"],
            trailing_steps=pos["trail_steps"],
            duration_hours=(last_bar - pos["entry_bar"]) * 4,
            risk_pct_used=pos["risk_pct_used"],
        )
        trades.append(trade)
        balance += pnl_usd

    return trades, equity_curve


# ── Analyse d'alternance W/L ──────────────────────────────────────────────────


def analyze_alternation(trades: list[CrashTrade]) -> dict:
    """
    Analyse l'alternance des résultats W/L.
    
    Mesure :
    - Taux de changement (W→L ou L→W) vs continuation (W→W ou L→L)
    - Autocorrélation lag-1 des PnL
    - Streaks (séries consécutives)
    """
    if len(trades) < 2:
        return {"error": "Pas assez de trades pour analyser"}

    outcomes = ["W" if t.pnl_usd > 0 else "L" for t in trades]
    n = len(outcomes)

    # Comptage des transitions
    changes = 0  # W→L ou L→W
    continuations = 0  # W→W ou L→L
    transitions = {"WW": 0, "WL": 0, "LW": 0, "LL": 0}

    for i in range(1, n):
        key = outcomes[i - 1] + outcomes[i]
        transitions[key] += 1
        if outcomes[i] != outcomes[i - 1]:
            changes += 1
        else:
            continuations += 1

    total_transitions = changes + continuations
    alternation_rate = changes / total_transitions if total_transitions else 0

    # Autocorrélation lag-1 des PnL
    pnls = [t.pnl_usd for t in trades]
    mean_pnl = sum(pnls) / n
    var_pnl = sum((p - mean_pnl) ** 2 for p in pnls) / n if n > 1 else 1
    if var_pnl > 0 and n > 1:
        autocorr = sum(
            (pnls[i] - mean_pnl) * (pnls[i - 1] - mean_pnl)
            for i in range(1, n)
        ) / ((n - 1) * var_pnl)
    else:
        autocorr = 0.0

    # Autocorrélation lag-1 binaire (1 = win, 0 = loss)
    binary = [1 if o == "W" else 0 for o in outcomes]
    mean_b = sum(binary) / n
    var_b = sum((b - mean_b) ** 2 for b in binary) / n
    if var_b > 0 and n > 1:
        autocorr_binary = sum(
            (binary[i] - mean_b) * (binary[i - 1] - mean_b)
            for i in range(1, n)
        ) / ((n - 1) * var_b)
    else:
        autocorr_binary = 0.0

    # Streaks
    streaks: list[tuple[str, int]] = []
    current_streak = outcomes[0]
    streak_len = 1
    for i in range(1, n):
        if outcomes[i] == current_streak:
            streak_len += 1
        else:
            streaks.append((current_streak, streak_len))
            current_streak = outcomes[i]
            streak_len = 1
    streaks.append((current_streak, streak_len))

    avg_streak = sum(s[1] for s in streaks) / len(streaks) if streaks else 0
    max_win_streak = max((s[1] for s in streaks if s[0] == "W"), default=0)
    max_loss_streak = max((s[1] for s in streaks if s[0] == "L"), default=0)

    # Probabilité conditionnelle
    prob_w_after_l = transitions["LW"] / (transitions["LW"] + transitions["LL"]) if (transitions["LW"] + transitions["LL"]) else 0
    prob_l_after_w = transitions["WL"] / (transitions["WW"] + transitions["WL"]) if (transitions["WW"] + transitions["WL"]) else 0
    prob_w_after_w = transitions["WW"] / (transitions["WW"] + transitions["WL"]) if (transitions["WW"] + transitions["WL"]) else 0
    prob_l_after_l = transitions["LL"] / (transitions["LW"] + transitions["LL"]) if (transitions["LW"] + transitions["LL"]) else 0

    return {
        "n_trades": n,
        "n_wins": outcomes.count("W"),
        "n_losses": outcomes.count("L"),
        "win_rate": outcomes.count("W") / n,
        "alternation_rate": alternation_rate,
        "changes": changes,
        "continuations": continuations,
        "transitions": transitions,
        "autocorr_pnl": autocorr,
        "autocorr_binary": autocorr_binary,
        "avg_streak_len": avg_streak,
        "max_win_streak": max_win_streak,
        "max_loss_streak": max_loss_streak,
        "prob_win_after_loss": prob_w_after_l,
        "prob_loss_after_win": prob_l_after_w,
        "prob_win_after_win": prob_w_after_w,
        "prob_loss_after_loss": prob_l_after_l,
        "sequence": "".join(outcomes),
    }


# ── Métriques de performance ──────────────────────────────────────────────────


def compute_performance(
    trades: list[CrashTrade],
    equity_curve: list[EquityPoint],
    initial_balance: float,
    years: float,
) -> dict:
    """Calcule les KPIs pour un run."""
    n = len(trades)
    final_eq = equity_curve[-1].equity if equity_curve else initial_balance
    total_return = (final_eq - initial_balance) / initial_balance
    cagr = (final_eq / initial_balance) ** (1 / years) - 1 if final_eq > 0 and years > 0 else 0

    # Drawdown
    peak = initial_balance
    max_dd = 0.0
    for pt in equity_curve:
        peak = max(peak, pt.equity)
        dd = (pt.equity - peak) / peak if peak else 0
        max_dd = min(max_dd, dd)

    # Trades stats
    if n:
        wins = [t for t in trades if t.pnl_usd > 0]
        losses = [t for t in trades if t.pnl_usd <= 0]
        win_rate = len(wins) / n
        gross_profit = sum(t.pnl_usd for t in wins) or 0
        gross_loss = abs(sum(t.pnl_usd for t in losses)) or 1e-9
        profit_factor = gross_profit / gross_loss
        avg_pnl = sum(t.pnl_usd for t in trades) / n
        avg_win = sum(t.pnl_usd for t in wins) / len(wins) if wins else 0
        avg_loss = sum(t.pnl_usd for t in losses) / len(losses) if losses else 0
        total_fees = sum(t.fees for t in trades)
    else:
        win_rate = profit_factor = avg_pnl = avg_win = avg_loss = total_fees = 0

    return {
        "n_trades": n,
        "final_equity": round(final_eq, 2),
        "total_return": total_return,
        "cagr": cagr,
        "max_drawdown": max_dd,
        "win_rate": win_rate,
        "profit_factor": profit_factor,
        "avg_pnl_usd": avg_pnl,
        "avg_win_usd": avg_win,
        "avg_loss_usd": avg_loss,
        "total_fees": total_fees,
    }


# ── Affichage ──────────────────────────────────────────────────────────────────


def print_alternation_report(analysis: dict) -> None:
    """Affiche le rapport d'alternance W/L."""
    print("\n" + "=" * 75)
    print("  📊  ANALYSE D'ALTERNANCE W/L — CRASHBOT")
    print("=" * 75)

    print(f"\n  Trades total : {analysis['n_trades']}")
    print(f"  Wins : {analysis['n_wins']}  |  Losses : {analysis['n_losses']}  |  Win Rate : {analysis['win_rate']:.1%}")

    print(f"\n  ── Alternance ──")
    print(f"  Taux d'alternance : {analysis['alternation_rate']:.1%}  (>50% = alternance, <50% = continuation)")
    print(f"  Changements (W→L / L→W) : {analysis['changes']}")
    print(f"  Continuations (W→W / L→L) : {analysis['continuations']}")

    print(f"\n  ── Transitions ──")
    t = analysis["transitions"]
    print(f"  W→W : {t['WW']:3d}  |  W→L : {t['WL']:3d}  →  P(L|W) = {analysis['prob_loss_after_win']:.1%}")
    print(f"  L→W : {t['LW']:3d}  |  L→L : {t['LL']:3d}  →  P(W|L) = {analysis['prob_win_after_loss']:.1%}")

    print(f"\n  ── Autocorrélation lag-1 ──")
    print(f"  PnL continu : {analysis['autocorr_pnl']:+.3f}  (négatif = alternance)")
    print(f"  Binaire W/L : {analysis['autocorr_binary']:+.3f}  (négatif = alternance)")

    print(f"\n  ── Streaks ──")
    print(f"  Streak moyen : {analysis['avg_streak_len']:.1f} trades")
    print(f"  Max win streak : {analysis['max_win_streak']}")
    print(f"  Max loss streak : {analysis['max_loss_streak']}")

    # Séquence (tronquée)
    seq = analysis["sequence"]
    if len(seq) > 80:
        print(f"\n  Séquence (premiers 80) : {seq[:80]}…")
    else:
        print(f"\n  Séquence : {seq}")

    # Verdict
    print(f"\n  ── VERDICT ──")
    alt_rate = analysis["alternation_rate"]
    acorr = analysis["autocorr_binary"]
    if alt_rate > 0.55 and acorr < -0.1:
        print("  ✅ ALTERNANCE SIGNIFICATIVE — L'hypothèse tient !")
        print("     → L'allocation adaptative (↑ après perte) pourrait être profitable.")
    elif alt_rate > 0.50:
        print("  🟡 LÉGÈRE ALTERNANCE — Marginalement au-dessus du random.")
        print("     → L'avantage est faible, mais testable.")
    else:
        print("  ❌ PAS D'ALTERNANCE — Les résultats ne montrent pas de pattern W/L.")
        print("     → L'allocation adaptative n'a pas de justification statistique.")
    print()


def print_comparison_report(results: dict[str, dict], analysis: dict) -> None:
    """Affiche le comparatif des 3 modes."""
    print("\n" + "=" * 75)
    print("  📈  COMPARAISON DES MODES D'ALLOCATION — CRASHBOT")
    print("=" * 75)

    headers = ["Métrique", "BASELINE (fixe)", "ADAPTIVE (↑loss)", "INVERSE (↑win)"]
    rows = [
        ("Trades", "n_trades", "d"),
        ("Equity finale", "final_equity", "$.2f"),
        ("Return total", "total_return", ".1%"),
        ("CAGR", "cagr", ".1%"),
        ("Max Drawdown", "max_drawdown", ".1%"),
        ("Win Rate", "win_rate", ".1%"),
        ("Profit Factor", "profit_factor", ".2f"),
        ("Avg PnL/trade", "avg_pnl_usd", "$+.2f"),
        ("Avg Win", "avg_win_usd", "$+.2f"),
        ("Avg Loss", "avg_loss_usd", "$+.2f"),
        ("Fees", "total_fees", "$.2f"),
    ]

    # Header
    print(f"\n  {'Métrique':<18} {'BASELINE':>14} {'ADAPTIVE':>14} {'INVERSE':>14}")
    print(f"  {'─' * 18} {'─' * 14} {'─' * 14} {'─' * 14}")

    for label, key, fmt in rows:
        vals = []
        for mode in ["baseline", "adaptive", "inverse"]:
            v = results[mode].get(key, 0)
            if fmt.startswith("$") and "%" not in fmt:
                actual_fmt = fmt[1:]  # strip $
                vals.append(f"${v:{actual_fmt}}")
            elif "%" in fmt:
                vals.append(f"{v:{fmt}}")
            else:
                vals.append(f"{v:{fmt}}")
        print(f"  {label:<18} {vals[0]:>14} {vals[1]:>14} {vals[2]:>14}")

    # Verdict
    print(f"\n  ── VERDICT ──")
    base_eq = results["baseline"]["final_equity"]
    adapt_eq = results["adaptive"]["final_equity"]
    inv_eq = results["inverse"]["final_equity"]

    best_mode = "baseline"
    best_eq = base_eq
    if adapt_eq > best_eq:
        best_mode = "adaptive"
        best_eq = adapt_eq
    if inv_eq > best_eq:
        best_mode = "inverse"
        best_eq = inv_eq

    delta_adapt = (adapt_eq - base_eq) / base_eq * 100 if base_eq else 0
    delta_inv = (inv_eq - base_eq) / base_eq * 100 if base_eq else 0

    print(f"  ADAPTIVE vs BASELINE : {'+' if delta_adapt > 0 else ''}{delta_adapt:.1f}%")
    print(f"  INVERSE  vs BASELINE : {'+' if delta_inv > 0 else ''}{delta_inv:.1f}%")

    if best_mode == "adaptive" and delta_adapt > 5:
        print(f"\n  ✅ L'ALLOCATION ADAPTIVE GAGNE ! (+{delta_adapt:.1f}% vs baseline)")
        print("     → L'hypothèse d'alternance est CONFIRMÉE par le PnL.")
    elif best_mode == "inverse" and delta_inv > 5:
        print(f"\n  ⚠️ L'INVERSE GAGNE (+{delta_inv:.1f}%) — les résultats se continuent plutôt qu'alterner.")
    else:
        print(f"\n  🟡 Différence marginale entre les modes — pas d'edge clair.")
    print()


def plot_comparison(
    all_equity: dict[str, list[EquityPoint]],
    start_date: datetime,
    end_date: datetime,
    initial_balance: float,
) -> Path:
    """Génère le graphique comparatif des equity curves."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 1, figsize=(14, 10), gridspec_kw={"height_ratios": [3, 1]})

    # Equity curves
    ax = axes[0]
    colors = {"baseline": "#888888", "adaptive": "#2ecc71", "inverse": "#e74c3c"}
    labels = {"baseline": "Baseline (risque fixe)", "adaptive": "Adaptive (↑ après perte)", "inverse": "Inverse (↑ après gain)"}

    for mode, eq in all_equity.items():
        dates = [datetime.fromtimestamp(pt.ts / 1000, tz=timezone.utc) for pt in eq]
        values = [pt.equity for pt in eq]
        ax.plot(dates, values, label=labels[mode], color=colors[mode], linewidth=1.5)

    ax.axhline(y=initial_balance, color="gray", linestyle="--", alpha=0.5, label="Capital initial")
    ax.set_title("CrashBot — Comparaison Allocation Adaptative", fontsize=14, fontweight="bold")
    ax.set_ylabel("Equity ($)")
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))

    # Drawdown
    ax2 = axes[1]
    for mode, eq in all_equity.items():
        dates = [datetime.fromtimestamp(pt.ts / 1000, tz=timezone.utc) for pt in eq]
        peak = initial_balance
        dd = []
        for pt in eq:
            peak = max(peak, pt.equity)
            dd.append((pt.equity - peak) / peak * 100 if peak else 0)
        ax2.fill_between(dates, dd, 0, alpha=0.3, color=colors[mode], label=labels[mode])
        ax2.plot(dates, dd, color=colors[mode], linewidth=0.8)

    ax2.set_ylabel("Drawdown (%)")
    ax2.set_xlabel("Date")
    ax2.grid(True, alpha=0.3)
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax2.xaxis.set_major_locator(mdates.MonthLocator(interval=3))

    plt.tight_layout()
    out_path = OUTPUT_DIR / "crashbot_adaptive_comparison.png"
    plt.savefig(out_path, dpi=150)
    plt.close()
    return out_path


# ── Main ───────────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(description="Backtest CrashBot — Allocation Adaptative")
    parser.add_argument("--pairs", type=str, default=None, help="Paires séparées par virgule")
    parser.add_argument("--years", type=int, default=3, help="Années de données (défaut: 3)")
    parser.add_argument("--balance", type=float, default=500, help="Capital initial ($)")
    parser.add_argument("--risk", type=float, default=0.05, help="Risque de base (%)")
    parser.add_argument("--drop", type=float, default=0.15, help="Seuil de crash (%)")
    parser.add_argument("--tp", type=float, default=0.08, help="Take profit (%)")
    parser.add_argument("--boost", type=float, default=1.8, help="Mult risque après perte/gain")
    parser.add_argument("--shrink", type=float, default=0.5, help="Mult risque après gain/perte")
    args = parser.parse_args()

    pairs = args.pairs.split(",") if args.pairs else DEFAULT_PAIRS
    end = datetime.now(timezone.utc)
    start = end - timedelta(days=args.years * 365)

    print(f"\n🤖 CrashBot Adaptive Backtest")
    print(f"   Paires : {len(pairs)} | Période : {start:%Y-%m-%d} → {end:%Y-%m-%d}")
    print(f"   Capital : ${args.balance:.0f} | Risque base : {args.risk:.0%}")
    print(f"   Drop seuil : {args.drop:.0%} | TP : {args.tp:.0%}")
    print(f"   Boost mult : {args.boost:.1f}× | Shrink mult : {args.shrink:.1f}×")

    # Config CrashBot
    cfg = CrashConfig(
        drop_threshold=args.drop,
        tp_pct=args.tp,
    )

    # Téléchargement des données
    print(f"\n📥 Téléchargement des données H4…")
    all_candles: dict[str, list[Candle]] = {}
    for pair in pairs:
        try:
            candles = download_candles(pair, start, end, interval="4h")
            if len(candles) > 100:
                all_candles[pair] = candles
                print(f"   ✅ {pair}: {len(candles)} bougies")
            else:
                print(f"   ⚠️ {pair}: seulement {len(candles)} bougies, skip")
        except Exception as e:
            print(f"   ❌ {pair}: {e}")

    if not all_candles:
        print("❌ Aucune donnée disponible")
        sys.exit(1)

    years_actual = (end - start).days / 365.25

    # ── 1. Run BASELINE ────────────────────────────────────────────────────
    print(f"\n🔵 Run BASELINE (risque fixe {args.risk:.0%})…")
    trades_base, eq_base = simulate_crashbot(
        all_candles, cfg, args.balance, args.risk,
        allocation_mode="baseline",
    )
    perf_base = compute_performance(trades_base, eq_base, args.balance, years_actual)
    print(f"   {perf_base['n_trades']} trades | Equity: ${perf_base['final_equity']:.2f} | PF: {perf_base['profit_factor']:.2f}")

    # ── 2. Analyse d'alternance ────────────────────────────────────────────
    analysis = analyze_alternation(trades_base)
    print_alternation_report(analysis)

    # ── 3. Run ADAPTIVE (↑ après perte) ────────────────────────────────────
    print(f"🟢 Run ADAPTIVE (↑{args.boost:.1f}× après perte, ↓{args.shrink:.1f}× après gain)…")
    trades_adapt, eq_adapt = simulate_crashbot(
        all_candles, cfg, args.balance, args.risk,
        allocation_mode="adaptive",
        boost_mult=args.boost,
        shrink_mult=args.shrink,
    )
    perf_adapt = compute_performance(trades_adapt, eq_adapt, args.balance, years_actual)
    print(f"   {perf_adapt['n_trades']} trades | Equity: ${perf_adapt['final_equity']:.2f} | PF: {perf_adapt['profit_factor']:.2f}")

    # ── 4. Run INVERSE (↑ après gain) ──────────────────────────────────────
    print(f"🔴 Run INVERSE (↑{args.boost:.1f}× après gain, ↓{args.shrink:.1f}× après perte)…")
    trades_inv, eq_inv = simulate_crashbot(
        all_candles, cfg, args.balance, args.risk,
        allocation_mode="inverse",
        boost_mult=args.boost,
        shrink_mult=args.shrink,
    )
    perf_inv = compute_performance(trades_inv, eq_inv, args.balance, years_actual)
    print(f"   {perf_inv['n_trades']} trades | Equity: ${perf_inv['final_equity']:.2f} | PF: {perf_inv['profit_factor']:.2f}")

    # ── 5. Comparaison ─────────────────────────────────────────────────────
    results = {
        "baseline": perf_base,
        "adaptive": perf_adapt,
        "inverse": perf_inv,
    }
    print_comparison_report(results, analysis)

    # ── 6. Graphique ───────────────────────────────────────────────────────
    all_equity = {
        "baseline": eq_base,
        "adaptive": eq_adapt,
        "inverse": eq_inv,
    }
    chart_path = plot_comparison(all_equity, start, end, args.balance)
    print(f"📊 Graphique sauvé : {chart_path}")

    # ── 7. Détail par paire (analyse d'alternance par paire) ───────────────
    print("\n" + "=" * 75)
    print("  📋  ALTERNANCE PAR PAIRE (BASELINE)")
    print("=" * 75)

    from collections import defaultdict
    by_pair: dict[str, list[CrashTrade]] = defaultdict(list)
    for t in trades_base:
        by_pair[t.symbol].append(t)

    print(f"\n  {'Paire':<12} {'Trades':>6} {'WR':>6} {'Alt%':>6} {'AutoCorr':>10} {'P(W|L)':>7} {'P(L|W)':>7} {'PnL':>10}")
    print(f"  {'─' * 12} {'─' * 6} {'─' * 6} {'─' * 6} {'─' * 10} {'─' * 7} {'─' * 7} {'─' * 10}")

    for pair in sorted(by_pair.keys()):
        pair_trades = by_pair[pair]
        if len(pair_trades) < 3:
            continue
        a = analyze_alternation(pair_trades)
        pnl = sum(t.pnl_usd for t in pair_trades)
        print(
            f"  {pair:<12} {a['n_trades']:>6d} {a['win_rate']:>5.0%} {a['alternation_rate']:>5.0%}"
            f" {a['autocorr_binary']:>+9.3f} {a['prob_win_after_loss']:>6.0%} {a['prob_loss_after_win']:>6.0%}"
            f" {'+' if pnl > 0 else ''}{pnl:>9.2f}"
        )
    print()


if __name__ == "__main__":
    main()
