#!/usr/bin/env python3
"""
Grid search Infinity Bot — trouve la meilleure config par paire.

Usage:
    python -m backtest.run_infinity_scanner
    python -m backtest.run_infinity_scanner --pairs ETH-USD,SOL-USD
    python -m backtest.run_infinity_scanner --pairs ETH-USD --years 4

Teste plusieurs combinaisons de paramètres clés :
  - buy_levels (seuils DCA)
  - stop_loss_pct
  - sell_levels (paliers TP)
  - trailing_high_period
"""

from __future__ import annotations

import argparse
import logging
import sys
from dataclasses import replace
from datetime import datetime, timezone
from itertools import product
from pathlib import Path

from backtest.data_loader import download_candles
from backtest.run_backtest_infinity import run_infinity_backtest, InfinityTrade
from src.core.infinity_engine import InfinityConfig

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(message)s")
logger = logging.getLogger(__name__)


# ── Grille de configs à tester ─────────────────────────────────────────────────

BUY_LEVEL_PRESETS = {
    "A (-5,-10,-15,-20,-25)": (-0.05, -0.10, -0.15, -0.20, -0.25),
    "B (-7,-12,-18,-24,-30)": (-0.07, -0.12, -0.18, -0.24, -0.30),
    "C (-10,-15,-20,-25,-30)": (-0.10, -0.15, -0.20, -0.25, -0.30),
    "D (-8,-14,-20,-26,-32)": (-0.08, -0.14, -0.20, -0.26, -0.32),
    "E (-10,-18,-25,-32,-40)": (-0.10, -0.18, -0.25, -0.32, -0.40),
    "F (-12,-20,-28,-35,-42)": (-0.12, -0.20, -0.28, -0.35, -0.42),
    "G (-15,-22,-30,-38,-45)": (-0.15, -0.22, -0.30, -0.38, -0.45),
}

SELL_LEVEL_PRESETS = {
    "s1 (+0.8,+1.5,+2.2,+3.0,+4.0)": (0.008, 0.015, 0.022, 0.030, 0.040),
    "s2 (+1.0,+2.0,+3.0,+4.5,+6.0)": (0.010, 0.020, 0.030, 0.045, 0.060),
    "s3 (+1.5,+3.0,+5.0,+7.0,+10.)": (0.015, 0.030, 0.050, 0.070, 0.100),
    "s4 (+2.0,+4.0,+6.0,+8.0,+12.)": (0.020, 0.040, 0.060, 0.080, 0.120),
}

STOP_LOSS_OPTIONS = [0.12, 0.15, 0.20, 0.25]

TRAILING_HIGH_OPTIONS = [48, 72, 120]  # 8j, 12j, 20j


# ── Métriques ──────────────────────────────────────────────────────────────────

def compute_metrics(
    trades: list[InfinityTrade],
    equity_curve: list[float],
    initial_balance: float,
) -> dict:
    """Calcule les métriques à partir des résultats de backtest."""
    if not trades:
        return {
            "trades": 0, "wr": 0, "pnl": 0, "pf": 0,
            "ret": 0, "dd": 0, "sharpe": 0, "sl_count": 0
        }

    wins = [t for t in trades if t.pnl_usd > 0]
    losses = [t for t in trades if t.pnl_usd <= 0]
    total_pnl = sum(t.pnl_usd for t in trades)
    gross_gain = sum(t.pnl_usd for t in wins) if wins else 0
    gross_loss = abs(sum(t.pnl_usd for t in losses)) if losses else 0
    pf = gross_gain / gross_loss if gross_loss > 0 else 99.0
    wr = len(wins) / len(trades) * 100 if trades else 0
    ret = total_pnl / initial_balance * 100

    # Max drawdown
    peak = initial_balance
    max_dd = 0
    for eq in equity_curve:
        if eq > peak:
            peak = eq
        dd = (peak - eq) / peak
        if dd > max_dd:
            max_dd = dd

    # Sharpe annualisé (simplifié)
    if len(equity_curve) > 1:
        returns = []
        for j in range(1, len(equity_curve)):
            if equity_curve[j - 1] > 0:
                returns.append(equity_curve[j] / equity_curve[j - 1] - 1)
        if returns:
            import statistics
            avg_r = statistics.mean(returns)
            std_r = statistics.stdev(returns) if len(returns) > 1 else 1e-9
            bars_per_year = 6 * 365  # H4
            sharpe = (avg_r / std_r) * (bars_per_year ** 0.5) if std_r > 0 else 0
        else:
            sharpe = 0
    else:
        sharpe = 0

    sl_count = sum(1 for t in trades if t.exit_reason == "STOP_LOSS")

    return {
        "trades": len(trades),
        "wr": round(wr, 1),
        "pnl": round(total_pnl, 2),
        "pf": round(pf, 2),
        "ret": round(ret, 2),
        "dd": round(max_dd * 100, 2),
        "sharpe": round(sharpe, 2),
        "sl_count": sl_count,
    }


# ── Score composite ────────────────────────────────────────────────────────────

def score_config(m: dict) -> float:
    """
    Score composite pour classer les configs.
    Pondère: rendement, PF, Sharpe, drawdown inversé, taux de SL.
    """
    if m["trades"] < 20:
        return -999  # Pas assez de trades

    ret_score = m["ret"]                           # +29% → 29
    pf_score = (m["pf"] - 1.0) * 50               # PF 1.37 → 18.5
    sharpe_score = m["sharpe"] * 15                # 0.48 → 7.2
    dd_penalty = m["dd"] * -2                      # 11.58% → -23.16
    sl_penalty = m["sl_count"] * -3                # 10 SL → -30

    return ret_score + pf_score + sharpe_score + dd_penalty + sl_penalty


# ── Runner ─────────────────────────────────────────────────────────────────────

def run_scan(
    pair: str,
    candles: list,
    initial_balance: float,
    capital_pct: float,
) -> list[dict]:
    """
    Lance le grid search sur toutes les combinaisons de paramètres.
    Retourne la liste triée par score (meilleur en premier).
    """
    results = []
    combos = list(product(
        BUY_LEVEL_PRESETS.items(),
        SELL_LEVEL_PRESETS.items(),
        STOP_LOSS_OPTIONS,
        TRAILING_HIGH_OPTIONS,
    ))
    total = len(combos)
    logger.info("🔍 %s — %d combinaisons à tester", pair, total)

    for idx, ((bl_name, buy_levels), (sl_name, sell_levels), stop_loss, trail_period) in enumerate(combos):
        if (idx + 1) % 50 == 0:
            logger.info("   ... %d/%d", idx + 1, total)

        config = InfinityConfig(
            initial_balance=initial_balance,
            buy_levels=buy_levels,
            entry_drop_pct=abs(buy_levels[0]),
            max_buy_levels=len(buy_levels),
            buy_pcts=(0.25, 0.20, 0.15, 0.10, 0.00)[:len(buy_levels)],
            buy_amounts=(100.0, 200.0, 300.0, 400.0, 0.0)[:len(buy_levels)],
            sell_levels=sell_levels,
            stop_loss_pct=stop_loss,
            trailing_high_period=trail_period,
        )

        try:
            trades, equity = run_infinity_backtest(
                candles=candles,
                config=config,
                initial_balance=initial_balance,
                capital_pct=capital_pct,
            )
        except Exception:
            continue

        m = compute_metrics(trades, equity, initial_balance)
        m["buy_preset"] = bl_name
        m["sell_preset"] = sl_name
        m["stop_loss"] = stop_loss
        m["trail_period"] = trail_period
        m["score"] = round(score_config(m), 2)
        results.append(m)

    results.sort(key=lambda x: x["score"], reverse=True)
    return results


# ── Affichage ──────────────────────────────────────────────────────────────────

def print_results(pair: str, results: list[dict], top_n: int = 10):
    """Affiche le top N des configs pour une paire."""
    print(f"\n{'=' * 120}")
    print(f"♾️  SCANNER INFINITY — {pair} — TOP {top_n}")
    print(f"{'=' * 120}")

    header = (
        f"{'#':>3} | {'Score':>6} | {'Ret%':>7} | {'PF':>5} | {'Sharpe':>6} | "
        f"{'DD%':>6} | {'WR%':>5} | {'Trd':>4} | {'SL':>3} | "
        f"{'Buy Preset':<28} | {'Sell Preset':<30} | {'SL%':>5} | {'Trail':>5}"
    )
    print(header)
    print("-" * 120)

    for rank, r in enumerate(results[:top_n], 1):
        line = (
            f"{rank:>3} | {r['score']:>6.1f} | {r['ret']:>6.2f}% | {r['pf']:>5.2f} | {r['sharpe']:>6.2f} | "
            f"{r['dd']:>5.2f}% | {r['wr']:>4.1f}% | {r['trades']:>4} | {r['sl_count']:>3} | "
            f"{r['buy_preset']:<28} | {r['sell_preset']:<30} | {r['stop_loss']*100:>4.0f}% | {r['trail_period']:>5}"
        )
        if rank == 1:
            print(f"🏆 {line}")
        else:
            print(f"   {line}")

    print(f"{'=' * 120}")


def print_summary_table(all_results: dict[str, list[dict]]):
    """Affiche un tableau comparatif de la meilleure config par paire."""
    print(f"\n{'=' * 130}")
    print(f"♾️  RÉSUMÉ — MEILLEURE CONFIG PAR PAIRE")
    print(f"{'=' * 130}")

    header = (
        f"{'Paire':<10} | {'Score':>6} | {'Ret%':>7} | {'PF':>5} | {'Sharpe':>6} | "
        f"{'DD%':>6} | {'WR%':>5} | {'Trd':>4} | {'SL':>3} | "
        f"{'Buy Preset':<28} | {'Sell Preset':<30} | {'SL%':>5} | {'Trail':>5}"
    )
    print(header)
    print("-" * 130)

    for pair, results in all_results.items():
        if not results:
            print(f"{pair:<10} | {'N/A':>6} | — pas assez de données —")
            continue
        r = results[0]  # meilleur
        emoji = "🟢" if r["ret"] > 20 else "🟡" if r["ret"] > 0 else "🔴"
        line = (
            f"{pair:<10} | {r['score']:>6.1f} | {r['ret']:>6.2f}% | {r['pf']:>5.2f} | {r['sharpe']:>6.2f} | "
            f"{r['dd']:>5.2f}% | {r['wr']:>4.1f}% | {r['trades']:>4} | {r['sl_count']:>3} | "
            f"{r['buy_preset']:<28} | {r['sell_preset']:<30} | {r['stop_loss']*100:>4.0f}% | {r['trail_period']:>5}"
        )
        print(f"{emoji} {line}")

    print(f"{'=' * 130}")


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Grid Search Infinity Bot")
    parser.add_argument("--pairs", default="ETH-USD,SOL-USD,XLM-USD,AAVE-USD,ADA-USD",
                        help="Paires séparées par des virgules")
    parser.add_argument("--years", type=float, default=6, help="Nombre d'années")
    parser.add_argument("--balance", type=float, default=1000, help="Balance initiale ($)")
    parser.add_argument("--capital-pct", type=float, default=0.65, help="% capital alloué")
    parser.add_argument("--top", type=int, default=10, help="Nombre de configs à afficher par paire")
    args = parser.parse_args()

    pairs = [p.strip() for p in args.pairs.split(",")]
    end = datetime(2026, 3, 6, tzinfo=timezone.utc)
    start_year = end.year - int(args.years)
    start = datetime(start_year, end.month, end.day, tzinfo=timezone.utc)

    all_results: dict[str, list[dict]] = {}

    for pair in pairs:
        logger.info("\n" + "=" * 80)
        logger.info("♾️  SCAN %s (%s → %s)", pair, start.date(), end.date())
        logger.info("=" * 80)

        try:
            candles = download_candles(pair, start, end, interval="4h")
            logger.info("   %d bougies H4 chargées", len(candles))
        except Exception as e:
            logger.error("❌ Impossible de charger %s: %s", pair, e)
            all_results[pair] = []
            continue

        if len(candles) < 200:
            logger.warning("⚠️ %s — pas assez de données (%d bougies)", pair, len(candles))
            all_results[pair] = []
            continue

        results = run_scan(pair, candles, args.balance, args.capital_pct)
        all_results[pair] = results

        print_results(pair, results, top_n=args.top)

    # Résumé final
    print_summary_table(all_results)


if __name__ == "__main__":
    main()
