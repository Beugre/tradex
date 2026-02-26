#!/usr/bin/env python
"""
Grid search paramètres RANGE — optimiser le return sans casser le PF.

Teste les combinaisons de :
  - max_simultaneous_positions : 3, 5, 8
  - risk_percent_range : 0.02, 0.03, 0.04
  - compound : False, True
"""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import datetime, timezone
from itertools import product

from dotenv import load_dotenv

load_dotenv()

from src import config
from backtest.data_loader import download_all_pairs
from backtest.simulator import BacktestConfig, BacktestEngine
from backtest.metrics import compute_metrics

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("grid_search")

PAIRS_20 = [
    "BTC-USD", "ETH-USD", "SOL-USD", "XRP-USD", "LINK-USD",
    "ADA-USD", "DOT-USD", "AVAX-USD", "DOGE-USD", "ATOM-USD",
    "UNI-USD", "NEAR-USD", "ALGO-USD", "LTC-USD", "ETC-USD",
    "FIL-USD", "AAVE-USD", "INJ-USD", "SAND-USD", "MANA-USD",
]

# ── Grille de paramètres ───────────────────────────────────────────────────────

GRID = {
    "max_pos": [3, 5, 8],
    "risk_pct": [0.02, 0.03, 0.04, 0.05],
    "compound": [False, True],
    "max_pos_pct": [0.20, 0.30],
}


def run_config(
    candles: dict,
    balance: float,
    max_pos: int,
    risk_pct: float,
    compound: bool,
    max_pos_pct: float,
) -> dict:
    cfg = BacktestConfig(
        initial_balance=balance,
        risk_percent_trend=config.RISK_PERCENT_TREND,
        risk_percent_range=risk_pct,
        entry_buffer_pct=config.ENTRY_BUFFER_PERCENT,
        sl_buffer_pct=config.SL_BUFFER_PERCENT,
        zero_risk_trigger_pct=config.ZERO_RISK_TRIGGER_PERCENT,
        zero_risk_lock_pct=config.ZERO_RISK_LOCK_PERCENT,
        trailing_stop_pct=config.TRAILING_STOP_PERCENT,
        max_position_pct=max_pos_pct,
        max_simultaneous_positions=max_pos,
        swing_lookback=config.SWING_LOOKBACK,
        range_width_min=config.RANGE_WIDTH_MIN,
        range_entry_buffer_pct=config.RANGE_ENTRY_BUFFER_PERCENT,
        range_sl_buffer_pct=config.RANGE_SL_BUFFER_PERCENT,
        range_cooldown_bars=config.RANGE_COOLDOWN_BARS,
        max_total_risk_pct=config.MAX_TOTAL_RISK_PERCENT,
        enable_trend=False,
        enable_range=True,
        compound=compound,
    )
    engine = BacktestEngine(candles, cfg)
    result = engine.run()
    return compute_metrics(result)


def main() -> None:
    parser = argparse.ArgumentParser(description="TradeX Grid Search — RANGE params")
    parser.add_argument("--balance", type=float, default=1000.0)
    parser.add_argument("--start", type=str, default="2022-02-20")
    parser.add_argument("--end", type=str, default="2026-02-20")
    args = parser.parse_args()

    start = datetime.strptime(args.start, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    end = datetime.strptime(args.end, "%Y-%m-%d").replace(tzinfo=timezone.utc)

    logger.info("🔍 Grid Search RANGE — %s → %s | $%.0f", start.date(), end.date(), args.balance)
    candles = download_all_pairs(PAIRS_20, start, end, interval="4h")

    # Générer toutes les combinaisons
    combos = list(product(
        GRID["max_pos"], GRID["risk_pct"], GRID["compound"], GRID["max_pos_pct"],
    ))
    logger.info("📊 %d combinaisons à tester", len(combos))

    results = []
    for i, (max_pos, risk_pct, compound, max_pos_pct) in enumerate(combos, 1):
        label = (
            f"pos={max_pos} risk={risk_pct:.0%} "
            f"{'CMP' if compound else 'FIX'} cap={max_pos_pct:.0%}"
        )
        print(f"\r  ⏳ [{i}/{len(combos)}] {label}…", end="", flush=True)

        m = run_config(candles, args.balance, max_pos, risk_pct, compound, max_pos_pct)
        results.append({
            "max_pos": max_pos,
            "risk_pct": risk_pct,
            "compound": compound,
            "max_pos_pct": max_pos_pct,
            "label": label,
            **m,
        })
    print()

    # Trier par return décroissant
    results.sort(key=lambda r: r["total_return"], reverse=True)

    # Affichage
    sep = "═" * 130
    print(f"\n{sep}")
    print("  🔍 GRID SEARCH RÉSULTATS — RANGE 20 paires (trié par Return)")
    print(sep)
    print(
        f"  {'#':>3s} │ {'MaxPos':>6s} │ {'Risk%':>5s} │ {'Mode':>4s} │ {'Cap%':>4s} │ "
        f"{'Return':>8s} │ {'CAGR':>7s} │ {'MaxDD':>7s} │ {'Sharpe':>7s} │ "
        f"{'PF':>5s} │ {'WR':>5s} │ {'Trades':>6s} │ {'Final$':>10s}"
    )
    print("  " + "─" * 126)

    for i, r in enumerate(results, 1):
        mode = "CMP" if r["compound"] else "FIX"
        # Highlight : PF > 1.3 ET MaxDD > -15%
        flag = ""
        if r["profit_factor"] >= 1.3 and r["max_drawdown"] > -0.15:
            flag = " ⭐"
        elif r["profit_factor"] >= 1.2 and r["max_drawdown"] > -0.10:
            flag = " ✅"

        print(
            f"  {i:>3d} │ {r['max_pos']:>6d} │ {r['risk_pct']:>4.0%} │ {mode:>4s} │ "
            f"{r['max_pos_pct']:>3.0%} │ {r['total_return']:>+7.1%} │ {r['cagr']:>+6.1%} │ "
            f"{r['max_drawdown']:>7.1%} │ {r['sharpe']:>7.2f} │ {r['profit_factor']:>5.2f} │ "
            f"{r['win_rate']:>4.0%} │ {r['n_trades']:>6d} │ ${r['final_equity']:>9,.2f}{flag}"
        )

        if i == 10:
            print("  " + "─" * 126)
            print("  … (suite tronquée)")
            break

    # Meilleur risque-ajusté (Sharpe > 0.8 ET MaxDD > -10%)
    safe = [r for r in results if r["sharpe"] > 0.8 and r["max_drawdown"] > -0.10]
    if safe:
        safe.sort(key=lambda r: r["total_return"], reverse=True)
        best = safe[0]
        print(f"\n  🏆 MEILLEUR risque-ajusté (Sharpe>0.8, DD>-10%) :")
        mode = "CMP" if best["compound"] else "FIX"
        print(
            f"     pos={best['max_pos']} risk={best['risk_pct']:.0%} {mode} "
            f"cap={best['max_pos_pct']:.0%}"
        )
        print(
            f"     Return: {best['total_return']:+.1%} | CAGR: {best['cagr']:+.1%} | "
            f"MaxDD: {best['max_drawdown']:.1%} | Sharpe: {best['sharpe']:.2f} | "
            f"PF: {best['profit_factor']:.2f} | {best['n_trades']} trades"
        )

    # Meilleur return absolu avec PF > 1.2
    aggro = [r for r in results if r["profit_factor"] > 1.2]
    if aggro:
        best_a = aggro[0]
        print(f"\n  🚀 MEILLEUR return absolu (PF>1.2) :")
        mode = "CMP" if best_a["compound"] else "FIX"
        print(
            f"     pos={best_a['max_pos']} risk={best_a['risk_pct']:.0%} {mode} "
            f"cap={best_a['max_pos_pct']:.0%}"
        )
        print(
            f"     Return: {best_a['total_return']:+.1%} | CAGR: {best_a['cagr']:+.1%} | "
            f"MaxDD: {best_a['max_drawdown']:.1%} | Sharpe: {best_a['sharpe']:.2f} | "
            f"PF: {best_a['profit_factor']:.2f} | {best_a['n_trades']} trades"
        )

    print(f"\n{sep}\n")


if __name__ == "__main__":
    main()
