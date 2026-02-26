#!/usr/bin/env python
"""
Backtest comparatif : RANGE only sur 5 paires vs 20 paires.

Mesure l'impact du scaling horizontal (plus de paires = plus de trades).
"""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import datetime, timezone

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
logger = logging.getLogger("scale_test")


PAIRS_5 = ["BTC-USD", "SOL-USD", "XRP-USD", "LINK-USD", "ETH-USD"]

PAIRS_20 = [
    "BTC-USD", "ETH-USD", "SOL-USD", "XRP-USD", "LINK-USD",
    "ADA-USD", "DOT-USD", "AVAX-USD", "DOGE-USD", "ATOM-USD",
    "UNI-USD", "NEAR-USD", "ALGO-USD", "LTC-USD", "ETC-USD",
    "FIL-USD", "AAVE-USD", "INJ-USD", "SAND-USD", "MANA-USD",
]


def _base_config(balance: float) -> BacktestConfig:
    return BacktestConfig(
        initial_balance=balance,
        risk_percent_trend=config.RISK_PERCENT_TREND,
        risk_percent_range=config.RISK_PERCENT_RANGE,
        entry_buffer_pct=config.ENTRY_BUFFER_PERCENT,
        sl_buffer_pct=config.SL_BUFFER_PERCENT,
        zero_risk_trigger_pct=config.ZERO_RISK_TRIGGER_PERCENT,
        zero_risk_lock_pct=config.ZERO_RISK_LOCK_PERCENT,
        trailing_stop_pct=config.TRAILING_STOP_PERCENT,
        max_position_pct=config.MAX_POSITION_PERCENT,
        max_simultaneous_positions=config.MAX_SIMULTANEOUS_POSITIONS,
        swing_lookback=config.SWING_LOOKBACK,
        range_width_min=config.RANGE_WIDTH_MIN,
        range_entry_buffer_pct=config.RANGE_ENTRY_BUFFER_PERCENT,
        range_sl_buffer_pct=config.RANGE_SL_BUFFER_PERCENT,
        range_cooldown_bars=config.RANGE_COOLDOWN_BARS,
        max_total_risk_pct=config.MAX_TOTAL_RISK_PERCENT,
        enable_trend=False,
        enable_range=True,
    )


def run_one(pairs: list[str], candles: dict, balance: float) -> tuple[dict, list]:
    cfg = _base_config(balance)
    # Filtrer les candles pour les paires demandées
    filtered = {p: candles[p] for p in pairs if p in candles}
    engine = BacktestEngine(filtered, cfg)
    result = engine.run()
    metrics = compute_metrics(result)
    return metrics, result.trades


def main() -> None:
    parser = argparse.ArgumentParser(description="TradeX Scale Test — 5 vs 20 paires")
    parser.add_argument("--balance", type=float, default=1000.0)
    parser.add_argument("--start", type=str, default="2022-02-20")
    parser.add_argument("--end", type=str, default="2026-02-20")
    args = parser.parse_args()

    start = datetime.strptime(args.start, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    end = datetime.strptime(args.end, "%Y-%m-%d").replace(tzinfo=timezone.utc)

    logger.info("🚀 Scale Test — %s → %s | $%.0f", start.date(), end.date(), args.balance)

    # Télécharger toutes les 20 paires (le cache gère les 5 existantes)
    logger.info("\n📥 Téléchargement des 20 paires H4…")
    all_candles = download_all_pairs(PAIRS_20, start, end, interval="4h")

    # ── Test 1 : 5 paires (baseline) ──────────────────────────────────────
    logger.info("\n" + "═" * 60)
    logger.info("  [1/2] RANGE only — 5 paires (baseline)")
    logger.info("═" * 60)
    m5, trades5 = run_one(PAIRS_5, all_candles, args.balance)

    # ── Test 2 : 20 paires ────────────────────────────────────────────────
    logger.info("\n" + "═" * 60)
    logger.info("  [2/2] RANGE only — 20 paires")
    logger.info("═" * 60)
    m20, trades20 = run_one(PAIRS_20, all_candles, args.balance)

    # ── Résultats ─────────────────────────────────────────────────────────
    sep = "═" * 90
    print(f"\n{sep}")
    print("  📊 COMPARAISON SCALING — RANGE only : 5 paires vs 20 paires")
    print(sep)

    header = (
        f"  {'Config':<28s} │ {'Return':>8s} │ {'CAGR':>7s} │ {'MaxDD':>7s} │ "
        f"{'Sharpe':>7s} │ {'PF':>5s} │ {'WR':>5s} │ {'Trades':>6s} │ "
        f"{'$/mois':>8s} │ {'Final$':>10s}"
    )
    print(header)
    print("  " + "─" * 86)

    for label, m in [("RANGE 5 paires", m5), ("RANGE 20 paires", m20)]:
        monthly = m.get("monthly_returns", [])
        avg_monthly_usd = 0
        if monthly:
            months = len(monthly)
            total_pnl = m["final_equity"] - args.balance
            avg_monthly_usd = total_pnl / months

        print(
            f"  {label:<28s} │ {m['total_return']:>+7.1%} │ {m['cagr']:>+6.1%} │ "
            f"{m['max_drawdown']:>7.1%} │ {m['sharpe']:>7.2f} │ {m['profit_factor']:>5.2f} │ "
            f"{m['win_rate']:>4.0%} │ {m['n_trades']:>6d} │ "
            f"${avg_monthly_usd:>+7.2f} │ ${m['final_equity']:>9,.2f}"
        )

    # Trades par mois
    print(f"\n  {'─' * 86}")
    n_months = 48  # ~4 ans
    print(f"  RANGE 5 paires  : {m5['n_trades'] / n_months:.1f} trades/mois")
    print(f"  RANGE 20 paires : {m20['n_trades'] / n_months:.1f} trades/mois")
    ratio = m20["n_trades"] / m5["n_trades"] if m5["n_trades"] else 0
    print(f"  Facteur : x{ratio:.1f}")

    # Détail par paire (20 paires)
    print(f"\n  {'─' * 86}")
    print("  📋 Détail par paire (20 paires) :")
    pair_stats: dict[str, dict] = {}
    for t in trades20:
        sym = t.symbol
        if sym not in pair_stats:
            pair_stats[sym] = {"n": 0, "pnl": 0.0, "wins": 0}
        pair_stats[sym]["n"] += 1
        pair_stats[sym]["pnl"] += t.pnl_usd
        if t.pnl_usd > 0:
            pair_stats[sym]["wins"] += 1

    # Trier par PnL décroissant
    sorted_pairs = sorted(pair_stats.items(), key=lambda x: x[1]["pnl"], reverse=True)
    for sym, s in sorted_pairs:
        wr = s["wins"] / s["n"] * 100 if s["n"] else 0
        emoji = "🟢" if s["pnl"] > 0 else "🔴"
        print(f"    {emoji} {sym:<10s} : {s['n']:>3d} trades | WR {wr:>4.0f}% | PnL ${s['pnl']:>+8.2f}")

    print(f"\n{sep}\n")


if __name__ == "__main__":
    main()
