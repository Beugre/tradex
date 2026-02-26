#!/usr/bin/env python
"""
Test de robustesse — RANGE avec paramètres verrouillés.

Paramètres fixes :
  - max_simultaneous_positions = 3
  - compound = True
  - max_position_pct = 0.30 (cap 30%)
  - risk_percent_range = 0.02 (2%)

31 paires avec ≥6 ans d'historique.

Périodes testées :
  1. 6 ans complet  : 2020-02-20 → 2026-02-20
  2. Sous-période 1 : 2020-02-20 → 2022-02-20 (bull + crash covid)
  3. Sous-période 2 : 2022-02-20 → 2024-02-20 (bear market)
  4. Sous-période 3 : 2024-02-20 → 2026-02-20 (post-halving)

Objectif : vérifier que l'edge est STABLE dans chaque sous-période.
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
logger = logging.getLogger("robustness")

# ── 31 paires avec ≥6 ans d'historique ─────────────────────────────────────────

PAIRS_31 = [
    "BTC-USD", "ETH-USD", "XRP-USD", "LINK-USD", "ADA-USD",
    "DOGE-USD", "ATOM-USD", "ALGO-USD", "LTC-USD", "ETC-USD",
    "MATIC-USD", "VET-USD", "THETA-USD", "FTM-USD", "CHZ-USD",
    "ENJ-USD", "BAT-USD", "ZIL-USD", "ICX-USD", "ONE-USD",
    "HBAR-USD", "IOTA-USD", "XTZ-USD", "EOS-USD", "NEO-USD",
    "DASH-USD", "ZEC-USD", "XLM-USD", "TRX-USD", "WAVES-USD",
    "KAVA-USD",
]

# ── Périodes ───────────────────────────────────────────────────────────────────

PERIODS = [
    ("6yr COMPLET (2020-2026)", "2020-02-20", "2026-02-20"),
    ("2yr P1 (2020-2022)", "2020-02-20", "2022-02-20"),
    ("2yr P2 (2022-2024)", "2022-02-20", "2024-02-20"),
    ("2yr P3 (2024-2026)", "2024-02-20", "2026-02-20"),
]


# ── Config verrouillée ─────────────────────────────────────────────────────────


def _locked_config(balance: float) -> BacktestConfig:
    """Config RANGE avec paramètres optimaux verrouillés."""
    return BacktestConfig(
        initial_balance=balance,
        # Trend (désactivé, mais on met les valeurs par défaut)
        risk_percent_trend=0.03,
        entry_buffer_pct=config.ENTRY_BUFFER_PERCENT,
        sl_buffer_pct=config.SL_BUFFER_PERCENT,
        zero_risk_trigger_pct=config.ZERO_RISK_TRIGGER_PERCENT,
        zero_risk_lock_pct=config.ZERO_RISK_LOCK_PERCENT,
        trailing_stop_pct=config.TRAILING_STOP_PERCENT,
        # ── VERROUILLÉ ──
        max_position_pct=0.30,            # cap 30%
        max_simultaneous_positions=3,     # max 3 positions
        compound=True,                    # sizing sur equity courante
        risk_percent_range=0.02,          # 2% risque
        # Range params (inchangés)
        swing_lookback=config.SWING_LOOKBACK,
        range_width_min=config.RANGE_WIDTH_MIN,
        range_entry_buffer_pct=config.RANGE_ENTRY_BUFFER_PERCENT,
        range_sl_buffer_pct=config.RANGE_SL_BUFFER_PERCENT,
        range_cooldown_bars=config.RANGE_COOLDOWN_BARS,
        max_total_risk_pct=config.MAX_TOTAL_RISK_PERCENT,
        # Stratégies
        enable_trend=False,
        enable_range=True,
    )


def run_period(
    all_candles: dict[str, list],
    balance: float,
    start: datetime,
    end: datetime,
) -> tuple[dict, list]:
    """Lance le backtest sur une période avec les candles pré-filtrées."""
    from src.core.models import Candle

    # Filtrer les candles pour la période demandée
    start_ms = int(start.timestamp() * 1000)
    end_ms = int(end.timestamp() * 1000)

    filtered: dict[str, list] = {}
    for pair, candles in all_candles.items():
        period_candles = [
            c for c in candles if start_ms <= c.timestamp <= end_ms
        ]
        if period_candles:
            filtered[pair] = period_candles

    cfg = _locked_config(balance)
    engine = BacktestEngine(filtered, cfg)
    result = engine.run()
    metrics = compute_metrics(result)
    return metrics, result.trades


def main() -> None:
    parser = argparse.ArgumentParser(description="TradeX Robustness Test — RANGE 31 paires")
    parser.add_argument("--balance", type=float, default=1000.0)
    args = parser.parse_args()

    # Dates extrêmes pour un seul téléchargement
    global_start = datetime(2020, 2, 20, tzinfo=timezone.utc)
    global_end = datetime(2026, 2, 20, tzinfo=timezone.utc)

    logger.info("🔬 ROBUSTNESS TEST — RANGE 31 paires | $%.0f", args.balance)
    logger.info("   Paramètres verrouillés : pos=3, compound=True, cap=30%%, risk=2%%")
    logger.info("   Périodes : 6yr complet + 3 sous-périodes de 2 ans\n")

    # ── Téléchargement unique (toute la période 6yr) ──
    logger.info("📥 Téléchargement des 31 paires H4 (2020-2026)…")
    all_candles = download_all_pairs(PAIRS_31, global_start, global_end, interval="4h")
    logger.info("✅ %d paires téléchargées\n", len(all_candles))

    # ── Run chaque période ─────────────────────────────────────────────────
    all_results = []

    for label, s, e in PERIODS:
        start = datetime.strptime(s, "%Y-%m-%d").replace(tzinfo=timezone.utc)
        end = datetime.strptime(e, "%Y-%m-%d").replace(tzinfo=timezone.utc)

        logger.info("⏳ %s…", label)
        metrics, trades = run_period(all_candles, args.balance, start, end)
        all_results.append((label, metrics, trades))
        logger.info("   ✅ %d trades | Return: %+.1f%%\n", metrics["n_trades"], metrics["total_return"] * 100)

    # ── Affichage comparatif ───────────────────────────────────────────────
    sep = "═" * 140
    print(f"\n{sep}")
    print("  🔬 ROBUSTNESS TEST — RANGE 31 paires | Params: pos=3, compound=True, cap=30%, risk=2%")
    print(sep)

    header = (
        f"  {'Période':<28s} │ {'Return':>8s} │ {'CAGR':>7s} │ {'MaxDD':>7s} │ "
        f"{'Sharpe':>7s} │ {'Sortino':>7s} │ {'PF':>5s} │ {'WR':>5s} │ "
        f"{'Trades':>6s} │ {'Avg PnL':>8s} │ {'Final$':>10s}"
    )
    print(header)
    print("  " + "─" * 136)

    for label, m, trades in all_results:
        pf_emoji = "✅" if m["profit_factor"] >= 1.3 else ("⚠️" if m["profit_factor"] >= 1.0 else "❌")

        print(
            f"  {label:<28s} │ {m['total_return']:>+7.1%} │ {m['cagr']:>+6.1%} │ "
            f"{m['max_drawdown']:>7.1%} │ {m['sharpe']:>7.2f} │ {m['sortino']:>7.2f} │ "
            f"{pf_emoji}{m['profit_factor']:>4.2f} │ {m['win_rate']:>4.0%} │ "
            f"{m['n_trades']:>6d} │ ${m['avg_pnl_usd']:>+7.2f} │ ${m['final_equity']:>9,.2f}"
        )

    # ── Analyse de stabilité ───────────────────────────────────────────────
    print(f"\n  {'─' * 136}")
    print("  📊 ANALYSE DE STABILITÉ")
    print(f"  {'─' * 136}")

    sub_results = all_results[1:]  # exclure le 6yr complet
    pfs = [m["profit_factor"] for _, m, _ in sub_results]
    sharpes = [m["sharpe"] for _, m, _ in sub_results]
    returns = [m["total_return"] for _, m, _ in sub_results]
    dds = [m["max_drawdown"] for _, m, _ in sub_results]

    import statistics
    pf_mean = statistics.mean(pfs)
    pf_std = statistics.stdev(pfs) if len(pfs) > 1 else 0
    sharpe_mean = statistics.mean(sharpes)
    ret_mean = statistics.mean(returns)

    print(f"\n  Sous-périodes (3 × 2 ans) :")
    print(f"    PF moyen     : {pf_mean:.2f} (σ = {pf_std:.2f})")
    print(f"    Sharpe moyen : {sharpe_mean:.2f}")
    print(f"    Return moyen : {ret_mean:+.1%} / 2 ans")
    print(f"    MaxDD range  : [{min(dds):.1%} .. {max(dds):.1%}]")

    # Verdict
    all_pf_positive = all(pf > 1.0 for pf in pfs)
    all_pf_strong = all(pf > 1.3 for pf in pfs)
    all_sharpe_ok = all(s > 0.5 for s in sharpes)
    all_dd_ok = all(dd > -0.20 for dd in dds)

    print(f"\n  🏁 VERDICT :")
    if all_pf_strong and all_sharpe_ok and all_dd_ok:
        print("     ✅✅ EDGE ROBUSTE — PF > 1.3, Sharpe > 0.5, DD < 20% dans TOUTES les périodes")
        print("     → Prêt pour production avec ces paramètres")
    elif all_pf_positive and all_dd_ok:
        print("     ✅ EDGE PRÉSENT mais variable — PF > 1.0 partout, DD contenus")
        print("     → Utilisable mais surveiller les périodes faibles")
    elif all_pf_positive:
        print("     ⚠️  EDGE FRAGILE — PF > 1.0 partout mais drawdown élevé dans certaines périodes")
        print("     → Réduire le risque ou diversifier davantage")
    else:
        losing_periods = [label for label, m, _ in sub_results if m["profit_factor"] < 1.0]
        print(f"     ❌ EDGE NON ROBUSTE — Périodes perdantes : {', '.join(losing_periods)}")
        print("     → L'edge n'est pas stable dans le temps")

    # ── Détail par paire (sur 6yr) ─────────────────────────────────────────
    _, m6yr, trades6yr = all_results[0]
    print(f"\n  {'─' * 136}")
    print("  📋 TOP / FLOP paires (6yr complet) :")

    pair_stats: dict[str, dict] = {}
    for t in trades6yr:
        sym = t.symbol
        if sym not in pair_stats:
            pair_stats[sym] = {"n": 0, "pnl": 0.0, "wins": 0}
        pair_stats[sym]["n"] += 1
        pair_stats[sym]["pnl"] += t.pnl_usd
        if t.pnl_usd > 0:
            pair_stats[sym]["wins"] += 1

    sorted_pairs = sorted(pair_stats.items(), key=lambda x: x[1]["pnl"], reverse=True)

    # Top 10
    print("\n  🟢 TOP 10 :")
    for sym, s in sorted_pairs[:10]:
        wr = s["wins"] / s["n"] * 100 if s["n"] else 0
        print(f"    {sym:<12s} : {s['n']:>3d} trades | WR {wr:>4.0f}% | PnL ${s['pnl']:>+9.2f}")

    # Flop 5
    if len(sorted_pairs) > 10:
        print("\n  🔴 FLOP 5 :")
        for sym, s in sorted_pairs[-5:]:
            wr = s["wins"] / s["n"] * 100 if s["n"] else 0
            print(f"    {sym:<12s} : {s['n']:>3d} trades | WR {wr:>4.0f}% | PnL ${s['pnl']:>+9.2f}")

    # Paires sans trades
    no_trades = set(PAIRS_31) - set(pair_stats.keys())
    if no_trades:
        print(f"\n  ⚪ Sans trades : {', '.join(sorted(no_trades))}")

    profitable_pairs = sum(1 for _, s in pair_stats.items() if s["pnl"] > 0)
    total_pairs = len(pair_stats)
    print(f"\n  📈 {profitable_pairs}/{total_pairs} paires profitables ({profitable_pairs/total_pairs*100:.0f}%)")

    print(f"\n{sep}\n")


if __name__ == "__main__":
    main()
