#!/usr/bin/env python3
"""
Walk-Forward Validation — Infinity Bot.

Train sur 2020-2024 → trouve la meilleure config par paire.
Test  sur 2024-2026 → applique cette config sur données inconnues.

Usage:
    python -m backtest.run_infinity_walkforward
    python -m backtest.run_infinity_walkforward --pairs ETH-USD,XLM-USD
"""

from __future__ import annotations

import argparse
import logging
from datetime import datetime, timezone

from backtest.data_loader import download_candles
from backtest.run_backtest_infinity import run_infinity_backtest, InfinityTrade
from backtest.run_infinity_scanner import (
    BUY_LEVEL_PRESETS,
    SELL_LEVEL_PRESETS,
    STOP_LOSS_OPTIONS,
    TRAILING_HIGH_OPTIONS,
    compute_metrics,
    score_config,
    run_scan,
)
from src.core.infinity_engine import InfinityConfig

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(message)s")
logger = logging.getLogger(__name__)

PAIRS_DEFAULT = "ETH-USD,SOL-USD,XLM-USD,AAVE-USD,ADA-USD"

# Périodes
TRAIN_START = datetime(2020, 1, 1, tzinfo=timezone.utc)
TRAIN_END   = datetime(2024, 1, 1, tzinfo=timezone.utc)
TEST_START  = datetime(2024, 1, 1, tzinfo=timezone.utc)
TEST_END    = datetime(2026, 3, 6, tzinfo=timezone.utc)


def build_config_from_result(r: dict, balance: float) -> InfinityConfig:
    """Reconstruit un InfinityConfig depuis un résultat du scanner."""
    buy_levels = BUY_LEVEL_PRESETS[r["buy_preset"]]
    sell_levels = SELL_LEVEL_PRESETS[r["sell_preset"]]
    return InfinityConfig(
        initial_balance=balance,
        buy_levels=buy_levels,
        entry_drop_pct=abs(buy_levels[0]),
        max_buy_levels=len(buy_levels),
        buy_pcts=(0.25, 0.20, 0.15, 0.10, 0.00)[:len(buy_levels)],
        buy_amounts=(100.0, 200.0, 300.0, 400.0, 0.0)[:len(buy_levels)],
        sell_levels=sell_levels,
        stop_loss_pct=r["stop_loss"],
        trailing_high_period=r["trail_period"],
    )


def main():
    parser = argparse.ArgumentParser(description="Walk-Forward Validation Infinity Bot")
    parser.add_argument("--pairs", default=PAIRS_DEFAULT, help="Paires séparées par des virgules")
    parser.add_argument("--balance", type=float, default=1000, help="Balance initiale ($)")
    parser.add_argument("--capital-pct", type=float, default=0.65, help="%% capital alloué")
    args = parser.parse_args()

    pairs = [p.strip() for p in args.pairs.split(",")]

    print()
    print("=" * 130)
    print("♾️  WALK-FORWARD VALIDATION — INFINITY BOT")
    print(f"   TRAIN: {TRAIN_START.date()} → {TRAIN_END.date()} (4 ans)")
    print(f"   TEST:  {TEST_START.date()} → {TEST_END.date()} (2.2 ans)")
    print("=" * 130)

    summary = []

    for pair in pairs:
        logger.info("\n" + "=" * 90)
        logger.info("♾️  %s", pair)
        logger.info("=" * 90)

        # ── 1) Charger toutes les données (6 ans) ──
        try:
            all_candles = download_candles(pair, TRAIN_START, TEST_END, interval="4h")
            logger.info("   %d bougies H4 totales", len(all_candles))
        except Exception as e:
            logger.error("❌ %s: %s", pair, e)
            summary.append({"pair": pair, "error": str(e)})
            continue

        # Séparer train / test
        train_candles = [c for c in all_candles if c.timestamp < int(TRAIN_END.timestamp() * 1000)]
        test_candles  = [c for c in all_candles if c.timestamp >= int(TEST_START.timestamp() * 1000)]

        logger.info("   Train: %d bougies | Test: %d bougies", len(train_candles), len(test_candles))

        if len(train_candles) < 500 or len(test_candles) < 200:
            logger.warning("⚠️ %s — pas assez de données", pair)
            summary.append({"pair": pair, "error": "insufficient data"})
            continue

        # ── 2) TRAIN: scanner sur 2020-2024 ──
        logger.info("📚 TRAIN phase — scanner 336 configs sur 2020→2024...")
        train_results = run_scan(pair, train_candles, args.balance, args.capital_pct)

        if not train_results or train_results[0]["score"] < 0:
            logger.warning("⚠️ %s — aucune config rentable en TRAIN", pair)
            summary.append({"pair": pair, "error": "no profitable config"})
            continue

        best_train = train_results[0]
        logger.info("🏆 TRAIN best: Score=%.1f | Ret=%.2f%% | PF=%.2f | DD=%.2f%% | %s + %s | SL=%.0f%% | Trail=%d",
                     best_train["score"], best_train["ret"], best_train["pf"], best_train["dd"],
                     best_train["buy_preset"], best_train["sell_preset"],
                     best_train["stop_loss"] * 100, best_train["trail_period"])

        # ── 3) TEST: appliquer la meilleure config sur 2024-2026 ──
        logger.info("🧪 TEST phase — appliquer la config TRAIN sur 2024→2026...")
        config = build_config_from_result(best_train, args.balance)

        try:
            test_trades, test_equity = run_infinity_backtest(
                candles=test_candles,
                config=config,
                initial_balance=args.balance,
                capital_pct=args.capital_pct,
            )
        except Exception as e:
            logger.error("❌ TEST %s: %s", pair, e)
            summary.append({"pair": pair, "error": str(e)})
            continue

        test_metrics = compute_metrics(test_trades, test_equity, args.balance)

        # ── 4) Vérifier robustesse: top 3 TRAIN sur TEST ──
        logger.info("🔬 Robustesse — test des configs rang 2 & 3 sur TEST...")
        robustness_results = []
        for rank, tr in enumerate(train_results[:3], 1):
            cfg = build_config_from_result(tr, args.balance)
            try:
                trades_r, equity_r = run_infinity_backtest(
                    candles=test_candles, config=cfg,
                    initial_balance=args.balance, capital_pct=args.capital_pct,
                )
                m_r = compute_metrics(trades_r, equity_r, args.balance)
                robustness_results.append((rank, tr, m_r))
            except Exception:
                continue

        # ── 5) Affichage ──
        print(f"\n{'─' * 100}")
        print(f"♾️  {pair} — WALK-FORWARD RESULTS")
        print(f"{'─' * 100}")

        print(f"\n  🏆 Config sélectionnée (meilleure TRAIN):")
        print(f"     Buy: {best_train['buy_preset']}")
        print(f"     Sell: {best_train['sell_preset']}")
        print(f"     SL: {best_train['stop_loss']*100:.0f}%  |  Trail: {best_train['trail_period']} bars")

        header = f"{'Phase':<8} | {'Ret%':>8} | {'PF':>6} | {'Sharpe':>7} | {'DD%':>7} | {'WR%':>6} | {'Trd':>4} | {'SL':>3}"
        print(f"\n  {header}")
        print(f"  {'-' * 70}")

        train_line = (f"  {'TRAIN':<8} | {best_train['ret']:>7.2f}% | {best_train['pf']:>6.2f} | "
                      f"{best_train['sharpe']:>7.2f} | {best_train['dd']:>6.2f}% | "
                      f"{best_train['wr']:>5.1f}% | {best_train['trades']:>4} | {best_train['sl_count']:>3}")
        print(train_line)

        test_line = (f"  {'TEST':<8} | {test_metrics['ret']:>7.2f}% | {test_metrics['pf']:>6.2f} | "
                     f"{test_metrics['sharpe']:>7.2f} | {test_metrics['dd']:>6.2f}% | "
                     f"{test_metrics['wr']:>5.1f}% | {test_metrics['trades']:>4} | {test_metrics['sl_count']:>3}")
        print(test_line)

        # Verdict
        test_ok = test_metrics["pf"] > 1.0 and test_metrics["ret"] > 0
        ratio = test_metrics["ret"] / best_train["ret"] * 100 if best_train["ret"] > 0 else 0
        emoji = "✅" if test_ok else "❌"
        print(f"\n  {emoji} TEST retient {ratio:.0f}% du rendement TRAIN")

        if test_ok and test_metrics["pf"] >= 1.5:
            verdict = "EXCELLENT — candidat solide pour le live"
        elif test_ok:
            verdict = "OK — profitable mais à surveiller"
        else:
            verdict = "ÉCHEC — overfitting probable, ne pas déployer"
        print(f"  📋 Verdict: {verdict}")

        # Robustesse
        if robustness_results:
            print(f"\n  🔬 Robustesse (top 3 TRAIN → TEST):")
            for rank, tr, mr in robustness_results:
                rob_emoji = "🟢" if mr["ret"] > 0 else "🔴"
                print(f"     {rob_emoji} Rang {rank}: TRAIN {tr['ret']:>6.2f}% → TEST {mr['ret']:>6.2f}%  "
                      f"(PF {mr['pf']:.2f}, DD {mr['dd']:.2f}%)")

        # Save for summary
        summary.append({
            "pair": pair,
            "train": best_train,
            "test": test_metrics,
            "verdict": verdict,
            "test_ok": test_ok,
            "ratio": ratio,
            "robustness": robustness_results,
        })

    # ── Résumé final ──
    print(f"\n\n{'=' * 120}")
    print(f"♾️  RÉSUMÉ WALK-FORWARD — TOUTES PAIRES")
    print(f"{'=' * 120}")

    header = (f"{'Paire':<10} | {'TRAIN Ret%':>10} | {'TEST Ret%':>10} | {'TEST PF':>8} | "
              f"{'TEST DD%':>8} | {'TEST WR%':>8} | {'Ratio':>6} | {'Verdict':<45}")
    print(header)
    print("-" * 120)

    for s in summary:
        if "error" in s:
            print(f"{'🔴':} {s['pair']:<10} | {'ERREUR: ' + s['error']}")
            continue

        emoji = "✅" if s["test_ok"] else "❌"
        line = (f"{emoji} {s['pair']:<10} | {s['train']['ret']:>9.2f}% | {s['test']['ret']:>9.2f}% | "
                f"{s['test']['pf']:>8.2f} | {s['test']['dd']:>7.2f}% | {s['test']['wr']:>7.1f}% | "
                f"{s['ratio']:>5.0f}% | {s['verdict']:<45}")
        print(line)

    print(f"{'=' * 120}")

    # Recommandation finale
    live_candidates = [s for s in summary if s.get("test_ok") and s["test"]["pf"] >= 1.5]
    if live_candidates:
        print(f"\n🚀 CANDIDATS LIVE ({len(live_candidates)}):")
        for s in live_candidates:
            print(f"   ✅ {s['pair']} — TEST: +{s['test']['ret']:.2f}%, PF {s['test']['pf']:.2f}")
            print(f"      Config: {s['train']['buy_preset']} + {s['train']['sell_preset']} "
                  f"| SL {s['train']['stop_loss']*100:.0f}% | Trail {s['train']['trail_period']}")
    else:
        print(f"\n⚠️ Aucun candidat solide pour le live (PF TEST < 1.5)")


if __name__ == "__main__":
    main()
