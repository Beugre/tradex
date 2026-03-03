#!/usr/bin/env python
"""
Backtest VWAP Mean Reversion Intraday — standalone + combiné 3 stratégies.

Usage :
    # VWAP seul, 6 ans
    PYTHONPATH=. python backtest/run_backtest_vwap.py --months 72 --no-show

    # Analyse de sensibilité
    PYTHONPATH=. python backtest/run_backtest_vwap.py --months 72 --sensitivity --no-show

    # Combiné 3 stratégies (VWAP + CrashBot + Trail Range) sur 6 ans
    PYTHONPATH=. python backtest/run_backtest_vwap.py --months 72 --combined --no-show
"""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

from backtest.data_loader import download_candles
from backtest.simulator_vwap import (
    VwapConfig,
    VwapEngine,
    VwapResult,
    VwapTrade,
    compute_vwap_metrics,
)

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("vwap_bt")

# ── Paires — top liquidité, bonnes pour VWAP reversion ────────────────────────

VWAP_PAIRS = [
    "BTC-USD", "ETH-USD", "SOL-USD", "XRP-USD", "BNB-USD",
    "DOGE-USD", "ADA-USD", "AVAX-USD", "DOT-USD", "LINK-USD",
    "SUI-USD", "NEAR-USD", "LTC-USD", "ARB-USD", "OP-USD",
    "PEPE-USD", "FET-USD", "RENDER-USD", "INJ-USD", "AAVE-USD",
]

OUTPUT_DIR = Path(__file__).parent / "output"


def main() -> None:
    parser = argparse.ArgumentParser(description="Backtest VWAP Mean Reversion Intraday")
    parser.add_argument("--balance", type=float, default=1000.0)
    parser.add_argument("--months", type=int, default=12)
    parser.add_argument("--start", type=str, default=None)
    parser.add_argument("--end", type=str, default=None)
    parser.add_argument("--interval", type=str, default="1h", choices=["1h", "30m"])
    parser.add_argument("--risk", type=float, default=0.02)
    parser.add_argument("--band-mult", type=float, default=1.5, help="VWAP band mult (σ)")
    parser.add_argument("--rsi-floor", type=float, default=25.0)
    parser.add_argument("--rsi-ceil", type=float, default=42.0)
    parser.add_argument("--tp-mode", type=str, default="vwap", choices=["vwap", "fixed"])
    parser.add_argument("--tp-pct", type=float, default=0.015)
    parser.add_argument("--sl-pct", type=float, default=0.012)
    parser.add_argument("--atr-sl", type=float, default=0.0, help="ATR mult for SL (0=fixed %%)")
    parser.add_argument("--timeout", type=int, default=24, help="Timeout bars")
    parser.add_argument("--cooldown", type=int, default=4, help="Cooldown bars")
    parser.add_argument("--sensitivity", action="store_true", help="Analyse de sensibilité")
    parser.add_argument("--combined", action="store_true", help="Backtest combiné 3 strats")
    parser.add_argument("--no-show", action="store_true")
    args = parser.parse_args()

    # Dates
    if args.start and args.end:
        start = datetime.strptime(args.start, "%Y-%m-%d").replace(tzinfo=timezone.utc)
        end = datetime.strptime(args.end, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    else:
        end = datetime.now(tz=timezone.utc)
        start = end - timedelta(days=args.months * 30)

    logger.info("🔵 VWAP Backtest — %s → %s (%s)", start.date(), end.date(), args.interval)

    # Téléchargement
    logger.info("📥 Téléchargement klines %s…", args.interval)
    candles_by_symbol = {}
    for pair in VWAP_PAIRS:
        candles = download_candles(pair, start, end, interval=args.interval)
        if candles:
            candles_by_symbol[pair] = candles
            logger.info("   ✅ %s : %d bougies", pair, len(candles))
        else:
            logger.warning("   ⚠️ %s : aucune donnée", pair)

    if not candles_by_symbol:
        logger.error("❌ Aucune donnée — abandon")
        sys.exit(1)

    base_cfg = VwapConfig(
        initial_balance=args.balance,
        risk_percent=args.risk,
        vwap_band_mult=args.band_mult,
        rsi_floor=args.rsi_floor,
        rsi_ceil=args.rsi_ceil,
        tp_mode=args.tp_mode,
        tp_pct=args.tp_pct,
        sl_pct=args.sl_pct,
        atr_sl_mult=args.atr_sl,
        timeout_bars=args.timeout,
        cooldown_bars=args.cooldown,
    )

    if args.sensitivity:
        _run_sensitivity(candles_by_symbol, base_cfg, args)
    elif args.combined:
        _run_combined(start, end, args)
    else:
        _run_single(candles_by_symbol, base_cfg, args)


# ── Run unique ─────────────────────────────────────────────────────────────────


def _run_single(candles: dict, cfg: VwapConfig, args) -> None:
    engine = VwapEngine(candles, cfg, interval=args.interval)
    result = engine.run()
    metrics = compute_vwap_metrics(result)
    _print_report(result, metrics)
    _generate_chart(result, metrics, show=not args.no_show)
    _save_trades_csv(result)


# ── Sensibilité ────────────────────────────────────────────────────────────────


def _run_sensitivity(candles: dict, base_cfg: VwapConfig, args) -> None:
    band_mults = [1.0, 1.25, 1.5, 2.0, 2.5]
    rsi_ceilings = [35, 40, 45]
    sl_pcts = [0.010, 0.012, 0.015, 0.020]
    tp_modes = [("vwap", 0), ("fixed", 0.015), ("fixed", 0.020)]
    timeout_bars = [12, 24, 48]

    results = []
    total = len(band_mults) * len(rsi_ceilings) * len(sl_pcts) * len(tp_modes) * len(timeout_bars)
    logger.info("🔬 Sensibilité : %d combinaisons", total)

    done = 0
    for band in band_mults:
        for rsi_c in rsi_ceilings:
            for sl in sl_pcts:
                for tp_m, tp_p in tp_modes:
                    for timeout in timeout_bars:
                        cfg = VwapConfig(
                            initial_balance=base_cfg.initial_balance,
                            risk_percent=base_cfg.risk_percent,
                            vwap_band_mult=band,
                            rsi_floor=base_cfg.rsi_floor,
                            rsi_ceil=rsi_c,
                            tp_mode=tp_m,
                            tp_pct=tp_p,
                            sl_pct=sl,
                            timeout_bars=timeout,
                            cooldown_bars=base_cfg.cooldown_bars,
                        )
                        engine = VwapEngine(candles, cfg, interval=args.interval)
                        result = engine.run()
                        metrics = compute_vwap_metrics(result)
                        results.append((metrics, cfg))
                        done += 1
                        if done % 50 == 0:
                            logger.info("   %d/%d done…", done, total)

    results.sort(key=lambda x: x[0]["total_return"], reverse=True)

    sep = "═" * 140
    print(f"\n{sep}")
    print("  🔬 SENSIBILITÉ — VWAP Mean Reversion")
    print(sep)
    print(
        f"  {'Band':>5s} | {'RSI≤':>5s} | {'SL%':>5s} | {'TP':>6s} | {'Tout':>4s} | "
        f"{'Trades':>6s} | {'WR':>6s} | {'PF':>6s} | "
        f"{'Return':>8s} | {'CAGR':>7s} | {'MaxDD':>7s} | {'Sharpe':>7s} | "
        f"{'AvgPnL':>8s} | {'AvgHold':>7s}"
    )
    print("  " + "─" * 136)

    for m, cfg in results[:50]:
        if m["n_trades"] == 0:
            continue
        tp_str = "VWAP" if cfg.tp_mode == "vwap" else f"{cfg.tp_pct:.1%}"
        print(
            f"  {cfg.vwap_band_mult:5.2f} | {cfg.rsi_ceil:5.0f} | {cfg.sl_pct:5.1%} | "
            f"{tp_str:>6s} | {cfg.timeout_bars:4d} | "
            f"{m['n_trades']:6d} | {m['win_rate']:6.1%} | {m['profit_factor']:6.2f} | "
            f"{m['total_return']:+7.1%} | {m['cagr']:+6.1%} | {m['max_drawdown']:7.1%} | "
            f"{m['sharpe']:7.2f} | "
            f"${m['avg_pnl_usd']:+7.2f} | {m['avg_hold_bars']:6.1f}b"
        )

    print(f"\n{sep}\n")

    if results and results[0][0]["n_trades"] > 0:
        best_m, best_cfg = results[0]
        tp_str = "VWAP" if best_cfg.tp_mode == "vwap" else f"{best_cfg.tp_pct:.1%}"
        print(
            f"  🏆 Meilleure : band={best_cfg.vwap_band_mult:.2f}σ, "
            f"RSI≤{best_cfg.rsi_ceil:.0f}, SL={best_cfg.sl_pct:.1%}, "
            f"TP={tp_str}, timeout={best_cfg.timeout_bars}b"
        )
        print(
            f"     Return={best_m['total_return']:+.1%}, WR={best_m['win_rate']:.0%}, "
            f"PF={best_m['profit_factor']:.2f}, CAGR={best_m['cagr']:+.1%}, "
            f"MaxDD={best_m['max_drawdown']:.1%}, Sharpe={best_m['sharpe']:.2f}\n"
        )


# ── Combiné 3 stratégies ──────────────────────────────────────────────────────


def _run_combined(start: datetime, end: datetime, args) -> None:
    """
    Simule 3 stratégies sur le même capital avec allocation dynamique.

    Trail Range  : simulator existant (H4)
    CrashBot     : simulator_antiliq (1m)
    VWAP Revert  : simulator_vwap (H1)

    Allocation PF-based (comme en production).
    """
    from backtest.simulator_antiliq import AntiliqConfig, AntiliqEngine, compute_antiliq_metrics
    from backtest.simulator_vwap import VwapConfig, VwapEngine, compute_vwap_metrics

    total_balance = args.balance
    logger.info("🔵🔥🟡 Backtest COMBINÉ 3 stratégies — $%.0f — %s → %s", total_balance, start.date(), end.date())

    # ── 1. VWAP Reversion (H1) ─────────────────────────────────────────────
    logger.info("\n" + "═" * 60)
    logger.info("  🔵 Stratégie 1 : VWAP Mean Reversion (H1)")
    logger.info("═" * 60)

    vwap_candles = {}
    for pair in VWAP_PAIRS:
        c = download_candles(pair, start, end, interval="1h")
        if c:
            vwap_candles[pair] = c

    vwap_cfg = VwapConfig(initial_balance=total_balance)
    vwap_engine = VwapEngine(vwap_candles, vwap_cfg, interval="1h")
    vwap_result = vwap_engine.run()
    vwap_metrics = compute_vwap_metrics(vwap_result)

    # ── 2. CrashBot / Antiliq (1m) ────────────────────────────────────────
    logger.info("\n" + "═" * 60)
    logger.info("  🔥 Stratégie 2 : CrashBot / Antiliq (1m)")
    logger.info("═" * 60)

    from backtest.run_backtest_antiliq import ANTILIQ_PAIRS
    crash_candles = {}
    for pair in ANTILIQ_PAIRS:
        c = download_candles(pair, start, end, interval="1m")
        if c:
            crash_candles[pair] = c

    crash_cfg = AntiliqConfig(
        initial_balance=total_balance,
        move_threshold_pct=0.04,
        volume_multiplier=0,
        tp_retrace_pct=0.3,
        trailing_sl=True,
        trailing_activation_pct=0.5,
        trailing_step_pct=0.3,
    )
    crash_engine = AntiliqEngine(crash_candles, crash_cfg)
    crash_result = crash_engine.run()
    crash_metrics = compute_antiliq_metrics(crash_result)

    # ── 3. Trail Range (H4) ───────────────────────────────────────────────
    logger.info("\n" + "═" * 60)
    logger.info("  🟡 Stratégie 3 : Trail Range (H4)")
    logger.info("═" * 60)

    trail_result, trail_metrics = _run_trail_range(start, end, total_balance)

    # ── 4. Rapport combiné ─────────────────────────────────────────────────
    _print_combined_report(
        vwap_result, vwap_metrics,
        crash_result, crash_metrics,
        trail_result, trail_metrics,
        total_balance,
    )
    _generate_combined_chart(
        vwap_result, vwap_metrics,
        crash_result, crash_metrics,
        trail_result, trail_metrics,
        total_balance,
        show=not args.no_show,
    )


def _run_trail_range(start: datetime, end: datetime, balance: float):
    """Charge et lance le backtest Trail Range H4 avec le simulateur existant."""
    try:
        from backtest.simulator import BacktestEngine, BacktestConfig
        from backtest.metrics import compute_metrics
        from backtest.data_loader import download_all_pairs

        trail_pairs = [
            "BTC-USD", "ETH-USD", "SOL-USD", "XRP-USD", "BNB-USD",
            "DOGE-USD", "ADA-USD", "AVAX-USD", "DOT-USD", "LINK-USD",
            "SUI-USD", "NEAR-USD", "LTC-USD", "ARB-USD", "OP-USD",
            "PEPE-USD", "FET-USD", "RENDER-USD", "INJ-USD", "AAVE-USD",
        ]

        candles = download_all_pairs(trail_pairs, start, end, interval="4h")

        cfg = BacktestConfig(initial_balance=balance)
        engine = BacktestEngine(candles, cfg)
        result = engine.run()
        metrics = compute_metrics(result)
        return result, metrics

    except Exception as e:
        logger.warning("⚠️ Trail Range non disponible : %s", e)
        return None, None


def _print_combined_report(
    vwap_result, vwap_m,
    crash_result, crash_m,
    trail_result, trail_m,
    total_balance: float,
) -> None:
    sep = "═" * 80

    print(f"\n{sep}")
    print("  🏦 BACKTEST COMBINÉ — 3 STRATÉGIES")
    print(f"  Capital initial : ${total_balance:,.0f}")
    if vwap_result:
        print(f"  Période : {vwap_result.start_date:%b %Y} → {vwap_result.end_date:%b %Y}")
    print(sep)

    strats = []
    if vwap_m:
        strats.append(("🔵 VWAP Reversion", vwap_m))
    if crash_m:
        strats.append(("🔥 CrashBot", crash_m))
    if trail_m:
        strats.append(("🟡 Trail Range", trail_m))

    print(f"\n  {'Stratégie':<22s} | {'Trades':>6s} | {'WR':>6s} | {'PF':>6s} | "
          f"{'Return':>8s} | {'CAGR':>7s} | {'MaxDD':>7s} | {'Sharpe':>7s} | {'Final $':>10s}")
    print("  " + "─" * 96)

    combined_equity = 0.0
    combined_trades = 0
    all_pnls = []

    for name, m in strats:
        final = m.get("final_equity", total_balance)
        n = m.get("n_trades", 0)
        combined_equity += final
        combined_trades += n

        # Collect PnLs for combined PF
        if name == "🔵 VWAP Reversion" and vwap_result:
            all_pnls.extend([(t.pnl_usd, name) for t in vwap_result.trades])
        elif name == "🔥 CrashBot" and crash_result:
            all_pnls.extend([(t.pnl_usd, name) for t in crash_result.trades])

        print(
            f"  {name:<22s} | {n:6d} | {m.get('win_rate',0):6.1%} | "
            f"{m.get('profit_factor',0):6.2f} | {m.get('total_return',0):+7.1%} | "
            f"{m.get('cagr',0):+6.1%} | {m.get('max_drawdown',0):7.1%} | "
            f"{m.get('sharpe',0):7.2f} | ${final:>9,.2f}"
        )

    # Combined weighted return (equal capital per strat)
    n_strats = len(strats)
    avg_equity = combined_equity / n_strats if n_strats > 0 else total_balance
    combined_return = (avg_equity - total_balance) / total_balance

    # Combined PF
    all_wins = sum(p for p, _ in all_pnls if p > 0) or 0
    all_losses = abs(sum(p for p, _ in all_pnls if p <= 0)) or 1e-9
    combined_pf = all_wins / all_losses

    print("  " + "─" * 96)
    print(f"  {'COMBINÉ (avg)':<22s} | {combined_trades:6d} | {'':>6s} | "
          f"{combined_pf:6.2f} | {combined_return:+7.1%} | "
          f"{'':>7s} | {'':>7s} | {'':>7s} | ${avg_equity:>9,.2f}")

    # Corrélation des rendements
    print(f"\n  📊 Analyse de corrélation")
    print("  " + "─" * 76)
    if vwap_result and crash_result:
        _print_correlation(vwap_result, crash_result, "VWAP", "CrashBot")
    if vwap_result and trail_result:
        _print_correlation_generic(
            vwap_result.equity_curve, _get_trail_eq(trail_result),
            "VWAP", "Trail Range"
        )
    if crash_result and trail_result:
        _print_correlation_generic(
            crash_result.equity_curve, _get_trail_eq(trail_result),
            "CrashBot", "Trail Range"
        )

    # Allocation suggestions
    vwap_pf = vwap_m.get("profit_factor", 0) if vwap_m else 0
    trail_pf = trail_m.get("profit_factor", 0) if trail_m else 0
    crash_pf = crash_m.get("profit_factor", 0) if crash_m else 0

    print(f"\n  🏦 Suggestion d'allocation")
    print("  " + "─" * 76)
    print(f"  PF VWAP={vwap_pf:.2f} | PF Crash={crash_pf:.2f} | PF Trail={trail_pf:.2f}")

    if trail_pf < 0.9:
        alloc = {"Trail": 10, "VWAP": 25, "Crash": 65}
    elif trail_pf <= 1.1:
        alloc = {"Trail": 20, "VWAP": 20, "Crash": 60}
    else:
        alloc = {"Trail": 35, "VWAP": 15, "Crash": 50}

    for name, pct in alloc.items():
        print(f"  → {name:>8s} : {pct:3d}% = ${total_balance * pct / 100:,.0f}")

    print(f"\n{sep}\n")


def _get_trail_eq(trail_result) -> list[tuple[int, float]]:
    """Extrait equity_curve du BacktestResult (format différent)."""
    if trail_result is None:
        return []
    eq = trail_result.equity_curve
    if not eq:
        return []
    # BacktestResult.equity_curve = list[EquityPoint(timestamp, equity)]
    return [(pt.timestamp, pt.equity) for pt in eq]


def _print_correlation(result_a, result_b, name_a: str, name_b: str) -> None:
    eq_a = result_a.equity_curve
    eq_b = result_b.equity_curve
    _print_correlation_generic(eq_a, eq_b, name_a, name_b)


def _print_correlation_generic(eq_a, eq_b, name_a: str, name_b: str) -> None:
    """Calcule et affiche la corrélation des rendements journaliers."""
    if len(eq_a) < 10 or len(eq_b) < 10:
        print(f"  {name_a} ↔ {name_b} : données insuffisantes")
        return

    # Rendements
    def daily_returns(eq):
        rets = []
        for i in range(1, len(eq)):
            prev = eq[i - 1][1]
            if prev > 0:
                rets.append((eq[i][0], (eq[i][1] - prev) / prev))
        return rets

    rets_a = daily_returns(eq_a)
    rets_b = daily_returns(eq_b)

    if not rets_a or not rets_b:
        print(f"  {name_a} ↔ {name_b} : pas de rendements")
        return

    # Aligner par timestamp approximatif (bucket journalier)
    from collections import defaultdict
    bucket_a: dict[int, list[float]] = defaultdict(list)
    bucket_b: dict[int, list[float]] = defaultdict(list)
    for ts, r in rets_a:
        day = ts // (86400 * 1000)
        bucket_a[day].append(r)
    for ts, r in rets_b:
        day = ts // (86400 * 1000)
        bucket_b[day].append(r)

    common = sorted(set(bucket_a.keys()) & set(bucket_b.keys()))
    if len(common) < 5:
        print(f"  {name_a} ↔ {name_b} : pas assez de jours communs ({len(common)})")
        return

    vals_a = [sum(bucket_a[d]) / len(bucket_a[d]) for d in common]
    vals_b = [sum(bucket_b[d]) / len(bucket_b[d]) for d in common]

    n = len(vals_a)
    mean_a = sum(vals_a) / n
    mean_b = sum(vals_b) / n
    cov = sum((vals_a[i] - mean_a) * (vals_b[i] - mean_b) for i in range(n)) / n
    std_a = (sum((v - mean_a) ** 2 for v in vals_a) / n) ** 0.5
    std_b = (sum((v - mean_b) ** 2 for v in vals_b) / n) ** 0.5

    if std_a > 0 and std_b > 0:
        corr = cov / (std_a * std_b)
    else:
        corr = 0.0

    emoji = "✅" if abs(corr) < 0.3 else ("⚠️" if abs(corr) < 0.6 else "❌")
    label = "décorrélé" if abs(corr) < 0.3 else ("faible" if abs(corr) < 0.6 else "corrélé")
    print(f"  {emoji} {name_a} ↔ {name_b} : ρ = {corr:+.3f} ({label}, {len(common)} jours)")


# ── Rapport ────────────────────────────────────────────────────────────────────


def _print_report(result: VwapResult, m: dict) -> None:
    cfg = result.config
    sep = "═" * 65

    print(f"\n{sep}")
    print(f"  🔵 VWAP Mean Reversion — {result.start_date:%b %Y} → {result.end_date:%b %Y}")
    print(f"  Paires : {len(result.pairs)} | Interval : H1")
    print(f"  Capital : ${result.initial_balance:,.0f} | Risque : {cfg.risk_percent:.0%}/trade")
    print(sep)

    print("\n  📈 Résultats globaux")
    print("  " + "─" * 61)
    print(f"  Capital final      : ${m['final_equity']:,.2f} ({m['total_return']:+.1%})")
    print(f"  CAGR               : {m['cagr']:+.1%}")
    print(f"  Max Drawdown       : {m['max_drawdown']:.1%}")
    print(f"  Sharpe Ratio       : {m['sharpe']:.2f}")
    print(f"  Sortino Ratio      : {m['sortino']:.2f}")

    print(f"\n  🎯 Trades")
    print("  " + "─" * 61)
    print(f"  Signaux détectés   : {m['n_signals']}")
    print(f"  Trades exécutés    : {m['n_trades']}")
    n = m["n_trades"]
    if n > 0:
        print(f"  Win Rate           : {m['win_rate']:.1%} ({int(m['win_rate'] * n)}/{n})")
        print(f"  Profit Factor      : {m['profit_factor']:.2f}")
        print(f"  PnL moyen          : ${m['avg_pnl_usd']:+.2f} ({m['avg_pnl_pct']:+.2%})")
        print(f"  Durée moy. trade   : {m['avg_hold_bars']:.1f} barres")

    if m["best_trade"]:
        b = m["best_trade"]
        print(f"  Meilleur trade     : ${b.pnl_usd:+.2f} ({b.pnl_pct:+.1%}) {b.symbol} [dev {b.deviation_pct:.1%}]")
    if m["worst_trade"]:
        w = m["worst_trade"]
        print(f"  Pire trade         : ${w.pnl_usd:+.2f} ({w.pnl_pct:+.1%}) {w.symbol} [dev {w.deviation_pct:.1%}]")

    if m["by_pair"]:
        print(f"\n  📊 Par paire")
        print("  " + "─" * 61)
        for pair, s in sorted(m["by_pair"].items(), key=lambda x: x[1]["pnl"], reverse=True):
            print(
                f"  {pair:12s} : {s['n']:3d} trades | WR {s['wr']:.0%}"
                f" | PF {s['pf']:.2f} | PnL ${s['pnl']:+.2f} | Hold {s['avg_hold']:.0f}b"
            )

    if m["by_exit"]:
        print(f"\n  📊 Par sortie")
        print("  " + "─" * 61)
        for reason, s in m["by_exit"].items():
            pct = s["n"] / n * 100 if n > 0 else 0
            print(
                f"  {reason:10s} : {s['n']:3d} trades ({pct:4.1f}%)"
                f" | WR {s['wr']:.0%} | PnL ${s['pnl']:+.2f}"
            )

    if m.get("by_deviation"):
        print(f"\n  📊 Par déviation VWAP")
        print("  " + "─" * 61)
        for bucket, s in m["by_deviation"].items():
            print(
                f"  Dev {bucket:6s} : {s['n']:3d} trades | WR {s['wr']:.0%}"
                f" | PF {s['pf']:.2f} | PnL ${s['pnl']:+.2f}"
            )

    print(f"\n  ⚙️  Configuration")
    print("  " + "─" * 61)
    print(f"  VWAP band          : {cfg.vwap_band_mult:.2f}σ")
    print(f"  RSI range          : {cfg.rsi_floor:.0f} – {cfg.rsi_ceil:.0f}")
    tp_str = "VWAP" if cfg.tp_mode == "vwap" else f"{cfg.tp_pct:.1%}"
    print(f"  TP                 : {tp_str}")
    sl_str = f"ATR×{cfg.atr_sl_mult:.1f}" if cfg.atr_sl_mult > 0 else f"{cfg.sl_pct:.1%}"
    print(f"  SL                 : {sl_str}")
    print(f"  Timeout            : {cfg.timeout_bars} barres")
    print(f"  Cooldown           : {cfg.cooldown_bars} barres")
    print(f"  Fee                : {cfg.fee_pct:.2%} | Slippage : {cfg.slippage_pct:.2%}")

    print(f"\n{sep}\n")


# ── Graphiques ─────────────────────────────────────────────────────────────────


def _generate_chart(result: VwapResult, metrics: dict, show: bool = True) -> Path:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mticker

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    plt.style.use("seaborn-v0_8-whitegrid")
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(3, 2, height_ratios=[3, 1, 1], hspace=0.35, wspace=0.3)

    eq = result.equity_curve
    dates = [datetime.fromtimestamp(ts / 1000, tz=timezone.utc) for ts, _ in eq]
    equities = [e for _, e in eq]
    trades = result.trades

    # Equity curve
    ax1 = fig.add_subplot(gs[0, :])
    ax1.fill_between(dates, equities, alpha=0.15, color="#2196F3")
    ax1.plot(dates, equities, color="#1565C0", linewidth=1.2, label="Equity")
    ax1.axhline(y=result.initial_balance, color="gray", linestyle="--", alpha=0.5)
    ax1.set_title(
        f"🔵 VWAP Mean Reversion — {result.start_date:%b %Y} → {result.end_date:%b %Y}  |  "
        f"${result.initial_balance:,.0f} → ${result.final_equity:,.2f} "
        f"({metrics['total_return']:+.1%})",
        fontsize=13, fontweight="bold", pad=15,
    )
    ax1.set_ylabel("Equity ($)")
    ax1.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x:,.0f}"))
    ax1.legend(loc="upper left")

    # Drawdown
    ax2 = fig.add_subplot(gs[1, 0])
    dd = metrics["dd_curve"]
    if dd and dates:
        ax2.fill_between(dates[:len(dd)], dd, alpha=0.3, color="#F44336")
        ax2.plot(dates[:len(dd)], dd, color="#D32F2F", linewidth=0.8)
    ax2.set_ylabel("Drawdown")
    ax2.set_title("Drawdown", fontsize=10)
    ax2.yaxis.set_major_formatter(mticker.PercentFormatter(1.0, decimals=0))

    # Trade distribution
    ax3 = fig.add_subplot(gs[1, 1])
    if trades:
        pnls = [t.pnl_pct * 100 for t in trades]
        colors = ["#4CAF50" if p >= 0 else "#F44336" for p in pnls]
        ax3.bar(range(len(pnls)), pnls, color=colors, alpha=0.7, width=0.9)
        ax3.axhline(y=0, color="gray", linewidth=0.5)
    ax3.set_ylabel("P&L %")
    ax3.set_title("Trades (chronologique)", fontsize=10)

    # VWAP deviation vs PnL
    ax4 = fig.add_subplot(gs[2, 0])
    if trades:
        devs = [abs(t.deviation_pct) * 100 for t in trades]
        pnl_pcts = [t.pnl_pct * 100 for t in trades]
        colors_s = ["#4CAF50" if p >= 0 else "#F44336" for p in pnl_pcts]
        ax4.scatter(devs, pnl_pcts, c=colors_s, alpha=0.6, s=20)
        ax4.axhline(y=0, color="gray", linewidth=0.5)
    ax4.set_xlabel("Déviation VWAP (%)")
    ax4.set_ylabel("P&L (%)")
    ax4.set_title("Déviation vs P&L", fontsize=10)

    # RSI at entry vs PnL
    ax5 = fig.add_subplot(gs[2, 1])
    if trades:
        rsis = [t.rsi_at_entry for t in trades]
        pnl_pcts = [t.pnl_pct * 100 for t in trades]
        colors_r = ["#4CAF50" if p >= 0 else "#F44336" for p in pnl_pcts]
        ax5.scatter(rsis, pnl_pcts, c=colors_r, alpha=0.6, s=20)
        ax5.axhline(y=0, color="gray", linewidth=0.5)
    ax5.set_xlabel("RSI à l'entrée")
    ax5.set_ylabel("P&L (%)")
    ax5.set_title("RSI vs P&L", fontsize=10)

    stats_text = (
        f"Band: {result.config.vwap_band_mult:.2f}σ  |  "
        f"RSI: {result.config.rsi_floor:.0f}-{result.config.rsi_ceil:.0f}  |  "
        f"WR: {metrics['win_rate']:.0%}  |  PF: {metrics['profit_factor']:.2f}  |  "
        f"MaxDD: {metrics['max_drawdown']:.1%}  |  Sharpe: {metrics['sharpe']:.2f}  |  "
        f"Trades: {metrics['n_trades']}  |  Avg hold: {metrics['avg_hold_bars']:.0f}b"
    )
    fig.text(0.5, 0.01, stats_text, ha="center", fontsize=10, style="italic",
             bbox=dict(boxstyle="round,pad=0.5", facecolor="#E3F2FD", alpha=0.8))

    plt.tight_layout(rect=[0, 0.03, 1, 1])
    chart_path = OUTPUT_DIR / f"vwap_{result.start_date:%Y%m%d}_{result.end_date:%Y%m%d}.png"
    fig.savefig(chart_path, dpi=150, bbox_inches="tight")
    logger.info("💹 Graphique : %s", chart_path)
    print(f"  💹 Graphique : {chart_path}")

    if show:
        try:
            plt.show()
        except Exception:
            pass
    return chart_path


def _generate_combined_chart(
    vwap_result, vwap_m,
    crash_result, crash_m,
    trail_result, trail_m,
    total_balance: float,
    show: bool = True,
) -> Path:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mticker

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(2, 1, figsize=(16, 10), height_ratios=[3, 1])

    ax1 = axes[0]

    # VWAP
    if vwap_result and vwap_result.equity_curve:
        dates_v = [datetime.fromtimestamp(ts / 1000, tz=timezone.utc) for ts, _ in vwap_result.equity_curve]
        eq_v = [e for _, e in vwap_result.equity_curve]
        ax1.plot(dates_v, eq_v, color="#2196F3", linewidth=1.5, label=f"🔵 VWAP ({vwap_m['total_return']:+.1%})", alpha=0.9)

    # CrashBot
    if crash_result and crash_result.equity_curve:
        dates_c = [datetime.fromtimestamp(ts / 1000, tz=timezone.utc) for ts, _ in crash_result.equity_curve]
        eq_c = [e for _, e in crash_result.equity_curve]
        ax1.plot(dates_c, eq_c, color="#FF5722", linewidth=1.5, label=f"🔥 CrashBot ({crash_m['total_return']:+.1%})", alpha=0.9)

    # Trail Range
    if trail_result:
        trail_eq = _get_trail_eq(trail_result)
        if trail_eq:
            dates_t = [datetime.fromtimestamp(ts / 1000, tz=timezone.utc) for ts, _ in trail_eq]
            eq_t = [e for _, e in trail_eq]
            ax1.plot(dates_t, eq_t, color="#FFC107", linewidth=1.5, label=f"🟡 Trail Range ({trail_m['total_return']:+.1%})", alpha=0.9)

    ax1.axhline(y=total_balance, color="gray", linestyle="--", alpha=0.4)
    ax1.set_title("🏦 Backtest Combiné — 3 Stratégies", fontsize=14, fontweight="bold", pad=15)
    ax1.set_ylabel("Equity ($)")
    ax1.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x:,.0f}"))
    ax1.legend(loc="upper left", fontsize=11)

    # Drawdowns comparés
    ax2 = axes[1]
    if vwap_m and vwap_m["dd_curve"] and vwap_result:
        dates_v = [datetime.fromtimestamp(ts / 1000, tz=timezone.utc) for ts, _ in vwap_result.equity_curve]
        dd_v = vwap_m["dd_curve"]
        ax2.plot(dates_v[:len(dd_v)], dd_v, color="#2196F3", linewidth=0.8, label="VWAP", alpha=0.7)
    if crash_m and crash_m["dd_curve"] and crash_result:
        dates_c = [datetime.fromtimestamp(ts / 1000, tz=timezone.utc) for ts, _ in crash_result.equity_curve]
        dd_c = crash_m["dd_curve"]
        ax2.plot(dates_c[:len(dd_c)], dd_c, color="#FF5722", linewidth=0.8, label="CrashBot", alpha=0.7)

    ax2.set_ylabel("Drawdown")
    ax2.set_title("Drawdowns comparés", fontsize=10)
    ax2.yaxis.set_major_formatter(mticker.PercentFormatter(1.0, decimals=0))
    ax2.legend(loc="lower left", fontsize=9)

    plt.tight_layout()
    chart_path = OUTPUT_DIR / "combined_3strats.png"
    fig.savefig(chart_path, dpi=150, bbox_inches="tight")
    logger.info("💹 Graphique combiné : %s", chart_path)
    print(f"  💹 Graphique combiné : {chart_path}")

    if show:
        try:
            plt.show()
        except Exception:
            pass
    return chart_path


# ── CSV ────────────────────────────────────────────────────────────────────────


def _save_trades_csv(result: VwapResult) -> None:
    import csv
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUTPUT_DIR / f"vwap_trades_{result.start_date:%Y%m%d}_{result.end_date:%Y%m%d}.csv"

    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "symbol", "entry_time", "exit_time", "entry_price", "exit_price",
            "size", "pnl_usd", "pnl_pct", "exit_reason",
            "vwap_at_entry", "deviation_pct", "rsi_at_entry", "hold_bars",
        ])
        for t in result.trades:
            entry_dt = datetime.fromtimestamp(t.entry_time / 1000, tz=timezone.utc)
            exit_dt = datetime.fromtimestamp(t.exit_time / 1000, tz=timezone.utc)
            w.writerow([
                t.symbol, entry_dt.isoformat(), exit_dt.isoformat(),
                f"{t.entry_price:.6f}", f"{t.exit_price:.6f}",
                f"{t.size:.8f}", f"{t.pnl_usd:.2f}", f"{t.pnl_pct:.4f}",
                t.exit_reason, f"{t.vwap_at_entry:.6f}",
                f"{t.deviation_pct:.4f}", f"{t.rsi_at_entry:.1f}", t.hold_bars,
            ])

    print(f"  📄 Trades CSV : {path}")


if __name__ == "__main__":
    main()
