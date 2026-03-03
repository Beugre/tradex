#!/usr/bin/env python
"""
Backtest Momentum Squeeze — compression→expansion de volatilité.

Stratégie décorrélée :
  - Trail Range (H4) : structure Dow / breakout macro → jours/semaines
  - CrashBot (1m)    : event-driven sur crashs extrêmes → minutes
  - Momentum Squeeze (15m) : compression BB→KC puis breakout directionnel → heures

Usage :
    # Momentum seul, 6 ans
    PYTHONPATH=. python backtest/run_backtest_momentum.py --months 72 --no-show

    # Analyse de sensibilité
    PYTHONPATH=. python backtest/run_backtest_momentum.py --months 72 --sensitivity --no-show

    # Combiné 3 stratégies (Momentum + CrashBot + Trail Range)
    PYTHONPATH=. python backtest/run_backtest_momentum.py --months 72 --combined --no-show
"""

from __future__ import annotations

import argparse
import logging
import sys
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path

from backtest.data_loader import download_candles
from backtest.simulator_momentum import (
    MomentumConfig,
    MomentumEngine,
    MomentumResult,
    MomentumTrade,
    compute_momentum_metrics,
)

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("momentum_bt")

# ── Paires — top liquidité, bonnes pour squeeze/breakout ──────────────────────

MOMENTUM_PAIRS = [
    "BTC-USD", "ETH-USD", "SOL-USD", "XRP-USD", "BNB-USD",
    "DOGE-USD", "ADA-USD", "AVAX-USD", "DOT-USD", "LINK-USD",
    "SUI-USD", "NEAR-USD", "LTC-USD", "ARB-USD", "OP-USD",
    "PEPE-USD", "FET-USD", "RENDER-USD", "INJ-USD", "AAVE-USD",
]

OUTPUT_DIR = Path(__file__).parent / "output"


def main() -> None:
    parser = argparse.ArgumentParser(description="Backtest Momentum Squeeze")
    parser.add_argument("--balance", type=float, default=1000.0)
    parser.add_argument("--months", type=int, default=12)
    parser.add_argument("--start", type=str, default=None)
    parser.add_argument("--end", type=str, default=None)
    parser.add_argument("--interval", type=str, default="15m", choices=["5m", "15m", "1h"])
    parser.add_argument("--risk", type=float, default=0.02)

    # Squeeze detection
    parser.add_argument("--bb-period", type=int, default=20)
    parser.add_argument("--bb-mult", type=float, default=2.0, help="Bollinger Bands σ mult")
    parser.add_argument("--kc-mult", type=float, default=1.5, help="Keltner Channel ATR mult")
    parser.add_argument("--squeeze-min", type=int, default=6, help="Min bars in squeeze")

    # Breakout confirmation
    parser.add_argument("--vol-confirm", type=float, default=1.3, help="Volume confirm mult (0=off)")
    parser.add_argument("--trend-filter", action="store_true", help="EMA50 trend filter")

    # Trade management
    parser.add_argument("--sl-atr", type=float, default=1.5, help="SL = N×ATR")
    parser.add_argument("--tp-atr", type=float, default=3.0, help="TP = N×ATR")
    parser.add_argument("--trailing", action="store_true", default=True)
    parser.add_argument("--no-trailing", dest="trailing", action="store_false")
    parser.add_argument("--trail-act", type=float, default=1.5, help="Trailing activation (ATR)")
    parser.add_argument("--trail-dist", type=float, default=1.0, help="Trailing distance (ATR)")
    parser.add_argument("--timeout", type=int, default=48, help="Timeout bars")
    parser.add_argument("--cooldown", type=int, default=8, help="Cooldown bars")

    # Modes
    parser.add_argument("--no-short", action="store_true", help="Long only")
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

    logger.info("🟡 Momentum Squeeze Backtest — %s → %s (%s)", start.date(), end.date(), args.interval)

    # Téléchargement
    logger.info("📥 Téléchargement klines %s…", args.interval)
    candles_by_symbol = {}
    for pair in MOMENTUM_PAIRS:
        candles = download_candles(pair, start, end, interval=args.interval)
        if candles:
            candles_by_symbol[pair] = candles
            logger.info("   ✅ %s : %d bougies", pair, len(candles))
        else:
            logger.warning("   ⚠️ %s : aucune donnée", pair)

    if not candles_by_symbol:
        logger.error("❌ Aucune donnée — abandon")
        sys.exit(1)

    base_cfg = MomentumConfig(
        initial_balance=args.balance,
        risk_percent=args.risk,
        bb_period=args.bb_period,
        bb_mult=args.bb_mult,
        kc_mult=args.kc_mult,
        squeeze_min_bars=args.squeeze_min,
        volume_confirm_mult=args.vol_confirm,
        require_trend_align=args.trend_filter,
        sl_atr_mult=args.sl_atr,
        tp_atr_mult=args.tp_atr,
        use_trailing=args.trailing,
        trail_activation_atr=args.trail_act,
        trail_distance_atr=args.trail_dist,
        timeout_bars=args.timeout,
        cooldown_bars=args.cooldown,
        allow_short=not args.no_short,
    )

    if args.sensitivity:
        _run_sensitivity(candles_by_symbol, base_cfg, args)
    elif args.combined:
        _run_combined(start, end, args)
    else:
        _run_single(candles_by_symbol, base_cfg, args)


# ── Run unique ─────────────────────────────────────────────────────────────────


def _run_single(candles: dict, cfg: MomentumConfig, args) -> None:
    engine = MomentumEngine(candles, cfg, interval=args.interval)
    result = engine.run()
    metrics = compute_momentum_metrics(result)
    _print_report(result, metrics)
    _generate_chart(result, metrics, show=not args.no_show)
    _save_trades_csv(result)


# ── Sensibilité ────────────────────────────────────────────────────────────────


def _run_sensitivity(candles: dict, base_cfg: MomentumConfig, args) -> None:
    # Grille de paramètres
    kc_mults = [1.0, 1.25, 1.5, 2.0]
    squeeze_mins = [4, 6, 8, 12]
    sl_atr_mults = [1.0, 1.5, 2.0]
    tp_atr_mults = [2.0, 3.0, 4.0, 5.0]
    trail_modes = [
        (True, 1.5, 1.0, "trail_1.5/1.0"),
        (True, 2.0, 1.5, "trail_2.0/1.5"),
        (False, 0, 0, "no_trail"),
    ]

    results = []
    total = len(kc_mults) * len(squeeze_mins) * len(sl_atr_mults) * len(tp_atr_mults) * len(trail_modes)
    logger.info("🔬 Sensibilité : %d combinaisons", total)

    done = 0
    for kc in kc_mults:
        for sq in squeeze_mins:
            for sl in sl_atr_mults:
                for tp in tp_atr_mults:
                    for use_trail, trail_act, trail_dist, trail_label in trail_modes:
                        cfg = MomentumConfig(
                            initial_balance=base_cfg.initial_balance,
                            risk_percent=base_cfg.risk_percent,
                            bb_period=base_cfg.bb_period,
                            bb_mult=base_cfg.bb_mult,
                            kc_mult=kc,
                            squeeze_min_bars=sq,
                            volume_confirm_mult=base_cfg.volume_confirm_mult,
                            require_trend_align=base_cfg.require_trend_align,
                            sl_atr_mult=sl,
                            tp_atr_mult=tp,
                            use_trailing=use_trail,
                            trail_activation_atr=trail_act,
                            trail_distance_atr=trail_dist,
                            timeout_bars=base_cfg.timeout_bars,
                            cooldown_bars=base_cfg.cooldown_bars,
                            allow_short=base_cfg.allow_short,
                        )
                        engine = MomentumEngine(candles, cfg, interval=args.interval)
                        result = engine.run()
                        metrics = compute_momentum_metrics(result)
                        results.append((metrics, cfg, trail_label))
                        done += 1
                        if done % 50 == 0:
                            logger.info("   %d/%d done…", done, total)

    results.sort(key=lambda x: x[0]["profit_factor"] if x[0]["n_trades"] >= 20 else -99, reverse=True)

    sep = "═" * 155
    print(f"\n{sep}")
    print("  🔬 SENSIBILITÉ — Momentum Squeeze")
    print(sep)
    print(
        f"  {'KC':>4s} | {'Sqz≥':>4s} | {'SL×':>4s} | {'TP×':>4s} | {'Trail':>14s} | "
        f"{'Trades':>6s} | {'Sqz':>5s} | {'WR':>6s} | {'PF':>6s} | "
        f"{'Return':>8s} | {'CAGR':>7s} | {'MaxDD':>7s} | {'Sharpe':>7s} | "
        f"{'AvgPnL':>8s} | {'AvgHold':>7s} | {'L/S':>5s}"
    )
    print("  " + "─" * 151)

    for m, cfg, trail_label in results[:60]:
        if m["n_trades"] < 5:
            continue
        by_side = m.get("by_side", {})
        longs = by_side.get("LONG", {}).get("n", 0)
        shorts = by_side.get("SHORT", {}).get("n", 0)
        print(
            f"  {cfg.kc_mult:4.2f} | {cfg.squeeze_min_bars:4d} | {cfg.sl_atr_mult:4.1f} | "
            f"{cfg.tp_atr_mult:4.1f} | {trail_label:>14s} | "
            f"{m['n_trades']:6d} | {m['n_squeezes']:5d} | {m['win_rate']:6.1%} | "
            f"{m['profit_factor']:6.2f} | "
            f"{m['total_return']:+7.1%} | {m['cagr']:+6.1%} | {m['max_drawdown']:7.1%} | "
            f"{m['sharpe']:7.2f} | "
            f"${m['avg_pnl_usd']:+7.2f} | {m['avg_hold_bars']:6.1f}b | {longs}/{shorts}"
        )

    print(f"\n{sep}\n")

    # Best
    viable = [(m, c, l) for m, c, l in results if m["n_trades"] >= 20]
    if viable:
        best_m, best_cfg, best_trail = viable[0]
        by_side = best_m.get("by_side", {})
        longs = by_side.get("LONG", {}).get("n", 0)
        shorts = by_side.get("SHORT", {}).get("n", 0)
        print(
            f"  🏆 Meilleure : KC={best_cfg.kc_mult:.2f}, Squeeze≥{best_cfg.squeeze_min_bars}, "
            f"SL={best_cfg.sl_atr_mult:.1f}×ATR, TP={best_cfg.tp_atr_mult:.1f}×ATR, {best_trail}"
        )
        print(
            f"     Return={best_m['total_return']:+.1%}, WR={best_m['win_rate']:.0%}, "
            f"PF={best_m['profit_factor']:.2f}, CAGR={best_m['cagr']:+.1%}, "
            f"MaxDD={best_m['max_drawdown']:.1%}, Sharpe={best_m['sharpe']:.2f}, "
            f"Trades={best_m['n_trades']} (L:{longs}/S:{shorts})\n"
        )


# ── Combiné 3 stratégies ──────────────────────────────────────────────────────


def _run_combined(start: datetime, end: datetime, args) -> None:
    from backtest.simulator_antiliq import AntiliqConfig, AntiliqEngine, compute_antiliq_metrics
    from backtest.run_backtest_antiliq import ANTILIQ_PAIRS

    total_balance = args.balance
    logger.info("🟡🔥🟢 Backtest COMBINÉ 3 stratégies — $%.0f — %s → %s",
                total_balance, start.date(), end.date())

    # ── 1. Momentum Squeeze (15m) ─────────────────────────────────────────
    logger.info("\n" + "═" * 60)
    logger.info("  🟡 Stratégie 1 : Momentum Squeeze (15m)")
    logger.info("═" * 60)

    mom_candles = {}
    for pair in MOMENTUM_PAIRS:
        c = download_candles(pair, start, end, interval=args.interval)
        if c:
            mom_candles[pair] = c

    mom_cfg = MomentumConfig(
        initial_balance=total_balance,
        risk_percent=args.risk,
        sl_atr_mult=args.sl_atr,
        tp_atr_mult=args.tp_atr,
        use_trailing=args.trailing,
        trail_activation_atr=args.trail_act,
        trail_distance_atr=args.trail_dist,
        timeout_bars=args.timeout,
        cooldown_bars=args.cooldown,
        allow_short=not args.no_short,
    )
    mom_engine = MomentumEngine(mom_candles, mom_cfg, interval=args.interval)
    mom_result = mom_engine.run()
    mom_metrics = compute_momentum_metrics(mom_result)

    # ── 2. CrashBot / Antiliq (1m) ────────────────────────────────────────
    logger.info("\n" + "═" * 60)
    logger.info("  🔥 Stratégie 2 : CrashBot / Antiliq (1m)")
    logger.info("═" * 60)

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
    logger.info("  🟢 Stratégie 3 : Trail Range (H4)")
    logger.info("═" * 60)

    trail_result, trail_metrics = _run_trail_range(start, end, total_balance)

    # ── 4. Rapport combiné ─────────────────────────────────────────────────
    _print_combined_report(
        mom_result, mom_metrics,
        crash_result, crash_metrics,
        trail_result, trail_metrics,
        total_balance,
    )
    _generate_combined_chart(
        mom_result, mom_metrics,
        crash_result, crash_metrics,
        trail_result, trail_metrics,
        total_balance,
        show=not args.no_show,
    )


def _run_trail_range(start: datetime, end: datetime, balance: float):
    """Lance le backtest Trail Range H4."""
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
    mom_result, mom_m,
    crash_result, crash_m,
    trail_result, trail_m,
    total_balance: float,
) -> None:
    sep = "═" * 90

    print(f"\n{sep}")
    print("  🏦 BACKTEST COMBINÉ — 3 STRATÉGIES (Momentum + CrashBot + Trail Range)")
    print(f"  Capital initial : ${total_balance:,.0f}")
    if mom_result:
        print(f"  Période : {mom_result.start_date:%b %Y} → {mom_result.end_date:%b %Y}")
    print(sep)

    strats = []
    if mom_m:
        strats.append(("🟡 Momentum Squeeze", mom_m))
    if crash_m:
        strats.append(("🔥 CrashBot", crash_m))
    if trail_m:
        strats.append(("🟢 Trail Range", trail_m))

    print(f"\n  {'Stratégie':<24s} | {'Trades':>6s} | {'WR':>6s} | {'PF':>6s} | "
          f"{'Return':>8s} | {'CAGR':>7s} | {'MaxDD':>7s} | {'Sharpe':>7s} | {'Final $':>10s}")
    print("  " + "─" * 100)

    combined_equity = 0.0
    combined_trades = 0
    all_pnls = []

    for name, m in strats:
        final = m.get("final_equity", total_balance)
        n = m.get("n_trades", 0)
        combined_equity += final
        combined_trades += n

        print(
            f"  {name:<24s} | {n:6d} | {m.get('win_rate', 0):6.1%} | "
            f"{m.get('profit_factor', 0):6.2f} | {m.get('total_return', 0):+7.1%} | "
            f"{m.get('cagr', 0):+6.1%} | {m.get('max_drawdown', 0):7.1%} | "
            f"{m.get('sharpe', 0):7.2f} | ${final:>9,.2f}"
        )

    n_strats = len(strats)
    avg_equity = combined_equity / n_strats if n_strats > 0 else total_balance
    combined_return = (avg_equity - total_balance) / total_balance

    print("  " + "─" * 100)
    print(f"  {'COMBINÉ (avg)':<24s} | {combined_trades:6d} | {'':>6s} | "
          f"{'':>6s} | {combined_return:+7.1%} | "
          f"{'':>7s} | {'':>7s} | {'':>7s} | ${avg_equity:>9,.2f}")

    # Corrélation
    print(f"\n  📊 Analyse de corrélation")
    print("  " + "─" * 76)
    if mom_result and crash_result:
        _print_correlation(mom_result.equity_curve, crash_result.equity_curve,
                           "Momentum", "CrashBot")
    if mom_result and trail_result:
        _print_correlation(mom_result.equity_curve, _get_trail_eq(trail_result),
                           "Momentum", "Trail Range")
    if crash_result and trail_result:
        _print_correlation(crash_result.equity_curve, _get_trail_eq(trail_result),
                           "CrashBot", "Trail Range")

    # Allocation
    mom_pf = mom_m.get("profit_factor", 0) if mom_m else 0
    trail_pf = trail_m.get("profit_factor", 0) if trail_m else 0
    crash_pf = crash_m.get("profit_factor", 0) if crash_m else 0

    print(f"\n  🏦 Suggestion d'allocation (basée sur PF)")
    print("  " + "─" * 76)
    print(f"  PF Momentum={mom_pf:.2f} | PF Crash={crash_pf:.2f} | PF Trail={trail_pf:.2f}")

    # Allocation proportionnelle au PF
    total_pf = max(mom_pf + crash_pf + trail_pf, 0.01)
    alloc_mom = int(round(mom_pf / total_pf * 100))
    alloc_crash = int(round(crash_pf / total_pf * 100))
    alloc_trail = 100 - alloc_mom - alloc_crash

    # Crash garde au moins 40% (sécurité)
    if alloc_crash < 40:
        diff = 40 - alloc_crash
        alloc_crash = 40
        alloc_mom = max(5, alloc_mom - diff // 2)
        alloc_trail = 100 - alloc_mom - alloc_crash

    for name, pct in [("Momentum", alloc_mom), ("CrashBot", alloc_crash), ("Trail", alloc_trail)]:
        print(f"  → {name:>10s} : {pct:3d}% = ${total_balance * pct / 100:,.0f}")

    print(f"\n{sep}\n")


def _get_trail_eq(trail_result) -> list[tuple[int, float]]:
    """Extrait equity_curve du BacktestResult."""
    if trail_result is None:
        return []
    eq = trail_result.equity_curve
    if not eq:
        return []
    return [(pt.timestamp, pt.equity) for pt in eq]


def _print_correlation(eq_a, eq_b, name_a: str, name_b: str) -> None:
    """Corrélation des rendements journaliers entre deux equity curves."""
    if len(eq_a) < 10 or len(eq_b) < 10:
        print(f"  {name_a} ↔ {name_b} : données insuffisantes")
        return

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


def _print_report(result: MomentumResult, m: dict) -> None:
    cfg = result.config
    sep = "═" * 65

    print(f"\n{sep}")
    print(f"  🟡 Momentum Squeeze — {result.start_date:%b %Y} → {result.end_date:%b %Y}")
    print(f"  Paires : {len(result.pairs)} | Interval : {cfg.bb_period}BB/{cfg.kc_mult:.1f}KC")
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
    print(f"  Squeezes détectés  : {m['n_squeezes']}")
    print(f"  Signaux breakout   : {m['n_signals']}")
    print(f"  Trades exécutés    : {m['n_trades']}")
    n = m["n_trades"]
    if n > 0:
        print(f"  Win Rate           : {m['win_rate']:.1%} ({int(m['win_rate'] * n)}/{n})")
        print(f"  Profit Factor      : {m['profit_factor']:.2f}")
        print(f"  PnL moyen          : ${m['avg_pnl_usd']:+.2f} ({m['avg_pnl_pct']:+.2%})")
        print(f"  Durée moy. trade   : {m['avg_hold_bars']:.1f} barres")

    if m["best_trade"]:
        b = m["best_trade"]
        print(f"  Meilleur trade     : ${b.pnl_usd:+.2f} ({b.pnl_pct:+.1%}) {b.side} {b.symbol} [squeeze {b.squeeze_bars}b]")
    if m["worst_trade"]:
        w = m["worst_trade"]
        print(f"  Pire trade         : ${w.pnl_usd:+.2f} ({w.pnl_pct:+.1%}) {w.side} {w.symbol} [squeeze {w.squeeze_bars}b]")

    # Par side
    if m.get("by_side"):
        print(f"\n  📊 Par direction")
        print("  " + "─" * 61)
        for side, s in sorted(m["by_side"].items()):
            print(
                f"  {side:8s} : {s['n']:3d} trades | WR {s['wr']:.0%}"
                f" | PF {s['pf']:.2f} | PnL ${s['pnl']:+.2f} | Hold {s['avg_hold']:.0f}b"
            )

    if m.get("by_pair"):
        print(f"\n  📊 Par paire")
        print("  " + "─" * 61)
        for pair, s in sorted(m["by_pair"].items(), key=lambda x: x[1]["pnl"], reverse=True):
            print(
                f"  {pair:12s} : {s['n']:3d} trades | WR {s['wr']:.0%}"
                f" | PF {s['pf']:.2f} | PnL ${s['pnl']:+.2f} | Hold {s['avg_hold']:.0f}b"
            )

    if m.get("by_exit"):
        print(f"\n  📊 Par sortie")
        print("  " + "─" * 61)
        for reason, s in m["by_exit"].items():
            pct = s["n"] / n * 100 if n > 0 else 0
            print(
                f"  {reason:10s} : {s['n']:3d} trades ({pct:4.1f}%)"
                f" | WR {s['wr']:.0%} | PnL ${s['pnl']:+.2f}"
            )

    if m.get("by_squeeze"):
        print(f"\n  📊 Par durée de squeeze")
        print("  " + "─" * 61)
        for bucket, s in m["by_squeeze"].items():
            print(
                f"  {bucket:10s} : {s['n']:3d} trades | WR {s['wr']:.0%}"
                f" | PF {s['pf']:.2f} | PnL ${s['pnl']:+.2f}"
            )

    if m.get("by_bbwidth"):
        print(f"\n  📊 Par largeur Bollinger (au breakout)")
        print("  " + "─" * 61)
        for bucket, s in m["by_bbwidth"].items():
            print(
                f"  {bucket:10s} : {s['n']:3d} trades | WR {s['wr']:.0%}"
                f" | PF {s['pf']:.2f} | PnL ${s['pnl']:+.2f}"
            )

    print(f"\n  ⚙️  Configuration")
    print("  " + "─" * 61)
    print(f"  Bollinger Bands    : {cfg.bb_period}p, {cfg.bb_mult:.1f}σ")
    print(f"  Keltner Channel    : {cfg.kc_period}p, {cfg.kc_mult:.1f}×ATR")
    print(f"  Squeeze min bars   : {cfg.squeeze_min_bars}")
    print(f"  Volume confirm     : {cfg.volume_confirm_mult:.1f}×")
    print(f"  SL                 : {cfg.sl_atr_mult:.1f}×ATR")
    print(f"  TP                 : {cfg.tp_atr_mult:.1f}×ATR")
    trail_str = f"act={cfg.trail_activation_atr:.1f}×ATR, dist={cfg.trail_distance_atr:.1f}×ATR" if cfg.use_trailing else "OFF"
    print(f"  Trailing           : {trail_str}")
    print(f"  Timeout            : {cfg.timeout_bars} barres")
    print(f"  Cooldown           : {cfg.cooldown_bars} barres")
    print(f"  Shorts             : {'OUI' if cfg.allow_short else 'NON'}")
    print(f"  Fee                : {cfg.fee_pct:.2%} | Slippage : {cfg.slippage_pct:.2%}")

    print(f"\n{sep}\n")


# ── Graphiques ─────────────────────────────────────────────────────────────────


def _generate_chart(result: MomentumResult, metrics: dict, show: bool = True) -> Path:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mticker

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    plt.style.use("seaborn-v0_8-whitegrid")
    fig = plt.figure(figsize=(16, 12))
    gs = fig.add_gridspec(3, 2, height_ratios=[3, 1.5, 1.5], hspace=0.35, wspace=0.3)

    eq = result.equity_curve
    dates = [datetime.fromtimestamp(ts / 1000, tz=timezone.utc) for ts, _ in eq]
    equities = [e for _, e in eq]
    trades = result.trades

    # 1. Equity curve
    ax1 = fig.add_subplot(gs[0, :])
    ax1.fill_between(dates, equities, alpha=0.15, color="#FFC107")
    ax1.plot(dates, equities, color="#F57F17", linewidth=1.2, label="Equity")
    ax1.axhline(y=result.initial_balance, color="gray", linestyle="--", alpha=0.5)

    # Mark trades on equity
    for t in trades:
        entry_dt = datetime.fromtimestamp(t.entry_time / 1000, tz=timezone.utc)
        color = "#4CAF50" if t.pnl_usd >= 0 else "#F44336"
        marker = "^" if t.side == "LONG" else "v"
        ax1.axvline(x=entry_dt, color=color, alpha=0.05, linewidth=0.5)

    ax1.set_title(
        f"Momentum Squeeze — {result.start_date:%b %Y} → {result.end_date:%b %Y}  |  "
        f"${result.initial_balance:,.0f} → ${result.final_equity:,.2f} "
        f"({metrics['total_return']:+.1%})",
        fontsize=13, fontweight="bold", pad=15,
    )
    ax1.set_ylabel("Equity ($)")
    ax1.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x:,.0f}"))
    ax1.legend(loc="upper left")

    # 2. Drawdown
    ax2 = fig.add_subplot(gs[1, 0])
    dd = metrics["dd_curve"]
    if dd and dates:
        ax2.fill_between(dates[:len(dd)], dd, alpha=0.3, color="#F44336")
        ax2.plot(dates[:len(dd)], dd, color="#D32F2F", linewidth=0.8)
    ax2.set_ylabel("Drawdown")
    ax2.set_title("Drawdown", fontsize=10)
    ax2.yaxis.set_major_formatter(mticker.PercentFormatter(1.0, decimals=0))

    # 3. Trade PnL (chronologique)
    ax3 = fig.add_subplot(gs[1, 1])
    if trades:
        pnls = [t.pnl_pct * 100 for t in trades]
        colors = ["#4CAF50" if p >= 0 else "#F44336" for p in pnls]
        ax3.bar(range(len(pnls)), pnls, color=colors, alpha=0.7, width=0.9)
        ax3.axhline(y=0, color="gray", linewidth=0.5)
    ax3.set_ylabel("P&L %")
    ax3.set_title("Trades (chronologique)", fontsize=10)

    # 4. Squeeze duration vs PnL
    ax4 = fig.add_subplot(gs[2, 0])
    if trades:
        squeezes = [t.squeeze_bars for t in trades]
        pnl_pcts = [t.pnl_pct * 100 for t in trades]
        colors_s = ["#4CAF50" if p >= 0 else "#F44336" for p in pnl_pcts]
        ax4.scatter(squeezes, pnl_pcts, c=colors_s, alpha=0.6, s=20)
        ax4.axhline(y=0, color="gray", linewidth=0.5)
    ax4.set_xlabel("Squeeze duration (bars)")
    ax4.set_ylabel("P&L (%)")
    ax4.set_title("Squeeze durée vs P&L", fontsize=10)

    # 5. BB Width vs PnL
    ax5 = fig.add_subplot(gs[2, 1])
    if trades:
        bbw = [t.bb_width_at_entry * 100 for t in trades]
        pnl_pcts = [t.pnl_pct * 100 for t in trades]
        colors_b = ["#4CAF50" if p >= 0 else "#F44336" for p in pnl_pcts]
        ax5.scatter(bbw, pnl_pcts, c=colors_b, alpha=0.6, s=20)
        ax5.axhline(y=0, color="gray", linewidth=0.5)
    ax5.set_xlabel("BB Width au breakout (%)")
    ax5.set_ylabel("P&L (%)")
    ax5.set_title("BB Width vs P&L", fontsize=10)

    stats_text = (
        f"KC: {result.config.kc_mult:.1f}  |  Squeeze≥{result.config.squeeze_min_bars}  |  "
        f"SL: {result.config.sl_atr_mult:.1f}×ATR  |  TP: {result.config.tp_atr_mult:.1f}×ATR  |  "
        f"WR: {metrics['win_rate']:.0%}  |  PF: {metrics['profit_factor']:.2f}  |  "
        f"MaxDD: {metrics['max_drawdown']:.1%}  |  Sharpe: {metrics['sharpe']:.2f}  |  "
        f"Trades: {metrics['n_trades']}  |  Squeezes: {metrics['n_squeezes']}"
    )
    fig.text(0.5, 0.01, stats_text, ha="center", fontsize=10, style="italic",
             bbox=dict(boxstyle="round,pad=0.5", facecolor="#FFF9C4", alpha=0.8))

    plt.tight_layout(rect=[0, 0.03, 1, 1])
    chart_path = OUTPUT_DIR / f"momentum_{result.start_date:%Y%m%d}_{result.end_date:%Y%m%d}.png"
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
    mom_result, mom_m,
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

    # Momentum
    if mom_result and mom_result.equity_curve:
        dates_m = [datetime.fromtimestamp(ts / 1000, tz=timezone.utc) for ts, _ in mom_result.equity_curve]
        eq_m = [e for _, e in mom_result.equity_curve]
        ax1.plot(dates_m, eq_m, color="#FFC107", linewidth=1.5,
                 label=f"Momentum ({mom_m['total_return']:+.1%})", alpha=0.9)

    # CrashBot
    if crash_result and crash_result.equity_curve:
        dates_c = [datetime.fromtimestamp(ts / 1000, tz=timezone.utc) for ts, _ in crash_result.equity_curve]
        eq_c = [e for _, e in crash_result.equity_curve]
        ax1.plot(dates_c, eq_c, color="#FF5722", linewidth=1.5,
                 label=f"CrashBot ({crash_m['total_return']:+.1%})", alpha=0.9)

    # Trail Range
    if trail_result:
        trail_eq = _get_trail_eq(trail_result)
        if trail_eq:
            dates_t = [datetime.fromtimestamp(ts / 1000, tz=timezone.utc) for ts, _ in trail_eq]
            eq_t = [e for _, e in trail_eq]
            ax1.plot(dates_t, eq_t, color="#4CAF50", linewidth=1.5,
                     label=f"Trail Range ({trail_m['total_return']:+.1%})", alpha=0.9)

    ax1.axhline(y=total_balance, color="gray", linestyle="--", alpha=0.4)
    ax1.set_title("Backtest Combiné — 3 Stratégies", fontsize=14, fontweight="bold", pad=15)
    ax1.set_ylabel("Equity ($)")
    ax1.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x:,.0f}"))
    ax1.legend(loc="upper left", fontsize=11)

    # Drawdowns
    ax2 = axes[1]
    if mom_m and mom_m["dd_curve"] and mom_result:
        dates_m = [datetime.fromtimestamp(ts / 1000, tz=timezone.utc) for ts, _ in mom_result.equity_curve]
        dd_m = mom_m["dd_curve"]
        ax2.plot(dates_m[:len(dd_m)], dd_m, color="#FFC107", linewidth=0.8, label="Momentum", alpha=0.7)
    if crash_m and crash_m["dd_curve"] and crash_result:
        dates_c = [datetime.fromtimestamp(ts / 1000, tz=timezone.utc) for ts, _ in crash_result.equity_curve]
        dd_c = crash_m["dd_curve"]
        ax2.plot(dates_c[:len(dd_c)], dd_c, color="#FF5722", linewidth=0.8, label="CrashBot", alpha=0.7)

    ax2.set_ylabel("Drawdown")
    ax2.set_title("Drawdowns comparés", fontsize=10)
    ax2.yaxis.set_major_formatter(mticker.PercentFormatter(1.0, decimals=0))
    ax2.legend(loc="lower left", fontsize=9)

    plt.tight_layout()
    chart_path = OUTPUT_DIR / "combined_3strats_momentum.png"
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


def _save_trades_csv(result: MomentumResult) -> None:
    import csv
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUTPUT_DIR / f"momentum_trades_{result.start_date:%Y%m%d}_{result.end_date:%Y%m%d}.csv"

    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "symbol", "side", "entry_time", "exit_time", "entry_price", "exit_price",
            "size", "pnl_usd", "pnl_pct", "exit_reason",
            "squeeze_bars", "atr_at_entry", "bb_width_at_entry", "hold_bars",
        ])
        for t in result.trades:
            entry_dt = datetime.fromtimestamp(t.entry_time / 1000, tz=timezone.utc)
            exit_dt = datetime.fromtimestamp(t.exit_time / 1000, tz=timezone.utc)
            w.writerow([
                t.symbol, t.side, entry_dt.isoformat(), exit_dt.isoformat(),
                f"{t.entry_price:.6f}", f"{t.exit_price:.6f}",
                f"{t.size:.8f}", f"{t.pnl_usd:.2f}", f"{t.pnl_pct:.4f}",
                t.exit_reason, t.squeeze_bars,
                f"{t.atr_at_entry:.6f}", f"{t.bb_width_at_entry:.6f}", t.hold_bars,
            ])

    print(f"  📄 Trades CSV : {path}")


if __name__ == "__main__":
    main()
