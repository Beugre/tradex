#!/usr/bin/env python3
"""
Risk Sweep — Teste différents risk_percent sur le portfolio 60% Crash / 40% Trail.
Télécharge les données une seule fois, puis boucle sur les niveaux de risque.
"""
import argparse
import logging
import math
import statistics
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

# ── Imports internes ──
from backtest.run_backtest_range_tptrail import (
    RangeTpTrailEngine,
    TpTrailConfig,
    download_all_pairs,
)
from backtest.run_backtest_dipbuy import DipBuyConfig, DipBuyEngine
from backtest.metrics import compute_metrics
from backtest.run_portfolio_sim import (
    TRAIL_PAIRS,
    CRASH_PAIRS,
    ALL_PAIRS,
    BotResult,
    build_equity_map,
    combined_equity_curve,
    compute_metrics_from_curve,
    monthly_returns_from_curve,
)
from backtest.simulator import EquityPoint

logging.basicConfig(
    level=logging.WARNING,  # silencieux pour le sweep
    format="[%(asctime)s] %(levelname)s %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("risk_sweep")


# ═══════════════════════════════════════════════════════════════════════════════
# Config factories avec risk_percent paramétrable
# ═══════════════════════════════════════════════════════════════════════════════

def trail_config(balance: float, risk_pct: float) -> TpTrailConfig:
    return TpTrailConfig(
        initial_balance=balance,
        step_pct=0.01,
        risk_percent=risk_pct,
        max_position_pct=0.30,
        max_simultaneous_positions=3,
        fee_pct=0.001,
        slippage_pct=0.001,
        swing_lookback=3,
        candle_window=100,
        range_width_min=0.02,
        range_entry_buffer_pct=0.002,
        range_sl_buffer_pct=0.003,
        range_cooldown_bars=3,
        compound=False,
    )


def crash_config(balance: float, risk_pct: float) -> DipBuyConfig:
    return DipBuyConfig(
        initial_balance=balance,
        interval="4h",
        lookback_bars=12,
        drop_threshold=0.20,
        tp_pct=0.08,
        atr_sl_mult=1.5,
        atr_period=14,
        trail_step_pct=0.005,
        trail_trigger_buffer=0.0005,
        risk_percent=risk_pct,
        max_simultaneous=5,
        cooldown_bars=6,
        fee_pct=0.00075,
        slippage_pct=0.001,
        sma_period=0,
        equity_sma_period=0,
        max_portfolio_heat=1.0,
        btc_trend_filter=False,
        min_drop_recovery=0.0,
        rsi_max=0.0,
        vol_spike_mult=0.0,
        min_wick_ratio=0.0,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# Simulation pour un risk_percent donné
# ═══════════════════════════════════════════════════════════════════════════════

def run_one_risk(
    risk_pct: float,
    trail_candles: dict,
    crash_candles: dict,
    balance: float,
    trail_pct: float,
    crash_pct: float,
    start: datetime,
    end: datetime,
) -> dict:
    """Lance Trail + Crash avec un risk_percent et renvoie les métriques."""
    trail_capital = balance * trail_pct
    crash_capital = balance * crash_pct

    # Trail
    t_cfg = trail_config(trail_capital, risk_pct)
    t_eng = RangeTpTrailEngine(trail_candles, t_cfg)
    t_res = t_eng.run()
    t_met = compute_metrics(t_res)

    # Crash
    c_cfg = crash_config(crash_capital, risk_pct)
    c_eng = DipBuyEngine(crash_candles, c_cfg)
    c_res = c_eng.run()
    c_met = compute_metrics(c_res)

    # Portfolio combiné
    bots = [
        BotResult("Trail", trail_pct, trail_capital, t_res,
                  build_equity_map(t_res.equity_curve), t_met, t_met["n_trades"]),
        BotResult("Crash", crash_pct, crash_capital, c_res,
                  build_equity_map(c_res.equity_curve), c_met, c_met["n_trades"]),
    ]

    all_ts = set()
    for b in bots:
        all_ts.update(b.equity_map.keys())
    timeline = sorted(all_ts)

    combined_eq = combined_equity_curve(bots, timeline)
    combined_m = compute_metrics_from_curve(combined_eq, balance, start, end)

    monthly = monthly_returns_from_curve(combined_eq, balance)
    positive_months = sum(1 for _, r in monthly if r > 0) if monthly else 0
    total_months = len(monthly) if monthly else 1
    worst_month = min((r for _, r in monthly), default=0)
    best_month = max((r for _, r in monthly), default=0)

    return {
        "risk_pct": risk_pct,
        # Trail
        "trail_return": t_met["total_return"],
        "trail_dd": t_met["max_drawdown"],
        "trail_sharpe": t_met["sharpe"],
        "trail_trades": t_met["n_trades"],
        "trail_wr": t_met["win_rate"],
        "trail_final": t_met["final_equity"],
        # Crash
        "crash_return": c_met["total_return"],
        "crash_dd": c_met["max_drawdown"],
        "crash_sharpe": c_met["sharpe"],
        "crash_trades": c_met["n_trades"],
        "crash_wr": c_met["win_rate"],
        "crash_final": c_met["final_equity"],
        # Portfolio
        "ptf_return": combined_m["total_return"],
        "ptf_cagr": combined_m["cagr"],
        "ptf_dd": combined_m["max_drawdown"],
        "ptf_sharpe": combined_m["sharpe"],
        "ptf_sortino": combined_m["sortino"],
        "ptf_final": combined_m["final_equity"],
        "ptf_trades": t_met["n_trades"] + c_met["n_trades"],
        # Monthly
        "months_positive": f"{positive_months}/{total_months}",
        "worst_month": worst_month,
        "best_month": best_month,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Risk Sweep 60/40 Portfolio")
    parser.add_argument("--months", type=int, default=72)
    parser.add_argument("--balance", type=float, default=1000.0)
    parser.add_argument("--trail-pct", type=float, default=0.40)
    parser.add_argument("--crash-pct", type=float, default=0.60)
    parser.add_argument(
        "--risks", type=str,
        default="0.01,0.02,0.03,0.04,0.05,0.06,0.08,0.10",
        help="Liste de risk_percent à tester (séparés par virgule)",
    )
    args = parser.parse_args()

    risk_levels = [float(r) for r in args.risks.split(",")]

    end = datetime.now(tz=timezone.utc)
    start = end - timedelta(days=args.months * 30)

    print(f"\n{'═' * 110}")
    print(f"  🎯 RISK SWEEP — Portfolio {args.trail_pct*100:.0f}% Trail / {args.crash_pct*100:.0f}% Crash")
    print(f"  📅 {start.date()} → {end.date()} ({args.months} mois)")
    print(f"  💰 Capital : ${args.balance:,.0f}")
    print(f"  🔍 Risk levels : {', '.join(f'{r:.0%}' for r in risk_levels)}")
    print(f"{'═' * 110}")

    # ── Téléchargement unique ──
    print(f"\n  📥 Téléchargement données H4…")
    all_candles = download_all_pairs(ALL_PAIRS, start, end, interval="4h")
    trail_candles = {p: all_candles[p] for p in TRAIL_PAIRS if p in all_candles}
    crash_candles = {p: all_candles[p] for p in CRASH_PAIRS if p in all_candles}
    print(f"  ✅ {len(all_candles)} paires chargées\n")

    # ── Sweep ──
    results = []
    for i, risk in enumerate(risk_levels):
        label = f"[{i+1}/{len(risk_levels)}] risk={risk:.0%}"
        print(f"  ⏳ {label}…", end="", flush=True)

        res = run_one_risk(
            risk_pct=risk,
            trail_candles=trail_candles,
            crash_candles=crash_candles,
            balance=args.balance,
            trail_pct=args.trail_pct,
            crash_pct=args.crash_pct,
            start=start,
            end=end,
        )
        results.append(res)
        print(f" ✅ Return {res['ptf_return']:+.1%} | DD {res['ptf_dd']:.1%} | Sharpe {res['ptf_sharpe']:.2f}")

    # ═══════════════════════════════════════════════════════════════════════
    # RAPPORT
    # ═══════════════════════════════════════════════════════════════════════
    print(f"\n{'═' * 110}")
    print(f"  📊 RÉSULTATS — RISK SWEEP ({args.trail_pct*100:.0f}% Trail / {args.crash_pct*100:.0f}% Crash)")
    print(f"{'═' * 110}")

    # ── Tableau principal : Portfolio ──
    header = (
        f"  {'Risk':>6s} │ {'Return':>8s} │ {'CAGR':>7s} │ "
        f"{'Sharpe':>6s} │ {'Sortino':>7s} │ {'MaxDD':>7s} │ "
        f"{'Final':>10s} │ {'Trades':>6s} │ {'Mois+':>7s} │ {'Pire':>7s}"
    )
    print(f"\n{header}")
    print("  " + "─" * (len(header) - 2))

    best_sharpe = max(r["ptf_sharpe"] for r in results)

    for r in results:
        is_best = r["ptf_sharpe"] == best_sharpe
        marker = " ⭐" if is_best else ""
        row = (
            f"  {r['risk_pct']:>5.0%} │ "
            f"{r['ptf_return']:>+7.1%} │ "
            f"{r['ptf_cagr']:>+6.1%} │ "
            f"{r['ptf_sharpe']:>6.2f} │ "
            f"{r['ptf_sortino']:>7.2f} │ "
            f"{r['ptf_dd']:>6.1%} │ "
            f"${r['ptf_final']:>9,.0f} │ "
            f"{r['ptf_trades']:>6d} │ "
            f"{r['months_positive']:>7s} │ "
            f"{r['worst_month']:>+6.1%}{marker}"
        )
        print(row)

    # ── Tableau détail par bot ──
    print(f"\n  📋 Détail par bot :")
    header2 = (
        f"  {'Risk':>6s} │ "
        f"{'T.Ret':>7s} │ {'T.DD':>6s} │ {'T.Sharpe':>8s} │ {'T.WR':>5s} │ "
        f"{'C.Ret':>7s} │ {'C.DD':>6s} │ {'C.Sharpe':>8s} │ {'C.WR':>5s}"
    )
    print(f"\n{header2}")
    print("  " + "─" * (len(header2) - 2))

    for r in results:
        row = (
            f"  {r['risk_pct']:>5.0%} │ "
            f"{r['trail_return']:>+6.1%} │ "
            f"{r['trail_dd']:>5.1%} │ "
            f"{r['trail_sharpe']:>8.2f} │ "
            f"{r['trail_wr']:>4.0%} │ "
            f"{r['crash_return']:>+6.1%} │ "
            f"{r['crash_dd']:>5.1%} │ "
            f"{r['crash_sharpe']:>8.2f} │ "
            f"{r['crash_wr']:>4.0%}"
        )
        print(row)

    # ── Recherche du sweet spot ──
    print(f"\n{'─' * 110}")
    print(f"  🎯 ANALYSE DU SWEET SPOT")
    print(f"{'─' * 110}")

    # Trouver le risk qui maximise Sharpe
    best = max(results, key=lambda r: r["ptf_sharpe"])
    print(f"\n  ⭐ Meilleur Sharpe : risk={best['risk_pct']:.0%} → "
          f"Sharpe {best['ptf_sharpe']:.2f} | CAGR {best['ptf_cagr']:+.1%} | DD {best['ptf_dd']:.1%}")

    # Trouver le risk avec DD ≈ -10%
    target_dd = -0.10
    closest_dd = min(results, key=lambda r: abs(r["ptf_dd"] - target_dd))
    print(f"  🎯 DD proche -10% : risk={closest_dd['risk_pct']:.0%} → "
          f"DD {closest_dd['ptf_dd']:.1%} | CAGR {closest_dd['ptf_cagr']:+.1%} | Sharpe {closest_dd['ptf_sharpe']:.2f}")

    # Trouver le risk max avec Sharpe ≥ 1.5
    good = [r for r in results if r["ptf_sharpe"] >= 1.5]
    if good:
        most_aggressive = max(good, key=lambda r: r["risk_pct"])
        print(f"  🏆 Max risk avec Sharpe≥1.5 : risk={most_aggressive['risk_pct']:.0%} → "
              f"Sharpe {most_aggressive['ptf_sharpe']:.2f} | CAGR {most_aggressive['ptf_cagr']:+.1%} | "
              f"DD {most_aggressive['ptf_dd']:.1%} | Final ${most_aggressive['ptf_final']:,.0f}")

    # Ratio DD/CAGR
    print(f"\n  📊 Efficience DD/CAGR (plus bas = meilleur) :")
    for r in results:
        ratio = abs(r["ptf_dd"]) / r["ptf_cagr"] if r["ptf_cagr"] > 0 else 99
        bar = "█" * int(ratio * 20)
        print(f"     risk={r['risk_pct']:>4.0%} : DD/CAGR = {ratio:.2f}  {bar}")

    print(f"\n{'═' * 110}\n")


if __name__ == "__main__":
    main()
