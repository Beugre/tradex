#!/usr/bin/env python
"""
Validation de robustesse — Donchian Trend Following (Daily).

Trois analyses complémentaires :
  1. Sensitivity sur la période complète (~5.8 ans) autour de Config G
  2. Walk-forward analysis (train/test glissant)
  3. Monte Carlo bootstrap (confiance sur les résultats)

Usage :
    # Tout lancer
    PYTHONPATH=. python backtest/run_robustness_donchian.py --no-show

    # Sensitivity uniquement
    PYTHONPATH=. python backtest/run_robustness_donchian.py --sensitivity --no-show

    # Walk-forward uniquement
    PYTHONPATH=. python backtest/run_robustness_donchian.py --walkforward --no-show

    # Monte Carlo uniquement
    PYTHONPATH=. python backtest/run_robustness_donchian.py --montecarlo --no-show
"""

from __future__ import annotations

import argparse
import logging
import math
import random
import sys
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path

import matplotlib

from backtest.data_loader import download_candles
from backtest.simulator_donchian import (
    DonchianConfig,
    DonchianEngine,
    DonchianResult,
    DonchianTrade,
    compute_donchian_metrics,
)

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("robustness")

# ── Config G (référence optimale) ─────────────────────────────────────────────

CONFIG_G = dict(
    initial_balance=1000.0,
    risk_percent=0.02,
    entry_period=30,
    exit_period=20,
    adx_period=14,
    adx_threshold=30.0,
    ema_period=200,
    use_ema_filter=False,
    atr_period=14,
    sl_atr_mult=3.0,
    use_donchian_exit=True,
    trail_atr_mult=3.0,
    max_positions=4,
    max_position_pct=0.25,
    cooldown_days=5,
    fee_pct=0.001,
    slippage_pct=0.0005,
    btc_regime_filter=True,
    btc_regime_period=200,
    btc_symbol="BTC-USD",
    excluded_pairs=["UNI-USD", "LINK-USD", "ADA-USD", "ARB-USD", "ATOM-USD"],
    allow_short=False,
    compound=True,
)

DONCHIAN_PAIRS = [
    "BTC-USD", "ETH-USD", "SOL-USD", "XRP-USD", "BNB-USD",
    "DOGE-USD", "ADA-USD", "AVAX-USD", "DOT-USD", "LINK-USD",
    "SUI-USD", "NEAR-USD", "LTC-USD", "ARB-USD", "OP-USD",
    "AAVE-USD", "UNI-USD", "ATOM-USD", "FIL-USD", "INJ-USD",
]

OUTPUT_DIR = Path(__file__).parent / "output"


# ═══════════════════════════════════════════════════════════════════════════════
#  UTILITAIRES
# ═══════════════════════════════════════════════════════════════════════════════


def _download_data(months: int = 72):
    """Télécharge les données 1d pour toutes les paires."""
    end = datetime.now(tz=timezone.utc)
    start = end - timedelta(days=months * 30)
    logger.info("📥 Téléchargement klines 1d — %s → %s", start.date(), end.date())

    candles_by_symbol: dict = {}
    for pair in DONCHIAN_PAIRS:
        try:
            candles = download_candles(pair, start, end, interval="1d")
            if candles:
                candles_by_symbol[pair] = candles
                logger.info("   ✅ %s : %d bougies", pair, len(candles))
        except ValueError as e:
            logger.warning("   ⚠️ %s : %s", pair, e)
    return candles_by_symbol


def _run_bt(candles: dict, overrides: dict | None = None) -> tuple[DonchianResult, dict]:
    """Lance un backtest avec Config G + overrides optionnels."""
    params = {**CONFIG_G}
    if overrides:
        params.update(overrides)
    cfg = DonchianConfig(**params)
    engine = DonchianEngine(candles, cfg, interval="1d")
    result = engine.run()
    metrics = compute_donchian_metrics(result)
    return result, metrics


def _filter_candles_by_dates(candles: dict, start_dt: datetime, end_dt: datetime) -> dict:
    """Filtre les bougies pour une fenêtre temporelle donnée."""
    start_ms = int(start_dt.timestamp() * 1000)
    end_ms = int(end_dt.timestamp() * 1000)
    filtered = {}
    for sym, bars in candles.items():
        sub = [c for c in bars if start_ms <= c.timestamp <= end_ms]
        if len(sub) > 50:  # Min bougies pour warmup
            filtered[sym] = sub
    return filtered


# ═══════════════════════════════════════════════════════════════════════════════
#  1. SENSITIVITY — Perturbation autour de Config G
# ═══════════════════════════════════════════════════════════════════════════════


def run_sensitivity(candles: dict) -> list[dict]:
    """Test de stabilité : perturbe chaque paramètre ±1-2 pas autour de Config G."""

    # Grille centrée sur Config G
    param_grid = {
        "entry_period":    [20, 25, 30, 35, 40, 55],
        "exit_period":     [10, 15, 20, 25],
        "adx_threshold":   [20.0, 25.0, 30.0, 35.0],
        "sl_atr_mult":     [2.0, 2.5, 3.0, 3.5, 4.0],
        "max_positions":   [2, 3, 4, 5, 6],
        "risk_percent":    [0.01, 0.015, 0.02, 0.03],
    }

    # Nombre total de combos (single-param variation = ~25 ; full grid réduit)
    # On fait une analyse 1-param-à-la-fois (OAT) pour clarté
    results = []

    # ── OAT : varier un paramètre, fixer les autres à Config G ──
    total = sum(len(v) for v in param_grid.values())
    logger.info("🔬 Sensitivity OAT : %d tests (1 param à la fois)", total)

    done = 0
    for param_name, values in param_grid.items():
        for val in values:
            # Skip les combos invalides
            overrides = {param_name: val}
            if param_name == "entry_period" and val <= CONFIG_G["exit_period"]:
                continue
            if param_name == "exit_period" and val >= CONFIG_G["entry_period"]:
                continue

            _, m = _run_bt(candles, overrides)
            results.append({
                "param": param_name,
                "value": val,
                "is_config_g": (val == CONFIG_G.get(param_name)),
                **{k: m.get(k) for k in [
                    "n_trades", "win_rate", "profit_factor",
                    "total_return", "cagr", "max_drawdown", "sharpe", "sortino",
                    "avg_pnl_usd", "avg_hold_days",
                ]},
            })
            done += 1
            if done % 5 == 0:
                logger.info("   %d/%d…", done, total)

    # ── Full grid réduit : entry × exit × adx × sl (pas risk/max_pos) ──
    entry_vals = [20, 30, 40]
    exit_vals = [10, 20]
    adx_vals = [25.0, 30.0]
    sl_vals = [2.5, 3.0, 3.5]

    grid_total = len(entry_vals) * len(exit_vals) * len(adx_vals) * len(sl_vals)
    logger.info("🔬 Sensitivity Grid : %d combos", grid_total)

    grid_results = []
    done = 0
    for entry_p in entry_vals:
        for exit_p in exit_vals:
            if exit_p >= entry_p:
                continue
            for adx_t in adx_vals:
                for sl_m in sl_vals:
                    overrides = {
                        "entry_period": entry_p,
                        "exit_period": exit_p,
                        "adx_threshold": adx_t,
                        "sl_atr_mult": sl_m,
                    }
                    _, m = _run_bt(candles, overrides)
                    grid_results.append({
                        "entry": entry_p,
                        "exit": exit_p,
                        "adx": adx_t,
                        "sl": sl_m,
                        **{k: m.get(k) for k in [
                            "n_trades", "win_rate", "profit_factor",
                            "total_return", "cagr", "max_drawdown", "sharpe",
                        ]},
                    })
                    done += 1

    return results, grid_results


def _print_sensitivity(oat_results: list[dict], grid_results: list[dict]) -> None:
    sep = "═" * 120
    print(f"\n{sep}")
    print("  🔬 1. SENSITIVITY — Perturbation OAT autour de Config G (5.8 ans)")
    print(sep)

    # Grouper par paramètre
    by_param = defaultdict(list)
    for r in oat_results:
        by_param[r["param"]].append(r)

    for param, rows in by_param.items():
        print(f"\n  ── {param} ──")
        print(f"  {'Value':>10s} | {'Trades':>6s} | {'WR':>6s} | {'PF':>6s} | "
              f"{'Return':>8s} | {'CAGR':>7s} | {'MaxDD':>7s} | {'Sharpe':>7s}")
        print(f"  {'─' * 80}")
        for r in sorted(rows, key=lambda x: x["value"]):
            marker = " ◀ G" if r["is_config_g"] else ""
            n = r["n_trades"] or 0
            pf_str = f"{r['profit_factor']:.2f}" if n >= 5 else "  n/a"
            print(
                f"  {str(r['value']):>10s} | {n:6d} | {r['win_rate']:6.1%} | "
                f"{pf_str:>6s} | {r['total_return']:+7.1%} | {r['cagr']:+6.1%} | "
                f"{r['max_drawdown']:7.1%} | {r['sharpe']:7.2f}{marker}"
            )

    # Robustesse : combien de voisins sont profitables ?
    profitable = sum(1 for r in oat_results if r["total_return"] > 0 and r["n_trades"] >= 5)
    total_valid = sum(1 for r in oat_results if r["n_trades"] >= 5)
    print(f"\n  📊 Robustesse OAT : {profitable}/{total_valid} configs voisines profitables "
          f"({profitable/total_valid:.0%})")

    # Sharpe moyen des voisins
    sharpes = [r["sharpe"] for r in oat_results if r["n_trades"] >= 5]
    if sharpes:
        avg_sharpe = sum(sharpes) / len(sharpes)
        min_sharpe = min(sharpes)
        max_sharpe = max(sharpes)
        print(f"  📊 Sharpe voisins : moy={avg_sharpe:.2f}, min={min_sharpe:.2f}, max={max_sharpe:.2f}")

    # Grid results
    print(f"\n  ── Grid réduit ({len(grid_results)} combos) ──")
    print(f"  {'Entry':>5s} | {'Exit':>4s} | {'ADX≥':>5s} | {'SL×':>4s} | "
          f"{'Trades':>6s} | {'WR':>6s} | {'PF':>6s} | "
          f"{'Return':>8s} | {'CAGR':>7s} | {'MaxDD':>7s} | {'Sharpe':>7s}")
    print(f"  {'─' * 95}")

    grid_sorted = sorted(grid_results, key=lambda x: x["sharpe"], reverse=True)
    for r in grid_sorted:
        n = r["n_trades"] or 0
        if n < 5:
            continue
        is_g = (r["entry"] == 30 and r["exit"] == 20 and
                r["adx"] == 30.0 and r["sl"] == 3.0)
        marker = " ◀ G" if is_g else ""
        print(
            f"  {r['entry']:5d} | {r['exit']:4d} | {r['adx']:5.0f} | {r['sl']:4.1f} | "
            f"{n:6d} | {r['win_rate']:6.1%} | {r['profit_factor']:6.2f} | "
            f"{r['total_return']:+7.1%} | {r['cagr']:+6.1%} | "
            f"{r['max_drawdown']:7.1%} | {r['sharpe']:7.2f}{marker}"
        )

    grid_profitable = sum(1 for r in grid_results if r["total_return"] > 0 and r["n_trades"] >= 5)
    grid_valid = sum(1 for r in grid_results if r["n_trades"] >= 5)
    print(f"\n  📊 Robustesse Grid : {grid_profitable}/{grid_valid} configs profitables "
          f"({grid_profitable/grid_valid:.0%})" if grid_valid else "")


# ═══════════════════════════════════════════════════════════════════════════════
#  2. WALK-FORWARD ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════════


def run_walkforward(candles: dict, train_months: int = 24, test_months: int = 12) -> list[dict]:
    """Walk-forward : entraîner sur N mois, tester sur les M suivants, rouler.

    Processus :
      1. Fenêtre in-sample (IS) : optimise entry_period et adx_threshold
      2. Fenêtre out-of-sample (OOS) : applique la meilleure config IS
      3. Avance de test_months et recommence
    """
    # Trouver la plage temporelle globale
    all_timestamps = set()
    for sym, bars in candles.items():
        for c in bars:
            all_timestamps.add(c.timestamp)
    if not all_timestamps:
        return []

    min_ts = min(all_timestamps)
    max_ts = max(all_timestamps)
    start_dt = datetime.fromtimestamp(min_ts / 1000, tz=timezone.utc)
    end_dt = datetime.fromtimestamp(max_ts / 1000, tz=timezone.utc)

    total_days = (end_dt - start_dt).days
    logger.info("🔄 Walk-Forward : %s → %s (%d jours)", start_dt.date(), end_dt.date(), total_days)
    logger.info("   Train=%d mois, Test=%d mois", train_months, test_months)

    # Paramètres à optimiser (grille réduite pour vitesse)
    opt_entry = [20, 25, 30, 40]
    opt_exit = [10, 15, 20]
    opt_adx = [25.0, 30.0, 35.0]
    opt_sl = [2.5, 3.0, 3.5]

    windows = []
    cursor = start_dt

    while cursor + timedelta(days=(train_months + test_months) * 30) <= end_dt:
        train_start = cursor
        train_end = cursor + timedelta(days=train_months * 30)
        test_start = train_end
        test_end = test_start + timedelta(days=test_months * 30)

        if test_end > end_dt:
            test_end = end_dt

        logger.info("   📐 Fenêtre : Train %s→%s | Test %s→%s",
                     train_start.date(), train_end.date(),
                     test_start.date(), test_end.date())

        # ── IN-SAMPLE : trouver la meilleure config ──
        train_candles = _filter_candles_by_dates(candles, train_start, train_end)
        if len(train_candles) < 3:
            logger.warning("      ⚠️ Pas assez de données train — skip")
            cursor += timedelta(days=test_months * 30)
            continue

        best_sharpe = -999
        best_params = {}
        best_metrics = {}

        for ep in opt_entry:
            for xp in opt_exit:
                if xp >= ep:
                    continue
                for adx in opt_adx:
                    for sl in opt_sl:
                        try:
                            _, m = _run_bt(train_candles, {
                                "entry_period": ep,
                                "exit_period": xp,
                                "adx_threshold": adx,
                                "sl_atr_mult": sl,
                            })
                            if m["n_trades"] < 5:
                                continue
                            score = m["sharpe"]
                            if score > best_sharpe:
                                best_sharpe = score
                                best_params = {"entry_period": ep, "exit_period": xp,
                                               "adx_threshold": adx, "sl_atr_mult": sl}
                                best_metrics = m
                        except Exception:
                            continue

        if not best_params:
            logger.warning("      ⚠️ Aucune config IS viable — skip")
            cursor += timedelta(days=test_months * 30)
            continue

        logger.info("      IS best: Entry=%d, Exit=%d, ADX≥%.0f, SL=%.1f | "
                     "Sharpe=%.2f, Return=%+.1f%%",
                     best_params["entry_period"], best_params["exit_period"],
                     best_params["adx_threshold"], best_params["sl_atr_mult"],
                     best_sharpe, best_metrics["total_return"] * 100)

        # ── OUT-OF-SAMPLE : appliquer la config IS ──
        test_candles = _filter_candles_by_dates(candles, test_start, test_end)
        if len(test_candles) < 3:
            logger.warning("      ⚠️ Pas assez de données test — skip")
            cursor += timedelta(days=test_months * 30)
            continue

        try:
            _, oos_m = _run_bt(test_candles, best_params)
        except Exception as e:
            logger.warning("      ⚠️ OOS error: %s", e)
            cursor += timedelta(days=test_months * 30)
            continue

        # Aussi tester Config G fixe sur OOS pour comparaison
        try:
            _, oos_g = _run_bt(test_candles)
        except Exception:
            oos_g = {"total_return": 0, "sharpe": 0, "max_drawdown": 0,
                     "profit_factor": 0, "n_trades": 0, "win_rate": 0}

        window = {
            "train_start": train_start,
            "train_end": train_end,
            "test_start": test_start,
            "test_end": test_end,
            "is_params": best_params,
            "is_sharpe": best_sharpe,
            "is_return": best_metrics["total_return"],
            "oos_return": oos_m["total_return"],
            "oos_sharpe": oos_m["sharpe"],
            "oos_maxdd": oos_m["max_drawdown"],
            "oos_pf": oos_m["profit_factor"],
            "oos_trades": oos_m["n_trades"],
            "oos_wr": oos_m["win_rate"],
            # Config G fixe sur OOS
            "oos_g_return": oos_g["total_return"],
            "oos_g_sharpe": oos_g["sharpe"],
            "oos_g_maxdd": oos_g["max_drawdown"],
            "oos_g_trades": oos_g["n_trades"],
        }
        windows.append(window)

        logger.info("      OOS optimisé: Return=%+.1f%%, Sharpe=%.2f, MaxDD=%.1f%%, Trades=%d",
                     oos_m["total_return"] * 100, oos_m["sharpe"],
                     oos_m["max_drawdown"] * 100, oos_m["n_trades"])
        logger.info("      OOS Config G: Return=%+.1f%%, Sharpe=%.2f, MaxDD=%.1f%%",
                     oos_g["total_return"] * 100, oos_g["sharpe"],
                     oos_g["max_drawdown"] * 100)

        cursor += timedelta(days=test_months * 30)

    return windows


def _print_walkforward(windows: list[dict]) -> None:
    sep = "═" * 130
    print(f"\n{sep}")
    print("  🔄 2. WALK-FORWARD ANALYSIS — Train 24 mois → Test 12 mois")
    print(sep)

    if not windows:
        print("  ⚠️ Pas assez de données pour le walk-forward")
        return

    print(f"\n  {'Fenêtre':>30s} | {'IS→OOS Config optimisée':^45s} | {'OOS Config G fixe':^30s}")
    print(f"  {'':>30s} | {'Return':>8s} {'Sharpe':>7s} {'MaxDD':>7s} {'PF':>6s} {'WR':>6s} {'N':>4s} | "
          f"{'Return':>8s} {'Sharpe':>7s} {'MaxDD':>7s} {'N':>4s}")
    print(f"  {'─' * 126}")

    for w in windows:
        label = f"{w['test_start'].date()} → {w['test_end'].date()}"
        params = f"E{w['is_params']['entry_period']}/X{w['is_params']['exit_period']}/A{w['is_params']['adx_threshold']:.0f}/S{w['is_params']['sl_atr_mult']:.1f}"

        n = w["oos_trades"]
        pf = w["oos_pf"] if n >= 3 else 0
        wr = w["oos_wr"] if n >= 3 else 0

        emoji = "🟢" if w["oos_return"] > 0 else "🔴"
        print(
            f"  {emoji} {label:>28s} | "
            f"{w['oos_return']:+7.1%} {w['oos_sharpe']:7.2f} {w['oos_maxdd']:7.1%} "
            f"{pf:6.2f} {wr:6.1%} {n:4d} | "
            f"{w['oos_g_return']:+7.1%} {w['oos_g_sharpe']:7.2f} {w['oos_g_maxdd']:7.1%} "
            f"{w['oos_g_trades']:4d}"
        )
        print(f"  {'':>3s} Params IS: {params}")

    # Résumé
    oos_returns = [w["oos_return"] for w in windows]
    oos_g_returns = [w["oos_g_return"] for w in windows]
    oos_sharpes = [w["oos_sharpe"] for w in windows]

    n_positive = sum(1 for r in oos_returns if r > 0)
    n_positive_g = sum(1 for r in oos_g_returns if r > 0)
    n_windows = len(windows)

    avg_oos = sum(oos_returns) / n_windows if n_windows else 0
    avg_oos_g = sum(oos_g_returns) / n_windows if n_windows else 0
    avg_sharpe = sum(oos_sharpes) / n_windows if n_windows else 0

    cumul_oos = 1.0
    for r in oos_returns:
        cumul_oos *= (1 + r)
    cumul_oos_g = 1.0
    for r in oos_g_returns:
        cumul_oos_g *= (1 + r)

    # Efficiency ratio = OOS Sharpe / IS Sharpe moyen
    is_sharpes = [w["is_sharpe"] for w in windows]
    avg_is_sharpe = sum(is_sharpes) / n_windows if n_windows else 1
    efficiency = avg_sharpe / avg_is_sharpe if avg_is_sharpe > 0 else 0

    print(f"\n  ── Résumé Walk-Forward ──")
    print(f"  Fenêtres OOS positives   : {n_positive}/{n_windows} ({n_positive/n_windows:.0%})")
    print(f"  Return OOS moyen (optim) : {avg_oos:+.1%}")
    print(f"  Return OOS moyen (cfg G) : {avg_oos_g:+.1%}")
    print(f"  Return cumulé OOS (optim): {(cumul_oos-1):+.1%}")
    print(f"  Return cumulé OOS (cfg G): {(cumul_oos_g-1):+.1%}")
    print(f"  Sharpe OOS moyen         : {avg_sharpe:.2f}")
    print(f"  Walk-Forward Efficiency  : {efficiency:.2f} (OOS/IS Sharpe)")
    print(f"  Config G stable ?        : {n_positive_g}/{n_windows} fenêtres positives")


# ═══════════════════════════════════════════════════════════════════════════════
#  3. MONTE CARLO BOOTSTRAP
# ═══════════════════════════════════════════════════════════════════════════════


def run_montecarlo(candles: dict, n_simulations: int = 2000) -> dict:
    """Monte Carlo par bootstrap des trades.

    Processus :
      1. Lancer le backtest Config G → obtenir la liste de trades
      2. Bootstrap : rééchantillonner N trades avec remplacement, N = len(trades)
      3. Recalculer equity curve, return, MaxDD, Sharpe pour chaque simulation
      4. Extraire les percentiles (5%, 25%, 50%, 75%, 95%)
    """
    # Backtest de référence
    result, ref_metrics = _run_bt(candles)
    trades = result.trades

    if len(trades) < 10:
        logger.warning("⚠️ Seulement %d trades — Monte Carlo peu fiable", len(trades))
        return {"error": "too_few_trades", "ref": ref_metrics}

    n_trades = len(trades)
    init = result.config.initial_balance

    logger.info("🎲 Monte Carlo : %d simulations, %d trades, capital $%.0f",
                n_simulations, n_trades, init)

    sim_returns = []
    sim_maxdds = []
    sim_sharpes = []
    sim_final_eq = []
    sim_max_consec_loss = []
    sim_equity_curves = []

    random.seed(42)  # Reproductibilité

    for sim_i in range(n_simulations):
        # Bootstrap : tirer n_trades trades avec remplacement
        boot_trades = random.choices(trades, k=n_trades)

        # Recalculer l'equity curve
        equity = init
        peak = init
        max_dd = 0.0
        daily_returns = []
        consecutive_losses = 0
        max_consecutive_losses = 0
        curve = [equity]

        for t in boot_trades:
            pnl_pct = t.pnl_pct
            prev_eq = equity
            # Appliquer le PnL proportionnel (compound)
            equity *= (1 + pnl_pct)
            curve.append(equity)

            # Daily return approximation
            daily_returns.append(pnl_pct)

            # Drawdown
            peak = max(peak, equity)
            dd = (equity - peak) / peak if peak > 0 else 0
            max_dd = min(max_dd, dd)

            # Consecutive losses
            if pnl_pct < 0:
                consecutive_losses += 1
                max_consecutive_losses = max(max_consecutive_losses, consecutive_losses)
            else:
                consecutive_losses = 0

        total_return = (equity - init) / init
        sim_returns.append(total_return)
        sim_maxdds.append(max_dd)
        sim_final_eq.append(equity)
        sim_max_consec_loss.append(max_consecutive_losses)

        # Sharpe approximatif
        if len(daily_returns) >= 2:
            mu = sum(daily_returns) / len(daily_returns)
            var = sum((r - mu) ** 2 for r in daily_returns) / (len(daily_returns) - 1)
            std = math.sqrt(var) if var > 0 else 1e-9
            sharpe = (mu / std) * math.sqrt(n_trades)  # Annualisé approximatif
        else:
            sharpe = 0
        sim_sharpes.append(sharpe)

        if sim_i < 200:  # Garder 200 curves pour le chart
            sim_equity_curves.append(curve)

    # Percentiles
    def _pct(data: list, p: float) -> float:
        s = sorted(data)
        idx = int(len(s) * p / 100)
        return s[min(idx, len(s) - 1)]

    return {
        "ref": ref_metrics,
        "n_simulations": n_simulations,
        "n_trades": n_trades,
        "returns": {
            "p5": _pct(sim_returns, 5),
            "p25": _pct(sim_returns, 25),
            "p50": _pct(sim_returns, 50),
            "p75": _pct(sim_returns, 75),
            "p95": _pct(sim_returns, 95),
            "mean": sum(sim_returns) / len(sim_returns),
        },
        "maxdd": {
            "p5": _pct(sim_maxdds, 5),
            "p25": _pct(sim_maxdds, 25),
            "p50": _pct(sim_maxdds, 50),
            "p75": _pct(sim_maxdds, 75),
            "p95": _pct(sim_maxdds, 95),
            "mean": sum(sim_maxdds) / len(sim_maxdds),
        },
        "sharpe": {
            "p5": _pct(sim_sharpes, 5),
            "p25": _pct(sim_sharpes, 25),
            "p50": _pct(sim_sharpes, 50),
            "p75": _pct(sim_sharpes, 75),
            "p95": _pct(sim_sharpes, 95),
            "mean": sum(sim_sharpes) / len(sim_sharpes),
        },
        "final_equity": {
            "p5": _pct(sim_final_eq, 5),
            "p25": _pct(sim_final_eq, 25),
            "p50": _pct(sim_final_eq, 50),
            "p75": _pct(sim_final_eq, 75),
            "p95": _pct(sim_final_eq, 95),
            "mean": sum(sim_final_eq) / len(sim_final_eq),
        },
        "max_consec_loss": {
            "p50": _pct(sim_max_consec_loss, 50),
            "p75": _pct(sim_max_consec_loss, 75),
            "p95": _pct(sim_max_consec_loss, 95),
            "max": max(sim_max_consec_loss),
        },
        "prob_profit": sum(1 for r in sim_returns if r > 0) / len(sim_returns),
        "prob_double": sum(1 for r in sim_returns if r > 1.0) / len(sim_returns),
        "prob_dd30": sum(1 for d in sim_maxdds if d < -0.30) / len(sim_maxdds),
        "equity_curves": sim_equity_curves,
        "all_returns": sim_returns,
        "all_maxdds": sim_maxdds,
    }


def _print_montecarlo(mc: dict) -> None:
    sep = "═" * 90
    print(f"\n{sep}")
    print(f"  🎲 3. MONTE CARLO BOOTSTRAP — {mc['n_simulations']} simulations, {mc['n_trades']} trades")
    print(sep)

    if "error" in mc:
        print(f"  ⚠️ {mc['error']}")
        return

    ref = mc["ref"]

    print(f"\n  Référence Config G :")
    print(f"     Return  : {ref['total_return']:+.1%}")
    print(f"     MaxDD   : {ref['max_drawdown']:.1%}")
    print(f"     Sharpe  : {ref['sharpe']:.2f}")
    print(f"     Trades  : {ref['n_trades']}")

    print(f"\n  {'Métrique':>20s} |   P5   |  P25   |  P50   |  P75   |  P95   |  Mean  |  Réf  ")
    print(f"  {'─' * 85}")

    def _row(name: str, data: dict, ref_val: float, fmt: str = "{:+.1%}"):
        vals = [data["p5"], data["p25"], data["p50"], data["p75"], data["p95"], data["mean"]]
        cells = " | ".join(f"{fmt.format(v):>6s}" for v in vals)
        print(f"  {name:>20s} | {cells} | {fmt.format(ref_val):>6s}")

    _row("Return", mc["returns"], ref["total_return"])
    _row("Max Drawdown", mc["maxdd"], ref["max_drawdown"])
    _row("Sharpe", mc["sharpe"], ref["sharpe"], "{:.2f}")

    feq = mc["final_equity"]
    print(f"\n  Equity finale ($1000 initial) :")
    print(f"     P5  = ${feq['p5']:,.0f} | P25 = ${feq['p25']:,.0f} | "
          f"P50 = ${feq['p50']:,.0f} | P75 = ${feq['p75']:,.0f} | P95 = ${feq['p95']:,.0f}")

    cl = mc["max_consec_loss"]
    print(f"\n  Série de pertes max consécutives :")
    print(f"     Médiane = {cl['p50']:.0f} | P75 = {cl['p75']:.0f} | "
          f"P95 = {cl['p95']:.0f} | Max = {cl['max']:.0f}")

    print(f"\n  Probabilités :")
    print(f"     P(profit > 0)   : {mc['prob_profit']:.1%}")
    print(f"     P(return > 100%): {mc['prob_double']:.1%}")
    print(f"     P(DD > 30%)     : {mc['prob_dd30']:.1%}")

    # Intervalle de confiance 90%
    r = mc["returns"]
    print(f"\n  📊 Intervalle de confiance 90% du return : [{r['p5']:+.1%}, {r['p95']:+.1%}]")
    d = mc["maxdd"]
    print(f"  📊 Intervalle de confiance 90% du MaxDD  : [{d['p5']:.1%}, {d['p95']:.1%}]")


# ═══════════════════════════════════════════════════════════════════════════════
#  CHARTS
# ═══════════════════════════════════════════════════════════════════════════════


def _generate_charts(mc: dict, wf: list[dict], show: bool = True) -> None:
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mtick

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(18, 12))
    fig.suptitle("Donchian Robustness — Monte Carlo & Walk-Forward", fontsize=14, fontweight="bold")

    # ── 1. Monte Carlo equity fan ──
    ax = axes[0][0]
    if mc and "equity_curves" in mc:
        curves = mc["equity_curves"]
        n_show = min(len(curves), 200)
        for i in range(n_show):
            alpha = 0.03
            ax.plot(curves[i], color="steelblue", alpha=alpha, linewidth=0.5)

        # Percentiles
        max_len = max(len(c) for c in curves[:n_show])
        p5_curve, p50_curve, p95_curve = [], [], []
        for j in range(max_len):
            vals = [c[j] for c in curves[:n_show] if j < len(c)]
            vals.sort()
            n = len(vals)
            p5_curve.append(vals[int(n * 0.05)])
            p50_curve.append(vals[int(n * 0.50)])
            p95_curve.append(vals[int(n * 0.95)])

        x = list(range(max_len))
        ax.plot(x, p5_curve, "r--", linewidth=1.5, label="P5")
        ax.plot(x, p50_curve, "b-", linewidth=2, label="P50 (médiane)")
        ax.plot(x, p95_curve, "g--", linewidth=1.5, label="P95")
        ax.axhline(1000, color="gray", linestyle=":", alpha=0.5)
        ax.fill_between(x, p5_curve, p95_curve, alpha=0.08, color="steelblue")

    ax.set_title(f"Monte Carlo Equity ({mc['n_simulations']} sims)")
    ax.set_ylabel("Equity ($)")
    ax.set_xlabel("Trade #")
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.3)

    # ── 2. Distribution des returns ──
    ax = axes[0][1]
    if mc and "all_returns" in mc:
        returns = mc["all_returns"]
        ax.hist(returns, bins=80, color="steelblue", alpha=0.7, edgecolor="white", linewidth=0.5)
        ref_ret = mc["ref"]["total_return"]
        ax.axvline(ref_ret, color="red", linewidth=2, linestyle="--", label=f"Réf: {ref_ret:+.0%}")
        ax.axvline(0, color="black", linewidth=1, linestyle="-", alpha=0.5)
        # P5/P95
        p5 = mc["returns"]["p5"]
        p95 = mc["returns"]["p95"]
        ax.axvline(p5, color="orange", linewidth=1.5, linestyle=":", label=f"P5: {p5:+.0%}")
        ax.axvline(p95, color="green", linewidth=1.5, linestyle=":", label=f"P95: {p95:+.0%}")
    ax.set_title("Distribution Monte Carlo des Returns")
    ax.set_xlabel("Return total")
    ax.xaxis.set_major_formatter(mtick.PercentFormatter(1.0))
    ax.legend()
    ax.grid(True, alpha=0.3)

    # ── 3. Distribution des MaxDD ──
    ax = axes[1][0]
    if mc and "all_maxdds" in mc:
        dds = mc["all_maxdds"]
        ax.hist(dds, bins=60, color="salmon", alpha=0.7, edgecolor="white", linewidth=0.5)
        ref_dd = mc["ref"]["max_drawdown"]
        ax.axvline(ref_dd, color="red", linewidth=2, linestyle="--", label=f"Réf: {ref_dd:.0%}")
        ax.axvline(-0.30, color="black", linewidth=1, linestyle=":", alpha=0.5, label="Seuil -30%")
    ax.set_title("Distribution Monte Carlo du Max Drawdown")
    ax.set_xlabel("Max Drawdown")
    ax.xaxis.set_major_formatter(mtick.PercentFormatter(1.0))
    ax.legend()
    ax.grid(True, alpha=0.3)

    # ── 4. Walk-Forward OOS returns ──
    ax = axes[1][1]
    if wf:
        labels = [f"{w['test_start'].strftime('%y/%m')}" for w in wf]
        oos_ret = [w["oos_return"] * 100 for w in wf]
        oos_g_ret = [w["oos_g_return"] * 100 for w in wf]

        x = range(len(wf))
        width = 0.35
        bars1 = ax.bar([i - width/2 for i in x], oos_ret, width, label="WF Optimisé",
                       color=["green" if r > 0 else "red" for r in oos_ret], alpha=0.7)
        bars2 = ax.bar([i + width/2 for i in x], oos_g_ret, width, label="Config G fixe",
                       color=["darkgreen" if r > 0 else "darkred" for r in oos_g_ret], alpha=0.5)

        ax.set_xticks(list(x))
        ax.set_xticklabels(labels, rotation=45)
        ax.axhline(0, color="black", linewidth=1)
    ax.set_title("Walk-Forward OOS Returns (%)")
    ax.set_ylabel("Return (%)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    outpath = OUTPUT_DIR / "donchian_robustness.png"
    plt.savefig(outpath, dpi=150)
    logger.info("📊 Chart → %s", outpath)
    if show:
        plt.show()
    plt.close()


# ═══════════════════════════════════════════════════════════════════════════════
#  4. SYNTHÈSE FINALE
# ═══════════════════════════════════════════════════════════════════════════════


def _print_synthesis(oat_results, grid_results, wf_windows, mc_results) -> None:
    sep = "═" * 90
    print(f"\n{sep}")
    print("  📋 SYNTHÈSE DE ROBUSTESSE — Donchian Trend Following")
    print(sep)

    scores = []

    # 1. Sensitivity
    if oat_results:
        valid = [r for r in oat_results if r["n_trades"] >= 5]
        profitable = sum(1 for r in valid if r["total_return"] > 0)
        pct = profitable / len(valid) if valid else 0
        sharpes = [r["sharpe"] for r in valid]
        avg_s = sum(sharpes) / len(sharpes) if sharpes else 0
        score_sens = "✅" if pct >= 0.8 else ("⚠️" if pct >= 0.6 else "❌")
        scores.append(("Sensitivity OAT", f"{profitable}/{len(valid)} ({pct:.0%}) profitables, Sharpe moy={avg_s:.2f}", score_sens))

    if grid_results:
        valid = [r for r in grid_results if r["n_trades"] >= 5]
        profitable = sum(1 for r in valid if r["total_return"] > 0)
        pct = profitable / len(valid) if valid else 0
        score_grid = "✅" if pct >= 0.8 else ("⚠️" if pct >= 0.6 else "❌")
        scores.append(("Sensitivity Grid", f"{profitable}/{len(valid)} ({pct:.0%}) profitables", score_grid))

    # 2. Walk-Forward
    if wf_windows:
        n = len(wf_windows)
        n_pos = sum(1 for w in wf_windows if w["oos_return"] > 0)
        n_pos_g = sum(1 for w in wf_windows if w["oos_g_return"] > 0)
        avg_ret = sum(w["oos_return"] for w in wf_windows) / n
        avg_ret_g = sum(w["oos_g_return"] for w in wf_windows) / n

        is_sharpes = [w["is_sharpe"] for w in wf_windows]
        oos_sharpes = [w["oos_sharpe"] for w in wf_windows]
        avg_is_s = sum(is_sharpes) / n
        avg_oos_s = sum(oos_sharpes) / n
        efficiency = avg_oos_s / avg_is_s if avg_is_s > 0 else 0

        score_wf = "✅" if n_pos_g >= n * 0.6 else ("⚠️" if n_pos_g >= n * 0.4 else "❌")
        scores.append(("Walk-Forward (Config G)", f"{n_pos_g}/{n} OOS positives, ret moy={avg_ret_g:+.1%}", score_wf))

        score_eff = "✅" if efficiency > 0.5 else ("⚠️" if efficiency > 0.25 else "❌")
        scores.append(("WF Efficiency", f"{efficiency:.2f} (Sharpe OOS/IS)", score_eff))

    # 3. Monte Carlo
    if mc_results and "error" not in mc_results:
        prob_profit = mc_results["prob_profit"]
        p5_return = mc_results["returns"]["p5"]
        p95_dd = mc_results["maxdd"]["p95"]
        median_return = mc_results["returns"]["p50"]

        score_mc = "✅" if prob_profit >= 0.80 else ("⚠️" if prob_profit >= 0.60 else "❌")
        scores.append(("Monte Carlo P(profit)", f"{prob_profit:.1%}", score_mc))

        score_p5 = "✅" if p5_return > 0 else ("⚠️" if p5_return > -0.20 else "❌")
        scores.append(("MC Return P5 (worst 5%)", f"{p5_return:+.1%}", score_p5))

        score_dd = "✅" if p95_dd > -0.40 else ("⚠️" if p95_dd > -0.50 else "❌")
        scores.append(("MC MaxDD P95", f"{p95_dd:.1%}", score_dd))

    # Affichage
    print()
    for name, detail, status in scores:
        print(f"  {status} {name:30s} : {detail}")

    n_pass = sum(1 for _, _, s in scores if s == "✅")
    n_warn = sum(1 for _, _, s in scores if s == "⚠️")
    n_fail = sum(1 for _, _, s in scores if s == "❌")
    total = len(scores)

    print(f"\n  {'─' * 60}")
    overall = "✅ ROBUSTE" if n_pass >= total * 0.7 and n_fail == 0 else (
        "⚠️ PARTIELLEMENT ROBUSTE" if n_fail <= 1 else "❌ NON ROBUSTE")
    print(f"  Verdict : {overall} ({n_pass}✅ {n_warn}⚠️ {n_fail}❌ / {total} critères)")
    print(f"\n{sep}\n")


# ═══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════════════════


def main() -> None:
    parser = argparse.ArgumentParser(description="Robustness analysis — Donchian (Daily)")
    parser.add_argument("--months", type=int, default=72, help="Mois de données (défaut=72 ≈ 6 ans)")
    parser.add_argument("--sensitivity", action="store_true", help="Sensitivity uniquement")
    parser.add_argument("--walkforward", action="store_true", help="Walk-forward uniquement")
    parser.add_argument("--montecarlo", action="store_true", help="Monte Carlo uniquement")
    parser.add_argument("--mc-sims", type=int, default=2000, help="Nombre de simulations MC")
    parser.add_argument("--wf-train", type=int, default=24, help="Mois train WF")
    parser.add_argument("--wf-test", type=int, default=12, help="Mois test WF")
    parser.add_argument("--no-show", action="store_true")
    args = parser.parse_args()

    if args.no_show:
        matplotlib.use("Agg")

    run_all = not (args.sensitivity or args.walkforward or args.montecarlo)

    # Données
    candles = _download_data(args.months)
    if not candles:
        logger.error("❌ Aucune donnée — abandon")
        sys.exit(1)

    logger.info("📦 %d paires chargées, %d–%d bougies",
                len(candles),
                min(len(v) for v in candles.values()),
                max(len(v) for v in candles.values()))

    oat_results = []
    grid_results = []
    wf_windows = []
    mc_results = {}

    # ── 1. Sensitivity ──
    if args.sensitivity or run_all:
        logger.info("\n" + "═" * 60)
        logger.info("  🔬 1/3 — SENSITIVITY ANALYSIS")
        logger.info("═" * 60)
        oat_results, grid_results = run_sensitivity(candles)
        _print_sensitivity(oat_results, grid_results)

    # ── 2. Walk-Forward ──
    if args.walkforward or run_all:
        logger.info("\n" + "═" * 60)
        logger.info("  🔄 2/3 — WALK-FORWARD ANALYSIS")
        logger.info("═" * 60)
        wf_windows = run_walkforward(candles, args.wf_train, args.wf_test)
        _print_walkforward(wf_windows)

    # ── 3. Monte Carlo ──
    if args.montecarlo or run_all:
        logger.info("\n" + "═" * 60)
        logger.info("  🎲 3/3 — MONTE CARLO BOOTSTRAP")
        logger.info("═" * 60)
        mc_results = run_montecarlo(candles, args.mc_sims)
        _print_montecarlo(mc_results)

    # ── Synthèse ──
    if run_all:
        _print_synthesis(oat_results, grid_results, wf_windows, mc_results)

    # ── Charts ──
    if mc_results or wf_windows:
        _generate_charts(mc_results, wf_windows, show=not args.no_show)

    logger.info("✅ Analyse de robustesse terminée")


if __name__ == "__main__":
    main()
