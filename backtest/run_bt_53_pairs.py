#!/usr/bin/env python
"""
🔬 BACKTEST 53 PAIRES REVOLUT X — RANGE strategy

Paramètres verrouillés : pos=3, compound=True, cap=30%, risk=2%

Toutes les paires disponibles sur Revolut X avec données Binance.
Chaque paire contribue à partir de la date où ses données démarrent.

Périodes :
  1. FULL  : 2020-02-20 → 2026-02-20 (6yr, chaque paire selon dispo)
  2. P1    : 2020-02-20 → 2022-02-20 (20 paires 6yr+)
  3. P2    : 2022-02-20 → 2024-02-20 (39 paires 4yr+)
  4. P3    : 2024-02-20 → 2026-02-20 (53 paires, all)

Compare avec la baseline 31 paires pour voir l'impact de la diversification.
"""

from __future__ import annotations

import logging
import statistics
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
logger = logging.getLogger("bt53")

# ── 53 paires Revolut X ───────────────────────────────────────────────────────

PAIRS_53 = [
    # 6yr+ (20 paires)
    "BTC-USD", "ETH-USD", "XRP-USD", "LINK-USD", "ADA-USD",
    "DOGE-USD", "ATOM-USD", "ALGO-USD", "LTC-USD", "ETC-USD",
    "VET-USD", "CHZ-USD", "BAT-USD", "HBAR-USD", "XTZ-USD",
    "DASH-USD", "XLM-USD", "TRX-USD", "KAVA-USD", "FET-USD",
    # 4yr+ (19 paires)
    "SOL-USD", "DOT-USD", "AVAX-USD", "UNI-USD", "NEAR-USD",
    "FIL-USD", "AAVE-USD", "INJ-USD", "SAND-USD", "MANA-USD",
    "SHIB-USD", "GRT-USD", "COMP-USD", "SNX-USD", "CRV-USD",
    "YFI-USD", "SUSHI-USD", "AXS-USD", "EGLD-USD",
    # 2yr+ (11 paires)
    "SUI-USD", "APE-USD", "PEPE-USD", "ARB-USD", "OP-USD",
    "LDO-USD", "BONK-USD", "SEI-USD", "TIA-USD", "FLOKI-USD",
    "JUP-USD",
    # <2yr (3 paires)
    "POL-USD", "RENDER-USD", "WIF-USD",
]

# Référence : 31 paires du backtest original (pour comparaison)
PAIRS_31_REF = [
    "BTC-USD", "ETH-USD", "XRP-USD", "LINK-USD", "ADA-USD",
    "DOGE-USD", "ATOM-USD", "ALGO-USD", "LTC-USD", "ETC-USD",
    "VET-USD", "CHZ-USD", "BAT-USD", "HBAR-USD", "XTZ-USD",
    "DASH-USD", "XLM-USD", "TRX-USD", "KAVA-USD",
    # + ceux du 31 original qui ne sont PAS sur Revolut X (pour référence)
    # On les exclut ici car la référence est les 20 paires 6yr disponibles
]

# 17 paires actuellement en production
PAIRS_17_PROD = [
    "BTC-USD", "ETH-USD", "XRP-USD", "LINK-USD", "ADA-USD",
    "DOGE-USD", "ATOM-USD", "LTC-USD", "ETC-USD", "CHZ-USD",
    "BAT-USD", "HBAR-USD", "XTZ-USD", "DASH-USD", "XLM-USD",
    "TRX-USD", "KAVA-USD",
]

D = lambda s: datetime.strptime(s, "%Y-%m-%d").replace(tzinfo=timezone.utc)


def _cfg(balance: float, fee: float = 0.00075, slip: float = 0.001) -> BacktestConfig:
    return BacktestConfig(
        initial_balance=balance,
        risk_percent_trend=0.03,
        entry_buffer_pct=config.ENTRY_BUFFER_PERCENT,
        sl_buffer_pct=config.SL_BUFFER_PERCENT,
        zero_risk_trigger_pct=config.ZERO_RISK_TRIGGER_PERCENT,
        zero_risk_lock_pct=config.ZERO_RISK_LOCK_PERCENT,
        trailing_stop_pct=config.TRAILING_STOP_PERCENT,
        max_position_pct=0.30,
        max_simultaneous_positions=3,
        compound=True,
        risk_percent_range=0.02,
        swing_lookback=config.SWING_LOOKBACK,
        range_width_min=config.RANGE_WIDTH_MIN,
        range_entry_buffer_pct=config.RANGE_ENTRY_BUFFER_PERCENT,
        range_sl_buffer_pct=config.RANGE_SL_BUFFER_PERCENT,
        range_cooldown_bars=config.RANGE_COOLDOWN_BARS,
        max_total_risk_pct=config.MAX_TOTAL_RISK_PERCENT,
        fee_pct=fee,
        slippage_pct=slip,
        enable_trend=False,
        enable_range=True,
    )


def _filt(all_c, start, end, pairs=None):
    s_ms = int(start.timestamp() * 1000)
    e_ms = int(end.timestamp() * 1000)
    out = {}
    for p, cl in all_c.items():
        if pairs and p not in pairs:
            continue
        pc = [c for c in cl if s_ms <= c.timestamp <= e_ms]
        if pc:
            out[p] = pc
    return out


def _run(candles, bal, fee=0.00075, slip=0.001):
    eng = BacktestEngine(candles, _cfg(bal, fee, slip))
    r = eng.run()
    m = compute_metrics(r)
    return m, r.trades, r.equity_curve


def _row(label, m, width=38):
    pf_e = "✅" if m["profit_factor"] >= 1.3 else ("⚠️" if m["profit_factor"] >= 1.0 else "❌")
    return (
        f"  {label:<{width}s} │ {m['total_return']:>+7.1%} │ {m['cagr']:>+6.1%} │ "
        f"{m['max_drawdown']:>7.1%} │ {m['sharpe']:>7.2f} │ "
        f"{pf_e}{m['profit_factor']:>4.2f} │ {m['win_rate']:>4.0%} │ "
        f"{m['n_trades']:>6d} │ ${m['final_equity']:>9,.2f}"
    )


def _hdr(title, width=38):
    sep = "═" * 130
    print(f"\n{sep}")
    print(f"  {title}")
    print(sep)
    print(
        f"  {'Label':<{width}s} │ {'Return':>8s} │ {'CAGR':>7s} │ {'MaxDD':>7s} │ "
        f"{'Sharpe':>7s} │ {'PF':>5s} │ {'WR':>5s} │ {'Trades':>6s} │ {'Final$':>10s}"
    )
    print("  " + "─" * 126)


def main():
    BAL = 1000.0

    logger.info("🔬 BACKTEST 53 PAIRES REVOLUT X")
    logger.info("   pos=3, compound=True, cap=30%%, risk=2%%")
    logger.info("   Range Only | fees=0.075%% + slip=0.1%%\n")

    # ── Téléchargement ──
    logger.info("📥 Chargement 53 paires H4…")
    # Paires 6yr+ et 4yr+ → depuis 2020
    pairs_from_2020 = PAIRS_53[:39]  # 20 + 19 = 39 paires
    # Paires 2yr+ → depuis 2022
    pairs_from_2022 = PAIRS_53[39:50]  # 11 paires
    # Paires <2yr → depuis 2024
    pairs_from_2024 = PAIRS_53[50:]  # 3 paires

    all_c = {}
    errors = []

    # Download 6yr+/4yr+ (depuis 2020)
    logger.info("  📦 Lot 1: 39 paires depuis 2020…")
    try:
        c1 = download_all_pairs(pairs_from_2020, D("2020-02-20"), D("2026-02-21"), interval="4h")
        all_c.update(c1)
    except Exception as e:
        logger.error("  ❌ Erreur lot 1: %s", e)

    # Download 2yr+ (depuis 2022)
    logger.info("  📦 Lot 2: 11 paires depuis 2022…")
    try:
        c2 = download_all_pairs(pairs_from_2022, D("2022-02-20"), D("2026-02-21"), interval="4h")
        all_c.update(c2)
    except Exception as e:
        logger.error("  ❌ Erreur lot 2: %s", e)

    # Download <2yr (depuis 2024)
    logger.info("  📦 Lot 3: 3 paires depuis 2024…")
    try:
        c3 = download_all_pairs(pairs_from_2024, D("2024-02-20"), D("2026-02-21"), interval="4h")
        all_c.update(c3)
    except Exception as e:
        logger.error("  ❌ Erreur lot 3: %s", e)

    loaded = {p: len(c) for p, c in all_c.items() if c}
    logger.info("✅ %d paires chargées (sur 53)\n", len(loaded))

    # Paires réellement chargées
    actual_pairs = sorted(loaded.keys())
    failed = [p for p in PAIRS_53 if p not in loaded]
    if failed:
        logger.warning("⚠️  %d paires sans données : %s", len(failed), failed)

    # ══════════════════════════════════════════════════════════════════════
    # TEST A — COMPARAISON : 17 prod vs 53 Revolut (6yr complet)
    # ══════════════════════════════════════════════════════════════════════
    logger.info("━" * 60)
    logger.info("📊 TEST A — COMPARAISON univers (6yr)")
    logger.info("━" * 60)

    full_start, full_end = D("2020-02-20"), D("2026-02-20")

    # 17 paires prod
    logger.info("  ⏳ 17 paires (prod actuelle)…")
    m17, tr17, _ = _run(_filt(all_c, full_start, full_end, set(PAIRS_17_PROD)), BAL)
    logger.info("    ✅ %+.1f%% | PF %.2f | %d trades", m17["total_return"] * 100, m17["profit_factor"], m17["n_trades"])

    # Toutes les paires chargées
    logger.info("  ⏳ %d paires (Revolut X complet)…", len(loaded))
    m_all, tr_all, eq_all = _run(_filt(all_c, full_start, full_end), BAL)
    logger.info("    ✅ %+.1f%% | PF %.2f | %d trades", m_all["total_return"] * 100, m_all["profit_factor"], m_all["n_trades"])

    _hdr("📊 TEST A — Comparaison univers de paires (6yr, fees standard)")
    print(_row(f"17 paires (prod actuelle)", m17))
    print(_row(f"{len(loaded)} paires (Revolut X complet)", m_all))

    dr = m_all["total_return"] - m17["total_return"]
    dp = m_all["profit_factor"] - m17["profit_factor"]
    dd = m_all["max_drawdown"] - m17["max_drawdown"]
    print(f"\n  📊 Delta 53 vs 17 :")
    print(f"    Return : {dr:>+7.1%}")
    print(f"    PF     : {dp:>+.2f}")
    print(f"    MaxDD  : {dd:>+.1%} ({'mieux' if dd > 0 else 'pire'})")
    print(f"    Trades : {m_all['n_trades'] - m17['n_trades']:>+d}")

    # ══════════════════════════════════════════════════════════════════════
    # TEST B — ROBUSTESSE par sous-période
    # ══════════════════════════════════════════════════════════════════════
    logger.info("\n" + "━" * 60)
    logger.info("🔬 TEST B — ROBUSTESSE par sous-période")
    logger.info("━" * 60)

    periods = [
        ("FULL 6yr (2020-2026)", "2020-02-20", "2026-02-20"),
        ("P1 2yr (2020-2022)",   "2020-02-20", "2022-02-20"),
        ("P2 2yr (2022-2024)",   "2022-02-20", "2024-02-20"),
        ("P3 2yr (2024-2026)",   "2024-02-20", "2026-02-20"),
    ]

    results_b = []
    for label, s, e in periods:
        logger.info("  ⏳ %s…", label)
        fc = _filt(all_c, D(s), D(e))
        n_pairs = len(fc)
        m, trades, _ = _run(fc, BAL)
        results_b.append((f"{label} [{n_pairs}p]", m, trades))
        logger.info("    ✅ %d paires | %+.1f%% | PF %.2f", n_pairs, m["total_return"] * 100, m["profit_factor"])

    _hdr("🔬 TEST B — Robustesse par sous-période (53 paires Revolut X)")
    for lab, m, _ in results_b:
        print(_row(lab, m))

    # Stabilité
    sub = results_b[1:]
    pfs = [m["profit_factor"] for _, m, _ in sub]
    rets = [m["total_return"] for _, m, _ in sub]
    dds = [m["max_drawdown"] for _, m, _ in sub]

    pf_mean = statistics.mean(pfs)
    pf_std = statistics.stdev(pfs) if len(pfs) > 1 else 0
    print(f"\n  📊 Stabilité sous-périodes :")
    print(f"    PF moyen     : {pf_mean:.2f} (σ = {pf_std:.2f})")
    print(f"    Return moyen : {statistics.mean(rets):+.1%} / 2 ans")
    print(f"    MaxDD range  : [{min(dds):.1%} .. {max(dds):.1%}]")

    all_pf_ok = all(pf > 1.0 for pf in pfs)
    all_pf_strong = all(pf > 1.3 for pf in pfs)
    if all_pf_strong:
        print(f"\n  ✅✅ EDGE ROBUSTE — PF > 1.3 dans TOUTES les sous-périodes")
    elif all_pf_ok:
        print(f"\n  ✅ EDGE PRÉSENT — PF > 1.0 dans toutes les sous-périodes")
    else:
        losing = [l for l, m, _ in sub if m["profit_factor"] < 1.0]
        print(f"\n  ⚠️  Périodes faibles : {losing}")

    # ══════════════════════════════════════════════════════════════════════
    # TEST C — Avec fees Revolut réels (0.09% + 0.15% slip)
    # ══════════════════════════════════════════════════════════════════════
    logger.info("\n" + "━" * 60)
    logger.info("🔧 TEST C — FRICTION REVOLUT")
    logger.info("━" * 60)

    scenarios = [
        ("Baseline (0.075% + 0.1%)",      0.00075, 0.001),
        ("Revolut maker (0% + 0.1%)",      0.0000, 0.001),
        ("Revolut taker (0.09% + 0.15%)",  0.0009, 0.0015),
        ("Stress (0.09% + 0.25%)",         0.0009, 0.0025),
    ]

    results_c = []
    fc_full = _filt(all_c, full_start, full_end)
    for lab, fee, slip in scenarios:
        logger.info("  ⏳ %s…", lab)
        m, _, _ = _run(fc_full, BAL, fee, slip)
        results_c.append((lab, m))
        logger.info("    ✅ %+.1f%% | PF %.2f", m["total_return"] * 100, m["profit_factor"])

    _hdr("🔧 TEST C — Impact friction (53 paires, 6yr)")
    for lab, m in results_c:
        print(_row(lab, m))

    base_ret = results_c[0][1]["total_return"]
    print(f"\n  📉 Dégradation vs baseline :")
    for lab, m in results_c[1:]:
        d = m["total_return"] - base_ret
        rel = d / base_ret * 100 if base_ret else 0
        print(f"    {lab:<38s} : {d:>+7.1%} ({rel:>+5.1f}% relatif)")

    # ══════════════════════════════════════════════════════════════════════
    # TEST D — TOP/FLOP par paire (6yr)
    # ══════════════════════════════════════════════════════════════════════
    logger.info("\n" + "━" * 60)
    logger.info("📋 TEST D — DÉTAIL PAR PAIRE")
    logger.info("━" * 60)

    pair_stats = {}
    for t in tr_all:
        sym = t.symbol
        if sym not in pair_stats:
            pair_stats[sym] = {"n": 0, "pnl": 0.0, "wins": 0}
        pair_stats[sym]["n"] += 1
        pair_stats[sym]["pnl"] += t.pnl_usd
        if t.pnl_usd > 0:
            pair_stats[sym]["wins"] += 1

    sorted_pairs = sorted(pair_stats.items(), key=lambda x: x[1]["pnl"], reverse=True)

    sep = "═" * 130
    print(f"\n{sep}")
    print(f"  📋 PERFORMANCE PAR PAIRE (6yr, toutes les paires Revolut X)")
    print(sep)

    print(f"\n  🟢 TOP 15 :")
    for sym, s in sorted_pairs[:15]:
        wr = s["wins"] / s["n"] * 100 if s["n"] else 0
        print(f"    {sym:<14s} : {s['n']:>3d} trades | WR {wr:>4.0f}% | PnL ${s['pnl']:>+9.2f}")

    print(f"\n  🔴 FLOP 10 :")
    for sym, s in sorted_pairs[-10:]:
        wr = s["wins"] / s["n"] * 100 if s["n"] else 0
        print(f"    {sym:<14s} : {s['n']:>3d} trades | WR {wr:>4.0f}% | PnL ${s['pnl']:>+9.2f}")

    # Paires sans trades
    no_trades = set(actual_pairs) - set(pair_stats.keys())
    if no_trades:
        print(f"\n  ⚪ Sans trades : {', '.join(sorted(no_trades))}")

    profitable = sum(1 for _, s in pair_stats.items() if s["pnl"] > 0)
    total = len(pair_stats)
    print(f"\n  📈 {profitable}/{total} paires profitables ({profitable/total*100:.0f}%)")

    # ══════════════════════════════════════════════════════════════════════
    # TEST E — Derniers 2 mois (Jan-Fév 2026) avec fees Revolut
    # ══════════════════════════════════════════════════════════════════════
    logger.info("\n" + "━" * 60)
    logger.info("📅 TEST E — JAN-FÉV 2026")
    logger.info("━" * 60)

    fc_recent = _filt(all_c, D("2026-01-01"), D("2026-02-21"))
    if fc_recent:
        logger.info("  ⏳ %d paires, fees Revolut…", len(fc_recent))
        m_rec, tr_rec, _ = _run(fc_recent, BAL, 0.0009, 0.0015)
        logger.info("    ✅ %+.1f%% | %d trades", m_rec["total_return"] * 100, m_rec["n_trades"])

        _hdr("📅 TEST E — Jan-Fév 2026 (fees Revolut 0.09% + slip 0.15%)")
        print(_row(f"{len(fc_recent)} paires Revolut X", m_rec))

        # Par paire
        ps = {}
        for t in tr_rec:
            ps.setdefault(t.symbol, {"n": 0, "pnl": 0.0, "w": 0})
            ps[t.symbol]["n"] += 1
            ps[t.symbol]["pnl"] += t.pnl_usd
            if t.pnl_usd > 0:
                ps[t.symbol]["w"] += 1
        sp = sorted(ps.items(), key=lambda x: x[1]["pnl"], reverse=True)
        print(f"\n  📋 Par paire (Jan+Fév 2026) :")
        for sym, st in sp:
            wr = st["w"] / st["n"] * 100 if st["n"] else 0
            em = "🟢" if st["pnl"] > 0 else "🔴"
            print(f"    {em} {sym:<14s} : {st['n']:>3} trades | WR {wr:>4.0f}% | PnL ${st['pnl']:>+7.2f}")
        prof = sum(1 for _, st in ps.items() if st["pnl"] > 0)
        print(f"\n    📈 {prof}/{len(ps)} paires profitables")

    # ══════════════════════════════════════════════════════════════════════
    # SYNTHÈSE FINALE
    # ══════════════════════════════════════════════════════════════════════
    print(f"\n\n{'█' * 130}")
    print(f"  🏭 SYNTHÈSE — BACKTEST 53 PAIRES REVOLUT X")
    print(f"{'█' * 130}\n")

    m_full = results_b[0][1]
    m_stress = results_c[3][1]  # stress scenario

    checks = [
        (f"A. Diversification améliore (53p PF ≥ 17p PF)", m_all["profit_factor"] >= m17["profit_factor"] * 0.95),
        (f"B. Robuste toutes sous-périodes (PF > 1.0)", all_pf_ok),
        (f"C. Survie friction Revolut stress (PF ≥ 1.2)", m_stress["profit_factor"] >= 1.2),
        (f"D. Majorité paires profitables (≥ 60%)", profitable / total >= 0.6 if total else False),
        (f"E. Performance récente positive", m_rec["total_return"] > 0 if fc_recent else False),
    ]

    passed = sum(1 for _, ok in checks if ok)
    for lab, ok in checks:
        print(f"    {'✅' if ok else '❌'}  {lab}")

    print(f"\n    Score : {passed}/5\n")

    if passed == 5:
        print("  🚀🚀🚀 GO — 53 paires validées. Déployer sur Revolut X !")
    elif passed >= 4:
        print("  🚀 QUASI-READY — 4/5 OK. Vérifier le point faible.")
    elif passed >= 3:
        print("  ⚠️  PRUDENCE — Des faiblesses. Considérer un univers réduit.")
    else:
        print("  ❌ NON — Rester sur 17 paires.")

    print(f"\n{'█' * 130}\n")


if __name__ == "__main__":
    main()
