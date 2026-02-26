#!/usr/bin/env python
"""
🏭 VALIDATION PRÉ-PRODUCTION — 5 tests critiques (optimisé)

Paramètres verrouillés : pos=3, compound=True, cap=30%, risk=2%

Tests :
  1) Slippage + fees réels   : baseline vs high-friction (0.09% + 0.25%)
  2) Walk-forward OOS         : train 2020-22 / validate 2022-23 / test 2023-24 / test 2024-26
  3) Sans pires paires        : retirer VET, ALGO, ENJ → 28 paires
  4) Tail risk extrême        : worst drawdowns, pires mois, VaR/CVaR, crashes
  5) Derniers 2 mois          : Janvier–Février 2026 (friction Revolut réelle)

Optimisation : la baseline 6yr est calculée 1 seule fois et réutilisée.
"""

from __future__ import annotations

import logging
import math
import statistics
import sys
from collections import defaultdict
from datetime import datetime, timezone
from typing import Optional

from dotenv import load_dotenv
load_dotenv()

from src import config
from src.core.models import Candle
from backtest.data_loader import download_all_pairs
from backtest.simulator import BacktestConfig, BacktestEngine, Trade, EquityPoint
from backtest.metrics import compute_metrics

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("prod_valid")

# ── Paires ─────────────────────────────────────────────────────────────────────

PAIRS_31 = [
    "BTC-USD", "ETH-USD", "XRP-USD", "LINK-USD", "ADA-USD",
    "DOGE-USD", "ATOM-USD", "ALGO-USD", "LTC-USD", "ETC-USD",
    "MATIC-USD", "VET-USD", "THETA-USD", "FTM-USD", "CHZ-USD",
    "ENJ-USD", "BAT-USD", "ZIL-USD", "ICX-USD", "ONE-USD",
    "HBAR-USD", "IOTA-USD", "XTZ-USD", "EOS-USD", "NEO-USD",
    "DASH-USD", "ZEC-USD", "XLM-USD", "TRX-USD", "WAVES-USD",
    "KAVA-USD",
]

PAIRS_FLOP = {"VET-USD", "ALGO-USD", "ENJ-USD"}

# ── Helpers ────────────────────────────────────────────────────────────────────


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


def _filt(all_c, s, e, pairs=None):
    s_ms, e_ms = int(s.timestamp() * 1000), int(e.timestamp() * 1000)
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
    return compute_metrics(r), r.trades, r.equity_curve


def _row(label, m):
    pf_e = "✅" if m["profit_factor"] >= 1.3 else ("⚠️" if m["profit_factor"] >= 1.0 else "❌")
    return (
        f"  {label:<34s} │ {m['total_return']:>+7.1%} │ {m['cagr']:>+6.1%} │ "
        f"{m['max_drawdown']:>7.1%} │ {m['sharpe']:>7.2f} │ "
        f"{pf_e}{m['profit_factor']:>4.2f} │ {m['win_rate']:>4.0%} │ "
        f"{m['n_trades']:>6d} │ ${m['final_equity']:>9,.2f}"
    )


def _hdr(title):
    sep = "═" * 120
    print(f"\n{sep}")
    print(f"  {title}")
    print(sep)
    print(
        f"  {'Label':<34s} │ {'Return':>8s} │ {'CAGR':>7s} │ {'MaxDD':>7s} │ "
        f"{'Sharpe':>7s} │ {'PF':>5s} │ {'WR':>5s} │ {'Trades':>6s} │ {'Final$':>10s}"
    )
    print("  " + "─" * 116)


def _find_dd_episodes(eq, init):
    if not eq:
        return []
    eps = []
    pk = init
    pk_ts = eq[0].timestamp
    tr = pk
    tr_ts = pk_ts
    in_dd = False
    for pt in eq:
        if pt.equity >= pk:
            if in_dd and (pk - tr) / pk > 0.005:
                eps.append({
                    "s": datetime.fromtimestamp(pk_ts / 1000, tz=timezone.utc).strftime("%Y-%m-%d"),
                    "t": datetime.fromtimestamp(tr_ts / 1000, tz=timezone.utc).strftime("%Y-%m-%d"),
                    "d": max((tr_ts - pk_ts) // (86400 * 1000), 1),
                    "dd": (tr - pk) / pk,
                    "loss": tr - pk,
                })
            pk = pt.equity
            pk_ts = pt.timestamp
            tr = pk
            tr_ts = pt.timestamp
            in_dd = False
        else:
            in_dd = True
            if pt.equity < tr:
                tr = pt.equity
                tr_ts = pt.timestamp
    if in_dd and (pk - tr) / pk > 0.005:
        eps.append({
            "s": datetime.fromtimestamp(pk_ts / 1000, tz=timezone.utc).strftime("%Y-%m-%d"),
            "t": datetime.fromtimestamp(tr_ts / 1000, tz=timezone.utc).strftime("%Y-%m-%d"),
            "d": max((tr_ts - pk_ts) // (86400 * 1000), 1),
            "dd": (tr - pk) / pk,
            "loss": tr - pk,
        })
    eps.sort(key=lambda e: e["dd"])
    return eps


D = lambda s: datetime.strptime(s, "%Y-%m-%d").replace(tzinfo=timezone.utc)


# ═══════════════════════════════════════════════════════════════════════════════

def main():
    BAL = 1000.0

    logger.info("🏭 VALIDATION PRÉ-PRODUCTION — 5 tests")
    logger.info("   pos=3, compound=True, cap=30%%, risk=2%%\n")

    # ── Téléchargement unique ──
    logger.info("📥 Chargement 31 paires H4 (2020→2026)…")
    all_c = download_all_pairs(PAIRS_31, D("2020-02-20"), D("2026-02-21"), interval="4h")
    logger.info("✅ %d paires\n", len(all_c))

    # ── BASELINE 6yr (1 seule fois, réutilisée par T1/T3/T4) ──
    logger.info("📊 Baseline 6yr (0.075%% fee + 0.1%% slip)…")
    f6 = _filt(all_c, D("2020-02-20"), D("2026-02-20"))
    m_base, tr_base, eq_base = _run(f6, BAL)
    logger.info("   ✅ Baseline: %+.1f%% | PF %.2f\n", m_base["total_return"] * 100, m_base["profit_factor"])

    # ══════════════════════════════════════════════════════════════════════
    # TEST 1 — Slippage + Fees
    # ══════════════════════════════════════════════════════════════════════
    logger.info("━" * 50)
    logger.info("🔧 TEST 1 — FRICTION")
    logger.info("━" * 50)

    scenarios = [
        ("Revolut réel (0.09% + 0.15%)", 0.0009, 0.0015),
        ("Stress (0.09% + 0.25%)",        0.0009, 0.0025),
        ("Extrême (0.10% + 0.30%)",       0.0010, 0.0030),
    ]

    t1_all = [("Baseline (0.075% + 0.1%)", m_base)]
    for lab, fee, slip in scenarios:
        logger.info("  ⏳ %s…", lab)
        m, _, _ = _run(f6, BAL, fee, slip)
        t1_all.append((lab, m))
        logger.info("    ✅ %+.1f%% | PF %.2f", m["total_return"] * 100, m["profit_factor"])

    _hdr("🔧 TEST 1 — Impact friction sur 6 ans / 31 paires")
    for lab, m in t1_all:
        print(_row(lab, m))

    br = m_base["total_return"]
    print(f"\n  📉 Dégradation vs baseline :")
    for lab, m in t1_all[1:]:
        d = m["total_return"] - br
        rel = d / br * 100 if br else 0
        print(f"    {lab:<34s} : {d:>+7.1%} ({rel:>+5.1f}% relatif)")

    m_stress = t1_all[2][1]
    t1_ok = m_stress["profit_factor"] >= 1.2 and m_stress["total_return"] > 0.3
    if t1_ok:
        print(f"\n  ✅✅ EDGE SURVIT — stress PF {m_stress['profit_factor']:.2f}, Return {m_stress['total_return']:+.1%}")
    else:
        print(f"\n  ⚠️  Edge dégradé — stress PF {m_stress['profit_factor']:.2f}, Return {m_stress['total_return']:+.1%}")

    # ══════════════════════════════════════════════════════════════════════
    # TEST 2 — Walk-Forward
    # ══════════════════════════════════════════════════════════════════════
    logger.info("\n" + "━" * 50)
    logger.info("🔬 TEST 2 — WALK-FORWARD")
    logger.info("━" * 50)

    wf = [
        ("TRAIN  (2020-2022)", "2020-02-20", "2022-02-20"),
        ("VALID  (2022-2023)", "2022-02-20", "2023-02-20"),
        ("TEST-1 (2023-2024)", "2023-02-20", "2024-02-20"),
        ("TEST-2 (2024-2026)", "2024-02-20", "2026-02-20"),
    ]

    t2_res = []
    for lab, s, e in wf:
        logger.info("  ⏳ %s…", lab)
        m, _, _ = _run(_filt(all_c, D(s), D(e)), BAL)
        t2_res.append((lab, m))
        logger.info("    ✅ %+.1f%% | PF %.2f", m["total_return"] * 100, m["profit_factor"])

    _hdr("🔬 TEST 2 — Walk-Forward (params JAMAIS ré-optimisés)")
    for lab, m in t2_res:
        print(_row(lab, m))

    train_pf = t2_res[0][1]["profit_factor"]
    oos = t2_res[1:]
    oos_pfs = [m["profit_factor"] for _, m in oos]
    oos_rets = [m["total_return"] for _, m in oos]
    ratio = statistics.mean(oos_pfs) / train_pf if train_pf else 0

    print(f"\n  📊 Analyse OOS :")
    print(f"    Train PF       : {train_pf:.2f}")
    print(f"    OOS PF moyen   : {statistics.mean(oos_pfs):.2f}")
    print(f"    OOS Return moy : {statistics.mean(oos_rets):+.1%}")
    print(f"    Ratio OOS/Train: {ratio:.2f} (>0.7 = bon, <0.5 = overfit)")

    t2_ok = all(pf > 1.0 for pf in oos_pfs) and ratio > 0.5
    if t2_ok and ratio > 0.7:
        print(f"\n  ✅✅ PAS D'OVERFIT — ratio {ratio:.2f}")
    elif t2_ok:
        print(f"\n  ✅ Edge OOS présent — ratio {ratio:.2f}")
    else:
        print(f"\n  ⚠️  Signes d'overfit — ratio {ratio:.2f}")

    # ══════════════════════════════════════════════════════════════════════
    # TEST 3 — Sans VET, ALGO, ENJ
    # ══════════════════════════════════════════════════════════════════════
    logger.info("\n" + "━" * 50)
    logger.info("🧹 TEST 3 — SANS PIRES PAIRES")
    logger.info("━" * 50)

    pairs_28 = [p for p in PAIRS_31 if p not in PAIRS_FLOP]
    logger.info("  ⏳ 28 paires…")
    m28, _, _ = _run(_filt(all_c, D("2020-02-20"), D("2026-02-20"), pairs_28), BAL)
    logger.info("    ✅ %+.1f%% | PF %.2f", m28["total_return"] * 100, m28["profit_factor"])

    _hdr("🧹 TEST 3 — Impact retrait pires paires (6 ans)")
    print(_row("31 paires (référence)", m_base))
    print(_row("28 paires (sans VET/ALGO/ENJ)", m28))

    dr = m28["total_return"] - m_base["total_return"]
    dp = m28["profit_factor"] - m_base["profit_factor"]
    dd = m28["max_drawdown"] - m_base["max_drawdown"]
    print(f"\n  📊 Delta :")
    print(f"    Return : {dr:>+7.1%}")
    print(f"    PF     : {dp:>+.2f}")
    print(f"    MaxDD  : {dd:>+.1%} ({'mieux' if dd > 0 else 'pire'})")

    t3_ok = m28["profit_factor"] >= 1.3
    if dr > 0 and dp > 0:
        print(f"\n  ✅ Flops = bruit → retirer VET/ALGO/ENJ de prod")
    else:
        print(f"\n  ℹ️  Impact neutre — garder 31 paires")

    # ══════════════════════════════════════════════════════════════════════
    # TEST 4 — Tail Risk (0 backtest supplémentaire)
    # ══════════════════════════════════════════════════════════════════════
    logger.info("\n" + "━" * 50)
    logger.info("💀 TEST 4 — TAIL RISK (réutilise baseline)")
    logger.info("━" * 50)

    sep = "═" * 120
    print(f"\n{sep}")
    print(f"  💀 TEST 4 — TAIL RISK EXTRÊME (6 ans / 31 paires)")
    print(sep)

    # 4a — Drawdown episodes
    print(f"\n  📉 TOP 5 drawdown episodes :")
    print(f"  {'#':>3} │ {'Début':>12} │ {'Creux':>12} │ {'Durée':>7} │ {'DD':>8} │ {'Perte $':>10}")
    print("  " + "─" * 65)

    for i, ep in enumerate(_find_dd_episodes(eq_base, BAL)[:5], 1):
        print(
            f"  {i:>3} │ {ep['s']:>12} │ {ep['t']:>12} │ "
            f"{ep['d']:>4}j   │ {ep['dd']:>7.1%} │ ${ep['loss']:>9.2f}"
        )

    # 4b — Pires mois
    monthly = m_base.get("monthly_returns", [])
    if monthly:
        print(f"\n  📅 5 pires mois :")
        for mo, ret in sorted(monthly, key=lambda x: x[1])[:5]:
            print(f"    {mo} : {ret:>+6.2%}")

    # 4c — Pires trades
    print(f"\n  🔥 10 pires trades :")
    print(f"  {'Paire':<12} │ {'Date':>10} │ {'Entrée $':>10} │ {'PnL $':>9} │ {'PnL %':>7} │ {'Raison':>10}")
    print("  " + "─" * 72)
    for t in sorted(tr_base, key=lambda t: t.pnl_usd)[:10]:
        dt = datetime.fromtimestamp(t.entry_time / 1000, tz=timezone.utc).strftime("%Y-%m-%d")
        print(
            f"  {t.symbol:<12} │ {dt:>10} │ ${t.entry_price:>9.4f} │ "
            f"${t.pnl_usd:>+8.2f} │ {t.pnl_pct:>+6.1%} │ {t.exit_reason:>10}"
        )

    # 4d — VaR/CVaR
    print(f"\n  📊 Value at Risk (rendements H4) :")
    if len(eq_base) > 100:
        rets = []
        for i in range(1, len(eq_base)):
            p = eq_base[i - 1].equity
            if p > 0:
                rets.append((eq_base[i].equity - p) / p)
        rets.sort()
        n = len(rets)
        var95 = rets[int(n * 0.05)]
        var99 = rets[int(n * 0.01)]
        cvar95 = statistics.mean(rets[:max(int(n * 0.05), 1)])
        cvar99 = statistics.mean(rets[:max(int(n * 0.01), 1)])
        ef = eq_base[-1].equity
        print(f"    VaR 95%  : {var95:>+.3%} (${ef * abs(var95):>7.2f} sur ${ef:,.0f})")
        print(f"    VaR 99%  : {var99:>+.3%} (${ef * abs(var99):>7.2f})")
        print(f"    CVaR 95% : {cvar95:>+.3%}")
        print(f"    CVaR 99% : {cvar99:>+.3%}")
        print(f"    Pire bar : {rets[0]:>+.3%}")

    # 4e — Crashes crypto
    print(f"\n  🌪️  Performance durant les crashes crypto :")
    crashes = [
        ("COVID mars 2020",       "2020-03-01", "2020-04-01"),
        ("BTC mai 2021 (-50%)",   "2021-05-01", "2021-07-01"),
        ("LUNA/UST mai 2022",     "2022-05-01", "2022-07-01"),
        ("FTX nov 2022",          "2022-11-01", "2023-01-01"),
        ("Bear bottom déc 2022",  "2022-12-01", "2023-03-01"),
    ]
    for lab, s, e in crashes:
        cs_ms = int(D(s).timestamp() * 1000)
        ce_ms = int(D(e).timestamp() * 1000)
        peq = [pt for pt in eq_base if cs_ms <= pt.timestamp <= ce_ms]
        if len(peq) >= 2:
            se, ee = peq[0].equity, peq[-1].equity
            pr = (ee - se) / se if se else 0
            pk = se
            pdd = 0
            for pt in peq:
                pk = max(pk, pt.equity)
                pdd = min(pdd, (pt.equity - pk) / pk)
            em = "🟢" if pr >= 0 else "🔴"
            print(f"    {em} {lab:<28s} : {pr:>+6.1%} (DD intra: {pdd:>+.1%})")
        else:
            print(f"    ⚪ {lab:<28s} : données insuffisantes")

    t4_ok = m_base["max_drawdown"] > -0.20
    print(f"\n  🏁 MaxDD global : {m_base['max_drawdown']:.1%} → {'✅ maîtrisé' if t4_ok else '⚠️ à surveiller'}")

    # ══════════════════════════════════════════════════════════════════════
    # TEST 5 — Derniers 2 mois
    # ══════════════════════════════════════════════════════════════════════
    logger.info("\n" + "━" * 50)
    logger.info("📅 TEST 5 — JAN-FÉV 2026")
    logger.info("━" * 50)

    t5_periods = [
        ("Janvier 2026",           "2026-01-01", "2026-02-01"),
        ("Février 2026 (partiel)", "2026-02-01", "2026-02-21"),
        ("Jan + Fév 2026",         "2026-01-01", "2026-02-21"),
    ]

    t5_res = []
    for lab, s, e in t5_periods:
        fc = _filt(all_c, D(s), D(e))
        if not fc:
            continue
        logger.info("  ⏳ %s…", lab)
        m, trades, _ = _run(fc, BAL, 0.0009, 0.0015)
        t5_res.append((lab, m, trades))
        logger.info("    ✅ %+.1f%% | %d trades", m["total_return"] * 100, m["n_trades"])

    if t5_res:
        _hdr("📅 TEST 5 — Performance récente (fees Revolut: 0.09% + slip 0.15%)")
        for lab, m, _ in t5_res:
            print(_row(lab, m))

        _, m_comb, tr_comb = t5_res[-1]
        ps = {}
        for t in tr_comb:
            ps.setdefault(t.symbol, {"n": 0, "pnl": 0.0, "w": 0})
            ps[t.symbol]["n"] += 1
            ps[t.symbol]["pnl"] += t.pnl_usd
            if t.pnl_usd > 0:
                ps[t.symbol]["w"] += 1

        sp = sorted(ps.items(), key=lambda x: x[1]["pnl"], reverse=True)
        print(f"\n  📋 Par paire (Jan+Fév 2026) :")
        for sym, st in sp[:10]:
            wr = st["w"] / st["n"] * 100 if st["n"] else 0
            em = "🟢" if st["pnl"] > 0 else "🔴"
            print(f"    {em} {sym:<12} : {st['n']:>3} trades | WR {wr:>4.0f}% | PnL ${st['pnl']:>+7.2f}")
        if len(sp) > 10:
            print(f"    … et {len(sp) - 10} autres")

        prof = sum(1 for _, st in ps.items() if st["pnl"] > 0)
        print(f"\n    📈 {prof}/{len(ps)} paires profitables")
        t5_ok = m_comb["total_return"] > 0 and m_comb["profit_factor"] >= 1.0
    else:
        t5_ok = False

    # ══════════════════════════════════════════════════════════════════════
    # SYNTHÈSE FINALE
    # ══════════════════════════════════════════════════════════════════════
    print(f"\n\n{'█' * 120}")
    print(f"  🏭 SYNTHÈSE FINALE — VALIDATION PRÉ-PRODUCTION")
    print(f"{'█' * 120}\n")

    checks = [
        ("1. Survie frictions réelles (stress PF≥1.2, Ret>30%)", t1_ok),
        ("2. Walk-forward pas d'overfit (OOS PF>1.0, ratio>0.5)", t2_ok),
        ("3. Robuste sans pires paires (PF≥1.3 sur 28 paires)",  t3_ok),
        ("4. Tail risk maîtrisé (MaxDD > -20%)",                  t4_ok),
        ("5. Performance récente positive (Jan-Fév 2026 PF≥1.0)", t5_ok),
    ]

    passed = sum(1 for _, ok in checks if ok)
    for lab, ok in checks:
        print(f"    {'✅' if ok else '❌'}  {lab}")

    print(f"\n    Score : {passed}/5\n")

    if passed == 5:
        print("  🚀🚀🚀 PRODUCTION-READY — Tous les tests passés.")
        print("         Déployer avec confiance sur Revolut X.")
    elif passed >= 4:
        print("  🚀 QUASI-READY — 4/5 OK. Corriger le point faible puis déployer.")
    elif passed >= 3:
        print("  ⚠️  PRUDENCE — Faiblesses détectées. Investiguer.")
    else:
        print("  ❌ NON PRÊT — Revoir la stratégie.")

    print(f"\n{'█' * 120}\n")


if __name__ == "__main__":
    main()
