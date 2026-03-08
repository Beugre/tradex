#!/usr/bin/env python3
"""London Breakout — Risk comparison 1.5-5%."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from backtest.data_loader import download_candles
from backtest.run_asian_deep import AsianConfig, simulate_asian, compute_kpis, print_table

def main():
    end = datetime.now(timezone.utc)
    start = end - timedelta(days=6 * 365)
    pairs = ["BTC-USD", "ETH-USD", "SOL-USD", "BNB-USD",
             "LINK-USD", "ADA-USD", "DOT-USD", "AVAX-USD"]
    tf = "4h"

    all_candles = {}
    for p in pairs:
        all_candles[p] = download_candles(p, start, end, interval=tf)
        print(f"  {p}: {len(all_candles[p])} bougies")

    # ── Full 6yr comparison ──
    configs = [
        ("R1.5%_MP1", 0.015, 1),
        ("R1.5%_MP2", 0.015, 2),
        ("R2%_MP1",   0.02,  1),
        ("R2%_MP2",   0.02,  2),
        ("R2%_MP3",   0.02,  3),
        ("R2.5%_MP1", 0.025, 1),
        ("R2.5%_MP2", 0.025, 2),
        ("R2.5%_MP3", 0.025, 3),
        ("R3%_MP1",   0.03,  1),
        ("R3%_MP2",   0.03,  2),
        ("R4%_MP1",   0.04,  1),
        ("R5%_MP1",   0.05,  1),
    ]

    kpis = []
    for label, risk, mp in configs:
        cfg = AsianConfig(
            asian_start_hour=8, asian_end_hour=16,
            sl_atr_mult=2.0, tp1_pct=0.02, tp2_pct=0.05,
            vol_mult=2.0, min_range_pct=0.015,
            risk_per_trade=risk, max_positions=mp, cooldown_bars=2,
        )
        r = simulate_asian(all_candles, cfg, 500.0, tf)
        r.label = label
        kpis.append(compute_kpis(r))

    print_table(sorted(kpis, key=lambda k: -k["pf"]),
                "London Breakout — Risk 1.5-5% (6 ans, $500)")

    # ── Walk-Forward 3yr/3yr ──
    min_len = min(len(c) for c in all_candles.values())
    split = int(min_len * 3 / 6)
    sb = 30

    print()
    print("=" * 110)
    print("  Walk-Forward 3yr/3yr — Risk comparison")
    print("=" * 110)
    hdr = (f"  {'Config':<16} {'PF_train':>8} {'PnL_tr':>9} {'Tr_tr':>6} {'DD_tr':>7}"
           f"  |  {'PF_test':>8} {'PnL_te':>9} {'Tr_te':>6} {'DD_te':>7}"
           f"  |  {'Stable':>7}")
    print(hdr)
    print("  " + "-" * 105)

    wf_configs = [
        ("R1.5%_MP1", 0.015, 1),
        ("R2%_MP1",   0.02,  1),
        ("R2%_MP2",   0.02,  2),
        ("R2.5%_MP1", 0.025, 1),
        ("R2.5%_MP2", 0.025, 2),
        ("R3%_MP1",   0.03,  1),
        ("R5%_MP1",   0.05,  1),
    ]

    for label, risk, mp in wf_configs:
        cfg = AsianConfig(
            asian_start_hour=8, asian_end_hour=16,
            sl_atr_mult=2.0, tp1_pct=0.02, tp2_pct=0.05,
            vol_mult=2.0, min_range_pct=0.015,
            risk_per_trade=risk, max_positions=mp, cooldown_bars=2,
        )
        rt = simulate_asian(all_candles, cfg, 500.0, tf, start_bar=sb, end_bar=split)
        rt.label = f"{label}_TR"
        kt = compute_kpis(rt)

        rte = simulate_asian(all_candles, cfg, 500.0, tf, start_bar=split, end_bar=min_len)
        rte.label = f"{label}_TE"
        kte = compute_kpis(rte)

        d = kte["pf"] - kt["pf"]
        tag = "YES" if abs(d) < 0.3 and kte["pf"] >= 1.0 else ("~" if kte["pf"] >= 0.9 else "NO")
        print(f"  {label:<16} {kt['pf']:>8.2f} {kt['pnl']:>+9.2f} {kt['trades']:>6} {kt['max_dd']:>6.1f}%"
              f"  |  {kte['pf']:>8.2f} {kte['pnl']:>+9.2f} {kte['trades']:>6} {kte['max_dd']:>6.1f}%"
              f"  |  {tag:>7}")

    print("  " + "-" * 105)

    # ── PnL/an ──
    print()
    print("=" * 80)
    print("  Rendement annualisé (PnL / 6 ans)")
    print("=" * 80)
    print(f"  {'Config':<16} {'PnL 6yr':>10} {'PnL/an':>10} {'DD':>7} {'PF':>7} {'Trades':>7}")
    print("  " + "-" * 62)
    for k in sorted(kpis, key=lambda x: -x["pf"]):
        pnl_an = k["pnl"] / 6
        print(f"  {k['label']:<16} {k['pnl']:>+10.2f} {pnl_an:>+10.2f} {k['max_dd']:>6.1f}% {k['pf']:>7.2f} {k['trades']:>7}")
    print("  " + "-" * 62)


if __name__ == "__main__":
    main()
