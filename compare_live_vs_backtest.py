#!/usr/bin/env python3
"""
═══════════════════════════════════════════════════════════════════════════════
  TradeX — Comparaison LIVE vs BACKTEST (3 derniers mois)
  
  Même logique exacte que le bot live (Dow Theory → NEUTRAL → Range BUY),
  3 variantes de SL :
    A) SL fixe (config live : range_low × 0.998)  
    B) SL = 1.0 × ATR
    C) SL = 1.5 × ATR
  
  + Récupération des trades réels du VPS pour comparer
═══════════════════════════════════════════════════════════════════════════════
"""

import argparse
import json
import subprocess
import sys
import os
import logging
from datetime import datetime, timezone, timedelta

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dotenv import load_dotenv
load_dotenv()

from backtest.data_loader import download_all_pairs, SYMBOL_MAP
from backtest.simulator import BacktestEngine, BacktestConfig, BacktestResult
from backtest.metrics import compute_metrics

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger("compare")


# ═══════════════════════════════════════════════════════════════════════════════
# 1) Récupérer les trades LIVE depuis le VPS
# ═══════════════════════════════════════════════════════════════════════════════

def fetch_live_trades() -> list[dict]:
    """Récupère les trades LIVE du VPS via SSH."""
    print("\n" + "═" * 80)
    print("  📡 RÉCUPÉRATION DES TRADES LIVE (VPS)")
    print("═" * 80)

    try:
        result = subprocess.run(
            ["ssh", "BOT-VPS", "cat /opt/tradex/data/state_binance.json"],
            capture_output=True, text=True, timeout=15,
        )
        if result.returncode != 0:
            print(f"  ❌ SSH échoué : {result.stderr.strip()}")
            return []
        data = json.loads(result.stdout)
    except Exception as e:
        print(f"  ⚠️ Impossible de récupérer les trades live : {e}")
        return []

    raw = data.get("positions", data) if isinstance(data, dict) else data
    positions = list(raw.values()) if isinstance(raw, dict) else raw

    closed = [p for p in positions if isinstance(p, dict) and p.get("status") == "CLOSED"]
    open_pos = [p for p in positions if isinstance(p, dict) and p.get("status") == "OPEN"]

    print(f"\n  📊 Positions trouvées : {len(positions)} total")
    print(f"     ├── Fermées : {len(closed)}")
    print(f"     └── Ouvertes : {len(open_pos)}")

    if closed:
        wins = sum(1 for t in closed if (t.get("pnl") or 0) > 0)
        total_pnl = sum(t.get("pnl") or 0 for t in closed)
        wr = wins / len(closed)
        sl_pcts = []
        for t in closed:
            e = t.get("entry_price") or 0
            s = t.get("sl_price") or 0
            if e: sl_pcts.append(abs(e - s) / e * 100)
        avg_sl = sum(sl_pcts) / len(sl_pcts) if sl_pcts else 0

        print(f"\n  📈 Résultats LIVE :")
        print(f"     {len(closed)} trades | {wins}W / {len(closed)-wins}L | WR {wr:.0%}")
        print(f"     PnL : ${total_pnl:+.2f} | SL moyen : {avg_sl:.2f}%")

        print(f"\n  📋 Détail :")
        for t in closed:
            sym = t.get("symbol", "?")
            pnl = t.get("pnl") or 0
            e = t.get("entry_price") or 0
            s = t.get("sl_price") or 0
            sl_pct = abs(e - s) / e * 100 if e else 0
            print(f"     {sym:16s} PnL=${pnl:+.2f}  SL={sl_pct:.2f}%")

    return closed


# ═══════════════════════════════════════════════════════════════════════════════
# 2) Backtest — même logique que le live, SL variable
# ═══════════════════════════════════════════════════════════════════════════════

def run_backtest(
    candles: dict,
    balance: float,
    atr_sl_mult: float = 0.0,
    min_sl_pct: float = 0.0,
    entry_buffer: float = 0.002,
    label: str = "",
) -> tuple[BacktestResult, dict]:
    """
    Backtest avec le moteur classique (identique au live).
    atr_sl_mult = 0 → SL fixe (live config)
    atr_sl_mult > 0 → SL = mult × ATR
    min_sl_pct > 0 → rejeter les trades avec SL < min_sl_pct
    """
    cfg = BacktestConfig(
        initial_balance=balance,
        risk_percent_range=0.02,          # 2% (comme live)
        entry_buffer_pct=entry_buffer,
        sl_buffer_pct=0.003,
        zero_risk_trigger_pct=0.02,
        zero_risk_lock_pct=0.005,
        trailing_stop_pct=0.02,
        max_position_pct=0.30,
        max_simultaneous_positions=3,
        swing_lookback=3,
        range_width_min=0.018,
        range_entry_buffer_pct=entry_buffer,
        range_sl_buffer_pct=0.003,
        range_cooldown_bars=3,
        range_min_sl_pct=min_sl_pct,
        # ATR-based SL
        range_atr_sl_mult=atr_sl_mult,
        range_atr_period=14,
        # Config standard
        fee_pct=0.00075,
        slippage_pct=0.001,
        enable_trend=False,               # RANGE ONLY
        enable_range=True,
        allow_short=False,
    )

    engine = BacktestEngine(candles, cfg)
    result = engine.run()
    metrics = compute_metrics(result)
    return result, metrics


# ═══════════════════════════════════════════════════════════════════════════════
# 3) Tableau comparatif
# ═══════════════════════════════════════════════════════════════════════════════

def print_comparison(
    live_trades: list[dict],
    results: list[tuple[str, BacktestResult, dict]],
    balance: float,
):
    sep = "═" * 95

    print(f"\n{sep}")
    print("  📊 COMPARAISON : LIVE  vs  BACKTEST (même logique, SL variable)")
    print(f"  📅 Derniers 3 mois | Capital : ${balance:,.0f}")
    print(sep)

    # ── Stats LIVE ──
    live_col = None
    if live_trades:
        n = len(live_trades)
        wins = sum(1 for t in live_trades if (t.get("pnl") or 0) > 0)
        pnl = sum(t.get("pnl") or 0 for t in live_trades)
        wr = wins / n if n else 0
        final = balance + pnl
        ret = pnl / balance if balance else 0
        avg = pnl / n if n else 0
        sl_pcts = []
        for t in live_trades:
            e = t.get("entry_price") or 0
            s = t.get("sl_price") or 0
            if e: sl_pcts.append(abs(e - s) / e * 100)
        avg_sl = sum(sl_pcts) / len(sl_pcts) if sl_pcts else 0
        live_col = {
            "trades": n, "wins": wins, "losses": n - wins,
            "wr": wr, "ret": ret, "pnl": pnl, "avg": avg,
            "final": final, "sl": f"{avg_sl:.2f}%",
            "dd": "N/A", "sharpe": "N/A", "pf": "N/A",
        }

    # ── Build columns ──
    cols = []
    if live_col:
        cols.append(("🔴 LIVE", live_col))
    for label, result, metrics in results:
        n = metrics.get("n_trades", 0)
        w = int(n * metrics.get("win_rate", 0))
        cols.append((label, {
            "trades": n, "wins": w, "losses": n - w,
            "wr": metrics.get("win_rate", 0),
            "ret": metrics.get("total_return", 0),
            "pnl": metrics.get("total_return", 0) * balance,
            "avg": metrics.get("avg_pnl_usd", 0),
            "final": metrics.get("final_equity", balance),
            "sl": "ATR" if "ATR" in label else "~0.20% fixe",
            "dd": f"{metrics.get('max_drawdown', 0):.1%}",
            "sharpe": f"{metrics.get('sharpe', 0):.2f}",
            "pf": f"{metrics.get('profit_factor', 0):.2f}",
        }))

    # ── Table ──
    header = "  {:20s}".format("Métrique")
    for name, _ in cols:
        header += f" │ {name:>18s}"
    print(f"\n{header}")
    print("  " + "─" * (22 + 21 * len(cols)))

    rows = [
        ("Trades", lambda c: f"{c['trades']}"),
        ("Wins / Losses", lambda c: f"{c['wins']}W / {c['losses']}L"),
        ("Win Rate", lambda c: f"{c['wr']:.1%}" if isinstance(c['wr'], float) else c['wr']),
        ("Return", lambda c: f"{c['ret']:+.1%}" if isinstance(c['ret'], float) else c['ret']),
        ("PnL total ($)", lambda c: f"${c['pnl']:+.2f}" if isinstance(c['pnl'], float) else c['pnl']),
        ("PnL moyen ($)", lambda c: f"${c['avg']:+.2f}" if isinstance(c['avg'], float) else c['avg']),
        ("Capital final", lambda c: f"${c['final']:,.2f}" if isinstance(c['final'], float) else c['final']),
        ("Max Drawdown", lambda c: c['dd']),
        ("Sharpe", lambda c: c['sharpe']),
        ("Profit Factor", lambda c: c['pf']),
        ("SL type", lambda c: c['sl']),
    ]

    for label, fmt in rows:
        line = f"  {label:20s}"
        for _, col in cols:
            line += f" │ {str(fmt(col)):>18s}"
        print(line)

    # ── Exit reasons par backtest ──
    for label, result, metrics in results:
        by_exit = metrics.get("by_exit_reason", {})
        if by_exit:
            print(f"\n  📋 Sorties — {label} :")
            for reason, stats in sorted(by_exit.items()):
                n = stats.get("n_trades", 0)
                w = int(n * stats.get("win_rate", 0))
                pnl = stats.get("total_pnl", 0)
                wr = stats.get("win_rate", 0)
                print(f"     {reason:16s} : {n:3d} trades | WR {wr:.0%} | PnL ${pnl:+.2f}")

    # ── Détail par paire (top/bottom) ──
    for label, result, metrics in results:
        by_pair = metrics.get("by_pair", {})
        active = {k: v for k, v in by_pair.items() if v.get("n_trades", 0) > 0}
        if active:
            sorted_pairs = sorted(active.items(), key=lambda x: x[1].get("total_pnl", 0), reverse=True)
            print(f"\n  🏆 Top/Bottom paires — {label} ({len(active)} actives) :")
            for pair, stats in sorted_pairs[:5]:
                n = stats.get("n_trades", 0)
                pnl = stats.get("total_pnl", 0)
                wr = stats.get("win_rate", 0)
                print(f"     ✅ {pair:14s} : {n:2d} trades | WR {wr:.0%} | ${pnl:+.2f}")
            if len(sorted_pairs) > 5:
                print(f"     ...")
                for pair, stats in sorted_pairs[-3:]:
                    n = stats.get("n_trades", 0)
                    pnl = stats.get("total_pnl", 0)
                    wr = stats.get("win_rate", 0)
                    print(f"     ❌ {pair:14s} : {n:2d} trades | WR {wr:.0%} | ${pnl:+.2f}")

    # ── Verdict ──
    print(f"\n{sep}")
    print("  🏆 VERDICT")
    print(sep)

    for label, result, metrics in results:
        ret = metrics.get("total_return", 0)
        wr = metrics.get("win_rate", 0)
        n = metrics.get("n_trades", 0)
        sharpe = metrics.get("sharpe", 0)
        print(f"\n  {label}")
        print(f"     Return {ret:+.1%} | WR {wr:.0%} | {n} trades | Sharpe {sharpe:.2f}")

    best = max(results, key=lambda x: x[2].get("total_return", 0))
    print(f"\n  → Meilleur : {best[0]} ({best[2].get('total_return', 0):+.1%})")

    print(f"\n{sep}\n")


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="TradeX — Comparaison LIVE vs BACKTEST (même logique, SL variable)"
    )
    parser.add_argument("--months", type=int, default=3)
    parser.add_argument("--balance", type=float, default=940.0)
    parser.add_argument("--no-live", action="store_true",
                        help="Ne pas récupérer les trades live (SSH)")
    parser.add_argument("--pairs-count", type=int, default=0,
                        help="Limiter le nombre de paires (0 = toutes)")
    args = parser.parse_args()

    end = datetime.now(timezone.utc)
    start = end - timedelta(days=args.months * 30)

    all_pairs = list(SYMBOL_MAP.keys())
    if args.pairs_count > 0:
        all_pairs = all_pairs[:args.pairs_count]

    print(f"\n{'═' * 95}")
    print(f"  🔬 TEST : range_width >= 1.8% + SL min 0.5%")
    print(f"  📅 {start:%Y-%m-%d} → {end:%Y-%m-%d} ({args.months} mois)")
    print(f"  💰 Capital : ${args.balance:,.0f} | 🪙 {len(all_pairs)} paires")
    print(f"  📐 A) SL fixe (ref, pas de filtre SL min)")
    print(f"     B) SL fixe + filtre SL > 0.5%")
    print(f"     C) SL fixe élargi (buffer 0.4%) + filtre SL > 0.5%")
    print(f"{'═' * 95}")

    # 1) Trades LIVE
    live_trades = [] if args.no_live else fetch_live_trades()

    # 2) Télécharger les données
    print(f"\n  📥 Téléchargement ({len(all_pairs)} paires, {args.months} mois)…")
    candles = download_all_pairs(all_pairs, start, end, interval="4h")
    candles = {p: c for p, c in candles.items() if len(c) > 50}
    print(f"  ✅ {len(candles)} paires chargées")

    # 3) Backtests
    results = []

    variants = [
        ("🟡 A) Ref (SL fixe, pas filtre)",  0.0, 0.0,   0.002),
        ("🟢 B) SL fixe + min 0.5%",         0.0, 0.005, 0.002),
        ("🔵 C) Buffer 0.4% + min 0.5%",     0.0, 0.005, 0.004),
    ]

    for label, atr_mult, min_sl, buf in variants:
        print(f"\n{'─' * 60}")
        print(f"  {label}")
        print(f"     entry_buffer={buf*100:.1f}% | SL min={min_sl*100:.1f}%")
        print(f"{'─' * 60}")

        r, m = run_backtest(candles, args.balance, atr_sl_mult=atr_mult,
                            min_sl_pct=min_sl, entry_buffer=buf, label=label)
        results.append((label, r, m))

    # 4) Comparaison
    print_comparison(live_trades, results, args.balance)


if __name__ == "__main__":
    main()
