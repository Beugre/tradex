"""
Comprehensive Firebase trading bots analysis.
Fetches ALL trades & daily snapshots for revolut, binance, binance-crashbot.
"""

import json
import os
import sys
from collections import defaultdict
from datetime import datetime, timezone
from statistics import mean, median

from google.cloud import firestore
from google.oauth2 import service_account

# ── Firebase connection ──────────────────────────────────────────────────────

CRED_PATH = os.path.join(os.path.dirname(__file__), "firebase-credentials.json")
credentials = service_account.Credentials.from_service_account_file(CRED_PATH)
db = firestore.Client(project="satochi-d38ec", credentials=credentials)

EXCHANGES = ["revolut", "binance", "binance-crashbot"]


def fetch_all_trades(exchange: str) -> list[dict]:
    """Fetch ALL trades for a given exchange."""
    docs = db.collection("trades").where("exchange", "==", exchange).stream()
    return [{**d.to_dict(), "_id": d.id} for d in docs]


def fetch_daily_snapshots(exchange: str) -> list[dict]:
    """Fetch ALL daily_snapshots for a given exchange."""
    docs = (
        db.collection("daily_snapshots")
        .where("exchange", "==", exchange)
        .order_by("date")
        .stream()
    )
    return [{**d.to_dict(), "_id": d.id} for d in docs]


def parse_dt(val) -> "Optional[datetime]":
    if val is None:
        return None
    if isinstance(val, datetime):
        return val
    try:
        return datetime.fromisoformat(str(val))
    except Exception:
        return None


def safe_float(val, default=0.0) -> float:
    try:
        return float(val)
    except (TypeError, ValueError):
        return default


def fmt(val, decimals=2) -> str:
    if val is None:
        return "N/A"
    return f"{val:,.{decimals}f}"


def fmt_pnl(val, decimals=2) -> str:
    if val is None:
        return "N/A"
    return f"${val:+,.{decimals}f}"


def pct(val, decimals=1) -> str:
    if val is None:
        return "N/A"
    return f"{val * 100:.{decimals}f}%"


# ── Analysis ─────────────────────────────────────────────────────────────────

def analyze_exchange(exchange: str, all_trades: list[dict]):
    print(f"\n{'═' * 80}")
    print(f"  📊 EXCHANGE: {exchange.upper()}")
    print(f"{'═' * 80}")

    # ── Separate closed vs open ──
    closed = [t for t in all_trades if t.get("closed_at") is not None]
    open_trades = [t for t in all_trades if t.get("status") in ("OPEN", "ZERO_RISK", "TRAILING")]

    # Sort closed by closed_at
    closed.sort(key=lambda t: str(t.get("closed_at", "")))

    total_closed = len(closed)
    total_open = len(open_trades)

    # ── P&L ──
    pnl_values = [safe_float(t.get("pnl_usd")) for t in closed if t.get("pnl_usd") is not None]
    pnl_net_values = [safe_float(t.get("pnl_net_usd")) for t in closed if t.get("pnl_net_usd") is not None]

    total_pnl_gross = sum(pnl_values) if pnl_values else 0
    total_pnl_net = sum(pnl_net_values) if pnl_net_values else 0

    wins = [p for p in pnl_values if p > 0]
    losses = [p for p in pnl_values if p < 0]
    breakeven = [p for p in pnl_values if p == 0]

    win_rate = len(wins) / total_closed if total_closed > 0 else 0
    avg_win = mean(wins) if wins else 0
    avg_loss = mean(losses) if losses else 0
    median_win = median(wins) if wins else 0
    median_loss = median(losses) if losses else 0

    sum_wins = sum(wins) if wins else 0
    sum_losses = abs(sum(losses)) if losses else 0
    profit_factor = sum_wins / sum_losses if sum_losses > 0 else float("inf")

    # Net P&L stats
    wins_net = [safe_float(t.get("pnl_net_usd")) for t in closed if safe_float(t.get("pnl_net_usd")) > 0]
    losses_net = [safe_float(t.get("pnl_net_usd")) for t in closed if safe_float(t.get("pnl_net_usd")) < 0]
    sum_wins_net = sum(wins_net) if wins_net else 0
    sum_losses_net = abs(sum(losses_net)) if losses_net else 0
    profit_factor_net = sum_wins_net / sum_losses_net if sum_losses_net > 0 else float("inf")

    # ── Max drawdown from cumulative PnL ──
    cum_pnl = 0.0
    peak = 0.0
    max_dd = 0.0
    max_dd_date = ""
    for t in closed:
        cum_pnl += safe_float(t.get("pnl_usd"))
        if cum_pnl > peak:
            peak = cum_pnl
        dd = peak - cum_pnl
        if dd > max_dd:
            max_dd = dd
            max_dd_date = str(t.get("closed_at", ""))[:10]

    # ── Best / Worst ──
    best_trade = max(closed, key=lambda t: safe_float(t.get("pnl_usd")), default=None)
    worst_trade = min(closed, key=lambda t: safe_float(t.get("pnl_usd")), default=None)

    # ── Last equity ──
    last_equity_trade = max(closed, key=lambda t: str(t.get("closed_at", "")), default=None)
    last_equity = safe_float(last_equity_trade.get("equity_after")) if last_equity_trade else None

    # ── Date range ──
    first_opened = min(closed, key=lambda t: str(t.get("opened_at", "")), default=None)
    last_closed_trade = max(closed, key=lambda t: str(t.get("closed_at", "")), default=None)

    first_date = str(first_opened.get("opened_at", ""))[:10] if first_opened else "N/A"
    last_date = str(last_closed_trade.get("closed_at", ""))[:10] if last_closed_trade else "N/A"

    # ── Holding time ──
    holding_times = [safe_float(t.get("holding_time_hours")) for t in closed if t.get("holding_time_hours") is not None]
    avg_holding = mean(holding_times) if holding_times else None
    median_holding = median(holding_times) if holding_times else None
    max_holding = max(holding_times) if holding_times else None
    min_holding = min(holding_times) if holding_times else None

    # ── Fees ──
    total_fees = sum(safe_float(t.get("fees_total")) for t in closed if t.get("fees_total") is not None)

    # ── PRINT SUMMARY ──
    print(f"\n  ── RÉSUMÉ GLOBAL ──")
    print(f"  Trades clôturés       : {total_closed}")
    print(f"  Positions ouvertes    : {total_open}")
    print(f"  Première trade        : {first_date}")
    print(f"  Dernière trade        : {last_date}")
    print(f"  P&L brut total        : {fmt_pnl(total_pnl_gross)}")
    print(f"  P&L net total (fees)  : {fmt_pnl(total_pnl_net)}")
    print(f"  Fees totales          : ${fmt(total_fees)}")
    print(f"  Last equity           : ${fmt(last_equity)}")
    print(f"  Win rate              : {pct(win_rate)}")
    print(f"  Wins / Losses / BE    : {len(wins)} / {len(losses)} / {len(breakeven)}")
    print(f"  Avg win               : {fmt_pnl(avg_win)}")
    print(f"  Avg loss              : {fmt_pnl(avg_loss)}")
    print(f"  Median win            : {fmt_pnl(median_win)}")
    print(f"  Median loss           : {fmt_pnl(median_loss)}")
    print(f"  Profit factor (brut)  : {fmt(profit_factor)}")
    print(f"  Profit factor (net)   : {fmt(profit_factor_net)}")
    print(f"  Max drawdown          : ${fmt(max_dd)} (around {max_dd_date})")
    print(f"  Avg holding time      : {fmt(avg_holding)} h")
    print(f"  Median holding time   : {fmt(median_holding)} h")
    print(f"  Min/Max holding       : {fmt(min_holding)} h / {fmt(max_holding)} h")

    if best_trade:
        print(f"\n  🏆 Best trade  : {best_trade.get('symbol')} | {fmt_pnl(safe_float(best_trade.get('pnl_usd')))} "
              f"({best_trade.get('side')}) | {str(best_trade.get('closed_at',''))[:10]}")
    if worst_trade:
        print(f"  💀 Worst trade : {worst_trade.get('symbol')} | {fmt_pnl(safe_float(worst_trade.get('pnl_usd')))} "
              f"({worst_trade.get('side')}) | {str(worst_trade.get('closed_at',''))[:10]}")

    # ── OPEN POSITIONS ──
    if open_trades:
        print(f"\n  ── POSITIONS OUVERTES ({total_open}) ──")
        for t in open_trades:
            status = t.get("status", "?")
            sym = t.get("symbol", "?")
            side = t.get("side", "?")
            entry = safe_float(t.get("entry_filled", t.get("entry_expected")))
            sl = safe_float(t.get("sl_price"))
            tp = safe_float(t.get("tp_price"))
            size = safe_float(t.get("size"))
            opened = str(t.get("opened_at", ""))[:16]
            strat = t.get("signal_type", "?")
            trailing = t.get("trailing_steps", "")
            trail_info = f" | trail_step={trailing}" if trailing else ""
            print(f"    [{status}] {sym} {side} @ {entry} | SL={sl} TP={tp} | size={size} | {strat} | opened={opened}{trail_info}")

    # ── BY SYMBOL ──
    print(f"\n  ── PAR SYMBOLE ──")
    by_symbol = defaultdict(list)
    for t in closed:
        by_symbol[t.get("symbol", "?")].append(t)

    for sym in sorted(by_symbol.keys()):
        trades_sym = by_symbol[sym]
        pnls = [safe_float(t.get("pnl_usd")) for t in trades_sym if t.get("pnl_usd") is not None]
        w = [p for p in pnls if p > 0]
        l = [p for p in pnls if p < 0]
        wr = len(w) / len(trades_sym) if trades_sym else 0
        total_p = sum(pnls)
        print(f"    {sym:15s} | {len(trades_sym):3d} trades | PnL={fmt_pnl(total_p):>12s} | WR={pct(wr):>7s} | "
              f"AvgW={fmt_pnl(mean(w)) if w else 'N/A':>10s} | AvgL={fmt_pnl(mean(l)) if l else 'N/A':>10s}")

    # ── BY STRATEGY ──
    print(f"\n  ── PAR STRATÉGIE ──")
    by_strategy = defaultdict(list)
    for t in closed:
        by_strategy[t.get("signal_type", "UNKNOWN")].append(t)

    for strat in sorted(by_strategy.keys()):
        trades_s = by_strategy[strat]
        pnls = [safe_float(t.get("pnl_usd")) for t in trades_s if t.get("pnl_usd") is not None]
        w = [p for p in pnls if p > 0]
        l = [p for p in pnls if p < 0]
        wr = len(w) / len(trades_s) if trades_s else 0
        total_p = sum(pnls)
        pf = sum(w) / abs(sum(l)) if l and sum(l) != 0 else float("inf")
        print(f"    {strat:15s} | {len(trades_s):3d} trades | PnL={fmt_pnl(total_p):>12s} | WR={pct(wr):>7s} | PF={fmt(pf)}")

    # ── BY EXIT REASON ──
    print(f"\n  ── PAR RAISON DE SORTIE ──")
    by_exit = defaultdict(list)
    for t in closed:
        by_exit[t.get("exit_reason", "UNKNOWN")].append(t)

    for reason in sorted(by_exit.keys()):
        trades_r = by_exit[reason]
        pnls = [safe_float(t.get("pnl_usd")) for t in trades_r if t.get("pnl_usd") is not None]
        total_p = sum(pnls)
        w = [p for p in pnls if p > 0]
        wr = len(w) / len(trades_r) if trades_r else 0
        print(f"    {str(reason):30s} | {len(trades_r):3d} trades | PnL={fmt_pnl(total_p):>12s} | WR={pct(wr):>7s}")

    # ── CRASHBOT-SPECIFIC: trail_steps distribution ──
    if exchange == "binance-crashbot":
        trail_steps_vals = [safe_float(t.get("trailing_steps")) for t in closed if t.get("trailing_steps") is not None]
        if trail_steps_vals:
            print(f"\n  ── CRASHBOT: TRAILING STEPS ──")
            print(f"    Trades avec trailing  : {len(trail_steps_vals)}")
            print(f"    Avg trail steps       : {fmt(mean(trail_steps_vals))}")
            print(f"    Max trail steps       : {fmt(max(trail_steps_vals))}")
            # Distribution
            step_counts = defaultdict(int)
            for s in trail_steps_vals:
                step_counts[int(s)] += 1
            for step in sorted(step_counts.keys()):
                print(f"    Step {step}: {step_counts[step]} trades")

    # ── MONTHLY BREAKDOWN ──
    print(f"\n  ── BREAKDOWN MENSUEL ──")
    by_month = defaultdict(list)
    for t in closed:
        closed_at = str(t.get("closed_at", ""))[:7]  # YYYY-MM
        if closed_at and len(closed_at) >= 7:
            by_month[closed_at].append(t)

    print(f"    {'Mois':>7s} | {'Trades':>6s} | {'P&L brut':>12s} | {'P&L net':>12s} | {'WR':>7s} | {'PF':>6s} | {'Wins':>4s} | {'Losses':>6s} | {'AvgWin':>10s} | {'AvgLoss':>10s}")
    print(f"    {'-' * 100}")

    monthly_cum_pnl = 0
    for month in sorted(by_month.keys()):
        trades_m = by_month[month]
        pnls = [safe_float(t.get("pnl_usd")) for t in trades_m if t.get("pnl_usd") is not None]
        pnls_net = [safe_float(t.get("pnl_net_usd")) for t in trades_m if t.get("pnl_net_usd") is not None]
        w = [p for p in pnls if p > 0]
        l = [p for p in pnls if p < 0]
        wr = len(w) / len(trades_m) if trades_m else 0
        pf = sum(w) / abs(sum(l)) if l and sum(l) != 0 else float("inf")
        total_gross = sum(pnls)
        total_net = sum(pnls_net) if pnls_net else 0
        monthly_cum_pnl += total_gross
        avg_w = mean(w) if w else 0
        avg_l = mean(l) if l else 0
        print(f"    {month:>7s} | {len(trades_m):>6d} | {fmt_pnl(total_gross):>12s} | {fmt_pnl(total_net):>12s} | "
              f"{pct(wr):>7s} | {fmt(pf):>6s} | {len(w):>4d} | {len(l):>6d} | {fmt_pnl(avg_w):>10s} | {fmt_pnl(avg_l):>10s}")

    print(f"    {'-' * 100}")
    print(f"    Cumul P&L brut mensuel : {fmt_pnl(monthly_cum_pnl)}")

    # ── WEEKLY BREAKDOWN (last 8 weeks) ──
    print(f"\n  ── BREAKDOWN HEBDOMADAIRE (dernières 8 semaines) ──")
    by_week = defaultdict(list)
    for t in closed:
        dt = parse_dt(t.get("closed_at"))
        if dt:
            week = dt.strftime("%Y-W%W")
            by_week[week].append(t)

    weeks_sorted = sorted(by_week.keys())[-8:]
    for week in weeks_sorted:
        trades_w = by_week[week]
        pnls = [safe_float(t.get("pnl_usd")) for t in trades_w if t.get("pnl_usd") is not None]
        w = [p for p in pnls if p > 0]
        l = [p for p in pnls if p < 0]
        wr = len(w) / len(trades_w) if trades_w else 0
        print(f"    {week} | {len(trades_w):>3d} trades | PnL={fmt_pnl(sum(pnls)):>12s} | WR={pct(wr):>7s}")

    # ── SIDE BREAKDOWN ──
    print(f"\n  ── PAR SIDE (BUY/SELL) ──")
    for side in ("buy", "sell"):
        trades_side = [t for t in closed if t.get("side") == side]
        if not trades_side:
            continue
        pnls = [safe_float(t.get("pnl_usd")) for t in trades_side if t.get("pnl_usd") is not None]
        w = [p for p in pnls if p > 0]
        l = [p for p in pnls if p < 0]
        wr = len(w) / len(trades_side) if trades_side else 0
        print(f"    {side.upper():>4s} | {len(trades_side):>4d} trades | PnL={fmt_pnl(sum(pnls)):>12s} | WR={pct(wr):>7s}")

    # ── TOP 10 BEST / WORST TRADES ──
    sorted_by_pnl = sorted(closed, key=lambda t: safe_float(t.get("pnl_usd")), reverse=True)

    print(f"\n  ── TOP 10 BEST TRADES ──")
    for t in sorted_by_pnl[:10]:
        print(f"    {t.get('symbol','?'):12s} | {t.get('side','?'):4s} | PnL={fmt_pnl(safe_float(t.get('pnl_usd'))):>10s} | "
              f"{t.get('signal_type','?'):8s} | {t.get('exit_reason','?'):20s} | {str(t.get('closed_at',''))[:10]}")

    print(f"\n  ── TOP 10 WORST TRADES ──")
    for t in sorted_by_pnl[-10:]:
        print(f"    {t.get('symbol','?'):12s} | {t.get('side','?'):4s} | PnL={fmt_pnl(safe_float(t.get('pnl_usd'))):>10s} | "
              f"{t.get('signal_type','?'):8s} | {t.get('exit_reason','?'):20s} | {str(t.get('closed_at',''))[:10]}")

    return closed


def analyze_daily_snapshots(exchange: str, snapshots: list[dict]):
    if not snapshots:
        print(f"\n  ── DAILY SNAPSHOTS: Aucun ──")
        return

    print(f"\n  ── DAILY SNAPSHOTS ({len(snapshots)} jours) ──")

    # Equity evolution
    equities = [(s.get("date", "?"), safe_float(s.get("equity")), safe_float(s.get("daily_pnl")), s.get("trades_today", 0)) for s in snapshots]
    equities.sort(key=lambda x: x[0])

    if equities:
        first_eq = equities[0]
        last_eq = equities[-1]
        min_eq = min(equities, key=lambda x: x[1])
        max_eq = max(equities, key=lambda x: x[1])

        print(f"    Première date     : {first_eq[0]} | Equity = ${fmt(first_eq[1])}")
        print(f"    Dernière date     : {last_eq[0]} | Equity = ${fmt(last_eq[1])}")
        print(f"    Min equity        : {min_eq[0]} | Equity = ${fmt(min_eq[1])}")
        print(f"    Max equity        : {max_eq[0]} | Equity = ${fmt(max_eq[1])}")

        # Equity drawdown
        peak_eq = 0
        max_eq_dd = 0
        max_eq_dd_date = ""
        for date, eq, _, _ in equities:
            if eq > peak_eq:
                peak_eq = eq
            dd = peak_eq - eq
            if dd > max_eq_dd:
                max_eq_dd = dd
                max_eq_dd_date = date
        print(f"    Max DD (equity)   : ${fmt(max_eq_dd)} (at {max_eq_dd_date})")

        # Daily P&L stats
        daily_pnls = [e[2] for e in equities if e[2] != 0]
        if daily_pnls:
            print(f"    Avg daily P&L     : {fmt_pnl(mean(daily_pnls))}")
            print(f"    Best day          : {fmt_pnl(max(daily_pnls))}")
            print(f"    Worst day         : {fmt_pnl(min(daily_pnls))}")

        # Last 14 days detail
        print(f"\n    Derniers 14 jours:")
        print(f"    {'Date':>12s} | {'Equity':>12s} | {'Daily PnL':>12s} | {'Trades':>6s}")
        print(f"    {'-' * 52}")
        for date, eq, dpnl, tc in equities[-14:]:
            print(f"    {date:>12s} | ${fmt(eq):>11s} | {fmt_pnl(dpnl):>12s} | {tc:>6}")


# ── MAIN ─────────────────────────────────────────────────────────────────────

def main():
    print("=" * 80)
    print("  🤖 TRADEX — ANALYSE COMPLÈTE DES BOTS FIREBASE")
    print(f"  Date: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}")
    print("=" * 80)

    all_data = {}

    for exchange in EXCHANGES:
        print(f"\n⏳ Fetching trades for {exchange}...")
        trades = fetch_all_trades(exchange)
        print(f"   → {len(trades)} trades fetched")

        print(f"⏳ Fetching daily snapshots for {exchange}...")
        snapshots = fetch_daily_snapshots(exchange)
        print(f"   → {len(snapshots)} snapshots fetched")

        all_data[exchange] = {"trades": trades, "snapshots": snapshots}

    # Analyze each exchange
    for exchange in EXCHANGES:
        closed = analyze_exchange(exchange, all_data[exchange]["trades"])
        analyze_daily_snapshots(exchange, all_data[exchange]["snapshots"])

    # ── CROSS-EXCHANGE SUMMARY ──
    print(f"\n{'═' * 80}")
    print(f"  📊 RÉSUMÉ CROSS-EXCHANGE")
    print(f"{'═' * 80}")

    grand_total_pnl = 0
    grand_total_trades = 0
    for exchange in EXCHANGES:
        trades = all_data[exchange]["trades"]
        closed = [t for t in trades if t.get("closed_at") is not None]
        pnl = sum(safe_float(t.get("pnl_usd")) for t in closed if t.get("pnl_usd") is not None)
        pnl_net = sum(safe_float(t.get("pnl_net_usd")) for t in closed if t.get("pnl_net_usd") is not None)
        open_c = len([t for t in trades if t.get("status") in ("OPEN", "ZERO_RISK", "TRAILING")])
        grand_total_pnl += pnl
        grand_total_trades += len(closed)

        # Last equity
        last_eq_trade = max(closed, key=lambda t: str(t.get("closed_at", "")), default=None)
        last_eq = safe_float(last_eq_trade.get("equity_after")) if last_eq_trade else 0

        print(f"  {exchange:20s} | {len(closed):>4d} closed | {open_c:>2d} open | PnL brut={fmt_pnl(pnl):>12s} | PnL net={fmt_pnl(pnl_net):>12s} | Equity=${fmt(last_eq)}")

    print(f"\n  TOTAL P&L brut tous bots : {fmt_pnl(grand_total_pnl)}")
    print(f"  TOTAL trades clôturés    : {grand_total_trades}")
    print(f"\n{'═' * 80}")
    print("  ✅ Analyse terminée")
    print(f"{'═' * 80}\n")


if __name__ == "__main__":
    main()
