"""
Comprehensive Firebase data dump for TradeX bots.
Fetches ALL trades, daily snapshots, open positions with full detail.
"""

import json
import os
import sys
from collections import defaultdict
from datetime import datetime, timezone

from google.cloud import firestore
from google.oauth2 import service_account

# ── Firebase connection ──────────────────────────────────────────────────────

CRED_PATH = os.path.join(os.path.dirname(__file__), "firebase-credentials.json")
credentials = service_account.Credentials.from_service_account_file(CRED_PATH)
db = firestore.Client(project="satochi-d38ec", credentials=credentials)

EXCHANGES = ["revolut", "binance", "binance-crashbot"]


def safe_float(val, default=0.0):
    try:
        return float(val)
    except (TypeError, ValueError):
        return default


def fmt_pnl(val):
    if val is None:
        return "N/A"
    return f"${val:+,.4f}"


def fetch_all_trades(exchange):
    docs = db.collection("trades").where("exchange", "==", exchange).stream()
    return [{**d.to_dict(), "_id": d.id} for d in docs]


def fetch_daily_snapshots(exchange):
    docs = (
        db.collection("daily_snapshots")
        .where("exchange", "==", exchange)
        .order_by("date")
        .stream()
    )
    return [{**d.to_dict(), "_id": d.id} for d in docs]


# ═══════════════════════════════════════════════════════════════════════════════
#  1 & 2 — CLOSED TRADES (all details)
# ═══════════════════════════════════════════════════════════════════════════════

def print_closed_trades(exchange, trades):
    closed = [t for t in trades if t.get("closed_at") is not None]
    closed.sort(key=lambda t: str(t.get("closed_at", "")))

    print(f"\n{'═' * 120}")
    print(f"  CLOSED TRADES — {exchange.upper()} ({len(closed)} trades)")
    print(f"{'═' * 120}")

    if not closed:
        print("  (aucun trade clôturé)")
        return closed

    # Header
    if exchange == "binance" or exchange == "binance-crashbot":
        print(f"  {'#':>3} | {'Symbol':>12} | {'Side':>4} | {'Strategy':>10} | {'PnL USD':>12} | {'PnL Net':>12} | "
              f"{'Exit Reason':>25} | {'Holding h':>9} | {'Entry':>12} | {'SL':>12} | {'TP':>12} | "
              f"{'Fees Tot':>10} | {'Maker/Taker':>11} | {'ExitFill':>10} | {'Opened At':>20} | {'Closed At':>20}")
        print(f"  {'-' * 210}")
    else:
        print(f"  {'#':>3} | {'Symbol':>12} | {'Side':>4} | {'Strategy':>10} | {'PnL USD':>12} | {'PnL Net':>12} | "
              f"{'Exit Reason':>25} | {'Holding h':>9} | {'Entry':>12} | {'SL':>12} | {'TP':>12} | "
              f"{'Opened At':>20} | {'Closed At':>20}")
        print(f"  {'-' * 180}")

    for i, t in enumerate(closed, 1):
        sym = t.get("symbol", "?")
        side = t.get("side", "?")
        strat = t.get("signal_type", "?")
        pnl = safe_float(t.get("pnl_usd"))
        pnl_net = safe_float(t.get("pnl_net_usd"))
        reason = str(t.get("exit_reason", "?"))
        holding = t.get("holding_time_hours")
        holding_s = f"{holding:.2f}" if holding is not None else "N/A"
        entry = safe_float(t.get("entry_filled", t.get("entry_expected")))
        sl = t.get("sl_price")
        sl_s = f"{safe_float(sl):.4f}" if sl is not None else "N/A"
        tp = t.get("tp_price")
        tp_s = f"{safe_float(tp):.4f}" if tp is not None else "N/A"
        opened = str(t.get("opened_at", ""))[:19]
        closed_at = str(t.get("closed_at", ""))[:19]

        if exchange in ("binance", "binance-crashbot"):
            fees = t.get("fees_total")
            fees_s = f"{safe_float(fees):.4f}" if fees is not None else "N/A"
            mot = t.get("maker_or_taker", "?")
            eft = t.get("exit_fill_type", "?")
            print(f"  {i:>3} | {sym:>12} | {side:>4} | {strat:>10} | {fmt_pnl(pnl):>12} | {fmt_pnl(pnl_net):>12} | "
                  f"{reason:>25} | {holding_s:>9} | {entry:>12.4f} | {sl_s:>12} | {tp_s:>12} | "
                  f"{fees_s:>10} | {str(mot):>11} | {str(eft):>10} | {opened:>20} | {closed_at:>20}")
        else:
            print(f"  {i:>3} | {sym:>12} | {side:>4} | {strat:>10} | {fmt_pnl(pnl):>12} | {fmt_pnl(pnl_net):>12} | "
                  f"{reason:>25} | {holding_s:>9} | {entry:>12.4f} | {sl_s:>12} | {tp_s:>12} | "
                  f"{opened:>20} | {closed_at:>20}")

    # ── PnL par symbole ──
    print(f"\n  ── PnL PAR SYMBOLE ({exchange.upper()}) ──")
    by_sym = defaultdict(lambda: {"pnl": 0, "pnl_net": 0, "count": 0, "wins": 0, "losses": 0})
    for t in closed:
        s = t.get("symbol", "?")
        p = safe_float(t.get("pnl_usd"))
        pn = safe_float(t.get("pnl_net_usd"))
        by_sym[s]["pnl"] += p
        by_sym[s]["pnl_net"] += pn
        by_sym[s]["count"] += 1
        if p > 0:
            by_sym[s]["wins"] += 1
        elif p < 0:
            by_sym[s]["losses"] += 1

    print(f"  {'Symbol':>15} | {'Trades':>6} | {'Wins':>4} | {'Losses':>6} | {'WR':>7} | {'PnL Brut':>14} | {'PnL Net':>14} | {'Status':>12}")
    print(f"  {'-' * 100}")
    for sym in sorted(by_sym.keys()):
        d = by_sym[sym]
        wr = d["wins"] / d["count"] if d["count"] > 0 else 0
        status = "✅ PROFIT" if d["pnl"] > 0 else ("🔴 LOSS" if d["pnl"] < 0 else "⚪ BE")
        print(f"  {sym:>15} | {d['count']:>6} | {d['wins']:>4} | {d['losses']:>6} | {wr:>6.1%} | {fmt_pnl(d['pnl']):>14} | {fmt_pnl(d['pnl_net']):>14} | {status:>12}")

    total_pnl = sum(d["pnl"] for d in by_sym.values())
    total_pnl_net = sum(d["pnl_net"] for d in by_sym.values())
    print(f"  {'TOTAL':>15} | {len(closed):>6} |      |        |        | {fmt_pnl(total_pnl):>14} | {fmt_pnl(total_pnl_net):>14} |")

    return closed


# ═══════════════════════════════════════════════════════════════════════════════
#  3 — CROSS-EXCHANGE COMPARISON
# ═══════════════════════════════════════════════════════════════════════════════

def compare_exchanges(all_data):
    print(f"\n{'═' * 120}")
    print(f"  CROSS-EXCHANGE PAIR COMPARISON (Revolut vs Binance)")
    print(f"{'═' * 120}")

    rev_closed = [t for t in all_data.get("revolut", []) if t.get("closed_at")]
    bin_closed = [t for t in all_data.get("binance", []) if t.get("closed_at")]

    rev_syms = defaultdict(lambda: {"pnl": 0, "count": 0})
    bin_syms = defaultdict(lambda: {"pnl": 0, "count": 0})

    for t in rev_closed:
        s = t.get("symbol", "?")
        rev_syms[s]["pnl"] += safe_float(t.get("pnl_usd"))
        rev_syms[s]["count"] += 1

    for t in bin_closed:
        s = t.get("symbol", "?")
        # Normalize symbol: Binance uses BTCUSDT, Revolut uses BTC-USD
        bin_syms[s]["pnl"] += safe_float(t.get("pnl_usd"))
        bin_syms[s]["count"] += 1

    all_syms = set(list(rev_syms.keys()) + list(bin_syms.keys()))

    print(f"\n  {'Symbol':>15} | {'Rev Trades':>10} | {'Rev PnL':>14} | {'Bin Trades':>10} | {'Bin PnL':>14} | {'Overlap':>8}")
    print(f"  {'-' * 85}")

    for sym in sorted(all_syms):
        r = rev_syms.get(sym)
        b = bin_syms.get(sym)
        r_count = r["count"] if r else 0
        r_pnl = r["pnl"] if r else 0
        b_count = b["count"] if b else 0
        b_pnl = b["pnl"] if b else 0
        overlap = "✅ BOTH" if r and b else ("REV" if r else "BIN")
        print(f"  {sym:>15} | {r_count:>10} | {fmt_pnl(r_pnl):>14} | {b_count:>10} | {fmt_pnl(b_pnl):>14} | {overlap:>8}")

    # Pairs on both
    common = set(rev_syms.keys()) & set(bin_syms.keys())
    if common:
        print(f"\n  ── PAIRES COMMUNES ({len(common)}) — Comparaison directe ──")
        for sym in sorted(common):
            r = rev_syms[sym]
            b = bin_syms[sym]
            better = "Revolut" if r["pnl"] > b["pnl"] else "Binance"
            diff = abs(r["pnl"] - b["pnl"])
            print(f"    {sym}: Revolut={fmt_pnl(r['pnl'])} ({r['count']} trades) | "
                  f"Binance={fmt_pnl(b['pnl'])} ({b['count']} trades) | "
                  f"Better: {better} (+${diff:.2f})")
    else:
        print(f"\n  Aucune paire commune trouvée (les formats de symboles diffèrent: Revolut=BTC-USD, Binance=BTCUSDT)")


# ═══════════════════════════════════════════════════════════════════════════════
#  4 — DAILY SNAPSHOTS (ALL)
# ═══════════════════════════════════════════════════════════════════════════════

def print_all_daily_snapshots(exchange, snapshots):
    print(f"\n{'═' * 120}")
    print(f"  ALL DAILY SNAPSHOTS — {exchange.upper()} ({len(snapshots)} entries)")
    print(f"{'═' * 120}")

    if not snapshots:
        print("  (aucun snapshot)")
        return

    # Detect all keys present in snapshots
    all_keys = set()
    for s in snapshots:
        all_keys.update(s.keys())
    all_keys -= {"_id"}
    
    # Print header — show all available fields
    print(f"\n  Available fields: {sorted(all_keys)}\n")

    # Print each snapshot with all its data
    for i, s in enumerate(snapshots, 1):
        date = s.get("date", "?")
        equity = s.get("equity")
        equity_s = f"${safe_float(equity):,.2f}" if equity is not None else "N/A"
        daily_pnl = s.get("daily_pnl")
        daily_pnl_s = fmt_pnl(safe_float(daily_pnl)) if daily_pnl is not None else "N/A"
        trades_today = s.get("trades_today", s.get("closed_today", "?"))
        
        # Extra fields
        extras = []
        for k in sorted(all_keys):
            if k in ("date", "equity", "daily_pnl", "trades_today", "closed_today", "exchange", "_id"):
                continue
            v = s.get(k)
            if v is not None:
                extras.append(f"{k}={v}")

        extra_str = " | ".join(extras) if extras else ""
        print(f"  {i:>4}. {date:>12} | Equity={equity_s:>12} | DailyPnL={daily_pnl_s:>12} | Trades={trades_today} | {extra_str}")


# ═══════════════════════════════════════════════════════════════════════════════
#  5 — CONFIG/PARAMETERS FROM TRADES
# ═══════════════════════════════════════════════════════════════════════════════

def print_config_from_trades(exchange, trades):
    print(f"\n{'═' * 120}")
    print(f"  CONFIG/PARAMETERS VISIBLE IN TRADES — {exchange.upper()}")
    print(f"{'═' * 120}")

    if not trades:
        print("  (aucun trade)")
        return

    # Collect unique values for config-like fields
    config_fields = [
        "risk_pct", "risk_amount_usd", "max_allowed_risk", "maker_or_taker",
        "maker_wait_seconds", "bot_version", "dry_run", "signal_type",
        "portfolio_risk_before", "sl_price_effective", "fiat_balance_at_entry",
        "equity_at_entry", "entry_slippage_pct",
    ]

    for field in config_fields:
        values = set()
        for t in trades:
            v = t.get(field)
            if v is not None:
                values.add(str(v))
        if values:
            if len(values) <= 20:
                print(f"  {field:>30s} : {', '.join(sorted(values))}")
            else:
                print(f"  {field:>30s} : ({len(values)} unique values)")

    # Also show a complete dump of one trade's keys to understand schema
    print(f"\n  ── FULL SCHEMA (all keys in first trade) ──")
    sample = trades[0]
    for k in sorted(sample.keys()):
        v = sample[k]
        vtype = type(v).__name__
        v_display = str(v)[:100]
        print(f"    {k:>30s} : [{vtype}] {v_display}")

    # Risk_usd per trade distribution
    risk_usds = [safe_float(t.get("risk_amount_usd")) for t in trades if t.get("risk_amount_usd") is not None]
    if risk_usds:
        print(f"\n  ── RISK USD PER TRADE ──")
        print(f"    Min : ${min(risk_usds):.2f}")
        print(f"    Max : ${max(risk_usds):.2f}")
        print(f"    Avg : ${sum(risk_usds)/len(risk_usds):.2f}")
        # Last 10
        last_10 = [safe_float(t.get("risk_amount_usd")) for t in sorted(trades, key=lambda t: str(t.get("opened_at", "")))[-10:] if t.get("risk_amount_usd") is not None]
        print(f"    Last 10 trades : {['$'+f'{r:.2f}' for r in last_10]}")


# ═══════════════════════════════════════════════════════════════════════════════
#  6 — OPEN POSITIONS
# ═══════════════════════════════════════════════════════════════════════════════

def print_open_positions(exchange, trades):
    open_trades = [t for t in trades if t.get("status") in ("OPEN", "ZERO_RISK", "TRAILING", "PENDING")]

    print(f"\n{'═' * 120}")
    print(f"  OPEN POSITIONS — {exchange.upper()} ({len(open_trades)} positions)")
    print(f"{'═' * 120}")

    if not open_trades:
        print("  (aucune position ouverte)")
        return

    for i, t in enumerate(open_trades, 1):
        print(f"\n  ── Position #{i} ──")
        sym = t.get("symbol", "?")
        side = t.get("side", "?")
        status = t.get("status", "?")
        entry = t.get("entry_filled", t.get("entry_expected"))
        sl = t.get("sl_price")
        tp = t.get("tp_price")
        size = t.get("size")
        size_usd = t.get("size_usd")
        risk_usd = t.get("risk_amount_usd")
        opened = str(t.get("opened_at", ""))[:19]
        strat = t.get("signal_type", "?")
        is_zr = t.get("is_zero_risk_applied")

        print(f"    Symbol          : {sym}")
        print(f"    Side            : {side}")
        print(f"    Status          : {status}")
        print(f"    Strategy        : {strat}")
        print(f"    Entry Filled    : {entry}")
        print(f"    SL Price        : {sl}")
        print(f"    TP Price        : {tp}")
        print(f"    Size            : {size}")
        print(f"    Size USD        : {size_usd}")
        print(f"    Risk USD        : {risk_usd}")
        print(f"    Zero Risk       : {is_zr}")
        print(f"    Opened At       : {opened}")

        # Crashbot-specific fields
        if exchange == "binance-crashbot":
            trail_steps = t.get("trailing_steps")
            peak_price = t.get("peak_price")
            trail_tp = t.get("trail_tp")
            trail_activation = t.get("trail_activation_price")
            print(f"    Trail Steps     : {trail_steps}")
            print(f"    Peak Price      : {peak_price}")
            print(f"    Trail TP        : {trail_tp}")
            print(f"    Trail Activation: {trail_activation}")

        # Dump ALL fields for full visibility
        print(f"    ── ALL FIELDS ──")
        for k in sorted(t.keys()):
            if k == "_id":
                continue
            print(f"      {k:>30s} = {t[k]}")


# ═══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    print("=" * 120)
    print("  🔥 TRADEX — FIREBASE DETAILED DATA DUMP")
    print(f"  Date: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}")
    print("=" * 120)

    all_trades_data = {}
    all_snapshots_data = {}

    # Fetch everything
    for exchange in EXCHANGES:
        print(f"\n⏳ Fetching {exchange}...")
        trades = fetch_all_trades(exchange)
        snapshots = fetch_daily_snapshots(exchange)
        all_trades_data[exchange] = trades
        all_snapshots_data[exchange] = snapshots
        print(f"   → {len(trades)} trades, {len(snapshots)} snapshots")

    # ═══ 1 & 2: Closed trades for each exchange ═══
    for exchange in EXCHANGES:
        print_closed_trades(exchange, all_trades_data[exchange])

    # ═══ 3: Cross-exchange comparison ═══
    compare_exchanges(all_trades_data)

    # ═══ 4: ALL daily snapshots ═══
    for exchange in EXCHANGES:
        print_all_daily_snapshots(exchange, all_snapshots_data[exchange])

    # ═══ 5: Config/parameters from binance trades ═══
    print_config_from_trades("binance", all_trades_data["binance"])
    print_config_from_trades("binance-crashbot", all_trades_data["binance-crashbot"])

    # ═══ 6: Open positions for all exchanges ═══
    for exchange in EXCHANGES:
        print_open_positions(exchange, all_trades_data[exchange])

    print(f"\n{'=' * 120}")
    print("  ✅ DUMP COMPLETE")
    print(f"{'=' * 120}\n")


if __name__ == "__main__":
    main()
