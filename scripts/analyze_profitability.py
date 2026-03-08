#!/usr/bin/env python3
"""
Analyse de rentabilité des bots TradeX.

Lit Firebase Firestore et produit un rapport complet :
- Trades clôturés : Win Rate, PnL, Profit Factor, avg W/L
- Positions ouvertes : entry, SL, taille, notionnel, durée
- Daily snapshots : évolution de l'equity
- Allocation Binance : régime courant
- Cycle Infinity : état DCA inversé
"""
from datetime import datetime, timezone
from google.cloud import firestore

db = firestore.Client()

# ─────────────────────────────────────────────────────────────────────────────
# 1. TRADES
# ─────────────────────────────────────────────────────────────────────────────
all_trades = list(db.collection("trades").stream())
print(f"Total trades en base : {len(all_trades)}")

# Bot label mapping
EXCHANGE_LABELS = {
    "binance": "Trail Range",
    "binance-crashbot": "CrashBot (DipBuy)",
    "revolut": "Momentum (Revolut X)",
}

by_exchange: dict = {}
for t in all_trades:
    d = t.to_dict()
    exch = d.get("exchange", "unknown")
    status = (d.get("status") or "OPEN").upper()

    if exch not in by_exchange:
        by_exchange[exch] = {
            "closed": [], "open": [], "trailing": [],
        }

    entry = d.get("entry_filled") or d.get("entry_expected") or 0
    size = d.get("size") or 0
    sl = d.get("sl_price") or 0
    tp = d.get("tp_price")
    side = d.get("side", "BUY")
    symbol = d.get("symbol", "?")
    raw_opened = d.get("opened_at", "")
    opened_at = raw_opened.isoformat() if hasattr(raw_opened, "isoformat") else str(raw_opened)
    signal = d.get("signal_type", "")
    size_usd = d.get("size_usd") or (size * entry if entry else 0)
    risk_usd = d.get("risk_amount_usd") or 0
    zr = d.get("is_zero_risk_applied", False)

    rec = {
        "symbol": symbol, "side": side, "entry": entry, "size": size,
        "size_usd": size_usd, "sl": sl, "tp": tp, "signal": signal,
        "opened_at": opened_at, "risk_usd": risk_usd, "zero_risk": zr,
    }

    if status == "CLOSED":
        rec["pnl_gross"] = d.get("pnl_usd") or 0
        rec["pnl_net"] = d.get("pnl_net_usd") or 0
        rec["pnl_pct"] = d.get("pnl_pct") or 0
        rec["pnl_net_pct"] = d.get("pnl_net_pct") or 0
        rec["exit_price"] = d.get("exit_price") or 0
        rec["exit_reason"] = d.get("exit_reason", "?")
        raw_closed = d.get("closed_at", "")
        rec["closed_at"] = raw_closed.isoformat() if hasattr(raw_closed, "isoformat") else str(raw_closed)
        rec["holding_h"] = d.get("holding_time_hours") or 0
        rec["fees"] = d.get("fees_total") or 0
        by_exchange[exch]["closed"].append(rec)
    elif status == "TRAILING":
        rec["trailing_steps"] = d.get("trailing_steps", 0)
        by_exchange[exch]["trailing"].append(rec)
    else:
        by_exchange[exch]["open"].append(rec)

now = datetime.now(timezone.utc)

print("=" * 70)
for exch in sorted(by_exchange.keys()):
    data = by_exchange[exch]
    label = EXCHANGE_LABELS.get(exch, exch.upper())
    closed = data["closed"]
    opens = data["open"]
    trails = data["trailing"]

    print(f"\n{'─' * 70}")
    print(f"  {label}  ({exch})")
    print(f"{'─' * 70}")

    # — Trades clôturés —
    wins = [t for t in closed if t["pnl_net"] >= 0]
    losses = [t for t in closed if t["pnl_net"] < 0]
    total_closed = len(closed)
    total_pnl = sum(t["pnl_net"] for t in closed)
    total_fees = sum(t["fees"] for t in closed)
    sum_gains = sum(t["pnl_net"] for t in wins)
    sum_losses = abs(sum(t["pnl_net"] for t in losses))
    pf = sum_gains / sum_losses if sum_losses > 0 else float("inf")
    wr = len(wins) / total_closed * 100 if total_closed > 0 else 0
    avg_w = sum_gains / len(wins) if wins else 0
    avg_l = sum_losses / len(losses) if losses else 0
    avg_hold = sum(t["holding_h"] for t in closed) / total_closed if total_closed else 0

    print(f"  Trades clôturés : {total_closed}  ({len(wins)}W / {len(losses)}L)")
    print(f"  Win Rate        : {wr:.1f}%")
    print(f"  PnL net total   : ${total_pnl:+.2f}  (fees: ${total_fees:.2f})")
    print(f"  Profit Factor   : {pf:.2f}")
    print(f"  Gain moyen      : ${avg_w:.2f}  |  Perte moyenne : ${avg_l:.2f}")
    print(f"  Durée moyenne   : {avg_hold:.1f}h")

    if closed:
        sorted_c = sorted(closed, key=lambda x: x["closed_at"], reverse=True)
        print(f"\n  5 derniers trades clôturés :")
        for t in sorted_c[:5]:
            emoji = "🟢" if t["pnl_net"] >= 0 else "🔴"
            dt = t["closed_at"][:16] if t["closed_at"] else "?"
            print(
                f"    {emoji} {t['symbol']:16s} {t['side']:4s}  "
                f"PnL=${t['pnl_net']:+.2f} ({t['pnl_net_pct']*100:+.2f}%)  "
                f"reason={t['exit_reason'][:25]}  [{dt}]"
            )

    # — Positions en trailing —
    if trails:
        print(f"\n  Positions TRAILING : {len(trails)}")
        for t in sorted(trails, key=lambda x: x["size_usd"], reverse=True)[:10]:
            print(
                f"    ↗ {t['symbol']:16s} {t['side']:4s}  "
                f"entry=${t['entry']:.4f}  size_usd=${t['size_usd']:.2f}  "
                f"steps={t['trailing_steps']}  "
                f"{'🔒ZR' if t['zero_risk'] else ''}"
            )

    # — Positions ouvertes —
    total_notional = sum(t["size_usd"] for t in opens)
    total_risk_open = sum(t["risk_usd"] for t in opens)
    zr_count = sum(1 for t in opens if t["zero_risk"])
    print(f"\n  Positions ouvertes : {len(opens)}  (trailing: {len(trails)})")
    print(f"  Notionnel total   : ${total_notional:,.2f}")
    print(f"  Risque total      : ${total_risk_open:,.2f}")
    print(f"  Zero-risk         : {zr_count}/{len(opens)}")

    if opens:
        sorted_o = sorted(opens, key=lambda x: x["size_usd"], reverse=True)
        print(f"\n  Top 10 positions ouvertes (par notionnel) :")
        for t in sorted_o[:10]:
            age_str = ""
            if t["opened_at"]:
                try:
                    opened = datetime.fromisoformat(t["opened_at"])
                    age_h = (now - opened).total_seconds() / 3600
                    if age_h < 24:
                        age_str = f"{age_h:.0f}h"
                    else:
                        age_str = f"{age_h/24:.1f}j"
                except (ValueError, TypeError):
                    pass
            zr_flag = " 🔒" if t["zero_risk"] else ""
            print(
                f"    • {t['symbol']:16s} {t['side']:4s}  "
                f"entry=${t['entry']:.4f}  SL=${t['sl']:.4f}  "
                f"${t['size_usd']:>9,.2f}  "
                f"risque=${t['risk_usd']:.2f}  "
                f"âge={age_str}{zr_flag}"
            )

print(f"\n{'=' * 70}")

# ─────────────────────────────────────────────────────────────────────────────
# 2. DAILY SNAPSHOTS
# ─────────────────────────────────────────────────────────────────────────────
print("\nDAILY SNAPSHOTS (10 derniers par exchange) :")
snaps = list(
    db.collection("daily_snapshots")
    .order_by("date", direction=firestore.Query.DESCENDING)
    .limit(30)
    .stream()
)
snap_by_exch: dict = {}
for s in snaps:
    d = s.to_dict()
    exch = d.get("exchange", "?")
    if exch not in snap_by_exch:
        snap_by_exch[exch] = []
    snap_by_exch[exch].append(d)

for exch in sorted(snap_by_exch.keys()):
    label = EXCHANGE_LABELS.get(exch, exch.upper())
    print(f"\n  {label} :")
    for d in snap_by_exch[exch][:10]:
        dt = d.get("date", "?")
        eq = d.get("equity", 0)
        pnl = d.get("daily_pnl", 0)
        nb = d.get("trades_today", 0)
        pos = d.get("positions", [])
        n_pos = len(pos) if isinstance(pos, list) else pos
        print(f"    {dt}  equity=${eq:>10,.2f}  daily_pnl=${pnl:+.2f}  trades={nb}  pos={n_pos}")

# ─────────────────────────────────────────────────────────────────────────────
# 3. ALLOCATION BINANCE
# ─────────────────────────────────────────────────────────────────────────────
print(f"\n{'=' * 70}")
print("ALLOCATION BINANCE :")
alloc = db.collection("allocation").document("current").get()
if alloc.exists:
    a = alloc.to_dict()
    total = a.get("total_balance", 0)
    print(f"  Régime    : {a.get('regime', '?')}")
    print(f"  Total     : ${total:,.2f}")
    tp = a.get("trail_pct", 0)
    cp = a.get("crash_pct", 0)
    print(f"  Trail     : {tp*100:.0f}%  →  ${a.get('trail_balance', 0):,.2f}")
    print(f"  CrashBot  : {cp*100:.0f}%  →  ${a.get('crash_balance', 0):,.2f}")
    print(f"  Trail PF  : {a.get('trail_pf', 0):.2f}  ({a.get('trail_trades', 0)} trades 90j)")
    print(f"  Raison    : {a.get('reason', '?')}")
    print(f"  Mis à jour: {a.get('updated_at', '?')[:19]}")
else:
    print("  (pas de données d'allocation)")

# ─────────────────────────────────────────────────────────────────────────────
# 4. CYCLE INFINITY
# ─────────────────────────────────────────────────────────────────────────────
print(f"\n{'=' * 70}")
print("CYCLE INFINITY (DCA inversé BTC) :")
cycle = db.collection("infinity_cycles").document("current").get()
if cycle.exists:
    c = cycle.to_dict()
    phase = c.get("phase", "?")
    ref = c.get("reference_price", 0)
    pmp = c.get("pmp", 0)
    total_size = c.get("total_size", 0)
    total_cost = c.get("total_cost", 0)
    n_buys = len(c.get("buys", []))
    n_sells = len(c.get("sells", []))
    meta = c.get("meta", {})
    print(f"  Phase           : {phase}")
    print(f"  Prix référence  : ${ref:,.2f}")
    print(f"  PMP             : ${pmp:,.2f}")
    print(f"  Position        : {total_size:.8f} BTC  (coût: ${total_cost:,.2f})")
    print(f"  Achats          : {n_buys}")
    print(f"  Ventes          : {n_sells}")
    if meta:
        print(f"  Dernière MAJ    : {meta.get('updated_at', '?')[:19]}")
else:
    print("  (pas de cycle en cours)")

# ─────────────────────────────────────────────────────────────────────────────
# 5. RÉSUMÉ GLOBAL
# ─────────────────────────────────────────────────────────────────────────────
print(f"\n{'=' * 70}")
print("RÉSUMÉ GLOBAL")
print(f"{'=' * 70}")
total_all_closed = 0
total_all_pnl = 0.0
total_all_open = 0
total_all_trailing = 0
total_all_notional = 0.0
for exch, data in by_exchange.items():
    label = EXCHANGE_LABELS.get(exch, exch)
    n_c = len(data["closed"])
    n_o = len(data["open"])
    n_t = len(data["trailing"])
    pnl = sum(t["pnl_net"] for t in data["closed"])
    notional = sum(t["size_usd"] for t in data["open"])
    total_all_closed += n_c
    total_all_pnl += pnl
    total_all_open += n_o
    total_all_trailing += n_t
    total_all_notional += notional
    wr = len([t for t in data["closed"] if t["pnl_net"] >= 0]) / n_c * 100 if n_c else 0
    print(f"  {label:30s}  closed={n_c:3d}  open={n_o:3d}  trail={n_t:3d}  PnL=${pnl:+9.2f}  WR={wr:5.1f}%  notional=${notional:>12,.2f}")

print(f"  {'─' * 68}")
print(f"  {'TOTAL':30s}  closed={total_all_closed:3d}  open={total_all_open:3d}  trail={total_all_trailing:3d}  PnL=${total_all_pnl:+9.2f}             notional=${total_all_notional:>12,.2f}")
