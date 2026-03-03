#!/usr/bin/env python3
"""
═══════════════════════════════════════════════════════════════════════════════
  TradeX — DIAGNOSTIC APPROFONDI : pourquoi le live a 3.4% WR ?
  
  Compare trade-par-trade :
    - LIVE Firebase (118 trades, 25 fév → 1 mar)
    - BACKTEST filtré à la même fenêtre 5 jours
  
  Analyses :
    1. Trades/jour (live vs BT dans la même fenêtre)
    2. Distribution SL distance (%)
    3. Distribution hold time
    4. Trades par paire (churning ?)
    5. Entry prices vs range boundaries
    6. Overlap des paires tradées
═══════════════════════════════════════════════════════════════════════════════
"""

import logging
import time as _time
import csv
import statistics
from datetime import datetime, timezone, timedelta
from pathlib import Path
from collections import Counter, defaultdict

import httpx
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dotenv import load_dotenv
load_dotenv()

from src.core.models import Candle
from backtest.simulator import BacktestEngine, BacktestConfig, Trade

logging.basicConfig(
    level=logging.WARNING,
    format="[%(asctime)s] %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger("diagnostic")

# ── Période live exacte ──────────────────────────────────────────────────────
LIVE_START = datetime(2026, 2, 25, tzinfo=timezone.utc)
LIVE_END   = datetime(2026, 3, 1, 23, 59, 59, tzinfo=timezone.utc)
LIVE_START_MS = int(LIVE_START.timestamp() * 1000)
LIVE_END_MS   = int(LIVE_END.timestamp() * 1000)

# Données : 3 mois de warmup
DATA_START = datetime(2025, 12, 1, tzinfo=timezone.utc)
DATA_END   = datetime(2026, 3, 2, tzinfo=timezone.utc)

CACHE_DIR = Path("backtest/data")

# ── Les 95 paires live ──────────────────────────────────────────────────────
LIVE_PAIRS_USDC = [
    "0GUSDC", "1000CATUSDC", "1000CHEEMSUSDC", "1000SATSUSDC", "1INCHUSDC",
    "1MBABYDOGEUSDC", "2ZUSDC", "A2ZUSDC", "ACHUSDC", "ACTUSDC",
    "ADAUSDC", "ALGOUSDC", "ALLOUSDC", "ALTUSDC", "APEUSDC",
    "APTUSDC", "ARBUSDC", "ASTERUSDC", "ATUSDC", "AVAXUSDC",
    "AXSUSDC", "BANANAS31USDC", "BANANAUSDC", "BANKUSDC", "BBUSDC",
    "BERAUSDC", "BLURUSDC", "BMTUSDC", "BNBUSDC", "BONKUSDC",
    "BROCCOLI714USDC", "BTCUSDC", "CATIUSDC", "CETUSUSDC", "CGPTUSDC",
    "COTIUSDC", "COWUSDC", "CRVUSDC", "CVCUSDC", "CVXUSDC",
    "CYBERUSDC", "DOLOUSDC", "EGLDUSDC", "ENSOUSDC", "ENSUSDC",
    "EPICUSDC", "ERAUSDC", "ESPUSDC", "ETCUSDC", "FFUSDC",
    "FOGOUSDC", "FUNUSDC", "FUSDC", "GMTUSDC", "GMXUSDC",
    "GPSUSDC", "HAEDALUSDC", "HOMEUSDC", "HUMAUSDC", "ICPUSDC",
    "IDEXUSDC", "INITUSDC", "IOTAUSDC", "JUPUSDC", "JUVUSDC",
    "KAIAUSDC", "KAITOUSDC", "KERNELUSDC", "KITEUSDC", "LAYERUSDC",
    "LDOUSDC", "LISTAUSDC", "LSKUSDC", "MANTAUSDC", "NMRUSDC",
    "ONDOUSDC", "OPENUSDC", "OSMOUSDC", "PEPEUSDC", "PUMPUSDC",
    "PYTHUSDC", "REZUSDC", "SENTUSDC", "SIGNUSDC", "SOPHUSDC",
    "STEEMUSDC", "STXUSDC", "TONUSDC", "TOWNSUSDC", "XVGUSDC",
    "YGGUSDC", "ZBTUSDC", "ZKPUSDC", "ZKUSDC", "ZROUSDC",
]


# ── Utilitaires download ────────────────────────────────────────────────────

def fetch_binance_klines(symbol: str, interval: str, start_ms: int, end_ms: int) -> list:
    url = "https://api.binance.com/api/v3/klines"
    params = {"symbol": symbol, "interval": interval,
              "startTime": start_ms, "endTime": end_ms, "limit": 1000}
    with httpx.Client(timeout=15) as client:
        r = client.get(url, params=params)
        r.raise_for_status()
        return r.json()


def download_pair(symbol_usdc: str, start: datetime, end: datetime,
                  interval: str = "4h") -> list[Candle]:
    for suffix in [symbol_usdc, symbol_usdc.replace("USDC", "USDT")]:
        cache = CACHE_DIR / f"{suffix}_{interval}_{start:%Y%m%d}_{end:%Y%m%d}.csv"
        if cache.exists():
            return _load_csv(cache)
        start_ms = int(start.timestamp() * 1000)
        end_ms = int(end.timestamp() * 1000)
        all_candles: list[Candle] = []
        cursor = start_ms
        try:
            while cursor < end_ms:
                klines = fetch_binance_klines(suffix, interval, cursor, end_ms)
                if not klines:
                    break
                for k in klines:
                    all_candles.append(Candle(
                        timestamp=int(k[0]), open=float(k[1]),
                        high=float(k[2]), low=float(k[3]),
                        close=float(k[4]), volume=float(k[5]),
                    ))
                cursor = int(klines[-1][0]) + 1
                _time.sleep(0.12)
            if all_candles:
                CACHE_DIR.mkdir(parents=True, exist_ok=True)
                _save_csv(all_candles, cache)
                return all_candles
        except httpx.HTTPStatusError:
            continue
    return []


def _load_csv(path: Path) -> list[Candle]:
    candles = []
    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            candles.append(Candle(
                timestamp=int(row["timestamp"]), open=float(row["open"]),
                high=float(row["high"]), low=float(row["low"]),
                close=float(row["close"]), volume=float(row["volume"]),
            ))
    return candles


def _save_csv(candles: list[Candle], path: Path) -> None:
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["timestamp", "open", "high", "low", "close", "volume"])
        w.writeheader()
        for c in candles:
            w.writerow({"timestamp": c.timestamp, "open": c.open, "high": c.high,
                         "low": c.low, "close": c.close, "volume": c.volume})


# ── Firebase query ───────────────────────────────────────────────────────────

def get_live_trades() -> list[dict]:
    """Récupère les trades CLOSED Binance depuis Firebase."""
    try:
        from src.firebase.client import get_documents
        trades = get_documents(
            "trades",
            filters=[
                ("exchange", "==", "binance"),
                ("status", "==", "CLOSED"),
            ],
        )
        print(f"  📦 Firebase : {len(trades)} trades CLOSED récupérés")

        # Inspecter les champs du premier trade
        if trades:
            sample = trades[0]
            print(f"  🔎 Champs disponibles : {sorted(sample.keys())}")
            # Afficher les champs timestamp pour debug
            for key in sorted(sample.keys()):
                val = sample[key]
                if 'time' in key.lower() or 'date' in key.lower() or 'ts' in key.lower():
                    print(f"     {key} = {val!r} (type: {type(val).__name__})")
            # Afficher les champs prix/sl
            for key in ['entry_price', 'exit_price', 'sl_price', 'tp_price', 'pnl_usd', 'pnl_net_usd', 'symbol', 'side', 'exit_reason']:
                if key in sample:
                    print(f"     {key} = {sample[key]!r}")
        return trades
    except Exception as e:
        print(f"  ⚠️ Firebase indisponible : {e}")
        import traceback
        traceback.print_exc()
        return []


# ── Helpers d'analyse ────────────────────────────────────────────────────────

def _ts_to_dt(ts_ms: int) -> datetime:
    return datetime.fromtimestamp(ts_ms / 1000, tz=timezone.utc)

def _ts_to_str(ts_ms: int) -> str:
    return _ts_to_dt(ts_ms).strftime("%d/%m %H:%M")

def _pct(a, b):
    """Retourne |a - b| / b en %."""
    if b == 0:
        return 0
    return abs(a - b) / b

def _distribution_summary(values: list[float], label: str, unit: str = "%") -> str:
    if not values:
        return f"  {label}: aucune donnée"
    mn = min(values)
    mx = max(values)
    med = statistics.median(values)
    avg = statistics.mean(values)
    return f"  {label}: min={mn:.3f}{unit} | med={med:.3f}{unit} | avg={avg:.3f}{unit} | max={mx:.3f}{unit}"


def _parse_ts(ts) -> int:
    """Parse un timestamp Firebase → ms."""
    if ts is None:
        return 0
    # google.cloud.firestore DatetimeWithNanoseconds / datetime
    if hasattr(ts, 'timestamp'):
        return int(ts.timestamp() * 1000)
    # google.protobuf.timestamp_pb2.Timestamp
    if hasattr(ts, 'seconds'):
        return int(ts.seconds * 1000 + ts.nanos // 1_000_000)
    if isinstance(ts, (int, float)):
        if ts < 1e12:  # seconds
            return int(ts * 1000)
        return int(ts)  # already ms
    if isinstance(ts, str):
        try:
            dt = datetime.fromisoformat(ts.replace('Z', '+00:00'))
            return int(dt.timestamp() * 1000)
        except:
            return 0
    return 0


def _normalize_symbol(sym: str) -> str:
    """Normalise un symbole Binance (ex: BONKUSDC → BONK-USD)."""
    for suffix in ("USDC", "USDT"):
        if sym.endswith(suffix):
            return sym[:-len(suffix)] + "-USD"
    return sym


# ══════════════════════════════════════════════════════════════════════════════

def main():
    sep = "═" * 90

    print(f"\n{sep}")
    print(f"  🔬 DIAGNOSTIC APPROFONDI — Pourquoi le live a 3.4% WR ?")
    print(f"  📅 Fenêtre live : {LIVE_START:%Y-%m-%d} → {LIVE_END:%Y-%m-%d}")
    print(f"  🎯 Objectif : identifier les différences structurelles BT vs Live")
    print(f"{sep}")

    # ═══════════════════════════════════════════════════════════════════════
    # 1. DONNÉES FIREBASE (LIVE)
    # ═══════════════════════════════════════════════════════════════════════
    print(f"\n{'─' * 60}")
    print(f"  📥 1. Récupération des trades live (Firebase)")
    print(f"{'─' * 60}")

    live_trades = get_live_trades()

    # Extraire les champs utiles
    live_data = []
    for t in live_trades:
        entry_price = t.get("entry_filled") or t.get("entry_expected") or t.get("entry_price", 0)
        exit_price = t.get("exit_price", 0)
        sl_price = t.get("sl_price", 0)
        tp_price = t.get("tp_price", 0)
        pnl = t.get("pnl_usd", 0)
        symbol = t.get("symbol", "?")
        side = t.get("side", "BUY")
        exit_reason = t.get("exit_reason", "?")
        hold_hours_fb = t.get("holding_time_hours", 0)

        # Chercher les timestamps (noms possibles)
        entry_ts = (t.get("entry_timestamp") or t.get("entry_time")
                    or t.get("opened_at") or t.get("created_at"))
        exit_ts = (t.get("exit_timestamp") or t.get("exit_time")
                   or t.get("closed_at") or t.get("updated_at"))

        entry_ms = _parse_ts(entry_ts)
        exit_ms = _parse_ts(exit_ts)

        # Normaliser le symbole : "BONKUSDC" → "BONK-USD"
        norm_symbol = _normalize_symbol(symbol)

        # SL distance
        e_px = float(entry_price) if entry_price else 0
        s_px = float(sl_price) if sl_price else 0
        sl_dist = abs(e_px - s_px) / e_px if e_px > 0 and s_px > 0 else 0
        # Hold time : use Firebase field if available, else compute 
        hold_h = float(hold_hours_fb) if hold_hours_fb else (
            (exit_ms - entry_ms) / (3600 * 1000) if entry_ms > 0 and exit_ms > 0 else 0
        )

        live_data.append({
            "symbol": symbol,
            "norm_symbol": norm_symbol,
            "side": side,
            "entry_price": e_px,
            "exit_price": float(exit_price) if exit_price else 0,
            "sl_price": s_px,
            "tp_price": float(tp_price) if tp_price else 0,
            "pnl": float(pnl) if pnl else 0,
            "exit_reason": exit_reason,
            "entry_ms": entry_ms,
            "exit_ms": exit_ms,
            "sl_dist_pct": sl_dist,
            "hold_hours": hold_h,
            "is_win": float(pnl) > 0 if pnl else False,
        })

    if live_data:
        print(f"  ✅ {len(live_data)} trades live parsés")
    else:
        print(f"  ❌ Pas de données live — on continue avec le BT seul")

    # ═══════════════════════════════════════════════════════════════════════
    # 2. BACKTEST
    # ═══════════════════════════════════════════════════════════════════════
    print(f"\n{'─' * 60}")
    print(f"  📥 2. Backtest des 95 paires (données en cache)")
    print(f"{'─' * 60}")

    candles: dict[str, list[Candle]] = {}
    for i, pair in enumerate(LIVE_PAIRS_USDC):
        key = pair.replace("USDC", "-USD")
        cs = download_pair(pair, DATA_START, DATA_END)
        if cs and len(cs) > 50:
            candles[key] = cs
        if (i + 1) % 20 == 0:
            print(f"     {i+1}/{len(LIVE_PAIRS_USDC)}…")

    print(f"  ✅ {len(candles)} paires chargées")

    cfg = BacktestConfig(
        initial_balance=940.0,
        risk_percent_range=0.02,
        entry_buffer_pct=0.002,
        sl_buffer_pct=0.003,
        zero_risk_trigger_pct=0.02,
        zero_risk_lock_pct=0.005,
        trailing_stop_pct=0.02,
        max_position_pct=0.30,
        max_simultaneous_positions=3,
        swing_lookback=3,
        range_width_min=0.018,
        range_entry_buffer_pct=0.002,
        range_sl_buffer_pct=0.003,
        range_cooldown_bars=3,
        range_min_sl_pct=0.0,
        range_atr_sl_mult=0.0,
        fee_pct=0.00075,
        slippage_pct=0.001,
        enable_trend=False,
        enable_range=True,
        allow_short=False,
        range_sl_on_close=False,  # intrabar = simule OCO
    )

    engine = BacktestEngine(candles, cfg)
    result = engine.run()

    # Filtrer les trades BT à la fenêtre live (25 fév → 1 mar)
    bt_all = result.trades
    bt_live_window = [t for t in bt_all if LIVE_START_MS <= t.entry_time <= LIVE_END_MS]
    bt_full = bt_all  # tous les trades sur 3 mois

    print(f"  📊 BT total : {len(bt_full)} trades (3 mois)")
    print(f"  📊 BT fenêtre live : {len(bt_live_window)} trades ({LIVE_START:%d/%m} → {LIVE_END:%d/%m})")

    # ═══════════════════════════════════════════════════════════════════════
    # 3. COMPARAISONS
    # ═══════════════════════════════════════════════════════════════════════

    print(f"\n{sep}")
    print(f"  📊 COMPARAISON STRUCTURELLE")
    print(f"{sep}")

    # ── 3a. Vue d'ensemble ──
    print(f"\n  ── 3a. VUE D'ENSEMBLE (fenêtre {LIVE_START:%d/%m} → {LIVE_END:%d/%m}) ──")
    print(f"  {'':30s} │ {'LIVE':>15s} │ {'BT (fenêtre)':>15s} │ {'BT (total)':>15s}")
    print(f"  {'─' * 30}─┼─{'─' * 15}─┼─{'─' * 15}─┼─{'─' * 15}")

    bt_w_wins = sum(1 for t in bt_live_window if t.pnl_usd > 0)
    bt_w_wr = bt_w_wins / len(bt_live_window) if bt_live_window else 0
    bt_w_pnl = sum(t.pnl_usd for t in bt_live_window)

    bt_f_wins = sum(1 for t in bt_full if t.pnl_usd > 0)
    bt_f_wr = bt_f_wins / len(bt_full) if bt_full else 0
    bt_f_pnl = sum(t.pnl_usd for t in bt_full)

    live_wins = sum(1 for t in live_data if t["is_win"])
    live_wr = live_wins / len(live_data) if live_data else 0
    live_pnl = sum(t["pnl"] for t in live_data)

    overview_rows = [
        ("Trades",      str(len(live_data)),           str(len(bt_live_window)),     str(len(bt_full))),
        ("Wins",        str(live_wins),                str(bt_w_wins),               str(bt_f_wins)),
        ("Win Rate",    f"{live_wr:.1%}",              f"{bt_w_wr:.1%}",             f"{bt_f_wr:.1%}"),
        ("PnL",         f"${live_pnl:+.2f}",           f"${bt_w_pnl:+.2f}",          f"${bt_f_pnl:+.2f}"),
        ("Trades/jour", f"{len(live_data)/5:.1f}",     f"{len(bt_live_window)/5:.1f}", f"{len(bt_full)/90:.1f}"),
    ]
    for label, v_l, v_bw, v_bf in overview_rows:
        print(f"  {label:30s} │ {v_l:>15s} │ {v_bw:>15s} │ {v_bf:>15s}")

    # ── 3b. Distribution SL distance ──
    print(f"\n  ── 3b. DISTRIBUTION SL DISTANCE ──")
    
    live_sl_dists = [t["sl_dist_pct"] * 100 for t in live_data if t["sl_dist_pct"] > 0]
    bt_w_sl_dists = [_pct(t.entry_price, t.entry_price - abs(t.entry_price - (t.entry_price - (t.entry_price - t.exit_price)))) * 100
                     for t in bt_live_window if t.exit_reason == "RANGE_SL"]
    
    # Mieux : calculer SL distance à partir du signal BT
    # Pour le BT, on n'a pas directement le SL price dans le Trade, mais on peut le déduire
    # Si exit_reason = RANGE_SL, exit_price = sl_price (en mode intrabar)
    bt_sl_dists_from_exit = []
    for t in bt_live_window:
        if t.exit_reason == "RANGE_SL":
            sl_dist = abs(t.entry_price - t.exit_price) / t.entry_price * 100
            bt_sl_dists_from_exit.append(sl_dist)

    bt_all_sl_dists = []
    for t in bt_live_window:
        # BT: SL = range_low * (1 - buffer) pour un BUY. 
        # entry = range_low * (1 + buffer).
        # Donc SL distance ≈ 2 * buffer ≈ 0.4%
        # On peut le calculer via entry vs exit pour les SL
        if t.exit_reason == "RANGE_SL":
            bt_all_sl_dists.append(abs(t.entry_price - t.exit_price) / t.entry_price * 100)

    print(f"\n  LIVE SL distances :")
    print(_distribution_summary(live_sl_dists, "SL distance"))
    if live_sl_dists:
        buckets = Counter()
        for d in live_sl_dists:
            if d < 0.2:
                buckets["< 0.2%"] += 1
            elif d < 0.4:
                buckets["0.2-0.4%"] += 1
            elif d < 0.6:
                buckets["0.4-0.6%"] += 1
            elif d < 1.0:
                buckets["0.6-1.0%"] += 1
            else:
                buckets["> 1.0%"] += 1
        print(f"  Distribution : {dict(sorted(buckets.items()))}")

    print(f"\n  BT SL distances (trades SL only, dans la fenêtre live) :")
    print(_distribution_summary(bt_all_sl_dists, "SL distance"))
    if bt_all_sl_dists:
        buckets = Counter()
        for d in bt_all_sl_dists:
            if d < 0.2:
                buckets["< 0.2%"] += 1
            elif d < 0.4:
                buckets["0.2-0.4%"] += 1
            elif d < 0.6:
                buckets["0.4-0.6%"] += 1
            elif d < 1.0:
                buckets["0.6-1.0%"] += 1
            else:
                buckets["> 1.0%"] += 1
        print(f"  Distribution : {dict(sorted(buckets.items()))}")

    # ── 3c. Distribution hold time ──
    print(f"\n  ── 3c. DISTRIBUTION HOLD TIME ──")

    live_holds = [t["hold_hours"] for t in live_data if t["hold_hours"] > 0]
    bt_holds = [(t.exit_time - t.entry_time) / 3600000 for t in bt_live_window if t.exit_time > t.entry_time]

    print(f"\n  LIVE :")
    print(_distribution_summary(live_holds, "Hold time", "h"))
    if live_holds:
        short = sum(1 for h in live_holds if h < 1)
        medium = sum(1 for h in live_holds if 1 <= h < 4)
        long_ = sum(1 for h in live_holds if h >= 4)
        print(f"  Distribution : <1h={short} | 1-4h={medium} | ≥4h={long_}")

    print(f"\n  BT (fenêtre live) :")
    print(_distribution_summary(bt_holds, "Hold time", "h"))
    if bt_holds:
        short = sum(1 for h in bt_holds if h < 1)
        medium = sum(1 for h in bt_holds if 1 <= h <= 8)
        long_ = sum(1 for h in bt_holds if h > 8)
        print(f"  Distribution : <1h={short} | 1-8h={medium} | >8h={long_}")

    # ── 3d. Trades par paire (churning ?) ──
    print(f"\n  ── 3d. TRADES PAR PAIRE (churning ?) ──")

    live_per_pair = Counter(t["norm_symbol"] for t in live_data)
    bt_per_pair = Counter(t.symbol for t in bt_live_window)

    live_top = live_per_pair.most_common(10)
    bt_top = bt_per_pair.most_common(10)

    print(f"\n  LIVE — Top 10 paires par nb de trades :")
    for pair, count in live_top:
        wins = sum(1 for t in live_data if t["norm_symbol"] == pair and t["is_win"])
        print(f"     {pair:20s} : {count:3d} trades ({wins}W / {count-wins}L)")

    print(f"\n  BT — Top 10 paires par nb de trades (fenêtre live) :")
    for pair, count in bt_top:
        wins = sum(1 for t in bt_live_window if t.symbol == pair and t.pnl_usd > 0)
        print(f"     {pair:20s} : {count:3d} trades ({wins}W / {count-wins}L)")

    # Paires uniques
    live_pairs_set = set(t["norm_symbol"] for t in live_data)
    bt_pairs_set = set(t.symbol for t in bt_live_window)
    overlap = live_pairs_set & bt_pairs_set
    live_only = live_pairs_set - bt_pairs_set
    bt_only = bt_pairs_set - live_pairs_set

    print(f"\n  Paires uniques LIVE : {len(live_pairs_set)}")
    print(f"  Paires uniques BT   : {len(bt_pairs_set)}")
    print(f"  Overlap              : {len(overlap)}")
    print(f"  Live only            : {len(live_only)} → {sorted(live_only)[:15]}")
    print(f"  BT only              : {len(bt_only)} → {sorted(bt_only)[:15]}")

    # ── 3e. Exit reasons ──
    print(f"\n  ── 3e. EXIT REASONS ──")
    
    live_reasons = Counter(t["exit_reason"] for t in live_data)
    bt_reasons = Counter(t.exit_reason for t in bt_live_window)

    print(f"\n  LIVE :")
    for reason, count in live_reasons.most_common():
        pnl_r = sum(t["pnl"] for t in live_data if t["exit_reason"] == reason)
        print(f"     {reason:40s} : {count:3d} ({pnl_r:+.2f} USD)")

    print(f"\n  BT (fenêtre live) :")
    for reason, count in bt_reasons.most_common():
        pnl_r = sum(t.pnl_usd for t in bt_live_window if t.exit_reason == reason)
        print(f"     {reason:40s} : {count:3d} ({pnl_r:+.2f} USD)")

    # ── 3f. Trades par jour ──
    print(f"\n  ── 3f. TRADES PAR JOUR ──")

    live_per_day = Counter()
    for t in live_data:
        if t["entry_ms"]:
            day = _ts_to_dt(t["entry_ms"]).strftime("%Y-%m-%d")
            live_per_day[day] += 1

    bt_per_day = Counter()
    for t in bt_live_window:
        day = _ts_to_dt(t.entry_time).strftime("%Y-%m-%d")
        bt_per_day[day] += 1

    all_days = sorted(set(live_per_day.keys()) | set(bt_per_day.keys()))
    print(f"\n  {'Date':15s} │ {'LIVE':>8s} │ {'BT':>8s}")
    print(f"  {'─' * 15}─┼─{'─' * 8}─┼─{'─' * 8}")
    for day in all_days:
        print(f"  {day:15s} │ {live_per_day.get(day, 0):>8d} │ {bt_per_day.get(day, 0):>8d}")

    # ── 3g. Analyse R:R ──
    print(f"\n  ── 3g. ANALYSE RISK/REWARD ──")

    # LIVE
    live_rr = []
    for t in live_data:
        entry = t["entry_price"]
        sl = t["sl_price"]
        tp = t["tp_price"]
        if entry and sl and tp and entry != sl:
            risk = abs(entry - sl)
            reward = abs(tp - entry)
            rr = reward / risk if risk > 0 else 0
            live_rr.append(rr)

    print(f"\n  LIVE R:R (reward/risk) :")
    print(_distribution_summary(live_rr, "R:R", ""))

    # ── 3g-bis. INTRA-CANDLE vs MULTI-CANDLE ──
    print(f"\n  ── 3g-bis. INTRA-CANDLE ANALYSIS ──")
    
    intra_candle = [t for t in live_data if 0 < t["hold_hours"] < 4]
    multi_candle = [t for t in live_data if t["hold_hours"] >= 4]
    
    ic_wins = sum(1 for t in intra_candle if t["is_win"])
    mc_wins = sum(1 for t in multi_candle if t["is_win"])
    ic_pnl = sum(t["pnl"] for t in intra_candle)
    mc_pnl = sum(t["pnl"] for t in multi_candle)
    
    print(f"\n  {'':25s} │ {'Intra-candle':>15s} │ {'Multi-candle':>15s}")
    print(f"  {'':25s} │ {'(hold < 4h)':>15s} │ {'(hold ≥ 4h)':>15s}")
    print(f"  {'─' * 25}─┼─{'─' * 15}─┼─{'─' * 15}")
    print(f"  {'Trades':25s} │ {len(intra_candle):>15d} │ {len(multi_candle):>15d}")
    print(f"  {'Wins':25s} │ {ic_wins:>15d} │ {mc_wins:>15d}")
    ic_wr = ic_wins / len(intra_candle) if intra_candle else 0
    mc_wr = mc_wins / len(multi_candle) if multi_candle else 0
    print(f"  {'Win Rate':25s} │ {ic_wr:>14.1%} │ {mc_wr:>14.1%}")
    print(f"  {'PnL':25s} │ ${ic_pnl:>+13.2f} │ ${mc_pnl:>+13.2f}")
    
    ic_sl = [t for t in intra_candle if "SL" in t["exit_reason"]]
    mc_sl = [t for t in multi_candle if "SL" in t["exit_reason"]]
    print(f"  {'SL exits':25s} │ {len(ic_sl):>15d} │ {len(mc_sl):>15d}")
    
    # Perte moyenne par SL
    ic_sl_losses = [t["pnl"] for t in ic_sl if t["pnl"] < 0]
    mc_sl_losses = [t["pnl"] for t in mc_sl if t["pnl"] < 0]
    ic_avg_loss = statistics.mean(ic_sl_losses) if ic_sl_losses else 0
    mc_avg_loss = statistics.mean(mc_sl_losses) if mc_sl_losses else 0
    print(f"  {'Perte moy. par SL':25s} │ ${ic_avg_loss:>+13.2f} │ ${mc_avg_loss:>+13.2f}")

    # SL distance distribution pour ceux qui ont des données
    ic_sl_dists = [t["sl_dist_pct"] * 100 for t in intra_candle if t["sl_dist_pct"] > 0]
    mc_sl_dists = [t["sl_dist_pct"] * 100 for t in multi_candle if t["sl_dist_pct"] > 0]
    
    if ic_sl_dists:
        print(f"\n  SL distance intra-candle :")
        print(_distribution_summary(ic_sl_dists, "SL distance"))
    if mc_sl_dists:
        print(f"  SL distance multi-candle :")
        print(_distribution_summary(mc_sl_dists, "SL distance"))

    # ── 3g-ter. Nombre de trades par paire qui ne sont PAS dans le BT ──
    print(f"\n  ── 3g-ter. PAIRES LIVE ABSENTES DU BT ──")
    
    # Trades live sur les 75 paires que le BT ne trade pas
    live_only_trades = [t for t in live_data if t["norm_symbol"] in live_only]
    lo_wins = sum(1 for t in live_only_trades if t["is_win"])
    lo_pnl = sum(t["pnl"] for t in live_only_trades)
    lo_wr = lo_wins / len(live_only_trades) if live_only_trades else 0
    
    # Trades live sur les 20 paires communes avec le BT
    overlap_live_trades = [t for t in live_data if t["norm_symbol"] in overlap]
    ol_wins = sum(1 for t in overlap_live_trades if t["is_win"])
    ol_pnl = sum(t["pnl"] for t in overlap_live_trades)
    ol_wr = ol_wins / len(overlap_live_trades) if overlap_live_trades else 0

    print(f"\n  {'':30s} │ {'Paires communes':>18s} │ {'Paires live-only':>18s}")
    print(f"  {'':30s} │ {'(BT détecte range)':>18s} │ {'(BT pas de range)':>18s}")
    print(f"  {'─' * 30}─┼─{'─' * 18}─┼─{'─' * 18}")
    print(f"  {'Paires':30s} │ {len(overlap):>18d} │ {len(live_only):>18d}")
    print(f"  {'Trades live':30s} │ {len(overlap_live_trades):>18d} │ {len(live_only_trades):>18d}")
    print(f"  {'Wins':30s} │ {ol_wins:>18d} │ {lo_wins:>18d}")
    print(f"  {'Win Rate':30s} │ {ol_wr:>17.1%} │ {lo_wr:>17.1%}")
    print(f"  {'PnL':30s} │ ${ol_pnl:>+16.2f} │ ${lo_pnl:>+16.2f}")

    # ── 3h. Détail des trades sur une paire commune ──
    if live_data and bt_live_window:
        # Trouver une paire avec beaucoup de trades dans les deux
        common_pairs_live = {p: c for p, c in live_per_pair.items() if p in bt_pairs_set}
        if common_pairs_live:
            # Paire avec le plus de trades live
            focus_pair = max(common_pairs_live, key=common_pairs_live.get)
            
            print(f"\n  ── 3h. ZOOM SUR {focus_pair} ──")
            
            focus_live = sorted([t for t in live_data if t["norm_symbol"] == focus_pair],
                               key=lambda t: t["entry_ms"])
            focus_bt = sorted([t for t in bt_live_window if t.symbol == focus_pair],
                             key=lambda t: t.entry_time)

            print(f"\n  LIVE ({len(focus_live)} trades) :")
            for t in focus_live[:12]:
                dt_str = _ts_to_str(t["entry_ms"]) if t["entry_ms"] else "?"
                print(f"     {dt_str} | {t['side']:4s} | Entry={t['entry_price']:.6f} | "
                      f"SL={t['sl_price']:.6f} | SL%={t['sl_dist_pct']*100:.2f}% | "
                      f"PnL={t['pnl']:+.3f} | Hold={t['hold_hours']:.1f}h | {t['exit_reason'][:30]}")

            print(f"\n  BT ({len(focus_bt)} trades) :")
            for t in focus_bt[:12]:
                sl_dist = abs(t.entry_price - t.exit_price) / t.entry_price * 100 if t.exit_reason == "RANGE_SL" else 0
                hold = (t.exit_time - t.entry_time) / 3600000
                dt_str = _ts_to_str(t.entry_time)
                print(f"     {dt_str} | {t.side.value:4s} | Entry={t.entry_price:.6f} | "
                      f"Exit={t.exit_price:.6f} | SL%={sl_dist:.2f}% | "
                      f"PnL={t.pnl_usd:+.3f} | Hold={hold:.1f}h | {t.exit_reason}")

    # ═══════════════════════════════════════════════════════════════════════
    # 4. VERDICT
    # ═══════════════════════════════════════════════════════════════════════

    print(f"\n{sep}")
    print(f"  🎯 SYNTHÈSE DIAGNOSTIC")
    print(f"{sep}")

    # Calculs pour le verdict
    live_avg_hold = statistics.mean(live_holds) if live_holds else 0
    bt_avg_hold = statistics.mean(bt_holds) if bt_holds else 0
    live_median_sl = statistics.median(live_sl_dists) if live_sl_dists else 0
    bt_median_sl = statistics.median(bt_all_sl_dists) if bt_all_sl_dists else 0
    
    live_sl_losses = sum(1 for t in live_data if "SL" in t["exit_reason"])
    live_sl_pct = live_sl_losses / len(live_data) * 100 if live_data else 0
    bt_sl_losses = sum(1 for t in bt_live_window if t.exit_reason == "RANGE_SL")
    bt_sl_pct = bt_sl_losses / len(bt_live_window) * 100 if bt_live_window else 0

    findings = []

    if live_data and len(live_data) / 5 > len(bt_live_window) / 5 * 2:
        findings.append(f"🔥 CHURNING : Live fait {len(live_data)/5:.0f} trades/jour vs BT {len(bt_live_window)/5:.0f}")

    if live_avg_hold and bt_avg_hold and live_avg_hold < bt_avg_hold * 0.5:
        findings.append(f"⏱️ HOLD TIME : Live {live_avg_hold:.1f}h vs BT {bt_avg_hold:.1f}h (live trop court)")

    if live_median_sl and bt_median_sl and live_median_sl < bt_median_sl * 0.8:
        findings.append(f"📏 SL DISTANCE : Live {live_median_sl:.2f}% vs BT {bt_median_sl:.2f}% (live SL trop serré)")

    if live_sl_pct > 90:
        findings.append(f"🛑 SL RATE : {live_sl_pct:.0f}% des trades live finissent en SL")

    if len(live_pairs_set) > len(bt_pairs_set) * 1.5:
        findings.append(f"🌐 PAIRES : Live trade {len(live_pairs_set)} paires vs BT {len(bt_pairs_set)} (BT ne détecte pas de range sur certaines)")

    if not findings:
        findings.append("Aucune anomalie flagrante détectée — les différences sont probablement subtiles")

    for f in findings:
        print(f"\n  {f}")

    # Recommandations
    print(f"\n\n  📋 DIFFÉRENCES STRUCTURELLES LIVE vs BT :")
    print(f"  ┌──────────────────────┬────────────────────────────────────────────┐")
    print(f"  │ Aspect               │ Différence                                 │")
    print(f"  ├──────────────────────┼────────────────────────────────────────────┤")
    print(f"  │ Entrée               │ LIVE=tick mid-candle, BT=open next candle │")
    print(f"  │ SL exécution         │ LIVE=OCO tick-by-tick, BT=c.low par bougie│")
    print(f"  │ # checks par bougie  │ LIVE=~1440 ticks/4h, BT=1 check/bougie   │")
    print(f"  │ Cooldown             │ LIVE=time-based, BT=timestamp-based       │")
    print(f"  │ Re-entry même bougie │ LIVE=OUI (ticker), BT=NON (open uniquem.) │")
    print(f"  │ Range detection      │ LIVE=continu, BT=au close de la bougie    │")
    print(f"  └──────────────────────┴────────────────────────────────────────────┘")

    print(f"\n  💡 PISTE PRIORITAIRE :")
    print(f"     Le BT n'entre qu'à l'OPEN de la bougie suivante (1 tentative)")
    print(f"     Le live entre dès que le ticker touche la buy zone (~1440 ticks/bougie)")
    print(f"     → Le live peut entrer à des prix intra-candle qui ne correspondent")
    print(f"       pas à une vraie structure (faux breakout intra-bougie)")
    print(f"     → Le live peut aussi re-entrer après un SL dans la MÊME bougie")

    print(f"\n{sep}\n")


if __name__ == "__main__":
    main()
