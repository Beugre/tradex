#!/usr/bin/env python3
"""
═══════════════════════════════════════════════════════════════════════════════
  TradeX — SL INTRABAR vs SL CLOSE : quantification exacte
  
  Mêmes 95 paires, même période, même config.
  A) SL intrabar (c.low ≤ SL) → simule OCO tick-by-tick
  B) SL close (c.close ≤ SL) → simule check SL au close H4
  + LIVE réel (Firebase) pour comparaison
═══════════════════════════════════════════════════════════════════════════════
"""

import logging
import time as _time
from datetime import datetime, timezone, timedelta
from pathlib import Path

import httpx
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dotenv import load_dotenv
load_dotenv()

from src.core.models import Candle
from backtest.simulator import BacktestEngine, BacktestConfig
from backtest.metrics import compute_metrics

logging.basicConfig(
    level=logging.WARNING,
    format="[%(asctime)s] %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger("replay")

# ── Les 95 paires exactes tradées en live (format Binance USDC) ──────────────

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

# On a besoin de plus de données avant le 25 fév pour que les swings soient
# calculés. On télécharge 3 mois (déc 2025 → 1 mar 2026).
DATA_START = datetime(2025, 12, 1, tzinfo=timezone.utc)
DATA_END   = datetime(2026, 3, 2, tzinfo=timezone.utc)

CACHE_DIR = Path("backtest/data")


# ── Téléchargement direct Binance (pas besoin de SYMBOL_MAP) ─────────────────

def fetch_binance_klines(symbol: str, interval: str, start_ms: int, end_ms: int) -> list:
    """Fetch klines depuis Binance API publique."""
    url = "https://api.binance.com/api/v3/klines"
    params = {
        "symbol": symbol,
        "interval": interval,
        "startTime": start_ms,
        "endTime": end_ms,
        "limit": 1000,
    }
    with httpx.Client(timeout=15) as client:
        r = client.get(url, params=params)
        r.raise_for_status()
        return r.json()


def download_pair(symbol_usdc: str, start: datetime, end: datetime,
                  interval: str = "4h") -> list[Candle]:
    """Télécharge les bougies H4 pour une paire USDC Binance."""
    # Essayer USDC d'abord, puis USDT si pas dispo
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
                        timestamp=int(k[0]),
                        open=float(k[1]),
                        high=float(k[2]),
                        low=float(k[3]),
                        close=float(k[4]),
                        volume=float(k[5]),
                    ))
                cursor = int(klines[-1][0]) + 1
                _time.sleep(0.12)

            if all_candles:
                CACHE_DIR.mkdir(parents=True, exist_ok=True)
                _save_csv(all_candles, cache)
                return all_candles
        except httpx.HTTPStatusError:
            continue  # Essayer le suffixe suivant

    return []


def _load_csv(path: Path) -> list[Candle]:
    candles = []
    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            candles.append(Candle(
                timestamp=int(row["timestamp"]),
                open=float(row["open"]),
                high=float(row["high"]),
                low=float(row["low"]),
                close=float(row["close"]),
                volume=float(row["volume"]),
            ))
    return candles


def _save_csv(candles: list[Candle], path: Path) -> None:
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["timestamp", "open", "high", "low", "close", "volume"])
        w.writeheader()
        for c in candles:
            w.writerow({
                "timestamp": c.timestamp,
                "open": c.open,
                "high": c.high,
                "low": c.low,
                "close": c.close,
                "volume": c.volume,
            })


import csv


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    sep = "═" * 90
    print(f"\n{sep}")
    print(f"  🔬 SL INTRABAR vs SL CLOSE — Quantification exacte")
    print(f"  📅 Données : {DATA_START:%Y-%m-%d} → {DATA_END:%Y-%m-%d}")
    print(f"  🪙 {len(LIVE_PAIRS_USDC)} paires (identiques au live Firebase)")
    print(f"  💰 Capital : $940 | Risque : 2%")
    print(f"  A) SL intrabar : c.low ≤ SL (simule OCO tick-by-tick)")
    print(f"  B) SL close    : c.close ≤ SL (simule check au close H4)")
    print(f"{sep}")

    # 1) Télécharger les données
    print(f"\n  📥 Téléchargement des {len(LIVE_PAIRS_USDC)} paires…")
    candles: dict[str, list[Candle]] = {}
    failed = []

    for i, pair in enumerate(LIVE_PAIRS_USDC):
        key = pair.replace("USDC", "-USD")
        cs = download_pair(pair, DATA_START, DATA_END)
        if cs and len(cs) > 50:
            candles[key] = cs
        else:
            failed.append(pair)
        if (i + 1) % 20 == 0:
            print(f"     {i+1}/{len(LIVE_PAIRS_USDC)}…")

    print(f"  ✅ {len(candles)} paires chargées | {len(failed)} échouées")
    if failed:
        print(f"  ⚠️ Échouées : {', '.join(failed[:15])}")

    # 2) Config de base (identique au live)
    base_cfg = dict(
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
    )

    # 3) Deux backtests
    results = []
    variants = [
        ("🔴 A) SL intrabar (c.low)",  False),
        ("🟢 B) SL close (c.close)",   True),
    ]

    for label, sl_on_close in variants:
        print(f"\n{'─' * 60}")
        print(f"  {label}")
        print(f"{'─' * 60}")

        cfg = BacktestConfig(**{**base_cfg, "range_sl_on_close": sl_on_close})
        engine = BacktestEngine(candles, cfg)
        result = engine.run()
        metrics = compute_metrics(result)
        results.append((label, result, metrics))

    # 4) Tableau comparatif
    print(f"\n{sep}")
    print(f"  📊 RÉSULTATS : SL INTRABAR vs SL CLOSE vs LIVE")
    print(sep)

    # Header
    header = f"  {'Métrique':25s}"
    header += f" │ {'🔴 LIVE (réel)':>22s}"
    for label, _, _ in results:
        header += f" │ {label:>30s}"
    print(f"\n{header}")
    print(f"  {'─' * 25}─┼─{'─' * 22}─┼─{'─' * 30}─┼─{'─' * 30}")

    # Données live
    live = {"trades": 118, "wins": 4, "wr": 0.034, "pnl": -104.31,
            "ret": -0.111, "final": 835.69, "fees": 36.85}

    rows = [
        ("Trades",       lambda m: str(m.get("n_trades", 0)),                         str(live["trades"])),
        ("Wins",         lambda m: str(int(m.get("n_trades",0) * m.get("win_rate",0))), str(live["wins"])),
        ("Win Rate",     lambda m: f"{m.get('win_rate', 0):.1%}",                      f"{live['wr']:.1%}"),
        ("Return",       lambda m: f"{m.get('total_return', 0):+.1%}",                 f"{live['ret']:+.1%}"),
        ("PnL net ($)",  lambda m: f"${m.get('total_return', 0) * 940:+.2f}",          f"${live['pnl']:+.2f}"),
        ("Capital final",lambda m: f"${m.get('final_equity', 940):,.2f}",              f"${live['final']:,.2f}"),
        ("Max Drawdown", lambda m: f"{m.get('max_drawdown', 0):.1%}",                  "N/A"),
        ("Sharpe",       lambda m: f"{m.get('sharpe', 0):.2f}",                        "N/A"),
        ("Profit Factor",lambda m: f"{m.get('profit_factor', 0):.2f}",                 "N/A"),
    ]

    for row_label, fmt_fn, live_val in rows:
        line = f"  {row_label:25s} │ {live_val:>22s}"
        for _, _, m in results:
            line += f" │ {fmt_fn(m):>30s}"
        print(line)

    # 5) Calcul de l'impact
    _, _, m_intra = results[0]
    _, _, m_close = results[1]

    n_intra = m_intra.get("n_trades", 0)
    n_close = m_close.get("n_trades", 0)
    wr_intra = m_intra.get("win_rate", 0)
    wr_close = m_close.get("win_rate", 0)
    ret_intra = m_intra.get("total_return", 0)
    ret_close = m_close.get("total_return", 0)
    pnl_intra = ret_intra * 940
    pnl_close = ret_close * 940
    sharpe_intra = m_intra.get("sharpe", 0)
    sharpe_close = m_close.get("sharpe", 0)
    pf_intra = m_intra.get("profit_factor", 0)
    pf_close = m_close.get("profit_factor", 0)

    print(f"\n{sep}")
    print(f"  🎯 IMPACT DU MODE SL")
    print(sep)

    print(f"\n  {'':25s} │ {'Intrabar':>15s} │ {'Close':>15s} │ {'Delta':>15s}")
    print(f"  {'─' * 25}─┼─{'─' * 15}─┼─{'─' * 15}─┼─{'─' * 15}")

    delta_items = [
        ("Trades",   str(n_intra),          str(n_close),          f"{n_close - n_intra:+d}"),
        ("Win Rate", f"{wr_intra:.1%}",     f"{wr_close:.1%}",     f"{(wr_close-wr_intra)*100:+.1f}pp"),
        ("PnL ($)",  f"${pnl_intra:+.2f}",  f"${pnl_close:+.2f}",  f"${pnl_close-pnl_intra:+.2f}"),
        ("Return",   f"{ret_intra:+.1%}",   f"{ret_close:+.1%}",   f"{(ret_close-ret_intra)*100:+.1f}pp"),
        ("Sharpe",   f"{sharpe_intra:.2f}",  f"{sharpe_close:.2f}", f"{sharpe_close-sharpe_intra:+.2f}"),
        ("PF",       f"{pf_intra:.2f}",     f"{pf_close:.2f}",     f"{pf_close-pf_intra:+.2f}"),
    ]

    for label, v_intra, v_close, delta in delta_items:
        print(f"  {label:25s} │ {v_intra:>15s} │ {v_close:>15s} │ {delta:>15s}")

    # SL saved: combien de SL en mode intrabar survivent en mode close
    sl_intra_count = sum(1 for t in (results[0][1].trades if hasattr(results[0][1], 'trades') else [])
                         if getattr(t, 'exit_reason', '') == 'RANGE_SL')
    sl_close_count = sum(1 for t in (results[1][1].trades if hasattr(results[1][1], 'trades') else [])
                         if getattr(t, 'exit_reason', '') == 'RANGE_SL')

    if sl_intra_count > 0:
        saved = sl_intra_count - sl_close_count
        print(f"\n  📊 SL déclenchés intrabar : {sl_intra_count}")
        print(f"     SL déclenchés close    : {sl_close_count}")
        print(f"     → {saved} SL évités ({saved/sl_intra_count*100:.0f}% de faux SL = mèches)")

    print(f"\n  🔑 Conclusion :")
    if pnl_close > pnl_intra:
        print(f"     Le mode SL close rapporte ${pnl_close - pnl_intra:+.2f} de plus")
        print(f"     ({(wr_close-wr_intra)*100:+.1f}pp de WR en plus)")
        print(f"     → Passer le live de OCO tick-by-tick à 'check SL au close H4'")
        print(f"        devrait rapprocher les performances du backtest")
    else:
        print(f"     Peu de différence entre les deux modes.")

    print(f"\n{sep}\n")


if __name__ == "__main__":
    main()
