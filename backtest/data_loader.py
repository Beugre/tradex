"""
Téléchargement de bougies H4 historiques depuis l'API Binance.

Cache les données en CSV dans backtest/data/ pour éviter de re-télécharger.
Endpoint public (pas d'auth requise) : GET /api/v3/klines.
"""

from __future__ import annotations

import csv
import logging
import time as _time
from datetime import datetime, timezone
from pathlib import Path

import httpx

from src.core.models import Candle

logger = logging.getLogger(__name__)

# ── Mapping symboles TradeX → Binance ──────────────────────────────────────────

SYMBOL_MAP: dict[str, str] = {
    # ── Paires originales (21) ─────────────────────────────────────────────
    "BTC-USD": "BTCUSDT",
    "ETH-USD": "ETHUSDT",
    "SOL-USD": "SOLUSDT",
    "XRP-USD": "XRPUSDT",
    "LINK-USD": "LINKUSDT",
    "SUI-USD": "SUIUSDT",
    "ADA-USD": "ADAUSDT",
    "DOT-USD": "DOTUSDT",
    "AVAX-USD": "AVAXUSDT",
    "DOGE-USD": "DOGEUSDT",
    "ATOM-USD": "ATOMUSDT",
    "UNI-USD": "UNIUSDT",
    "NEAR-USD": "NEARUSDT",
    "ALGO-USD": "ALGOUSDT",
    "LTC-USD": "LTCUSDT",
    "ETC-USD": "ETCUSDT",
    "FIL-USD": "FILUSDT",
    "AAVE-USD": "AAVEUSDT",
    "INJ-USD": "INJUSDT",
    "SAND-USD": "SANDUSDT",
    "MANA-USD": "MANAUSDT",
    # ── Paires ajoutées pour 6yr backtest (21 nouvelles) ───────────────────
    "MATIC-USD": "MATICUSDT",
    "VET-USD": "VETUSDT",
    "THETA-USD": "THETAUSDT",
    "FTM-USD": "FTMUSDT",
    "CHZ-USD": "CHZUSDT",
    "ENJ-USD": "ENJUSDT",
    "BAT-USD": "BATUSDT",
    "ZIL-USD": "ZILUSDT",
    "ICX-USD": "ICXUSDT",
    "ONE-USD": "ONEUSDT",
    "HBAR-USD": "HBARUSDT",
    "IOTA-USD": "IOTAUSDT",
    "XTZ-USD": "XTZUSDT",
    "EOS-USD": "EOSUSDT",
    "NEO-USD": "NEOUSDT",
    "DASH-USD": "DASHUSDT",
    "ZEC-USD": "ZECUSDT",
    "XLM-USD": "XLMUSDT",
    "TRX-USD": "TRXUSDT",
    "WAVES-USD": "WAVESUSDT",
    "KAVA-USD": "KAVAUSDT",
    # ── Paires Revolut X supplémentaires ───────────────────────────────────
    "APE-USD": "APEUSDT",
    "SHIB-USD": "SHIBUSDT",
    "PEPE-USD": "PEPEUSDT",
    "ARB-USD": "ARBUSDT",
    "OP-USD": "OPUSDT",
    "POL-USD": "POLUSDT",
    "GRT-USD": "GRTUSDT",
    "COMP-USD": "COMPUSDT",
    "SNX-USD": "SNXUSDT",
    "CRV-USD": "CRVUSDT",
    "LDO-USD": "LDOUSDT",
    "YFI-USD": "YFIUSDT",
    "SUSHI-USD": "SUSHIUSDT",
    "AXS-USD": "AXSUSDT",
    "EGLD-USD": "EGLDUSDT",
    "RENDER-USD": "RENDERUSDT",
    "FET-USD": "FETUSDT",
    "BONK-USD": "BONKUSDT",
    "WIF-USD": "WIFUSDT",
    "SEI-USD": "SEIUSDT",
    "TIA-USD": "TIAUSDT",
    "FLOKI-USD": "FLOKIUSDT",
    "JUP-USD": "JUPUSDT",
    # ── Antiliq bot ────────────────────────────────────────────────────────
    "BNB-USD": "BNBUSDT",
}

BINANCE_URLS = [
    "https://api.binance.com",
    "https://api1.binance.com",
    "https://api2.binance.com",
    "https://api3.binance.com",
]

CACHE_DIR = Path(__file__).parent / "data"
_MAX_PER_REQ = 1000


# ── API publique ───────────────────────────────────────────────────────────────


def download_all_pairs(
    pairs: list[str],
    start: datetime,
    end: datetime,
    interval: str = "4h",
) -> dict[str, list[Candle]]:
    """Télécharge les bougies H4 pour toutes les paires."""
    result: dict[str, list[Candle]] = {}
    for pair in pairs:
        logger.info("📥 %s (%s → %s)…", pair, start.date(), end.date())
        candles = download_candles(pair, start, end, interval)
        result[pair] = candles
        logger.info("   ✅ %s : %d bougies", pair, len(candles))
    return result


def download_candles(
    symbol: str,
    start: datetime,
    end: datetime,
    interval: str = "4h",
    use_cache: bool = True,
) -> list[Candle]:
    """Télécharge avec pagination et cache CSV."""
    bsym = SYMBOL_MAP.get(symbol)
    if bsym is None:
        raise ValueError(f"Symbole inconnu : {symbol}")

    cache = CACHE_DIR / f"{bsym}_{interval}_{start:%Y%m%d}_{end:%Y%m%d}.csv"
    if use_cache and cache.exists():
        candles = _load_csv(cache)
        logger.info("   📦 Cache : %d bougies (%s)", len(candles), cache.name)
        return candles

    start_ms = int(start.timestamp() * 1000)
    end_ms = int(end.timestamp() * 1000)
    all_candles: list[Candle] = []
    cursor = start_ms

    while cursor < end_ms:
        klines = _fetch(bsym, interval, cursor, end_ms)
        if not klines:
            break
        for k in klines:
            all_candles.append(
                Candle(
                    timestamp=int(k[0]),
                    open=float(k[1]),
                    high=float(k[2]),
                    low=float(k[3]),
                    close=float(k[4]),
                    volume=float(k[5]),
                )
            )
        cursor = int(klines[-1][0]) + 1
        _time.sleep(0.15)

    if all_candles:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        _save_csv(all_candles, cache)

    return all_candles


# ── Helpers ────────────────────────────────────────────────────────────────────


def _fetch(symbol: str, interval: str, start_ms: int, end_ms: int) -> list:
    params = {
        "symbol": symbol,
        "interval": interval,
        "startTime": start_ms,
        "endTime": end_ms,
        "limit": _MAX_PER_REQ,
    }
    for base in BINANCE_URLS:
        try:
            with httpx.Client(timeout=30) as c:
                r = c.get(f"{base}/api/v3/klines", params=params)
                r.raise_for_status()
                return r.json()
        except Exception as exc:  # noqa: BLE001
            logger.warning("  ⚠️ %s échec : %s", base, exc)
    raise RuntimeError(f"Impossible de récupérer les klines {symbol}")


def _save_csv(candles: list[Candle], path: Path) -> None:
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["timestamp", "open", "high", "low", "close", "volume"])
        for c in candles:
            w.writerow([c.timestamp, c.open, c.high, c.low, c.close, c.volume])


def _load_csv(path: Path) -> list[Candle]:
    candles: list[Candle] = []
    with open(path) as f:
        for row in csv.DictReader(f):
            candles.append(
                Candle(
                    timestamp=int(row["timestamp"]),
                    open=float(row["open"]),
                    high=float(row["high"]),
                    low=float(row["low"]),
                    close=float(row["close"]),
                    volume=float(row["volume"]),
                )
            )
    return candles


# ── BTC D1 pour filtre EMA200 ─────────────────────────────────────────────────


def download_btc_d1(
    start: datetime,
    end: datetime,
    use_cache: bool = True,
) -> list[Candle]:
    """Télécharge les bougies DAILY BTC pour le calcul de l'EMA200.

    On ajoute ~250 jours de marge avant `start` pour que l'EMA200
    soit déjà « warm » au début de la période de backtest.
    """
    from datetime import timedelta

    warmup_days = 250  # > 200 pour que l'EMA soit stable
    real_start = start - timedelta(days=warmup_days)

    logger.info(
        "📥 BTC-USD D1 (EMA warmup %s → backtest %s → %s)…",
        real_start.date(), start.date(), end.date(),
    )
    candles = download_candles(
        "BTC-USD", real_start, end, interval="1d", use_cache=use_cache,
    )
    logger.info("   ✅ BTC D1 : %d bougies", len(candles))
    return candles


def download_all_pairs_d1(
    pairs: list[str],
    start: datetime,
    end: datetime,
    use_cache: bool = True,
) -> dict[str, list[Candle]]:
    """Télécharge les bougies D1 pour toutes les paires.

    Warmup de 250 jours pour :
    - EMA200 BTC (250 jours)
    - Swing detection D1 (~50 jours suffisent mais on prend la marge EMA)
    """
    from datetime import timedelta

    warmup_days = 250
    real_start = start - timedelta(days=warmup_days)

    result: dict[str, list[Candle]] = {}
    for pair in pairs:
        logger.info("📥 %s D1 (%s → %s)…", pair, real_start.date(), end.date())
        candles = download_candles(
            pair, real_start, end, interval="1d", use_cache=use_cache,
        )
        result[pair] = candles
        logger.info("   ✅ %s D1 : %d bougies", pair, len(candles))
    return result
