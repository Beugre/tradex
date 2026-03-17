"""
On-chain metrics fetcher — sources gratuites (pas d'API key).

Fournit le MVRV ratio via CoinMetrics Community API.
Module sans état — une seule fonction publique `fetch_mvrv()`.
"""

from __future__ import annotations

import logging
import time
from typing import Optional

import requests

logger = logging.getLogger("tradex.onchain")

# ── CoinMetrics Community API (gratuit, pas d'API key) ─────────────────────────
_COINMETRICS_URL = (
    "https://community-api.coinmetrics.io/v4/timeseries/asset-metrics"
)
_MVRV_METRIC = "CapMVRVCur"
_TIMEOUT = 10

# Simple in-memory cache to avoid hitting the API too often
_cache: dict[str, tuple[float, float]] = {}   # key → (value, timestamp)
_CACHE_TTL = 3600  # 1 hour


def fetch_mvrv(asset: str = "btc") -> Optional[float]:
    """Fetch the latest MVRV ratio for the given asset.

    Uses CoinMetrics Community API (free, no auth).

    Args:
        asset: Asset identifier (default "btc").

    Returns:
        MVRV ratio as float, or None if unavailable.
    """
    cache_key = f"mvrv_{asset}"
    now = time.time()

    # Check cache
    if cache_key in _cache:
        val, ts = _cache[cache_key]
        if now - ts < _CACHE_TTL:
            return val

    try:
        resp = requests.get(
            _COINMETRICS_URL,
            params={
                "assets": asset,
                "metrics": _MVRV_METRIC,
                "frequency": "1d",
                "page_size": 2,
            },
            timeout=_TIMEOUT,
        )
        resp.raise_for_status()
        data = resp.json().get("data", [])
        if not data:
            logger.warning("MVRV: pas de données retournées par CoinMetrics")
            return None

        # Take the latest entry (last in the sorted-ascending list)
        latest = data[-1]
        mvrv = float(latest.get(_MVRV_METRIC, 0))
        if mvrv <= 0:
            logger.warning("MVRV: valeur invalide (%s)", mvrv)
            return None

        _cache[cache_key] = (mvrv, now)
        logger.debug("MVRV %s = %.4f (date: %s)", asset, mvrv, latest.get("time", "?"))
        return mvrv

    except requests.RequestException as e:
        logger.warning("MVRV fetch failed: %s", e)
        return None
    except (ValueError, KeyError, IndexError) as e:
        logger.warning("MVRV parse error: %s", e)
        return None
