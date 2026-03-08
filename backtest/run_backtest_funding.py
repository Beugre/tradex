#!/usr/bin/env python3
"""
Backtest : Funding Rate Reversion (FRR)

Exploite les extremes du funding rate sur les perpétuels Binance
pour signaler des entrées spot (long only).

Règles :
  LONG :
    - funding_rate < seuil négatif (ex: -0.03%)
    - RSI(14) < 40
    - Price > EMA200 (H4)
  SL : 2 × ATR(14)
  TP : TP1 +2% (40%), TP2 +4% (40%), TP3 trailing 1.5×ATR (20%)
  Breakeven après TP1.

  Filtre : pas de SHORT (spot only).

Données funding : Binance Futures API (publique, pas d'auth).

Usage :
    python -m backtest.run_backtest_funding --years 6 --balance 500
    python -m backtest.run_backtest_funding --years 6 --balance 500 --variants
"""

from __future__ import annotations

import argparse
import csv
import logging
import time as _time
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path

import httpx
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from backtest.data_loader import download_candles, SYMBOL_MAP
from src.core.models import Candle
from src.core.indicators import ema, sma, atr_series, rsi_series

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(message)s")
logger = logging.getLogger(__name__)

OUTPUT_DIR = Path(__file__).parent / "output"
OUTPUT_DIR.mkdir(exist_ok=True)
CACHE_DIR = Path(__file__).parent / "data"
CACHE_DIR.mkdir(exist_ok=True)

PAIRS = ["BTC-USD", "ETH-USD", "SOL-USD", "BNB-USD", "LINK-USD"]

MAKER_FEE = 0.0
TAKER_FEE = 0.0009

# Binance Futures symbol map (USDT perps)
FUTURES_SYMBOL_MAP: dict[str, str] = {
    "BTC-USD": "BTCUSDT",
    "ETH-USD": "ETHUSDT",
    "SOL-USD": "SOLUSDT",
    "BNB-USD": "BNBUSDT",
    "LINK-USD": "LINKUSDT",
    "ADA-USD": "ADAUSDT",
    "DOT-USD": "DOTUSDT",
    "AVAX-USD": "AVAXUSDT",
    "XRP-USD": "XRPUSDT",
    "DOGE-USD": "DOGEUSDT",
    "LTC-USD": "LTCUSDT",
    "AAVE-USD": "AAVEUSDT",
}


# ══════════════════════════════════════════════════════════════════════════════
#  FUNDING RATE DATA
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class FundingEntry:
    timestamp: int  # ms
    rate: float     # e.g. 0.0001 = 0.01%


def download_funding_rates(
    symbol: str,
    start: datetime,
    end: datetime,
    use_cache: bool = True,
) -> list[FundingEntry]:
    """Télécharge les funding rates depuis Binance Futures API (public)."""
    fsym = FUTURES_SYMBOL_MAP.get(symbol)
    if fsym is None:
        raise ValueError(f"Pas de symbole futures pour {symbol}")

    cache = CACHE_DIR / f"funding_{fsym}_{start:%Y%m%d}_{end:%Y%m%d}.csv"
    if use_cache and cache.exists():
        entries = _load_funding_csv(cache)
        logger.info("   📦 Funding cache : %d entrées (%s)", len(entries), cache.name)
        return entries

    start_ms = int(start.timestamp() * 1000)
    end_ms = int(end.timestamp() * 1000)
    all_entries: list[FundingEntry] = []
    cursor = start_ms

    urls = [
        "https://fapi.binance.com",
    ]

    while cursor < end_ms:
        params = {
            "symbol": fsym,
            "startTime": cursor,
            "endTime": end_ms,
            "limit": 1000,
        }
        data = None
        for base in urls:
            try:
                with httpx.Client(timeout=30) as c:
                    r = c.get(f"{base}/fapi/v1/fundingRate", params=params)
                    r.raise_for_status()
                    data = r.json()
                    break
            except Exception as exc:
                logger.warning("  ⚠️ Funding %s : %s", base, exc)

        if not data:
            logger.warning("  ⚠️ Pas de données funding pour %s à partir de %d", fsym, cursor)
            break

        for entry in data:
            all_entries.append(FundingEntry(
                timestamp=int(entry["fundingTime"]),
                rate=float(entry["fundingRate"]),
            ))

        if len(data) < 1000:
            break
        cursor = int(data[-1]["fundingTime"]) + 1
        _time.sleep(0.2)

    if all_entries:
        _save_funding_csv(all_entries, cache)

    logger.info("   ✅ Funding %s : %d entrées", symbol, len(all_entries))
    return all_entries


def _save_funding_csv(entries: list[FundingEntry], path: Path) -> None:
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["timestamp", "rate"])
        for e in entries:
            w.writerow([e.timestamp, e.rate])


def _load_funding_csv(path: Path) -> list[FundingEntry]:
    entries: list[FundingEntry] = []
    with open(path) as f:
        for row in csv.DictReader(f):
            entries.append(FundingEntry(
                timestamp=int(row["timestamp"]),
                rate=float(row["rate"]),
            ))
    return entries


def align_funding_to_candles(
    funding: list[FundingEntry],
    candles: list[Candle],
) -> list[float]:
    """Aligne les funding rates aux bougies H4.

    Chaque bougie H4 reçoit le funding rate le plus récent
    dont le timestamp ≤ au timestamp de la bougie.
    Le funding arrive toutes les 8h, donc certaines bougies H4
    partagent le même funding.
    """
    rates = [0.0] * len(candles)
    f_idx = 0
    last_rate = 0.0

    for i, c in enumerate(candles):
        while f_idx < len(funding) and funding[f_idx].timestamp <= c.timestamp:
            last_rate = funding[f_idx].rate
            f_idx += 1
        rates[i] = last_rate

    return rates


# ══════════════════════════════════════════════════════════════════════════════
#  CONFIG
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class FRRConfig:
    """Configuration pour Funding Rate Reversion."""
    # ── Funding thresholds ──
    funding_long_threshold: float = -0.0003    # < -0.03% → LONG signal
    # (funding_rate est exprimé en décimal : -0.0003 = -0.03%)

    # ── RSI ──
    rsi_period: int = 14
    rsi_max_long: float = 40.0

    # ── EMA trend filter ──
    ema_period: int = 200

    # ── ATR ──
    atr_period: int = 14

    # ── SL ──
    sl_atr_mult: float = 2.0

    # ── TP ladder ──
    tp1_pct: float = 0.02              # +2%
    tp2_pct: float = 0.04              # +4%
    tp1_share: float = 0.40
    tp2_share: float = 0.40
    tp3_share: float = 0.20

    # ── TP3 trailing ──
    tp3_trail_atr_mult: float = 1.5    # trailing stop = 1.5×ATR

    # ── Breakeven ──
    breakeven_after_tp1: bool = True

    # ── Risk ──
    risk_per_trade: float = 0.015      # 1.5% equity
    max_positions: int = 4
    max_exposure_pct: float = 0.50

    # ── Cooldown ──
    cooldown_bars: int = 6             # ~24h en H4

    # ── Funding persistence ──
    funding_lookback: int = 1          # Nombre de funding consécutifs négatifs requis


# ══════════════════════════════════════════════════════════════════════════════
#  DATA STRUCTURES
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class Trade:
    symbol: str
    side: str
    entry_price: float
    exit_price: float
    size: float
    size_usd: float
    entry_time: int
    exit_time: int
    pnl_usd: float
    pnl_pct: float
    exit_reason: str
    hold_bars: int = 0
    funding_at_entry: float = 0.0


@dataclass
class BacktestResult:
    label: str
    trades: list[Trade]
    equity_curve: list[float]
    initial_balance: float
    final_equity: float
    config_desc: dict


@dataclass
class _OpenPos:
    symbol: str
    entry_price: float
    sl_price: float
    initial_size: float
    initial_size_usd: float
    remaining_size: float
    remaining_size_usd: float
    entry_bar: int
    entry_ts: int
    funding_at_entry: float
    tp1_hit: bool = False
    tp2_hit: bool = False
    tp1_price: float = 0.0
    tp2_price: float = 0.0
    # TP3 trailing
    trail_active: bool = False
    trail_highest: float = 0.0
    trail_stop: float = 0.0


@dataclass
class _PairState:
    cooldown_until: int = 0


# ══════════════════════════════════════════════════════════════════════════════
#  INDICATORS
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class FRRIndicators:
    rsi: list[float] = field(default_factory=list)
    ema200: list[float] = field(default_factory=list)
    atr: list[float] = field(default_factory=list)
    funding_aligned: list[float] = field(default_factory=list)


def compute_frr_indicators(
    candles: list[Candle],
    funding: list[FundingEntry],
    cfg: FRRConfig,
) -> FRRIndicators:
    ind = FRRIndicators()
    ind.rsi = rsi_series(candles, cfg.rsi_period)
    closes = [c.close for c in candles]
    ind.ema200 = ema(closes, cfg.ema_period)
    ind.atr = atr_series(candles, cfg.atr_period)
    ind.funding_aligned = align_funding_to_candles(funding, candles)
    return ind


# ══════════════════════════════════════════════════════════════════════════════
#  SIMULATOR
# ══════════════════════════════════════════════════════════════════════════════

def simulate_frr(
    all_candles: dict[str, list[Candle]],
    all_funding: dict[str, list[FundingEntry]],
    cfg: FRRConfig,
    initial_balance: float = 500.0,
) -> BacktestResult:
    balance = initial_balance
    positions: list[_OpenPos] = []
    closed_trades: list[Trade] = []
    equity_curve: list[float] = [balance]

    all_ind: dict[str, FRRIndicators] = {}
    for symbol in all_candles:
        funding = all_funding.get(symbol, [])
        all_ind[symbol] = compute_frr_indicators(all_candles[symbol], funding, cfg)

    states: dict[str, _PairState] = {sym: _PairState() for sym in all_candles}
    min_len = min(len(c) for c in all_candles.values())

    start_bar = max(cfg.ema_period + 10, cfg.atr_period + 5, cfg.rsi_period + 5)

    for bar_idx in range(start_bar, min_len):

        # ── Gestion positions ouvertes ──
        for pos in positions[:]:
            c = all_candles[pos.symbol][bar_idx]
            ind = all_ind[pos.symbol]
            hold = bar_idx - pos.entry_bar

            # Check SL
            if c.low <= pos.sl_price:
                exit_price = pos.sl_price
                pnl_pct = (exit_price - pos.entry_price) / pos.entry_price
                pnl_usd = pos.remaining_size_usd * pnl_pct - pos.remaining_size_usd * TAKER_FEE
                balance += pos.remaining_size_usd + pnl_usd
                reason = "BE" if pos.tp1_hit and cfg.breakeven_after_tp1 and pos.sl_price >= pos.entry_price else "SL"
                closed_trades.append(Trade(
                    symbol=pos.symbol, side="LONG",
                    entry_price=pos.entry_price, exit_price=exit_price,
                    size=pos.remaining_size, size_usd=pos.remaining_size_usd,
                    entry_time=pos.entry_ts, exit_time=c.timestamp,
                    pnl_usd=pnl_usd, pnl_pct=pnl_pct * 100,
                    exit_reason=reason, hold_bars=hold,
                    funding_at_entry=pos.funding_at_entry,
                ))
                positions.remove(pos)
                continue

            # Check trailing stop (TP3 runner)
            if pos.trail_active:
                if c.high > pos.trail_highest:
                    pos.trail_highest = c.high
                    current_atr = ind.atr[bar_idx] if bar_idx < len(ind.atr) else ind.atr[-1]
                    pos.trail_stop = pos.trail_highest - cfg.tp3_trail_atr_mult * current_atr
                    # Trail stop ne descend jamais en-dessous du TP2
                    pos.trail_stop = max(pos.trail_stop, pos.tp2_price)

                if c.low <= pos.trail_stop:
                    exit_price = pos.trail_stop
                    close_size = pos.remaining_size
                    close_size_usd = pos.remaining_size_usd
                    pnl_pct = (exit_price - pos.entry_price) / pos.entry_price
                    pnl_usd = close_size_usd * pnl_pct - close_size_usd * MAKER_FEE
                    balance += close_size_usd + pnl_usd
                    closed_trades.append(Trade(
                        symbol=pos.symbol, side="LONG",
                        entry_price=pos.entry_price, exit_price=exit_price,
                        size=close_size, size_usd=close_size_usd,
                        entry_time=pos.entry_ts, exit_time=c.timestamp,
                        pnl_usd=pnl_usd, pnl_pct=pnl_pct * 100,
                        exit_reason="TRAIL", hold_bars=hold,
                        funding_at_entry=pos.funding_at_entry,
                    ))
                    positions.remove(pos)
                    continue

            # TP1
            if not pos.tp1_hit and c.high >= pos.tp1_price:
                close_size = pos.initial_size * cfg.tp1_share
                close_size_usd = pos.initial_size_usd * cfg.tp1_share
                pnl_pct = (pos.tp1_price - pos.entry_price) / pos.entry_price
                pnl_usd = close_size_usd * pnl_pct - close_size_usd * MAKER_FEE
                balance += close_size_usd + pnl_usd
                closed_trades.append(Trade(
                    symbol=pos.symbol, side="LONG",
                    entry_price=pos.entry_price, exit_price=pos.tp1_price,
                    size=close_size, size_usd=close_size_usd,
                    entry_time=pos.entry_ts, exit_time=c.timestamp,
                    pnl_usd=pnl_usd, pnl_pct=pnl_pct * 100,
                    exit_reason="TP1", hold_bars=hold,
                    funding_at_entry=pos.funding_at_entry,
                ))
                pos.tp1_hit = True
                pos.remaining_size -= close_size
                pos.remaining_size_usd -= close_size_usd
                if cfg.breakeven_after_tp1:
                    pos.sl_price = pos.entry_price

            # TP2
            if not pos.tp2_hit and pos.tp1_hit and c.high >= pos.tp2_price:
                close_size = pos.initial_size * cfg.tp2_share
                close_size_usd = pos.initial_size_usd * cfg.tp2_share
                pnl_pct = (pos.tp2_price - pos.entry_price) / pos.entry_price
                pnl_usd = close_size_usd * pnl_pct - close_size_usd * MAKER_FEE
                balance += close_size_usd + pnl_usd
                closed_trades.append(Trade(
                    symbol=pos.symbol, side="LONG",
                    entry_price=pos.entry_price, exit_price=pos.tp2_price,
                    size=close_size, size_usd=close_size_usd,
                    entry_time=pos.entry_ts, exit_time=c.timestamp,
                    pnl_usd=pnl_usd, pnl_pct=pnl_pct * 100,
                    exit_reason="TP2", hold_bars=hold,
                    funding_at_entry=pos.funding_at_entry,
                ))
                pos.tp2_hit = True
                pos.remaining_size -= close_size
                pos.remaining_size_usd -= close_size_usd
                # Activer trailing stop pour TP3
                pos.trail_active = True
                pos.trail_highest = c.high
                current_atr = ind.atr[bar_idx] if bar_idx < len(ind.atr) else ind.atr[-1]
                pos.trail_stop = pos.trail_highest - cfg.tp3_trail_atr_mult * current_atr

            if pos.remaining_size_usd < 1:
                positions.remove(pos)

        # ── Détection de nouvelles entrées ──
        if len(positions) < cfg.max_positions and balance > 10:
            total_exposure = sum(p.remaining_size_usd for p in positions)
            current_eq = balance + total_exposure
            max_exposure = current_eq * cfg.max_exposure_pct

            for symbol in all_candles:
                if any(p.symbol == symbol for p in positions):
                    continue
                if len(positions) >= cfg.max_positions:
                    break

                state = states[symbol]
                if bar_idx < state.cooldown_until:
                    continue

                candles = all_candles[symbol]
                c = candles[bar_idx]
                ind = all_ind[symbol]

                # ── 1. Funding rate check ──
                fr = ind.funding_aligned[bar_idx] if bar_idx < len(ind.funding_aligned) else 0
                if fr >= cfg.funding_long_threshold:
                    continue  # Funding pas assez négatif

                # ── 2. RSI check ──
                rsi_val = ind.rsi[bar_idx] if bar_idx < len(ind.rsi) else 50
                if rsi_val >= cfg.rsi_max_long:
                    continue

                # ── 3. EMA200 trend filter (price > EMA200) ──
                ema_val = ind.ema200[bar_idx] if bar_idx < len(ind.ema200) else 0
                if ema_val <= 0:
                    continue
                if c.close <= ema_val:
                    continue

                # ── 4. ATR & sizing ──
                current_atr = ind.atr[bar_idx] if bar_idx < len(ind.atr) else 0
                if current_atr <= 0:
                    continue

                entry_price = c.close
                sl_price = entry_price - cfg.sl_atr_mult * current_atr
                sl_distance = entry_price - sl_price
                if sl_distance <= 0:
                    continue

                equity = balance + sum(p.remaining_size_usd for p in positions)
                risk_amount = equity * cfg.risk_per_trade
                size = risk_amount / sl_distance
                size_usd = size * entry_price

                remaining_exposure = max_exposure - total_exposure
                if size_usd > remaining_exposure:
                    size_usd = remaining_exposure
                    size = size_usd / entry_price if entry_price > 0 else 0

                if size_usd < 5:
                    continue

                # TP prices
                tp1_price = entry_price * (1 + cfg.tp1_pct)
                tp2_price = entry_price * (1 + cfg.tp2_pct)

                # Execute
                fee = size_usd * MAKER_FEE
                balance -= size_usd + fee
                total_exposure += size_usd

                positions.append(_OpenPos(
                    symbol=symbol, entry_price=entry_price,
                    sl_price=sl_price,
                    initial_size=size, initial_size_usd=size_usd,
                    remaining_size=size, remaining_size_usd=size_usd,
                    entry_bar=bar_idx, entry_ts=c.timestamp,
                    funding_at_entry=fr,
                    tp1_price=tp1_price, tp2_price=tp2_price,
                ))
                state.cooldown_until = bar_idx + cfg.cooldown_bars

        # ── Equity tracking ──
        pos_value = sum(
            p.remaining_size * all_candles[p.symbol][min(bar_idx, len(all_candles[p.symbol]) - 1)].close
            for p in positions
        )
        equity_curve.append(balance + pos_value)

    # Clôturer positions restantes
    for pos in positions:
        last = all_candles[pos.symbol][min_len - 1]
        pnl_pct = (last.close - pos.entry_price) / pos.entry_price
        pnl_usd = pos.remaining_size_usd * pnl_pct - pos.remaining_size_usd * TAKER_FEE
        balance += pos.remaining_size_usd + pnl_usd
        closed_trades.append(Trade(
            symbol=pos.symbol, side="LONG",
            entry_price=pos.entry_price, exit_price=last.close,
            size=pos.remaining_size, size_usd=pos.remaining_size_usd,
            entry_time=pos.entry_ts, exit_time=last.timestamp,
            pnl_usd=pnl_usd, pnl_pct=pnl_pct * 100,
            exit_reason="END", hold_bars=min_len - 1 - pos.entry_bar,
            funding_at_entry=pos.funding_at_entry,
        ))

    return BacktestResult(
        label="", trades=closed_trades, equity_curve=equity_curve,
        initial_balance=initial_balance,
        final_equity=equity_curve[-1] if equity_curve else initial_balance,
        config_desc={},
    )


# ══════════════════════════════════════════════════════════════════════════════
#  VARIANTS
# ══════════════════════════════════════════════════════════════════════════════

def _define_variants(
    run_variants: bool = False,
) -> list[tuple[str, FRRConfig, dict, list[str] | None]]:
    configs = []

    # ── BASE : funding < -0.03%, RSI < 40, EMA200 ──
    configs.append(("FRR_BASE", FRRConfig(), {
        "funding": "<-0.03%", "rsi": "<40", "ema": "200",
        "sl": "2×ATR", "tp": "2%/4%/trail",
    }, None))

    # ── 3 paires (BTC, ETH, SOL) ──
    configs.append(("FRR_3P", FRRConfig(), {
        "pairs": "BTC,ETH,SOL",
    }, ["BTC-USD", "ETH-USD", "SOL-USD"]))

    if run_variants:
        # ── V1 : Funding < -0.01% (plus permissif) ──
        configs.append(("FRR_FR01", FRRConfig(
            funding_long_threshold=-0.0001,
        ), {"funding": "<-0.01%"}, None))

        # ── V2 : Funding < -0.05% (plus strict) ──
        configs.append(("FRR_FR05", FRRConfig(
            funding_long_threshold=-0.0005,
        ), {"funding": "<-0.05%"}, None))

        # ── V3 : RSI < 50 (plus permissif) ──
        configs.append(("FRR_RSI50", FRRConfig(
            rsi_max_long=50.0,
        ), {"rsi": "<50"}, None))

        # ── V4 : RSI < 30 (plus strict) ──
        configs.append(("FRR_RSI30", FRRConfig(
            rsi_max_long=30.0,
        ), {"rsi": "<30"}, None))

        # ── V5 : Sans filtre EMA (pas de trend filter) ──
        configs.append(("FRR_NO_EMA", FRRConfig(
            ema_period=1,  # essentially disabled
        ), {"ema": "off"}, None))

        # ── V6 : SL 1.5×ATR (plus serré) ──
        configs.append(("FRR_SL15", FRRConfig(
            sl_atr_mult=1.5,
        ), {"sl": "1.5×ATR"}, None))

        # ── V7 : SL 3×ATR (plus large) ──
        configs.append(("FRR_SL3", FRRConfig(
            sl_atr_mult=3.0,
        ), {"sl": "3×ATR"}, None))

        # ── V8 : TP 3%/6%/trail (plus large) ──
        configs.append(("FRR_WIDE_TP", FRRConfig(
            tp1_pct=0.03,
            tp2_pct=0.06,
        ), {"tp": "3%/6%/trail"}, None))

        # ── V9 : TP 1%/2%/trail (plus serré) ──
        configs.append(("FRR_TIGHT_TP", FRRConfig(
            tp1_pct=0.01,
            tp2_pct=0.02,
        ), {"tp": "1%/2%/trail"}, None))

        # ── V10 : Combo permissif (FR<-0.01%, RSI<50, no EMA) ──
        configs.append(("FRR_LOOSE", FRRConfig(
            funding_long_threshold=-0.0001,
            rsi_max_long=50.0,
            ema_period=1,
        ), {"combo": "FR-0.01%+RSI50+noEMA"}, None))

        # ── V11 : Combo strict (FR<-0.05%, RSI<30) ──
        configs.append(("FRR_STRICT", FRRConfig(
            funding_long_threshold=-0.0005,
            rsi_max_long=30.0,
        ), {"combo": "FR-0.05%+RSI30"}, None))

        # ── V12 : Risk 2.5% ──
        configs.append(("FRR_RISK25", FRRConfig(
            risk_per_trade=0.025,
            max_exposure_pct=0.60,
        ), {"risk": "2.5%"}, None))

        # ── V13 : Sans breakeven ──
        configs.append(("FRR_NO_BE", FRRConfig(
            breakeven_after_tp1=False,
        ), {"BE": "off"}, None))

    return configs


# ══════════════════════════════════════════════════════════════════════════════
#  REPORTING
# ══════════════════════════════════════════════════════════════════════════════

def compute_kpis(result: BacktestResult) -> dict:
    trades = result.trades
    if not trades:
        return {
            "label": result.label, "trades": 0, "win_rate": 0, "pf": 0,
            "pnl": 0, "avg_pnl": 0, "max_dd": 0, "final": result.initial_balance,
            "rr": 0, "avg_hold": 0, "avg_win_pct": 0, "avg_loss_pct": 0,
        }

    wins = [t for t in trades if t.pnl_usd > 0]
    losses = [t for t in trades if t.pnl_usd <= 0]
    total_gains = sum(t.pnl_usd for t in wins) if wins else 0
    total_losses = abs(sum(t.pnl_usd for t in losses)) if losses else 0.001

    peak = result.equity_curve[0]
    max_dd = 0
    for eq in result.equity_curve:
        if eq > peak:
            peak = eq
        dd = (peak - eq) / peak * 100 if peak > 0 else 0
        if dd > max_dd:
            max_dd = dd

    avg_win = (total_gains / len(wins)) if wins else 0
    avg_loss = (total_losses / len(losses)) if losses else 0.001

    return {
        "label": result.label,
        "trades": len(trades),
        "win_rate": len(wins) / len(trades) * 100 if trades else 0,
        "pf": total_gains / total_losses if total_losses > 0 else 0,
        "pnl": result.final_equity - result.initial_balance,
        "avg_pnl": sum(t.pnl_usd for t in trades) / len(trades) if trades else 0,
        "max_dd": max_dd,
        "final": result.final_equity,
        "rr": avg_win / avg_loss if avg_loss > 0 else 0,
        "avg_hold": sum(t.hold_bars for t in trades) / len(trades) if trades else 0,
        "avg_win_pct": (sum(t.pnl_pct for t in wins) / len(wins)) if wins else 0,
        "avg_loss_pct": (sum(t.pnl_pct for t in losses) / len(losses)) if losses else 0,
    }


def print_table(kpis_list: list[dict], title: str) -> None:
    print(f"\n{'='*120}")
    print(f"  {title}")
    print(f"{'='*120}")
    hdr = f"{'Config':<22} {'Trades':>6} {'WR%':>7} {'PF':>7} {'PnL $':>9} {'Avg PnL':>8} {'Max DD%':>8} {'Final $':>9} {'R:R':>5} {'Avg Hold':>9}"
    print(hdr)
    print("-" * 120)
    for k in kpis_list:
        print(
            f"{k['label']:<22} {k['trades']:>6} {k['win_rate']:>6.1f}% {k['pf']:>7.2f} "
            f"{k['pnl']:>+9.2f}$ {k['avg_pnl']:>+7.2f}$ {k['max_dd']:>7.1f}% "
            f"{k['final']:>9.2f}$ {k['rr']:>5.2f} {k['avg_hold']:>8.1f}b"
        )
    print("-" * 120)
    if kpis_list:
        best = max(kpis_list, key=lambda k: k["pf"])
        print(f"  Meilleur PF : {best['label']} (PF {best['pf']:.2f}, PnL {best['pnl']:+.2f}$, WR {best['win_rate']:.1f}%)")


def print_exit_breakdown(results: list[BacktestResult], prefix: str) -> None:
    print(f"\n{'='*70}")
    print(f"  {prefix} — Répartition des sorties")
    print(f"{'='*70}")
    for res in results:
        if not res.trades:
            continue
        counter = Counter(t.exit_reason for t in res.trades)
        print(f"\n  {res.label}:")
        for reason, count in counter.most_common():
            pct = count / len(res.trades) * 100
            avg = sum(t.pnl_usd for t in res.trades if t.exit_reason == reason) / count
            avg_p = sum(t.pnl_pct for t in res.trades if t.exit_reason == reason) / count
            print(f"    {reason:<16}: {count:>4} ({pct:>5.1f}%)  avg PnL: {avg:>+.2f}$  avg %: {avg_p:>+.2f}%")


def print_per_pair(results: list[BacktestResult]) -> None:
    for res in results[:3]:
        if not res.trades:
            continue
        print(f"\n{'='*70}")
        print(f"  {res.label} — Stats par paire")
        print(f"{'='*70}")
        pairs = sorted(set(t.symbol for t in res.trades))
        print(f"  {'Paire':<12} {'Trades':>6} {'WR%':>7} {'PnL $':>9} {'Avg FR%':>9} {'Avg Win%':>9} {'Avg Loss%':>10}")
        print(f"  {'-'*65}")
        for pair in pairs:
            pt = [t for t in res.trades if t.symbol == pair]
            w = [t for t in pt if t.pnl_usd > 0]
            l = [t for t in pt if t.pnl_usd <= 0]
            wr = len(w) / len(pt) * 100 if pt else 0
            pnl = sum(t.pnl_usd for t in pt)
            avg_fr = sum(t.funding_at_entry for t in pt) / len(pt) * 100 if pt else 0  # en %
            awp = sum(t.pnl_pct for t in w) / len(w) if w else 0
            alp = sum(t.pnl_pct for t in l) / len(l) if l else 0
            print(f"  {pair:<12} {len(pt):>6} {wr:>6.1f}% {pnl:>+9.2f}$ {avg_fr:>+8.4f}% {awp:>+8.2f}% {alp:>+9.2f}%")


def print_funding_stats(all_funding: dict[str, list[FundingEntry]]) -> None:
    """Affiche des statistiques sur les funding rates téléchargés."""
    print(f"\n{'='*70}")
    print(f"  Statistiques Funding Rate")
    print(f"{'='*70}")
    for symbol, entries in sorted(all_funding.items()):
        if not entries:
            print(f"  {symbol}: aucune donnée")
            continue
        rates = [e.rate for e in entries]
        neg = [r for r in rates if r < 0]
        very_neg = [r for r in rates if r < -0.0003]
        print(
            f"  {symbol:<12}: {len(entries):>5} entrées | "
            f"min: {min(rates)*100:>+.4f}% | max: {max(rates)*100:>+.4f}% | "
            f"mean: {sum(rates)/len(rates)*100:>+.4f}% | "
            f"négatifs: {len(neg)} ({len(neg)/len(rates)*100:.1f}%) | "
            f"< -0.03%: {len(very_neg)} ({len(very_neg)/len(rates)*100:.1f}%)"
        )


def plot_equity(results: list[BacktestResult], title: str, filename: str) -> None:
    fig, ax = plt.subplots(figsize=(14, 6))
    for res in results:
        if res.trades:
            ax.plot(res.equity_curve, label=res.label, linewidth=1)
    ax.axhline(y=results[0].initial_balance, color="grey", linestyle="--", alpha=0.5)
    ax.set_title(title)
    ax.set_xlabel("Bar")
    ax.set_ylabel("Equity ($)")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)
    path = OUTPUT_DIR / filename
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\n  Chart : {path}")


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    parser = argparse.ArgumentParser(description="Backtest Funding Rate Reversion")
    parser.add_argument("--years", type=int, default=6, help="Années de données")
    parser.add_argument("--balance", type=float, default=500, help="Capital initial ($)")
    parser.add_argument("--variants", action="store_true", help="Tester aussi les variantes")
    args = parser.parse_args()

    end = datetime.now(timezone.utc)
    start = end - timedelta(days=args.years * 365)

    print(f"\n{'═'*70}")
    print(f"  Funding Rate Reversion — Téléchargement H4 ({args.years} ans)")
    print(f"  Paires : {', '.join(PAIRS)}")
    print(f"{'═'*70}")

    # Download H4 candles
    all_candles: dict[str, list[Candle]] = {}
    for pair in PAIRS:
        logger.info("  %s H4…", pair)
        candles = download_candles(pair, start, end, interval="4h")
        all_candles[pair] = candles
        n_days = len(candles) // 6
        logger.info("    %s : %d bougies H4 (%d jours)", pair, len(candles), n_days)

    # Download funding rates
    print(f"\n{'═'*70}")
    print(f"  Téléchargement Funding Rates (Binance Futures)")
    print(f"{'═'*70}")

    all_funding: dict[str, list[FundingEntry]] = {}
    for pair in PAIRS:
        logger.info("  %s funding…", pair)
        try:
            funding = download_funding_rates(pair, start, end)
            all_funding[pair] = funding
        except Exception as exc:
            logger.warning("  ⚠️ Funding %s échoué : %s", pair, exc)
            all_funding[pair] = []

    print_funding_stats(all_funding)

    # Run variants
    configs = _define_variants(run_variants=args.variants)
    results: list[BacktestResult] = []
    kpis_list: list[dict] = []

    for label, cfg, desc, pairs_override in configs:
        logger.info("  %s…", label)
        candles_run = {k: v for k, v in all_candles.items() if k in pairs_override} if pairs_override else all_candles
        funding_run = {k: v for k, v in all_funding.items() if k in (pairs_override or all_funding)}
        if not candles_run:
            continue
        result = simulate_frr(candles_run, funding_run, cfg, initial_balance=args.balance)
        result.label = label
        result.config_desc = desc
        results.append(result)
        kpis = compute_kpis(result)
        kpis_list.append(kpis)

    n_bars = min(len(c) for c in all_candles.values())
    print_table(kpis_list, f"Funding Rate Reversion — H4 ({n_bars} bars, {args.years} ans, ${args.balance:.0f})")
    print_exit_breakdown(results, "FRR")
    print_per_pair(results)
    plot_equity(results, "Funding Rate Reversion (H4)", "frr_equity.png")

    # Verdict
    valid = [k for k in kpis_list if k["trades"] >= 10]
    if valid:
        best = max(valid, key=lambda k: k["pf"])
        print(f"\n{'═'*70}")
        if best["pf"] >= 1.5:
            print(f"  ✅ PROMETTEUR : {best['label']} — PF {best['pf']:.2f}, WR {best['win_rate']:.1f}%, PnL {best['pnl']:+.2f}$")
            print(f"     DD {best['max_dd']:.1f}% — R:R {best['rr']:.2f} — {best['trades']} trades")
        elif best["pf"] >= 1.0:
            print(f"  ⚠️  MARGINAL : {best['label']} — PF {best['pf']:.2f}, PnL {best['pnl']:+.2f}$, {best['trades']} trades")
        else:
            print(f"  ❌ NON RENTABLE : Meilleur PF = {best['pf']:.2f} ({best['label']}, {best['trades']} trades)")
        print(f"{'═'*70}\n")
    else:
        print(f"\n{'═'*70}")
        print(f"  ⚠️  PAS ASSEZ DE TRADES (< 10) sur toutes les configs")
        print(f"{'═'*70}\n")


if __name__ == "__main__":
    main()
