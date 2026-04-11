#!/usr/bin/env python3
"""
Backtest — 3 Stratégies Classiques (Spot)

┌─────────────────────────────────────────────────────────────────────────────┐
│ STRAT 1 — Trend Following (15m)                                             │
│   EMA50 > EMA200 + prix > EMA50 + RSI 50–65                                │
│   Pullback vers EMA50 + bougie de rejet haussière                           │
│   Exit : trailing stop ou TP fixe, SL -1.5%                                │
├─────────────────────────────────────────────────────────────────────────────┤
│ STRAT 2 — Mean Reversion (15m)                                              │
│   Marché en range ( |EMA50-EMA200|/EMA200 < seuil )                         │
│   RSI < 30 → BUY — TP = retour vers EMA50, SL = -1%                        │
├─────────────────────────────────────────────────────────────────────────────┤
│ STRAT 3 — Breakout Volatility (5m)                                          │
│   ATR compressé + spike volume + cassure résistance                         │
│   TP = +1%, SL = -0.6%                                                      │
└─────────────────────────────────────────────────────────────────────────────┘

Paires  : BTC-USD, ETH-USD, SOL-USD, BNB-USD, XRP-USD (Big5)
Capital : $1 000 | Frais : 0.10% entrée + 0.10% sortie

Usage :
    python3 -m backtest.run_backtest_3strats
    python3 -m backtest.run_backtest_3strats --balance 1000 --years 3
    python3 -m backtest.run_backtest_3strats --start 2022-01-01 --end 2025-01-01
    python3 -m backtest.run_backtest_3strats --strat all    # toutes (défaut)
    python3 -m backtest.run_backtest_3strats --strat tf     # Trend Following seul
    python3 -m backtest.run_backtest_3strats --strat mr     # Mean Reversion seul
    python3 -m backtest.run_backtest_3strats --strat brk    # Breakout seul
"""

from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

from backtest.data_loader import download_candles
from src.core.models import Candle

logging.basicConfig(level=logging.WARNING, format="[%(asctime)s] %(message)s")
logger = logging.getLogger(__name__)

PAIRS_BIG5 = ["BTC-USD", "ETH-USD", "SOL-USD", "BNB-USD", "XRP-USD"]


# ══════════════════════════════════════════════════════════════════════════════
# Indicateurs techniques
# ══════════════════════════════════════════════════════════════════════════════

def _ema(closes: list[float], period: int) -> list[float]:
    """EMA classique, seed sur SMA(period)."""
    result = [0.0] * len(closes)
    if len(closes) < period:
        return result
    result[period - 1] = sum(closes[:period]) / period
    k = 2.0 / (period + 1)
    for i in range(period, len(closes)):
        result[i] = closes[i] * k + result[i - 1] * (1.0 - k)
    return result


def _rsi(closes: list[float], period: int = 14) -> list[float]:
    """RSI Wilder."""
    result = [0.0] * len(closes)
    if len(closes) <= period:
        return result
    gains, losses = [], []
    for i in range(1, period + 1):
        d = closes[i] - closes[i - 1]
        gains.append(max(d, 0.0))
        losses.append(max(-d, 0.0))
    avg_gain = sum(gains) / period
    avg_loss = sum(losses) / period
    result[period] = (100.0 - 100.0 / (1.0 + avg_gain / avg_loss)) if avg_loss else 100.0
    for i in range(period + 1, len(closes)):
        d = closes[i] - closes[i - 1]
        avg_gain = (avg_gain * (period - 1) + max(d, 0.0)) / period
        avg_loss = (avg_loss * (period - 1) + max(-d, 0.0)) / period
        result[i] = (100.0 - 100.0 / (1.0 + avg_gain / avg_loss)) if avg_loss else 100.0
    return result


def _atr(candles: list[Candle], period: int = 14) -> list[float]:
    """ATR Wilder."""
    result = [0.0] * len(candles)
    if len(candles) < 2:
        return result
    trs = [0.0]
    for i in range(1, len(candles)):
        h, l, pc = candles[i].high, candles[i].low, candles[i - 1].close
        trs.append(max(h - l, abs(h - pc), abs(l - pc)))
    if len(trs) < period + 1:
        return result
    atr_v = sum(trs[1: period + 1]) / period
    result[period] = atr_v
    for i in range(period + 1, len(candles)):
        atr_v = (atr_v * (period - 1) + trs[i]) / period
        result[i] = atr_v
    return result


def _sma(values: list[float], period: int) -> list[float]:
    """SMA simple."""
    result = [0.0] * len(values)
    for i in range(period - 1, len(values)):
        result[i] = sum(values[i - period + 1: i + 1]) / period
    return result


# ══════════════════════════════════════════════════════════════════════════════
# Dataclass résultat de trade
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class TradeResult:
    entry_price: float
    exit_price: float
    pnl_pct: float      # net (frais inclus) en fraction de la mise
    pnl_abs: float      # en dollars
    is_win: bool
    exit_reason: str = ""


# ══════════════════════════════════════════════════════════════════════════════
# Configs
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class TFConfig:
    """Trend Following."""
    name: str
    ema_fast: int = 50
    ema_slow: int = 200
    rsi_min: float = 50.0
    rsi_max: float = 65.0
    pullback_pct: float = 0.012   # prix doit descendre à < EMA50 × (1 + pullback_pct)
    pullback_bars: int = 4         # fenêtre de vérification du pullback
    sl_pct: float = 0.015          # SL fixe  -1.5%
    trail_pct: float = 0.015       # trailing 1.5% sous le peak
    tp_fixed: float = 0.0          # 0 = trailing, >0 = TP fixe
    alloc_pct: float = 0.25        # fraction du capital par trade
    max_pos: int = 2
    entry_fee: float = 0.001
    exit_fee: float = 0.001
    cooldown: int = 5
    # ── Filtres régime v2 ──
    slope_bars: int = 5             # fenêtre pour calculer la pente EMA50
    slope_min_pct: float = 0.0      # EMA50 doit monter de X% sur slope_bars, 0=désactivé
    atr_min_ratio: float = 0.0      # ATR > ATR_MA × ratio pour entrer, 0=désactivé
    rsi_rising: bool = False        # RSI doit monter sur 2 bars consécutifs
    # ── Pyramiding v2 ──
    pyramid_enabled: bool = False
    pyramid_rsi_min: float = 55.0
    pyramid_alloc_pct: float = 0.15
    pyramid_max: int = 1


@dataclass
class MRConfig:
    """Mean Reversion."""
    name: str
    ema_fast: int = 50
    ema_slow: int = 200
    range_thr: float = 0.020       # |EMA50-EMA200|/EMA200 < seuil → range
    rsi_oversold: float = 30.0
    sl_pct: float = 0.010
    tp_pct: float = 0.020          # TP fixe fallback
    tp_use_ema: bool = True        # TP au retour sur EMA50
    alloc_pct: float = 0.25
    max_pos: int = 2
    entry_fee: float = 0.001
    exit_fee: float = 0.001
    cooldown: int = 5


@dataclass
class BRKConfig:
    """Breakout Volatility."""
    name: str
    atr_period: int = 14
    atr_ma_period: int = 20
    atr_comp_ratio: float = 0.70   # ATR[i-1] < ATR_MA[i-1] × ratio → compression
    resistance_bars: int = 20      # résistance = max(highs[-N:])
    vol_ma_period: int = 20
    vol_spike: float = 1.5         # volume > vol_MA × spike
    tp_pct: float = 0.010
    sl_pct: float = 0.006
    alloc_pct: float = 0.20
    max_pos: int = 3
    entry_fee: float = 0.001
    exit_fee: float = 0.001
    cooldown: int = 10


# ══════════════════════════════════════════════════════════════════════════════
# Strat 1 — Trend Following
# ══════════════════════════════════════════════════════════════════════════════

def _run_tf_pair(
    candles: list[Candle],
    cfg: TFConfig,
    initial_balance: float,
) -> tuple[float, list[TradeResult], list[float]]:
    n = len(candles)
    warmup = cfg.ema_slow + 5
    if n < warmup + 10:
        return initial_balance, [], [initial_balance] * n

    closes = [c.close for c in candles]
    highs  = [c.high  for c in candles]
    lows   = [c.low   for c in candles]
    opens  = [c.open  for c in candles]

    ema_f  = _ema(closes, cfg.ema_fast)
    ema_s  = _ema(closes, cfg.ema_slow)
    rsi_v  = _rsi(closes, 14)
    atr_v  = _atr(candles, 14) if cfg.atr_min_ratio > 0 else [0.0] * n
    atr_ma = _sma(atr_v, 20)   if cfg.atr_min_ratio > 0 else [0.0] * n

    balance = initial_balance
    equity  = [initial_balance] * n
    trades: list[TradeResult] = []

    # Chaque position : {entry, size, cost, peak, sl, tp, pyramided}
    positions: list[dict] = []
    cooldown = 0

    for i in range(warmup, n):
        price = closes[i]
        ef = ema_f[i]
        es = ema_s[i]
        rsi = rsi_v[i]

        if cooldown > 0:
            cooldown -= 1

        # ── Gestion positions ouvertes ───────────────────────────────────
        still_open: list[dict] = []
        for pos in positions:
            # Update trailing peak
            if price > pos["peak"]:
                pos["peak"] = price
            trail_stop = pos["peak"] * (1.0 - cfg.trail_pct)

            reason = ""
            exit_p = price
            if price <= pos["sl"]:
                reason = "SL"
                exit_p = pos["sl"]
            elif cfg.tp_fixed > 0 and price >= pos["tp"]:
                reason = "TP"
                exit_p = pos["tp"]
            elif cfg.tp_fixed == 0 and price < trail_stop and price > pos["entry"]:
                reason = "TRAIL"
                exit_p = trail_stop
            elif ef < es:
                # Tendance cassée → on sort
                reason = "TREND_BREAK"
                exit_p = price

            if reason:
                net = pos["size"] * exit_p * (1.0 - cfg.exit_fee)
                pnl = net - pos["cost"]
                balance += net
                trades.append(TradeResult(pos["entry"], exit_p, pnl / pos["cost"], pnl, pnl > 0, reason))
                cooldown = cfg.cooldown
            else:
                still_open.append(pos)

        positions = still_open

        # ── Filtres régime v2 ────────────────────────────────────────────
        slope_ok = True
        if cfg.slope_min_pct > 0 and i >= cfg.slope_bars:
            ref = ema_f[i - cfg.slope_bars]
            slope_ok = ref > 0 and (ema_f[i] - ref) / ref >= cfg.slope_min_pct

        atr_ok = True
        if cfg.atr_min_ratio > 0 and atr_ma[i] > 0:
            atr_ok = atr_v[i] >= atr_ma[i] * cfg.atr_min_ratio

        rsi_up_ok = True
        if cfg.rsi_rising and i >= 2:
            rsi_up_ok = rsi_v[i] > rsi_v[i - 1] > rsi_v[i - 2]

        # ── Conditions d'entrée ──────────────────────────────────────────
        if (
            cooldown == 0
            and len(positions) < cfg.max_pos
            and ef > es                  # uptrend
            and price > ef               # prix au-dessus EMA50
            and cfg.rsi_min <= rsi <= cfg.rsi_max
            and slope_ok
            and atr_ok
            and rsi_up_ok
        ):
            # Pullback : l'un des N dernières bougies a son low ≤ EMA50 × (1 + pullback_pct)
            start_pb = max(warmup, i - cfg.pullback_bars)
            touched = any(
                lows[j] <= ema_f[j] * (1.0 + cfg.pullback_pct)
                for j in range(start_pb, i)
                if ema_f[j] > 0
            )
            # Bougie de rejet haussière
            bullish = closes[i] > opens[i] and closes[i] > closes[i - 1]

            if touched and bullish:
                cost = balance * cfg.alloc_pct
                if cost <= 0:
                    equity[i] = balance + sum(p["size"] * price for p in positions)
                    continue
                fee_in = cost * cfg.entry_fee
                size = (cost - fee_in) / price
                balance -= cost
                tp_price = price * (1.0 + cfg.tp_fixed) if cfg.tp_fixed > 0 else 0.0
                positions.append({
                    "entry":     price,
                    "size":      size,
                    "cost":      cost,
                    "peak":      price,
                    "sl":        price * (1.0 - cfg.sl_pct),
                    "tp":        tp_price,
                    "pyramided": False,
                })

        # ── Pyramiding ───────────────────────────────────────────────────
        if cfg.pyramid_enabled:
            for pos in positions:
                if (
                    not pos["pyramided"]
                    and rsi > cfg.pyramid_rsi_min
                    and price > pos["entry"] * 1.005   # au moins +0.5% en profit
                    and ef > es
                    and slope_ok
                    and balance > 0
                ):
                    extra_cost = balance * cfg.pyramid_alloc_pct
                    if extra_cost > 1.0:
                        fee_in = extra_cost * cfg.entry_fee
                        extra_size = (extra_cost - fee_in) / price
                        total_cost = pos["cost"] + extra_cost
                        total_size = pos["size"] + extra_size
                        pos["entry"]     = total_cost / total_size
                        pos["size"]      = total_size
                        pos["cost"]      = total_cost
                        pos["pyramided"] = True
                        balance         -= extra_cost

        equity[i] = balance + sum(p["size"] * price for p in positions)

    # Clôture forcée fin de backtest
    if candles:
        last_price = candles[-1].close
        for pos in positions:
            net = pos["size"] * last_price * (1.0 - cfg.exit_fee)
            pnl = net - pos["cost"]
            balance += net
            trades.append(TradeResult(pos["entry"], last_price, pnl / pos["cost"], pnl, pnl > 0, "END"))

    return balance, trades, equity


# ══════════════════════════════════════════════════════════════════════════════
# Strat 2 — Mean Reversion
# ══════════════════════════════════════════════════════════════════════════════

def _run_mr_pair(
    candles: list[Candle],
    cfg: MRConfig,
    initial_balance: float,
) -> tuple[float, list[TradeResult], list[float]]:
    n = len(candles)
    warmup = cfg.ema_slow + 5
    if n < warmup + 10:
        return initial_balance, [], [initial_balance] * n

    closes = [c.close for c in candles]

    ema_f = _ema(closes, cfg.ema_fast)
    ema_s = _ema(closes, cfg.ema_slow)
    rsi_v = _rsi(closes, 14)

    balance = initial_balance
    equity  = [initial_balance] * n
    trades: list[TradeResult] = []
    positions: list[dict] = []
    cooldown = 0

    for i in range(warmup, n):
        price = closes[i]
        ef = ema_f[i]
        es = ema_s[i]
        rsi = rsi_v[i]

        if cooldown > 0:
            cooldown -= 1

        if es == 0:
            equity[i] = balance + sum(p["size"] * price for p in positions)
            continue

        in_range = abs(ef - es) / es < cfg.range_thr

        # ── Gestion positions ────────────────────────────────────────────
        still_open: list[dict] = []
        for pos in positions:
            reason = ""
            exit_p = price

            # TP : retour sur EMA50 (et prix profitable)
            if cfg.tp_use_ema and ef > 0 and price >= ef and price > pos["entry"]:
                reason = "TP_EMA"
            elif price >= pos["entry"] * (1.0 + cfg.tp_pct):
                reason = "TP"
            elif price <= pos["sl"]:
                reason = "SL"
                exit_p = pos["sl"]
            # Si le marché devient tendanciel (EMA diverge), on coupe
            elif es > 0 and abs(ef - es) / es > cfg.range_thr * 2.0:
                reason = "TREND_BREAK"

            if reason:
                net = pos["size"] * exit_p * (1.0 - cfg.exit_fee)
                pnl = net - pos["cost"]
                balance += net
                trades.append(TradeResult(pos["entry"], exit_p, pnl / pos["cost"], pnl, pnl > 0, reason))
                cooldown = cfg.cooldown
            else:
                still_open.append(pos)

        positions = still_open

        # ── Entrée ───────────────────────────────────────────────────────
        if (
            cooldown == 0
            and len(positions) < cfg.max_pos
            and in_range
            and rsi < cfg.rsi_oversold
        ):
            cost = balance * cfg.alloc_pct
            if cost <= 0:
                equity[i] = balance + sum(p["size"] * price for p in positions)
                continue
            fee_in = cost * cfg.entry_fee
            size = (cost - fee_in) / price
            balance -= cost
            positions.append({
                "entry": price,
                "size":  size,
                "cost":  cost,
                "sl":    price * (1.0 - cfg.sl_pct),
            })

        equity[i] = balance + sum(p["size"] * price for p in positions)

    if candles:
        last_price = candles[-1].close
        for pos in positions:
            net = pos["size"] * last_price * (1.0 - cfg.exit_fee)
            pnl = net - pos["cost"]
            balance += net
            trades.append(TradeResult(pos["entry"], last_price, pnl / pos["cost"], pnl, pnl > 0, "END"))

    return balance, trades, equity


# ══════════════════════════════════════════════════════════════════════════════
# Strat 3 — Breakout Volatility
# ══════════════════════════════════════════════════════════════════════════════

def _run_brk_pair(
    candles: list[Candle],
    cfg: BRKConfig,
    initial_balance: float,
) -> tuple[float, list[TradeResult], list[float]]:
    n = len(candles)
    warmup = max(cfg.atr_ma_period, cfg.resistance_bars, cfg.vol_ma_period) + 5
    if n < warmup + 10:
        return initial_balance, [], [initial_balance] * n

    closes  = [c.close  for c in candles]
    highs   = [c.high   for c in candles]
    volumes = [c.volume for c in candles]

    atr_v  = _atr(candles, cfg.atr_period)
    atr_ma = _sma(atr_v, cfg.atr_ma_period)
    vol_ma = _sma(volumes, cfg.vol_ma_period)

    balance = initial_balance
    equity  = [initial_balance] * n
    trades: list[TradeResult] = []
    positions: list[dict] = []
    cooldown = 0

    for i in range(warmup, n):
        price    = closes[i]
        vol      = volumes[i]
        prev_atr = atr_v[i - 1]
        prev_ama = atr_ma[i - 1]
        vma      = vol_ma[i]

        if cooldown > 0:
            cooldown -= 1

        # ── Gestion positions ────────────────────────────────────────────
        still_open: list[dict] = []
        for pos in positions:
            reason = ""
            exit_p = price
            if price >= pos["tp"]:
                reason = "TP"
                exit_p = pos["tp"]
            elif price <= pos["sl"]:
                reason = "SL"
                exit_p = pos["sl"]

            if reason:
                net = pos["size"] * exit_p * (1.0 - cfg.exit_fee)
                pnl = net - pos["cost"]
                balance += net
                trades.append(TradeResult(pos["entry"], exit_p, pnl / pos["cost"], pnl, pnl > 0, reason))
                cooldown = cfg.cooldown
            else:
                still_open.append(pos)

        positions = still_open

        # ── Entrée ───────────────────────────────────────────────────────
        if prev_ama == 0 or vma == 0:
            equity[i] = balance + sum(p["size"] * price for p in positions)
            continue

        # Compression : ATR était bas avant la bougie courante
        compressed = prev_atr < prev_ama * cfg.atr_comp_ratio
        # Volume spike sur la bougie actuelle
        vol_ok = vol > vma * cfg.vol_spike
        # Cassure de la résistance locale (max des N bougies précédentes)
        resistance = max(highs[max(0, i - cfg.resistance_bars): i])
        breakout = price > resistance

        if (
            cooldown == 0
            and len(positions) < cfg.max_pos
            and compressed
            and vol_ok
            and breakout
        ):
            cost = balance * cfg.alloc_pct
            if cost <= 0:
                equity[i] = balance + sum(p["size"] * price for p in positions)
                continue
            fee_in = cost * cfg.entry_fee
            size = (cost - fee_in) / price
            balance -= cost
            positions.append({
                "entry": price,
                "size":  size,
                "cost":  cost,
                "tp":    price * (1.0 + cfg.tp_pct),
                "sl":    price * (1.0 - cfg.sl_pct),
            })

        equity[i] = balance + sum(p["size"] * price for p in positions)

    if candles:
        last_price = candles[-1].close
        for pos in positions:
            net = pos["size"] * last_price * (1.0 - cfg.exit_fee)
            pnl = net - pos["cost"]
            balance += net
            trades.append(TradeResult(pos["entry"], last_price, pnl / pos["cost"], pnl, pnl > 0, "END"))

    return balance, trades, equity


# ══════════════════════════════════════════════════════════════════════════════
# Métriques
# ══════════════════════════════════════════════════════════════════════════════

def _compute_metrics(
    trades: list[TradeResult],
    equity: list[float],
    initial: float,
) -> dict:
    n = len(trades)
    if n == 0:
        return {"n": 0, "wr": 0.0, "pf": 0.0, "final": initial, "dd": 0.0}

    wins   = [t for t in trades if t.is_win]
    losses = [t for t in trades if not t.is_win]
    gp = sum(t.pnl_abs for t in wins)
    gl = abs(sum(t.pnl_abs for t in losses)) or 1e-9
    pf = gp / gl

    peak = dd = 0.0
    for v in equity:
        if v > peak:
            peak = v
        if peak > 0:
            dd = min(dd, (v - peak) / peak)

    return {
        "n":     n,
        "wr":    len(wins) / n,
        "pf":    pf,
        "final": equity[-1] if equity else initial,
        "dd":    dd,
    }


# ══════════════════════════════════════════════════════════════════════════════
# Runners (download + agrégation multi-paires)
# ══════════════════════════════════════════════════════════════════════════════

def _run_tf(
    cfgs: list[TFConfig],
    candles_by_pair: dict[str, list[Candle]],
    initial: float,
    pairs: list[str],
) -> list[tuple[TFConfig, dict]]:
    results = []
    for cfg in cfgs:
        allT: list[TradeResult] = []
        allE: list[float] = []
        per_pair = initial / len(pairs)
        for pair in pairs:
            cds = candles_by_pair.get(pair, [])
            if not cds:
                continue
            bal, trades, eq = _run_tf_pair(cds, cfg, per_pair)
            allT.extend(trades)
            if not allE:
                allE = list(eq)
            else:
                allE = [a + b for a, b in zip(allE, eq)]
        m = _compute_metrics(allT, allE, initial)
        # Recalcul final_balance réel
        m["final"] = sum(
            _run_tf_pair(candles_by_pair[p], cfg, initial / len(pairs))[0]
            for p in pairs if candles_by_pair.get(p)
        )
        results.append((cfg, m))
    return results


def _run_mr(
    cfgs: list[MRConfig],
    candles_by_pair: dict[str, list[Candle]],
    initial: float,
    pairs: list[str],
) -> list[tuple[MRConfig, dict]]:
    results = []
    for cfg in cfgs:
        allT: list[TradeResult] = []
        allE: list[float] = []
        per_pair = initial / len(pairs)
        for pair in pairs:
            cds = candles_by_pair.get(pair, [])
            if not cds:
                continue
            bal, trades, eq = _run_mr_pair(cds, cfg, per_pair)
            allT.extend(trades)
            if not allE:
                allE = list(eq)
            else:
                allE = [a + b for a, b in zip(allE, eq)]
        m = _compute_metrics(allT, allE, initial)
        m["final"] = sum(
            _run_mr_pair(candles_by_pair[p], cfg, initial / len(pairs))[0]
            for p in pairs if candles_by_pair.get(p)
        )
        results.append((cfg, m))
    return results


def _run_brk(
    cfgs: list[BRKConfig],
    candles_by_pair: dict[str, list[Candle]],
    initial: float,
    pairs: list[str],
) -> list[tuple[BRKConfig, dict]]:
    results = []
    for cfg in cfgs:
        allT: list[TradeResult] = []
        allE: list[float] = []
        per_pair = initial / len(pairs)
        for pair in pairs:
            cds = candles_by_pair.get(pair, [])
            if not cds:
                continue
            bal, trades, eq = _run_brk_pair(cds, cfg, per_pair)
            allT.extend(trades)
            if not allE:
                allE = list(eq)
            else:
                allE = [a + b for a, b in zip(allE, eq)]
        m = _compute_metrics(allT, allE, initial)
        m["final"] = sum(
            _run_brk_pair(candles_by_pair[p], cfg, initial / len(pairs))[0]
            for p in pairs if candles_by_pair.get(p)
        )
        results.append((cfg, m))
    return results


# ══════════════════════════════════════════════════════════════════════════════
# Affichage
# ══════════════════════════════════════════════════════════════════════════════

_W = 86


def _header(title: str, subtitle: str) -> None:
    print("\n" + "═" * _W)
    print(f"  {title}")
    print(f"  {subtitle}")
    print("═" * _W)


def _table_row(name: str, m: dict, initial: float, extra: str = "") -> None:
    pnl = m["final"] - initial
    sign = "+" if pnl >= 0 else ""
    pf_str = f"{m['pf']:.2f}" if m["pf"] != float("inf") else "  ∞  "
    print(
        f"  {name:28s} | {m['wr']:>6.1%} | {pf_str:>7s} | {sign}{pnl:>9.2f}$ | {m['dd']:>7.1%} | {m['n']:>6d}  {extra}"
    )


def _table_header() -> None:
    h = f"  {'Config':28s} | {'  WR':>6s} | {'   PF':>7s} | {'  PnL ($)':>11s} | {'  DD max':>7s} | {'Trades':>6s}"
    print(h)
    print("  " + "─" * (_W - 2))


def _print_tf(results: list[tuple[TFConfig, dict]], initial: float) -> None:
    _header(
        "STRAT 1 — TREND FOLLOWING v2 (EMA50/200 + filtres régime + pyramiding)",
        f"Timeframe : 15m | Capital : ${initial:,.0f} | Frais : 0.10%+0.10% | Big5",
    )
    _table_header()
    for cfg, m in sorted(results, key=lambda x: x[1]["pf"], reverse=True):
        trail_str = f"trail {cfg.trail_pct:.0%}" if cfg.tp_fixed == 0 else f"TP {cfg.tp_fixed:.0%}"
        parts = [f"RSI {cfg.rsi_min:.0f}–{cfg.rsi_max:.0f}", f"SL {cfg.sl_pct:.1%}", trail_str, f"alloc {cfg.alloc_pct:.0%}"]
        if cfg.slope_min_pct > 0:
            parts.append(f"pente≥{cfg.slope_min_pct*100:.2f}%")
        if cfg.atr_min_ratio > 0:
            parts.append(f"atr≥{cfg.atr_min_ratio:.0%}MA")
        if cfg.rsi_rising:
            parts.append("rsi↑")
        if cfg.pyramid_enabled:
            parts.append("pyramid")
        extra = "[" + " | ".join(parts) + "]"
        _table_row(cfg.name, m, initial, extra)

    # ── Groupes ──────────────────────────────────────────────────────────
    by_name = {cfg.name: (cfg, m) for cfg, m in results}
    groups = [
        ("G1 — Référence",    ["TF_BASE", "TF_STRICT", "TF_WIDE_RSI", "TF_R40", "TF_TP3"]),
        ("G2 — Pente EMA50",  ["TF_SLOPE_02", "TF_SLOPE_05", "TF_SLOPE_10"]),
        ("G3 — Filtre ATR",   ["TF_ATR_70", "TF_ATR_85", "TF_ATR_100"]),
        ("G4 — RSI montant",  ["TF_RSI_UP"]),
        ("G5 — Combo",        ["TF_COMBO", "TF_COMBO_R40", "TF_COMBO_WIDE"]),
        ("G6 — Pyramiding",   ["TF_PYRAMID", "TF_COMBO_PYRAMID"]),
        ("G7 — SLOPE_10 × Pyramid × ATR", [
            "TF_S10_PYR", "TF_S10_ATR100", "TF_S10_ATR100_PYR",
            "TF_S10_RSI_PYR", "TF_S10_FULL", "TF_S10_FULL_PYR",
            "TF_S15_FULL", "TF_S15_FULL_PYR", "TF_S20_FULL", "TF_S20_FULL_PYR",
        ]),
    ]
    for grp_title, names in groups:
        found = [(n, by_name[n]) for n in names if n in by_name]
        if not found:
            continue
        print(f"\n  {grp_title} :")
        for name, (cfg, m) in found:
            pnl = m["final"] - initial
            pf_str = f"{m['pf']:.2f}" if m["pf"] != float("inf") else "  ∞"
            sign = "+" if pnl >= 0 else ""
            print(f"    {name:22s}  PF {pf_str:>6s}  WR {m['wr']:.1%}  PnL ${sign}{pnl:.2f}  DD {m['dd']:.1%}  [{m['n']}t]")

    best_pf  = max(results, key=lambda x: x[1]["pf"])
    best_pnl = max(results, key=lambda x: x[1]["final"])
    print("\n  " + "─" * (_W - 2))
    print(f"  ★ Meilleur PF  : {best_pf[0].name:22s}  PF {best_pf[1]['pf']:.2f}  |  PnL ${best_pf[1]['final']-initial:+.2f}  |  DD {best_pf[1]['dd']:.1%}")
    print(f"  ★ Meilleur PnL : {best_pnl[0].name:22s}  PF {best_pnl[1]['pf']:.2f}  |  PnL ${best_pnl[1]['final']-initial:+.2f}  |  DD {best_pnl[1]['dd']:.1%}")


def _print_mr(results: list[tuple[MRConfig, dict]], initial: float) -> None:
    _header(
        "STRAT 2 — MEAN REVERSION (range EMA + RSI < seuil → achat → EMA50)",
        f"Timeframe : 15m | Capital : ${initial:,.0f} | Frais : 0.10%+0.10% | Big5",
    )
    _table_header()
    for cfg, m in sorted(results, key=lambda x: x[1]["pf"], reverse=True):
        extra = f"[range ≤{cfg.range_thr:.0%} | RSI<{cfg.rsi_oversold:.0f} | SL {cfg.sl_pct:.1%} | alloc {cfg.alloc_pct:.0%}]"
        _table_row(cfg.name, m, initial, extra)

    best_pf  = max(results, key=lambda x: x[1]["pf"])
    best_pnl = max(results, key=lambda x: x[1]["final"])
    print("  " + "─" * (_W - 2))
    print(f"  ★ Meilleur PF  : {best_pf[0].name:20s}  PF {best_pf[1]['pf']:.2f}  |  PnL ${best_pf[1]['final']-initial:+.2f}  |  DD {best_pf[1]['dd']:.1%}")
    print(f"  ★ Meilleur PnL : {best_pnl[0].name:20s}  PF {best_pnl[1]['pf']:.2f}  |  PnL ${best_pnl[1]['final']-initial:+.2f}  |  DD {best_pnl[1]['dd']:.1%}")


def _print_brk(results: list[tuple[BRKConfig, dict]], initial: float) -> None:
    _header(
        "STRAT 3 — BREAKOUT VOLATILITY (compression ATR + spike vol + cassure)",
        f"Timeframe : 5m | Capital : ${initial:,.0f} | Frais : 0.10%+0.10% | Big5",
    )
    _table_header()
    for cfg, m in sorted(results, key=lambda x: x[1]["pf"], reverse=True):
        extra = f"[comp {cfg.atr_comp_ratio:.2f} | vol×{cfg.vol_spike:.1f} | TP {cfg.tp_pct:.1%} / SL {cfg.sl_pct:.1%} | alloc {cfg.alloc_pct:.0%}]"
        _table_row(cfg.name, m, initial, extra)

    best_pf  = max(results, key=lambda x: x[1]["pf"])
    best_pnl = max(results, key=lambda x: x[1]["final"])
    print("  " + "─" * (_W - 2))
    print(f"  ★ Meilleur PF  : {best_pf[0].name:20s}  PF {best_pf[1]['pf']:.2f}  |  PnL ${best_pf[1]['final']-initial:+.2f}  |  DD {best_pf[1]['dd']:.1%}")
    print(f"  ★ Meilleur PnL : {best_pnl[0].name:20s}  PF {best_pnl[1]['pf']:.2f}  |  PnL ${best_pnl[1]['final']-initial:+.2f}  |  DD {best_pnl[1]['dd']:.1%}")


def _print_summary(
    tf_res: list[tuple] | None,
    mr_res: list[tuple] | None,
    brk_res: list[tuple] | None,
    initial: float,
) -> None:
    print("\n" + "═" * _W)
    print("  RÉSUMÉ COMPARATIF — Meilleure variante par stratégie")
    print("═" * _W)
    _table_header()
    if tf_res:
        best = max(tf_res, key=lambda x: x[1]["pf"])
        _table_row(f"TF  ▸ {best[0].name}", best[1], initial)
    if mr_res:
        best = max(mr_res, key=lambda x: x[1]["pf"])
        _table_row(f"MR  ▸ {best[0].name}", best[1], initial)
    if brk_res:
        best = max(brk_res, key=lambda x: x[1]["pf"])
        _table_row(f"BRK ▸ {best[0].name}", best[1], initial)
    print("═" * _W)
    print()


# ══════════════════════════════════════════════════════════════════════════════
# Variantes
# ══════════════════════════════════════════════════════════════════════════════

def _tf_variants() -> list[TFConfig]:
    _b = dict(rsi_min=50, rsi_max=65, sl_pct=0.015, trail_pct=0.015, tp_fixed=0.0, alloc_pct=0.25)

    # ══ G1 — Référence ══
    g1 = [
        TFConfig(name="TF_BASE",     **_b),
        TFConfig(name="TF_STRICT",   **{**_b, "rsi_max": 60, "pullback_pct": 0.008}),
        TFConfig(name="TF_WIDE_RSI", **{**_b, "rsi_min": 45, "rsi_max": 70, "trail_pct": 0.020}),
        TFConfig(name="TF_R40",      **{**_b, "alloc_pct": 0.40}),
        TFConfig(name="TF_TP3",      **{**_b, "tp_fixed": 0.030}),
    ]
    # ══ G2 — Filtre pente EMA50 (évite marchés plats / faux débuts de tendance) ══
    g2 = [
        TFConfig(name="TF_SLOPE_02",  **{**_b, "slope_min_pct": 0.0002}),
        TFConfig(name="TF_SLOPE_05",  **{**_b, "slope_min_pct": 0.0005}),
        TFConfig(name="TF_SLOPE_10",  **{**_b, "slope_min_pct": 0.0010}),
    ]
    # ══ G3 — Filtre ATR (pas de marché compressé / range étroit) ══
    g3 = [
        TFConfig(name="TF_ATR_70",   **{**_b, "atr_min_ratio": 0.70}),
        TFConfig(name="TF_ATR_85",   **{**_b, "atr_min_ratio": 0.85}),
        TFConfig(name="TF_ATR_100",  **{**_b, "atr_min_ratio": 1.00}),
    ]
    # ══ G4 — RSI montant (momentum croissant) ══
    g4 = [
        TFConfig(name="TF_RSI_UP",   **{**_b, "rsi_rising": True}),
    ]
    # ══ G5 — Combo (pente + ATR + RSI montant) ══
    g5 = [
        TFConfig(name="TF_COMBO",      **{**_b, "slope_min_pct": 0.0003, "atr_min_ratio": 0.80, "rsi_rising": True}),
        TFConfig(name="TF_COMBO_R40",  **{**_b, "slope_min_pct": 0.0003, "atr_min_ratio": 0.80, "rsi_rising": True, "alloc_pct": 0.40}),
        TFConfig(name="TF_COMBO_WIDE", **{**_b, "slope_min_pct": 0.0003, "atr_min_ratio": 0.80, "rsi_rising": True, "rsi_min": 45, "rsi_max": 70}),
    ]
    # ══ G6 — Pyramiding (renforcement sur continuations) ══
    g6 = [
        TFConfig(name="TF_PYRAMID",       **{**_b, "pyramid_enabled": True}),
        TFConfig(name="TF_COMBO_PYRAMID", **{**_b, "slope_min_pct": 0.0003, "atr_min_ratio": 0.80, "rsi_rising": True, "pyramid_enabled": True}),
    ]
    # ══ G7 — SLOPE_10 × pyramid × ATR (maximisation PF) ══
    g7 = [
        # Slope seul + pyramid
        TFConfig(name="TF_S10_PYR",          **{**_b, "slope_min_pct": 0.0010, "pyramid_enabled": True}),
        # Meilleurs deux filtres individuels combinés
        TFConfig(name="TF_S10_ATR100",       **{**_b, "slope_min_pct": 0.0010, "atr_min_ratio": 1.00}),
        TFConfig(name="TF_S10_ATR100_PYR",   **{**_b, "slope_min_pct": 0.0010, "atr_min_ratio": 1.00, "pyramid_enabled": True}),
        # Slope + RSI montant + pyramid
        TFConfig(name="TF_S10_RSI_PYR",      **{**_b, "slope_min_pct": 0.0010, "rsi_rising": True, "pyramid_enabled": True}),
        # Full stack — filtre maximum
        TFConfig(name="TF_S10_FULL",         **{**_b, "slope_min_pct": 0.0010, "atr_min_ratio": 1.00, "rsi_rising": True}),
        TFConfig(name="TF_S10_FULL_PYR",     **{**_b, "slope_min_pct": 0.0010, "atr_min_ratio": 1.00, "rsi_rising": True, "pyramid_enabled": True}),
        # Pente encore plus stricte (0.15% / 0.20%) — cherche le PF max
        TFConfig(name="TF_S15_FULL",         **{**_b, "slope_min_pct": 0.0015, "atr_min_ratio": 1.00, "rsi_rising": True}),
        TFConfig(name="TF_S15_FULL_PYR",     **{**_b, "slope_min_pct": 0.0015, "atr_min_ratio": 1.00, "rsi_rising": True, "pyramid_enabled": True}),
        TFConfig(name="TF_S20_FULL",         **{**_b, "slope_min_pct": 0.0020, "atr_min_ratio": 1.00, "rsi_rising": True}),
        TFConfig(name="TF_S20_FULL_PYR",     **{**_b, "slope_min_pct": 0.0020, "atr_min_ratio": 1.00, "rsi_rising": True, "pyramid_enabled": True}),
    ]
    return g1 + g2 + g3 + g4 + g5 + g6 + g7


def _mr_variants() -> list[MRConfig]:
    return [
        # BASE : range 2%, RSI<30, SL -1%, TP EMA50
        MRConfig(name="MR_BASE",     range_thr=0.020, rsi_oversold=30, sl_pct=0.010, alloc_pct=0.25),
        # STRICT : range 1.5%, RSI<25
        MRConfig(name="MR_STRICT",   range_thr=0.015, rsi_oversold=25, sl_pct=0.010, alloc_pct=0.25),
        # WIDE : range 3%, RSI<35
        MRConfig(name="MR_WIDE",     range_thr=0.030, rsi_oversold=35, sl_pct=0.015, alloc_pct=0.25),
        # TP fixe +2% (sans attendre EMA50)
        MRConfig(name="MR_TP2",      range_thr=0.020, rsi_oversold=30, sl_pct=0.010, alloc_pct=0.25, tp_use_ema=False, tp_pct=0.020),
        # Alloc 40%
        MRConfig(name="MR_R40",      range_thr=0.020, rsi_oversold=30, sl_pct=0.010, alloc_pct=0.40),
    ]


def _brk_variants() -> list[BRKConfig]:
    return [
        # BASE : compression 0.70, vol×1.5, TP 1%, SL 0.6%
        BRKConfig(name="BRK_BASE",   atr_comp_ratio=0.70, vol_spike=1.5, tp_pct=0.010, sl_pct=0.006, alloc_pct=0.20),
        # STRICT : compression 0.65, vol×2.0
        BRKConfig(name="BRK_STRICT", atr_comp_ratio=0.65, vol_spike=2.0, tp_pct=0.010, sl_pct=0.006, alloc_pct=0.20),
        # TP élargi 1.5%, SL 0.8%
        BRKConfig(name="BRK_WIDE",   atr_comp_ratio=0.70, vol_spike=1.5, tp_pct=0.015, sl_pct=0.008, alloc_pct=0.20),
        # Plus de résistance lookback
        BRKConfig(name="BRK_RES30",  atr_comp_ratio=0.70, vol_spike=1.5, tp_pct=0.010, sl_pct=0.006, alloc_pct=0.20, resistance_bars=30),
        # Alloc 30%
        BRKConfig(name="BRK_R30",    atr_comp_ratio=0.70, vol_spike=1.5, tp_pct=0.010, sl_pct=0.006, alloc_pct=0.30),
    ]


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    parser = argparse.ArgumentParser(description="Backtest 3 stratégies classiques")
    parser.add_argument("--balance", type=float, default=1000.0, help="Capital initial ($)")
    parser.add_argument("--years",   type=float, default=3.0,    help="Durée en années (défaut : 3)")
    parser.add_argument("--start",   type=str,   default=None,   help="Date début YYYY-MM-DD")
    parser.add_argument("--end",     type=str,   default=None,   help="Date fin YYYY-MM-DD")
    parser.add_argument("--strat",   type=str,   default="all",  choices=["all", "tf", "mr", "brk"],
                        help="Stratégie(s) à backtester (défaut : all)")
    args = parser.parse_args()

    now = datetime.now(timezone.utc)
    if args.start:
        start_dt = datetime.strptime(args.start, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    else:
        start_dt = now - timedelta(days=int(args.years * 365.25))
    end_dt = datetime.strptime(args.end, "%Y-%m-%d").replace(tzinfo=timezone.utc) if args.end else now

    pairs  = PAIRS_BIG5
    strat  = args.strat
    init   = args.balance

    run_tf_flag  = strat in ("all", "tf")
    run_mr_flag  = strat in ("all", "mr")
    run_brk_flag = strat in ("all", "brk")

    # ── Téléchargement 15m (Strat 1 + 2) ────────────────────────────────
    candles_15m: dict[str, list[Candle]] = {}
    if run_tf_flag or run_mr_flag:
        print(f"\n📥 Téléchargement 15m ({start_dt:%Y-%m-%d} → {end_dt:%Y-%m-%d})…")
        for p in pairs:
            cds = download_candles(p, start_dt, end_dt, interval="15m")
            candles_15m[p] = cds
            print(f"  ✓ {p}: {len(cds):,} bougies")

    # ── Téléchargement 5m (Strat 3) ──────────────────────────────────────
    candles_5m: dict[str, list[Candle]] = {}
    if run_brk_flag:
        print(f"\n📥 Téléchargement 5m ({start_dt:%Y-%m-%d} → {end_dt:%Y-%m-%d})…")
        for p in pairs:
            cds = download_candles(p, start_dt, end_dt, interval="5m")
            candles_5m[p] = cds
            print(f"  ✓ {p}: {len(cds):,} bougies")

    tf_results  = None
    mr_results  = None
    brk_results = None

    # ── Strat 1 ──────────────────────────────────────────────────────────
    if run_tf_flag:
        print("\n⚙️  Strat 1 — Trend Following…")
        tf_results = _run_tf(_tf_variants(), candles_15m, init, pairs)
        _print_tf(tf_results, init)

    # ── Strat 2 ──────────────────────────────────────────────────────────
    if run_mr_flag:
        print("\n⚙️  Strat 2 — Mean Reversion…")
        mr_results = _run_mr(_mr_variants(), candles_15m, init, pairs)
        _print_mr(mr_results, init)

    # ── Strat 3 ──────────────────────────────────────────────────────────
    if run_brk_flag:
        print("\n⚙️  Strat 3 — Breakout Volatility…")
        brk_results = _run_brk(_brk_variants(), candles_5m, init, pairs)
        _print_brk(brk_results, init)

    # ── Résumé comparatif ─────────────────────────────────────────────────
    if strat == "all":
        _print_summary(tf_results, mr_results, brk_results, init)


if __name__ == "__main__":
    main()
