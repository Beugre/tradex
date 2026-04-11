#!/usr/bin/env python3
"""
Backtest — Dip from Rolling High (Spot, Big5)

Strategie :
  - Signal d'entree : prix baisse de X% par rapport au high des N derniers jours
  - Exit : TP fixe OU trailing stop (activation +5%, SL verrouille a 0% = breakeven garanti)
  - Pas de SL en dessous du prix d'entree

Parametres explores :
  - Lookback : 7 jours ou 14 jours (en barres H4 : 7j = 42b, 14j = 84b)
  - Drop : -10% ou -20% du high
  - TP : +5%, +10%, +15%
  - Trailing : activation +5%, trail 3% sous le pic, SL minimum = entree (0% perte max)

Paires : Big5 (BTC, ETH, SOL, BNB, XRP) — les altcoins ont prouve leur toxicite
Capital : $1,000 | Risk : 20% par trade

Usage:
    python3 -m backtest.run_backtest_dip_from_high --compare
    python3 -m backtest.run_backtest_dip_from_high --balance 1000 --years 2
    python3 -m backtest.run_backtest_dip_from_high --years 3
"""

from __future__ import annotations

import argparse
import logging
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from backtest.data_loader import download_candles
from src.core.models import Candle

logging.basicConfig(level=logging.WARNING, format="[%(asctime)s] %(message)s")
logger = logging.getLogger(__name__)

OUTPUT_DIR = Path(__file__).parent / "output"

PAIRS_BIG5 = ["BTC-USD", "ETH-USD", "SOL-USD", "BNB-USD", "XRP-USD"]


# ── Constante ─────────────────────────────────────────────────────────────────

H4_PER_DAY = 6  # 6 bougies H4 par jour


# ── Config ─────────────────────────────────────────────────────────────────────


@dataclass
class DipHighConfig:
    """Configuration pour la strategie dip-from-rolling-high."""
    name: str = "DIP_HIGH"

    # ── Signal d'entree ──
    lookback_days: int = 14            # Fenetre du rolling high (en jours)
    drop_pct: float = 0.20             # Baisse requise depuis le high (-20%)

    # ── TP fixe ──
    tp_pct: float = 0.10               # Take profit (+10%)
    tp_enabled: bool = True

    # ── Trailing stop ──
    trailing_enabled: bool = False
    trailing_activation_pct: float = 0.05   # Active le trailing quand prix >= entree * 1.05
    trailing_distance_pct: float = 0.03     # Trail 3% sous le pic
    # SL minimum = entree (garantit 0% de perte une fois le trailing active)

    # ── Risk / Sizing ──
    risk_pct: float = 0.20             # 20% du capital par position
    max_simultaneous: int = 5          # max 5 positions simultanees (1 par paire)

    # ── Cooldown ──
    cooldown_bars: int = 6             # 1 jour H4 entre deux entrees sur la meme paire

    # ── Hard Stop-Loss ──
    sl_pct: float = 0.0                # 0 = desactive | ex: 0.30 = SL a -30% du prix d'entree

    # ── Regime filter (BTC MA200) ──
    regime_filter_enabled: bool = False  # True = entrees bloquees si BTC < MA200
    ma_period_days: int = 200            # Periode de la MA (en jours)

    # ── Max duration ──
    max_bars_in_trade: int = 0         # 0 = illimite

    # ── Fees ──
    entry_fee_pct: float = 0.001       # 0.1% (Binance taker, conservative)
    exit_fee_pct: float = 0.001


# ── Structures ─────────────────────────────────────────────────────────────────


@dataclass
class DipHighTrade:
    symbol: str
    entry_bar: int
    entry_price: float
    entry_ts: int
    tp_price: float
    size: float
    high_at_entry: float = 0.0         # Rolling high a l'entree
    drop_pct_at_entry: float = 0.0     # % de baisse effectif
    exit_bar: int = 0
    exit_price: float = 0.0
    exit_ts: int = 0
    exit_reason: str = ""
    pnl_usd: float = 0.0
    pnl_pct: float = 0.0
    fees: float = 0.0
    duration_bars: int = 0
    # Trailing state
    trailing_active: bool = False
    max_price_seen: float = 0.0


@dataclass
class EquityPoint:
    ts: int
    equity: float


# ── Indicateurs ────────────────────────────────────────────────────────────────


def rolling_high_series(highs: list[float], period: int) -> list[float]:
    """High glissant sur N barres (inclut la barre courante)."""
    n = len(highs)
    result = [0.0] * n
    for i in range(n):
        start = max(0, i - period + 1)
        result[i] = max(highs[start: i + 1])
    return result


# ── Simulation ─────────────────────────────────────────────────────────────────


def _ma_series(closes: list[float], period: int) -> list[float]:
    """SMA sur N barres (NaN = 0.0 tant que pas assez de donnees)."""
    result = [0.0] * len(closes)
    for i in range(period - 1, len(closes)):
        result[i] = sum(closes[i - period + 1: i + 1]) / period
    return result


def run_pair(
    candles: list[Candle],
    cfg: DipHighConfig,
    balance: float,
    symbol: str,
    regime_mask: list[bool] | None = None,
) -> tuple[list[DipHighTrade], list[EquityPoint], float]:
    """Simule la strategie dip-from-high sur une paire.

    regime_mask : liste de bool, meme longueur que candles. True = entree autorisee.
                  Si None, toutes les entrees sont autorisees.
    """
    n = len(candles)
    lookback_bars = cfg.lookback_days * H4_PER_DAY
    if n < lookback_bars + 2:
        return [], [], balance

    highs = [c.high for c in candles]
    roll_high = rolling_high_series(highs, lookback_bars)

    trades: list[DipHighTrade] = []
    equity_curve: list[EquityPoint] = []
    open_trades: list[DipHighTrade] = []
    cooldown_until: int = 0

    for i in range(lookback_bars, n):
        c = candles[i]

        # ── Gestion sorties ──
        new_open: list[DipHighTrade] = []
        for t in open_trades:
            closed = False

            # Mise a jour du pic (trailing)
            if cfg.trailing_enabled:
                t.max_price_seen = max(t.max_price_seen, c.high)

            # ── TP fixe ──
            if cfg.tp_enabled and c.high >= t.tp_price:
                t.exit_price = t.tp_price
                t.exit_reason = "TP"
                closed = True

            # ── Trailing stop ──
            elif cfg.trailing_enabled and t.max_price_seen >= t.entry_price * (1 + cfg.trailing_activation_pct):
                t.trailing_active = True
                # SL = max(prix_entree, pic * (1 - trail_dist)) → jamais en perte
                trail_stop = max(
                    t.entry_price,  # garantit 0% de perte (breakeven minimum)
                    t.max_price_seen * (1 - cfg.trailing_distance_pct),
                )
                if c.low <= trail_stop:
                    t.exit_price = trail_stop
                    t.exit_reason = "TRAIL"
                    closed = True

            # ── Hard Stop-Loss ──
            elif cfg.sl_pct > 0 and c.low <= t.entry_price * (1 - cfg.sl_pct):
                t.exit_price = t.entry_price * (1 - cfg.sl_pct)
                t.exit_reason = "SL"
                closed = True

            # ── Max duration ──
            elif cfg.max_bars_in_trade > 0 and (i - t.entry_bar) >= cfg.max_bars_in_trade:
                t.exit_price = c.close
                t.exit_reason = "TIMEOUT"
                closed = True

            if closed:
                t.exit_bar = i
                t.exit_ts = c.timestamp
                exit_fee = t.size * t.exit_price * cfg.exit_fee_pct
                entry_fee = t.size * t.entry_price * cfg.entry_fee_pct
                t.fees = entry_fee + exit_fee
                t.pnl_usd = (t.exit_price - t.entry_price) * t.size - t.fees
                t.pnl_pct = (t.exit_price - t.entry_price) / t.entry_price if t.entry_price > 0 else 0
                t.duration_bars = t.exit_bar - t.entry_bar
                balance += t.pnl_usd + (t.size * t.entry_price)
                trades.append(t)
            else:
                new_open.append(t)

        open_trades = new_open

        # ── Signal d'entree ──
        regime_ok = (regime_mask is None) or (i < len(regime_mask) and regime_mask[i])
        if regime_ok and i > cooldown_until and len(open_trades) < cfg.max_simultaneous:
            if not any(t.symbol == symbol for t in open_trades):
                ref_high = roll_high[i - 1]  # high des N barres precedentes (exclu barre courante)
                if ref_high > 0:
                    actual_drop = (ref_high - c.close) / ref_high
                    if actual_drop >= cfg.drop_pct:
                        alloc = balance * cfg.risk_pct
                        if alloc >= 1.0 and c.close > 0:
                            size = alloc / c.close
                            tp_price = c.close * (1 + cfg.tp_pct) if cfg.tp_enabled else float("inf")

                            trade = DipHighTrade(
                                symbol=symbol,
                                entry_bar=i,
                                entry_price=c.close,
                                entry_ts=c.timestamp,
                                tp_price=tp_price,
                                size=size,
                                high_at_entry=ref_high,
                                drop_pct_at_entry=actual_drop,
                                max_price_seen=c.close,
                            )
                            balance -= alloc
                            open_trades.append(trade)
                            cooldown_until = i + cfg.cooldown_bars

        equity_curve.append(EquityPoint(c.timestamp, balance + _unrealized(open_trades, c.close)))

    # Force-close les positions ouvertes a la fin
    if open_trades:
        last = candles[-1]
        for t in open_trades:
            t.exit_bar = n - 1
            t.exit_price = last.close
            t.exit_ts = last.timestamp
            t.exit_reason = "END"
            exit_fee = t.size * t.exit_price * cfg.exit_fee_pct
            entry_fee = t.size * t.entry_price * cfg.entry_fee_pct
            t.fees = exit_fee + entry_fee
            t.pnl_usd = (t.exit_price - t.entry_price) * t.size - t.fees
            t.pnl_pct = (t.exit_price - t.entry_price) / t.entry_price if t.entry_price > 0 else 0
            t.duration_bars = t.exit_bar - t.entry_bar
            balance += t.pnl_usd + (t.size * t.entry_price)
            trades.append(t)

    return trades, equity_curve, balance


def _unrealized(open_trades: list[DipHighTrade], current_price: float) -> float:
    return sum((current_price - t.entry_price) * t.size for t in open_trades)


# ── Variantes ─────────────────────────────────────────────────────────────────


def get_variants() -> list[DipHighConfig]:
    """19 variantes organisees en 5 groupes."""
    return [
        # ══ G1 — Lookback 14j, Drop -20% ══════════════════════════════════════
        DipHighConfig(name="L14_D20_TP5",   lookback_days=14, drop_pct=0.20, tp_pct=0.05),
        DipHighConfig(name="L14_D20_TP10",  lookback_days=14, drop_pct=0.20, tp_pct=0.10),
        DipHighConfig(name="L14_D20_TP15",  lookback_days=14, drop_pct=0.20, tp_pct=0.15),
        DipHighConfig(name="L14_D20_TRAIL", lookback_days=14, drop_pct=0.20,
                      tp_enabled=False, trailing_enabled=True),

        # ══ G2 — Lookback 14j, Drop -10% ══════════════════════════════════════
        DipHighConfig(name="L14_D10_TP5",   lookback_days=14, drop_pct=0.10, tp_pct=0.05),
        DipHighConfig(name="L14_D10_TP10",  lookback_days=14, drop_pct=0.10, tp_pct=0.10),
        DipHighConfig(name="L14_D10_TP15",  lookback_days=14, drop_pct=0.10, tp_pct=0.15),
        DipHighConfig(name="L14_D10_TRAIL", lookback_days=14, drop_pct=0.10,
                      tp_enabled=False, trailing_enabled=True),

        # ══ G3 — Lookback 7j, Drop -20% ═══════════════════════════════════════
        DipHighConfig(name="L7_D20_TP5",    lookback_days=7,  drop_pct=0.20, tp_pct=0.05),
        DipHighConfig(name="L7_D20_TP10",   lookback_days=7,  drop_pct=0.20, tp_pct=0.10),
        DipHighConfig(name="L7_D20_TP15",   lookback_days=7,  drop_pct=0.20, tp_pct=0.15),
        DipHighConfig(name="L7_D20_TRAIL",  lookback_days=7,  drop_pct=0.20,
                      tp_enabled=False, trailing_enabled=True),

        # ══ G4 — Lookback 7j, Drop -10% ═══════════════════════════════════════
        DipHighConfig(name="L7_D10_TP5",    lookback_days=7,  drop_pct=0.10, tp_pct=0.05),
        DipHighConfig(name="L7_D10_TP10",   lookback_days=7,  drop_pct=0.10, tp_pct=0.10),
        DipHighConfig(name="L7_D10_TP15",   lookback_days=7,  drop_pct=0.10, tp_pct=0.15),
        DipHighConfig(name="L7_D10_TRAIL",  lookback_days=7,  drop_pct=0.10,
                      tp_enabled=False, trailing_enabled=True),

        # ══ G5 — Combos TP + Trailing ensemble (TP agit comme filet de securite) ══
        # TP fixe + trailing : le trailing peut sortir avant le TP si pic atteint
        DipHighConfig(name="L14_D20_TP15+TRAIL", lookback_days=14, drop_pct=0.20,
                      tp_pct=0.15, trailing_enabled=True),
        DipHighConfig(name="L14_D10_TP10+TRAIL", lookback_days=14, drop_pct=0.10,
                      tp_pct=0.10, trailing_enabled=True),
        DipHighConfig(name="L7_D20_TP10+TRAIL",  lookback_days=7,  drop_pct=0.20,
                      tp_pct=0.10, trailing_enabled=True),

        # ══ G6 — L7_D20 : SL dur + Timeout (reduce les trades END catastrophiques) ══
        # Objectif : conserver le PF eleve de L7_D20 tout en limitant le DD
        DipHighConfig(name="L7_D20_TP5_SL30",   lookback_days=7, drop_pct=0.20,
                      tp_pct=0.05, sl_pct=0.30),
        DipHighConfig(name="L7_D20_TP5_SL35",   lookback_days=7, drop_pct=0.20,
                      tp_pct=0.05, sl_pct=0.35),
        DipHighConfig(name="L7_D20_TP10_SL30",  lookback_days=7, drop_pct=0.20,
                      tp_pct=0.10, sl_pct=0.30),
        DipHighConfig(name="L7_D20_TP10_SL35",  lookback_days=7, drop_pct=0.20,
                      tp_pct=0.10, sl_pct=0.35),
        DipHighConfig(name="L7_D20_TRAIL_SL30", lookback_days=7, drop_pct=0.20,
                      tp_enabled=False, trailing_enabled=True, sl_pct=0.30),
        DipHighConfig(name="L7_D20_TP5_TO30",   lookback_days=7, drop_pct=0.20,
                      tp_pct=0.05, max_bars_in_trade=30 * H4_PER_DAY),
        DipHighConfig(name="L7_D20_TP5_TO60",   lookback_days=7, drop_pct=0.20,
                      tp_pct=0.05, max_bars_in_trade=60 * H4_PER_DAY),
        DipHighConfig(name="L7_D20_TP10_TO30",  lookback_days=7, drop_pct=0.20,
                      tp_pct=0.10, max_bars_in_trade=30 * H4_PER_DAY),

        # ══ G8 — Filtre regime BTC MA200 (bloque entrees quand BTC < MA200) ══
        DipHighConfig(name="L7_D20_TP5_MA200",   lookback_days=7, drop_pct=0.20,
                      tp_pct=0.05, regime_filter_enabled=True),
        DipHighConfig(name="L7_D20_TP10_MA200",  lookback_days=7, drop_pct=0.20,
                      tp_pct=0.10, regime_filter_enabled=True),
        DipHighConfig(name="L7_D20_TRAIL_MA200", lookback_days=7, drop_pct=0.20,
                      tp_enabled=False, trailing_enabled=True, regime_filter_enabled=True),
        DipHighConfig(name="L7_D20_TP5_R15_MA200", lookback_days=7, drop_pct=0.20,
                      tp_pct=0.05, risk_pct=0.15, regime_filter_enabled=True),

        # ══ G9 — MA plus courte : MA50 et MA100 (sortie du bear plus rapide) ══
        DipHighConfig(name="L7_D20_TP5_MA50",    lookback_days=7, drop_pct=0.20,
                      tp_pct=0.05, regime_filter_enabled=True, ma_period_days=50),
        DipHighConfig(name="L7_D20_TP5_MA100",   lookback_days=7, drop_pct=0.20,
                      tp_pct=0.05, regime_filter_enabled=True, ma_period_days=100),
        DipHighConfig(name="L7_D20_TP10_MA50",   lookback_days=7, drop_pct=0.20,
                      tp_pct=0.10, regime_filter_enabled=True, ma_period_days=50),
        DipHighConfig(name="L7_D20_TP10_MA100",  lookback_days=7, drop_pct=0.20,
                      tp_pct=0.10, regime_filter_enabled=True, ma_period_days=100),
        DipHighConfig(name="L7_D20_TP5_R15_MA50",  lookback_days=7, drop_pct=0.20,
                      tp_pct=0.05, risk_pct=0.15, regime_filter_enabled=True, ma_period_days=50),
        DipHighConfig(name="L7_D20_TP5_R15_MA100", lookback_days=7, drop_pct=0.20,
                      tp_pct=0.05, risk_pct=0.15, regime_filter_enabled=True, ma_period_days=100),

        # ══ G10 — Max positions = 1 (anti-correlation, pas de surexposition simultanee) ══
        DipHighConfig(name="L7_D20_TP5_P1",       lookback_days=7, drop_pct=0.20,
                      tp_pct=0.05, max_simultaneous=1),
        DipHighConfig(name="L7_D20_TP10_P1",      lookback_days=7, drop_pct=0.20,
                      tp_pct=0.10, max_simultaneous=1),
        DipHighConfig(name="L7_D20_TP5_MA200_P1", lookback_days=7, drop_pct=0.20,
                      tp_pct=0.05, regime_filter_enabled=True, max_simultaneous=1),
        DipHighConfig(name="L7_D20_TP10_MA200_P1",lookback_days=7, drop_pct=0.20,
                      tp_pct=0.10, regime_filter_enabled=True, max_simultaneous=1),
        DipHighConfig(name="L7_D20_TP5_R15_MA200_P1", lookback_days=7, drop_pct=0.20,
                      tp_pct=0.05, risk_pct=0.15, regime_filter_enabled=True, max_simultaneous=1),

        # ══ G11 — Risk agressif 25%/30% sur meilleures configs PnL max ══
        DipHighConfig(name="L7_D20_TP5_R25",  lookback_days=7, drop_pct=0.20,
                      tp_pct=0.05, risk_pct=0.25),
        DipHighConfig(name="L7_D20_TP5_R30",  lookback_days=7, drop_pct=0.20,
                      tp_pct=0.05, risk_pct=0.30),
        DipHighConfig(name="L7_D20_TP10_R25", lookback_days=7, drop_pct=0.20,
                      tp_pct=0.10, risk_pct=0.25),
        DipHighConfig(name="L7_D20_TP10_R30", lookback_days=7, drop_pct=0.20,
                      tp_pct=0.10, risk_pct=0.30),

        # ══ G12 — Risk agressif 25%/30% + filtre MA200 (meilleur des deux mondes) ══
        DipHighConfig(name="L7_D20_TP5_R25_MA200",  lookback_days=7, drop_pct=0.20,
                      tp_pct=0.05, risk_pct=0.25, regime_filter_enabled=True),
        DipHighConfig(name="L7_D20_TP5_R30_MA200",  lookback_days=7, drop_pct=0.20,
                      tp_pct=0.05, risk_pct=0.30, regime_filter_enabled=True),
        DipHighConfig(name="L7_D20_TP10_R25_MA200", lookback_days=7, drop_pct=0.20,
                      tp_pct=0.10, risk_pct=0.25, regime_filter_enabled=True),
        DipHighConfig(name="L7_D20_TP10_R30_MA200", lookback_days=7, drop_pct=0.20,
                      tp_pct=0.10, risk_pct=0.30, regime_filter_enabled=True),

        # ══ G7 — Risk 10% et 15% (vs 20% reference) sur meilleures configs ══
        DipHighConfig(name="L7_D20_TP5_R10",  lookback_days=7, drop_pct=0.20,
                      tp_pct=0.05, risk_pct=0.10),
        DipHighConfig(name="L7_D20_TP5_R15",  lookback_days=7, drop_pct=0.20,
                      tp_pct=0.05, risk_pct=0.15),
        DipHighConfig(name="L7_D20_TP10_R10", lookback_days=7, drop_pct=0.20,
                      tp_pct=0.10, risk_pct=0.10),
        DipHighConfig(name="L7_D20_TP10_R15", lookback_days=7, drop_pct=0.20,
                      tp_pct=0.10, risk_pct=0.15),
        DipHighConfig(name="L7_D20_TRAIL_R10", lookback_days=7, drop_pct=0.20,
                      tp_enabled=False, trailing_enabled=True, risk_pct=0.10),
        DipHighConfig(name="L7_D20_TRAIL_R15", lookback_days=7, drop_pct=0.20,
                      tp_enabled=False, trailing_enabled=True, risk_pct=0.15),
    ]


# ── Metriques ──────────────────────────────────────────────────────────────────


def _compute_metrics(
    trades: list[DipHighTrade],
    equity_curve: list[EquityPoint],
    initial_balance: float,
    final_balance: float,
    years: float,
) -> dict:
    import math
    n = len(trades)
    total_return = (final_balance - initial_balance) / initial_balance if initial_balance > 0 else 0
    cagr = (final_balance / initial_balance) ** (1 / max(years, 0.01)) - 1 if final_balance > 0 else 0

    # Drawdown
    peak = initial_balance
    max_dd = 0.0
    for pt in equity_curve:
        peak = max(peak, pt.equity)
        dd = (pt.equity - peak) / peak if peak > 0 else 0
        max_dd = min(max_dd, dd)

    # Sharpe
    returns = []
    for i in range(1, len(equity_curve)):
        prev = equity_curve[i - 1].equity
        if prev > 0:
            returns.append((equity_curve[i].equity - prev) / prev)
    sharpe = 0.0
    if len(returns) > 1:
        mean_r = sum(returns) / len(returns)
        std_r = (sum((r - mean_r) ** 2 for r in returns) / len(returns)) ** 0.5
        if std_r > 0:
            sharpe = (mean_r / std_r) * math.sqrt(365 * H4_PER_DAY)

    if n:
        wins = [t for t in trades if t.pnl_usd > 0]
        losses = [t for t in trades if t.pnl_usd <= 0]
        win_rate = len(wins) / n
        gross_profit = sum(t.pnl_usd for t in wins) or 0
        gross_loss = abs(sum(t.pnl_usd for t in losses)) or 1e-9
        profit_factor = gross_profit / gross_loss
        avg_pnl = sum(t.pnl_usd for t in trades) / n
        avg_pnl_pct = sum(t.pnl_pct for t in trades) / n
        avg_duration = sum(t.duration_bars for t in trades) / n
        total_fees = sum(t.fees for t in trades)
        best = max(trades, key=lambda t: t.pnl_usd)
        worst = min(trades, key=lambda t: t.pnl_usd)
        max_consec = _max_consec_losses(trades)
        avg_drop_entry = sum(t.drop_pct_at_entry for t in trades) / n
    else:
        win_rate = profit_factor = avg_pnl = avg_pnl_pct = avg_duration = 0
        total_fees = avg_drop_entry = 0
        best = worst = None
        max_consec = 0

    # Par paire
    by_pair: dict[str, dict] = {}
    pair_trades: dict[str, list] = defaultdict(list)
    for t in trades:
        pair_trades[t.symbol].append(t)
    for sym, pts in sorted(pair_trades.items()):
        w = [t for t in pts if t.pnl_usd > 0]
        by_pair[sym] = {
            "n": len(pts),
            "wr": len(w) / len(pts) if pts else 0,
            "pnl": sum(t.pnl_usd for t in pts),
            "avg_dur": sum(t.duration_bars for t in pts) / len(pts) if pts else 0,
        }

    # Par motif de sortie
    by_exit: dict[str, dict] = {}
    exit_trades: dict[str, list] = defaultdict(list)
    for t in trades:
        exit_trades[t.exit_reason].append(t)
    for reason, ets in sorted(exit_trades.items()):
        w = [t for t in ets if t.pnl_usd > 0]
        by_exit[reason] = {
            "n": len(ets),
            "wr": len(w) / len(ets) if ets else 0,
            "pnl": sum(t.pnl_usd for t in ets),
        }

    return {
        "total_return": total_return, "cagr": cagr, "max_drawdown": max_dd, "sharpe": sharpe,
        "win_rate": win_rate, "profit_factor": profit_factor, "n_trades": n,
        "avg_pnl": avg_pnl, "avg_pnl_pct": avg_pnl_pct, "avg_duration": avg_duration,
        "total_fees": total_fees, "max_consec_losses": max_consec, "avg_drop_entry": avg_drop_entry,
        "best": best, "worst": worst, "by_pair": by_pair, "by_exit": by_exit,
        "final_balance": final_balance,
    }


def _max_consec_losses(trades: list) -> int:
    max_c = consec = 0
    for t in trades:
        if t.pnl_usd <= 0:
            consec += 1
            max_c = max(max_c, consec)
        else:
            consec = 0
    return max_c


# ── Rapport console ────────────────────────────────────────────────────────────


def _print_report(cfg: DipHighConfig, m: dict, initial_balance: float, years: float) -> None:
    sep = "=" * 80
    trail_s = f"TRAIL +{cfg.trailing_activation_pct:.0%}/dist {cfg.trailing_distance_pct:.0%}" if cfg.trailing_enabled else "OFF"
    tp_s = f"+{cfg.tp_pct:.0%}" if cfg.tp_enabled else "OFF"
    print(f"\n{sep}")
    print(f"  DIP FROM HIGH — {cfg.name}")
    print(f"  Capital: ${initial_balance:,.0f} | Lookback: {cfg.lookback_days}j"
          f" | Drop: -{cfg.drop_pct:.0%} | TP: {tp_s} | Trail: {trail_s}")
    print(f"  Risk/trade: {cfg.risk_pct:.0%} | Fees: {cfg.entry_fee_pct:.2%} maker + {cfg.exit_fee_pct:.2%} taker")
    print(sep)
    print(f"\n  RESULTATS GLOBAUX")
    print("  " + "-" * 76)
    print(f"  Capital final      : ${m['final_balance']:,.2f} ({m['total_return']:+.1%})")
    print(f"  CAGR               : {m['cagr']:.1%}")
    print(f"  Max Drawdown       : {m['max_drawdown']:.1%}")
    print(f"  Sharpe Ratio       : {m['sharpe']:.2f}")
    print(f"  Win Rate           : {m['win_rate']:.1%} ({int(m['win_rate'] * m['n_trades'])}/{m['n_trades']})")
    print(f"  Profit Factor      : {m['profit_factor']:.2f}")
    print(f"  Trades             : {m['n_trades']}")
    print(f"  Baisse moy. entree : {m['avg_drop_entry']:.1%}")
    print(f"  PnL moyen          : ${m['avg_pnl']:+.2f} ({m['avg_pnl_pct']:+.2%})")
    print(f"  Duree moy. trade   : {m['avg_duration']:.0f} barres ({m['avg_duration'] / H4_PER_DAY:.0f}j)")
    print(f"  Total fees         : ${m['total_fees']:.2f}")
    print(f"  Max pertes consec. : {m['max_consec_losses']}")
    if m["best"]:
        b = m["best"]
        print(f"  Meilleur trade     : ${b.pnl_usd:+.2f} ({b.pnl_pct:+.1%}) {b.symbol}")
    if m["worst"]:
        w = m["worst"]
        print(f"  Pire trade         : ${w.pnl_usd:+.2f} ({w.pnl_pct:+.1%}) {w.symbol}")
    if m["by_pair"]:
        print(f"\n  PAR PAIRE")
        print("  " + "-" * 76)
        for sym, s in m["by_pair"].items():
            print(f"  {sym:12s} : {s['n']:3d} trades | WR {s['wr']:.0%}"
                  f" | PnL ${s['pnl']:+.2f} | Duree moy {s['avg_dur'] / H4_PER_DAY:.0f}j")
    if m["by_exit"]:
        print(f"\n  PAR MOTIF DE SORTIE")
        print("  " + "-" * 76)
        for reason, s in m["by_exit"].items():
            print(f"  {reason:12s} : {s['n']:3d} trades | WR {s['wr']:.0%} | PnL ${s['pnl']:+.2f}")
    print(f"\n{sep}\n")


# ── Comparaison ────────────────────────────────────────────────────────────────


def _print_comparison(results: list[tuple[DipHighConfig, dict]], initial_balance: float) -> None:
    sep = "=" * 110
    print(f"\n{sep}")
    print(f"  COMPARAISON DIP-FROM-HIGH — {len(results)} variantes | Big5 | 20%/trade")
    print(sep)

    header = (f"  {'Variante':22s} | {'Look':>4s} | {'Drop':>5s} | {'TP':>5s} | {'Trail':>5s}"
              f" | {'Trades':>6s} | {'WR':>6s} | {'PF':>5s}"
              f" | {'PnL $':>9s} | {'DD':>6s} | {'Sharpe':>6s} | {'Dur':>5s}")
    print(header)
    print("  " + "-" * 106)

    for cfg, m in results:
        trail_s = f"+{cfg.trailing_activation_pct:.0%}" if cfg.trailing_enabled else "OFF"
        tp_s = f"+{cfg.tp_pct:.0%}" if cfg.tp_enabled else "OFF"
        pnl = m["final_balance"] - initial_balance
        print(f"  {cfg.name:22s} | {cfg.lookback_days:3d}j | -{cfg.drop_pct:.0%}"
              f" | {tp_s:>5s} | {trail_s:>5s}"
              f" | {m['n_trades']:>6d} | {m['win_rate']:5.1%} | {m['profit_factor']:5.2f}"
              f" | ${pnl:>+8.2f} | {m['max_drawdown']:>5.1%} | {m['sharpe']:>6.2f}"
              f" | {m['avg_duration'] / H4_PER_DAY:>4.0f}j")

    print(sep)

    by_name = {cfg.name: (cfg, m) for cfg, m in results}
    print(f"\n  COMPARAISONS CLES :")
    print("  " + "-" * 76)

    # G1 vs G2 : Drop -20% vs -10% (lookback 14j)
    print(f"\n  Lookback 14j \u2014 Drop -20% vs -10% :")
    for tp_str, tp_val in [("TP5", 0.05), ("TP10", 0.10), ("TP15", 0.15)]:
        n20 = f"L14_D20_{tp_str}"
        n10 = f"L14_D10_{tp_str}"
        if n20 in by_name and n10 in by_name:
            _, m20 = by_name[n20]
            _, m10 = by_name[n10]
            p20 = m20["final_balance"] - initial_balance
            p10 = m10["final_balance"] - initial_balance
            print(f"    TP+{tp_val:.0%}:  -20% \u2192 PF {m20['profit_factor']:.2f} {m20['n_trades']:3d}t PnL ${p20:+.2f}"
                  f"  |  -10% \u2192 PF {m10['profit_factor']:.2f} {m10['n_trades']:3d}t PnL ${p10:+.2f}")

    # G3 vs G4 : Drop -20% vs -10% (lookback 7j)
    print(f"\n  Lookback 7j \u2014 Drop -20% vs -10% :")
    for tp_str, tp_val in [("TP5", 0.05), ("TP10", 0.10), ("TP15", 0.15)]:
        n20 = f"L7_D20_{tp_str}"
        n10 = f"L7_D10_{tp_str}"
        if n20 in by_name and n10 in by_name:
            _, m20 = by_name[n20]
            _, m10 = by_name[n10]
            p20 = m20["final_balance"] - initial_balance
            p10 = m10["final_balance"] - initial_balance
            print(f"    TP+{tp_val:.0%}:  -20% \u2192 PF {m20['profit_factor']:.2f} {m20['n_trades']:3d}t PnL ${p20:+.2f}"
                  f"  |  -10% \u2192 PF {m10['profit_factor']:.2f} {m10['n_trades']:3d}t PnL ${p10:+.2f}")

    # Trailing vs TP fixe
    print(f"\n  Trailing (activ. +5%, SL=breakeven) vs TP fixe :")
    for lookback, drop, group in [(14, 20, "L14_D20"), (14, 10, "L14_D10"),
                                   (7, 20, "L7_D20"), (7, 10, "L7_D10")]:
        ref = f"{group}_TP10"
        trail = f"{group}_TRAIL"
        if ref in by_name and trail in by_name:
            _, m_tp = by_name[ref]
            _, m_tr = by_name[trail]
            p_tp = m_tp["final_balance"] - initial_balance
            p_tr = m_tr["final_balance"] - initial_balance
            print(f"    {group}:  TP+10% \u2192 PF {m_tp['profit_factor']:.2f} PnL ${p_tp:+.2f}"
                  f"  |  Trail \u2192 PF {m_tr['profit_factor']:.2f} PnL ${p_tr:+.2f}")

    # G5 : combos
    g5 = ["L14_D20_TP15+TRAIL", "L14_D10_TP10+TRAIL", "L7_D20_TP10+TRAIL"]
    found = [(n, by_name[n]) for n in g5 if n in by_name]
    if found:
        print(f"\n  G5 \u2014 TP + Trailing combines :")
        for name, (cfg, m) in found:
            pnl = m["final_balance"] - initial_balance
            print(f"    {name:25s} : PF {m['profit_factor']:.2f} | {m['n_trades']:3d}t"
                  f" | WR {m['win_rate']:.1%} | PnL ${pnl:+.2f} | DD {m['max_drawdown']:.1%}")

    # G6 : SL / Timeout vs reference sans SL
    g6 = [
        "L7_D20_TP5_SL30", "L7_D20_TP5_SL35",
        "L7_D20_TP10_SL30", "L7_D20_TP10_SL35",
        "L7_D20_TRAIL_SL30",
        "L7_D20_TP5_TO30", "L7_D20_TP5_TO60", "L7_D20_TP10_TO30",
    ]
    found_g6 = [(n, by_name[n]) for n in g6 if n in by_name]
    if found_g6:
        print(f"\n  G6 \u2014 SL dur & Timeout sur L7_D20 (vs reference sans protection) :")
        refs = {"TP5": "L7_D20_TP5", "TP10": "L7_D20_TP10", "TRAIL": "L7_D20_TRAIL"}
        for ref_label, ref_name in refs.items():
            if ref_name in by_name:
                _, mr = by_name[ref_name]
                pr = mr["final_balance"] - initial_balance
                print(f"    REF {ref_name:20s} : PF {mr['profit_factor']:.2f}"
                      f" | WR {mr['win_rate']:.1%} | PnL ${pr:+.2f} | DD {mr['max_drawdown']:.1%}")
        print()
        for name, (cfg, m) in found_g6:
            pnl = m["final_balance"] - initial_balance
            sl_s = f"SL-{cfg.sl_pct:.0%}" if cfg.sl_pct > 0 else "    "
            to_s = f"TO{cfg.max_bars_in_trade // H4_PER_DAY}j" if cfg.max_bars_in_trade > 0 else "    "
            print(f"    {name:25s} [{sl_s or to_s}] : PF {m['profit_factor']:.2f}"
                  f" | WR {m['win_rate']:.1%} | PnL ${pnl:+.2f} | DD {m['max_drawdown']:.1%}"
                  f" | {m['n_trades']}t")

    # G7 : Risk 10% / 15% vs 20% de reference
    g7 = ["L7_D20_TP5_R10", "L7_D20_TP5_R15", "L7_D20_TP10_R10", "L7_D20_TP10_R15",
          "L7_D20_TRAIL_R10", "L7_D20_TRAIL_R15"]
    found_g7 = [(n, by_name[n]) for n in g7 if n in by_name]
    if found_g7:
        print(f"\n  G7 \u2014 Impact du risk/trade sur L7_D20 (ref = 20%) :")
        refs_g7 = [("L7_D20_TP5", "TP5 R20%"), ("L7_D20_TP10", "TP10 R20%"), ("L7_D20_TRAIL", "TRAIL R20%")]
        for ref_name, label in refs_g7:
            if ref_name in by_name:
                _, mr = by_name[ref_name]
                pr = mr["final_balance"] - initial_balance
                print(f"    REF {label:15s} : PF {mr['profit_factor']:.2f}"
                      f" | WR {mr['win_rate']:.1%} | PnL ${pr:+.2f} | DD {mr['max_drawdown']:.1%}")
        print()
        for name, (cfg, m) in found_g7:
            pnl = m["final_balance"] - initial_balance
            print(f"    {name:25s} [R{cfg.risk_pct:.0%}] : PF {m['profit_factor']:.2f}"
                  f" | WR {m['win_rate']:.1%} | PnL ${pnl:+.2f} | DD {m['max_drawdown']:.1%}"
                  f" | {m['n_trades']}t")

    # G8 : Regime filter MA200
    g8 = ["L7_D20_TP5_MA200", "L7_D20_TP10_MA200", "L7_D20_TRAIL_MA200", "L7_D20_TP5_R15_MA200"]
    found_g8 = [(n, by_name[n]) for n in g8 if n in by_name]
    if found_g8:
        print(f"\n  G8 \u2014 Filtre regime BTC MA200 (vs reference sans filtre) :")
        refs_g8 = [("L7_D20_TP5", "TP5 no-filter"), ("L7_D20_TP10", "TP10 no-filter"),
                   ("L7_D20_TRAIL", "TRAIL no-filter"), ("L7_D20_TP5_R15", "TP5_R15 no-filter")]
        for ref_name, label in refs_g8:
            if ref_name in by_name:
                _, mr = by_name[ref_name]
                pr = mr["final_balance"] - initial_balance
                print(f"    REF {label:20s} : PF {mr['profit_factor']:.2f}"
                      f" | WR {mr['win_rate']:.1%} | PnL ${pr:+.2f} | DD {mr['max_drawdown']:.1%}")
        print()
        for name, (cfg, m) in found_g8:
            pnl = m["final_balance"] - initial_balance
            n_blocked = m.get("n_trades", 0)
            print(f"    {name:25s} [MA200] : PF {m['profit_factor']:.2f}"
                  f" | WR {m['win_rate']:.1%} | PnL ${pnl:+.2f} | DD {m['max_drawdown']:.1%}"
                  f" | {m['n_trades']}t")

    # G9 : MA50 / MA100 vs MA200
    g9 = ["L7_D20_TP5_MA50", "L7_D20_TP5_MA100", "L7_D20_TP10_MA50", "L7_D20_TP10_MA100",
          "L7_D20_TP5_R15_MA50", "L7_D20_TP5_R15_MA100"]
    found_g9 = [(n, by_name[n]) for n in g9 if n in by_name]
    if found_g9:
        print(f"\n  G9 \u2014 MA plus courte (MA50/MA100 vs MA200) :\n")
        refs_g9 = [("L7_D20_TP5_MA200", "TP5+MA200"), ("L7_D20_TP10_MA200", "TP10+MA200"),
                   ("L7_D20_TP5_R15_MA200", "TP5_R15+MA200")]
        for ref_name, label in refs_g9:
            if ref_name in by_name:
                _, mr = by_name[ref_name]
                pr = mr["final_balance"] - initial_balance
                print(f"    REF {label:20s} : PF {mr['profit_factor']:.2f}"
                      f" | WR {mr['win_rate']:.1%} | PnL ${pr:+.2f} | DD {mr['max_drawdown']:.1%} | {mr['n_trades']}t")
        print()
        for name, (cfg, m) in found_g9:
            pnl = m["final_balance"] - initial_balance
            ma_s = f"MA{cfg.ma_period_days}"
            print(f"    {name:28s} [{ma_s}] : PF {m['profit_factor']:.2f}"
                  f" | WR {m['win_rate']:.1%} | PnL ${pnl:+.2f} | DD {m['max_drawdown']:.1%} | {m['n_trades']}t")

    # G10 : max_simultaneous = 1
    g10 = ["L7_D20_TP5_P1", "L7_D20_TP10_P1", "L7_D20_TP5_MA200_P1", "L7_D20_TP10_MA200_P1",
           "L7_D20_TP5_R15_MA200_P1"]
    found_g10 = [(n, by_name[n]) for n in g10 if n in by_name]
    if found_g10:
        print(f"\n  G10 \u2014 Max 1 position simultanee (vs 5 ref) :\n")
        refs_g10 = [("L7_D20_TP5", "TP5 P5"), ("L7_D20_TP10", "TP10 P5"),
                    ("L7_D20_TP5_MA200", "TP5+MA200 P5"), ("L7_D20_TP10_MA200", "TP10+MA200 P5"),
                    ("L7_D20_TP5_R15_MA200", "TP5_R15+MA200 P5")]
        for ref_name, label in refs_g10:
            if ref_name in by_name:
                _, mr = by_name[ref_name]
                pr = mr["final_balance"] - initial_balance
                print(f"    REF {label:22s} : PF {mr['profit_factor']:.2f}"
                      f" | WR {mr['win_rate']:.1%} | PnL ${pr:+.2f} | DD {mr['max_drawdown']:.1%} | {mr['n_trades']}t")
        print()
        for name, (cfg, m) in found_g10:
            pnl = m["final_balance"] - initial_balance
            print(f"    {name:28s} [P1] : PF {m['profit_factor']:.2f}"
                  f" | WR {m['win_rate']:.1%} | PnL ${pnl:+.2f} | DD {m['max_drawdown']:.1%} | {m['n_trades']}t")

    # G11 : Risk agressif 25%/30%
    g11 = ["L7_D20_TP5_R25", "L7_D20_TP5_R30", "L7_D20_TP10_R25", "L7_D20_TP10_R30"]
    found_g11 = [(n, by_name[n]) for n in g11 if n in by_name]
    if found_g11:
        print(f"\n  G11 \u2014 Risk agressif 25%/30% (PnL max, vs R20% ref) :\n")
        refs_g11 = [("L7_D20_TP5", "TP5 R20%"), ("L7_D20_TP10", "TP10 R20%")]
        for ref_name, label in refs_g11:
            if ref_name in by_name:
                _, mr = by_name[ref_name]
                pr = mr["final_balance"] - initial_balance
                print(f"    REF {label:15s} : PF {mr['profit_factor']:.2f}"
                      f" | WR {mr['win_rate']:.1%} | PnL ${pr:+.2f} | DD {mr['max_drawdown']:.1%} | {mr['n_trades']}t")
        print()
        for name, (cfg, m) in found_g11:
            pnl = m["final_balance"] - initial_balance
            print(f"    {name:25s} [R{cfg.risk_pct:.0%}] : PF {m['profit_factor']:.2f}"
                  f" | WR {m['win_rate']:.1%} | PnL ${pnl:+.2f} | DD {m['max_drawdown']:.1%} | {m['n_trades']}t")

    # G12 : Risk agressif + MA200 (meilleur des deux mondes)
    g12 = ["L7_D20_TP5_R25_MA200", "L7_D20_TP5_R30_MA200", "L7_D20_TP10_R25_MA200", "L7_D20_TP10_R30_MA200"]
    found_g12 = [(n, by_name[n]) for n in g12 if n in by_name]
    if found_g12:
        print(f"\n  G12 \u2014 Risk agressif 25%/30% + filtre MA200 (best of both worlds) :\n")
        refs_g12 = [("L7_D20_TP5_MA200", "TP5 R20%+MA200"), ("L7_D20_TP10_MA200", "TP10 R20%+MA200")]
        for ref_name, label in refs_g12:
            if ref_name in by_name:
                _, mr = by_name[ref_name]
                pr = mr["final_balance"] - initial_balance
                print(f"    REF {label:18s} : PF {mr['profit_factor']:.2f}"
                      f" | WR {mr['win_rate']:.1%} | PnL ${pr:+.2f} | DD {mr['max_drawdown']:.1%} | {mr['n_trades']}t")
        print()
        for name, (cfg, m) in found_g12:
            pnl = m["final_balance"] - initial_balance
            print(f"    {name:28s} [R{cfg.risk_pct:.0%}+MA200] : PF {m['profit_factor']:.2f}"
                  f" | WR {m['win_rate']:.1%} | PnL ${pnl:+.2f} | DD {m['max_drawdown']:.1%} | {m['n_trades']}t")

    # Classement PnL (top 5)
    ranked_by_pnl = sorted(results, key=lambda x: x[1]["final_balance"], reverse=True)[:5]
    print(f"\n  \u2605 TOP 5 PAR PnL :")
    for cfg, m in ranked_by_pnl:
        pnl = m["final_balance"] - initial_balance
        print(f"    {cfg.name:30s} : PnL ${pnl:+.2f} | PF {m['profit_factor']:.2f}"
              f" | WR {m['win_rate']:.1%} | DD {m['max_drawdown']:.1%}")

    best_cfg, best_m = max(results, key=lambda x: x[1]["profit_factor"])
    pnl_best = best_m["final_balance"] - initial_balance
    print(f"\n  \u2605 MEILLEURE VARIANTE (PF) : {best_cfg.name}"
          f" | PF {best_m['profit_factor']:.2f} | WR {best_m['win_rate']:.1%}"
          f" | PnL ${pnl_best:+.2f} | DD {best_m['max_drawdown']:.1%}")
    print(f"\n{sep}\n")


# ── Charts ─────────────────────────────────────────────────────────────────────


def _generate_chart(cfg: DipHighConfig, equity_curve: list[EquityPoint],
                    initial_balance: float) -> None:
    if not equity_curve:
        return
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUTPUT_DIR / f"diphi_{cfg.name}.png"
    dates = [datetime.fromtimestamp(pt.ts / 1000, tz=timezone.utc) for pt in equity_curve]
    equities = [pt.equity for pt in equity_curve]
    fig, ax = plt.subplots(figsize=(14, 4))
    ax.plot(dates, equities, linewidth=1.0, color="#FF6B35")
    ax.axhline(initial_balance, color="gray", linestyle="--", alpha=0.5)
    ax.set_title(f"Dip-from-High — {cfg.name}")
    ax.set_ylabel("Equity ($)")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=100)
    plt.close(fig)
    logger.warning("Chart saved: %s", path)


# ── Main ───────────────────────────────────────────────────────────────────────


def _build_regime_mask(btc_candles: list[Candle], ma_period_days: int) -> list[bool]:
    """Retourne True pour chaque barre ou BTC est au-dessus de sa MA (en jours)."""
    period_bars = ma_period_days * H4_PER_DAY
    closes = [c.close for c in btc_candles]
    ma = _ma_series(closes, period_bars)
    return [closes[i] >= ma[i] > 0 for i in range(len(closes))]


def _run_variants(
    variants: list[DipHighConfig],
    all_candles: dict[str, list[Candle]],
    initial_balance: float,
    years: float,
    generate_charts: bool = True,
) -> list[tuple[DipHighConfig, dict]]:
    # Pre-calcul du regime mask BTC (reutilise par tous les cfgs avec filtre actif)
    btc_regime_cache: dict[int, list[bool]] = {}
    btc_candles = all_candles.get("BTC-USD", [])

    results: list[tuple[DipHighConfig, dict]] = []
    for cfg in variants:
        balance = initial_balance
        all_trades: list[DipHighTrade] = []
        merged_equity: list[EquityPoint] = []

        # Regime mask : calcule une fois par periode MA
        regime_mask: list[bool] | None = None
        if cfg.regime_filter_enabled and btc_candles:
            if cfg.ma_period_days not in btc_regime_cache:
                btc_regime_cache[cfg.ma_period_days] = _build_regime_mask(btc_candles, cfg.ma_period_days)
            regime_mask = btc_regime_cache[cfg.ma_period_days]

        for pair in sorted(all_candles.keys()):
            tr, eq, balance = run_pair(all_candles[pair], cfg, balance, pair, regime_mask)
            all_trades.extend(tr)
            merged_equity.extend(eq)

        merged_equity.sort(key=lambda p: p.ts)
        m = _compute_metrics(all_trades, merged_equity, initial_balance, balance, years)
        results.append((cfg, m))
        pnl = balance - initial_balance
        trail_s = "TRAIL" if cfg.trailing_enabled else "    "
        print(f"  {cfg.name:25s} {trail_s} : {len(all_trades):3d}t"
              f" | PF {m['profit_factor']:.2f} | PnL ${pnl:+.2f} | WR {m['win_rate']:.1%}")

        if generate_charts:
            _generate_chart(cfg, merged_equity, initial_balance)

    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Backtest Dip-from-Rolling-High")
    parser.add_argument("--balance", type=float, default=1000.0)
    parser.add_argument("--years", type=float, default=2.0)
    parser.add_argument("--interval", type=str, default="4h")
    parser.add_argument("--compare", action="store_true")
    parser.add_argument("--start", type=str, default=None, help="Date debut YYYY-MM-DD")
    parser.add_argument("--end", type=str, default=None, help="Date fin YYYY-MM-DD")
    args = parser.parse_args()

    initial_balance = args.balance
    interval = args.interval

    if args.start and args.end:
        start = datetime.strptime(args.start, "%Y-%m-%d").replace(tzinfo=timezone.utc)
        end = datetime.strptime(args.end, "%Y-%m-%d").replace(tzinfo=timezone.utc)
        years = (end - start).days / 365.25
        period_label = f"{args.start} → {args.end}"
    else:
        years = args.years
        end = datetime.now(timezone.utc)
        start = end - timedelta(days=int(years * 365.25))
        period_label = f"{years:.1f} ans"

    print(f"\n📥 Telechargement {len(PAIRS_BIG5)} paires ({interval}, {period_label})...")
    all_candles: dict[str, list[Candle]] = {}
    for pair in PAIRS_BIG5:
        candles = download_candles(pair, start, end, interval=interval)
        if candles:
            all_candles[pair] = candles
            print(f"  ✓ {pair}: {len(candles)} bougies")
        else:
            print(f"  ✗ {pair}: aucune donnee")

    if not all_candles:
        print("❌ Aucune donnee.")
        return

    if args.compare:
        variants = get_variants()
        print(f"\n🔬 Comparaison de {len(variants)} variantes (Big5, {period_label})...")
        results = _run_variants(variants, all_candles, initial_balance, years)
        _print_comparison(results, initial_balance)
        best_cfg, best_m = max(results, key=lambda x: x[1]["profit_factor"])
        _print_report(best_cfg, best_m, initial_balance, years)
    else:
        cfg = DipHighConfig()
        print(f"\n🔬 Run unique : {cfg.name}...")
        results = _run_variants([cfg], all_candles, initial_balance, years)
        _print_report(cfg, results[0][1], initial_balance, years)


if __name__ == "__main__":
    main()
