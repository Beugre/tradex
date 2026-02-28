#!/usr/bin/env python
"""
Backtest comparatif : RANGE classique vs RANGE RSI+ATR+Trailing.

Variante légère :
  Entrée :
    - Dow Theory range (NEUTRAL) + prix dans la buy zone
    - RSI(14) < 35  (seul filtre technique ajouté)
  SL :
    - 1.3 × ATR(14) sous le swing low du range
  TP :
    - Pas de TP fixe — trailing stop pur
    - Trailing activé après trail_activation × ATR de profit
    - Trail distance = trail_atr_mult × ATR sous le plus haut

Comparaison A/B avec le Range classique (Dow Theory seul, TP = range_mid).
"""

from __future__ import annotations

import argparse
import logging
import math
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

from dotenv import load_dotenv
load_dotenv()

from src.core.models import (
    Candle,
    OrderSide,
    Position,
    PositionStatus,
    RangeState,
    StrategyType,
    TrendDirection,
    TrendState,
)
from src.core.swing_detector import detect_swings
from src.core.trend_engine import determine_trend
from src.core.strategy_mean_rev import build_range_from_trend
from src.core.risk_manager import calculate_position_size
from backtest.data_loader import download_all_pairs
from backtest.simulator import (
    BacktestConfig,
    BacktestEngine,
    BacktestResult,
    Trade,
    EquityPoint,
)
from backtest.metrics import compute_metrics

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("range_rsi")


# ════════════════════════════════════════════════════════════════════════════════
# Indicateurs techniques
# ════════════════════════════════════════════════════════════════════════════════


def rsi(closes: list[float], period: int = 14) -> Optional[float]:
    """Relative Strength Index (Wilder's smoothing)."""
    if len(closes) < period + 1:
        return None
    deltas = [closes[i] - closes[i - 1] for i in range(1, len(closes))]

    gains = [max(d, 0) for d in deltas[:period]]
    losses = [abs(min(d, 0)) for d in deltas[:period]]
    avg_gain = sum(gains) / period
    avg_loss = sum(losses) / period

    for d in deltas[period:]:
        avg_gain = (avg_gain * (period - 1) + max(d, 0)) / period
        avg_loss = (avg_loss * (period - 1) + abs(min(d, 0))) / period

    if avg_loss == 0:
        return 100.0
    rs = avg_gain / avg_loss
    return 100.0 - (100.0 / (1.0 + rs))


def atr(candles: list[Candle], period: int = 14) -> Optional[float]:
    """Average True Range (Wilder's smoothing)."""
    if len(candles) < period + 1:
        return None
    trs: list[float] = []
    for i in range(1, len(candles)):
        h = candles[i].high
        l = candles[i].low
        pc = candles[i - 1].close
        tr = max(h - l, abs(h - pc), abs(l - pc))
        trs.append(tr)

    a = sum(trs[:period]) / period
    for tr in trs[period:]:
        a = (a * (period - 1) + tr) / period
    return a


# ════════════════════════════════════════════════════════════════════════════════
# Configuration & structures
# ════════════════════════════════════════════════════════════════════════════════


@dataclass
class RangeRsiConfig:
    """Paramètres de la stratégie Range RSI + ATR SL + Trailing."""
    initial_balance: float = 1000.0

    # ── RSI ──
    rsi_period: int = 14
    rsi_threshold: float = 35.0

    # ── ATR ──
    atr_period: int = 14
    atr_sl_mult: float = 1.3          # SL = swing_low - 1.3 × ATR

    # ── Trailing stop ──
    trail_activation_atr: float = 1.0  # Activer trailing après 1×ATR de profit
    trail_distance_atr: float = 1.5    # Trail SL = highest - 1.5×ATR
    # Alternative : trailing en % (utilisé si trail_mode == "pct")
    trail_mode: str = "atr"            # "atr" ou "pct"
    trail_activation_pct: float = 0.01 # 1% de profit pour activer
    trail_distance_pct: float = 0.015  # 1.5% sous le plus haut

    # ── Risk ──
    risk_percent: float = 0.02
    max_position_pct: float = 0.30
    max_simultaneous_positions: int = 3
    fee_pct: float = 0.00075
    slippage_pct: float = 0.001

    # ── Range Dow ──
    swing_lookback: int = 3
    candle_window: int = 100
    range_width_min: float = 0.02
    range_entry_buffer_pct: float = 0.002

    # ── Compound ──
    compound: bool = False


@dataclass
class RsiTrade:
    """Trade terminé."""
    symbol: str
    side: OrderSide
    entry_price: float
    exit_price: float
    size: float
    entry_time: int
    exit_time: int
    pnl_usd: float
    pnl_pct: float
    exit_reason: str
    rsi_at_entry: Optional[float] = None
    atr_at_entry: Optional[float] = None
    highest_reached: float = 0.0       # plus haut atteint pendant le trade
    bars_held: int = 0                 # nombre de bougies dans le trade


@dataclass
class RsiPosition:
    symbol: str
    side: OrderSide
    entry_price: float
    sl_price: float
    size: float
    entry_time: int
    atr_at_entry: float
    # Trailing
    trailing_active: bool = False
    trailing_sl: Optional[float] = None
    highest_since_entry: float = 0.0
    bars_count: int = 0
    # Indicateurs snapshot
    rsi_at_entry: Optional[float] = None


# ════════════════════════════════════════════════════════════════════════════════
# Moteur de backtest RANGE RSI
# ════════════════════════════════════════════════════════════════════════════════


class RangeRsiEngine:
    """Backtest RANGE + RSI filter + ATR SL + Trailing stop."""

    def __init__(
        self,
        candles_by_symbol: dict[str, list[Candle]],
        config: RangeRsiConfig,
    ) -> None:
        self.candles = candles_by_symbol
        self.cfg = config
        self.pairs = list(candles_by_symbol.keys())

        self.cash: float = config.initial_balance
        self.positions: list[RsiPosition] = []
        self.closed_trades: list[RsiTrade] = []
        self.equity_curve: list[EquityPoint] = []

        self.trends: dict[str, Optional[TrendState]] = {p: None for p in self.pairs}
        self.ranges: dict[str, Optional[RangeState]] = {p: None for p in self.pairs}

        self._pending: dict[str, Optional[dict]] = {p: None for p in self.pairs}
        self.last_close: dict[str, float] = {}

        self._idx: dict[tuple[str, int], Candle] = {}
        for sym, clist in candles_by_symbol.items():
            for c in clist:
                self._idx[(sym, c.timestamp)] = c

    def run(self) -> BacktestResult:
        for mod in (
            "src.core.swing_detector",
            "src.core.trend_engine",
            "src.core.strategy_mean_rev",
        ):
            logging.getLogger(mod).setLevel(logging.WARNING)

        timeline = self._build_timeline()
        total = len(timeline)
        logger.info(
            "📊 Range RSI : %d barres, %d paires, $%.0f",
            total, len(self.pairs), self.cfg.initial_balance,
        )

        for i, ts in enumerate(timeline):
            for sym in self.pairs:
                c = self._idx.get((sym, ts))
                if c:
                    self.last_close[sym] = c.close

            # Ordre anti-lookahead :
            # 1) Gérer exits (SL/trailing)
            # 2) Exécuter signaux pending
            # 3) Analyser et générer nouveaux signaux
            self._manage_exits(ts)
            self._execute_pending(ts)
            self._analyze(ts)
            self._record_equity(ts)

            if (i + 1) % 500 == 0 or i == total - 1:
                eq = self.equity_curve[-1].equity
                print(
                    f"\r   ⏳ {i+1}/{total} ({100*(i+1)/total:.0f}%) "
                    f"| Equity: ${eq:,.2f} | Trades: {len(self.closed_trades)}",
                    end="", flush=True,
                )
        print()

        self._close_remaining(timeline[-1])
        final_eq = self.equity_curve[-1].equity if self.equity_curve else self.cash

        trades = [
            Trade(
                symbol=t.symbol,
                strategy=StrategyType.RANGE,
                side=t.side,
                entry_price=t.entry_price,
                exit_price=t.exit_price,
                size=t.size,
                entry_time=t.entry_time,
                exit_time=t.exit_time,
                pnl_usd=t.pnl_usd,
                pnl_pct=t.pnl_pct,
                exit_reason=t.exit_reason,
            )
            for t in self.closed_trades
        ]

        return BacktestResult(
            trades=trades,
            equity_curve=self.equity_curve,
            initial_balance=self.cfg.initial_balance,
            final_equity=final_eq,
            start_date=datetime.fromtimestamp(timeline[0] / 1000, tz=timezone.utc),
            end_date=datetime.fromtimestamp(timeline[-1] / 1000, tz=timezone.utc),
            pairs=self.pairs,
        )

    def _build_timeline(self) -> list[int]:
        ts_set: set[int] = set()
        for clist in self.candles.values():
            for c in clist:
                ts_set.add(c.timestamp)
        return sorted(ts_set)

    def _visible(self, symbol: str, up_to_ts: int) -> list[Candle]:
        clist = self.candles[symbol]
        vis = [c for c in clist if c.timestamp <= up_to_ts]
        return vis[-self.cfg.candle_window:]

    # ── Gestion des exits ──────────────────────────────────────────────────

    def _manage_exits(self, ts: int) -> None:
        to_close: list[tuple[RsiPosition, float, str]] = []

        for pos in self.positions:
            c = self._idx.get((pos.symbol, ts))
            if c is None:
                continue

            pos.bars_count += 1

            # Track highest
            if c.high > pos.highest_since_entry:
                pos.highest_since_entry = c.high

            # ── SL check (prioritaire — worst-case) ──
            eff_sl = pos.trailing_sl if pos.trailing_active and pos.trailing_sl else pos.sl_price
            if c.low <= eff_sl:
                reason = "TRAILING_SL" if pos.trailing_active else "RSI_SL"
                to_close.append((pos, eff_sl, reason))
                continue

            # ── Trailing stop management ──
            if self.cfg.trail_mode == "atr":
                activation_dist = self.cfg.trail_activation_atr * pos.atr_at_entry
                trail_dist = self.cfg.trail_distance_atr * pos.atr_at_entry
            else:  # pct
                activation_dist = pos.entry_price * self.cfg.trail_activation_pct
                trail_dist = pos.highest_since_entry * self.cfg.trail_distance_pct

            # Activer le trailing
            if not pos.trailing_active:
                profit = c.close - pos.entry_price
                if profit >= activation_dist:
                    pos.trailing_active = True
                    if self.cfg.trail_mode == "atr":
                        pos.trailing_sl = pos.highest_since_entry - trail_dist
                    else:
                        pos.trailing_sl = pos.highest_since_entry * (1 - self.cfg.trail_distance_pct)
                    # Ne pas descendre sous le SL initial
                    if pos.trailing_sl < pos.sl_price:
                        pos.trailing_sl = pos.sl_price

            # Mettre à jour le trailing (ne remonte jamais)
            if pos.trailing_active and pos.trailing_sl:
                if self.cfg.trail_mode == "atr":
                    new_trail = pos.highest_since_entry - trail_dist
                else:
                    new_trail = pos.highest_since_entry * (1 - self.cfg.trail_distance_pct)

                if new_trail > pos.trailing_sl:
                    pos.trailing_sl = new_trail

        for pos, exit_px, reason in to_close:
            self._close_position(pos, exit_px, ts, reason)

    # ── Exécuter les signaux pending ───────────────────────────────────────

    def _execute_pending(self, ts: int) -> None:
        if len(self.positions) >= self.cfg.max_simultaneous_positions:
            return

        for sym in self.pairs:
            if len(self.positions) >= self.cfg.max_simultaneous_positions:
                break
            if any(p.symbol == sym for p in self.positions):
                continue

            sig = self._pending.get(sym)
            if sig is None:
                continue

            c = self._idx.get((sym, ts))
            if c is None:
                continue

            # Entrer si open est dans la buy zone (anti-lookahead)
            if c.open > sig["buy_zone"]:
                continue

            entry_price = c.open * (1 + self.cfg.slippage_pct)

            if sig["sl_price"] >= entry_price:
                continue

            # Sizing
            sizing_balance = self.cash if self.cfg.compound else self.cfg.initial_balance
            size = calculate_position_size(
                account_balance=sizing_balance,
                risk_percent=self.cfg.risk_percent,
                entry_price=entry_price,
                sl_price=sig["sl_price"],
                max_position_percent=self.cfg.max_position_pct,
            )
            if size <= 0:
                continue

            cost = size * entry_price * (1 + self.cfg.fee_pct)
            if cost > self.cash:
                size = self.cash / (entry_price * (1 + self.cfg.fee_pct))
                cost = size * entry_price * (1 + self.cfg.fee_pct)
            if size <= 0 or cost > self.cash:
                continue

            self.cash -= cost
            pos = RsiPosition(
                symbol=sym,
                side=OrderSide.BUY,
                entry_price=entry_price,
                sl_price=sig["sl_price"],
                size=size,
                entry_time=ts,
                atr_at_entry=sig["atr"],
                highest_since_entry=entry_price,
                rsi_at_entry=sig.get("rsi"),
            )
            self.positions.append(pos)
            self._pending[sym] = None

    # ── Analyse : Dow range + RSI ──────────────────────────────────────────

    def _analyze(self, ts: int) -> None:
        for sym in self.pairs:
            vis = self._visible(sym, ts)
            min_len = max(
                2 * self.cfg.swing_lookback + 1,
                self.cfg.rsi_period + 1,
                self.cfg.atr_period + 1,
            )
            if len(vis) < min_len:
                continue

            # ── 1) Dow Theory : détecter le range ──
            swings = detect_swings(vis, self.cfg.swing_lookback)
            if not swings:
                continue
            trend = determine_trend(swings, sym)
            self.trends[sym] = trend

            # Invalider le range si tendance non-NEUTRAL
            if trend.direction != TrendDirection.NEUTRAL:
                self.ranges[sym] = None
                self._pending[sym] = None
                # Fermer les positions (trend breakout)
                for pos in list(self.positions):
                    if pos.symbol == sym:
                        px = self.last_close.get(sym, pos.entry_price)
                        self._close_position(pos, px, ts, "TREND_BREAK")
                continue

            # Construire le range
            rs = build_range_from_trend(trend, self.cfg.range_width_min)
            if rs is None:
                self._pending[sym] = None
                continue
            self.ranges[sym] = rs

            # ── 2) Calculer RSI + ATR ──
            closes = [c.close for c in vis]
            rsi_val = rsi(closes, self.cfg.rsi_period)
            atr_val = atr(vis, self.cfg.atr_period)

            if rsi_val is None or atr_val is None:
                continue

            # ── 3) Conditions d'entrée : buy zone + RSI < seuil ──
            buy_zone = rs.range_low * (1 + self.cfg.range_entry_buffer_pct)
            current_candle = vis[-1]

            # Prix dans la buy zone (proche du range_low)
            in_buy_zone = current_candle.close <= buy_zone

            # RSI survendu
            rsi_ok = rsi_val < self.cfg.rsi_threshold

            if in_buy_zone and rsi_ok:
                # SL = swing_low - 1.3 × ATR
                sl_price = rs.range_low - self.cfg.atr_sl_mult * atr_val

                self._pending[sym] = {
                    "side": OrderSide.BUY,
                    "buy_zone": buy_zone,
                    "sl_price": sl_price,
                    "rsi": rsi_val,
                    "atr": atr_val,
                }
            else:
                self._pending[sym] = None

    # ── Close position ─────────────────────────────────────────────────────

    def _close_position(
        self, pos: RsiPosition, exit_price: float, ts: int, reason: str,
    ) -> None:
        revenue = pos.size * exit_price * (1 - self.cfg.fee_pct)
        pnl = pos.size * (
            exit_price * (1 - self.cfg.fee_pct)
            - pos.entry_price * (1 + self.cfg.fee_pct)
        )
        self.cash += revenue
        pnl_pct = pnl / (pos.size * pos.entry_price) if pos.entry_price else 0

        self.closed_trades.append(RsiTrade(
            symbol=pos.symbol,
            side=pos.side,
            entry_price=pos.entry_price,
            exit_price=exit_price,
            size=pos.size,
            entry_time=pos.entry_time,
            exit_time=ts,
            pnl_usd=pnl,
            pnl_pct=pnl_pct,
            exit_reason=reason,
            rsi_at_entry=pos.rsi_at_entry,
            atr_at_entry=pos.atr_at_entry,
            highest_reached=pos.highest_since_entry,
            bars_held=pos.bars_count,
        ))
        if pos in self.positions:
            self.positions.remove(pos)

    def _close_remaining(self, last_ts: int) -> None:
        for pos in list(self.positions):
            px = self.last_close.get(pos.symbol, pos.entry_price)
            self._close_position(pos, px, last_ts, "END")

    def _record_equity(self, ts: int) -> None:
        unrealized = sum(
            p.size * self.last_close.get(p.symbol, p.entry_price)
            for p in self.positions
        )
        self.equity_curve.append(EquityPoint(ts, self.cash + unrealized))


# ════════════════════════════════════════════════════════════════════════════════
# Rapport comparatif
# ════════════════════════════════════════════════════════════════════════════════


def print_comparison(
    m_classic: dict,
    m_rsi: dict,
    res_classic: BacktestResult,
    res_rsi: BacktestResult,
    rsi_trades: list[RsiTrade],
) -> None:
    """Affiche un tableau comparatif RANGE classique vs RANGE RSI+Trailing."""
    sep = "═" * 80
    print(f"\n{sep}")
    print("  🔬 COMPARAISON : RANGE CLASSIQUE vs RANGE RSI+ATR+Trailing")
    print(f"  📅 {res_classic.start_date:%b %Y} → {res_classic.end_date:%b %Y}")
    print(f"  🪙 {len(res_classic.pairs)} paires | Capital: ${res_classic.initial_balance:,.0f}")
    print(sep)

    metrics_list = [
        ("Return", "total_return", "{:+.1%}"),
        ("CAGR", "cagr", "{:+.1%}"),
        ("Max Drawdown", "max_drawdown", "{:.1%}"),
        ("Sharpe", "sharpe", "{:.2f}"),
        ("Sortino", "sortino", "{:.2f}"),
        ("Win Rate", "win_rate", "{:.1%}"),
        ("Profit Factor", "profit_factor", "{:.2f}"),
        ("Trades", "n_trades", "{:d}"),
        ("PnL moyen ($)", "avg_pnl_usd", "{:+.2f}"),
        ("PnL moyen (%)", "avg_pnl_pct", "{:+.2%}"),
        ("Capital final", "final_equity", "${:,.2f}"),
    ]

    print(f"\n  {'Métrique':<22s} │ {'CLASSIQUE':>14s} │ {'RSI+Trail':>14s} │ {'Δ':>10s}")
    print("  " + "─" * 70)

    for label, key, fmt in metrics_list:
        v_c = m_classic.get(key, 0)
        v_r = m_rsi.get(key, 0)

        s_c = fmt.format(v_c) if isinstance(v_c, (int, float)) else str(v_c)
        s_r = fmt.format(v_r) if isinstance(v_r, (int, float)) else str(v_r)

        if isinstance(v_c, (int, float)) and isinstance(v_r, (int, float)):
            delta = v_r - v_c
            if key in ("total_return", "cagr", "max_drawdown", "win_rate", "avg_pnl_pct"):
                s_d = f"{delta:+.1%}"
            elif key == "n_trades":
                s_d = f"{delta:+d}"
            else:
                s_d = f"{delta:+.2f}"
        else:
            s_d = ""

        better = ""
        if key in ("max_drawdown",):
            better = "✅" if v_r > v_c else ("❌" if v_r < v_c else "")
        elif key in ("total_return", "cagr", "sharpe", "sortino", "win_rate",
                      "profit_factor", "final_equity", "avg_pnl_usd", "avg_pnl_pct"):
            better = "✅" if v_r > v_c else ("❌" if v_r < v_c else "")

        print(f"  {label:<22s} │ {s_c:>14s} │ {s_r:>14s} │ {s_d:>8s} {better}")

    # Stats additionnelles RSI
    if rsi_trades:
        print(f"\n  📊 Détail RANGE RSI+Trailing")
        print("  " + "─" * 70)

        # Par motif de sortie
        exit_reasons: dict[str, list[RsiTrade]] = {}
        for t in rsi_trades:
            exit_reasons.setdefault(t.exit_reason, []).append(t)

        for reason, tlist in sorted(exit_reasons.items()):
            n = len(tlist)
            wins = sum(1 for t in tlist if t.pnl_usd > 0)
            pnl = sum(t.pnl_usd for t in tlist)
            wr = wins / n if n else 0
            avg_bars = sum(t.bars_held for t in tlist) / n if n else 0
            print(
                f"  {reason:16s} : {n:3d} trades | WR {wr:.0%} | "
                f"PnL ${pnl:+.2f} | Avg {avg_bars:.0f} barres"
            )

        # Indicateurs moyens
        rsi_vals = [t.rsi_at_entry for t in rsi_trades if t.rsi_at_entry is not None]
        atr_vals = [t.atr_at_entry for t in rsi_trades if t.atr_at_entry is not None]
        if rsi_vals:
            print(f"\n  RSI moyen à l'entrée  : {sum(rsi_vals)/len(rsi_vals):.1f}")
        if atr_vals:
            print(f"  ATR moyen à l'entrée  : {sum(atr_vals)/len(atr_vals):.4f}")

        # Stats trailing
        trail_trades = [t for t in rsi_trades if t.exit_reason == "TRAILING_SL"]
        if trail_trades:
            avg_bars_trail = sum(t.bars_held for t in trail_trades) / len(trail_trades)
            avg_pnl_trail = sum(t.pnl_pct for t in trail_trades) / len(trail_trades)
            max_gain = max(t.pnl_pct for t in trail_trades) if trail_trades else 0
            print(f"\n  🎯 Trades avec trailing stop :")
            print(f"     Durée moyenne     : {avg_bars_trail:.0f} bougies ({avg_bars_trail*4:.0f}h)")
            print(f"     PnL moyen         : {avg_pnl_trail:+.2%}")
            print(f"     Meilleur trade    : {max_gain:+.2%}")

        # Top 5 trades
        sorted_trades = sorted(rsi_trades, key=lambda t: t.pnl_usd, reverse=True)
        print(f"\n  🏆 Top 5 trades :")
        for t in sorted_trades[:5]:
            print(
                f"     {t.symbol:10s} | {t.pnl_usd:+8.2f} USD ({t.pnl_pct:+.1%}) | "
                f"{t.bars_held} barres | RSI={t.rsi_at_entry:.0f} | "
                f"Exit: {t.exit_reason}"
            )

        print(f"\n  💩 Bottom 5 trades :")
        for t in sorted_trades[-5:]:
            print(
                f"     {t.symbol:10s} | {t.pnl_usd:+8.2f} USD ({t.pnl_pct:+.1%}) | "
                f"{t.bars_held} barres | RSI={t.rsi_at_entry:.0f} | "
                f"Exit: {t.exit_reason}"
            )

    print(f"\n{sep}\n")


# ════════════════════════════════════════════════════════════════════════════════
# Main
# ════════════════════════════════════════════════════════════════════════════════


PAIRS_20 = [
    "BTC-USD", "ETH-USD", "SOL-USD", "XRP-USD", "LINK-USD",
    "ADA-USD", "DOT-USD", "AVAX-USD", "DOGE-USD", "ATOM-USD",
    "UNI-USD", "NEAR-USD", "ALGO-USD", "LTC-USD", "ETC-USD",
    "FIL-USD", "AAVE-USD", "INJ-USD", "SAND-USD", "MANA-USD",
]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="TradeX Backtest — RANGE classique vs RANGE RSI+ATR SL+Trailing"
    )
    parser.add_argument("--balance", type=float, default=1000.0)
    parser.add_argument("--start", default="2022-02-20")
    parser.add_argument("--end", default="2026-02-20")
    parser.add_argument("--pairs", type=int, default=20)
    # RSI
    parser.add_argument("--rsi-threshold", type=float, default=35.0)
    # ATR SL
    parser.add_argument("--atr-sl-mult", type=float, default=1.3)
    # Trailing
    parser.add_argument("--trail-mode", choices=["atr", "pct"], default="atr",
                        help="Mode trailing : ATR ou pourcentage")
    parser.add_argument("--trail-activation-atr", type=float, default=1.0,
                        help="ATR multiples avant activation trailing")
    parser.add_argument("--trail-distance-atr", type=float, default=1.5,
                        help="ATR multiples pour distance trailing")
    parser.add_argument("--trail-activation-pct", type=float, default=0.01,
                        help="% profit avant activation (mode pct)")
    parser.add_argument("--trail-distance-pct", type=float, default=0.015,
                        help="% sous le plus haut (mode pct)")
    parser.add_argument("--no-show", action="store_true")
    args = parser.parse_args()

    start = datetime.strptime(args.start, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    end = datetime.strptime(args.end, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    pairs = PAIRS_20[:args.pairs]

    trail_desc = (
        f"Trail ATR: activation={args.trail_activation_atr}×ATR, "
        f"distance={args.trail_distance_atr}×ATR"
        if args.trail_mode == "atr"
        else f"Trail %: activation={args.trail_activation_pct:.1%}, "
             f"distance={args.trail_distance_pct:.1%}"
    )

    print(f"\n{'═'*80}")
    print(f"  🔬 RANGE CLASSIQUE vs RANGE RSI+ATR+Trailing — A/B Test")
    print(f"  📅 {start:%Y-%m-%d} → {end:%Y-%m-%d} | {len(pairs)} paires | ${args.balance:,.0f}")
    print(f"  RSI < {args.rsi_threshold} | SL = swing_low - {args.atr_sl_mult}×ATR")
    print(f"  {trail_desc}")
    print(f"{'═'*80}\n")

    # ── Télécharger les données ──
    logger.info("📥 Téléchargement des données…")
    candles = download_all_pairs(pairs, start, end, interval="4h")

    # ═══════════════════════════════════════════════════════════════════════
    # A) RANGE CLASSIQUE (Dow Theory seul, TP = range_mid)
    # ═══════════════════════════════════════════════════════════════════════
    print("\n" + "─" * 60)
    print("  🅰️  RANGE CLASSIQUE (Dow Theory, TP=range_mid)")
    print("─" * 60)

    from src import config
    cfg_classic = BacktestConfig(
        initial_balance=args.balance,
        risk_percent_range=0.02,
        entry_buffer_pct=getattr(config, "ENTRY_BUFFER_PERCENT", 0.002),
        sl_buffer_pct=getattr(config, "SL_BUFFER_PERCENT", 0.003),
        zero_risk_trigger_pct=getattr(config, "ZERO_RISK_TRIGGER_PERCENT", 0.02),
        zero_risk_lock_pct=getattr(config, "ZERO_RISK_LOCK_PERCENT", 0.005),
        trailing_stop_pct=getattr(config, "TRAILING_STOP_PERCENT", 0.02),
        max_position_pct=0.30,
        max_simultaneous_positions=3,
        swing_lookback=getattr(config, "SWING_LOOKBACK", 3),
        range_width_min=getattr(config, "RANGE_WIDTH_MIN", 0.02),
        range_entry_buffer_pct=getattr(config, "RANGE_ENTRY_BUFFER_PERCENT", 0.002),
        range_sl_buffer_pct=getattr(config, "RANGE_SL_BUFFER_PERCENT", 0.003),
        range_cooldown_bars=getattr(config, "RANGE_COOLDOWN_BARS", 3),
        enable_trend=False,
        enable_range=True,
    )
    engine_classic = BacktestEngine(candles, cfg_classic)
    result_classic = engine_classic.run()
    m_classic = compute_metrics(result_classic)

    # ═══════════════════════════════════════════════════════════════════════
    # B) RANGE RSI + ATR SL + Trailing
    # ═══════════════════════════════════════════════════════════════════════
    print("\n" + "─" * 60)
    print("  🅱️  RANGE RSI + ATR SL + Trailing")
    print("─" * 60)

    cfg_rsi = RangeRsiConfig(
        initial_balance=args.balance,
        rsi_threshold=args.rsi_threshold,
        atr_sl_mult=args.atr_sl_mult,
        trail_mode=args.trail_mode,
        trail_activation_atr=args.trail_activation_atr,
        trail_distance_atr=args.trail_distance_atr,
        trail_activation_pct=args.trail_activation_pct,
        trail_distance_pct=args.trail_distance_pct,
    )
    engine_rsi = RangeRsiEngine(candles, cfg_rsi)
    result_rsi = engine_rsi.run()
    m_rsi = compute_metrics(result_rsi)

    # ═══════════════════════════════════════════════════════════════════════
    # Comparaison A/B
    # ═══════════════════════════════════════════════════════════════════════
    print_comparison(m_classic, m_rsi, result_classic, result_rsi, engine_rsi.closed_trades)

    # ── Graphiques ──
    from backtest.report import generate_report
    print("  📊 Graphique CLASSIQUE :")
    generate_report(result_classic, m_classic, show=False)
    print("  📊 Graphique RSI+Trailing :")
    generate_report(result_rsi, m_rsi, show=not args.no_show)


if __name__ == "__main__":
    main()
