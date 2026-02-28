#!/usr/bin/env python
"""
Backtest RSI Bot — Mean Reversion pure.

🧠 Logique :
   LONG  : RSI < oversold  (ex: 30) → acheter le dip, sortir quand RSI remonte
   SHORT : RSI > overbought (ex: 70) → shorter le sommet, sortir quand RSI redescend

   SL : ATR × mult
   TP : ATR × mult OU RSI revient en zone neutre
   Trail optionnel après TP

   Modes : LONG, SHORT, BIDIR (les deux)

Usage :
  PYTHONPATH=. .venv/bin/python backtest/run_backtest_rsi.py --no-show
  PYTHONPATH=. .venv/bin/python backtest/run_backtest_rsi.py --months 24 --sensitivity --no-show
  PYTHONPATH=. .venv/bin/python backtest/run_backtest_rsi.py --mode SHORT --months 24 --no-show
"""

from __future__ import annotations

import argparse
import logging
import sys
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
load_dotenv()

from src.core.models import (
    Candle,
    OrderSide,
    StrategyType,
)
from backtest.data_loader import download_all_pairs, download_btc_d1
from backtest.simulator import BacktestResult, Trade, EquityPoint
from backtest.metrics import compute_metrics

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("rsi_bot")

OUTPUT_DIR = Path(__file__).parent / "output"


# ════════════════════════════════════════════════════════════════════════════════
# Indicateurs
# ════════════════════════════════════════════════════════════════════════════════


def compute_rsi(closes: list[float], period: int = 14) -> Optional[float]:
    """RSI Wilder classique."""
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


def compute_atr(candles: list[Candle], period: int = 14) -> Optional[float]:
    if len(candles) < period + 1:
        return None
    trs: list[float] = []
    for i in range(1, len(candles)):
        h, l, pc = candles[i].high, candles[i].low, candles[i - 1].close
        trs.append(max(h - l, abs(h - pc), abs(l - pc)))
    a = sum(trs[:period]) / period
    for tr in trs[period:]:
        a = (a * (period - 1) + tr) / period
    return a


# ════════════════════════════════════════════════════════════════════════════════
# Config
# ════════════════════════════════════════════════════════════════════════════════


class RSIMode(Enum):
    LONG = "LONG"
    SHORT = "SHORT"
    BIDIR = "BIDIR"


@dataclass
class RSIConfig:
    initial_balance: float = 1000.0
    mode: RSIMode = RSIMode.BIDIR

    # ── RSI ──
    rsi_period: int = 14
    rsi_oversold: float = 30.0       # LONG quand RSI < ce seuil
    rsi_overbought: float = 70.0     # SHORT quand RSI > ce seuil
    rsi_exit_long: float = 55.0      # Sortie LONG quand RSI repasse au-dessus
    rsi_exit_short: float = 45.0     # Sortie SHORT quand RSI repasse en-dessous

    # ── Gestion de position ──
    atr_period: int = 14
    sl_atr_mult: float = 1.5         # SL = entry ± ATR × mult
    tp_atr_mult: float = 2.5         # TP = entry ± ATR × mult
    use_rsi_exit: bool = True         # Sortir aussi sur RSI neutre (pas seulement SL/TP)
    trail_after_tp: bool = True       # Trailing après que le TP soit touché
    trail_atr_mult: float = 2.0       # Distance trailing = ATR × mult

    # ── Risk ──
    risk_percent: float = 0.02
    max_position_pct: float = 0.30
    max_simultaneous: int = 3
    fee_pct: float = 0.00075
    slippage_pct: float = 0.001
    margin_fee_pct: float = 0.0002    # Pour les shorts seulement

    # ── Cooldown ──
    cooldown_bars: int = 3            # Barres d'attente entre trades sur même paire
    max_consec_losses: int = 3
    cooldown_after_losses: int = 6

    # ── Macro filter ──
    use_macro_filter: bool = False    # BTC EMA200 D1 (bear → short ok, bull → long ok)
    macro_ema_period: int = 200

    # ── Fenêtre ──
    candle_window: int = 80


# ════════════════════════════════════════════════════════════════════════════════
# Structures
# ════════════════════════════════════════════════════════════════════════════════


@dataclass
class RSIPosition:
    symbol: str
    side: str               # "LONG" ou "SHORT"
    entry_price: float
    sl_price: float
    tp_price: float
    size: float
    entry_time: int
    rsi_at_entry: float
    best_price: float = 0.0  # meilleur prix en faveur
    is_trailing: bool = False
    trailing_sl: float = 0.0


@dataclass
class RSITrade:
    symbol: str
    side: str
    entry_price: float
    exit_price: float
    size: float
    entry_time: int
    exit_time: int
    pnl_usd: float
    pnl_pct: float
    exit_reason: str
    rsi_at_entry: float


# ════════════════════════════════════════════════════════════════════════════════
# Moteur RSI
# ════════════════════════════════════════════════════════════════════════════════


class RSIEngine:
    """
    Bot RSI mean-reversion pur.
    LONG quand RSI < oversold, SHORT quand RSI > overbought.
    """

    def __init__(
        self,
        candles_by_symbol: dict[str, list[Candle]],
        config: RSIConfig,
        btc_d1_candles: list[Candle] | None = None,
    ) -> None:
        self.candles = candles_by_symbol
        self.cfg = config
        self.pairs = list(candles_by_symbol.keys())

        self.cash: float = config.initial_balance
        self.positions: list[RSIPosition] = []
        self.closed_trades: list[RSITrade] = []
        self.equity_curve: list[EquityPoint] = []

        self._consec_losses: dict[str, int] = {p: 0 for p in self.pairs}
        self._cooldown_until: dict[str, int] = {p: 0 for p in self.pairs}
        self._last_trade_ts: dict[str, int] = {p: 0 for p in self.pairs}
        self.last_close: dict[str, float] = {}

        # Index rapide
        self._idx: dict[tuple[str, int], Candle] = {}
        for sym, clist in candles_by_symbol.items():
            for c in clist:
                self._idx[(sym, c.timestamp)] = c

        # ── Macro filter BTC EMA200 D1 ──
        self._ema_mode: dict[int, bool] = {}
        self._ema_ts_sorted: list[int] = []
        self._btc_bullish: bool = True  # par défaut bull
        if config.use_macro_filter and btc_d1_candles:
            self._build_ema_lookup(btc_d1_candles)

    # ── Macro EMA200 ───────────────────────────────────────────────────────

    def _build_ema_lookup(self, d1_candles: list[Candle]) -> None:
        period = self.cfg.macro_ema_period
        if len(d1_candles) < period:
            logger.warning("⚠️ %d bougies D1 < EMA%d — filtre désactivé", len(d1_candles), period)
            return
        closes = [c.close for c in d1_candles]
        sma_seed = sum(closes[:period]) / period
        k = 2.0 / (period + 1)
        ema_val = sma_seed
        for i in range(period, len(closes)):
            ema_val = closes[i] * k + ema_val * (1 - k)
            ts = d1_candles[i].timestamp
            self._ema_mode[ts] = closes[i] > ema_val
        self._ema_ts_sorted = sorted(self._ema_mode.keys())
        n_bull = sum(1 for v in self._ema_mode.values() if v)
        n_bear = len(self._ema_mode) - n_bull
        logger.info("📊 Macro EMA%d : %d j BULL, %d j BEAR", period, n_bull, n_bear)

    def _update_macro(self, ts: int) -> None:
        if not self._ema_mode:
            return
        import bisect
        idx = bisect.bisect_right(self._ema_ts_sorted, ts) - 1
        if idx >= 0:
            self._btc_bullish = self._ema_mode[self._ema_ts_sorted[idx]]

    # ── Run ─────────────────────────────────────────────────────────────────

    def run(self) -> BacktestResult:
        timeline = self._build_timeline()
        total = len(timeline)
        logger.info(
            "📊 RSI Bot [%s] : %d barres, %d paires, $%.0f",
            self.cfg.mode.value, total, len(self.pairs), self.cfg.initial_balance,
        )

        for i, ts in enumerate(timeline):
            for sym in self.pairs:
                c = self._idx.get((sym, ts))
                if c:
                    self.last_close[sym] = c.close

            self._update_macro(ts)
            self._manage_exits(ts)
            self._analyze(ts)
            self._record_equity(ts)

            if (i + 1) % 500 == 0 or i == total - 1:
                eq = self.equity_curve[-1].equity
                print(
                    f"\r   ⏳ {i+1}/{total} ({100*(i+1)/total:.0f}%) "
                    f"| Eq: ${eq:,.2f} | Trades: {len(self.closed_trades)} "
                    f"| Pos: {len(self.positions)}",
                    end="", flush=True,
                )
        print()

        self._close_remaining(timeline[-1])
        final_eq = self.equity_curve[-1].equity if self.equity_curve else self.cash

        trades = [
            Trade(
                symbol=t.symbol,
                strategy=StrategyType.BREAKOUT,
                side=OrderSide.BUY if t.side == "LONG" else OrderSide.SELL,
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

    # ── Helpers ────────────────────────────────────────────────────────────

    def _build_timeline(self) -> list[int]:
        ts_set: set[int] = set()
        for clist in self.candles.values():
            for c in clist:
                ts_set.add(c.timestamp)
        return sorted(ts_set)

    def _visible(self, symbol: str, ts: int) -> list[Candle]:
        clist = self.candles[symbol]
        vis = [c for c in clist if c.timestamp <= ts]
        return vis[-self.cfg.candle_window:]

    # ── Exits ──────────────────────────────────────────────────────────────

    def _manage_exits(self, ts: int) -> None:
        to_close: list[tuple[RSIPosition, float, str]] = []

        for pos in self.positions:
            c = self._idx.get((pos.symbol, ts))
            if c is None:
                continue

            is_long = pos.side == "LONG"

            # Track best price
            if is_long:
                pos.best_price = max(pos.best_price, c.high)
            else:
                pos.best_price = min(pos.best_price, c.low) if pos.best_price > 0 else c.low

            # ── SL ──
            if is_long and c.low <= pos.sl_price:
                to_close.append((pos, pos.sl_price, "SL"))
                continue
            if not is_long and c.high >= pos.sl_price:
                to_close.append((pos, pos.sl_price, "SL"))
                continue

            # ── TP / Trailing ──
            if is_long and not pos.is_trailing and c.high >= pos.tp_price:
                if self.cfg.trail_after_tp:
                    pos.is_trailing = True
                    vis = self._visible(pos.symbol, ts)
                    atr_val = compute_atr(vis, self.cfg.atr_period) or (pos.entry_price * 0.02)
                    pos.trailing_sl = c.high - atr_val * self.cfg.trail_atr_mult
                    pos.trailing_sl = max(pos.trailing_sl, pos.entry_price)  # au moins breakeven
                    continue
                else:
                    to_close.append((pos, pos.tp_price, "TP"))
                    continue

            if not is_long and not pos.is_trailing and c.low <= pos.tp_price:
                if self.cfg.trail_after_tp:
                    pos.is_trailing = True
                    vis = self._visible(pos.symbol, ts)
                    atr_val = compute_atr(vis, self.cfg.atr_period) or (pos.entry_price * 0.02)
                    pos.trailing_sl = c.low + atr_val * self.cfg.trail_atr_mult
                    pos.trailing_sl = min(pos.trailing_sl, pos.entry_price)
                    continue
                else:
                    to_close.append((pos, pos.tp_price, "TP"))
                    continue

            # ── Trailing stop ──
            if pos.is_trailing:
                vis = self._visible(pos.symbol, ts)
                atr_val = compute_atr(vis, self.cfg.atr_period) or (pos.entry_price * 0.02)
                trail_dist = atr_val * self.cfg.trail_atr_mult

                if is_long:
                    new_trail = pos.best_price - trail_dist
                    new_trail = max(new_trail, pos.entry_price)
                    if new_trail > pos.trailing_sl:
                        pos.trailing_sl = new_trail
                    if c.low <= pos.trailing_sl:
                        to_close.append((pos, pos.trailing_sl, "TRAIL"))
                        continue
                else:
                    new_trail = pos.best_price + trail_dist
                    new_trail = min(new_trail, pos.entry_price)
                    if new_trail < pos.trailing_sl:
                        pos.trailing_sl = new_trail
                    if c.high >= pos.trailing_sl:
                        to_close.append((pos, pos.trailing_sl, "TRAIL"))
                        continue

            # ── RSI exit (mean reversion complete) ──
            if self.cfg.use_rsi_exit:
                vis = self._visible(pos.symbol, ts)
                closes = [cc.close for cc in vis]
                rsi_val = compute_rsi(closes, self.cfg.rsi_period)
                if rsi_val is not None:
                    if is_long and rsi_val >= self.cfg.rsi_exit_long:
                        to_close.append((pos, c.close, "RSI_EXIT"))
                        continue
                    if not is_long and rsi_val <= self.cfg.rsi_exit_short:
                        to_close.append((pos, c.close, "RSI_EXIT"))
                        continue

        for pos, exit_px, reason in to_close:
            self._close_position(pos, exit_px, ts, reason)

    # ── Analyse — Entrées ──────────────────────────────────────────────────

    def _analyze(self, ts: int) -> None:
        if len(self.positions) >= self.cfg.max_simultaneous:
            return

        for sym in self.pairs:
            if len(self.positions) >= self.cfg.max_simultaneous:
                break
            if any(p.symbol == sym for p in self.positions):
                continue

            # Cooldown
            if ts < self._cooldown_until.get(sym, 0):
                continue
            bar_ms = 4 * 3600 * 1000
            if ts - self._last_trade_ts.get(sym, 0) < self.cfg.cooldown_bars * bar_ms:
                continue

            vis = self._visible(sym, ts)
            min_bars = max(self.cfg.rsi_period + 2, self.cfg.atr_period + 2)
            if len(vis) < min_bars:
                continue

            closes = [c.close for c in vis]
            rsi_val = compute_rsi(closes, self.cfg.rsi_period)
            if rsi_val is None:
                continue

            atr_val = compute_atr(vis, self.cfg.atr_period)
            if atr_val is None or atr_val <= 0:
                continue

            current = vis[-1]

            # ── LONG : RSI oversold ──
            if (
                self.cfg.mode in (RSIMode.LONG, RSIMode.BIDIR)
                and rsi_val < self.cfg.rsi_oversold
            ):
                # Macro : en mode macro, ne pas longer si BTC bearish ?
                # En fait pour mean reversion on peut longer même en bear (c'est un rebond)
                # Mais optionnellement on peut filtrer
                if self.cfg.use_macro_filter and not self._btc_bullish:
                    pass  # skip longs en bear si macro filter actif
                else:
                    entry_px = current.close * (1 + self.cfg.slippage_pct)
                    sl_px = entry_px - atr_val * self.cfg.sl_atr_mult
                    tp_px = entry_px + atr_val * self.cfg.tp_atr_mult

                    if self._open_position(sym, "LONG", entry_px, sl_px, tp_px,
                                           atr_val, rsi_val, ts):
                        continue

            # ── SHORT : RSI overbought ──
            if (
                self.cfg.mode in (RSIMode.SHORT, RSIMode.BIDIR)
                and rsi_val > self.cfg.rsi_overbought
            ):
                if self.cfg.use_macro_filter and self._btc_bullish:
                    pass  # skip shorts en bull si macro filter actif
                else:
                    entry_px = current.close * (1 - self.cfg.slippage_pct)
                    sl_px = entry_px + atr_val * self.cfg.sl_atr_mult
                    tp_px = entry_px - atr_val * self.cfg.tp_atr_mult

                    self._open_position(sym, "SHORT", entry_px, sl_px, tp_px,
                                        atr_val, rsi_val, ts)

    # ── Open / Close position ──────────────────────────────────────────────

    def _open_position(
        self, sym: str, side: str, entry_px: float,
        sl_px: float, tp_px: float, atr_val: float,
        rsi_val: float, ts: int,
    ) -> bool:
        is_long = side == "LONG"
        sl_dist = abs(entry_px - sl_px)
        if sl_dist <= 0:
            return False

        sizing_balance = self.cfg.initial_balance  # pas de compound
        risk_amount = sizing_balance * self.cfg.risk_percent
        size = risk_amount / sl_dist

        # Plafond
        max_usd = sizing_balance * self.cfg.max_position_pct
        if size * entry_px > max_usd:
            size = max_usd / entry_px

        if size <= 0:
            return False

        cost = size * entry_px * self.cfg.fee_pct
        if cost > self.cash:
            return False

        self.cash -= cost

        pos = RSIPosition(
            symbol=sym,
            side=side,
            entry_price=entry_px,
            sl_price=sl_px,
            tp_price=tp_px,
            size=size,
            entry_time=ts,
            rsi_at_entry=rsi_val,
            best_price=entry_px,
        )
        self.positions.append(pos)
        return True

    def _close_position(self, pos: RSIPosition, exit_px: float, ts: int, reason: str) -> None:
        is_long = pos.side == "LONG"
        fee = pos.size * exit_px * self.cfg.fee_pct

        if is_long:
            pnl_raw = pos.size * (exit_px - pos.entry_price)
        else:
            pnl_raw = pos.size * (pos.entry_price - exit_px)
            # Margin cost for shorts
            duration_bars = max(1, (ts - pos.entry_time) / (4 * 3600 * 1000))
            margin_cost = pos.size * pos.entry_price * self.cfg.margin_fee_pct * duration_bars
            pnl_raw -= margin_cost

        pnl_net = pnl_raw - fee
        self.cash += pnl_net

        pnl_pct = pnl_net / (pos.size * pos.entry_price) if pos.entry_price else 0

        self.closed_trades.append(RSITrade(
            symbol=pos.symbol,
            side=pos.side,
            entry_price=pos.entry_price,
            exit_price=exit_px,
            size=pos.size,
            entry_time=pos.entry_time,
            exit_time=ts,
            pnl_usd=pnl_net,
            pnl_pct=pnl_pct,
            exit_reason=reason,
            rsi_at_entry=pos.rsi_at_entry,
        ))

        if pos in self.positions:
            self.positions.remove(pos)

        self._last_trade_ts[pos.symbol] = ts

        # Cooldown on consecutive losses
        sym = pos.symbol
        if pnl_net < 0:
            self._consec_losses[sym] = self._consec_losses.get(sym, 0) + 1
            if self._consec_losses[sym] >= self.cfg.max_consec_losses:
                pause_ms = self.cfg.cooldown_after_losses * 4 * 3600 * 1000
                self._cooldown_until[sym] = ts + pause_ms
                self._consec_losses[sym] = 0
        else:
            self._consec_losses[sym] = 0

    def _close_remaining(self, last_ts: int) -> None:
        for pos in list(self.positions):
            px = self.last_close.get(pos.symbol, pos.entry_price)
            self._close_position(pos, px, last_ts, "END")

    def _record_equity(self, ts: int) -> None:
        unrealized = 0.0
        for p in self.positions:
            price = self.last_close.get(p.symbol, p.entry_price)
            if p.side == "LONG":
                unrealized += p.size * (price - p.entry_price)
            else:
                unrealized += p.size * (p.entry_price - price)
        self.equity_curve.append(EquityPoint(ts, self.cash + unrealized))


# ════════════════════════════════════════════════════════════════════════════════
# Rapport
# ════════════════════════════════════════════════════════════════════════════════


def print_report(result: BacktestResult, m: dict, raw_trades: list[RSITrade], cfg: RSIConfig) -> None:
    sep = "═" * 80
    print(f"\n{sep}")
    print(f"  📊 RSI BOT [{cfg.mode.value}] — Résultats")
    print(f"  📅 {result.start_date.date()} → {result.end_date.date()}")
    print(f"  RSI: oversold={cfg.rsi_oversold:.0f} / overbought={cfg.rsi_overbought:.0f}")
    print(sep)

    print(f"\n  💰 Capital  : ${result.initial_balance:,.2f} → ${result.final_equity:,.2f}")
    print(f"  📈 Return   : {m['total_return']:+.1%}")
    print(f"  📊 CAGR     : {m['cagr']:+.1%}")
    print(f"  📉 Max DD   : {m['max_drawdown']:.1%}")
    print(f"  📐 Sharpe   : {m['sharpe']:.2f}")
    print(f"  📐 Sortino  : {m['sortino']:.2f}")
    print(f"  🎯 Trades   : {m['n_trades']}")
    print(f"  ✅ Win Rate : {m['win_rate']:.1%}")
    print(f"  💹 PF       : {m['profit_factor']:.2f}")
    print(f"  💵 PnL Moy  : ${m['avg_pnl_usd']:+.2f}")

    # Side breakdown
    longs = [t for t in raw_trades if t.side == "LONG"]
    shorts = [t for t in raw_trades if t.side == "SHORT"]
    if longs:
        l_wins = sum(1 for t in longs if t.pnl_usd > 0)
        l_pnl = sum(t.pnl_usd for t in longs)
        print(f"\n  📗 LONG  : {len(longs)} trades | WR {l_wins/len(longs):.0%} | PnL ${l_pnl:+.2f}")
    if shorts:
        s_wins = sum(1 for t in shorts if t.pnl_usd > 0)
        s_pnl = sum(t.pnl_usd for t in shorts)
        print(f"  📕 SHORT : {len(shorts)} trades | WR {s_wins/len(shorts):.0%} | PnL ${s_pnl:+.2f}")

    # Par paire (top 5 + bottom 5)
    by_pair = m.get("by_pair", {})
    if by_pair:
        sorted_pairs = sorted(by_pair.items(), key=lambda x: x[1]["pnl"], reverse=True)
        print(f"\n  📊 Top 5 paires :")
        for pair, stats in sorted_pairs[:5]:
            print(f"     {pair:10s} : {stats['n']:3d} trades | WR {stats['wr']:.0%} | PF {stats['pf']:.2f} | PnL ${stats['pnl']:+.2f}")
        if len(sorted_pairs) > 5:
            print(f"  📊 Bottom 5 paires :")
            for pair, stats in sorted_pairs[-5:]:
                print(f"     {pair:10s} : {stats['n']:3d} trades | WR {stats['wr']:.0%} | PF {stats['pf']:.2f} | PnL ${stats['pnl']:+.2f}")

    # Par exit
    by_exit = m.get("by_exit", {})
    if by_exit:
        print(f"\n  🚪 Par exit :")
        for reason, stats in sorted(by_exit.items()):
            print(f"     {reason:10s} : {stats['n']:3d} trades | PnL ${stats['pnl']:+.2f}")

    # RSI stats
    if raw_trades:
        wins = [t for t in raw_trades if t.pnl_usd > 0]
        losses = [t for t in raw_trades if t.pnl_usd <= 0]
        if wins:
            avg_rsi = sum(t.rsi_at_entry for t in wins) / len(wins)
            print(f"\n  🏆 RSI moy wins  : {avg_rsi:.1f}")
        if losses:
            avg_rsi = sum(t.rsi_at_entry for t in losses) / len(losses)
            print(f"  ❌ RSI moy losses: {avg_rsi:.1f}")

        best = max(raw_trades, key=lambda t: t.pnl_usd)
        worst = min(raw_trades, key=lambda t: t.pnl_usd)
        print(f"\n  🏅 Best  : {best.symbol} {best.side} RSI={best.rsi_at_entry:.0f} → ${best.pnl_usd:+.2f} ({best.pnl_pct:+.1%})")
        print(f"  💀 Worst : {worst.symbol} {worst.side} RSI={worst.rsi_at_entry:.0f} → ${worst.pnl_usd:+.2f} ({worst.pnl_pct:+.1%})")

    print(f"\n{sep}\n")


# ════════════════════════════════════════════════════════════════════════════════
# Graphiques
# ════════════════════════════════════════════════════════════════════════════════


def generate_charts(result: BacktestResult, m: dict, raw_trades: list[RSITrade],
                    cfg: RSIConfig, show: bool = True) -> Path:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mticker

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(3, 1, figsize=(16, 14), height_ratios=[3, 1, 1.5])
    plt.style.use("seaborn-v0_8-whitegrid")

    # 1. Equity
    ax1 = axes[0]
    eq = result.equity_curve
    dates = [datetime.fromtimestamp(p.timestamp / 1000, tz=timezone.utc) for p in eq]
    equities = [p.equity for p in eq]
    ax1.plot(dates, equities, color="#2196F3", linewidth=1.5, alpha=0.9)
    ax1.fill_between(dates, result.initial_balance, equities, alpha=0.1, color="#2196F3")
    ax1.axhline(y=result.initial_balance, color="gray", linestyle="--", alpha=0.4)
    ax1.set_title(
        f"RSI Bot [{cfg.mode.value}] -- Equity Curve\n"
        f"RSI {cfg.rsi_oversold:.0f}/{cfg.rsi_overbought:.0f} | "
        f"Return: {m['total_return']:+.1%} | Sharpe: {m['sharpe']:.2f} | "
        f"WR: {m['win_rate']:.0%} | PF: {m['profit_factor']:.2f} | "
        f"Trades: {m['n_trades']}",
        fontsize=12, fontweight="bold",
    )
    ax1.set_ylabel("Equity ($)")
    ax1.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x:,.0f}"))

    # 2. Drawdown
    ax2 = axes[1]
    peak = equities[0]
    dd = []
    for e in equities:
        peak = max(peak, e)
        dd.append((e - peak) / peak if peak else 0)
    ax2.fill_between(dates, dd, alpha=0.3, color="#F44336")
    ax2.plot(dates, dd, color="#F44336", linewidth=0.8)
    ax2.set_ylabel("Drawdown")
    ax2.yaxis.set_major_formatter(mticker.PercentFormatter(1.0, decimals=0))

    # 3. PnL par trade (coloré par side)
    ax3 = axes[2]
    if raw_trades:
        trade_dates = [datetime.fromtimestamp(t.exit_time / 1000, tz=timezone.utc) for t in raw_trades]
        pnls = [t.pnl_usd for t in raw_trades]
        colors = []
        for t in raw_trades:
            if t.pnl_usd > 0:
                colors.append("#4CAF50" if t.side == "LONG" else "#81C784")
            else:
                colors.append("#F44336" if t.side == "LONG" else "#EF9A9A")
        ax3.bar(trade_dates, pnls, color=colors, alpha=0.7, width=0.5)
        ax3.axhline(y=0, color="gray", linestyle="-", alpha=0.4)
    ax3.set_ylabel("PnL ($)")
    ax3.set_title("PnL par trade (vert=LONG, rouge clair=SHORT)", fontsize=10)

    plt.tight_layout()
    chart_path = OUTPUT_DIR / "rsi_bot_backtest.png"
    fig.savefig(chart_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  📊 Chart: {chart_path}")

    if show:
        try:
            import subprocess
            subprocess.run(["open", str(chart_path)], check=False)
        except Exception:
            pass

    return chart_path


# ════════════════════════════════════════════════════════════════════════════════
# Sensibilité
# ════════════════════════════════════════════════════════════════════════════════


def run_sensitivity(
    candles: dict[str, list[Candle]],
    base_cfg: RSIConfig,
    btc_d1: list[Candle] | None = None,
) -> None:
    print("\n" + "═" * 100)
    print("  🔬 Analyse de sensibilité — RSI Bot")
    if base_cfg.use_macro_filter:
        print("  🌍 Macro filter BTC EMA200 D1 : ACTIF")
    print("═" * 100)

    configs = [
        # ── Seuils RSI ──
        ("RSI 30/70 BIDIR",       {"rsi_oversold": 30, "rsi_overbought": 70, "mode": RSIMode.BIDIR}),
        ("RSI 25/75 BIDIR",       {"rsi_oversold": 25, "rsi_overbought": 75, "mode": RSIMode.BIDIR}),
        ("RSI 20/80 BIDIR",       {"rsi_oversold": 20, "rsi_overbought": 80, "mode": RSIMode.BIDIR}),
        ("RSI 35/65 BIDIR",       {"rsi_oversold": 35, "rsi_overbought": 65, "mode": RSIMode.BIDIR}),

        # ── Mode ──
        ("RSI 30/70 LONG only",   {"rsi_oversold": 30, "rsi_overbought": 70, "mode": RSIMode.LONG}),
        ("RSI 30/70 SHORT only",  {"rsi_oversold": 30, "rsi_overbought": 70, "mode": RSIMode.SHORT}),

        # ── Exit RSI ──
        ("Exit RSI 50/50",        {"rsi_exit_long": 50, "rsi_exit_short": 50}),
        ("Exit RSI 60/40",        {"rsi_exit_long": 60, "rsi_exit_short": 40}),
        ("No RSI exit (SL/TP)",   {"use_rsi_exit": False}),

        # ── SL/TP ──
        ("SL 1.0 / TP 2.0",      {"sl_atr_mult": 1.0, "tp_atr_mult": 2.0}),
        ("SL 2.0 / TP 3.0",      {"sl_atr_mult": 2.0, "tp_atr_mult": 3.0}),
        ("SL 1.5 / TP 4.0",      {"sl_atr_mult": 1.5, "tp_atr_mult": 4.0}),

        # ── Trail ──
        ("No trailing",           {"trail_after_tp": False}),
        ("Trail ATR×1.5",         {"trail_after_tp": True, "trail_atr_mult": 1.5}),

        # ── Macro filter ──
        ("Macro filter ON",       {"use_macro_filter": True}),
        ("Macro filter OFF",      {"use_macro_filter": False}),
    ]

    header = (
        f"  {'Config':<25s} │ {'Trades':>6s} │ {'L/S':>7s} │ "
        f"{'WR':>5s} │ {'PF':>6s} │ {'Return':>8s} │ "
        f"{'Sharpe':>6s} │ {'MaxDD':>7s} │ {'Final':>10s}"
    )
    print(f"\n{header}")
    print("  " + "─" * (len(header) - 2))

    for label, overrides in configs:
        cfg = RSIConfig(initial_balance=base_cfg.initial_balance)
        # Copier les valeurs de base
        for attr in (
            "rsi_period", "rsi_oversold", "rsi_overbought",
            "rsi_exit_long", "rsi_exit_short",
            "atr_period", "sl_atr_mult", "tp_atr_mult",
            "use_rsi_exit", "trail_after_tp", "trail_atr_mult",
            "risk_percent", "fee_pct", "slippage_pct", "margin_fee_pct",
            "use_macro_filter", "mode",
        ):
            setattr(cfg, attr, getattr(base_cfg, attr))
        # Appliquer les overrides
        for k, v in overrides.items():
            setattr(cfg, k, v)

        engine = RSIEngine(candles, cfg, btc_d1_candles=btc_d1)
        res = engine.run()
        m = compute_metrics(res)

        n_long = sum(1 for t in engine.closed_trades if t.side == "LONG")
        n_short = sum(1 for t in engine.closed_trades if t.side == "SHORT")
        ls_str = f"{n_long}L/{n_short}S"

        row = (
            f"  {label:<25s} │ {m['n_trades']:>6d} │ {ls_str:>7s} │ "
            f"{m['win_rate']:>4.0%} │ {m['profit_factor']:>6.2f} │ "
            f"{m['total_return']:>+7.1%} │ {m['sharpe']:>6.2f} │ "
            f"{m['max_drawdown']:>6.1%} │ ${m['final_equity']:>9,.2f}"
        )
        print(row)

    print()


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
    parser = argparse.ArgumentParser(description="Backtest RSI Bot — Mean Reversion")
    parser.add_argument("--months", type=int, default=12)
    parser.add_argument("--balance", type=float, default=1000.0)
    parser.add_argument("--pairs", type=int, default=20)
    parser.add_argument("--no-show", action="store_true")
    parser.add_argument("--sensitivity", action="store_true")
    parser.add_argument("--mode", type=str, default="BIDIR", choices=["LONG", "SHORT", "BIDIR"])
    parser.add_argument("--rsi-oversold", type=float, default=30.0)
    parser.add_argument("--rsi-overbought", type=float, default=70.0)
    parser.add_argument("--rsi-exit-long", type=float, default=55.0)
    parser.add_argument("--rsi-exit-short", type=float, default=45.0)
    parser.add_argument("--sl-atr", type=float, default=1.5)
    parser.add_argument("--tp-atr", type=float, default=2.5)
    parser.add_argument("--no-trail", action="store_true")
    parser.add_argument("--no-rsi-exit", action="store_true")
    parser.add_argument("--macro-filter", action="store_true")
    args = parser.parse_args()

    end = datetime.now(tz=timezone.utc)
    start = end - timedelta(days=args.months * 30)
    pairs = PAIRS_20[:args.pairs]
    mode = RSIMode[args.mode]

    print(f"\n{'═' * 80}")
    print(f"  📊 RSI BOT — Backtest Mean Reversion")
    print(f"  📅 {start.date()} → {end.date()} ({args.months} mois)")
    print(f"  💰 ${args.balance:,.0f} | {len(pairs)} paires | Mode: {mode.value}")
    print(f"  RSI: oversold={args.rsi_oversold:.0f} / overbought={args.rsi_overbought:.0f}")
    print(f"  Exit RSI: long≥{args.rsi_exit_long:.0f} / short≤{args.rsi_exit_short:.0f}")
    print(f"  SL: {args.sl_atr}×ATR | TP: {args.tp_atr}×ATR | Trail: {'ON' if not args.no_trail else 'OFF'}")
    print(f"  RSI exit: {'ON' if not args.no_rsi_exit else 'OFF'} | Macro: {'ON' if args.macro_filter else 'OFF'}")
    print(f"{'═' * 80}\n")

    # Download
    logger.info("📥 Téléchargement données H4…")
    candles = download_all_pairs(pairs, start, end, interval="4h")

    btc_d1 = None
    if args.macro_filter or args.sensitivity:
        logger.info("📥 Téléchargement BTC D1 pour macro filter…")
        btc_d1 = download_btc_d1(start, end)

    # Config
    cfg = RSIConfig(
        initial_balance=args.balance,
        mode=mode,
        rsi_oversold=args.rsi_oversold,
        rsi_overbought=args.rsi_overbought,
        rsi_exit_long=args.rsi_exit_long,
        rsi_exit_short=args.rsi_exit_short,
        sl_atr_mult=args.sl_atr,
        tp_atr_mult=args.tp_atr,
        trail_after_tp=not args.no_trail,
        use_rsi_exit=not args.no_rsi_exit,
        use_macro_filter=args.macro_filter,
    )

    # Run principal
    print("\n" + "─" * 60)
    print(f"  📊 RSI BOT [{mode.value}] — Mean Reversion")
    print("─" * 60)

    engine = RSIEngine(candles, cfg, btc_d1_candles=btc_d1)
    result = engine.run()
    metrics = compute_metrics(result)

    print_report(result, metrics, engine.closed_trades, cfg)
    generate_charts(result, metrics, engine.closed_trades, cfg, show=not args.no_show)

    if args.sensitivity:
        run_sensitivity(candles, cfg, btc_d1)


if __name__ == "__main__":
    main()
