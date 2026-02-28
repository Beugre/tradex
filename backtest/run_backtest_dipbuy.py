#!/usr/bin/env python
"""
Backtest DIP BUY — Acheter quand une crypto chute de -X% en 24h.

🧠 Logique :
   1. Signal : close actuel / close 24h avant ≤ -(THRESHOLD)%
   2. Buy au close de la bougie signal (+ slippage)
   3. SL initial = entry × (1 - SL_PCT)           ex: -2%
   4. TP initial = entry × (1 + TP_PCT)           ex: +4%
   5. Step trailing quand high → TP - buffer :
      - SL verrouille au niveau du TP précédent
      - TP s'étend de +STEP_PCT
      → Step 1 : à ~+3.95%, SL → +4%, TP → +5%
      → Step 2 : à ~+4.95%, SL → +5%, TP → +6%
      → Etc.
   6. Si prix redescend sous SL → exit (profit locké ou perte initiale)

Usage :
  PYTHONPATH=. .venv/bin/python backtest/run_backtest_dipbuy.py --no-show
  PYTHONPATH=. .venv/bin/python backtest/run_backtest_dipbuy.py --timeframe 1d --no-show
  PYTHONPATH=. .venv/bin/python backtest/run_backtest_dipbuy.py --compare --no-show
  PYTHONPATH=. .venv/bin/python backtest/run_backtest_dipbuy.py --sensitivity --no-show
  PYTHONPATH=. .venv/bin/python backtest/run_backtest_dipbuy.py --threshold 0.10 --sl 0.02 --tp 0.04 --step 0.01 --no-show
"""

from __future__ import annotations

import argparse
import logging
import sys
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
load_dotenv()

from src.core.models import Candle, OrderSide, StrategyType
from backtest.data_loader import download_all_pairs
from backtest.simulator import BacktestResult, Trade, EquityPoint
from backtest.metrics import compute_metrics

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("dipbuy")

OUTPUT_DIR = Path(__file__).parent / "output"


# ════════════════════════════════════════════════════════════════════════════════
# Config
# ════════════════════════════════════════════════════════════════════════════════


@dataclass
class DipBuyConfig:
    initial_balance: float = 1000.0

    # ── Signal ──
    drop_threshold: float = 0.10       # -10% en 24h pour déclencher
    lookback_bars: int = 6             # 6×4h = 24h  (auto-set selon interval)

    # ── Entry / Exit ──
    sl_pct: float = 0.02               # SL = -2% sous l'entrée
    tp_pct: float = 0.04               # TP = +4% au-dessus de l'entrée

    # ── Step trailing ──
    trail_trigger_buffer: float = 0.0005   # trigger trail à TP - 0.05% de l'entry
    trail_step_pct: float = 0.01           # TP extend par step = +1%

    # ── Risk ──
    risk_percent: float = 0.02
    max_position_pct: float = 0.30
    max_simultaneous: int = 5
    fee_pct: float = 0.00075
    slippage_pct: float = 0.001

    # ── Cooldown ──
    cooldown_bars: int = 6

    # ── Filters anti-DD ──
    sma_period: int = 0              # 0 = désactivé. Ex: 50 → only buy if close > SMA(50)
    equity_sma_period: int = 0       # 0 = désactivé. Ex: 30 → stop trading si equity < SMA(equity)
    max_portfolio_heat: float = 1.0  # Max % du capital exposé en risque simultané (1.0 = illimité)
    btc_trend_filter: bool = False   # Si True, ne trade que si BTC > SMA(50)
    min_drop_recovery: float = 0.0   # 0 = désactivé. Ex: 0.02 → signal seulement si le low de la bougie est > 2% au-dessus du low 24h (rebond amorcé)

    # ── Filtres capitulation ──
    rsi_max: float = 0.0             # 0 = désactivé. Ex: 18 → RSI(14) doit être < 18
    rsi_period: int = 14             # Période RSI
    vol_spike_mult: float = 0.0      # 0 = désactivé. Ex: 2.0 → volume > 2× moyenne(20)
    vol_avg_period: int = 20         # Période pour la moyenne du volume
    min_wick_ratio: float = 0.0      # 0 = désactivé. Ex: 1.5 → mèche basse > 1.5× corps

    # ── SL dynamique ATR ──
    atr_sl_mult: float = 0.0         # 0 = désactivé (utilise sl_pct). Ex: 1.5 → SL = entry - 1.5×ATR
    atr_period: int = 14             # Période ATR

    # ── Timeframe ──
    interval: str = "4h"


# ════════════════════════════════════════════════════════════════════════════════
# Structures
# ════════════════════════════════════════════════════════════════════════════════


@dataclass
class DipPosition:
    symbol: str
    entry_price: float
    sl_price: float
    tp_price: float
    size: float
    entry_time: int
    trail_steps: int = 0
    best_price: float = 0.0
    drop_pct: float = 0.0


@dataclass
class DipTrade:
    symbol: str
    entry_price: float
    exit_price: float
    size: float
    entry_time: int
    exit_time: int
    pnl_usd: float
    pnl_pct: float
    exit_reason: str
    trail_steps: int
    drop_pct: float


# ════════════════════════════════════════════════════════════════════════════════
# Moteur
# ════════════════════════════════════════════════════════════════════════════════


class DipBuyEngine:
    """Backtest dip-buy : achète les crashs -X% en 24h, step trailing."""

    def __init__(
        self,
        candles_by_symbol: dict[str, list[Candle]],
        config: DipBuyConfig,
    ) -> None:
        self.candles = candles_by_symbol
        self.cfg = config
        self.pairs = list(candles_by_symbol.keys())

        self.cash: float = config.initial_balance
        self.positions: list[DipPosition] = []
        self.closed_trades: list[DipTrade] = []
        self.equity_curve: list[EquityPoint] = []

        self._last_trade_ts: dict[str, int] = {p: 0 for p in self.pairs}
        self.last_close: dict[str, float] = {}

        # Index rapide : (symbol, timestamp) → Candle
        self._idx: dict[tuple[str, int], Candle] = {}
        self._candle_lists: dict[str, list[Candle]] = {}
        for sym, clist in candles_by_symbol.items():
            sorted_c = sorted(clist, key=lambda c: c.timestamp)
            self._candle_lists[sym] = sorted_c
            for c in sorted_c:
                self._idx[(sym, c.timestamp)] = c

        # Pré-calculer le close N bars avant pour chaque (sym, ts)
        self._lookback_close: dict[str, dict[int, float]] = {}
        for sym, clist in self._candle_lists.items():
            lb: dict[int, float] = {}
            for i, c in enumerate(clist):
                if i >= config.lookback_bars:
                    lb[c.timestamp] = clist[i - config.lookback_bars].close
            self._lookback_close[sym] = lb

        # Pré-calculer SMA pour chaque (sym, ts)
        self._sma: dict[str, dict[int, float]] = {}
        if config.sma_period > 0:
            for sym, clist in self._candle_lists.items():
                sma_map: dict[int, float] = {}
                closes = [c.close for c in clist]
                p = config.sma_period
                for i in range(p - 1, len(clist)):
                    sma_map[clist[i].timestamp] = sum(closes[i - p + 1 : i + 1]) / p
                self._sma[sym] = sma_map

        # BTC SMA pour filtre tendance macro
        self._btc_sma: dict[int, float] = {}
        if config.btc_trend_filter and "BTC-USD" in self._candle_lists:
            clist = self._candle_lists["BTC-USD"]
            closes = [c.close for c in clist]
            p = 50  # SMA 50 pour BTC
            for i in range(p - 1, len(clist)):
                self._btc_sma[clist[i].timestamp] = sum(closes[i - p + 1 : i + 1]) / p

        # Low 24h lookback pour min_drop_recovery
        self._lookback_low: dict[str, dict[int, float]] = {}
        if config.min_drop_recovery > 0:
            for sym, clist in self._candle_lists.items():
                low_map: dict[int, float] = {}
                for i in range(config.lookback_bars, len(clist)):
                    low_map[clist[i].timestamp] = min(
                        c.low for c in clist[i - config.lookback_bars : i + 1]
                    )
                self._lookback_low[sym] = low_map

        # Pré-calculer RSI pour chaque (sym, ts)
        self._rsi: dict[str, dict[int, float]] = {}
        if config.rsi_max > 0:
            for sym, clist in self._candle_lists.items():
                rsi_map: dict[int, float] = {}
                p = config.rsi_period
                if len(clist) > p:
                    # Init avec SMA des gains/pertes
                    gains = []
                    losses = []
                    for j in range(1, p + 1):
                        delta = clist[j].close - clist[j - 1].close
                        gains.append(max(delta, 0))
                        losses.append(max(-delta, 0))
                    avg_gain = sum(gains) / p
                    avg_loss = sum(losses) / p
                    if avg_loss > 0:
                        rs = avg_gain / avg_loss
                        rsi_map[clist[p].timestamp] = 100 - 100 / (1 + rs)
                    else:
                        rsi_map[clist[p].timestamp] = 100.0
                    # Smoothed (Wilder)
                    for j in range(p + 1, len(clist)):
                        delta = clist[j].close - clist[j - 1].close
                        avg_gain = (avg_gain * (p - 1) + max(delta, 0)) / p
                        avg_loss = (avg_loss * (p - 1) + max(-delta, 0)) / p
                        if avg_loss > 0:
                            rs = avg_gain / avg_loss
                            rsi_map[clist[j].timestamp] = 100 - 100 / (1 + rs)
                        else:
                            rsi_map[clist[j].timestamp] = 100.0
                self._rsi[sym] = rsi_map

        # Pré-calculer ATR pour chaque (sym, ts)
        self._atr: dict[str, dict[int, float]] = {}
        if config.atr_sl_mult > 0:
            for sym, clist in self._candle_lists.items():
                atr_map: dict[int, float] = {}
                p = config.atr_period
                if len(clist) > p:
                    trs: list[float] = []
                    for j in range(1, p + 1):
                        tr = max(
                            clist[j].high - clist[j].low,
                            abs(clist[j].high - clist[j - 1].close),
                            abs(clist[j].low - clist[j - 1].close),
                        )
                        trs.append(tr)
                    atr = sum(trs) / p
                    atr_map[clist[p].timestamp] = atr
                    for j in range(p + 1, len(clist)):
                        tr = max(
                            clist[j].high - clist[j].low,
                            abs(clist[j].high - clist[j - 1].close),
                            abs(clist[j].low - clist[j - 1].close),
                        )
                        atr = (atr * (p - 1) + tr) / p
                        atr_map[clist[j].timestamp] = atr
                self._atr[sym] = atr_map

        # Pré-calculer volume moyen pour chaque (sym, ts)
        self._vol_avg: dict[str, dict[int, float]] = {}
        if config.vol_spike_mult > 0:
            for sym, clist in self._candle_lists.items():
                va_map: dict[int, float] = {}
                p = config.vol_avg_period
                for i in range(p, len(clist)):
                    vols = [clist[j].volume for j in range(i - p, i)]
                    va_map[clist[i].timestamp] = sum(vols) / p
                self._vol_avg[sym] = va_map

        # Bar duration en ms
        if config.interval == "1d":
            self._bar_ms = 24 * 3600 * 1000
        elif config.interval == "1h":
            self._bar_ms = 3600 * 1000
        else:  # 4h
            self._bar_ms = 4 * 3600 * 1000

    # ── Run ─────────────────────────────────────────────────────────────────

    def run(self) -> BacktestResult:
        timeline = self._build_timeline()
        total = len(timeline)

        logger.info(
            "📊 DIP BUY [%s] : %d barres, %d paires, $%.0f | drop≥%.0f%% | "
            "SL=%.0f%% TP=%.0f%% step=%.0f%%",
            self.cfg.interval.upper(), total, len(self.pairs),
            self.cfg.initial_balance,
            self.cfg.drop_threshold * 100, self.cfg.sl_pct * 100,
            self.cfg.tp_pct * 100, self.cfg.trail_step_pct * 100,
        )

        for i, ts in enumerate(timeline):
            for sym in self.pairs:
                c = self._idx.get((sym, ts))
                if c:
                    self.last_close[sym] = c.close

            self._manage_positions(ts)
            self._scan_signals(ts)
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
                strategy=StrategyType.RANGE,
                side=OrderSide.BUY,
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

    # ── Position management ────────────────────────────────────────────────

    def _manage_positions(self, ts: int) -> None:
        to_close: list[tuple[DipPosition, float, str]] = []

        for pos in self.positions:
            c = self._idx.get((pos.symbol, ts))
            if c is None:
                continue

            pos.best_price = max(pos.best_price, c.high)

            # 1. SL check (pessimistic — vérifié en premier)
            if c.low <= pos.sl_price:
                reason = f"TRAIL_SL{pos.trail_steps}" if pos.trail_steps > 0 else "SL"
                to_close.append((pos, pos.sl_price, reason))
                continue

            # 2. TP / Step trailing
            if self.cfg.trail_step_pct > 0:
                # Step trailing : peut cascader sur un gros mouvement
                safety = 0
                while safety < 50:
                    trigger = pos.tp_price - pos.entry_price * self.cfg.trail_trigger_buffer
                    if c.high >= trigger:
                        # Verrouiller SL au TP actuel, étendre TP
                        pos.sl_price = pos.tp_price
                        pos.tp_price += pos.entry_price * self.cfg.trail_step_pct
                        pos.trail_steps += 1
                        safety += 1
                    else:
                        break
            else:
                # Pas de trailing → TP direct
                if c.high >= pos.tp_price:
                    to_close.append((pos, pos.tp_price, "TP"))
                    continue

        for pos, exit_px, reason in to_close:
            self._close_position(pos, exit_px, ts, reason)

    # ── Signal scan ────────────────────────────────────────────────────────

    def _scan_signals(self, ts: int) -> None:
        if len(self.positions) >= self.cfg.max_simultaneous:
            return

        for sym in self.pairs:
            if len(self.positions) >= self.cfg.max_simultaneous:
                break
            if any(p.symbol == sym for p in self.positions):
                continue

            # Cooldown
            if ts - self._last_trade_ts.get(sym, 0) < self.cfg.cooldown_bars * self._bar_ms:
                continue

            c = self._idx.get((sym, ts))
            if c is None:
                continue

            # Vérifier le drop sur 24h
            close_ago = self._lookback_close.get(sym, {}).get(ts)
            if close_ago is None or close_ago <= 0:
                continue

            drop = (c.close - close_ago) / close_ago  # négatif si baisse
            if drop > -self.cfg.drop_threshold:
                continue

            # ── Filtre SMA : prix doit être > SMA (dip dans un uptrend) ──
            if self.cfg.sma_period > 0:
                sma_val = self._sma.get(sym, {}).get(ts)
                if sma_val is None or c.close < sma_val:
                    continue  # Pas d'achat en dessous de la SMA

            # ── Filtre BTC trend : BTC doit être > SMA(50) ──
            if self.cfg.btc_trend_filter and self._btc_sma:
                btc_sma = self._btc_sma.get(ts)
                btc_close = self.last_close.get("BTC-USD")
                if btc_sma is not None and btc_close is not None:
                    if btc_close < btc_sma:
                        continue  # Pas d'achat si BTC en downtrend

            # ── Filtre recovery : le prix doit avoir rebondi un minimum ──
            if self.cfg.min_drop_recovery > 0:
                low_24h = self._lookback_low.get(sym, {}).get(ts)
                if low_24h is not None and low_24h > 0:
                    recovery = (c.close - low_24h) / low_24h
                    if recovery < self.cfg.min_drop_recovery:
                        continue  # Pas assez de rebond

            # ── Filtre equity curve : stop si equity < SMA(equity) ──
            if self.cfg.equity_sma_period > 0 and len(self.equity_curve) >= self.cfg.equity_sma_period:
                recent_eq = [ep.equity for ep in self.equity_curve[-self.cfg.equity_sma_period:]]
                eq_sma = sum(recent_eq) / len(recent_eq)
                if self.equity_curve[-1].equity < eq_sma:
                    continue  # Equity en drawdown, on pause

            # ── Filtre portfolio heat : risque total ouvert ──
            if self.cfg.max_portfolio_heat < 1.0:
                total_risk = sum(
                    p.size * abs(p.entry_price - p.sl_price)
                    for p in self.positions
                )
                current_eq = self.equity_curve[-1].equity if self.equity_curve else self.cash
                if current_eq > 0 and total_risk / current_eq >= self.cfg.max_portfolio_heat:
                    continue  # Trop de risque ouvert

            # ── Filtre RSI : RSI doit être < seuil (capitulation) ──
            if self.cfg.rsi_max > 0:
                rsi_val = self._rsi.get(sym, {}).get(ts)
                if rsi_val is None or rsi_val >= self.cfg.rsi_max:
                    continue  # RSI pas assez bas

            # ── Filtre volume spike : volume > X× moyenne ──
            if self.cfg.vol_spike_mult > 0:
                vol_avg = self._vol_avg.get(sym, {}).get(ts)
                if vol_avg is None or vol_avg <= 0 or c.volume < vol_avg * self.cfg.vol_spike_mult:
                    continue  # Volume pas assez élevé

            # ── Filtre wick ratio : mèche basse > X× corps ──
            if self.cfg.min_wick_ratio > 0:
                body = abs(c.close - c.open)
                lower_wick = min(c.open, c.close) - c.low
                if body <= 0:
                    body = 0.0001  # éviter div/0
                if lower_wick / body < self.cfg.min_wick_ratio:
                    continue  # Pas de mèche de rejet

            # SIGNAL ! Buy au close + slippage
            entry_px = c.close * (1 + self.cfg.slippage_pct)

            # SL : ATR dynamique ou % fixe
            if self.cfg.atr_sl_mult > 0:
                atr_val = self._atr.get(sym, {}).get(ts)
                if atr_val and atr_val > 0:
                    sl_px = entry_px - self.cfg.atr_sl_mult * atr_val
                else:
                    sl_px = entry_px * (1 - self.cfg.sl_pct)
            else:
                sl_px = entry_px * (1 - self.cfg.sl_pct)

            tp_px = entry_px * (1 + self.cfg.tp_pct)

            # Sizing (risk-based)
            sl_dist = entry_px - sl_px
            if sl_dist <= 0:
                continue
            risk_amount = self.cfg.initial_balance * self.cfg.risk_percent
            size = risk_amount / sl_dist

            max_usd = self.cfg.initial_balance * self.cfg.max_position_pct
            if size * entry_px > max_usd:
                size = max_usd / entry_px

            if size <= 0:
                continue

            entry_fee = size * entry_px * self.cfg.fee_pct
            if entry_fee > self.cash:
                continue

            self.cash -= entry_fee

            pos = DipPosition(
                symbol=sym,
                entry_price=entry_px,
                sl_price=sl_px,
                tp_price=tp_px,
                size=size,
                entry_time=ts,
                best_price=entry_px,
                drop_pct=drop,
            )
            self.positions.append(pos)

    # ── Close / Equity ─────────────────────────────────────────────────────

    def _close_position(
        self, pos: DipPosition, exit_px: float, ts: int, reason: str,
    ) -> None:
        exit_fee = pos.size * exit_px * self.cfg.fee_pct
        entry_fee = pos.size * pos.entry_price * self.cfg.fee_pct
        pnl_raw = pos.size * (exit_px - pos.entry_price)
        pnl_net = pnl_raw - exit_fee  # entry_fee déjà déduit du cash

        self.cash += pnl_net

        pnl_pct = (pnl_net + entry_fee) / (pos.size * pos.entry_price) if pos.entry_price else 0
        # pnl total (both fees) for reporting
        pnl_total = pnl_raw - entry_fee - exit_fee

        self.closed_trades.append(DipTrade(
            symbol=pos.symbol,
            entry_price=pos.entry_price,
            exit_price=exit_px,
            size=pos.size,
            entry_time=pos.entry_time,
            exit_time=ts,
            pnl_usd=pnl_total,
            pnl_pct=pnl_total / (pos.size * pos.entry_price) if pos.entry_price else 0,
            exit_reason=reason,
            trail_steps=pos.trail_steps,
            drop_pct=pos.drop_pct,
        ))

        if pos in self.positions:
            self.positions.remove(pos)
        self._last_trade_ts[pos.symbol] = ts

    def _close_remaining(self, last_ts: int) -> None:
        for pos in list(self.positions):
            px = self.last_close.get(pos.symbol, pos.entry_price)
            self._close_position(pos, px, last_ts, "END")

    def _record_equity(self, ts: int) -> None:
        unrealized = 0.0
        for p in self.positions:
            price = self.last_close.get(p.symbol, p.entry_price)
            unrealized += p.size * (price - p.entry_price)
        self.equity_curve.append(EquityPoint(ts, self.cash + unrealized))


# ════════════════════════════════════════════════════════════════════════════════
# Rapport
# ════════════════════════════════════════════════════════════════════════════════


def print_report(
    result: BacktestResult, m: dict,
    raw_trades: list[DipTrade], cfg: DipBuyConfig,
) -> None:
    sep = "═" * 80
    print(f"\n{sep}")
    print(f"  📉 DIP BUY [{cfg.interval.upper()}] — Résultats")
    print(f"  📅 {result.start_date.date()} → {result.end_date.date()}")
    print(
        f"  Drop ≥ {cfg.drop_threshold*100:.0f}% | "
        f"SL={cfg.sl_pct*100:.0f}% | TP={cfg.tp_pct*100:.0f}% | "
        f"Step={cfg.trail_step_pct*100:.0f}%"
    )
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

    if not raw_trades:
        print(f"\n{sep}\n")
        return

    # ── Trail stats ──
    trailed = [t for t in raw_trades if t.trail_steps > 0]
    if trailed:
        avg_steps = sum(t.trail_steps for t in trailed) / len(trailed)
        max_steps = max(t.trail_steps for t in trailed)
        trail_pnl = sum(t.pnl_usd for t in trailed)
        print(
            f"\n  🔄 Trailing : {len(trailed)} trades trailés "
            f"({len(trailed)/len(raw_trades)*100:.0f}%)"
        )
        print(
            f"     Steps moy: {avg_steps:.1f} | max: {max_steps} | "
            f"PnL trail: ${trail_pnl:+.2f}"
        )

    # ── Exit reasons ──
    reasons: dict[str, list[DipTrade]] = {}
    for t in raw_trades:
        reasons.setdefault(t.exit_reason, []).append(t)

    print(f"\n  🚪 Par exit :")
    for reason, trades in sorted(reasons.items()):
        pnl = sum(t.pnl_usd for t in trades)
        wr = sum(1 for t in trades if t.pnl_usd > 0) / len(trades) if trades else 0
        print(f"     {reason:14s} : {len(trades):3d} trades | WR {wr:.0%} | PnL ${pnl:+.2f}")

    # ── Drop stats ──
    drops = [abs(t.drop_pct) for t in raw_trades]
    print(
        f"\n  📊 Drop signal : moy={sum(drops)/len(drops)*100:.1f}% | "
        f"max={max(drops)*100:.1f}%"
    )

    # ── Par paire ──
    pair_pnl: dict[str, float] = {}
    pair_count: dict[str, int] = {}
    for t in raw_trades:
        pair_pnl[t.symbol] = pair_pnl.get(t.symbol, 0) + t.pnl_usd
        pair_count[t.symbol] = pair_count.get(t.symbol, 0) + 1

    sorted_pairs = sorted(pair_pnl.items(), key=lambda x: x[1], reverse=True)
    print(f"\n  📊 Top 5 paires :")
    for pair, pnl in sorted_pairs[:5]:
        print(f"     {pair:10s} : {pair_count[pair]:3d} trades | PnL ${pnl:+.2f}")
    if len(sorted_pairs) > 5:
        print(f"  📊 Bottom 5 :")
        for pair, pnl in sorted_pairs[-5:]:
            print(f"     {pair:10s} : {pair_count[pair]:3d} trades | PnL ${pnl:+.2f}")

    # ── Best / worst trade ──
    best = max(raw_trades, key=lambda t: t.pnl_usd)
    worst = min(raw_trades, key=lambda t: t.pnl_usd)
    print(
        f"\n  🏅 Best  : {best.symbol} drop={abs(best.drop_pct)*100:.0f}% "
        f"steps={best.trail_steps} → ${best.pnl_usd:+.2f}"
    )
    print(
        f"  💀 Worst : {worst.symbol} drop={abs(worst.drop_pct)*100:.0f}% "
        f"steps={worst.trail_steps} → ${worst.pnl_usd:+.2f}"
    )

    print(f"\n{sep}\n")


# ════════════════════════════════════════════════════════════════════════════════
# Graphique
# ════════════════════════════════════════════════════════════════════════════════


def generate_chart(
    result: BacktestResult, m: dict, raw_trades: list[DipTrade],
    cfg: DipBuyConfig, show: bool = True,
) -> Path:
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
        f"DIP BUY [{cfg.interval.upper()}] — Drop≥{cfg.drop_threshold*100:.0f}% | "
        f"SL={cfg.sl_pct*100:.0f}% TP={cfg.tp_pct*100:.0f}% "
        f"Step={cfg.trail_step_pct*100:.0f}%\n"
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

    # 3. PnL par trade (couleur par trail steps)
    ax3 = axes[2]
    if raw_trades:
        trade_dates = [
            datetime.fromtimestamp(t.exit_time / 1000, tz=timezone.utc)
            for t in raw_trades
        ]
        pnls = [t.pnl_usd for t in raw_trades]
        colors = []
        for t in raw_trades:
            if t.trail_steps > 0:
                colors.append("#4CAF50" if t.pnl_usd > 0 else "#FF9800")
            else:
                colors.append("#2196F3" if t.pnl_usd > 0 else "#F44336")
        ax3.bar(trade_dates, pnls, color=colors, alpha=0.7, width=0.5)
        ax3.axhline(y=0, color="gray", linestyle="-", alpha=0.4)
    ax3.set_ylabel("PnL ($)")
    ax3.set_title(
        "PnL par trade (vert=trailé win, orange=trailé loss, "
        "bleu=TP, rouge=SL)", fontsize=10,
    )

    plt.tight_layout()
    chart_path = OUTPUT_DIR / f"dipbuy_{cfg.interval}_backtest.png"
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
    candles: dict[str, list[Candle]], base_cfg: DipBuyConfig,
) -> None:
    print("\n" + "═" * 105)
    print(f"  🔬 Analyse de sensibilité — DIP BUY [{base_cfg.interval.upper()}]")
    print("═" * 105)

    # Base commune pour 48h+D20
    B = {"lookback_bars": 12, "drop_threshold": 0.20}

    configs = [
        # ═══ RÉFÉRENCES ═══
        ("24h D10 SL5T10", {"sl_pct": 0.05, "tp_pct": 0.10}),
        ("48h+D20 BASE",  {**B, "sl_pct": 0.05, "tp_pct": 0.10}),

        # ═══ 1. TP initial (6%, 8%, 10%) ═══
        ("TP6",           {**B, "sl_pct": 0.05, "tp_pct": 0.06}),
        ("TP8",           {**B, "sl_pct": 0.05, "tp_pct": 0.08}),
        ("TP10",          {**B, "sl_pct": 0.05, "tp_pct": 0.10}),
        ("TP12",          {**B, "sl_pct": 0.05, "tp_pct": 0.12}),

        # ═══ 2. Trail step agressif ═══
        ("Step 0.5%",     {**B, "sl_pct": 0.05, "tp_pct": 0.08, "trail_step_pct": 0.005}),
        ("Step 1%",       {**B, "sl_pct": 0.05, "tp_pct": 0.08, "trail_step_pct": 0.01}),
        ("Step 2%",       {**B, "sl_pct": 0.05, "tp_pct": 0.08, "trail_step_pct": 0.02}),

        # ═══ 3. SL ATR dynamique ═══
        ("ATR 1.0",       {**B, "tp_pct": 0.10, "atr_sl_mult": 1.0}),
        ("ATR 1.5",       {**B, "tp_pct": 0.10, "atr_sl_mult": 1.5}),
        ("ATR 2.0",       {**B, "tp_pct": 0.10, "atr_sl_mult": 2.0}),
        ("ATR 2.5",       {**B, "tp_pct": 0.10, "atr_sl_mult": 2.5}),

        # ═══ 4. RSI capitulation ═══
        ("RSI<30",        {**B, "sl_pct": 0.05, "tp_pct": 0.10, "rsi_max": 30}),
        ("RSI<25",        {**B, "sl_pct": 0.05, "tp_pct": 0.10, "rsi_max": 25}),
        ("RSI<20",        {**B, "sl_pct": 0.05, "tp_pct": 0.10, "rsi_max": 20}),
        ("RSI<18",        {**B, "sl_pct": 0.05, "tp_pct": 0.10, "rsi_max": 18}),
        ("RSI<15",        {**B, "sl_pct": 0.05, "tp_pct": 0.10, "rsi_max": 15}),

        # ═══ 5. Volume spike ═══
        ("Vol>1.5x",      {**B, "sl_pct": 0.05, "tp_pct": 0.10, "vol_spike_mult": 1.5}),
        ("Vol>2x",        {**B, "sl_pct": 0.05, "tp_pct": 0.10, "vol_spike_mult": 2.0}),
        ("Vol>3x",        {**B, "sl_pct": 0.05, "tp_pct": 0.10, "vol_spike_mult": 3.0}),

        # ═══ 6. Wick ratio ═══
        ("Wick>1.0",      {**B, "sl_pct": 0.05, "tp_pct": 0.10, "min_wick_ratio": 1.0}),
        ("Wick>1.5",      {**B, "sl_pct": 0.05, "tp_pct": 0.10, "min_wick_ratio": 1.5}),
        ("Wick>2.0",      {**B, "sl_pct": 0.05, "tp_pct": 0.10, "min_wick_ratio": 2.0}),

        # ═══ 7. Risk % réduit ═══
        ("Risk 1%",       {**B, "sl_pct": 0.05, "tp_pct": 0.10, "risk_percent": 0.01}),
        ("Risk 1.5%",     {**B, "sl_pct": 0.05, "tp_pct": 0.10, "risk_percent": 0.015}),
        ("Risk 2%",       {**B, "sl_pct": 0.05, "tp_pct": 0.10, "risk_percent": 0.02}),

        # ═══ 8. Capitulation complète (RSI + Vol + Wick) ═══
        ("CAP RSI18+V2",  {**B, "sl_pct": 0.05, "tp_pct": 0.10, "rsi_max": 18, "vol_spike_mult": 2.0}),
        ("CAP RSI25+V2",  {**B, "sl_pct": 0.05, "tp_pct": 0.10, "rsi_max": 25, "vol_spike_mult": 2.0}),
        ("CAP R25+V2+W",  {**B, "sl_pct": 0.05, "tp_pct": 0.10, "rsi_max": 25, "vol_spike_mult": 2.0, "min_wick_ratio": 1.5}),
        ("CAP R30+V1.5",  {**B, "sl_pct": 0.05, "tp_pct": 0.10, "rsi_max": 30, "vol_spike_mult": 1.5}),
        ("CAP FULL",      {**B, "sl_pct": 0.05, "tp_pct": 0.10, "rsi_max": 18, "vol_spike_mult": 2.0, "min_wick_ratio": 1.5}),

        # ═══ 9. ATR + capitulation ═══
        ("ATR1.5+RSI25",  {**B, "tp_pct": 0.10, "atr_sl_mult": 1.5, "rsi_max": 25}),
        ("ATR1.5+RSI30",  {**B, "tp_pct": 0.10, "atr_sl_mult": 1.5, "rsi_max": 30}),
        ("ATR2+RSI30",    {**B, "tp_pct": 0.10, "atr_sl_mult": 2.0, "rsi_max": 30}),
        ("ATR1.5+V2",     {**B, "tp_pct": 0.10, "atr_sl_mult": 1.5, "vol_spike_mult": 2.0}),

        # ═══ 10. Best combos avec risk ajusté ═══
        ("R1%+ATR1.5",    {**B, "tp_pct": 0.10, "atr_sl_mult": 1.5, "risk_percent": 0.01}),
        ("R1.5%+ATR1.5",  {**B, "tp_pct": 0.10, "atr_sl_mult": 1.5, "risk_percent": 0.015}),
        ("R1%+RSI30",     {**B, "sl_pct": 0.05, "tp_pct": 0.10, "rsi_max": 30, "risk_percent": 0.01}),
        ("R1%+RSI25+V2",  {**B, "sl_pct": 0.05, "tp_pct": 0.10, "rsi_max": 25, "vol_spike_mult": 2.0, "risk_percent": 0.01}),

        # ═══ 11. TP ajustés + ATR ═══
        ("TP6+ATR1.5",    {**B, "tp_pct": 0.06, "atr_sl_mult": 1.5}),
        ("TP8+ATR1.5",    {**B, "tp_pct": 0.08, "atr_sl_mult": 1.5}),
        ("TP8+ATR1.5+S.5",{**B, "tp_pct": 0.08, "atr_sl_mult": 1.5, "trail_step_pct": 0.005}),
    ]

    header = (
        f"  {'Config':<16s} │ {'Trades':>6s} │ {'WR':>5s} │ {'PF':>6s} │ "
        f"{'Return':>8s} │ {'Sharpe':>6s} │ {'MaxDD':>7s} │ "
        f"{'Final':>10s} │ {'Trail%':>6s}"
    )
    print(f"\n{header}")
    print("  " + "─" * (len(header) - 2))

    for label, overrides in configs:
        cfg = DipBuyConfig(
            initial_balance=base_cfg.initial_balance,
            interval=base_cfg.interval,
            lookback_bars=base_cfg.lookback_bars,
        )
        # Copier les valeurs de base
        for attr in (
            "drop_threshold", "sl_pct", "tp_pct",
            "trail_trigger_buffer", "trail_step_pct",
            "risk_percent", "fee_pct", "slippage_pct",
            "max_simultaneous", "cooldown_bars",
            "sma_period", "equity_sma_period", "max_portfolio_heat",
            "btc_trend_filter", "min_drop_recovery",
            "rsi_max", "rsi_period", "vol_spike_mult", "vol_avg_period",
            "min_wick_ratio", "atr_sl_mult", "atr_period",
        ):
            setattr(cfg, attr, getattr(base_cfg, attr))
        # Overrides
        for k, v in overrides.items():
            setattr(cfg, k, v)

        engine = DipBuyEngine(candles, cfg)
        res = engine.run()
        m = compute_metrics(res)

        trailed = sum(1 for t in engine.closed_trades if t.trail_steps > 0)
        trail_pct = (
            trailed / len(engine.closed_trades) * 100
            if engine.closed_trades else 0
        )

        row = (
            f"  {label:<16s} │ {m['n_trades']:>6d} │ {m['win_rate']:>4.0%} │ "
            f"{m['profit_factor']:>6.2f} │ {m['total_return']:>+7.1%} │ "
            f"{m['sharpe']:>6.2f} │ {m['max_drawdown']:>6.1%} │ "
            f"${m['final_equity']:>9,.2f} │ {trail_pct:>5.0f}%"
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
    parser = argparse.ArgumentParser(
        description="Backtest Dip Buy — crashs 24h avec step trailing",
    )
    parser.add_argument("--months", type=int, default=24)
    parser.add_argument("--balance", type=float, default=1000.0)
    parser.add_argument("--pairs", type=int, default=20)
    parser.add_argument(
        "--timeframe", type=str, default="4h",
        choices=["4h", "1d"],
    )
    parser.add_argument(
        "--threshold", type=float, default=0.10,
        help="Drop threshold (0.10 = 10%%)",
    )
    parser.add_argument("--sl", type=float, default=0.02, help="SL (0.02 = 2%%)")
    parser.add_argument("--tp", type=float, default=0.04, help="TP (0.04 = 4%%)")
    parser.add_argument(
        "--step", type=float, default=0.01,
        help="Trail step (0.01 = 1%%)",
    )
    parser.add_argument(
        "--lookback-hours", type=int, default=24,
        help="Fenêtre de drop en heures (24, 48, 72…)",
    )
    parser.add_argument("--no-show", action="store_true")
    parser.add_argument("--sensitivity", action="store_true")
    parser.add_argument(
        "--compare", action="store_true",
        help="Comparer H4 vs D1",
    )
    args = parser.parse_args()

    end = datetime.now(tz=timezone.utc)
    start = end - timedelta(days=args.months * 30)
    pairs = PAIRS_20[: args.pairs]

    print(f"\n{'═' * 80}")
    print(f"  📉 DIP BUY — Acheter les crashs crypto")
    print(f"  📅 {start.date()} → {end.date()} ({args.months} mois)")
    print(f"  💰 ${args.balance:,.0f} | {len(pairs)} paires")
    print(f"  Signal: drop ≥ {args.threshold*100:.0f}% en {args.lookback_hours}h")
    print(
        f"  SL: -{args.sl*100:.0f}% | TP: +{args.tp*100:.0f}% | "
        f"Trail step: +{args.step*100:.0f}%"
    )
    print(f"{'═' * 80}\n")

    # ── Mode comparaison H4 vs D1 ──────────────────────────────────────────
    if args.compare:
        results: dict[str, tuple] = {}

        for tf, lb, iv in [("H4", 6, "4h"), ("D1", 1, "1d")]:
            print(f"\n{'─' * 60}")
            print(f"  📊 DIP BUY — {tf}")
            print(f"{'─' * 60}")

            logger.info("📥 Téléchargement données %s…", tf)
            candles = download_all_pairs(pairs, start, end, interval=iv)

            cfg = DipBuyConfig(
                initial_balance=args.balance,
                drop_threshold=args.threshold,
                sl_pct=args.sl,
                tp_pct=args.tp,
                trail_step_pct=args.step,
                interval=iv,
                lookback_bars=lb,
            )

            engine = DipBuyEngine(candles, cfg)
            res = engine.run()
            m = compute_metrics(res)
            results[tf] = (res, m, engine.closed_trades, cfg)

        # ── Tableau comparatif ──
        print(f"\n{'═' * 70}")
        print(f"  📊 COMPARAISON H4 vs D1")
        print(f"{'═' * 70}")

        header = f"  {'Métrique':<16s} │ {'H4':>20s} │ {'D1':>20s}"
        print(f"\n{header}")
        print("  " + "─" * 62)

        metrics_keys = [
            ("Trades", "n_trades", "{:d}"),
            ("Win Rate", "win_rate", "{:.1%}"),
            ("Profit Factor", "profit_factor", "{:.2f}"),
            ("Return", "total_return", "{:+.1%}"),
            ("CAGR", "cagr", "{:+.1%}"),
            ("Sharpe", "sharpe", "{:.2f}"),
            ("Sortino", "sortino", "{:.2f}"),
            ("Max DD", "max_drawdown", "{:.1%}"),
            ("Final $", "final_equity", "${:,.2f}"),
            ("Avg PnL $", "avg_pnl_usd", "${:+.2f}"),
        ]

        for label, key, fmt in metrics_keys:
            v_h4 = results["H4"][1].get(key, 0)
            v_d1 = results["D1"][1].get(key, 0)
            s_h4 = fmt.format(v_h4)
            s_d1 = fmt.format(v_d1)
            print(f"  {label:<16s} │ {s_h4:>20s} │ {s_d1:>20s}")

        # Trail breakdown
        for tf in ("H4", "D1"):
            trades = results[tf][2]
            trailed = [t for t in trades if t.trail_steps > 0]
            trail_pct = len(trailed) / len(trades) * 100 if trades else 0
            avg_s = (
                sum(t.trail_steps for t in trailed) / len(trailed)
                if trailed else 0
            )
            print(
                f"\n  {tf} trailing : {len(trailed)} trades trailés "
                f"({trail_pct:.0f}%) | steps moy: {avg_s:.1f}"
            )

        print(f"\n{'═' * 70}\n")

        # Rapports détaillés + graphiques
        for tf in ("H4", "D1"):
            res, m, raw, cfg = results[tf]
            print_report(res, m, raw, cfg)
            generate_chart(res, m, raw, cfg, show=not args.no_show)

        return

    # ── Mode simple (un seul timeframe) ────────────────────────────────────
    if args.timeframe == "1d":
        lookback = max(1, args.lookback_hours // 24)
        interval = "1d"
    else:
        lookback = max(1, args.lookback_hours // 4)
        interval = "4h"

    cfg = DipBuyConfig(
        initial_balance=args.balance,
        drop_threshold=args.threshold,
        sl_pct=args.sl,
        tp_pct=args.tp,
        trail_step_pct=args.step,
        interval=interval,
        lookback_bars=lookback,
    )

    logger.info("📥 Téléchargement données %s…", interval.upper())
    candles = download_all_pairs(pairs, start, end, interval=interval)

    print(f"\n{'─' * 60}")
    print(f"  📊 DIP BUY — {interval.upper()}")
    print(f"{'─' * 60}")

    engine = DipBuyEngine(candles, cfg)
    result = engine.run()
    metrics = compute_metrics(result)

    print_report(result, metrics, engine.closed_trades, cfg)
    generate_chart(result, metrics, engine.closed_trades, cfg, show=not args.no_show)

    if args.sensitivity:
        run_sensitivity(candles, cfg)


if __name__ == "__main__":
    main()
