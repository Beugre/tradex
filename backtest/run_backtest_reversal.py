#!/usr/bin/env python
"""
Backtest Extreme Oversold Reversal — Long Liquidation Bounce.

🧠 Logique :
   On NE trade PAS la tendance. On trade les EXCÈS.
   En bear market, les liquidations sont exagérées → rebonds mécaniques.

   Conditions d'entrée (ALL required) :
   1) Dump détecté : bougie rouge > dump_atr_mult × ATR (défaut 2×)
   2) Volume spike : volume > vol_spike_mult × SMA volume (défaut 2.5×)
   3) RSI extrême : RSI < rsi_threshold (défaut 20)
   4) Distance VWAP : prix < VWAP - vwap_dist_mult × σ (défaut 2σ)
   5) Confirmation : au moins une de ces conditions :
      a) Bougie verte après le dump (clôture > open)
      b) Mèche basse significative (> 50% du range)

   Sortie :
   - TP court : retour VWAP OU tp_pct fixe (1–3%)
   - SL serré : sl_atr_mult × ATR sous l'entrée
   - Pas de trailing — on prend le rebond rapide et on sort
   - Time stop : max_holding_bars (éviter l'immobilisation)

   C'est du scalping-swing, pas du swing large.
   On cherche 1–2% propre, faible exposition, trades rapides.

Usage :
  PYTHONPATH=. .venv/bin/python backtest/run_backtest_reversal.py --no-show
  PYTHONPATH=. .venv/bin/python backtest/run_backtest_reversal.py --months 24 --sensitivity --no-show
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
from backtest.data_loader import download_all_pairs, download_btc_d1
from backtest.simulator import BacktestResult, Trade, EquityPoint
from backtest.metrics import compute_metrics

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("reversal")

OUTPUT_DIR = Path(__file__).parent / "output"


# ════════════════════════════════════════════════════════════════════════════════
# Indicateurs
# ════════════════════════════════════════════════════════════════════════════════


def compute_rsi(closes: list[float], period: int = 14) -> Optional[float]:
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
    return 100.0 - 100.0 / (1.0 + avg_gain / avg_loss)


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


def compute_vwap(candles: list[Candle], period: int = 20) -> Optional[tuple[float, float]]:
    """VWAP rolling + écart-type σ sur `period` bougies.

    Retourne (vwap, sigma) ou None.
    En l'absence de données tick, on approx avec (H+L+C)/3 × Volume.
    """
    if len(candles) < period:
        return None
    window = candles[-period:]
    cum_vol = 0.0
    cum_tp_vol = 0.0
    for c in window:
        tp = (c.high + c.low + c.close) / 3.0
        cum_vol += c.volume
        cum_tp_vol += tp * c.volume
    if cum_vol <= 0:
        return None
    vwap = cum_tp_vol / cum_vol

    # Sigma = écart-type pondéré par volume
    var_sum = 0.0
    for c in window:
        tp = (c.high + c.low + c.close) / 3.0
        var_sum += c.volume * (tp - vwap) ** 2
    sigma = (var_sum / cum_vol) ** 0.5 if cum_vol > 0 else 0.0

    return vwap, sigma


def volume_sma(volumes: list[float], period: int = 20) -> Optional[float]:
    if len(volumes) < period:
        return None
    return sum(volumes[-period:]) / period


# ════════════════════════════════════════════════════════════════════════════════
# Config
# ════════════════════════════════════════════════════════════════════════════════


@dataclass
class ReversalConfig:
    initial_balance: float = 1000.0

    # ── Détection du dump ──
    atr_period: int = 14
    dump_atr_mult: float = 2.0        # Bougie rouge > N × ATR = dump
    vol_period: int = 20
    vol_spike_mult: float = 2.5       # Volume > N × SMA = spike

    # ── RSI extrême ──
    rsi_period: int = 14
    rsi_threshold: float = 20.0       # RSI < seuil = oversold extrême

    # ── VWAP distance ──
    vwap_period: int = 20             # Rolling VWAP period
    vwap_dist_mult: float = 2.0       # Prix < VWAP - N × σ
    use_vwap_filter: bool = True

    # ── Confirmation ──
    require_confirmation: bool = True  # Green candle ou big lower wick
    wick_ratio_min: float = 0.50      # Mèche basse > 50% du range

    # ── Gestion de position ──
    sl_atr_mult: float = 1.0          # SL = entry - ATR × mult (serré)
    tp_pct: float = 0.02              # TP fixe = +2%
    tp_at_vwap: bool = True           # OU TP au retour VWAP (le premier atteint)
    max_holding_bars: int = 48        # Time stop (48×15min=12h par défaut)
    use_trail: bool = False           # Pas de trailing par défaut

    # ── Risk ──
    risk_percent: float = 0.02
    max_position_pct: float = 0.25    # Faible exposition
    max_simultaneous: int = 3
    fee_pct: float = 0.00075
    slippage_pct: float = 0.001

    # ── Cooldown ──
    cooldown_bars: int = 2            # Min 2 barres entre trades
    max_consec_losses: int = 3
    cooldown_after_losses: int = 6

    # ── Macro filter ──
    use_macro_filter: bool = False    # Uniquement en bear (BTC < EMA200)
    macro_ema_period: int = 200

    # ── Timeframe ──
    interval: str = "15m"             # Interval des bougies
    bar_duration_ms: int = 15 * 60 * 1000  # Durée d'une barre en ms
    candle_window: int = 100          # Plus large pour M15


# ════════════════════════════════════════════════════════════════════════════════
# Structures
# ════════════════════════════════════════════════════════════════════════════════


@dataclass
class RevPosition:
    symbol: str
    entry_price: float
    sl_price: float
    tp_price: float
    tp_vwap: Optional[float]          # VWAP target au moment de l'entrée
    size: float
    entry_time: int
    entry_bar_idx: int
    rsi_at_entry: float
    dump_size_atr: float              # Taille du dump en multiples d'ATR
    vol_spike: float                  # Ratio volume vs SMA
    best_price: float = 0.0


@dataclass
class RevTrade:
    symbol: str
    entry_price: float
    exit_price: float
    size: float
    entry_time: int
    exit_time: int
    pnl_usd: float
    pnl_pct: float
    exit_reason: str
    rsi_at_entry: float
    dump_size_atr: float
    vol_spike: float
    holding_bars: int


# ════════════════════════════════════════════════════════════════════════════════
# Moteur Reversal
# ════════════════════════════════════════════════════════════════════════════════


class ReversalEngine:
    """
    Extreme Oversold Reversal — Long liquidation bounces.
    On ne trade pas la tendance. On trade les excès.
    """

    def __init__(
        self,
        candles_by_symbol: dict[str, list[Candle]],
        config: ReversalConfig,
        btc_d1_candles: list[Candle] | None = None,
    ) -> None:
        self.candles = candles_by_symbol
        self.cfg = config
        self.pairs = list(candles_by_symbol.keys())

        self.cash: float = config.initial_balance
        self.positions: list[RevPosition] = []
        self.closed_trades: list[RevTrade] = []
        self.equity_curve: list[EquityPoint] = []

        self._consec_losses: dict[str, int] = {p: 0 for p in self.pairs}
        self._cooldown_until: dict[str, int] = {p: 0 for p in self.pairs}
        self._last_trade_ts: dict[str, int] = {p: 0 for p in self.pairs}
        self.last_close: dict[str, float] = {}
        self._bar_count: int = 0

        # Index rapide
        self._idx: dict[tuple[str, int], Candle] = {}
        self._ts_arrays: dict[str, list[int]] = {}  # timestamps triés par symbole
        for sym, clist in candles_by_symbol.items():
            self._ts_arrays[sym] = [c.timestamp for c in clist]
            for c in clist:
                self._idx[(sym, c.timestamp)] = c

        # Macro filter
        self._ema_mode: dict[int, bool] = {}
        self._ema_ts_sorted: list[int] = []
        self._btc_bullish: bool = True
        if config.use_macro_filter and btc_d1_candles:
            self._build_ema_lookup(btc_d1_candles)

    # ── Macro EMA200 ───────────────────────────────────────────────────────

    def _build_ema_lookup(self, d1_candles: list[Candle]) -> None:
        period = self.cfg.macro_ema_period
        if len(d1_candles) < period:
            logger.warning("⚠️ %d D1 < EMA%d — filtre désactivé", len(d1_candles), period)
            return
        closes = [c.close for c in d1_candles]
        sma_seed = sum(closes[:period]) / period
        k = 2.0 / (period + 1)
        ema_val = sma_seed
        for i in range(period, len(closes)):
            ema_val = closes[i] * k + ema_val * (1 - k)
            self._ema_mode[d1_candles[i].timestamp] = closes[i] > ema_val
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
            "⚡ Reversal Bot : %d barres, %d paires, $%.0f",
            total, len(self.pairs), self.cfg.initial_balance,
        )

        for i, ts in enumerate(timeline):
            self._bar_count = i
            for sym in self.pairs:
                c = self._idx.get((sym, ts))
                if c:
                    self.last_close[sym] = c.close

            self._update_macro(ts)
            self._manage_exits(ts, i)
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

    def _visible(self, symbol: str, ts: int) -> list[Candle]:
        import bisect
        clist = self.candles[symbol]
        ts_arr = self._ts_arrays[symbol]
        end = bisect.bisect_right(ts_arr, ts)
        start = max(0, end - self.cfg.candle_window)
        return clist[start:end]

    # ── Exits ──────────────────────────────────────────────────────────────

    def _manage_exits(self, ts: int, bar_idx: int) -> None:
        to_close: list[tuple[RevPosition, float, str]] = []

        for pos in self.positions:
            c = self._idx.get((pos.symbol, ts))
            if c is None:
                continue

            pos.best_price = max(pos.best_price, c.high)
            holding = bar_idx - pos.entry_bar_idx

            # ── SL ──
            if c.low <= pos.sl_price:
                to_close.append((pos, pos.sl_price, "SL"))
                continue

            # ── TP fixe ──
            if c.high >= pos.tp_price:
                to_close.append((pos, pos.tp_price, "TP"))
                continue

            # ── TP VWAP (retour au VWAP) ──
            if self.cfg.tp_at_vwap and pos.tp_vwap is not None:
                if c.high >= pos.tp_vwap:
                    exit_px = min(c.high, pos.tp_vwap)
                    to_close.append((pos, exit_px, "VWAP"))
                    continue

            # ── Time stop ──
            if holding >= self.cfg.max_holding_bars:
                to_close.append((pos, c.close, "TIME"))
                continue

        for pos, exit_px, reason in to_close:
            self._close_position(pos, exit_px, ts, reason, bar_idx)

    # ── Analyse — Détection des excès ──────────────────────────────────────

    def _analyze(self, ts: int) -> None:
        if len(self.positions) >= self.cfg.max_simultaneous:
            return

        # Macro filter : si activé, ne trader que quand BTC < EMA200 (bear)
        # MAIS pour le reversal, on peut aussi capter les rebonds en bull
        # Donc le macro filter est optionnel et inverse : on trade en bear seulement
        if self.cfg.use_macro_filter and self._btc_bullish:
            return

        for sym in self.pairs:
            if len(self.positions) >= self.cfg.max_simultaneous:
                break
            if any(p.symbol == sym for p in self.positions):
                continue

            # Cooldowns
            if ts < self._cooldown_until.get(sym, 0):
                continue
            bar_ms = self.cfg.bar_duration_ms
            if ts - self._last_trade_ts.get(sym, 0) < self.cfg.cooldown_bars * bar_ms:
                continue

            vis = self._visible(sym, ts)
            min_bars = max(
                self.cfg.atr_period + 2,
                self.cfg.rsi_period + 2,
                self.cfg.vol_period + 1,
                self.cfg.vwap_period + 1,
            )
            if len(vis) < min_bars:
                continue

            current = vis[-1]
            closes = [c.close for c in vis]
            volumes = [c.volume for c in vis]

            # ════════════════════════════════════════════════════════════════
            # 1) DUMP : bougie rouge > N × ATR
            # ════════════════════════════════════════════════════════════════
            atr_val = compute_atr(vis, self.cfg.atr_period)
            if atr_val is None or atr_val <= 0:
                continue

            candle_range = current.open - current.close  # > 0 si rouge
            is_dump = candle_range > self.cfg.dump_atr_mult * atr_val
            if not is_dump:
                # Aussi checker la bougie précédente (confirmation après le dump)
                if len(vis) >= 2:
                    prev = vis[-2]
                    prev_range = prev.open - prev.close
                    if prev_range > self.cfg.dump_atr_mult * atr_val:
                        # Le dump était la bougie précédente, on est sur la confirmation
                        is_dump = True
                        candle_range = prev_range
                    else:
                        continue
                else:
                    continue

            dump_atr_ratio = candle_range / atr_val

            # ════════════════════════════════════════════════════════════════
            # 2) VOLUME SPIKE : volume > N × SMA
            # ════════════════════════════════════════════════════════════════
            vol_avg = volume_sma(volumes[:-1], self.cfg.vol_period)  # SMA sans la bougie courante
            if vol_avg is None or vol_avg <= 0:
                continue
            # Prendre le max entre volume courant et précédent
            vol_current = max(current.volume, vis[-2].volume if len(vis) >= 2 else 0)
            vol_ratio = vol_current / vol_avg
            if vol_ratio < self.cfg.vol_spike_mult:
                continue

            # ════════════════════════════════════════════════════════════════
            # 3) RSI EXTRÊME : RSI < seuil
            # ════════════════════════════════════════════════════════════════
            rsi_val = compute_rsi(closes, self.cfg.rsi_period)
            if rsi_val is None or rsi_val >= self.cfg.rsi_threshold:
                continue

            # ════════════════════════════════════════════════════════════════
            # 4) DISTANCE VWAP (optionnel)
            # ════════════════════════════════════════════════════════════════
            vwap_val = None
            if self.cfg.use_vwap_filter:
                vwap_data = compute_vwap(vis, self.cfg.vwap_period)
                if vwap_data is None:
                    continue
                vwap_val, sigma = vwap_data
                if sigma <= 0:
                    continue
                # Prix doit être significativement sous le VWAP
                dist = (vwap_val - current.close) / sigma
                if dist < self.cfg.vwap_dist_mult:
                    continue  # Pas assez loin du VWAP
            else:
                # Sans VWAP filter, calculer quand même pour le TP
                vwap_data = compute_vwap(vis, self.cfg.vwap_period)
                if vwap_data:
                    vwap_val = vwap_data[0]

            # ════════════════════════════════════════════════════════════════
            # 5) CONFIRMATION : green candle ou mèche basse
            # ════════════════════════════════════════════════════════════════
            if self.cfg.require_confirmation:
                is_green = current.close > current.open
                total_range = current.high - current.low
                lower_wick = min(current.open, current.close) - current.low
                has_wick = total_range > 0 and (lower_wick / total_range) >= self.cfg.wick_ratio_min

                if not is_green and not has_wick:
                    continue

            # ════════════════════════════════════════════════════════════════
            # 6) ENTRÉE LONG
            # ════════════════════════════════════════════════════════════════
            entry_px = current.close * (1 + self.cfg.slippage_pct)
            sl_px = entry_px - atr_val * self.cfg.sl_atr_mult
            tp_px = entry_px * (1 + self.cfg.tp_pct)
            tp_vwap = vwap_val  # VWAP comme target alternatif

            if sl_px <= 0 or tp_px <= entry_px:
                continue

            # Sizing
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

            cost = size * entry_px * self.cfg.fee_pct
            if cost > self.cash:
                continue

            self.cash -= cost

            pos = RevPosition(
                symbol=sym,
                entry_price=entry_px,
                sl_price=sl_px,
                tp_price=tp_px,
                tp_vwap=tp_vwap,
                size=size,
                entry_time=ts,
                entry_bar_idx=self._bar_count,
                rsi_at_entry=rsi_val,
                dump_size_atr=dump_atr_ratio,
                vol_spike=vol_ratio,
                best_price=entry_px,
            )
            self.positions.append(pos)

    # ── Close position ─────────────────────────────────────────────────────

    def _close_position(
        self, pos: RevPosition, exit_px: float, ts: int, reason: str, bar_idx: int,
    ) -> None:
        fee = pos.size * exit_px * self.cfg.fee_pct
        pnl_raw = pos.size * (exit_px - pos.entry_price)
        pnl_net = pnl_raw - fee
        self.cash += pnl_net

        pnl_pct = pnl_net / (pos.size * pos.entry_price) if pos.entry_price else 0
        holding = bar_idx - pos.entry_bar_idx

        self.closed_trades.append(RevTrade(
            symbol=pos.symbol,
            entry_price=pos.entry_price,
            exit_price=exit_px,
            size=pos.size,
            entry_time=pos.entry_time,
            exit_time=ts,
            pnl_usd=pnl_net,
            pnl_pct=pnl_pct,
            exit_reason=reason,
            rsi_at_entry=pos.rsi_at_entry,
            dump_size_atr=pos.dump_size_atr,
            vol_spike=pos.vol_spike,
            holding_bars=holding,
        ))

        if pos in self.positions:
            self.positions.remove(pos)

        self._last_trade_ts[pos.symbol] = ts
        sym = pos.symbol
        if pnl_net < 0:
            self._consec_losses[sym] = self._consec_losses.get(sym, 0) + 1
            if self._consec_losses[sym] >= self.cfg.max_consec_losses:
                pause_ms = self.cfg.cooldown_after_losses * self.cfg.bar_duration_ms
                self._cooldown_until[sym] = ts + pause_ms
                self._consec_losses[sym] = 0
        else:
            self._consec_losses[sym] = 0

    def _close_remaining(self, last_ts: int) -> None:
        for pos in list(self.positions):
            px = self.last_close.get(pos.symbol, pos.entry_price)
            self._close_position(pos, px, last_ts, "END", self._bar_count)

    def _record_equity(self, ts: int) -> None:
        unrealized = 0.0
        for p in self.positions:
            price = self.last_close.get(p.symbol, p.entry_price)
            unrealized += p.size * (price - p.entry_price)
        self.equity_curve.append(EquityPoint(ts, self.cash + unrealized))


# ════════════════════════════════════════════════════════════════════════════════
# Rapport
# ════════════════════════════════════════════════════════════════════════════════


def print_report(result: BacktestResult, m: dict, raw: list[RevTrade], cfg: ReversalConfig) -> None:
    sep = "═" * 80
    print(f"\n{sep}")
    print(f"  ⚡ EXTREME OVERSOLD REVERSAL — Résultats")
    print(f"  📅 {result.start_date.date()} → {result.end_date.date()}")
    bar_h = cfg.bar_duration_ms / 3_600_000
    print(f"  ⏱️  Interval: {cfg.interval} | Dump > {cfg.dump_atr_mult}×ATR | Vol > {cfg.vol_spike_mult}×SMA | RSI < {cfg.rsi_threshold}")
    print(f"  TP: {cfg.tp_pct:.1%} + VWAP={'ON' if cfg.tp_at_vwap else 'OFF'} | SL: {cfg.sl_atr_mult}×ATR | MaxHold: {cfg.max_holding_bars}b ({cfg.max_holding_bars * bar_h:.0f}h)")
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

    # Par paire
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

    # Stats de timing
    if raw:
        avg_hold = sum(t.holding_bars for t in raw) / len(raw)
        wins = [t for t in raw if t.pnl_usd > 0]
        losses = [t for t in raw if t.pnl_usd <= 0]
        bar_h = cfg.bar_duration_ms / 3_600_000

        print(f"\n  ⏱️  Holding moy : {avg_hold:.1f} barres ({avg_hold * bar_h:.1f}h)")
        if wins:
            avg_hold_w = sum(t.holding_bars for t in wins) / len(wins)
            avg_rsi_w = sum(t.rsi_at_entry for t in wins) / len(wins)
            avg_dump_w = sum(t.dump_size_atr for t in wins) / len(wins)
            avg_vol_w = sum(t.vol_spike for t in wins) / len(wins)
            print(f"  🏆 Wins  : hold={avg_hold_w:.1f}b | RSI={avg_rsi_w:.1f} | dump={avg_dump_w:.1f}×ATR | vol={avg_vol_w:.1f}×")
        if losses:
            avg_hold_l = sum(t.holding_bars for t in losses) / len(losses)
            avg_rsi_l = sum(t.rsi_at_entry for t in losses) / len(losses)
            avg_dump_l = sum(t.dump_size_atr for t in losses) / len(losses)
            avg_vol_l = sum(t.vol_spike for t in losses) / len(losses)
            print(f"  ❌ Losses: hold={avg_hold_l:.1f}b | RSI={avg_rsi_l:.1f} | dump={avg_dump_l:.1f}×ATR | vol={avg_vol_l:.1f}×")

        best = max(raw, key=lambda t: t.pnl_usd)
        worst = min(raw, key=lambda t: t.pnl_usd)
        print(f"\n  🏅 Best  : {best.symbol} RSI={best.rsi_at_entry:.0f} dump={best.dump_size_atr:.1f}×ATR → ${best.pnl_usd:+.2f} ({best.pnl_pct:+.1%}) [{best.holding_bars}b]")
        print(f"  💀 Worst : {worst.symbol} RSI={worst.rsi_at_entry:.0f} dump={worst.dump_size_atr:.1f}×ATR → ${worst.pnl_usd:+.2f} ({worst.pnl_pct:+.1%}) [{worst.holding_bars}b]")

    print(f"\n{sep}\n")


# ════════════════════════════════════════════════════════════════════════════════
# Graphiques
# ════════════════════════════════════════════════════════════════════════════════


def generate_charts(result: BacktestResult, m: dict, raw: list[RevTrade],
                    cfg: ReversalConfig, show: bool = True) -> Path:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mticker

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(3, 1, figsize=(16, 14), height_ratios=[3, 1, 1.5])
    plt.style.use("seaborn-v0_8-whitegrid")

    eq = result.equity_curve
    dates = [datetime.fromtimestamp(p.timestamp / 1000, tz=timezone.utc) for p in eq]
    equities = [p.equity for p in eq]

    # 1. Equity
    ax1 = axes[0]
    ax1.plot(dates, equities, color="#FF9800", linewidth=1.5, alpha=0.9)
    ax1.fill_between(dates, result.initial_balance, equities, alpha=0.1, color="#FF9800")
    ax1.axhline(y=result.initial_balance, color="gray", linestyle="--", alpha=0.4)
    ax1.set_title(
        f"Extreme Oversold Reversal -- Equity\n"
        f"RSI<{cfg.rsi_threshold:.0f} | Dump>{cfg.dump_atr_mult}×ATR | "
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

    # 3. PnL par trade + scatter RSI
    ax3 = axes[2]
    if raw:
        trade_dates = [datetime.fromtimestamp(t.exit_time / 1000, tz=timezone.utc) for t in raw]
        pnls = [t.pnl_usd for t in raw]
        colors = ["#4CAF50" if p > 0 else "#F44336" for p in pnls]
        ax3.bar(trade_dates, pnls, color=colors, alpha=0.7, width=0.5)
        ax3.axhline(y=0, color="gray", linestyle="-", alpha=0.4)
    ax3.set_ylabel("PnL ($)")
    ax3.set_title("PnL par trade", fontsize=10)

    plt.tight_layout()
    chart_path = OUTPUT_DIR / "reversal_bot_backtest.png"
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
    base_cfg: ReversalConfig,
    btc_d1: list[Candle] | None = None,
) -> None:
    print("\n" + "═" * 105)
    print("  🔬 Analyse de sensibilité — Extreme Oversold Reversal")
    print("═" * 105)

    configs = [
        # ── RSI threshold ──
        ("RSI < 20",              {"rsi_threshold": 20}),
        ("RSI < 25",              {"rsi_threshold": 25}),
        ("RSI < 15",              {"rsi_threshold": 15}),
        ("RSI < 30",              {"rsi_threshold": 30}),

        # ── Dump intensity ──
        ("Dump > 2.0×ATR",        {"dump_atr_mult": 2.0}),
        ("Dump > 1.5×ATR",        {"dump_atr_mult": 1.5}),
        ("Dump > 2.5×ATR",        {"dump_atr_mult": 2.5}),
        ("Dump > 3.0×ATR",        {"dump_atr_mult": 3.0}),

        # ── Volume spike ──
        ("Vol > 2.0×SMA",         {"vol_spike_mult": 2.0}),
        ("Vol > 3.0×SMA",         {"vol_spike_mult": 3.0}),
        ("Vol > 1.5×SMA",         {"vol_spike_mult": 1.5}),

        # ── TP ──
        ("TP 1%",                 {"tp_pct": 0.01}),
        ("TP 2%",                 {"tp_pct": 0.02}),
        ("TP 3%",                 {"tp_pct": 0.03}),
        ("TP 5%",                 {"tp_pct": 0.05}),
        ("VWAP only (no % TP)",   {"tp_pct": 0.50, "tp_at_vwap": True}),

        # ── SL ──
        ("SL 0.5×ATR",            {"sl_atr_mult": 0.5}),
        ("SL 1.0×ATR",            {"sl_atr_mult": 1.0}),
        ("SL 1.5×ATR",            {"sl_atr_mult": 1.5}),
        ("SL 2.0×ATR",            {"sl_atr_mult": 2.0}),

        # ── Time stop (adapté au timeframe) ──
        ("Hold max 24 bars",      {"max_holding_bars": 24}),
        ("Hold max 48 bars",      {"max_holding_bars": 48}),
        ("Hold max 96 bars",      {"max_holding_bars": 96}),
        ("Hold max 192 bars",     {"max_holding_bars": 192}),

        # ── Confirmation ──
        ("No confirmation",       {"require_confirmation": False}),
        ("No VWAP filter",        {"use_vwap_filter": False}),

        # ── Relaxed combo ──
        ("Relaxed: RSI<25 dump>1.5 vol>2.0",
            {"rsi_threshold": 25, "dump_atr_mult": 1.5, "vol_spike_mult": 2.0}),
        ("Ultra strict: RSI<15 dump>3.0 vol>3.0",
            {"rsi_threshold": 15, "dump_atr_mult": 3.0, "vol_spike_mult": 3.0}),

        # ── Macro filter ──
        ("Macro filter ON",       {"use_macro_filter": True}),
        ("Macro filter OFF",      {"use_macro_filter": False}),
    ]

    header = (
        f"  {'Config':<35s} │ {'Trades':>6s} │ {'WR':>5s} │ "
        f"{'PF':>6s} │ {'Return':>8s} │ {'Sharpe':>6s} │ "
        f"{'MaxDD':>7s} │ {'AvgHold':>7s} │ {'Final':>10s}"
    )
    print(f"\n{header}")
    print("  " + "─" * (len(header) - 2))

    for label, overrides in configs:
        cfg = ReversalConfig(initial_balance=base_cfg.initial_balance)
        # Copy base
        for attr in (
            "atr_period", "dump_atr_mult", "vol_period", "vol_spike_mult",
            "rsi_period", "rsi_threshold", "vwap_period", "vwap_dist_mult",
            "use_vwap_filter", "require_confirmation", "wick_ratio_min",
            "sl_atr_mult", "tp_pct", "tp_at_vwap", "max_holding_bars",
            "use_trail", "risk_percent", "fee_pct", "slippage_pct",
            "use_macro_filter", "interval", "bar_duration_ms",
        ):
            setattr(cfg, attr, getattr(base_cfg, attr))
        for k, v in overrides.items():
            setattr(cfg, k, v)

        engine = ReversalEngine(candles, cfg, btc_d1_candles=btc_d1)
        res = engine.run()
        m = compute_metrics(res)

        avg_hold = (
            sum(t.holding_bars for t in engine.closed_trades) / max(1, len(engine.closed_trades))
        )

        row = (
            f"  {label:<35s} │ {m['n_trades']:>6d} │ {m['win_rate']:>4.0%} │ "
            f"{m['profit_factor']:>6.2f} │ {m['total_return']:>+7.1%} │ "
            f"{m['sharpe']:>6.2f} │ {m['max_drawdown']:>6.1%} │ "
            f"{avg_hold:>5.1f}b │ ${m['final_equity']:>9,.2f}"
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
    parser = argparse.ArgumentParser(description="Backtest Extreme Oversold Reversal")
    parser.add_argument("--months", type=int, default=12)
    parser.add_argument("--balance", type=float, default=1000.0)
    parser.add_argument("--pairs", type=int, default=20)
    parser.add_argument("--no-show", action="store_true")
    parser.add_argument("--sensitivity", action="store_true")
    parser.add_argument("--rsi", type=float, default=20.0, help="RSI threshold")
    parser.add_argument("--dump-atr", type=float, default=2.0, help="Dump min en ATR mult")
    parser.add_argument("--vol-spike", type=float, default=2.5, help="Volume spike mult")
    parser.add_argument("--tp-pct", type=float, default=0.02, help="TP fixe (%%)")
    parser.add_argument("--sl-atr", type=float, default=1.0, help="SL ATR mult")
    parser.add_argument("--max-hold", type=int, default=48, help="Max holding bars (48×15min=12h)")
    parser.add_argument("--no-vwap", action="store_true", help="Désactiver filtre VWAP")
    parser.add_argument("--no-confirm", action="store_true", help="Pas de confirmation")
    parser.add_argument("--macro-filter", action="store_true", help="Macro BTC EMA200 filter")
    parser.add_argument("--interval", type=str, default="15m", help="Interval (15m, 1h, 4h)")
    args = parser.parse_args()

    # Map interval → durée en ms
    INTERVAL_MS = {
        "1m": 60_000, "3m": 180_000, "5m": 300_000,
        "15m": 900_000, "30m": 1_800_000,
        "1h": 3_600_000, "2h": 7_200_000, "4h": 14_400_000,
    }
    bar_ms = INTERVAL_MS.get(args.interval, 900_000)

    end = datetime.now(tz=timezone.utc)
    start = end - timedelta(days=args.months * 30)
    pairs = PAIRS_20[:args.pairs]

    print(f"\n{'═' * 80}")
    print(f"  ⚡ EXTREME OVERSOLD REVERSAL — Backtest")
    print(f"  📅 {start.date()} → {end.date()} ({args.months} mois)")
    print(f"  💰 ${args.balance:,.0f} | {len(pairs)} paires | ⏱️  {args.interval}")
    print(f"  RSI < {args.rsi} | Dump > {args.dump_atr}×ATR | Vol > {args.vol_spike}×SMA")
    print(f"  TP: {args.tp_pct:.1%} + VWAP={'ON' if not args.no_vwap else 'OFF'} | SL: {args.sl_atr}×ATR")
    print(f"  MaxHold: {args.max_hold} bars ({args.max_hold * bar_ms / 3_600_000:.0f}h)")
    print(f"  Confirm: {'OFF' if args.no_confirm else 'ON'} | Macro: {'ON' if args.macro_filter else 'OFF'}")
    print(f"{'═' * 80}\n")

    logger.info("📥 Téléchargement %s…", args.interval)
    candles = download_all_pairs(pairs, start, end, interval=args.interval)

    btc_d1 = None
    if args.macro_filter or args.sensitivity:
        logger.info("📥 BTC D1 pour macro filter…")
        btc_d1 = download_btc_d1(start, end)

    cfg = ReversalConfig(
        initial_balance=args.balance,
        rsi_threshold=args.rsi,
        dump_atr_mult=args.dump_atr,
        vol_spike_mult=args.vol_spike,
        tp_pct=args.tp_pct,
        sl_atr_mult=args.sl_atr,
        max_holding_bars=args.max_hold,
        use_vwap_filter=not args.no_vwap,
        require_confirmation=not args.no_confirm,
        use_macro_filter=args.macro_filter,
        interval=args.interval,
        bar_duration_ms=bar_ms,
    )

    print("\n" + "─" * 60)
    print("  ⚡ EXTREME OVERSOLD REVERSAL")
    print("─" * 60)

    engine = ReversalEngine(candles, cfg, btc_d1_candles=btc_d1)
    result = engine.run()
    metrics = compute_metrics(result)

    print_report(result, metrics, engine.closed_trades, cfg)
    generate_charts(result, metrics, engine.closed_trades, cfg, show=not args.no_show)

    if args.sensitivity:
        run_sensitivity(candles, cfg, btc_d1)


if __name__ == "__main__":
    main()
