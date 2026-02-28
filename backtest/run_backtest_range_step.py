#!/usr/bin/env python
"""
Backtest comparatif : RANGE classique vs RANGE RSI + Step Trailing.

Variante :
  Entrée :
    - Dow Theory range (NEUTRAL) + prix dans la buy zone
    - RSI(14) < 35  (seul filtre technique)
  SL :
    - Classique serré : range_low × (1 - buffer%)
  TP :
    - Step trailing à paliers de 1% :
      · Palier 0 : SL = range_low, cible = entry + 1%
      · Palier 1 (prix atteint +1%) : SL → entry (breakeven), cible → +2%
      · Palier 2 (prix atteint +2%) : SL → entry + 1%, cible → +3%
      · Palier N : SL = entry × (1 + step% × (N-1)), cible = entry × (1 + step% × (N+1))
    → Dès le premier palier, on ne peut plus perdre.

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
logger = logging.getLogger("range_step")


# ════════════════════════════════════════════════════════════════════════════════
# Indicateur RSI
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


# ════════════════════════════════════════════════════════════════════════════════
# Configuration & structures
# ════════════════════════════════════════════════════════════════════════════════


@dataclass
class StepConfig:
    """Paramètres de la stratégie Range RSI + Step Trailing."""
    initial_balance: float = 1000.0

    # ── RSI ──
    rsi_period: int = 14
    rsi_threshold: float = 35.0

    # ── Step trailing ──
    step_pct: float = 0.01           # 1% par palier

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
    range_sl_buffer_pct: float = 0.003

    # ── Compound ──
    compound: bool = False


@dataclass
class StepTrade:
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
    steps_reached: int = 0             # combien de paliers atteints
    highest_reached: float = 0.0
    bars_held: int = 0


@dataclass
class StepPosition:
    symbol: str
    side: OrderSide
    entry_price: float
    initial_sl: float                  # SL classique (range_low - buffer)
    current_sl: float                  # SL courant (évolue avec les paliers)
    size: float
    entry_time: int
    # Step trailing state
    steps_completed: int = 0           # nombre de paliers franchis
    highest_since_entry: float = 0.0
    bars_count: int = 0
    # Snapshot
    rsi_at_entry: Optional[float] = None


# ════════════════════════════════════════════════════════════════════════════════
# Moteur de backtest
# ════════════════════════════════════════════════════════════════════════════════


class RangeStepEngine:
    """Backtest RANGE + RSI filter + Step Trailing (1% paliers)."""

    def __init__(
        self,
        candles_by_symbol: dict[str, list[Candle]],
        config: StepConfig,
    ) -> None:
        self.candles = candles_by_symbol
        self.cfg = config
        self.pairs = list(candles_by_symbol.keys())

        self.cash: float = config.initial_balance
        self.positions: list[StepPosition] = []
        self.closed_trades: list[StepTrade] = []
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
            "📊 Range Step : %d barres, %d paires, $%.0f, step=%.1f%%",
            total, len(self.pairs), self.cfg.initial_balance,
            self.cfg.step_pct * 100,
        )

        for i, ts in enumerate(timeline):
            for sym in self.pairs:
                c = self._idx.get((sym, ts))
                if c:
                    self.last_close[sym] = c.close

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

    # ── Gestion des exits : Step Trailing ──────────────────────────────────

    def _manage_exits(self, ts: int) -> None:
        to_close: list[tuple[StepPosition, float, str]] = []

        for pos in self.positions:
            c = self._idx.get((pos.symbol, ts))
            if c is None:
                continue

            pos.bars_count += 1

            # Track highest
            if c.high > pos.highest_since_entry:
                pos.highest_since_entry = c.high

            # ── SL check (prioritaire — worst-case) ──
            if c.low <= pos.current_sl:
                if pos.steps_completed == 0:
                    reason = "STEP_SL"       # SL initial touché → perte
                else:
                    reason = "STEP_TRAIL"    # trailing touché → gain verrouillé
                to_close.append((pos, pos.current_sl, reason))
                continue

            # ── Step trailing : vérifier si le prix a franchi le palier suivant ──
            # On vérifie avec le HIGH (le prix a bien atteint ce niveau)
            next_step = pos.steps_completed + 1
            next_target = pos.entry_price * (1 + self.cfg.step_pct * next_step)

            while c.high >= next_target:
                # Palier franchi !
                pos.steps_completed = next_step

                # Déplacer le SL : toujours 1 step derrière
                if next_step == 1:
                    # Premier palier : SL → breakeven (entry)
                    pos.current_sl = pos.entry_price
                else:
                    # Paliers suivants : SL → entry + (N-1) × step
                    pos.current_sl = pos.entry_price * (
                        1 + self.cfg.step_pct * (next_step - 1)
                    )

                # Vérifier le palier suivant (le prix peut avoir traversé
                # plusieurs paliers dans une seule bougie)
                next_step = pos.steps_completed + 1
                next_target = pos.entry_price * (1 + self.cfg.step_pct * next_step)

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

            # Anti-lookahead : entrer si open dans la buy zone
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
            pos = StepPosition(
                symbol=sym,
                side=OrderSide.BUY,
                entry_price=entry_price,
                initial_sl=sig["sl_price"],
                current_sl=sig["sl_price"],
                size=size,
                entry_time=ts,
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
            )
            if len(vis) < min_len:
                continue

            # ── 1) Dow Theory : détecter le range ──
            swings = detect_swings(vis, self.cfg.swing_lookback)
            if not swings:
                continue
            trend = determine_trend(swings, sym)
            self.trends[sym] = trend

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

            # ── 2) RSI ──
            closes = [c.close for c in vis]
            rsi_val = rsi(closes, self.cfg.rsi_period)
            if rsi_val is None:
                continue

            # ── 3) Conditions d'entrée : buy zone + RSI < seuil ──
            buy_zone = rs.range_low * (1 + self.cfg.range_entry_buffer_pct)
            current_candle = vis[-1]

            in_buy_zone = current_candle.close <= buy_zone
            rsi_ok = rsi_val < self.cfg.rsi_threshold

            if in_buy_zone and rsi_ok:
                # SL classique serré
                sl_price = rs.range_low * (1 - self.cfg.range_sl_buffer_pct)

                self._pending[sym] = {
                    "side": OrderSide.BUY,
                    "buy_zone": buy_zone,
                    "sl_price": sl_price,
                    "rsi": rsi_val,
                }
            else:
                self._pending[sym] = None

    # ── Close position ─────────────────────────────────────────────────────

    def _close_position(
        self, pos: StepPosition, exit_price: float, ts: int, reason: str,
    ) -> None:
        revenue = pos.size * exit_price * (1 - self.cfg.fee_pct)
        pnl = pos.size * (
            exit_price * (1 - self.cfg.fee_pct)
            - pos.entry_price * (1 + self.cfg.fee_pct)
        )
        self.cash += revenue
        pnl_pct = pnl / (pos.size * pos.entry_price) if pos.entry_price else 0

        self.closed_trades.append(StepTrade(
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
            steps_reached=pos.steps_completed,
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
    m_step: dict,
    res_classic: BacktestResult,
    res_step: BacktestResult,
    step_trades: list[StepTrade],
    step_pct: float,
) -> None:
    sep = "═" * 80
    print(f"\n{sep}")
    print(f"  🔬 COMPARAISON : RANGE CLASSIQUE vs RANGE RSI + Step Trail ({step_pct:.0%})")
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

    label_b = "RSI+Step"
    print(f"\n  {'Métrique':<22s} │ {'CLASSIQUE':>14s} │ {label_b:>14s} │ {'Δ':>10s}")
    print("  " + "─" * 70)

    for label, key, fmt in metrics_list:
        v_c = m_classic.get(key, 0)
        v_s = m_step.get(key, 0)

        s_c = fmt.format(v_c) if isinstance(v_c, (int, float)) else str(v_c)
        s_s = fmt.format(v_s) if isinstance(v_s, (int, float)) else str(v_s)

        if isinstance(v_c, (int, float)) and isinstance(v_s, (int, float)):
            delta = v_s - v_c
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
            better = "✅" if v_s > v_c else ("❌" if v_s < v_c else "")
        elif key in ("total_return", "cagr", "sharpe", "sortino", "win_rate",
                      "profit_factor", "final_equity", "avg_pnl_usd", "avg_pnl_pct"):
            better = "✅" if v_s > v_c else ("❌" if v_s < v_c else "")

        print(f"  {label:<22s} │ {s_c:>14s} │ {s_s:>14s} │ {s_d:>8s} {better}")

    if step_trades:
        print(f"\n  📊 Détail RSI + Step Trailing ({step_pct:.0%})")
        print("  " + "─" * 70)

        # Par motif de sortie
        exit_reasons: dict[str, list[StepTrade]] = {}
        for t in step_trades:
            exit_reasons.setdefault(t.exit_reason, []).append(t)

        for reason, tlist in sorted(exit_reasons.items()):
            n = len(tlist)
            wins = sum(1 for t in tlist if t.pnl_usd > 0)
            pnl = sum(t.pnl_usd for t in tlist)
            wr = wins / n if n else 0
            avg_bars = sum(t.bars_held for t in tlist) / n if n else 0
            avg_steps = sum(t.steps_reached for t in tlist) / n if n else 0
            print(
                f"  {reason:16s} : {n:3d} trades | WR {wr:.0%} | "
                f"PnL ${pnl:+.2f} | Avg {avg_bars:.0f} barres | "
                f"Avg steps: {avg_steps:.1f}"
            )

        # Distribution des paliers
        print(f"\n  🪜 Distribution des paliers atteints :")
        max_step = max((t.steps_reached for t in step_trades), default=0)
        for s in range(max_step + 1):
            count = sum(1 for t in step_trades if t.steps_reached == s)
            pnl = sum(t.pnl_usd for t in step_trades if t.steps_reached == s)
            if count > 0:
                bar = "█" * max(1, int(count / max(1, len(step_trades)) * 40))
                label_s = "initial SL" if s == 0 else f"+{s * step_pct:.0%} verrouillé"
                print(
                    f"     Step {s:2d} ({label_s:>16s}) : "
                    f"{count:4d} trades | PnL ${pnl:+8.2f} {bar}"
                )

        # RSI moyen
        rsi_vals = [t.rsi_at_entry for t in step_trades if t.rsi_at_entry is not None]
        if rsi_vals:
            print(f"\n  RSI moyen à l'entrée  : {sum(rsi_vals)/len(rsi_vals):.1f}")

        # Trades ayant atteint au moins 1 palier
        protected = [t for t in step_trades if t.steps_reached >= 1]
        if protected:
            prot_pnl = sum(t.pnl_usd for t in protected)
            prot_wr = sum(1 for t in protected if t.pnl_usd > 0) / len(protected)
            print(f"\n  🛡️  Trades protégés (≥ 1 palier) :")
            print(f"     Nombre            : {len(protected)} / {len(step_trades)} "
                  f"({len(protected)/len(step_trades):.0%})")
            print(f"     WR                : {prot_wr:.0%}")
            print(f"     PnL total         : ${prot_pnl:+.2f}")

        # Top 5 trades
        sorted_trades = sorted(step_trades, key=lambda t: t.pnl_usd, reverse=True)
        print(f"\n  🏆 Top 5 trades :")
        for t in sorted_trades[:5]:
            print(
                f"     {t.symbol:10s} | {t.pnl_usd:+8.2f} USD ({t.pnl_pct:+.1%}) | "
                f"{t.bars_held} barres | {t.steps_reached} steps | RSI={t.rsi_at_entry:.0f} | "
                f"Exit: {t.exit_reason}"
            )

        print(f"\n  💩 Bottom 5 trades :")
        for t in sorted_trades[-5:]:
            print(
                f"     {t.symbol:10s} | {t.pnl_usd:+8.2f} USD ({t.pnl_pct:+.1%}) | "
                f"{t.bars_held} barres | {t.steps_reached} steps | RSI={t.rsi_at_entry:.0f} | "
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
        description="TradeX Backtest — RANGE classique vs RANGE RSI + Step Trailing"
    )
    parser.add_argument("--balance", type=float, default=1000.0)
    parser.add_argument("--start", default="2022-02-20")
    parser.add_argument("--end", default="2026-02-20")
    parser.add_argument("--pairs", type=int, default=20)
    parser.add_argument("--rsi-threshold", type=float, default=35.0)
    parser.add_argument("--step-pct", type=float, default=0.01,
                        help="Taille du palier (ex: 0.01 = 1%%)")
    parser.add_argument("--no-show", action="store_true")
    args = parser.parse_args()

    start = datetime.strptime(args.start, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    end = datetime.strptime(args.end, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    pairs = PAIRS_20[:args.pairs]

    print(f"\n{'═'*80}")
    print(f"  🔬 RANGE CLASSIQUE vs RANGE RSI + Step Trailing — A/B Test")
    print(f"  📅 {start:%Y-%m-%d} → {end:%Y-%m-%d} | {len(pairs)} paires | ${args.balance:,.0f}")
    print(f"  RSI < {args.rsi_threshold} | SL classique (serré)")
    print(f"  Step trailing : paliers de {args.step_pct:.0%}")
    print(f"  → Dès +{args.step_pct:.0%} : SL → breakeven (plus de perte possible)")
    print(f"  → +{2*args.step_pct:.0%} : SL → +{args.step_pct:.0%} verrouillé, etc.")
    print(f"{'═'*80}\n")

    # ── Données ──
    logger.info("📥 Téléchargement des données…")
    candles = download_all_pairs(pairs, start, end, interval="4h")

    # ═══════════════════════════════════════════════════════════════════════
    # A) RANGE CLASSIQUE
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
    # B) RANGE RSI + Step Trailing
    # ═══════════════════════════════════════════════════════════════════════
    print("\n" + "─" * 60)
    print(f"  🅱️  RANGE RSI < {args.rsi_threshold} + Step Trailing ({args.step_pct:.0%})")
    print("─" * 60)

    cfg_step = StepConfig(
        initial_balance=args.balance,
        rsi_threshold=args.rsi_threshold,
        step_pct=args.step_pct,
    )
    engine_step = RangeStepEngine(candles, cfg_step)
    result_step = engine_step.run()
    m_step = compute_metrics(result_step)

    # ═══════════════════════════════════════════════════════════════════════
    # Comparaison A/B
    # ═══════════════════════════════════════════════════════════════════════
    print_comparison(
        m_classic, m_step, result_classic, result_step,
        engine_step.closed_trades, args.step_pct,
    )

    # ── Graphiques ──
    from backtest.report import generate_report
    print("  📊 Graphique CLASSIQUE :")
    generate_report(result_classic, m_classic, show=False)
    print("  📊 Graphique RSI+Step :")
    generate_report(result_step, m_step, show=not args.no_show)


if __name__ == "__main__":
    main()
