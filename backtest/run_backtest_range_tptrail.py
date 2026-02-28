#!/usr/bin/env python
"""
Backtest comparatif : RANGE classique vs RANGE + Trailing à partir du TP.

Variante :
  Entrée : identique classique (Dow Theory range, buy zone)
  SL     : identique classique (range_low × (1 - buffer%))
  TP     : PAS de fermeture au range_mid — à la place :
           quand le prix atteint range_mid → activer trailing
           SL monte au breakeven, puis suit par paliers de 1%
           → Le trade peut capturer des mouvements au-delà du range_mid
"""

from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional

from dotenv import load_dotenv
load_dotenv()

from src.core.models import (
    Candle, OrderSide, RangeState, StrategyType,
    TrendDirection, TrendState,
)
from src.core.swing_detector import detect_swings
from src.core.trend_engine import determine_trend
from src.core.strategy_mean_rev import build_range_from_trend
from src.core.risk_manager import calculate_position_size
from backtest.data_loader import download_all_pairs
from backtest.simulator import (
    BacktestConfig, BacktestEngine, BacktestResult, Trade, EquityPoint,
)
from backtest.metrics import compute_metrics

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("range_tp_trail")


# ════════════════════════════════════════════════════════════════════════════════
@dataclass
class TpTrailConfig:
    initial_balance: float = 1000.0
    step_pct: float = 0.01
    risk_percent: float = 0.02
    max_position_pct: float = 0.30
    max_simultaneous_positions: int = 3
    fee_pct: float = 0.00075
    slippage_pct: float = 0.001
    swing_lookback: int = 3
    candle_window: int = 100
    range_width_min: float = 0.02
    range_entry_buffer_pct: float = 0.002
    range_sl_buffer_pct: float = 0.003
    range_cooldown_bars: int = 3
    compound: bool = False


@dataclass
class TpTrade:
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
    steps_reached: int = 0
    bars_held: int = 0
    tp_level: float = 0.0


@dataclass
class TpPosition:
    symbol: str
    side: OrderSide
    entry_price: float
    sl_price: float
    tp_level: float          # range_mid — seuil d'activation du trailing
    current_sl: float
    size: float
    entry_time: int
    trailing_active: bool = False
    steps_completed: int = 0
    highest_since_entry: float = 0.0
    bars_count: int = 0


# ════════════════════════════════════════════════════════════════════════════════
class RangeTpTrailEngine:
    """RANGE classique, mais au lieu de clôturer au TP → trailing."""

    def __init__(self, candles_by_symbol: dict[str, list[Candle]],
                 config: TpTrailConfig) -> None:
        self.candles = candles_by_symbol
        self.cfg = config
        self.pairs = list(candles_by_symbol.keys())

        self.cash = config.initial_balance
        self.positions: list[TpPosition] = []
        self.closed_trades: list[TpTrade] = []
        self.equity_curve: list[EquityPoint] = []

        self.trends: dict[str, Optional[TrendState]] = {p: None for p in self.pairs}
        self.ranges: dict[str, Optional[RangeState]] = {p: None for p in self.pairs}
        self.cooldown_until: dict[str, int] = {p: 0 for p in self.pairs}

        self._pending: dict[str, Optional[dict]] = {p: None for p in self.pairs}
        self.last_close: dict[str, float] = {}

        self._idx: dict[tuple[str, int], Candle] = {}
        for sym, clist in candles_by_symbol.items():
            for c in clist:
                self._idx[(sym, c.timestamp)] = c

    def run(self) -> BacktestResult:
        for mod in ("src.core.swing_detector", "src.core.trend_engine",
                     "src.core.strategy_mean_rev"):
            logging.getLogger(mod).setLevel(logging.WARNING)

        timeline = self._build_timeline()
        total = len(timeline)
        logger.info("📊 Range TP-Trail : %d barres, %d paires, $%.0f",
                     total, len(self.pairs), self.cfg.initial_balance)

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
                print(f"\r   ⏳ {i+1}/{total} ({100*(i+1)/total:.0f}%) "
                      f"| Equity: ${eq:,.2f} | Trades: {len(self.closed_trades)}",
                      end="", flush=True)
        print()

        self._close_remaining(timeline[-1])
        final_eq = self.equity_curve[-1].equity if self.equity_curve else self.cash

        trades = [
            Trade(symbol=t.symbol, strategy=StrategyType.RANGE, side=t.side,
                  entry_price=t.entry_price, exit_price=t.exit_price, size=t.size,
                  entry_time=t.entry_time, exit_time=t.exit_time,
                  pnl_usd=t.pnl_usd, pnl_pct=t.pnl_pct, exit_reason=t.exit_reason)
            for t in self.closed_trades
        ]

        return BacktestResult(
            trades=trades, equity_curve=self.equity_curve,
            initial_balance=self.cfg.initial_balance, final_equity=final_eq,
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

    # ── Exits ──────────────────────────────────────────────────────────────

    def _manage_exits(self, ts: int) -> None:
        to_close: list[tuple[TpPosition, float, str]] = []

        for pos in self.positions:
            c = self._idx.get((pos.symbol, ts))
            if c is None:
                continue

            pos.bars_count += 1
            if c.high > pos.highest_since_entry:
                pos.highest_since_entry = c.high

            # ── 1) SL — PRIORITAIRE ──
            if c.low <= pos.current_sl:
                if pos.trailing_active:
                    reason = "TP_TRAIL"
                else:
                    reason = "RANGE_SL"
                to_close.append((pos, pos.current_sl, reason))
                if not pos.trailing_active:
                    self.cooldown_until[pos.symbol] = (
                        ts + self.cfg.range_cooldown_bars * 4 * 3600 * 1000
                    )
                continue

            # ── 2) Activation trailing quand prix atteint TP (range_mid) ──
            if not pos.trailing_active and c.high >= pos.tp_level:
                pos.trailing_active = True
                # SL → TP (range_mid)
                pos.current_sl = pos.tp_level
                pos.steps_completed = 1

            # ── 3) Step trailing au-delà du TP ──
            if pos.trailing_active:
                next_step = pos.steps_completed + 1
                next_target = pos.tp_level * (1 + self.cfg.step_pct * (next_step - 1))

                while c.high >= next_target:
                    pos.steps_completed = next_step
                    # SL = TP + (N-1) × step%
                    pos.current_sl = pos.tp_level * (
                        1 + self.cfg.step_pct * (next_step - 1)
                    )
                    next_step = pos.steps_completed + 1
                    next_target = pos.tp_level * (1 + self.cfg.step_pct * (next_step - 1))

        for pos, exit_px, reason in to_close:
            self._close_position(pos, exit_px, ts, reason)

    # ── Pending entries ────────────────────────────────────────────────────

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

            if ts < self.cooldown_until.get(sym, 0):
                continue
            if c.open > sig["buy_zone"]:
                continue

            entry_price = c.open * (1 + self.cfg.slippage_pct)
            if sig["sl_price"] >= entry_price:
                continue
            if sig["tp_price"] <= entry_price:
                continue

            sizing_balance = self.cash if self.cfg.compound else self.cfg.initial_balance
            size = calculate_position_size(
                account_balance=sizing_balance, risk_percent=self.cfg.risk_percent,
                entry_price=entry_price, sl_price=sig["sl_price"],
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
            self.positions.append(TpPosition(
                symbol=sym, side=OrderSide.BUY,
                entry_price=entry_price, sl_price=sig["sl_price"],
                tp_level=sig["tp_price"], current_sl=sig["sl_price"],
                size=size, entry_time=ts,
                highest_since_entry=entry_price,
            ))
            self._pending[sym] = None

    # ── Analyse Dow (identique classique) ──────────────────────────────────

    def _analyze(self, ts: int) -> None:
        for sym in self.pairs:
            vis = self._visible(sym, ts)
            if len(vis) < 2 * self.cfg.swing_lookback + 1:
                continue

            swings = detect_swings(vis, self.cfg.swing_lookback)
            if not swings:
                continue
            trend = determine_trend(swings, sym)
            self.trends[sym] = trend

            if trend.direction != TrendDirection.NEUTRAL:
                self.ranges[sym] = None
                self._pending[sym] = None
                for pos in list(self.positions):
                    if pos.symbol == sym:
                        px = self.last_close.get(sym, pos.entry_price)
                        self._close_position(pos, px, ts, "TREND_BREAK")
                continue

            rs = build_range_from_trend(trend, self.cfg.range_width_min)
            if rs is None:
                self._pending[sym] = None
                continue
            self.ranges[sym] = rs

            buy_zone = rs.range_low * (1 + self.cfg.range_entry_buffer_pct)
            sl_price = rs.range_low * (1 - self.cfg.range_sl_buffer_pct)

            self._pending[sym] = {
                "side": OrderSide.BUY, "buy_zone": buy_zone,
                "sl_price": sl_price, "tp_price": rs.range_mid,
            }

    # ── Close / equity ─────────────────────────────────────────────────────

    def _close_position(self, pos: TpPosition, exit_price: float,
                        ts: int, reason: str) -> None:
        revenue = pos.size * exit_price * (1 - self.cfg.fee_pct)
        pnl = pos.size * (
            exit_price * (1 - self.cfg.fee_pct)
            - pos.entry_price * (1 + self.cfg.fee_pct)
        )
        self.cash += revenue
        pnl_pct = pnl / (pos.size * pos.entry_price) if pos.entry_price else 0

        self.closed_trades.append(TpTrade(
            symbol=pos.symbol, side=pos.side,
            entry_price=pos.entry_price, exit_price=exit_price,
            size=pos.size, entry_time=pos.entry_time, exit_time=ts,
            pnl_usd=pnl, pnl_pct=pnl_pct, exit_reason=reason,
            steps_reached=pos.steps_completed, bars_held=pos.bars_count,
            tp_level=pos.tp_level,
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
# Rapport
# ════════════════════════════════════════════════════════════════════════════════

def print_comparison(m_classic: dict, m_trail: dict,
                     res_classic: BacktestResult, res_trail: BacktestResult,
                     trail_trades: list[TpTrade]) -> None:
    sep = "═" * 80
    print(f"\n{sep}")
    print(f"  🔬 COMPARAISON : RANGE CLASSIQUE vs RANGE Trailing après TP")
    print(f"  📅 {res_classic.start_date:%b %Y} → {res_classic.end_date:%b %Y}")
    print(f"  🪙 {len(res_classic.pairs)} paires | Capital: ${res_classic.initial_balance:,.0f}")
    print(f"  📐 SL classique | Pas de TP fixe → trailing activé au range_mid")
    print(sep)

    metrics = [
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

    print(f"\n  {'Métrique':<22s} │ {'CLASSIQUE':>14s} │ {'Trail@TP':>14s} │ {'Δ':>10s}")
    print("  " + "─" * 70)

    for label, key, fmt in metrics:
        v_c = m_classic.get(key, 0)
        v_t = m_trail.get(key, 0)
        s_c = fmt.format(v_c) if isinstance(v_c, (int, float)) else str(v_c)
        s_t = fmt.format(v_t) if isinstance(v_t, (int, float)) else str(v_t)

        if isinstance(v_c, (int, float)) and isinstance(v_t, (int, float)):
            delta = v_t - v_c
            if key in ("total_return", "cagr", "max_drawdown", "win_rate", "avg_pnl_pct"):
                s_d = f"{delta:+.1%}"
            elif key == "n_trades":
                s_d = f"{delta:+d}"
            else:
                s_d = f"{delta:+.2f}"
        else:
            s_d = ""

        better = ""
        if key == "max_drawdown":
            better = "✅" if v_t > v_c else ("❌" if v_t < v_c else "")
        elif key in ("total_return", "cagr", "sharpe", "sortino", "win_rate",
                      "profit_factor", "final_equity", "avg_pnl_usd", "avg_pnl_pct"):
            better = "✅" if v_t > v_c else ("❌" if v_t < v_c else "")

        print(f"  {label:<22s} │ {s_c:>14s} │ {s_t:>14s} │ {s_d:>8s} {better}")

    if trail_trades:
        print(f"\n  📊 Détail Trailing après TP")
        print("  " + "─" * 70)

        exit_reasons: dict[str, list[TpTrade]] = {}
        for t in trail_trades:
            exit_reasons.setdefault(t.exit_reason, []).append(t)

        for reason, tlist in sorted(exit_reasons.items()):
            n = len(tlist)
            wins = sum(1 for t in tlist if t.pnl_usd > 0)
            pnl = sum(t.pnl_usd for t in tlist)
            wr = wins / n if n else 0
            avg_bars = sum(t.bars_held for t in tlist) / n if n else 0
            avg_steps = sum(t.steps_reached for t in tlist) / n if n else 0
            print(f"  {reason:16s} : {n:3d} trades | WR {wr:.0%} | "
                  f"PnL ${pnl:+.2f} | Avg {avg_bars:.0f}b | Steps {avg_steps:.1f}")

        # Trades qui ont atteint le TP (trailing activé)
        activated = [t for t in trail_trades if t.steps_reached >= 1]
        not_act = [t for t in trail_trades if t.steps_reached == 0]
        if activated:
            act_pnl = sum(t.pnl_usd for t in activated)
            act_wr = sum(1 for t in activated if t.pnl_usd > 0) / len(activated)
            print(f"\n  🛡️  Trades avec trailing activé (atteint range_mid) :")
            print(f"     Nombre            : {len(activated)} / {len(trail_trades)} "
                  f"({len(activated)/len(trail_trades):.0%})")
            print(f"     WR                : {act_wr:.0%}")
            print(f"     PnL total         : ${act_pnl:+.2f}")

        # Distribution des steps (au-delà du TP)
        tp_trail = [t for t in trail_trades if t.exit_reason == "TP_TRAIL"]
        if tp_trail:
            avg_pnl = sum(t.pnl_pct for t in tp_trail) / len(tp_trail)
            max_gain = max(t.pnl_pct for t in tp_trail)
            avg_steps = sum(t.steps_reached for t in tp_trail) / len(tp_trail)
            print(f"\n  🎯 Trades sortis par trailing (au-delà du TP) :")
            print(f"     Nombre            : {len(tp_trail)}")
            print(f"     PnL moyen         : {avg_pnl:+.2%}")
            print(f"     Meilleur          : {max_gain:+.2%}")
            print(f"     Steps moyen       : {avg_steps:.1f}")

        # Comparaison : classique ferme au TP pour un gain ~fixe
        # Cette variante : les mêmes trades continuent au-delà
        if activated:
            classic_equiv = sum(
                t.size * (t.tp_level * (1 - 0.00075)
                          - t.entry_price * (1 + 0.00075))
                for t in activated
            )
            actual_pnl = sum(t.pnl_usd for t in activated)
            print(f"\n  📐 Comparaison pour les {len(activated)} trades qui ont atteint le TP :")
            print(f"     Classique (fermer au TP) : ${classic_equiv:+.2f}")
            print(f"     Trailing (laisser courir): ${actual_pnl:+.2f}")
            diff = actual_pnl - classic_equiv
            emoji = "✅" if diff > 0 else "❌"
            print(f"     Différence               : ${diff:+.2f} {emoji}")

        sorted_trades = sorted(trail_trades, key=lambda t: t.pnl_usd, reverse=True)
        print(f"\n  🏆 Top 5 trades :")
        for t in sorted_trades[:5]:
            print(f"     {t.symbol:10s} | {t.pnl_usd:+8.2f} ({t.pnl_pct:+.1%}) | "
                  f"{t.bars_held}b | {t.steps_reached} steps | {t.exit_reason}")
        print(f"\n  💩 Bottom 5 trades :")
        for t in sorted_trades[-5:]:
            print(f"     {t.symbol:10s} | {t.pnl_usd:+8.2f} ({t.pnl_pct:+.1%}) | "
                  f"{t.bars_held}b | {t.steps_reached} steps | {t.exit_reason}")

    print(f"\n{sep}\n")


# ════════════════════════════════════════════════════════════════════════════════
PAIRS_20 = [
    "BTC-USD", "ETH-USD", "SOL-USD", "XRP-USD", "LINK-USD",
    "ADA-USD", "DOT-USD", "AVAX-USD", "DOGE-USD", "ATOM-USD",
    "UNI-USD", "NEAR-USD", "ALGO-USD", "LTC-USD", "ETC-USD",
    "FIL-USD", "AAVE-USD", "INJ-USD", "SAND-USD", "MANA-USD",
]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="RANGE classique vs RANGE Trailing à partir du TP"
    )
    parser.add_argument("--balance", type=float, default=1000.0)
    parser.add_argument("--start", default="2022-02-20")
    parser.add_argument("--end", default="2026-02-20")
    parser.add_argument("--pairs", type=int, default=20)
    parser.add_argument("--step-pct", type=float, default=0.01)
    parser.add_argument("--no-show", action="store_true")
    args = parser.parse_args()

    start = datetime.strptime(args.start, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    end = datetime.strptime(args.end, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    pairs = PAIRS_20[:args.pairs]

    print(f"\n{'═'*80}")
    print(f"  🔬 RANGE CLASSIQUE vs RANGE Trailing après TP — A/B Test")
    print(f"  📅 {start:%Y-%m-%d} → {end:%Y-%m-%d} | {len(pairs)} paires | ${args.balance:,.0f}")
    print(f"  SL classique | Au lieu de fermer au TP → trailing activé")
    print(f"  → SL monte au breakeven quand prix atteint range_mid")
    print(f"  → Puis step trailing {args.step_pct:.0%} pour laisser courir")
    print(f"{'═'*80}\n")

    logger.info("📥 Téléchargement des données…")
    candles = download_all_pairs(pairs, start, end, interval="4h")

    # A) Classique
    print("\n" + "─" * 60)
    print("  🅰️  RANGE CLASSIQUE (TP = range_mid)")
    print("─" * 60)

    from src import config
    cfg_classic = BacktestConfig(
        initial_balance=args.balance, risk_percent_range=0.02,
        entry_buffer_pct=getattr(config, "ENTRY_BUFFER_PERCENT", 0.002),
        sl_buffer_pct=getattr(config, "SL_BUFFER_PERCENT", 0.003),
        zero_risk_trigger_pct=getattr(config, "ZERO_RISK_TRIGGER_PERCENT", 0.02),
        zero_risk_lock_pct=getattr(config, "ZERO_RISK_LOCK_PERCENT", 0.005),
        trailing_stop_pct=getattr(config, "TRAILING_STOP_PERCENT", 0.02),
        max_position_pct=0.30, max_simultaneous_positions=3,
        swing_lookback=getattr(config, "SWING_LOOKBACK", 3),
        range_width_min=getattr(config, "RANGE_WIDTH_MIN", 0.02),
        range_entry_buffer_pct=getattr(config, "RANGE_ENTRY_BUFFER_PERCENT", 0.002),
        range_sl_buffer_pct=getattr(config, "RANGE_SL_BUFFER_PERCENT", 0.003),
        range_cooldown_bars=getattr(config, "RANGE_COOLDOWN_BARS", 3),
        enable_trend=False, enable_range=True,
    )
    engine_classic = BacktestEngine(candles, cfg_classic)
    result_classic = engine_classic.run()
    m_classic = compute_metrics(result_classic)

    # B) Trailing après TP
    print("\n" + "─" * 60)
    print(f"  🅱️  RANGE Trailing après TP (step {args.step_pct:.0%})")
    print("─" * 60)

    cfg_trail = TpTrailConfig(
        initial_balance=args.balance, step_pct=args.step_pct,
        range_entry_buffer_pct=getattr(config, "RANGE_ENTRY_BUFFER_PERCENT", 0.002),
        range_sl_buffer_pct=getattr(config, "RANGE_SL_BUFFER_PERCENT", 0.003),
        range_cooldown_bars=getattr(config, "RANGE_COOLDOWN_BARS", 3),
        swing_lookback=getattr(config, "SWING_LOOKBACK", 3),
        range_width_min=getattr(config, "RANGE_WIDTH_MIN", 0.02),
    )
    engine_trail = RangeTpTrailEngine(candles, cfg_trail)
    result_trail = engine_trail.run()
    m_trail = compute_metrics(result_trail)

    # Comparaison
    print_comparison(m_classic, m_trail, result_classic, result_trail,
                     engine_trail.closed_trades)

    from backtest.report import generate_report
    print("  📊 Graphique CLASSIQUE :")
    generate_report(result_classic, m_classic, show=False)
    print("  📊 Graphique Trail@TP :")
    generate_report(result_trail, m_trail, show=not args.no_show)


if __name__ == "__main__":
    main()
