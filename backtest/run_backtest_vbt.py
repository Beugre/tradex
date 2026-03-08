#!/usr/bin/env python3
"""
Backtest : VBT (Volatility Breakout Trend)

Acheter les cassures de compression de volatilité dans une tendance haussière.

Règles :
  - Trend filter Daily : Close > EMA200, EMA50 > EMA200
  - Compression H4 : ATR14 < ATR50 × 0.7
  - Entry H4 : Close > Highest High(20) ET Volume > Volume MA(20)
  - SL : entry - 1.8× ATR
  - Exit : Trailing stop 2× ATR (pas de TP fixe)
  - Risk : 1% equity, max 4 positions

Usage :
    python -m backtest.run_backtest_vbt
    python -m backtest.run_backtest_vbt --years 6 --balance 500
    python -m backtest.run_backtest_vbt --years 6 --balance 500 --variants
"""

from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from backtest.data_loader import download_candles
from src.core.models import Candle
from src.core.momentum_engine import ema, sma, atr_series, rolling_max

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(message)s")
logger = logging.getLogger(__name__)

OUTPUT_DIR = Path(__file__).parent / "output"
OUTPUT_DIR.mkdir(exist_ok=True)

PAIRS = ["BTC-USD", "ETH-USD", "SOL-USD", "BNB-USD", "LINK-USD", "AVAX-USD"]

# Revolut X fees
MAKER_FEE = 0.0
TAKER_FEE = 0.0009

H4_PER_DAY = 6


# ══════════════════════════════════════════════════════════════════════════════
#  CONFIG
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class VBTConfig:
    """Configuration pour Volatility Breakout Trend."""
    # ── Trend filter (Daily) ──
    ema_trend_slow: int = 200
    ema_trend_fast: int = 50

    # ── Compression (H4) ──
    atr_fast_period: int = 14
    atr_slow_period: int = 50
    compression_ratio: float = 0.70    # ATR14 < ATR50 × ratio

    # ── Breakout entry (H4) ──
    breakout_lookback: int = 20        # Highest high des N dernières bougies
    vol_ma_period: int = 20            # Volume > MA(N)
    vol_mult: float = 1.0             # Volume ≥ mult × MA

    # ── SL ──
    sl_atr_mult: float = 1.8

    # ── Trailing stop (seule sortie) ──
    trail_atr_mult: float = 2.0

    # ── Risk ──
    risk_per_trade: float = 0.01       # 1% equity
    max_positions: int = 4
    max_exposure_pct: float = 0.50     # Max 50% equity en positions

    # ── Cooldown ──
    cooldown_bars: int = 6             # 24h cooldown par paire


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
    """Position ouverte avec trailing stop."""
    symbol: str
    entry_price: float
    sl_price: float
    size: float
    size_usd: float
    entry_bar: int
    entry_ts: int
    highest_since_entry: float = 0.0
    trail_stop: float = 0.0


@dataclass
class _PairState:
    cooldown_until: int = 0


# ══════════════════════════════════════════════════════════════════════════════
#  RESAMPLE H4 → DAILY
# ══════════════════════════════════════════════════════════════════════════════

def resample_h4_to_daily(candles: list[Candle]) -> list[Candle]:
    daily: list[Candle] = []
    for i in range(0, len(candles) - H4_PER_DAY + 1, H4_PER_DAY):
        group = candles[i:i + H4_PER_DAY]
        daily.append(Candle(
            timestamp=group[0].timestamp,
            open=group[0].open,
            high=max(c.high for c in group),
            low=min(c.low for c in group),
            close=group[-1].close,
            volume=sum(c.volume for c in group),
        ))
    return daily


# ══════════════════════════════════════════════════════════════════════════════
#  INDICATORS
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class VBTIndicators:
    # H4
    atr_fast: list[float] = field(default_factory=list)    # ATR14
    atr_slow: list[float] = field(default_factory=list)    # ATR50
    highest_high: list[float] = field(default_factory=list) # Rolling max high
    vol_ma: list[float] = field(default_factory=list)       # Volume MA

    # Daily (mapped to H4 via // H4_PER_DAY)
    d_ema200: list[float] = field(default_factory=list)
    d_ema50: list[float] = field(default_factory=list)
    d_close: list[float] = field(default_factory=list)


def compute_vbt_indicators(candles: list[Candle], cfg: VBTConfig) -> VBTIndicators:
    ind = VBTIndicators()

    closes = [c.close for c in candles]
    highs = [c.high for c in candles]
    volumes = [c.volume for c in candles]

    ind.atr_fast = atr_series(candles, cfg.atr_fast_period)
    ind.atr_slow = atr_series(candles, cfg.atr_slow_period)
    ind.highest_high = rolling_max(highs, cfg.breakout_lookback)
    ind.vol_ma = sma(volumes, cfg.vol_ma_period)

    # Daily
    daily = resample_h4_to_daily(candles)
    d_closes = [c.close for c in daily]
    ind.d_ema200 = ema(d_closes, cfg.ema_trend_slow)
    ind.d_ema50 = ema(d_closes, cfg.ema_trend_fast)
    ind.d_close = d_closes

    return ind


# ══════════════════════════════════════════════════════════════════════════════
#  SIMULATOR
# ══════════════════════════════════════════════════════════════════════════════

def simulate_vbt(
    all_candles: dict[str, list[Candle]],
    cfg: VBTConfig,
    initial_balance: float = 500.0,
) -> BacktestResult:
    balance = initial_balance
    positions: list[_OpenPos] = []
    closed_trades: list[Trade] = []
    equity_curve: list[float] = [balance]

    # Pré-calcul indicateurs
    all_ind: dict[str, VBTIndicators] = {}
    for symbol, candles in all_candles.items():
        all_ind[symbol] = compute_vbt_indicators(candles, cfg)

    states: dict[str, _PairState] = {sym: _PairState() for sym in all_candles}
    min_len = min(len(c) for c in all_candles.values())

    # Warmup : EMA200 daily = 200j × 6 = 1200 bars + ATR50
    start_bar = max(1200, cfg.atr_slow_period + 10)

    for bar_idx in range(start_bar, min_len):

        # ── Gestion positions ouvertes ──
        for pos in positions[:]:
            c = all_candles[pos.symbol][bar_idx]
            ind = all_ind[pos.symbol]
            hold = bar_idx - pos.entry_bar

            # Update trailing stop
            if c.high > pos.highest_since_entry:
                pos.highest_since_entry = c.high
                atr_now = ind.atr_fast[bar_idx] if bar_idx < len(ind.atr_fast) else 0
                new_trail = pos.highest_since_entry - cfg.trail_atr_mult * atr_now
                if new_trail > pos.trail_stop:
                    pos.trail_stop = new_trail

            # Check SL / trailing stop (prendre le plus haut des deux)
            effective_stop = max(pos.sl_price, pos.trail_stop)
            if c.low <= effective_stop:
                exit_price = effective_stop
                pnl_pct = (exit_price - pos.entry_price) / pos.entry_price
                pnl_usd = pos.size_usd * pnl_pct - pos.size_usd * TAKER_FEE
                balance += pos.size_usd + pnl_usd

                reason = "TRAIL" if pos.trail_stop >= pos.sl_price else "SL"
                closed_trades.append(Trade(
                    symbol=pos.symbol, side="LONG",
                    entry_price=pos.entry_price, exit_price=exit_price,
                    size=pos.size, size_usd=pos.size_usd,
                    entry_time=pos.entry_ts, exit_time=c.timestamp,
                    pnl_usd=pnl_usd, pnl_pct=pnl_pct * 100,
                    exit_reason=reason, hold_bars=hold,
                ))
                positions.remove(pos)
                continue

        # ── Détection de nouvelles entrées ──
        if len(positions) < cfg.max_positions and balance > 10:
            total_exposure = sum(p.size_usd for p in positions)
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

                # ── 1. Trend filter Daily ──
                d_idx = bar_idx // H4_PER_DAY
                if d_idx >= len(ind.d_ema200) or d_idx >= len(ind.d_ema50) or d_idx >= len(ind.d_close):
                    continue

                d_close = ind.d_close[d_idx]
                d_ema200 = ind.d_ema200[d_idx]
                d_ema50 = ind.d_ema50[d_idx]

                if d_close <= d_ema200:
                    continue
                if d_ema50 <= d_ema200:
                    continue

                # ── 2. Compression check ──
                atr_fast = ind.atr_fast[bar_idx] if bar_idx < len(ind.atr_fast) else 0
                atr_slow = ind.atr_slow[bar_idx] if bar_idx < len(ind.atr_slow) else 0
                if atr_slow <= 0:
                    continue
                if atr_fast >= cfg.compression_ratio * atr_slow:
                    continue  # Pas comprimé → skip

                # ── 3. Breakout signal ──
                prev_highest = ind.highest_high[bar_idx - 1] if bar_idx - 1 < len(ind.highest_high) else 0
                if prev_highest <= 0:
                    continue
                if c.close <= prev_highest:
                    continue  # Pas de cassure

                # Volume confirmation
                vol_ma = ind.vol_ma[bar_idx] if bar_idx < len(ind.vol_ma) else 0
                if vol_ma <= 0:
                    continue
                if c.volume < cfg.vol_mult * vol_ma:
                    continue

                # ── 4. Sizing ──
                if atr_fast <= 0:
                    continue
                sl_distance = cfg.sl_atr_mult * atr_fast
                sl_price = c.close - sl_distance
                entry_price = c.close

                equity = balance + sum(p.size_usd for p in positions)
                risk_amount = equity * cfg.risk_per_trade
                size = risk_amount / sl_distance
                size_usd = size * entry_price

                # Cap exposition
                remaining_exposure = max_exposure - total_exposure
                if size_usd > remaining_exposure:
                    size_usd = remaining_exposure
                    size = size_usd / entry_price if entry_price > 0 else 0

                if size_usd < 5:
                    continue

                # Execute
                fee = size_usd * MAKER_FEE
                balance -= size_usd + fee
                total_exposure += size_usd

                positions.append(_OpenPos(
                    symbol=symbol, entry_price=entry_price,
                    sl_price=sl_price,
                    size=size, size_usd=size_usd,
                    entry_bar=bar_idx, entry_ts=c.timestamp,
                    highest_since_entry=c.high,
                    trail_stop=c.high - cfg.trail_atr_mult * atr_fast,
                ))
                state.cooldown_until = bar_idx + cfg.cooldown_bars

        # ── Equity tracking ──
        pos_value = sum(
            p.size * all_candles[p.symbol][min(bar_idx, len(all_candles[p.symbol]) - 1)].close
            for p in positions
        )
        equity_curve.append(balance + pos_value)

    # Clôturer positions restantes
    for pos in positions:
        last = all_candles[pos.symbol][min_len - 1]
        pnl_pct = (last.close - pos.entry_price) / pos.entry_price
        pnl_usd = pos.size_usd * pnl_pct - pos.size_usd * TAKER_FEE
        balance += pos.size_usd + pnl_usd
        closed_trades.append(Trade(
            symbol=pos.symbol, side="LONG",
            entry_price=pos.entry_price, exit_price=last.close,
            size=pos.size, size_usd=pos.size_usd,
            entry_time=pos.entry_ts, exit_time=last.timestamp,
            pnl_usd=pnl_usd, pnl_pct=pnl_pct * 100,
            exit_reason="END", hold_bars=min_len - 1 - pos.entry_bar,
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

def _define_variants(run_variants: bool = False) -> list[tuple[str, VBTConfig, dict, list[str] | None]]:
    configs = []

    # ── BASE : Config proposée ──
    configs.append(("VBT_BASE", VBTConfig(), {
        "entry": "BreakoutHH20 + Compression + VolConf",
        "sl": "1.8×ATR",
        "exit": "Trail 2×ATR",
        "risk": "1%", "max_pos": "4",
    }, None))

    # ── 3 paires (BTC, ETH, SOL) ──
    configs.append(("VBT_3PAIRS", VBTConfig(), {
        "entry": "BreakoutHH20 + Compression + VolConf",
        "sl": "1.8×ATR",
        "exit": "Trail 2×ATR",
        "pairs": "BTC,ETH,SOL",
    }, ["BTC-USD", "ETH-USD", "SOL-USD"]))

    if run_variants:
        # ── V1 : SL serré 1.2 ATR ──
        configs.append(("VBT_TIGHT_SL", VBTConfig(
            sl_atr_mult=1.2,
        ), {
            "sl": "1.2×ATR", "exit": "Trail 2×ATR",
        }, None))

        # ── V2 : Trail serré 1.5 ATR ──
        configs.append(("VBT_TIGHT_TRAIL", VBTConfig(
            trail_atr_mult=1.5,
        ), {
            "sl": "1.8×ATR", "exit": "Trail 1.5×ATR",
        }, None))

        # ── V3 : Trail large 3 ATR ──
        configs.append(("VBT_WIDE_TRAIL", VBTConfig(
            trail_atr_mult=3.0,
        ), {
            "sl": "1.8×ATR", "exit": "Trail 3×ATR",
        }, None))

        # ── V4 : Compression très stricte (0.5) ──
        configs.append(("VBT_SQUEEZE", VBTConfig(
            compression_ratio=0.50,
        ), {
            "compression": "ATR14 < 0.5×ATR50",
        }, None))

        # ── V5 : Lookback 30 (range plus long) ──
        configs.append(("VBT_HH30", VBTConfig(
            breakout_lookback=30,
        ), {
            "breakout": "HH30 (range plus long)",
        }, None))

        # ── V6 : Risk 2% ──
        configs.append(("VBT_RISK2", VBTConfig(
            risk_per_trade=0.02,
            max_exposure_pct=0.60,
        ), {
            "risk": "2%", "max_exp": "60%",
        }, None))

        # ── V7 : Pas de filtre volume ──
        configs.append(("VBT_NO_VOL", VBTConfig(
            vol_mult=0.0,
        ), {
            "vol": "désactivé",
        }, None))

        # ── V8 : Compression 0.8 (plus permissif) ──
        configs.append(("VBT_LOOSE_COMP", VBTConfig(
            compression_ratio=0.80,
        ), {
            "compression": "ATR14 < 0.8×ATR50",
        }, None))

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
            "rr": 0, "avg_hold": 0,
        }

    wins = [t for t in trades if t.pnl_usd > 0]
    losses = [t for t in trades if t.pnl_usd <= 0]
    total_gains = sum(t.pnl_usd for t in wins) if wins else 0
    total_losses = abs(sum(t.pnl_usd for t in losses)) if losses else 0.001

    # Max DD
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
    print(f"\n{'='*110}")
    print(f"  {title}")
    print(f"{'='*110}")
    hdr = f"{'Config':<20} {'Trades':>6} {'WR%':>7} {'PF':>7} {'PnL $':>9} {'Avg PnL':>8} {'Max DD%':>8} {'Final $':>9} {'R:R':>5} {'Avg Hold':>9}"
    print(hdr)
    print("-" * 110)
    for k in kpis_list:
        print(
            f"{k['label']:<20} {k['trades']:>6} {k['win_rate']:>6.1f}% {k['pf']:>7.2f} "
            f"{k['pnl']:>+9.2f}$ {k['avg_pnl']:>+7.2f}$ {k['max_dd']:>7.1f}% "
            f"{k['final']:>9.2f}$ {k['rr']:>5.2f} {k['avg_hold']:>8.1f}b"
        )
    print("-" * 110)
    best = max(kpis_list, key=lambda k: k["pf"])
    print(f"  Meilleur PF : {best['label']} (PF {best['pf']:.2f}, PnL {best['pnl']:+.2f}$, WR {best['win_rate']:.1f}%)")


def print_exit_breakdown(results: list[BacktestResult], prefix: str) -> None:
    from collections import Counter
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
            avg_pct = sum(t.pnl_pct for t in res.trades if t.exit_reason == reason) / count
            print(f"    {reason:<16}: {count:>4} ({pct:>5.1f}%)  avg PnL: {avg:>+.2f}$  avg %: {avg_pct:>+.2f}%")


def print_per_pair(results: list[BacktestResult]) -> None:
    for res in results:
        if not res.trades:
            continue
        print(f"\n{'='*70}")
        print(f"  {res.label} — Stats par paire")
        print(f"{'='*70}")
        pairs = sorted(set(t.symbol for t in res.trades))
        print(f"  {'Paire':<12} {'Trades':>6} {'WR%':>7} {'PnL $':>9} {'Avg Win%':>9} {'Avg Loss%':>10}")
        print(f"  {'-'*55}")
        for pair in pairs:
            pair_trades = [t for t in res.trades if t.symbol == pair]
            wins = [t for t in pair_trades if t.pnl_usd > 0]
            losses = [t for t in pair_trades if t.pnl_usd <= 0]
            wr = len(wins) / len(pair_trades) * 100 if pair_trades else 0
            pnl = sum(t.pnl_usd for t in pair_trades)
            avg_win_pct = sum(t.pnl_pct for t in wins) / len(wins) if wins else 0
            avg_loss_pct = sum(t.pnl_pct for t in losses) / len(losses) if losses else 0
            print(f"  {pair:<12} {len(pair_trades):>6} {wr:>6.1f}% {pnl:>+9.2f}$ {avg_win_pct:>+8.2f}% {avg_loss_pct:>+9.2f}%")


def plot_equity(results: list[BacktestResult], title: str, filename: str) -> None:
    fig, ax = plt.subplots(figsize=(14, 6))
    for res in results:
        ax.plot(res.equity_curve, label=res.label, linewidth=1)
    ax.axhline(y=results[0].initial_balance, color="grey", linestyle="--", alpha=0.5)
    ax.set_title(title)
    ax.set_xlabel("Bar (H4)")
    ax.set_ylabel("Equity ($)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    path = OUTPUT_DIR / filename
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\n  Chart : {path}")


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    parser = argparse.ArgumentParser(description="Backtest Volatility Breakout Trend")
    parser.add_argument("--years", type=int, default=6, help="Années de données H4")
    parser.add_argument("--balance", type=float, default=500, help="Capital initial ($)")
    parser.add_argument("--variants", action="store_true", help="Tester aussi les variantes")
    args = parser.parse_args()

    end = datetime.now(timezone.utc)
    start = end - timedelta(days=args.years * 365)

    print(f"\n{'═'*70}")
    print(f"  VOLATILITY BREAKOUT TREND — Téléchargement H4 ({args.years} ans)")
    print(f"  Paires : {', '.join(PAIRS)}")
    print(f"{'═'*70}")

    all_candles: dict[str, list[Candle]] = {}
    for pair in PAIRS:
        logger.info("  %s H4…", pair)
        candles = download_candles(pair, start, end, interval="4h")
        all_candles[pair] = candles
        logger.info("    %s : %d bougies H4 (%d jours)", pair, len(candles), len(candles) // 6)

    configs = _define_variants(run_variants=args.variants)
    results: list[BacktestResult] = []
    kpis_list: list[dict] = []

    for label, cfg, desc, pairs_override in configs:
        logger.info("  %s…", label)
        candles_for_run = {k: v for k, v in all_candles.items() if k in pairs_override} if pairs_override else all_candles
        if not candles_for_run:
            logger.warning("  Aucune paire disponible pour %s, skip", label)
            continue
        result = simulate_vbt(candles_for_run, cfg, initial_balance=args.balance)
        result.label = label
        result.config_desc = desc
        results.append(result)
        kpis = compute_kpis(result)
        kpis_list.append(kpis)

    n_bars = min(len(c) for c in all_candles.values())
    print_table(kpis_list, f"VOLATILITY BREAKOUT TREND — H4 ({n_bars} bars, {args.years} ans, ${args.balance:.0f})")
    print_exit_breakdown(results, "VBT")
    print_per_pair(results)
    plot_equity(results, "Volatility Breakout Trend", "vbt_equity.png")

    # Résumé
    best = max(kpis_list, key=lambda k: k["pf"])
    print(f"\n{'═'*70}")
    if best["pf"] >= 1.5:
        print(f"  ✅ PROMETTEUR : {best['label']} — PF {best['pf']:.2f}, WR {best['win_rate']:.1f}%, PnL {best['pnl']:+.2f}$")
        print(f"     DD {best['max_dd']:.1f}% — R:R {best['rr']:.2f}")
    elif best["pf"] >= 1.0:
        print(f"  ⚠️  MARGINAL : {best['label']} — PF {best['pf']:.2f}, PnL {best['pnl']:+.2f}$")
        print(f"     Tester --variants pour optimiser")
    else:
        print(f"  ❌ NON RENTABLE : Meilleur PF = {best['pf']:.2f}")
    print(f"{'═'*70}\n")


if __name__ == "__main__":
    main()
