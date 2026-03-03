"""
Moteur de backtest pour la stratégie VWAP Mean Reversion Intraday.

Logique :
  - Timeframe H1 (ou 30m).
  - Calcule le VWAP journalier glissant + écart-type (bandes).
  - Signal LONG si :  prix ≤ VWAP − K×σ  ET  RSI(14) entre rsi_floor et rsi_ceil
  - TP = retour au VWAP  (ou entry × (1 + tp_pct))
  - SL = entry × (1 − sl_pct)   ou  1×ATR sous l'entrée
  - Timeout : N barres max (fin de "session" quotidienne)

Anticorrélé au CrashBot (dump 48h) et au Trail Range (expansion H4)
car exploite des excès **intraday** courts.
"""

from __future__ import annotations

import logging
import math
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

from src.core.models import Candle

logger = logging.getLogger(__name__)


# ── Configuration ──────────────────────────────────────────────────────────────


@dataclass
class VwapConfig:
    """Paramètres de la stratégie VWAP Mean Reversion."""

    initial_balance: float = 1000.0
    risk_percent: float = 0.02          # 2% risque par trade

    # Signal
    vwap_band_mult: float = 1.5         # Entrée à VWAP − K×σ
    rsi_period: int = 14                # Période RSI
    rsi_floor: float = 25.0             # RSI min pour entrer (pas trop extrême = pas un crash)
    rsi_ceil: float = 42.0              # RSI max pour entrer

    # Trade management
    tp_mode: str = "vwap"               # "vwap" = TP au VWAP, "fixed" = tp_pct fixe
    tp_pct: float = 0.015               # TP fixe 1.5% (si tp_mode="fixed")
    sl_pct: float = 0.012               # SL 1.2%
    atr_sl_mult: float = 0.0            # Si >0, SL = entry − mult×ATR (override sl_pct)
    atr_period: int = 14                # Période ATR

    # Timeout
    timeout_bars: int = 24              # Max 24 barres (= 24h en H1, 12h en 30m)

    # Cooldown
    cooldown_bars: int = 4              # Pas de nouveau trade sur même paire pendant N barres

    # Filtres
    require_volume_decline: bool = False # Volume décroissant (stabilisation)
    vol_decline_lookback: int = 3       # Comparer volume actuel vs N barres avant

    # Contraintes
    max_positions: int = 3              # Max positions simultanées
    max_position_pct: float = 0.40      # Max 40% du capital par position

    # Frais
    fee_pct: float = 0.001              # 0.1% taker fee
    slippage_pct: float = 0.0005        # 0.05% slippage


# ── Structures ─────────────────────────────────────────────────────────────────


@dataclass
class VwapTrade:
    """Trade terminé."""

    symbol: str
    entry_price: float
    exit_price: float
    size: float
    entry_time: int             # timestamp ms
    exit_time: int
    pnl_usd: float
    pnl_pct: float
    exit_reason: str            # TP, SL, TIMEOUT, END
    vwap_at_entry: float        # VWAP au moment de l'entrée
    deviation_pct: float        # Écart prix/VWAP à l'entrée
    rsi_at_entry: float
    hold_bars: int


@dataclass
class VwapPosition:
    """Position ouverte."""

    symbol: str
    entry_price: float
    tp_price: float
    sl_price: float
    size: float
    cost: float
    entry_time: int
    entry_bar_idx: int
    vwap_at_entry: float
    deviation_pct: float
    rsi_at_entry: float


@dataclass
class VwapResult:
    """Résultat complet du backtest."""

    trades: list[VwapTrade]
    equity_curve: list[tuple[int, float]]   # (timestamp_ms, equity)
    initial_balance: float
    final_equity: float
    n_signals: int
    n_traded: int
    config: VwapConfig
    start_date: datetime
    end_date: datetime
    pairs: list[str]


# ── Indicateurs techniques ─────────────────────────────────────────────────────


def _compute_rsi(closes: list[float], period: int = 14) -> float:
    """RSI classique sur les `period` dernières clôtures."""
    if len(closes) < period + 1:
        return 50.0  # neutre
    gains = []
    losses = []
    for i in range(len(closes) - period, len(closes)):
        delta = closes[i] - closes[i - 1]
        if delta > 0:
            gains.append(delta)
            losses.append(0.0)
        else:
            gains.append(0.0)
            losses.append(abs(delta))
    avg_gain = sum(gains) / period
    avg_loss = sum(losses) / period
    if avg_loss == 0:
        return 100.0
    rs = avg_gain / avg_loss
    return 100.0 - (100.0 / (1.0 + rs))


def _compute_atr(candles: list[Candle], period: int = 14) -> float:
    """Average True Range sur les `period` dernières bougies."""
    if len(candles) < period + 1:
        return 0.0
    trs = []
    for i in range(len(candles) - period, len(candles)):
        c = candles[i]
        prev_close = candles[i - 1].close
        tr = max(c.high - c.low, abs(c.high - prev_close), abs(c.low - prev_close))
        trs.append(tr)
    return sum(trs) / len(trs)


def _compute_vwap_session(candles: list[Candle], session_len: int) -> tuple[float, float]:
    """Calcule le VWAP et l'écart-type sur les N dernières bougies (session glissante).

    VWAP = Σ(typical_price × volume) / Σ(volume)
    σ    = sqrt(Σ(volume × (typical_price − VWAP)²) / Σ(volume))
    """
    if len(candles) < session_len:
        session = candles
    else:
        session = candles[-session_len:]

    total_vol = 0.0
    total_tp_vol = 0.0

    for c in session:
        tp = (c.high + c.low + c.close) / 3.0
        total_tp_vol += tp * c.volume
        total_vol += c.volume

    if total_vol == 0:
        return candles[-1].close, 0.0

    vwap = total_tp_vol / total_vol

    # Écart-type pondéré par volume
    var_sum = 0.0
    for c in session:
        tp = (c.high + c.low + c.close) / 3.0
        var_sum += c.volume * (tp - vwap) ** 2

    std = math.sqrt(var_sum / total_vol) if total_vol > 0 else 0.0

    return vwap, std


# ── Moteur ─────────────────────────────────────────────────────────────────────


class VwapEngine:
    """Simule la stratégie VWAP mean reversion bar par bar."""

    # Nombre de barres dans une "session" VWAP (≈ 1 journée)
    # H1 → 24 barres, 30m → 48 barres
    SESSION_LEN_H1 = 24
    SESSION_LEN_30M = 48

    def __init__(
        self,
        candles_by_symbol: dict[str, list[Candle]],
        config: VwapConfig,
        interval: str = "1h",
    ) -> None:
        self.candles = candles_by_symbol
        self.cfg = config
        self.interval = interval
        self.session_len = self.SESSION_LEN_30M if "30" in interval else self.SESSION_LEN_H1

        self.cash = config.initial_balance
        self.positions: dict[str, VwapPosition] = {}
        self.trades: list[VwapTrade] = []
        self.equity_curve: list[tuple[int, float]] = []
        self.cooldowns: dict[str, int] = {}  # symbol → bar_idx cooldown_until
        self.n_signals = 0
        self.n_traded = 0

    def run(self) -> VwapResult:
        symbols = sorted(self.candles.keys())
        if not symbols:
            raise ValueError("Aucune donnée fournie")

        min_len = min(len(self.candles[s]) for s in symbols)
        warmup = max(self.session_len, self.cfg.rsi_period + 2, self.cfg.atr_period + 2)

        if min_len <= warmup:
            raise ValueError(f"Pas assez de bougies ({min_len}), besoin d'au moins {warmup}")

        logger.info(
            "🔵 VWAP Engine — %d paires, %d bougies %s, capital $%.0f, session=%d barres",
            len(symbols), min_len, self.interval, self.cfg.initial_balance, self.session_len,
        )

        for i in range(warmup, min_len):
            ts = self.candles[symbols[0]][i].timestamp

            # 1. Check exits
            for sym in list(self.positions.keys()):
                if sym in self.candles and i < len(self.candles[sym]):
                    self._check_exit(sym, i)

            # 2. Check entries
            if len(self.positions) < self.cfg.max_positions:
                for sym in symbols:
                    if sym in self.positions:
                        continue
                    if len(self.positions) >= self.cfg.max_positions:
                        break
                    if i >= len(self.candles[sym]):
                        continue
                    # Cooldown
                    if sym in self.cooldowns and i < self.cooldowns[sym]:
                        continue

                    self._check_entry(sym, i)

            # 3. Equity snapshot (toutes les heures)
            if i % 6 == 0:  # Snapshot every 6 bars
                eq = self._compute_equity(i, symbols)
                self.equity_curve.append((ts, eq))

        # Close remaining positions
        for sym in list(self.positions.keys()):
            last_idx = len(self.candles[sym]) - 1
            self._close_position(sym, self.candles[sym][last_idx].close, last_idx, "END")

        # Final equity
        final_eq = self.cash
        first_ts = self.candles[symbols[0]][warmup].timestamp
        last_ts = self.candles[symbols[0]][min_len - 1].timestamp

        if self.equity_curve:
            self.equity_curve.append((last_ts, final_eq))

        return VwapResult(
            trades=self.trades,
            equity_curve=self.equity_curve,
            initial_balance=self.cfg.initial_balance,
            final_equity=final_eq,
            n_signals=self.n_signals,
            n_traded=self.n_traded,
            config=self.cfg,
            start_date=datetime.fromtimestamp(first_ts / 1000, tz=timezone.utc),
            end_date=datetime.fromtimestamp(last_ts / 1000, tz=timezone.utc),
            pairs=list(self.candles.keys()),
        )

    # ── Entry logic ────────────────────────────────────────────────────────

    def _check_entry(self, symbol: str, idx: int) -> None:
        bars = self.candles[symbol]
        candle = bars[idx]
        price = candle.close

        # VWAP + bandes
        session_bars = bars[max(0, idx - self.session_len + 1): idx + 1]
        vwap, std = _compute_vwap_session(session_bars, self.session_len)

        if std == 0:
            return

        # Bande basse
        lower_band = vwap - self.cfg.vwap_band_mult * std

        if price > lower_band:
            return  # Prix pas assez bas

        # RSI
        closes = [b.close for b in bars[max(0, idx - self.cfg.rsi_period - 1): idx + 1]]
        rsi = _compute_rsi(closes, self.cfg.rsi_period)

        if rsi < self.cfg.rsi_floor or rsi > self.cfg.rsi_ceil:
            return  # RSI hors zone

        # Filtre volume décroissant (optionnel)
        if self.cfg.require_volume_decline and idx >= self.cfg.vol_decline_lookback:
            recent_vol = candle.volume
            past_vol = sum(
                bars[idx - j].volume for j in range(1, self.cfg.vol_decline_lookback + 1)
            ) / self.cfg.vol_decline_lookback
            if recent_vol > past_vol * 1.2:  # Volume encore en hausse → skip
                return

        self.n_signals += 1

        # Sizing
        deviation_pct = (price - vwap) / vwap

        # SL
        if self.cfg.atr_sl_mult > 0:
            atr = _compute_atr(
                bars[max(0, idx - self.cfg.atr_period): idx + 1],
                self.cfg.atr_period,
            )
            sl_price = price - self.cfg.atr_sl_mult * atr if atr > 0 else price * (1 - self.cfg.sl_pct)
        else:
            sl_price = price * (1 - self.cfg.sl_pct)

        sl_distance = abs(price - sl_price)
        if sl_distance <= 0:
            return

        # TP
        if self.cfg.tp_mode == "vwap":
            tp_price = vwap  # Retour au VWAP
        else:
            tp_price = price * (1 + self.cfg.tp_pct)

        # Position sizing
        risk_amount = self.cash * self.cfg.risk_percent
        size = risk_amount / sl_distance
        cost = size * price

        # Contraintes
        max_cost = self.cash * self.cfg.max_position_pct
        if cost > max_cost:
            size = max_cost / price
            cost = size * price

        if cost > self.cash * 0.95:  # Garder 5% de marge
            return

        # Slippage
        entry_price = price * (1 + self.cfg.slippage_pct)
        fee = cost * self.cfg.fee_pct

        self.cash -= (cost + fee)
        self.n_traded += 1

        self.positions[symbol] = VwapPosition(
            symbol=symbol,
            entry_price=entry_price,
            tp_price=tp_price,
            sl_price=sl_price,
            size=size,
            cost=cost,
            entry_time=candle.timestamp,
            entry_bar_idx=idx,
            vwap_at_entry=vwap,
            deviation_pct=deviation_pct,
            rsi_at_entry=rsi,
        )

        logger.debug(
            "🔵 ENTRY %s @ %.6f | VWAP=%.6f (%.1f%%) | RSI=%.1f | TP=%.6f SL=%.6f",
            symbol, entry_price, vwap, deviation_pct * 100, rsi, tp_price, sl_price,
        )

    # ── Exit logic ─────────────────────────────────────────────────────────

    def _check_exit(self, symbol: str, idx: int) -> None:
        pos = self.positions[symbol]
        candle = self.candles[symbol][idx]

        # TP dynamique : recalculer le VWAP si mode vwap
        if self.cfg.tp_mode == "vwap":
            session_bars = self.candles[symbol][max(0, idx - self.session_len + 1): idx + 1]
            vwap, _ = _compute_vwap_session(session_bars, self.session_len)
            pos.tp_price = vwap

        # SL hit (sur le low intrabar)
        if candle.low <= pos.sl_price:
            self._close_position(symbol, pos.sl_price, idx, "SL")
            return

        # TP hit (sur le high intrabar)
        if candle.high >= pos.tp_price:
            self._close_position(symbol, pos.tp_price, idx, "TP")
            return

        # Timeout
        bars_held = idx - pos.entry_bar_idx
        if bars_held >= self.cfg.timeout_bars:
            self._close_position(symbol, candle.close, idx, "TIMEOUT")
            return

    # ── Close position ─────────────────────────────────────────────────────

    def _close_position(self, symbol: str, exit_price: float, idx: int, reason: str) -> None:
        pos = self.positions.pop(symbol)

        # Slippage + fee
        actual_exit = exit_price * (1 - self.cfg.slippage_pct) if reason != "TP" else exit_price
        fee = pos.size * actual_exit * self.cfg.fee_pct

        proceeds = pos.size * actual_exit - fee
        self.cash += proceeds

        pnl_usd = proceeds - pos.cost
        pnl_pct = pnl_usd / pos.cost if pos.cost > 0 else 0

        bars_held = idx - pos.entry_bar_idx

        self.trades.append(VwapTrade(
            symbol=symbol,
            entry_price=pos.entry_price,
            exit_price=actual_exit,
            size=pos.size,
            entry_time=pos.entry_time,
            exit_time=self.candles[symbol][idx].timestamp,
            pnl_usd=pnl_usd,
            pnl_pct=pnl_pct,
            exit_reason=reason,
            vwap_at_entry=pos.vwap_at_entry,
            deviation_pct=pos.deviation_pct,
            rsi_at_entry=pos.rsi_at_entry,
            hold_bars=bars_held,
        ))

        # Cooldown
        self.cooldowns[symbol] = idx + self.cfg.cooldown_bars

        emoji = "✅" if pnl_usd >= 0 else "❌"
        logger.debug(
            "%s CLOSE %s @ %.6f (%s) | PnL $%.2f (%.2f%%) | %d bars | dev=%.1f%%",
            emoji, symbol, actual_exit, reason, pnl_usd, pnl_pct * 100,
            bars_held, pos.deviation_pct * 100,
        )

    # ── Equity ─────────────────────────────────────────────────────────────

    def _compute_equity(self, idx: int, symbols: list[str]) -> float:
        eq = self.cash
        for sym, pos in self.positions.items():
            if idx < len(self.candles[sym]):
                eq += pos.size * self.candles[sym][idx].close
        return eq


# ── Métriques ──────────────────────────────────────────────────────────────────


def compute_vwap_metrics(result: VwapResult) -> dict:
    """Calcule les KPIs du backtest VWAP."""
    trades = result.trades
    eq = result.equity_curve
    init = result.initial_balance
    final = result.final_equity
    days = max((result.end_date - result.start_date).days, 1)
    years = days / 365.25

    total_return = (final - init) / init if init > 0 else 0
    cagr = (final / init) ** (1 / years) - 1 if final > 0 and years > 0 else 0

    # Drawdown
    peak = init
    max_dd = 0.0
    dd_curve: list[float] = []
    for _, equity in eq:
        peak = max(peak, equity)
        dd = (equity - peak) / peak if peak > 0 else 0
        dd_curve.append(dd)
        max_dd = min(max_dd, dd)

    # Sharpe
    returns: list[float] = []
    for i in range(1, len(eq)):
        prev_eq = eq[i - 1][1]
        if prev_eq > 0:
            returns.append((eq[i][1] - prev_eq) / prev_eq)

    # Periods per year dépend de l'intervalle de snapshot
    sharpe = _sharpe(returns)
    sortino = _sortino(returns)

    # Trades
    n = len(trades)
    if n > 0:
        wins = [t for t in trades if t.pnl_usd > 0]
        losses = [t for t in trades if t.pnl_usd <= 0]
        win_rate = len(wins) / n
        gross_profit = sum(t.pnl_usd for t in wins) or 0
        gross_loss = abs(sum(t.pnl_usd for t in losses)) or 1e-9
        profit_factor = gross_profit / gross_loss
        avg_pnl_usd = sum(t.pnl_usd for t in trades) / n
        avg_pnl_pct = sum(t.pnl_pct for t in trades) / n
        avg_hold = sum(t.hold_bars for t in trades) / n
        best = max(trades, key=lambda t: t.pnl_usd)
        worst = min(trades, key=lambda t: t.pnl_usd)
    else:
        win_rate = profit_factor = avg_pnl_usd = avg_pnl_pct = avg_hold = 0
        best = worst = None

    by_pair = _group_trades(trades, lambda t: t.symbol)
    by_exit = _group_trades(trades, lambda t: t.exit_reason)

    # Par tranche de déviation VWAP
    def dev_bucket(t: VwapTrade) -> str:
        pct = abs(t.deviation_pct) * 100
        if pct < 1.5:
            return "<1.5%"
        elif pct < 2.5:
            return "1.5-2.5%"
        elif pct < 4:
            return "2.5-4%"
        else:
            return ">4%"

    by_deviation = _group_trades(trades, dev_bucket)

    return {
        "total_return": total_return,
        "cagr": cagr,
        "max_drawdown": max_dd,
        "sharpe": sharpe,
        "sortino": sortino,
        "n_trades": n,
        "n_signals": result.n_signals,
        "win_rate": win_rate,
        "profit_factor": profit_factor,
        "avg_pnl_usd": avg_pnl_usd,
        "avg_pnl_pct": avg_pnl_pct,
        "avg_hold_bars": avg_hold,
        "best_trade": best,
        "worst_trade": worst,
        "by_pair": by_pair,
        "by_exit": by_exit,
        "by_deviation": by_deviation,
        "dd_curve": dd_curve,
        "years": years,
        "days": days,
        "final_equity": final,
    }


# ── Helpers ────────────────────────────────────────────────────────────────────


def _sharpe(returns: list[float], periods_per_year: float = 1460) -> float:
    """Sharpe annualisé. 1460 = ~6 snapshots/jour × 365."""
    if len(returns) < 2:
        return 0.0
    mu = sum(returns) / len(returns)
    var = sum((r - mu) ** 2 for r in returns) / (len(returns) - 1)
    std = math.sqrt(var) if var > 0 else 1e-9
    return (mu / std) * math.sqrt(periods_per_year)


def _sortino(returns: list[float], periods_per_year: float = 1460) -> float:
    if len(returns) < 2:
        return 0.0
    mu = sum(returns) / len(returns)
    neg = [r for r in returns if r < 0]
    if not neg:
        return 99.0
    down_var = sum(r ** 2 for r in neg) / len(neg)
    down_std = math.sqrt(down_var) if down_var > 0 else 1e-9
    return (mu / down_std) * math.sqrt(periods_per_year)


def _group_trades(trades: list[VwapTrade], key_fn) -> dict[str, dict]:
    groups: dict[str, list[VwapTrade]] = defaultdict(list)
    for t in trades:
        groups[key_fn(t)].append(t)
    out: dict[str, dict] = {}
    for k, tlist in sorted(groups.items()):
        n = len(tlist)
        wins = sum(1 for t in tlist if t.pnl_usd > 0)
        pnl = sum(t.pnl_usd for t in tlist)
        gp = sum(t.pnl_usd for t in tlist if t.pnl_usd > 0) or 0
        gl = abs(sum(t.pnl_usd for t in tlist if t.pnl_usd <= 0)) or 1e-9
        avg_hold = sum(t.hold_bars for t in tlist) / n if n else 0
        out[k] = {
            "n": n,
            "wins": wins,
            "wr": wins / n if n else 0,
            "pnl": pnl,
            "pf": gp / gl,
            "avg_pct": sum(t.pnl_pct for t in tlist) / n if n else 0,
            "avg_hold": avg_hold,
        }
    return out
