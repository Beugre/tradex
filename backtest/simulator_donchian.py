"""
Moteur de backtest — Donchian Trend Following (Daily).

Logique (inspirée des Turtle Traders, adaptée crypto) :
  - Timeframe : 1D (bougies journalières)
  - Entrée LONG : close > plus haut des N derniers jours (Donchian upper)
  - Entrée SHORT : close < plus bas des N derniers jours (Donchian lower)
  - Sortie : trailing sur canal Donchian plus court (exit_period)
    - LONG fermé si close < plus bas des exit_period derniers jours
    - SHORT fermé si close > plus haut des exit_period derniers jours
  - Filtre :
    - ADX(14) > adx_threshold → confirme qu'un trend existe
    - EMA(200) filter optionnel → direction de la tendance macro
  - SL initial : sl_atr_mult × ATR(14) depuis l'entrée

Décorrélé des autres stratégies :
  - Trail Range (H4) : structure Dow (HH/HL/LH/LL), range mean reversion
  - CrashBot (1m)    : event-driven sur crashs extrêmes
  - Momentum (15m)   : BB squeeze → MACD breakout

Le Donchian est un trend-following pur sur daily, ride les grosses
tendances macro. Peu de trades, gros R:R, faible corrélation.
"""

from __future__ import annotations

import logging
import math
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional, Any

from src.core.models import Candle

logger = logging.getLogger(__name__)


# ── Configuration ──────────────────────────────────────────────────────────────


@dataclass
class DonchianConfig:
    """Paramètres de la stratégie Donchian Trend Following."""

    initial_balance: float = 1000.0
    risk_percent: float = 0.02          # 2% risque par trade

    # ── Donchian Channels ──
    entry_period: int = 20              # Breakout = plus haut/bas des N jours
    exit_period: int = 10               # Exit = canal court (N/2 classique)

    # ── Filtre ADX ──
    adx_period: int = 14
    adx_threshold: float = 20.0        # ADX > 20 = marché en tendance

    # ── Filtre EMA macro ──
    ema_period: int = 200               # EMA longue pour filtre directionnel
    use_ema_filter: bool = False        # Désactivé par défaut (Turtles ne l'utilisent pas)

    # ── Stop Loss ──
    atr_period: int = 14
    sl_atr_mult: float = 2.0           # SL = 2×ATR (large, daily)

    # ── Trailing ──
    use_donchian_exit: bool = True      # True = exit via Donchian court, False = ATR trailing
    trail_atr_mult: float = 3.0        # Si donchian_exit=False, trailing à 3×ATR

    # ── Constraints ──
    max_positions: int = 6              # Max positions simultanées
    max_position_pct: float = 0.25      # Max 25% du capital par position
    cooldown_days: int = 5              # Cooldown par paire après clôture

    # ── Fees ──
    fee_pct: float = 0.001              # 0.1% taker
    slippage_pct: float = 0.0005        # 0.05% slippage

    # ── Filtre régime BTC ──
    btc_regime_filter: bool = False     # Si True, bloque les longs quand BTC < SMA(btc_regime_period)
    btc_regime_period: int = 200        # Période SMA pour le filtre régime
    btc_symbol: str = "BTC-USD"          # Symbole BTC dans les données

    # ── Filtrage de paires ──
    excluded_pairs: list = field(default_factory=list)  # Paires exclues du trading

    # ── Mode ──
    allow_short: bool = True
    compound: bool = True               # Réinvestir les gains


# ── Structures ─────────────────────────────────────────────────────────────────


@dataclass
class DonchianTrade:
    """Trade terminé."""
    symbol: str
    side: str                   # "LONG" ou "SHORT"
    entry_price: float
    exit_price: float
    size: float
    entry_time: int             # timestamp ms
    exit_time: int
    pnl_usd: float
    pnl_pct: float
    exit_reason: str            # DONCHIAN_EXIT, SL, TRAILING, TIMEOUT, END
    hold_days: int
    atr_at_entry: float


@dataclass
class DonchianPosition:
    """Position ouverte."""
    symbol: str
    side: str
    entry_price: float
    sl_price: float
    size: float
    cost: float
    entry_time: int
    entry_bar_idx: int
    atr_at_entry: float
    peak_price: float           # Best price (pour trailing ATR si utilisé)


@dataclass
class DonchianResult:
    """Résultat complet du backtest."""
    trades: list[DonchianTrade]
    equity_curve: list[tuple[int, float]]   # (timestamp_ms, equity)
    initial_balance: float
    final_equity: float
    n_signals: int              # Breakouts détectés (avant filtres)
    n_filtered: int             # Filtrés par ADX/EMA
    config: DonchianConfig
    start_date: datetime
    end_date: datetime
    pairs: list[str]


# ── Indicateurs (O(n) full-series) ─────────────────────────────────────────────


def _ema_series(values: list[float], period: int) -> list[float]:
    """EMA complète O(n)."""
    if not values:
        return []
    result = [values[0]]
    k = 2.0 / (period + 1)
    for i in range(1, len(values)):
        result.append(values[i] * k + result[-1] * (1 - k))
    return result


def _atr_series(candles: list[Candle], period: int) -> list[float]:
    """ATR Wilder-smoothed O(n)."""
    n = len(candles)
    if n < 2:
        return [0.0] * n
    trs = [candles[0].high - candles[0].low]
    for i in range(1, n):
        c = candles[i]
        prev = candles[i - 1]
        tr = max(c.high - c.low, abs(c.high - prev.close), abs(c.low - prev.close))
        trs.append(tr)
    # Wilder smoothing
    result = [0.0] * n
    if n < period + 1:
        result[-1] = sum(trs) / n if n else 0
        return result
    # Première valeur = SMA des period premiers TR
    first_atr = sum(trs[1:period + 1]) / period
    result[period] = first_atr
    for i in range(period + 1, n):
        result[i] = (result[i - 1] * (period - 1) + trs[i]) / period
    return result


def _adx_series(candles: list[Candle], period: int) -> list[float]:
    """ADX Wilder-smoothed O(n). Retourne liste de même longueur."""
    n = len(candles)
    result = [0.0] * n
    if n < 2 * period + 2:
        return result

    # +DM, -DM, TR
    plus_dm = [0.0]
    minus_dm = [0.0]
    tr_list = [candles[0].high - candles[0].low]
    for i in range(1, n):
        h, l = candles[i].high, candles[i].low
        ph, pl, pc = candles[i - 1].high, candles[i - 1].low, candles[i - 1].close
        tr = max(h - l, abs(h - pc), abs(l - pc))
        tr_list.append(tr)
        up = h - ph
        down = pl - l
        plus_dm.append(up if up > down and up > 0 else 0.0)
        minus_dm.append(down if down > up and down > 0 else 0.0)

    # Wilder smooth helper
    def _wilder(vals: list[float], p: int) -> list[float]:
        out = [0.0] * len(vals)
        if len(vals) < p + 1:
            return out
        out[p] = sum(vals[1:p + 1]) / p
        for i in range(p + 1, len(vals)):
            out[i] = (out[i - 1] * (p - 1) + vals[i]) / p
        return out

    smooth_tr = _wilder(tr_list, period)
    smooth_plus = _wilder(plus_dm, period)
    smooth_minus = _wilder(minus_dm, period)

    # DX series
    dx_list = [0.0] * n
    for i in range(period, n):
        st = smooth_tr[i]
        if st <= 0:
            continue
        pdi = 100.0 * smooth_plus[i] / st
        mdi = 100.0 * smooth_minus[i] / st
        denom = pdi + mdi
        if denom > 0:
            dx_list[i] = 100.0 * abs(pdi - mdi) / denom

    # ADX = Wilder smooth du DX
    start = 2 * period
    if start < n:
        result[start] = sum(dx_list[period:start + 1]) / (period + 1) if (start - period + 1) > 0 else 0
        for i in range(start + 1, n):
            result[i] = (result[i - 1] * (period - 1) + dx_list[i]) / period

    return result


def _donchian_channel(
    candles: list[Candle], period: int,
) -> tuple[list[float], list[float]]:
    """Donchian Channel O(n×period). upper[i] = max(high[i-period..i-1]), lower[i] = min(low[i-period..i-1]).

    Note: on exclut la barre courante (lookback pur).
    """
    n = len(candles)
    upper = [0.0] * n
    lower = [0.0] * n
    for i in range(period, n):
        window_highs = [candles[j].high for j in range(i - period, i)]
        window_lows = [candles[j].low for j in range(i - period, i)]
        upper[i] = max(window_highs)
        lower[i] = min(window_lows)
    return upper, lower


# ── Moteur ─────────────────────────────────────────────────────────────────────


class DonchianEngine:
    """Simule la stratégie Donchian Trend Following bar par bar (daily)."""

    def __init__(
        self,
        candles_by_symbol: dict[str, list[Candle]],
        config: DonchianConfig,
        interval: str = "1d",
    ) -> None:
        self.candles = candles_by_symbol
        self.cfg = config
        self.interval = interval

        self.cash = config.initial_balance
        self.positions: dict[str, DonchianPosition] = {}
        self.trades: list[DonchianTrade] = []
        self.equity_curve: list[tuple[int, float]] = []
        self.cooldowns: dict[str, int] = {}
        self.n_signals = 0
        self.n_filtered = 0

        # Indicateurs pré-calculés
        self._ind: dict[str, dict] = {}

    def _precompute(self) -> None:
        """Pré-calcule tous les indicateurs O(n) par paire."""
        for sym, bars in self.candles.items():
            closes = [b.close for b in bars]

            # Donchian channels
            entry_upper, entry_lower = _donchian_channel(bars, self.cfg.entry_period)
            exit_upper, exit_lower = _donchian_channel(bars, self.cfg.exit_period)

            # ATR
            atr_vals = _atr_series(bars, self.cfg.atr_period)

            # ADX
            adx_vals = _adx_series(bars, self.cfg.adx_period)

            # EMA 200
            ema_vals = _ema_series(closes, self.cfg.ema_period) if self.cfg.use_ema_filter else [0.0] * len(bars)

            self._ind[sym] = {
                "closes": closes,
                "entry_upper": entry_upper,
                "entry_lower": entry_lower,
                "exit_upper": exit_upper,
                "exit_lower": exit_lower,
                "atr": atr_vals,
                "adx": adx_vals,
                "ema": ema_vals,
            }

        # ── Filtre régime BTC : SMA de BTC ──
        self._btc_regime: list[bool] = []  # True = bull regime (ok pour longs)
        if self.cfg.btc_regime_filter and self.cfg.btc_symbol in self.candles:
            btc_closes = [b.close for b in self.candles[self.cfg.btc_symbol]]
            period = self.cfg.btc_regime_period
            n_btc = len(btc_closes)
            self._btc_regime = [True] * n_btc  # Défaut = bull
            # SMA glissante O(n)
            if n_btc >= period:
                running_sum = sum(btc_closes[:period])
                for i in range(period, n_btc):
                    sma_val = running_sum / period
                    self._btc_regime[i] = btc_closes[i] > sma_val
                    running_sum += btc_closes[i] - btc_closes[i - period]

        logger.info("📊 Indicateurs pré-calculés pour %d paires", len(self._ind))

    def run(self) -> DonchianResult:
        symbols = sorted(self.candles.keys())
        if not symbols:
            raise ValueError("Aucune donnée fournie")

        warmup = max(
            self.cfg.entry_period + 5,
            self.cfg.exit_period + 5,
            2 * self.cfg.adx_period + 5,
            self.cfg.atr_period + 5,
            self.cfg.ema_period + 5 if self.cfg.use_ema_filter else 0,
        )

        # ── Timeline globale basée sur timestamps ──
        # Chaque paire a sa propre longueur ; on construit une timeline
        # unifiée à partir de la paire la plus longue (max_len).
        max_len = max(len(self.candles[s]) for s in symbols)

        # Vérifier qu'au moins une paire a assez de bougies
        valid_symbols = [s for s in symbols if len(self.candles[s]) > warmup]
        if not valid_symbols:
            raise ValueError(f"Aucune paire n'a assez de bougies (besoin > {warmup})")

        # Paire de référence pour les timestamps = la plus longue
        ref_sym = max(symbols, key=lambda s: len(self.candles[s]))
        ref_len = len(self.candles[ref_sym])

        # Index de départ par paire (warmup indépendant)
        # On construit un mapping timestamp → index par paire pour
        # synchroniser les barres. Les bougies daily ont le même
        # timestamp d'ouverture sur Binance pour toutes les paires.
        ts_to_idx: dict[str, dict[int, int]] = {}
        for sym in symbols:
            ts_to_idx[sym] = {}
            for idx, candle in enumerate(self.candles[sym]):
                ts_to_idx[sym][candle.timestamp] = idx
        self._ts_to_idx = ts_to_idx  # Accessible dans _check_entry

        logger.info(
            "🔵 Donchian Trend Following — %d paires, %d–%d bougies %s, capital $%.0f",
            len(symbols),
            min(len(self.candles[s]) for s in symbols),
            max_len,
            self.interval,
            self.cfg.initial_balance,
        )

        self._precompute()

        first_ts = None
        last_ts = None

        for ref_i in range(warmup, ref_len):
            ts = self.candles[ref_sym][ref_i].timestamp
            if first_ts is None:
                first_ts = ts
            last_ts = ts

            # 1. Check exits
            for sym in list(self.positions.keys()):
                idx = ts_to_idx.get(sym, {}).get(ts)
                if idx is not None:
                    self._check_exit(sym, idx)

            # 2. Check entries
            if len(self.positions) < self.cfg.max_positions:
                for sym in symbols:
                    if sym in self.positions:
                        continue
                    if len(self.positions) >= self.cfg.max_positions:
                        break

                    idx = ts_to_idx.get(sym, {}).get(ts)
                    if idx is None or idx < warmup:
                        continue  # Paire pas encore dispo ou pas assez de warmup

                    if sym in self.cooldowns and idx < self.cooldowns[sym]:
                        continue
                    self._check_entry(sym, idx)

            # 3. Equity snapshot (chaque jour)
            eq = self._compute_equity_ts(ts, ts_to_idx)
            self.equity_curve.append((ts, eq))

        # Close remaining positions
        for sym in list(self.positions.keys()):
            last_idx = len(self.candles[sym]) - 1
            self._close_position(sym, self.candles[sym][last_idx].close, last_idx, "END")

        final_eq = self.cash

        if self.equity_curve:
            self.equity_curve.append((last_ts, final_eq))

        return DonchianResult(
            trades=self.trades,
            equity_curve=self.equity_curve,
            initial_balance=self.cfg.initial_balance,
            final_equity=final_eq,
            n_signals=self.n_signals,
            n_filtered=self.n_filtered,
            config=self.cfg,
            start_date=datetime.fromtimestamp(first_ts / 1000, tz=timezone.utc),
            end_date=datetime.fromtimestamp(last_ts / 1000, tz=timezone.utc),
            pairs=list(self.candles.keys()),
        )

    # ── Entry ──────────────────────────────────────────────────────────────

    def _check_entry(self, symbol: str, idx: int) -> None:
        ind = self._ind[symbol]
        candle = self.candles[symbol][idx]
        price = candle.close

        entry_upper = ind["entry_upper"][idx]
        entry_lower = ind["entry_lower"][idx]

        if entry_upper <= 0 or entry_lower <= 0:
            return

        # Breakout detection
        side = None
        if price > entry_upper:
            side = "LONG"
        elif price < entry_lower:
            side = "SHORT"
        else:
            return  # Pas de breakout

        if side == "SHORT" and not self.cfg.allow_short:
            return

        # ── Paire exclue ? ──
        if symbol in self.cfg.excluded_pairs:
            return

        self.n_signals += 1

        # ── Filtre régime BTC (si activé) ──
        if self.cfg.btc_regime_filter and side == "LONG" and self._btc_regime:
            # Trouver l'index BTC correspondant à ce timestamp
            btc_sym = self.cfg.btc_symbol
            if btc_sym in self._ts_to_idx:
                candle_ts = self.candles[symbol][idx].timestamp
                btc_idx = self._ts_to_idx.get(btc_sym, {}).get(candle_ts)
                if btc_idx is not None and btc_idx < len(self._btc_regime):
                    if not self._btc_regime[btc_idx]:
                        self.n_filtered += 1
                        return  # Bear regime — pas de longs

        # ── Filtre ADX ──
        adx_val = ind["adx"][idx]
        if adx_val < self.cfg.adx_threshold:
            self.n_filtered += 1
            return  # Marché pas assez en tendance

        # ── Filtre EMA directionnel (optionnel) ──
        if self.cfg.use_ema_filter:
            ema_val = ind["ema"][idx]
            if side == "LONG" and price < ema_val:
                self.n_filtered += 1
                return
            if side == "SHORT" and price > ema_val:
                self.n_filtered += 1
                return

        # ── ATR pour SL ──
        atr_val = ind["atr"][idx]
        if atr_val <= 0:
            return

        # ── SL ──
        if side == "LONG":
            sl_price = price - self.cfg.sl_atr_mult * atr_val
        else:
            sl_price = price + self.cfg.sl_atr_mult * atr_val

        sl_distance = abs(price - sl_price)
        if sl_distance <= 0:
            return

        # ── Position sizing ──
        base_capital = self.cash if self.cfg.compound else self.cfg.initial_balance
        risk_amount = base_capital * self.cfg.risk_percent
        size = risk_amount / sl_distance
        cost = size * price

        max_cost = self.cash * self.cfg.max_position_pct
        if cost > max_cost:
            size = max_cost / price
            cost = size * price

        if cost > self.cash * 0.95:
            return  # Pas assez de cash

        # ── Entry avec slippage ──
        slip = self.cfg.slippage_pct
        if side == "LONG":
            entry_price = price * (1 + slip)
        else:
            entry_price = price * (1 - slip)

        fee = cost * self.cfg.fee_pct
        self.cash -= (cost + fee)

        self.positions[symbol] = DonchianPosition(
            symbol=symbol,
            side=side,
            entry_price=entry_price,
            sl_price=sl_price,
            size=size,
            cost=cost,
            entry_time=candle.timestamp,
            entry_bar_idx=idx,
            atr_at_entry=atr_val,
            peak_price=entry_price,
        )

        logger.debug(
            "🔵 ENTRY %s %s @ %.4f | SL %.4f | ATR %.4f | ADX %.1f",
            side, symbol, entry_price, sl_price, atr_val, adx_val,
        )

    # ── Exit ───────────────────────────────────────────────────────────────

    def _check_exit(self, symbol: str, idx: int) -> None:
        pos = self.positions[symbol]
        candle = self.candles[symbol][idx]
        ind = self._ind[symbol]

        if pos.side == "LONG":
            # Update peak
            if candle.high > pos.peak_price:
                pos.peak_price = candle.high

            # SL hit
            if candle.low <= pos.sl_price:
                self._close_position(symbol, pos.sl_price, idx, "SL")
                return

            # Donchian exit : close < exit_lower
            if self.cfg.use_donchian_exit:
                exit_lower = ind["exit_lower"][idx]
                if exit_lower > 0 and candle.close < exit_lower:
                    self._close_position(symbol, candle.close, idx, "DONCHIAN_EXIT")
                    return
            else:
                # ATR trailing
                new_sl = pos.peak_price - self.cfg.trail_atr_mult * pos.atr_at_entry
                if new_sl > pos.sl_price:
                    pos.sl_price = new_sl

        else:  # SHORT
            # Update peak (lowest)
            if candle.low < pos.peak_price:
                pos.peak_price = candle.low

            # SL hit
            if candle.high >= pos.sl_price:
                self._close_position(symbol, pos.sl_price, idx, "SL")
                return

            # Donchian exit : close > exit_upper
            if self.cfg.use_donchian_exit:
                exit_upper = ind["exit_upper"][idx]
                if exit_upper > 0 and candle.close > exit_upper:
                    self._close_position(symbol, candle.close, idx, "DONCHIAN_EXIT")
                    return
            else:
                # ATR trailing
                new_sl = pos.peak_price + self.cfg.trail_atr_mult * pos.atr_at_entry
                if new_sl < pos.sl_price:
                    pos.sl_price = new_sl

    # ── Close ──────────────────────────────────────────────────────────────

    def _close_position(self, symbol: str, exit_price: float, idx: int, reason: str) -> None:
        pos = self.positions.pop(symbol)

        slip = self.cfg.slippage_pct
        if reason in ("SL", "END"):
            if pos.side == "LONG":
                actual_exit = exit_price * (1 - slip)
            else:
                actual_exit = exit_price * (1 + slip)
        else:
            actual_exit = exit_price

        fee = pos.size * actual_exit * self.cfg.fee_pct

        if pos.side == "LONG":
            proceeds = pos.size * actual_exit - fee
            pnl_usd = proceeds - pos.cost
        else:
            proceeds = pos.size * (2 * pos.entry_price - actual_exit) - fee
            pnl_usd = pos.size * (pos.entry_price - actual_exit) - fee - (pos.cost * self.cfg.fee_pct)

        self.cash += pos.cost + pnl_usd
        pnl_pct = pnl_usd / pos.cost if pos.cost > 0 else 0
        hold_bars = idx - pos.entry_bar_idx

        self.trades.append(DonchianTrade(
            symbol=symbol,
            side=pos.side,
            entry_price=pos.entry_price,
            exit_price=actual_exit,
            size=pos.size,
            entry_time=pos.entry_time,
            exit_time=self.candles[symbol][idx].timestamp,
            pnl_usd=pnl_usd,
            pnl_pct=pnl_pct,
            exit_reason=reason,
            hold_days=hold_bars,
            atr_at_entry=pos.atr_at_entry,
        ))

        self.cooldowns[symbol] = idx + self.cfg.cooldown_days

        emoji = "✅" if pnl_usd >= 0 else "❌"
        logger.debug(
            "%s CLOSE %s %s @ %.4f (%s) | PnL $%.2f (%.2f%%) | %d days",
            emoji, pos.side, symbol, actual_exit, reason,
            pnl_usd, pnl_pct * 100, hold_bars,
        )

    # ── Equity ─────────────────────────────────────────────────────────────

    def _compute_equity(self, idx: int, symbols: list[str]) -> float:
        eq = self.cash
        for sym, pos in self.positions.items():
            if idx < len(self.candles[sym]):
                current_price = self.candles[sym][idx].close
                if pos.side == "LONG":
                    eq += pos.size * current_price
                else:
                    unrealized = pos.size * (pos.entry_price - current_price)
                    eq += pos.cost + unrealized
        return eq

    def _compute_equity_ts(self, ts: int, ts_to_idx: dict[str, dict[int, int]]) -> float:
        """Compute equity using timestamp-based index lookup."""
        eq = self.cash
        for sym, pos in self.positions.items():
            idx = ts_to_idx.get(sym, {}).get(ts)
            if idx is not None:
                current_price = self.candles[sym][idx].close
                if pos.side == "LONG":
                    eq += pos.size * current_price
                else:
                    unrealized = pos.size * (pos.entry_price - current_price)
                    eq += pos.cost + unrealized
            else:
                # Paire pas dispo à ce timestamp, utiliser le dernier prix connu
                eq += pos.cost  # Valeur d'entrée (approximation)
        return eq


# ── Métriques ──────────────────────────────────────────────────────────────────


def compute_donchian_metrics(result: DonchianResult) -> dict:
    """Calcule les KPIs du backtest Donchian."""
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
    for _, equity in eq:
        peak = max(peak, equity)
        dd = (equity - peak) / peak if peak > 0 else 0
        max_dd = min(max_dd, dd)

    # Returns for Sharpe/Sortino (daily snapshots)
    returns: list[float] = []
    for i in range(1, len(eq)):
        prev_eq = eq[i - 1][1]
        if prev_eq > 0:
            returns.append((eq[i][1] - prev_eq) / prev_eq)

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
        avg_hold = sum(t.hold_days for t in trades) / n
        best = max(trades, key=lambda t: t.pnl_usd)
        worst = min(trades, key=lambda t: t.pnl_usd)
    else:
        win_rate = profit_factor = avg_pnl_usd = avg_pnl_pct = avg_hold = 0
        best = worst = None

    by_pair = _group_trades(trades, lambda t: t.symbol)
    by_exit = _group_trades(trades, lambda t: t.exit_reason)
    by_side = _group_trades(trades, lambda t: t.side)

    # Par durée de trade
    def hold_bucket(t: DonchianTrade) -> str:
        if t.hold_days <= 3:
            return "1-3 days"
        elif t.hold_days <= 10:
            return "4-10 days"
        elif t.hold_days <= 30:
            return "11-30 days"
        else:
            return "30+ days"

    by_hold = _group_trades(trades, hold_bucket)

    return {
        "total_return": total_return,
        "cagr": cagr,
        "max_drawdown": max_dd,
        "sharpe": sharpe,
        "sortino": sortino,
        "n_trades": n,
        "n_signals": result.n_signals,
        "n_filtered": result.n_filtered,
        "win_rate": win_rate,
        "profit_factor": profit_factor,
        "avg_pnl_usd": avg_pnl_usd,
        "avg_pnl_pct": avg_pnl_pct,
        "avg_hold_days": avg_hold,
        "best_trade": best,
        "worst_trade": worst,
        "by_pair": by_pair,
        "by_exit": by_exit,
        "by_side": by_side,
        "by_hold": by_hold,
        "years": years,
        "days": days,
        "final_equity": final,
    }


# ── Helpers ────────────────────────────────────────────────────────────────────


def _sharpe(returns: list[float], periods_per_year: float = 365.0) -> float:
    """Sharpe annualisé (daily)."""
    if len(returns) < 2:
        return 0.0
    mu = sum(returns) / len(returns)
    var = sum((r - mu) ** 2 for r in returns) / (len(returns) - 1)
    std = math.sqrt(var) if var > 0 else 1e-9
    return (mu / std) * math.sqrt(periods_per_year)


def _sortino(returns: list[float], periods_per_year: float = 365.0) -> float:
    if len(returns) < 2:
        return 0.0
    mu = sum(returns) / len(returns)
    neg = [r for r in returns if r < 0]
    if not neg:
        return 99.0
    down_var = sum(r ** 2 for r in neg) / len(neg)
    down_std = math.sqrt(down_var) if down_var > 0 else 1e-9
    return (mu / down_std) * math.sqrt(periods_per_year)


def _group_trades(trades: list[DonchianTrade], key_fn) -> dict[str, dict]:
    groups: dict[str, list[DonchianTrade]] = defaultdict(list)
    for t in trades:
        groups[key_fn(t)].append(t)
    out: dict[str, dict] = {}
    for k, tlist in sorted(groups.items()):
        n = len(tlist)
        wins = sum(1 for t in tlist if t.pnl_usd > 0)
        pnl = sum(t.pnl_usd for t in tlist)
        gp = sum(t.pnl_usd for t in tlist if t.pnl_usd > 0) or 0
        gl = abs(sum(t.pnl_usd for t in tlist if t.pnl_usd <= 0)) or 1e-9
        avg_hold = sum(t.hold_days for t in tlist) / n if n else 0
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
