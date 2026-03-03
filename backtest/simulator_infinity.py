"""
Moteur de backtest — Infinity Bot

DCA inversé sur baisse + vente en paliers progressifs.
Timeframe : H4 (RSI, volume, prix).
Single pair (BTC-USD par défaut).

Logique :
  1. WAITING  → Attend les conditions d'entrée (prix < ref, RSI<35, vol>MA)
  2. ACCUMULATING → Achète en DCA sur chaque palier de baisse (-5%,-10%,...)
  3. DISTRIBUTING → Vend en paliers progressifs (+0.8%,+1.5%,...) quand RSI>50
  Stop-loss : -15% du PMP → vente market (taker 0.09%)
  Override  : +20% du PMP → vente complète
  Cycle reset après vente complète ou stop.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

from src.core.models import Candle
from src.core.infinity_engine import (
    InfinityConfig,
    InfinityPhase,
    InfinityExitReason,
    InfinityCycle,
    InfinityBuyLevel,
    InfinitySellLevel,
    rsi_series,
    sma_series,
    check_first_entry,
    compute_buy_size,
    check_buy_conditions,
    check_sell_conditions,
    check_override_sell,
    check_stop_loss,
)

logger = logging.getLogger("infinity_sim")


# ── Trade record ───────────────────────────────────────────────────────────────


@dataclass
class InfinityTrade:
    """Un cycle complet (entrée → sortie)."""
    cycle_num: int
    entry_date: str
    exit_date: str
    reference_price: float
    pmp: float
    exit_price: float
    exit_reason: str
    n_buys: int
    n_sells: int
    total_invested: float
    total_proceeds: float
    pnl: float
    pnl_pct: float
    duration_bars: int
    fees_paid: float


# ── Result ─────────────────────────────────────────────────────────────────────


@dataclass
class InfinityResult:
    """Résultat d'un backtest Infinity Bot."""
    trades: list[InfinityTrade]
    equity_curve: list[float]
    initial_balance: float
    final_equity: float
    config: InfinityConfig
    start_date: datetime
    end_date: datetime
    pair: str
    n_cycles: int
    n_stops: int
    n_overrides: int
    n_tp_complete: int
    n_timeouts: int


# ── Simulateur ─────────────────────────────────────────────────────────────────


class InfinityEngine:
    """Moteur de backtest bar-by-bar pour l'Infinity Bot."""

    def __init__(
        self,
        candles: list[Candle],
        config: InfinityConfig,
        pair: str = "BTC-USD",
    ) -> None:
        self.candles = candles
        self.cfg = config
        self.pair = pair

        # État
        self.cash = config.initial_balance
        self.cycle: Optional[InfinityCycle] = None
        self.reference_price: float = candles[0].close if candles else 0.0

        # Compteurs
        self.trades: list[InfinityTrade] = []
        self.equity_curve: list[float] = []
        self.n_cycles = 0
        self.n_stops = 0
        self.n_overrides = 0
        self.n_tp_complete = 0
        self.n_timeouts = 0
        self.total_fees = 0.0

        # Cooldown
        self._cooldown_until = 0
        self._consecutive_stops = 0

        # Indicateurs pré-calculés
        self._rsi: list[float] = []
        self._vol_ma: list[float] = []
        self._volumes: list[float] = []
        self._trailing_high: list[float] = []

    def _precompute(self) -> None:
        """Pré-calcule RSI, volume MA et trailing high."""
        closes = [c.close for c in self.candles]
        volumes = [c.volume for c in self.candles]
        self._rsi = rsi_series(closes, self.cfg.rsi_period)
        self._vol_ma = sma_series(volumes, self.cfg.volume_ma_len)
        self._volumes = volumes

        # Trailing high = max(close) sur les N dernières bougies
        n = len(closes)
        period = self.cfg.trailing_high_period
        self._trailing_high = [0.0] * n
        for i in range(n):
            start = max(0, i - period + 1)
            self._trailing_high[i] = max(closes[start: i + 1])

    def run(self) -> InfinityResult:
        """Exécute le backtest."""
        n = len(self.candles)
        if n < 50:
            raise ValueError(f"Pas assez de données : {n} bougies")

        self._precompute()
        warmup = max(self.cfg.rsi_period + 5, self.cfg.volume_ma_len + 5)

        logger.info(
            "🔷 Infinity Bot — %s, %d bougies H4, $%.0f",
            self.pair, n, self.cfg.initial_balance,
        )

        for idx in range(warmup, n):
            candle = self.candles[idx]
            close = candle.close
            rsi = self._rsi[idx]
            volume = self._volumes[idx]
            vol_ma = self._vol_ma[idx]
            trailing_high = self._trailing_high[idx]

            if self.cycle is None:
                # Hors cycle → le référence suit le trailing high
                self.reference_price = trailing_high
                # Phase WAITING — chercher une entrée
                self._try_first_entry(idx, close, rsi, volume, vol_ma, trailing_high)
            else:
                # Phase ACCUMULATING ou DISTRIBUTING
                self._manage_cycle(idx, close, rsi, volume, vol_ma)

            # Equity = cash + valeur positions ouvertes
            position_value = 0.0
            if self.cycle and self.cycle.size_remaining > 0:
                position_value = self.cycle.size_remaining * close
            self.equity_curve.append(self.cash + position_value)

        # Fermer position ouverte à la fin
        if self.cycle and self.cycle.size_remaining > 0:
            self._close_cycle(n - 1, self.candles[-1].close, "END")

        first_ts = self.candles[warmup].timestamp
        last_ts = self.candles[-1].timestamp

        return InfinityResult(
            trades=self.trades,
            equity_curve=self.equity_curve,
            initial_balance=self.cfg.initial_balance,
            final_equity=self.equity_curve[-1] if self.equity_curve else self.cfg.initial_balance,
            config=self.cfg,
            start_date=datetime.fromtimestamp(first_ts / 1000, tz=timezone.utc),
            end_date=datetime.fromtimestamp(last_ts / 1000, tz=timezone.utc),
            pair=self.pair,
            n_cycles=self.n_cycles,
            n_stops=self.n_stops,
            n_overrides=self.n_overrides,
            n_tp_complete=self.n_tp_complete,
            n_timeouts=self.n_timeouts,
        )

    # ── Première entrée ────────────────────────────────────────────────────

    def _try_first_entry(
        self, idx: int, close: float, rsi: float, volume: float, vol_ma: float,
        trailing_high: float,
    ) -> None:
        """Tente d'ouvrir le premier achat d'un nouveau cycle."""
        # Cooldown
        if idx < self._cooldown_until:
            return
        # Trop de stops consécutifs
        if self._consecutive_stops >= self.cfg.max_consecutive_stops:
            return

        if not check_first_entry(
            close, trailing_high, self.cfg.entry_drop_pct,
            rsi, self.cfg.first_entry_rsi_max, volume, vol_ma,
            self.cfg.require_volume_entry,
        ):
            return

        # Calcul montant 1er achat (proportionnel à l'equity si activé)
        current_equity = self.cash  # Pas de position ouverte ici
        if self.cfg.scale_with_equity:
            pct = self.cfg.buy_pcts[0] if self.cfg.buy_pcts else 0.15
            amount = current_equity * pct
        else:
            amount = self.cfg.buy_amounts[0]
            if amount <= 0:
                amount = min(self.cash, current_equity * 0.15)

        # RSI gate
        if rsi >= self.cfg.rsi_half_buy:
            return  # RSI > 50, pas d'achat
        if rsi >= self.cfg.rsi_full_buy:
            amount *= 0.5  # RSI 30-50 → moitié

        if amount <= 0 or amount > self.cash:
            return

        # Exécuter l'achat (maker)
        fee = amount * self.cfg.maker_fee
        net_amount = amount - fee
        size = net_amount / close
        self.cash -= amount
        self.total_fees += fee

        buy = InfinityBuyLevel(
            level=0,
            price=close,
            size=size,
            cost=amount,
            bar_idx=idx,
            timestamp=self.candles[idx].timestamp,
        )

        self.cycle = InfinityCycle(
            reference_price=self.reference_price,
            phase=InfinityPhase.ACCUMULATING,
            buys=[buy],
            total_size=size,
            total_cost=amount,
            pmp=close,
            size_remaining=size,
            cycle_start_bar=idx,
            cycle_start_ts=self.candles[idx].timestamp,
        )

        self.n_cycles += 1
        self._consecutive_stops = 0

        logger.debug(
            "🟢 ENTRY #%d @ %.2f | RSI=%.1f | $%.2f → %.6f BTC",
            self.n_cycles, close, rsi, amount, size,
        )

    # ── Gestion du cycle ───────────────────────────────────────────────────

    def _manage_cycle(
        self, idx: int, close: float, rsi: float, volume: float, vol_ma: float,
    ) -> None:
        """Gère un cycle actif : achats DCA, ventes, stop, breakeven."""
        cycle = self.cycle
        assert cycle is not None

        # ── 1. Breakeven stop (si activé et TP1 atteint) ──
        if (
            self.cfg.use_breakeven_stop
            and cycle.breakeven_active
            and close <= cycle.pmp
        ):
            self._close_cycle(idx, close, "BREAKEVEN")
            return

        # ── 2. Stop-loss check (-15% du PMP) ──
        if check_stop_loss(close, cycle.pmp, self.cfg.stop_loss_pct):
            self._close_cycle(idx, close, "STOP_LOSS")
            return

        # ── 3. Override sell (+20% du PMP) ──
        if check_override_sell(close, cycle.pmp, self.cfg.override_sell_pct):
            self._close_cycle(idx, close, "OVERRIDE_SELL")
            return

        # ── 4. Timeout ──
        bars_in = idx - cycle.cycle_start_bar
        if bars_in >= self.cfg.cycle_timeout_bars:
            self._close_cycle(idx, close, "TIMEOUT")
            return

        # ── 5. Achats DCA (si encore en accumulation) ──
        if cycle.phase == InfinityPhase.ACCUMULATING:
            self._try_dca_buy(idx, close, rsi, volume, vol_ma)

        # ── 6. Ventes par paliers ──
        self._try_sell_levels(idx, close, rsi)

    def _try_dca_buy(
        self, idx: int, close: float, rsi: float, volume: float, vol_ma: float,
    ) -> None:
        """Tente d'acheter au prochain palier DCA."""
        cycle = self.cycle
        assert cycle is not None

        n_buys = len(cycle.buys)
        if n_buys >= self.cfg.max_buy_levels:
            return

        # Vérifier conditions d'achat
        if not check_buy_conditions(
            close, cycle.pmp, rsi, self.cfg.rsi_half_buy, volume, vol_ma,
        ):
            return

        # Vérifier que le prix a atteint le palier
        target_drop = self.cfg.buy_levels[n_buys]
        target_price = cycle.reference_price * (1 + target_drop)

        if close > target_price:
            return

        # Max investi ?
        current_equity = self.cash + (cycle.size_remaining * close if cycle.size_remaining > 0 else 0)
        max_invested = current_equity * self.cfg.max_invested_pct
        if cycle.total_cost >= max_invested:
            return

        # Montant (proportionnel à l'equity si activé)
        if self.cfg.scale_with_equity:
            pct = self.cfg.buy_pcts[n_buys] if n_buys < len(self.cfg.buy_pcts) else 0.05
            target_amount = current_equity * pct
        else:
            if n_buys < len(self.cfg.buy_amounts):
                target_amount = self.cfg.buy_amounts[n_buys]
            else:
                target_amount = 0
            if target_amount <= 0:
                target_amount = min(
                    self.cash,
                    max_invested - cycle.total_cost,
                    current_equity * 0.15,
                )

        amount = compute_buy_size(
            rsi, self.cfg.rsi_full_buy, self.cfg.rsi_half_buy,
            target_amount, self.cash, max_invested, cycle.total_cost,
        )

        if amount < 5:  # Minimum 5$
            return

        # Exécuter (maker)
        fee = amount * self.cfg.maker_fee
        net_amount = amount - fee
        size = net_amount / close
        self.cash -= amount
        self.total_fees += fee

        buy = InfinityBuyLevel(
            level=n_buys,
            price=close,
            size=size,
            cost=amount,
            bar_idx=idx,
            timestamp=self.candles[idx].timestamp,
        )

        cycle.buys.append(buy)
        cycle.total_size += size
        cycle.total_cost += amount
        cycle.size_remaining += size
        cycle.recalc_pmp()

        logger.debug(
            "🔽 DCA L%d @ %.2f | PMP=%.2f | RSI=%.1f | $%.2f",
            n_buys, close, cycle.pmp, rsi, amount,
        )

    def _try_sell_levels(self, idx: int, close: float, rsi: float) -> None:
        """Vérifie et exécute les ventes par paliers."""
        cycle = self.cycle
        assert cycle is not None

        if cycle.size_remaining <= 0:
            return

        any_sold = False

        for level_idx, sell_pct in enumerate(self.cfg.sell_levels):
            if level_idx in cycle.sell_levels_hit:
                continue

            if not check_sell_conditions(
                close, cycle.pmp, sell_pct, rsi, self.cfg.rsi_sell_min,
            ):
                continue

            # Quelle fraction vendre
            sell_fraction = self.cfg.sell_pcts[level_idx] if level_idx < len(self.cfg.sell_pcts) else 0.20
            sell_size = cycle.total_size * sell_fraction

            # Ne pas vendre plus que ce qui reste
            sell_size = min(sell_size, cycle.size_remaining)
            if sell_size <= 0:
                continue

            # Exécuter (maker)
            proceeds = sell_size * close
            fee = proceeds * self.cfg.maker_fee
            net_proceeds = proceeds - fee
            self.cash += net_proceeds
            self.total_fees += fee

            cycle.size_remaining -= sell_size
            cycle.total_proceeds += net_proceeds
            cycle.sell_levels_hit.add(level_idx)

            sell = InfinitySellLevel(
                level=level_idx,
                price=close,
                size=sell_size,
                proceeds=net_proceeds,
                bar_idx=idx,
                timestamp=self.candles[idx].timestamp,
            )
            cycle.sells.append(sell)
            any_sold = True

            # Activer le breakeven stop après le level configuré
            if level_idx == self.cfg.breakeven_after_level and self.cfg.use_breakeven_stop:
                cycle.breakeven_active = True

            logger.debug(
                "🟡 SELL L%d @ %.2f (+%.1f%%) | %.6f BTC → $%.2f | remain=%.6f",
                level_idx, close, sell_pct * 100, sell_size, net_proceeds,
                cycle.size_remaining,
            )

        # Passer en phase DISTRIBUTING dès le premier sell
        if any_sold and cycle.phase == InfinityPhase.ACCUMULATING:
            cycle.phase = InfinityPhase.DISTRIBUTING

        # Si tout vendu → cycle terminé proprement
        if cycle.size_remaining <= 1e-12 and any_sold:
            self._record_trade(idx, close, InfinityExitReason.TP_COMPLETE.value)
            self.n_tp_complete += 1
            self.reference_price = close  # Nouveau référence = dernier sell
            self._cooldown_until = idx + self.cfg.cooldown_bars
            self.cycle = None

    def _close_cycle(self, idx: int, close: float, reason: str) -> None:
        """Ferme un cycle : vend tout au marché."""
        cycle = self.cycle
        if cycle is None or cycle.size_remaining <= 0:
            self.cycle = None
            return

        # Vente market (taker fee pour stop, maker pour override/timeout/breakeven)
        is_stop = reason == "STOP_LOSS"
        fee_rate = self.cfg.taker_fee if is_stop else self.cfg.maker_fee
        proceeds = cycle.size_remaining * close
        fee = proceeds * fee_rate
        net = proceeds - fee
        self.cash += net
        self.total_fees += fee

        cycle.total_proceeds += net
        cycle.size_remaining = 0.0

        self._record_trade(idx, close, reason)

        # Compteurs
        if reason == "STOP_LOSS":
            self.n_stops += 1
            self._consecutive_stops += 1
        elif reason == "OVERRIDE_SELL":
            self.n_overrides += 1
            self._consecutive_stops = 0
        elif reason == "TIMEOUT":
            self.n_timeouts += 1
            self._consecutive_stops = 0
        elif reason == "BREAKEVEN":
            self._consecutive_stops = 0    # Breakeven n'est pas un stop
        else:
            self._consecutive_stops = 0

        # Nouveau référence
        self.reference_price = close
        self._cooldown_until = idx + self.cfg.cooldown_bars
        self.cycle = None

    def _record_trade(self, idx: int, exit_price: float, reason: str) -> None:
        """Enregistre un trade dans l'historique."""
        cycle = self.cycle
        if cycle is None:
            return

        pnl = cycle.total_proceeds - cycle.total_cost
        pnl_pct = (pnl / cycle.total_cost * 100) if cycle.total_cost > 0 else 0.0
        duration = idx - cycle.cycle_start_bar

        entry_ts = cycle.cycle_start_ts
        exit_ts = self.candles[idx].timestamp
        entry_date = datetime.fromtimestamp(entry_ts / 1000, tz=timezone.utc).strftime("%Y-%m-%d")
        exit_date = datetime.fromtimestamp(exit_ts / 1000, tz=timezone.utc).strftime("%Y-%m-%d")

        trade = InfinityTrade(
            cycle_num=self.n_cycles,
            entry_date=entry_date,
            exit_date=exit_date,
            reference_price=cycle.reference_price,
            pmp=cycle.pmp,
            exit_price=exit_price,
            exit_reason=reason,
            n_buys=len(cycle.buys),
            n_sells=len(cycle.sells),
            total_invested=cycle.total_cost,
            total_proceeds=cycle.total_proceeds,
            pnl=pnl,
            pnl_pct=pnl_pct,
            duration_bars=duration,
            fees_paid=0.0,  # Tracked globally
        )
        self.trades.append(trade)


# ── Métriques ──────────────────────────────────────────────────────────────────


def compute_infinity_metrics(result: InfinityResult) -> dict:
    """Calcule les métriques de performance."""
    trades = result.trades
    if not trades:
        return {}

    pnls = [t.pnl for t in trades]
    pnl_pcts = [t.pnl_pct for t in trades]
    winners = [p for p in pnls if p > 0]
    losers = [p for p in pnls if p < 0]

    total_pnl = sum(pnls)
    win_rate = len(winners) / len(pnls) * 100 if pnls else 0

    gross_profit = sum(winners) if winners else 0
    gross_loss = abs(sum(losers)) if losers else 0
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")

    avg_pnl = total_pnl / len(pnls) if pnls else 0
    avg_pnl_pct = sum(pnl_pcts) / len(pnl_pcts) if pnl_pcts else 0

    avg_win = sum(winners) / len(winners) if winners else 0
    avg_loss = sum(losers) / len(losers) if losers else 0

    # Durées
    durations = [t.duration_bars for t in trades]
    avg_duration = sum(durations) / len(durations) if durations else 0

    # Max drawdown sur equity curve
    eq = result.equity_curve
    max_dd = 0.0
    peak = eq[0] if eq else result.initial_balance
    for v in eq:
        if v > peak:
            peak = v
        dd = (v - peak) / peak
        if dd < max_dd:
            max_dd = dd

    # Return
    total_return = (result.final_equity - result.initial_balance) / result.initial_balance

    # Période en jours
    if result.start_date and result.end_date:
        days = (result.end_date - result.start_date).days
    else:
        days = len(eq) * 4 / 24  # H4 bars

    months = days / 30.44
    monthly_return = total_return / months if months > 0 else 0
    cagr = (1 + total_return) ** (365.25 / max(days, 1)) - 1 if days > 0 else 0

    # Sharpe (annualisé)
    import math
    if len(pnl_pcts) > 1:
        mean_r = sum(pnl_pcts) / len(pnl_pcts)
        var_r = sum((p - mean_r) ** 2 for p in pnl_pcts) / (len(pnl_pcts) - 1)
        std_r = math.sqrt(var_r) if var_r > 0 else 1
        # Annualisation : ~2190 bougies H4 / an, ~N cycles / an
        cycles_per_year = len(pnls) / max(days / 365.25, 0.01)
        sharpe = (mean_r / std_r) * math.sqrt(cycles_per_year) if std_r > 0 else 0
    else:
        sharpe = 0

    return {
        "total_return": total_return,
        "monthly_return": monthly_return,
        "cagr": cagr,
        "max_dd": max_dd,
        "sharpe": sharpe,
        "n_trades": len(pnls),
        "win_rate": win_rate,
        "profit_factor": profit_factor,
        "avg_pnl": avg_pnl,
        "avg_pnl_pct": avg_pnl_pct,
        "avg_win": avg_win,
        "avg_loss": avg_loss,
        "avg_duration_bars": avg_duration,
        "total_pnl": total_pnl,
        "gross_profit": gross_profit,
        "gross_loss": gross_loss,
        "days": days,
        "months": months,
    }
