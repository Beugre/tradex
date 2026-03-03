"""
Moteur de backtest — Smart Infinity Grid 2.0

Logique :
  - Timeframes : H1 (signaux, RSI, ATR, EMA200) + H4 (filtre tendance EMA200)
  - Activation : ATR H1 élevé + RSI H1 < 35 + EMA200 H1 intacte
  - Entrée initiale : RSI H1 < 30 + bougie de rejet (mèche basse ≥ 1.5× body)
  - Grid dynamique : 5 niveaux à -1.5%, -3%, -5%, -7%, -10% avec sizing croissant
  - Sortie : Scale-out sur 3 TP dynamiques basés sur le PMP
    - TP1 = PMP + 0.8% → vend 40%
    - TP2 = PMP + 1.5% → vend 35%
    - TP3 = PMP + 2.5% → vend 25% (reliquat)
  - Stop global : EMA200 H4 cassée | RSI < 25 pendant 12 bougies | DD > 20%

Exécution : maker-only (0% fee Revolut X).
Mode : long-only (dip-buy grid).
Capital : 20% max par cycle, 3 cycles simultanés max.
"""

from __future__ import annotations

import logging
import math
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

from src.core.models import Candle
from src.core.grid_engine import (
    GridConfig,
    GridMode,
    GridPhase,
    GridCycle,
    GridLevel,
    GridStopReason,
    ema_series,
    sma_series,
    atr_series,
    rsi_series,
    check_h4_trend_ok,
    check_activation,
    check_entry_signal,
    check_bounce_entry,
    compute_grid_prices,
    compute_pmp,
    compute_tp_prices,
    check_global_stop,
    check_volatility_filter,
    micro_dip_config,
    deep_dip_config,
)

logger = logging.getLogger(__name__)


# ── Structures ─────────────────────────────────────────────────────────────────


@dataclass
class GridTrade:
    """Un trade (cycle) terminé."""
    symbol: str
    n_levels: int               # Nombre de niveaux remplis
    initial_entry: float        # Prix d'entrée initial
    pmp: float                  # Prix moyen pondéré final
    total_size: float           # Taille totale achetée
    total_cost: float           # Coût total
    total_proceeds: float       # Produit total des ventes
    entry_time: int             # Timestamp ms du début du cycle
    exit_time: int              # Timestamp ms de fin
    pnl_usd: float
    pnl_pct: float
    exit_reason: str            # TP_COMPLETE, GLOBAL_STOP, TIMEOUT, END
    tp1_hit: bool
    tp2_hit: bool
    tp3_hit: bool
    hold_bars: int              # Durée en bougies H1
    max_unrealized_dd: float    # Pire drawdown intra-cycle


@dataclass
class GridResult:
    """Résultat complet du backtest."""
    trades: list[GridTrade]
    equity_curve: list[tuple[int, float]]  # (timestamp_ms, equity)
    initial_balance: float
    final_equity: float
    config: GridConfig
    start_date: datetime
    end_date: datetime
    pairs: list[str]
    # Compteurs
    n_activations: int
    n_entries: int
    n_global_stops: int
    n_tp_complete: int
    n_timeouts: int
    n_filtered: int


# ── Helpers H1 → H4 ───────────────────────────────────────────────────────────


def _resample_h1_to_h4(candles_h1: list[Candle]) -> list[Candle]:
    """Resample H1 → H4 (agrège 4 bougies H1 consécutives)."""
    result: list[Candle] = []
    buffer: list[Candle] = []
    for c in candles_h1:
        buffer.append(c)
        if len(buffer) == 4:
            h4 = Candle(
                timestamp=buffer[0].timestamp,
                open=buffer[0].open,
                high=max(b.high for b in buffer),
                low=min(b.low for b in buffer),
                close=buffer[-1].close,
                volume=sum(b.volume for b in buffer),
            )
            result.append(h4)
            buffer = []
    return result


# ── Moteur ─────────────────────────────────────────────────────────────────────


class GridEngine:
    """Simule la stratégie Smart Infinity Grid 2.0 bar par bar sur H1."""

    def __init__(
        self,
        candles_by_symbol: dict[str, list[Candle]],
        config: GridConfig,
    ) -> None:
        self.candles = candles_by_symbol
        self.cfg = config

        self.cash = config.initial_balance
        self.cycles: dict[str, GridCycle] = {}      # Cycles actifs
        self.trades: list[GridTrade] = []
        self.equity_curve: list[tuple[int, float]] = []
        self.cooldowns: dict[str, int] = {}          # symbol → bar_idx

        # Compteurs
        self.n_activations = 0
        self.n_entries = 0
        self.n_global_stops = 0
        self.n_tp_complete = 0
        self.n_timeouts = 0
        self.n_filtered = 0

        # Safety
        self._consecutive_losses = 0
        self._daily_pnl: dict[str, float] = defaultdict(float)
        self._safety_paused = False
        self._safety_paused_until_day = ""

        # Pré-calculés
        self._ind_h1: dict[str, dict] = {}
        self._ind_h4: dict[str, dict] = {}
        self._h4_candles: dict[str, list[Candle]] = {}

    def _precompute(self) -> None:
        """Pré-calcule tous les indicateurs."""
        for sym, bars in self.candles.items():
            closes = [b.close for b in bars]
            volumes = [b.volume for b in bars]

            self._ind_h1[sym] = {
                "closes": closes,
                "volumes": volumes,
                "ema200": ema_series(closes, self.cfg.ema200_h1_period),
                "ema20": ema_series(closes, 20),
                "rsi": rsi_series(closes, self.cfg.rsi_h1_period),
                "atr": atr_series(bars, self.cfg.atr_h1_period),
                "atr_ma": sma_series(
                    atr_series(bars, self.cfg.atr_h1_period),
                    self.cfg.atr_h1_ma_period,
                ),
            }

            # H4
            h4 = _resample_h1_to_h4(bars)
            self._h4_candles[sym] = h4
            h4_closes = [c.close for c in h4]
            self._ind_h4[sym] = {
                "closes": h4_closes,
                "ema200": ema_series(h4_closes, self.cfg.ema200_h4_period),
            }

        logger.info("📊 Indicateurs pré-calculés pour %d paires (H1+H4)", len(self._ind_h1))

    def _h1_to_h4_idx(self, h1_idx: int) -> int:
        """Convertit un index H1 en index H4."""
        return h1_idx // 4

    def run(self) -> GridResult:
        """Exécute le backtest bar par bar."""
        symbols = sorted(self.candles.keys())
        if not symbols:
            raise ValueError("Aucune donnée")

        min_len = min(len(self.candles[s]) for s in symbols)
        warmup = max(250, self.cfg.ema200_h1_period + 20)

        if min_len <= warmup:
            raise ValueError(f"Données insuffisantes ({min_len} barres, min {warmup})")

        logger.info(
            "🔷 Smart Grid 2.0 — %d paires, %d bougies H1, $%.0f",
            len(symbols), min_len, self.cfg.initial_balance,
        )

        self._precompute()

        for i in range(warmup, min_len):
            ts = self.candles[symbols[0]][i].timestamp
            day_str = datetime.fromtimestamp(ts / 1000, tz=timezone.utc).strftime("%Y-%m-%d")

            # Reset safety si nouveau jour
            if self._safety_paused and day_str != self._safety_paused_until_day:
                self._safety_paused = False
                self._consecutive_losses = 0

            # 1. Gérer les cycles actifs (exits, grid fills, TP checks)
            for sym in list(self.cycles.keys()):
                if i < len(self.candles[sym]):
                    self._manage_cycle(sym, i, day_str)

            # 2. Chercher de nouvelles entrées
            if not self._safety_paused:
                active_count = len(self.cycles)
                if active_count < self.cfg.max_simultaneous_cycles:
                    for sym in symbols:
                        if sym in self.cycles:
                            continue
                        if active_count >= self.cfg.max_simultaneous_cycles:
                            break
                        if i >= len(self.candles[sym]):
                            continue
                        if sym in self.cooldowns and i < self.cooldowns[sym]:
                            continue
                        if self._try_enter(sym, i):
                            active_count += 1

            # 3. Equity snapshot (toutes les 4 barres = 4h en H1)
            if i % 4 == 0:
                eq = self._compute_equity(i, symbols)
                self.equity_curve.append((ts, eq))

        # Close remaining
        for sym in list(self.cycles.keys()):
            last_idx = len(self.candles[sym]) - 1
            self._close_cycle(sym, self.candles[sym][last_idx].close, last_idx, "END", "")

        final_eq = self.cash
        first_ts = self.candles[symbols[0]][warmup].timestamp
        last_ts = self.candles[symbols[0]][min_len - 1].timestamp

        if self.equity_curve:
            self.equity_curve.append((last_ts, final_eq))

        return GridResult(
            trades=self.trades,
            equity_curve=self.equity_curve,
            initial_balance=self.cfg.initial_balance,
            final_equity=final_eq,
            config=self.cfg,
            start_date=datetime.fromtimestamp(first_ts / 1000, tz=timezone.utc),
            end_date=datetime.fromtimestamp(last_ts / 1000, tz=timezone.utc),
            pairs=list(self.candles.keys()),
            n_activations=self.n_activations,
            n_entries=self.n_entries,
            n_global_stops=self.n_global_stops,
            n_tp_complete=self.n_tp_complete,
            n_timeouts=self.n_timeouts,
            n_filtered=self.n_filtered,
        )

    # ── Entrée ─────────────────────────────────────────────────────────────

    def _try_enter(self, symbol: str, idx: int) -> bool:
        """Tente d'ouvrir un nouveau cycle de grid."""
        ind_h1 = self._ind_h1[symbol]
        candle = self.candles[symbol][idx]

        # Blacklist check
        if symbol in self.cfg.blacklist:
            self.n_filtered += 1
            return False

        # Filtre H4 : EMA200 intacte
        h4_idx = self._h1_to_h4_idx(idx)
        ind_h4 = self._ind_h4[symbol]
        if h4_idx >= len(ind_h4["ema200"]) or h4_idx < 1:
            self.n_filtered += 1
            return False

        ema200_h4 = ind_h4["ema200"][h4_idx]
        close_h4 = ind_h4["closes"][h4_idx]

        if not check_h4_trend_ok(ema200_h4, close_h4, self.cfg.ema200_break_pct):
            self.n_filtered += 1
            return False

        # Filtre volatilité ATR/Price
        atr_val_pre = ind_h1["atr"][idx]
        close_pre = ind_h1["closes"][idx]
        if not check_volatility_filter(atr_val_pre, close_pre, self.cfg.max_atr_price_ratio):
            self.n_filtered += 1
            return False

        # Activation H1 : ATR élevé + RSI < seuil + EMA200 intacte
        rsi_val = ind_h1["rsi"][idx]
        atr_val = ind_h1["atr"][idx]
        atr_ma_val = ind_h1["atr_ma"][idx]
        ema200_h1 = ind_h1["ema200"][idx]
        ema20_h1 = ind_h1["ema20"][idx]
        close_h1 = ind_h1["closes"][idx]

        # Chemin 1 : Activation classique (RSI survendu)
        classic_activated = check_activation(
            rsi_val, atr_val, atr_ma_val, ema200_h1, close_h1,
            self.cfg.rsi_activation, self.cfg.ema200_break_pct,
        )

        # Chemin 2 : Bounce (rebond technique sur EMA20)
        bounce_activated = False
        if self.cfg.bounce_enabled and not classic_activated:
            # Le bounce ne nécessite pas ATR > MA ni RSI extrême
            # Juste EMA200 intacte + EMA20 touchée
            ema200_ok = ema200_h1 > 0 and close_h1 > ema200_h1 * (1 - self.cfg.ema200_break_pct)
            if ema200_ok:
                bounce_activated = check_bounce_entry(
                    rsi_val, candle, ema20_h1,
                    self.cfg.bounce_rsi_min,
                    self.cfg.bounce_rsi_max,
                    self.cfg.bounce_ema_touch_pct,
                )

        if not classic_activated and not bounce_activated:
            self.n_filtered += 1
            return False

        self.n_activations += 1

        # Signal d'entrée
        entry_ok = False
        if classic_activated:
            # Classique : RSI < 30 + bougie de rejet
            entry_ok = check_entry_signal(
                rsi_val, candle,
                self.cfg.rsi_entry, self.cfg.rejection_wick_ratio,
            )
        if not entry_ok and bounce_activated:
            # Bounce : la condition d'activation EST le signal d'entrée
            entry_ok = True

        if not entry_ok:
            return False

        # ✅ Entrée ! Ouvrir le cycle
        self.n_entries += 1

        # Calcul de la taille de base
        cycle_budget = self.cash * self.cfg.capital_per_cycle_pct
        # Répartir le budget sur tous les niveaux potentiels
        total_weight = sum(self.cfg.grid_multipliers)
        base_size_usd = cycle_budget / total_weight
        level0_usd = base_size_usd * self.cfg.grid_multipliers[0]

        entry_price = candle.close
        slip = entry_price * self.cfg.slippage_pct
        actual_entry = entry_price * (1 + self.cfg.slippage_pct)  # Long buy → slippage haut
        fee = level0_usd * self.cfg.fee_pct

        size = (level0_usd - fee) / actual_entry
        cost = size * actual_entry + fee

        if cost > self.cash:
            return False

        self.cash -= cost

        level0 = GridLevel(
            level=0,
            entry_price=actual_entry,
            size=size,
            cost=cost,
            bar_idx=idx,
            timestamp=candle.timestamp,
        )

        cycle = GridCycle(
            symbol=symbol,
            phase=GridPhase.ACCUMULATING,
            initial_entry_price=actual_entry,
            levels_filled=[level0],
            total_size=size,
            total_cost=cost,
            pmp=actual_entry,
            peak_price=candle.high,
            cycle_start_bar=idx,
            cycle_start_ts=candle.timestamp,
            size_remaining=size,
        )

        self.cycles[symbol] = cycle

        logger.debug(
            "🔷 GRID ENTRY %s @ %.4f | RSI=%.1f ATR=%.4f | budget=$%.2f size=%.6f",
            symbol, actual_entry, rsi_val, atr_val, cycle_budget, size,
        )

        return True

    # ── Gestion du cycle ───────────────────────────────────────────────────

    def _manage_cycle(self, symbol: str, idx: int, day_str: str) -> None:
        """Gère un cycle actif : grid fills, TP checks, stop global."""
        cycle = self.cycles[symbol]
        candle = self.candles[symbol][idx]
        ind_h1 = self._ind_h1[symbol]

        close = candle.close
        rsi_val = ind_h1["rsi"][idx]

        # ── Stop global check ──
        h4_idx = self._h1_to_h4_idx(idx)
        ind_h4 = self._ind_h4[symbol]

        ema200_h4 = 0.0
        if h4_idx < len(ind_h4["ema200"]):
            ema200_h4 = ind_h4["ema200"][h4_idx]

        # RSI sustained low tracking
        if rsi_val < self.cfg.rsi_stop_threshold:
            cycle.rsi_below_stop_count += 1
        else:
            cycle.rsi_below_stop_count = 0

        # Calcul unrealized PnL
        unrealized_value = cycle.total_size * close if cycle.total_size > 0 else 0
        # Inclure le PnL déjà réalisé par les scale-outs
        effective_cost = cycle.total_cost - cycle.realized_proceeds
        unrealized_pnl = unrealized_value - effective_cost if effective_cost > 0 else unrealized_value - cycle.total_cost

        # ATR courant pour stop adaptatif
        atr_current = ind_h1["atr"][idx]

        stop_reason = check_global_stop(
            close=close,
            ema200_h4=ema200_h4,
            ema_break_pct=self.cfg.ema_stop_break_pct,
            rsi_h1=rsi_val,
            rsi_stop_threshold=self.cfg.rsi_stop_threshold,
            rsi_below_count=cycle.rsi_below_stop_count,
            rsi_stop_bars=self.cfg.rsi_stop_bars,
            cycle_cost=cycle.total_cost,
            cycle_unrealized=unrealized_pnl,
            max_dd_pct=self.cfg.max_drawdown_pct,
            use_ema_stop=self.cfg.ema200_h4_stop,
            pmp=cycle.pmp,
            pmp_stop_pct=self.cfg.pmp_stop_pct,
            atr=atr_current,
            atr_stop_multiplier=self.cfg.atr_stop_multiplier,
        )

        if stop_reason:
            self.n_global_stops += 1
            self._close_cycle(symbol, close, idx, f"STOP_{stop_reason}", day_str)
            return

        # ── Timeout ──
        bars_in_cycle = idx - cycle.cycle_start_bar
        if bars_in_cycle >= self.cfg.cycle_timeout_bars:
            self.n_timeouts += 1
            self._close_cycle(symbol, close, idx, "TIMEOUT", day_str)
            return

        # ── Grid fills : vérifier si le prix a atteint un nouveau niveau ──
        if cycle.phase == GridPhase.ACCUMULATING:
            self._check_grid_fills(symbol, idx)

        # ── TP checks : scale-out progressif ──
        if cycle.size_remaining > 0:
            self._check_tp(symbol, idx, day_str)

    def _check_grid_fills(self, symbol: str, idx: int) -> None:
        """Vérifie si le prix a atteint un nouveau niveau de grid."""
        cycle = self.cycles[symbol]
        candle = self.candles[symbol][idx]

        n_filled = len(cycle.levels_filled)
        if n_filled >= self.cfg.grid_levels:
            return  # Tous les niveaux sont remplis

        # Calculer les prix de grid restants
        grid_prices = compute_grid_prices(
            cycle.initial_entry_price,
            self.cfg.grid_distances,
        )

        for level_idx in range(n_filled, self.cfg.grid_levels):
            target_price = grid_prices[level_idx]

            # Le prix a-t-il touché ce niveau ? (low ≤ target)
            if candle.low <= target_price:
                # Budget du cycle
                cycle_budget = self.cfg.initial_balance * self.cfg.capital_per_cycle_pct
                total_weight = sum(self.cfg.grid_multipliers)
                base_size_usd = cycle_budget / total_weight
                level_usd = base_size_usd * self.cfg.grid_multipliers[level_idx]

                actual_price = target_price * (1 + self.cfg.slippage_pct)
                fee = level_usd * self.cfg.fee_pct
                size = (level_usd - fee) / actual_price
                cost = size * actual_price + fee

                if cost > self.cash:
                    continue  # Pas assez de cash

                self.cash -= cost

                new_level = GridLevel(
                    level=level_idx,
                    entry_price=actual_price,
                    size=size,
                    cost=cost,
                    bar_idx=idx,
                    timestamp=candle.timestamp,
                )

                cycle.levels_filled.append(new_level)
                cycle.total_size += size
                cycle.total_cost += cost
                cycle.size_remaining += size
                cycle.recalc_pmp()

                logger.debug(
                    "🔽 GRID FILL L%d %s @ %.4f | PMP=%.4f | levels=%d/%d",
                    level_idx, symbol, actual_price, cycle.pmp,
                    len(cycle.levels_filled), self.cfg.grid_levels,
                )

    def _check_tp(self, symbol: str, idx: int, day_str: str) -> None:
        """Vérifie les take-profits progressifs."""
        cycle = self.cycles[symbol]
        candle = self.candles[symbol][idx]
        close = candle.close

        tp1, tp2, tp3 = compute_tp_prices(
            cycle.pmp, self.cfg.tp1_pct, self.cfg.tp2_pct, self.cfg.tp3_pct,
        )

        # TP1
        if not cycle.tp1_hit and candle.high >= tp1:
            sell_size = cycle.total_size * self.cfg.tp1_exit_pct
            if sell_size > cycle.size_remaining:
                sell_size = cycle.size_remaining
            if sell_size > 0:
                actual_exit = tp1 * (1 - self.cfg.slippage_pct)
                fee = sell_size * actual_exit * self.cfg.fee_pct
                proceeds = sell_size * actual_exit - fee
                self.cash += proceeds
                cycle.size_remaining -= sell_size
                cycle.realized_proceeds += proceeds
                cycle.realized_pnl += proceeds - (sell_size * cycle.pmp)
                cycle.tp1_hit = True
                logger.debug("📈 TP1 %s @ %.4f | sold %.6f | remaining %.6f",
                             symbol, actual_exit, sell_size, cycle.size_remaining)

        # TP2
        if not cycle.tp2_hit and cycle.tp1_hit and candle.high >= tp2:
            sell_size = cycle.total_size * self.cfg.tp2_exit_pct
            if sell_size > cycle.size_remaining:
                sell_size = cycle.size_remaining
            if sell_size > 0:
                actual_exit = tp2 * (1 - self.cfg.slippage_pct)
                fee = sell_size * actual_exit * self.cfg.fee_pct
                proceeds = sell_size * actual_exit - fee
                self.cash += proceeds
                cycle.size_remaining -= sell_size
                cycle.realized_proceeds += proceeds
                cycle.realized_pnl += proceeds - (sell_size * cycle.pmp)
                cycle.tp2_hit = True
                logger.debug("📈 TP2 %s @ %.4f | sold %.6f | remaining %.6f",
                             symbol, actual_exit, sell_size, cycle.size_remaining)

        # TP3
        if not cycle.tp3_hit and cycle.tp2_hit and candle.high >= tp3:
            sell_size = cycle.size_remaining
            if sell_size > 0:
                actual_exit = tp3 * (1 - self.cfg.slippage_pct)
                fee = sell_size * actual_exit * self.cfg.fee_pct
                proceeds = sell_size * actual_exit - fee
                self.cash += proceeds
                cycle.size_remaining -= sell_size
                cycle.realized_proceeds += proceeds
                cycle.realized_pnl += proceeds - (sell_size * cycle.pmp)
                cycle.tp3_hit = True
                logger.debug("📈 TP3 %s @ %.4f | sold %.6f | CYCLE COMPLETE",
                             symbol, actual_exit, sell_size)

        # Si tout vendu → clôturer le cycle
        if cycle.size_remaining <= 1e-12:
            self.n_tp_complete += 1
            self._close_cycle(symbol, close, idx, "TP_COMPLETE", day_str)

    # ── Clôture de cycle ───────────────────────────────────────────────────

    def _close_cycle(self, symbol: str, exit_price: float, idx: int, reason: str, day_str: str) -> None:
        """Clôture un cycle de grid."""
        cycle = self.cycles.pop(symbol)

        # Vendre le restant si nécessaire
        remaining_proceeds = 0.0
        if cycle.size_remaining > 1e-12:
            actual_exit = exit_price * (1 - self.cfg.slippage_pct)
            fee = cycle.size_remaining * actual_exit * self.cfg.fee_pct
            remaining_proceeds = cycle.size_remaining * actual_exit - fee
            self.cash += remaining_proceeds

        total_proceeds = cycle.realized_proceeds + remaining_proceeds
        pnl_usd = total_proceeds - cycle.total_cost
        pnl_pct = pnl_usd / cycle.total_cost if cycle.total_cost > 0 else 0
        bars_held = idx - cycle.cycle_start_bar

        # Max unrealized DD pendant le cycle
        max_dd = 0.0
        if cycle.total_cost > 0:
            # Calculer le pire DD rétrospectif sur toute la durée du cycle
            for bi in range(cycle.cycle_start_bar, min(idx + 1, len(self.candles[symbol]))):
                low = self.candles[symbol][bi].low
                min_val = cycle.total_size * low
                dd = (min_val - cycle.total_cost) / cycle.total_cost
                max_dd = min(max_dd, dd)

        self.trades.append(GridTrade(
            symbol=symbol,
            n_levels=len(cycle.levels_filled),
            initial_entry=cycle.initial_entry_price,
            pmp=cycle.pmp,
            total_size=cycle.total_size,
            total_cost=cycle.total_cost,
            total_proceeds=total_proceeds,
            entry_time=cycle.cycle_start_ts,
            exit_time=self.candles[symbol][idx].timestamp,
            pnl_usd=pnl_usd,
            pnl_pct=pnl_pct,
            exit_reason=reason,
            tp1_hit=cycle.tp1_hit,
            tp2_hit=cycle.tp2_hit,
            tp3_hit=cycle.tp3_hit,
            hold_bars=bars_held,
            max_unrealized_dd=max_dd,
        ))

        self.cooldowns[symbol] = idx + self.cfg.cooldown_bars

        # Safety tracking
        if pnl_usd < 0:
            self._consecutive_losses += 1
        else:
            self._consecutive_losses = 0

        # Daily PnL
        if day_str:
            self._daily_pnl[day_str] += pnl_usd
            daily_loss_pct = self._daily_pnl[day_str] / self.cfg.initial_balance
            if daily_loss_pct <= -self.cfg.daily_loss_limit_pct:
                self._safety_paused = True
                self._safety_paused_until_day = day_str
                logger.debug("🛑 Daily loss limit hit: $%.2f (%.2f%%)",
                             self._daily_pnl[day_str], daily_loss_pct * 100)

        if self._consecutive_losses >= self.cfg.max_consecutive_losses:
            self._safety_paused = True
            self._safety_paused_until_day = day_str
            logger.debug("🛑 %d consecutive losses — paused", self._consecutive_losses)

        logger.debug(
            "🔷 CYCLE %s %s — %d levels, PMP=%.4f, PnL=$%.2f (%.2f%%) in %d bars",
            reason, symbol, len(cycle.levels_filled), cycle.pmp,
            pnl_usd, pnl_pct * 100, bars_held,
        )

    # ── Equity ─────────────────────────────────────────────────────────────

    def _compute_equity(self, idx: int, symbols: list[str]) -> float:
        """Calcule l'equity totale (cash + positions ouvertes)."""
        eq = self.cash
        for sym, cycle in self.cycles.items():
            if idx < len(self.candles[sym]) and cycle.size_remaining > 0:
                current_price = self.candles[sym][idx].close
                eq += cycle.size_remaining * current_price
        return eq


# ── Métriques ──────────────────────────────────────────────────────────────────


def compute_grid_metrics(result: GridResult) -> dict:
    """Calcule les KPIs du backtest Smart Grid 2.0."""
    trades = result.trades
    eq = result.equity_curve
    init = result.initial_balance
    final = result.final_equity
    days = max((result.end_date - result.start_date).days, 1)
    years = days / 365.25

    total_return = (final - init) / init if init > 0 else 0
    cagr = (final / init) ** (1 / years) - 1 if final > 0 and years > 0 else 0
    monthly_return = (1 + total_return) ** (30.4375 / days) - 1 if days > 0 else 0

    # Drawdown
    peak = init
    max_dd = 0.0
    for _, equity in eq:
        peak = max(peak, equity)
        dd = (equity - peak) / peak if peak > 0 else 0
        max_dd = min(max_dd, dd)

    # Sharpe / Sortino sur returns 4h
    returns: list[float] = []
    for i in range(1, len(eq)):
        prev_eq = eq[i - 1][1]
        if prev_eq > 0:
            returns.append((eq[i][1] - prev_eq) / prev_eq)

    sharpe = _sharpe(returns)
    sortino = _sortino(returns)

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
        avg_levels = sum(t.n_levels for t in trades) / n
        best = max(trades, key=lambda t: t.pnl_usd)
        worst = min(trades, key=lambda t: t.pnl_usd)
        trades_per_day = n / days if days > 0 else 0
        tp1_pct = sum(1 for t in trades if t.tp1_hit) / n
        tp2_pct = sum(1 for t in trades if t.tp2_hit) / n
        tp3_pct = sum(1 for t in trades if t.tp3_hit) / n
    else:
        win_rate = profit_factor = avg_pnl_usd = avg_pnl_pct = 0
        avg_hold = avg_levels = trades_per_day = 0
        tp1_pct = tp2_pct = tp3_pct = 0
        best = worst = None

    by_pair = _group_trades(trades, lambda t: t.symbol)
    by_exit = _group_trades(trades, lambda t: t.exit_reason)
    by_levels = _group_trades(trades, lambda t: f"L{t.n_levels}")

    return {
        "total_return": total_return,
        "cagr": cagr,
        "monthly_return": monthly_return,
        "max_drawdown": max_dd,
        "sharpe": sharpe,
        "sortino": sortino,
        "n_trades": n,
        "trades_per_day": trades_per_day,
        "win_rate": win_rate,
        "profit_factor": profit_factor,
        "avg_pnl_usd": avg_pnl_usd,
        "avg_pnl_pct": avg_pnl_pct,
        "avg_hold_bars": avg_hold,
        "avg_levels": avg_levels,
        "tp1_hit_pct": tp1_pct,
        "tp2_hit_pct": tp2_pct,
        "tp3_hit_pct": tp3_pct,
        "best_trade": best,
        "worst_trade": worst,
        "by_pair": by_pair,
        "by_exit": by_exit,
        "by_levels": by_levels,
        # Compteurs
        "n_activations": result.n_activations,
        "n_entries": result.n_entries,
        "n_global_stops": result.n_global_stops,
        "n_tp_complete": result.n_tp_complete,
        "n_timeouts": result.n_timeouts,
        "n_filtered": result.n_filtered,
        # Meta
        "days": days,
        "years": years,
        "final_equity": final,
    }


# ── Helpers ────────────────────────────────────────────────────────────────────


def _sharpe(returns: list[float], periods_per_year: float = 2190) -> float:
    """Sharpe annualisé. 2190 = 6 snapshots/jour × 365."""
    if len(returns) < 2:
        return 0.0
    mu = sum(returns) / len(returns)
    var = sum((r - mu) ** 2 for r in returns) / (len(returns) - 1)
    std = math.sqrt(var) if var > 0 else 1e-9
    return (mu / std) * math.sqrt(periods_per_year)


def _sortino(returns: list[float], periods_per_year: float = 2190) -> float:
    if len(returns) < 2:
        return 0.0
    mu = sum(returns) / len(returns)
    neg = [r for r in returns if r < 0]
    if not neg:
        return 99.0
    down_var = sum(r ** 2 for r in neg) / len(neg)
    down_std = math.sqrt(down_var) if down_var > 0 else 1e-9
    return (mu / down_std) * math.sqrt(periods_per_year)


def _group_trades(trades: list[GridTrade], key_fn) -> dict[str, dict]:
    groups: dict[str, list[GridTrade]] = defaultdict(list)
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
        avg_levels = sum(t.n_levels for t in tlist) / n if n else 0
        out[k] = {
            "n": n,
            "wins": wins,
            "wr": wins / n if n else 0,
            "pnl": pnl,
            "pf": gp / gl,
            "avg_pct": sum(t.pnl_pct for t in tlist) / n if n else 0,
            "avg_hold": avg_hold,
            "avg_levels": avg_levels,
        }
    return out


# ── Dual-mode engine ──────────────────────────────────────────────────────────


@dataclass
class DualGridResult:
    """Résultat combiné MICRO + DEEP."""
    micro_result: GridResult
    deep_result: GridResult
    combined_trades: list[GridTrade]
    combined_equity: list[tuple[int, float]]
    initial_balance: float
    final_equity: float
    start_date: datetime
    end_date: datetime
    pairs_micro: list[str]
    pairs_deep: list[str]
    monthly_stop_events: int           # Nombre de mois stoppés par le -6% mensuel


class DualGridEngine:
    """
    Combine MICRO DIP (H1) + DEEP DIP (H4) sur un capital partagé.

    Le capital est splitté : micro_pct / deep_pct (default 60/40).
    Chaque mode a son propre moteur GridEngine indépendant.
    Stop global portefeuille si perte mensuelle > monthly_stop_pct.
    """

    def __init__(
        self,
        candles_h1: dict[str, list[Candle]],
        candles_h4: dict[str, list[Candle]],
        micro_cfg: GridConfig,
        deep_cfg: GridConfig,
        initial_balance: float = 1000.0,
        micro_pct: float = 0.60,
        deep_pct: float = 0.40,
        monthly_stop_pct: float = 0.06,
    ) -> None:
        self.initial_balance = initial_balance
        self.micro_pct = micro_pct
        self.deep_pct = deep_pct
        self.monthly_stop_pct = monthly_stop_pct

        micro_balance = initial_balance * micro_pct
        deep_balance = initial_balance * deep_pct

        # Configure les balances
        micro_cfg.initial_balance = micro_balance
        deep_cfg.initial_balance = deep_balance

        self.micro_engine = GridEngine(candles_h1, micro_cfg)
        self.deep_engine = GridEngine(candles_h4, deep_cfg)

    def run(self) -> DualGridResult:
        """Exécute les deux moteurs et combine les résultats."""
        logger.info(
            "🔷 DUAL MODE — MICRO ($%.0f, %d paires H1) + DEEP ($%.0f, %d paires H4)",
            self.micro_engine.cfg.initial_balance,
            len(self.micro_engine.candles),
            self.deep_engine.cfg.initial_balance,
            len(self.deep_engine.candles),
        )

        micro_result = self.micro_engine.run()
        deep_result = self.deep_engine.run()

        # Combiner les trades
        all_trades = []
        for t in micro_result.trades:
            all_trades.append(GridTrade(
                symbol=f"[M]{t.symbol}",
                n_levels=t.n_levels,
                initial_entry=t.initial_entry,
                pmp=t.pmp,
                total_size=t.total_size,
                total_cost=t.total_cost,
                total_proceeds=t.total_proceeds,
                entry_time=t.entry_time,
                exit_time=t.exit_time,
                pnl_usd=t.pnl_usd,
                pnl_pct=t.pnl_pct,
                exit_reason=t.exit_reason,
                tp1_hit=t.tp1_hit,
                tp2_hit=t.tp2_hit,
                tp3_hit=t.tp3_hit,
                hold_bars=t.hold_bars,
                max_unrealized_dd=t.max_unrealized_dd,
            ))
        for t in deep_result.trades:
            all_trades.append(GridTrade(
                symbol=f"[D]{t.symbol}",
                n_levels=t.n_levels,
                initial_entry=t.initial_entry,
                pmp=t.pmp,
                total_size=t.total_size,
                total_cost=t.total_cost,
                total_proceeds=t.total_proceeds,
                entry_time=t.entry_time,
                exit_time=t.exit_time,
                pnl_usd=t.pnl_usd,
                pnl_pct=t.pnl_pct,
                exit_reason=t.exit_reason,
                tp1_hit=t.tp1_hit,
                tp2_hit=t.tp2_hit,
                tp3_hit=t.tp3_hit,
                hold_bars=t.hold_bars,
                max_unrealized_dd=t.max_unrealized_dd,
            ))

        all_trades.sort(key=lambda t: t.entry_time)

        # Combiner les equity curves
        combined_eq = self._merge_equity_curves(
            micro_result.equity_curve, deep_result.equity_curve, self.initial_balance,
        )

        # Stop mensuel (-6%)
        monthly_stops = self._count_monthly_stops(all_trades)

        final_eq = micro_result.final_equity + deep_result.final_equity

        start_date = min(micro_result.start_date, deep_result.start_date)
        end_date = max(micro_result.end_date, deep_result.end_date)

        return DualGridResult(
            micro_result=micro_result,
            deep_result=deep_result,
            combined_trades=all_trades,
            combined_equity=combined_eq,
            initial_balance=self.initial_balance,
            final_equity=final_eq,
            start_date=start_date,
            end_date=end_date,
            pairs_micro=micro_result.pairs,
            pairs_deep=deep_result.pairs,
            monthly_stop_events=monthly_stops,
        )

    def _merge_equity_curves(
        self,
        eq_micro: list[tuple[int, float]],
        eq_deep: list[tuple[int, float]],
        initial: float,
    ) -> list[tuple[int, float]]:
        """Merge les deux equity curves en une seule courbe combinée."""
        micro_map: dict[int, float] = {ts: eq for ts, eq in eq_micro}
        deep_map: dict[int, float] = {ts: eq for ts, eq in eq_deep}

        all_ts = sorted(set(micro_map.keys()) | set(deep_map.keys()))
        if not all_ts:
            return [(0, initial)]

        last_micro = initial * self.micro_pct
        last_deep = initial * self.deep_pct
        combined: list[tuple[int, float]] = []

        for ts in all_ts:
            if ts in micro_map:
                last_micro = micro_map[ts]
            if ts in deep_map:
                last_deep = deep_map[ts]
            combined.append((ts, last_micro + last_deep))

        return combined

    def _count_monthly_stops(self, trades: list[GridTrade]) -> int:
        """Compte combien de mois auraient déclenché le stop mensuel -6%."""
        monthly_pnl: dict[str, float] = defaultdict(float)
        for t in trades:
            month = datetime.fromtimestamp(t.exit_time / 1000, tz=timezone.utc).strftime("%Y-%m")
            monthly_pnl[month] += t.pnl_usd

        stops = 0
        for month, pnl in monthly_pnl.items():
            if pnl / self.initial_balance <= -self.monthly_stop_pct:
                stops += 1
                logger.info("🛑 Stop mensuel %s : $%.2f (%.1f%%)", month, pnl, pnl / self.initial_balance * 100)
        return stops


def compute_dual_metrics(result: DualGridResult) -> dict:
    """KPIs combinés pour le dual-mode."""
    trades = result.combined_trades
    eq = result.combined_equity
    init = result.initial_balance
    final = result.final_equity
    days = max((result.end_date - result.start_date).days, 1)
    years = days / 365.25

    total_return = (final - init) / init if init > 0 else 0
    cagr = (final / init) ** (1 / years) - 1 if final > 0 and years > 0 else 0
    monthly_return = (1 + total_return) ** (30.4375 / days) - 1 if days > 0 else 0

    # Drawdown
    peak = init
    max_dd = 0.0
    for _, equity in eq:
        peak = max(peak, equity)
        dd = (equity - peak) / peak if peak > 0 else 0
        max_dd = min(max_dd, dd)

    # Sharpe / Sortino
    returns: list[float] = []
    for i in range(1, len(eq)):
        prev_eq = eq[i - 1][1]
        if prev_eq > 0:
            returns.append((eq[i][1] - prev_eq) / prev_eq)

    sharpe = _sharpe(returns)
    sortino = _sortino(returns)

    n = len(trades)
    micro_trades = [t for t in trades if t.symbol.startswith("[M]")]
    deep_trades = [t for t in trades if t.symbol.startswith("[D]")]

    if n > 0:
        wins = [t for t in trades if t.pnl_usd > 0]
        losses = [t for t in trades if t.pnl_usd <= 0]
        win_rate = len(wins) / n
        gross_profit = sum(t.pnl_usd for t in wins) or 0
        gross_loss = abs(sum(t.pnl_usd for t in losses)) or 1e-9
        profit_factor = gross_profit / gross_loss
        avg_pnl_usd = sum(t.pnl_usd for t in trades) / n
        avg_pnl_pct = sum(t.pnl_pct for t in trades) / n
        trades_per_day = n / days
        best = max(trades, key=lambda t: t.pnl_usd)
        worst = min(trades, key=lambda t: t.pnl_usd)
    else:
        win_rate = profit_factor = avg_pnl_usd = avg_pnl_pct = trades_per_day = 0
        best = worst = None

    return {
        "total_return": total_return,
        "cagr": cagr,
        "monthly_return": monthly_return,
        "max_drawdown": max_dd,
        "sharpe": sharpe,
        "sortino": sortino,
        "n_trades": n,
        "n_micro": len(micro_trades),
        "n_deep": len(deep_trades),
        "micro_pnl": sum(t.pnl_usd for t in micro_trades),
        "deep_pnl": sum(t.pnl_usd for t in deep_trades),
        "trades_per_day": trades_per_day,
        "win_rate": win_rate,
        "profit_factor": profit_factor,
        "avg_pnl_usd": avg_pnl_usd,
        "avg_pnl_pct": avg_pnl_pct,
        "best_trade": best,
        "worst_trade": worst,
        "days": days,
        "years": years,
        "final_equity": final,
        "monthly_stop_events": result.monthly_stop_events,
    }
