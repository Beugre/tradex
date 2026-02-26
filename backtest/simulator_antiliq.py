"""
Moteur de backtest pour la stratégie Antiliq.

Simule le trading contrarian sur cascades de liquidation :
  1. Détecte les dumps anormaux (>X% en 5 min + volume élevé)
  2. BUY au marché (long) pour jouer le rebond
  3. Sortie : TP (retrace 50%), SL (extension 50%), ou timeout (30 min)

Itère bar par bar sur des bougies 1m pour reproduire fidèlement
le comportement du bot en production.
"""

from __future__ import annotations

import logging
import math
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

from src.core.models import Candle
from src.core.flow_detector import detect_abnormal_flow, FlowSignal

logger = logging.getLogger(__name__)


# ── Configuration ──────────────────────────────────────────────────────────────


@dataclass
class AntiliqConfig:
    """Paramètres du backtest antiliq — miroir du .env de production."""

    initial_balance: float = 1000.0
    risk_percent: float = 0.02          # 2% risque par trade

    # Détection
    move_window: int = 5                # Fenêtre de détection (minutes)
    move_threshold_pct: float = 0.03    # 3% minimum de dump
    volume_multiplier: float = 1.5      # Volume 1.5× la moyenne 1h
    vol_avg_window: int = 60            # Baseline volume (60 bougies = 1h)

    # Trade
    tp_retrace_pct: float = 0.5         # TP à 50% de retrace
    sl_extension_pct: float = 0.5       # SL à 50% d'extension
    timeout_minutes: int = 30           # Durée max d'un trade (minutes)
    cooldown_minutes: int = 60          # Cooldown par paire après un trade

    # Trailing SL
    trailing_sl: bool = True             # Activer le trailing SL
    trailing_activation_pct: float = 0.3 # Activer après 30% de retrace vers TP
    trailing_step_pct: float = 0.5       # Le SL remonte de 50% de la distance parcourue

    # Contraintes
    max_positions: int = 2              # Max positions simultanées
    max_position_pct: float = 0.40      # Max 40% du capital par position

    # Frais
    fee_pct: float = 0.001              # 0.1% taker fee (par côté)
    slippage_pct: float = 0.0015        # 0.15% slippage adverse (spread en cascade)


# ── Structures de résultat ─────────────────────────────────────────────────────


@dataclass
class AntiliqPosition:
    """Position ouverte en cours de simulation."""

    symbol: str
    entry_price: float          # Prix d'entrée effectif (avec slippage)
    tp_price: float
    sl_price: float             # SL initial (fixe)
    size: float                 # Quantité en base asset
    cost: float                 # Coût total (size × entry + fee)
    entry_time: int             # Timestamp ms
    timeout_time: int           # Timestamp ms — clôture forcée
    signal: FlowSignal          # Signal d'origine
    peak_price: float = 0.0    # Prix le plus haut depuis l'entrée (pour trailing)
    current_sl: float = 0.0    # SL dynamique (trailing), initialisé = sl_price


@dataclass
class AntiliqTrade:
    """Trade terminé (ouverture → fermeture)."""

    symbol: str
    entry_price: float
    exit_price: float
    size: float
    entry_time: int
    exit_time: int
    pnl_usd: float
    pnl_pct: float
    exit_reason: str            # TP, SL, TIMEOUT, END
    move_pct: float             # Amplitude du dump d'origine
    volume_ratio: float         # Ratio volume du signal
    hold_minutes: int           # Durée du trade en minutes


@dataclass
class AntiliqResult:
    """Résultat complet d'un backtest antiliq."""

    trades: list[AntiliqTrade]
    equity_curve: list[tuple[int, float]]   # (timestamp_ms, equity)
    initial_balance: float
    final_equity: float
    n_signals: int              # Signaux détectés (y compris ignorés)
    n_traded: int               # Signaux ayant donné lieu à un trade
    config: AntiliqConfig
    start_date: datetime
    end_date: datetime
    pairs: list[str]


# ── Moteur de simulation ──────────────────────────────────────────────────────


class AntiliqEngine:
    """Simule la stratégie antiliq bar par bar sur klines 1m."""

    def __init__(
        self,
        candles_by_symbol: dict[str, list[Candle]],
        config: AntiliqConfig,
    ) -> None:
        self.candles = candles_by_symbol
        self.cfg = config
        self.cash = config.initial_balance
        self.positions: dict[str, AntiliqPosition] = {}
        self.trades: list[AntiliqTrade] = []
        self.equity_curve: list[tuple[int, float]] = []
        self.cooldowns: dict[str, int] = {}     # symbol → cooldown_until (ts ms)
        self.n_signals = 0
        self.n_traded = 0

    def run(self) -> AntiliqResult:
        """Exécute le backtest et retourne les résultats."""
        symbols = sorted(self.candles.keys())
        if not symbols:
            raise ValueError("Aucune donnée de bougies fournie")

        min_len = min(len(self.candles[s]) for s in symbols)
        min_lookback = self.cfg.vol_avg_window + self.cfg.move_window + 2

        if min_len <= min_lookback:
            raise ValueError(
                f"Pas assez de bougies ({min_len}) — besoin d'au moins {min_lookback}"
            )

        logger.info(
            "🔥 Antiliq Engine — %d paires, %d bougies 1m, capital $%.0f",
            len(symbols), min_len, self.cfg.initial_balance,
        )

        # ── Boucle principale : chaque barre 1m ───────────────────────────
        for i in range(min_lookback, min_len):
            ts = self.candles[symbols[0]][i].timestamp

            # 1. Vérifier les positions ouvertes (sorties)
            for sym in list(self.positions.keys()):
                if sym in self.candles and i < len(self.candles[sym]):
                    candle = self.candles[sym][i]
                    self._check_exit(sym, candle)

            # 2. Chercher de nouveaux signaux (entrées)
            if len(self.positions) < self.cfg.max_positions:
                for sym in symbols:
                    if sym in self.positions:
                        continue
                    if len(self.positions) >= self.cfg.max_positions:
                        break
                    if i >= len(self.candles[sym]):
                        continue

                    # Cooldown actif ?
                    candle_ts = self.candles[sym][i].timestamp
                    if sym in self.cooldowns and candle_ts < self.cooldowns[sym]:
                        continue

                    # Fenêtre de détection
                    lookback_start = max(0, i - self.cfg.vol_avg_window - self.cfg.move_window)
                    recent = self.candles[sym][lookback_start : i + 1]

                    signal = detect_abnormal_flow(
                        recent,
                        symbol=sym,
                        move_window=self.cfg.move_window,
                        move_threshold_pct=self.cfg.move_threshold_pct,
                        volume_multiplier=self.cfg.volume_multiplier,
                        vol_avg_window=self.cfg.vol_avg_window,
                        tp_retrace_pct=self.cfg.tp_retrace_pct,
                        sl_extension_pct=self.cfg.sl_extension_pct,
                    )
                    if signal:
                        self.n_signals += 1
                        self._open_position(signal)

            # 3. Enregistrer l'equity (toutes les 60 barres = 1h)
            if i % 60 == 0:
                eq = self._compute_equity(i)
                self.equity_curve.append((ts, eq))

        # ── Clôturer les positions restantes ───────────────────────────────
        for sym in list(self.positions.keys()):
            last_idx = len(self.candles[sym]) - 1
            last_candle = self.candles[sym][last_idx]
            self._close_position(sym, last_candle.close, last_candle.timestamp, "END")

        # Equity finale
        final_eq = self.cash
        if self.equity_curve:
            self.equity_curve.append((self.equity_curve[-1][0], final_eq))

        # Dates
        first_ts = self.candles[symbols[0]][min_lookback].timestamp
        last_ts = self.candles[symbols[0]][min_len - 1].timestamp
        start_dt = datetime.fromtimestamp(first_ts / 1000, tz=timezone.utc)
        end_dt = datetime.fromtimestamp(last_ts / 1000, tz=timezone.utc)

        logger.info(
            "✅ Terminé — %d signaux détectés, %d trades exécutés, equity $%.2f → $%.2f",
            self.n_signals, self.n_traded, self.cfg.initial_balance, final_eq,
        )

        return AntiliqResult(
            trades=self.trades,
            equity_curve=self.equity_curve,
            initial_balance=self.cfg.initial_balance,
            final_equity=final_eq,
            n_signals=self.n_signals,
            n_traded=self.n_traded,
            config=self.cfg,
            start_date=start_dt,
            end_date=end_dt,
            pairs=symbols,
        )

    # ── Ouverture de position ──────────────────────────────────────────────

    def _open_position(self, signal: FlowSignal) -> None:
        """Ouvre une position BUY après détection d'un dump."""
        # Slippage adverse (on achète plus cher en pleine cascade)
        entry = signal.entry_price * (1 + self.cfg.slippage_pct)

        # Sizing basé sur le risque
        sl_distance = abs(entry - signal.sl_price)
        if sl_distance <= 0:
            return

        equity = self._compute_equity_fast()
        risk_amount = equity * self.cfg.risk_percent
        size = risk_amount / sl_distance

        # Contrainte : max % du capital par position
        max_cost = equity * self.cfg.max_position_pct
        cost = size * entry
        if cost > max_cost:
            size = max_cost / entry
            cost = size * entry

        # Vérifier qu'on a le cash
        entry_fee = cost * self.cfg.fee_pct
        total_cost = cost + entry_fee
        if total_cost > self.cash * 0.95:
            size = (self.cash * 0.95) / (entry * (1 + self.cfg.fee_pct))
            cost = size * entry
            entry_fee = cost * self.cfg.fee_pct
            total_cost = cost + entry_fee

        # Min notional
        if cost < 10:
            return

        # Déduire le coût du cash
        self.cash -= total_cost

        timeout_ts = signal.timestamp + self.cfg.timeout_minutes * 60 * 1000

        self.positions[signal.symbol] = AntiliqPosition(
            symbol=signal.symbol,
            entry_price=entry,
            tp_price=signal.tp_price,
            sl_price=signal.sl_price,
            size=size,
            cost=total_cost,
            entry_time=signal.timestamp,
            timeout_time=timeout_ts,
            signal=signal,
            peak_price=entry,
            current_sl=signal.sl_price,
        )
        self.n_traded += 1

        logger.debug(
            "📥 OPEN %s @ %.4f | TP=%.4f SL=%.4f | size=%.6f | dump=%.2f%%",
            signal.symbol, entry, signal.tp_price, signal.sl_price,
            size, signal.move_pct * 100,
        )

    # ── Vérification des sorties ───────────────────────────────────────────

    def _check_exit(self, symbol: str, candle: Candle) -> None:
        """Vérifie si une position doit être clôturée sur cette bougie."""
        pos = self.positions.get(symbol)
        if not pos:
            return

        # ── Trailing SL : remonter le SL quand le prix progresse ──────────
        if self.cfg.trailing_sl:
            # Mettre à jour le peak (plus haut atteint)
            if candle.high > pos.peak_price:
                pos.peak_price = candle.high

            # Activation : le prix a retracé au moins X% vers le TP
            tp_dist = pos.tp_price - pos.entry_price   # distance entry → TP
            retrace = pos.peak_price - pos.entry_price  # progression actuelle
            activation_threshold = tp_dist * self.cfg.trailing_activation_pct

            if retrace >= activation_threshold and tp_dist > 0:
                # Nouveau SL = entry + (peak - entry) × trailing_step_pct
                new_sl = pos.entry_price + retrace * self.cfg.trailing_step_pct
                if new_sl > pos.current_sl:
                    pos.current_sl = new_sl
                    logger.debug(
                        "🔄 TRAIL %s : peak=%.4f → SL remonté à %.4f",
                        symbol, pos.peak_price, pos.current_sl,
                    )

        # SL effectif (trailing ou fixe)
        effective_sl = pos.current_sl if self.cfg.trailing_sl else pos.sl_price

        # SL en premier (pessimiste : on suppose SL touché avant TP)
        if candle.low <= effective_sl:
            # Déterminer le motif : TRAIL_SL si le SL a été remonté au-dessus de l'initial
            reason = "TRAIL_SL" if effective_sl > pos.sl_price else "SL"
            self._close_position(symbol, effective_sl, candle.timestamp, reason)
            return

        # TP
        if candle.high >= pos.tp_price:
            self._close_position(symbol, pos.tp_price, candle.timestamp, "TP")
            return

        # Timeout
        if candle.timestamp >= pos.timeout_time:
            self._close_position(symbol, candle.close, candle.timestamp, "TIMEOUT")
            return

    # ── Clôture de position ────────────────────────────────────────────────

    def _close_position(
        self, symbol: str, exit_price: float, exit_time: int, reason: str
    ) -> None:
        """Clôture une position et enregistre le trade."""
        pos = self.positions.pop(symbol, None)
        if not pos:
            return

        # Slippage adverse sur la vente (on vend moins cher)
        actual_exit = exit_price * (1 - self.cfg.slippage_pct)

        # Proceeds de la vente
        proceeds = pos.size * actual_exit
        exit_fee = proceeds * self.cfg.fee_pct
        net_proceeds = proceeds - exit_fee

        # Rendre le cash
        self.cash += net_proceeds

        # PnL net (tout inclus : slippage + fees des deux côtés)
        pnl_usd = net_proceeds - pos.cost
        pnl_pct = pnl_usd / pos.cost if pos.cost > 0 else 0

        # Durée du trade
        hold_ms = exit_time - pos.entry_time
        hold_minutes = max(1, hold_ms // 60_000)

        # Cooldown
        self.cooldowns[symbol] = exit_time + self.cfg.cooldown_minutes * 60_000

        trade = AntiliqTrade(
            symbol=symbol,
            entry_price=pos.entry_price,
            exit_price=actual_exit,
            size=pos.size,
            entry_time=pos.entry_time,
            exit_time=exit_time,
            pnl_usd=pnl_usd,
            pnl_pct=pnl_pct,
            exit_reason=reason,
            move_pct=pos.signal.move_pct,
            volume_ratio=pos.signal.volume_ratio,
            hold_minutes=hold_minutes,
        )
        self.trades.append(trade)

        emoji = "✅" if pnl_usd >= 0 else "❌"
        logger.debug(
            "%s CLOSE %s @ %.4f (%s) | PnL $%.2f (%.2f%%) | hold %dmin | dump %.1f%%",
            emoji, symbol, actual_exit, reason, pnl_usd, pnl_pct * 100,
            hold_minutes, pos.signal.move_pct * 100,
        )

    # ── Equity ─────────────────────────────────────────────────────────────

    def _compute_equity(self, current_idx: int) -> float:
        """Equity = cash + valeur mark-to-market des positions ouvertes."""
        eq = self.cash
        for sym, pos in self.positions.items():
            if current_idx < len(self.candles[sym]):
                current_price = self.candles[sym][current_idx].close
                eq += pos.size * current_price
        return eq

    def _compute_equity_fast(self) -> float:
        """Equity approximative (cash + coût des positions)."""
        eq = self.cash
        for pos in self.positions.values():
            eq += pos.cost  # Approximation : cost ≈ market value
        return eq


# ── Métriques ──────────────────────────────────────────────────────────────────

# Bougies 1m → 525,960 barres par an
_BARS_PER_YEAR_1M = 60 * 24 * 365.25


def compute_antiliq_metrics(result: AntiliqResult) -> dict:
    """Calcule les KPIs du backtest antiliq."""
    trades = result.trades
    eq = result.equity_curve
    init = result.initial_balance
    final = result.final_equity
    days = max((result.end_date - result.start_date).days, 1)
    years = days / 365.25

    # ── Rendement ──
    total_return = (final - init) / init if init > 0 else 0
    cagr = (final / init) ** (1 / years) - 1 if final > 0 and years > 0 else 0

    # ── Drawdown ──
    peak = init
    max_dd = 0.0
    dd_curve: list[float] = []
    for _, equity in eq:
        peak = max(peak, equity)
        dd = (equity - peak) / peak if peak > 0 else 0
        dd_curve.append(dd)
        max_dd = min(max_dd, dd)

    # ── Sharpe (sur rendements horaires) ──
    returns: list[float] = []
    for i in range(1, len(eq)):
        prev_eq = eq[i - 1][1]
        if prev_eq > 0:
            returns.append((eq[i][1] - prev_eq) / prev_eq)

    sharpe = _sharpe(returns, periods_per_year=24 * 365.25)  # Horaire
    sortino = _sortino(returns, periods_per_year=24 * 365.25)

    # ── Trades ──
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
        avg_hold = sum(t.hold_minutes for t in trades) / n
        best = max(trades, key=lambda t: t.pnl_usd)
        worst = min(trades, key=lambda t: t.pnl_usd)
    else:
        win_rate = profit_factor = avg_pnl_usd = avg_pnl_pct = avg_hold = 0
        best = worst = None

    # ── Par paire ──
    by_pair = _group_trades(trades, lambda t: t.symbol)

    # ── Par exit reason ──
    by_exit = _group_trades(trades, lambda t: t.exit_reason)

    # ── Par tranche de dump ──
    def dump_bucket(t: AntiliqTrade) -> str:
        pct = abs(t.move_pct) * 100
        if pct < 3:
            return "<3%"
        elif pct < 4:
            return "3-4%"
        elif pct < 5:
            return "4-5%"
        else:
            return ">5%"

    by_dump = _group_trades(trades, dump_bucket)

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
        "avg_hold_min": avg_hold,
        "best_trade": best,
        "worst_trade": worst,
        "by_pair": by_pair,
        "by_exit": by_exit,
        "by_dump_size": by_dump,
        "dd_curve": dd_curve,
        "years": years,
        "days": days,
        "final_equity": final,
    }


# ── Helpers métriques ──────────────────────────────────────────────────────────


def _sharpe(returns: list[float], periods_per_year: float = 8760) -> float:
    if len(returns) < 2:
        return 0.0
    mu = sum(returns) / len(returns)
    var = sum((r - mu) ** 2 for r in returns) / (len(returns) - 1)
    std = math.sqrt(var) if var > 0 else 1e-9
    return (mu / std) * math.sqrt(periods_per_year)


def _sortino(returns: list[float], periods_per_year: float = 8760) -> float:
    if len(returns) < 2:
        return 0.0
    mu = sum(returns) / len(returns)
    neg = [r for r in returns if r < 0]
    if not neg:
        return 99.0
    down_var = sum(r ** 2 for r in neg) / len(neg)
    down_std = math.sqrt(down_var) if down_var > 0 else 1e-9
    return (mu / down_std) * math.sqrt(periods_per_year)


def _group_trades(
    trades: list[AntiliqTrade], key_fn
) -> dict[str, dict]:
    groups: dict[str, list[AntiliqTrade]] = defaultdict(list)
    for t in trades:
        groups[key_fn(t)].append(t)
    out: dict[str, dict] = {}
    for k, tlist in sorted(groups.items()):
        n = len(tlist)
        wins = sum(1 for t in tlist if t.pnl_usd > 0)
        pnl = sum(t.pnl_usd for t in tlist)
        gp = sum(t.pnl_usd for t in tlist if t.pnl_usd > 0) or 0
        gl = abs(sum(t.pnl_usd for t in tlist if t.pnl_usd <= 0)) or 1e-9
        avg_hold = sum(t.hold_minutes for t in tlist) / n if n else 0
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
