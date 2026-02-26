"""
Moteur de backtest Breakout Volatility Expansion.

Simule la stratégie bar-par-bar sur données H4 historiques.
Trailing stop pour laisser courir les gagnants.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

from src.core.models import Candle, OrderSide
from src.core.breakout_detector import (
    BreakoutConfig,
    BreakoutDirection,
    BreakoutSignal,
    atr,
    adx as calc_adx,
    bollinger_bands,
    bb_width as calc_bb_width,
    donchian_channel,
    sma,
)

logger = logging.getLogger(__name__)


# ── Structures ─────────────────────────────────────────────────────────────────


@dataclass
class BreakoutTrade:
    """Un trade breakout terminé."""
    symbol: str
    direction: BreakoutDirection
    entry_price: float
    exit_price: float
    size: float                  # en unités de base
    entry_time: int              # timestamp ms
    exit_time: int               # timestamp ms
    pnl_usd: float
    pnl_pct: float
    exit_reason: str             # SL, TRAILING_SL, END
    sl_price: float              # SL au moment de la sortie
    adx_at_entry: float
    bb_width_at_entry: float
    volume_ratio_at_entry: float


@dataclass
class BreakoutPosition:
    """Position ouverte."""
    symbol: str
    direction: BreakoutDirection
    entry_price: float
    size: float
    entry_time: int
    sl_price: float
    trailing_sl: float           # trailing stop (ajusté au fur et à mesure)
    highest_since_entry: float   # plus haut atteint (pour trailing LONG)
    lowest_since_entry: float    # plus bas atteint (pour trailing SHORT)
    adx_at_entry: float
    bb_width_at_entry: float
    volume_ratio_at_entry: float


@dataclass
class EquityPoint:
    timestamp: int
    equity: float


@dataclass
class BreakoutResult:
    """Résultat complet du backtest breakout."""
    trades: list[BreakoutTrade]
    equity_curve: list[EquityPoint]
    initial_balance: float
    final_equity: float
    start_date: datetime
    end_date: datetime
    pairs: list[str]


@dataclass
class BreakoutSimConfig:
    """Paramètres complets du simulateur breakout."""
    # ── Stratégie (repris de BreakoutConfig) ──
    bb_period: int = 20
    bb_std: float = 2.0
    bb_width_expansion: float = 1.2
    donchian_period: int = 20
    adx_period: int = 14
    adx_threshold: float = 25.0
    vol_avg_period: int = 20
    vol_multiplier: float = 1.2
    sl_atr_mult: float = 1.5
    atr_period: int = 14
    allow_short: bool = True

    # ── Trailing Stop ──
    trailing_atr_mult: float = 2.0   # trailing stop = ATR * mult
    activate_trailing_pct: float = 0.01  # activer trailing après +1%

    # ── 1️⃣ Filtre EMA200 Daily (HTF) ──
    use_ema_filter: bool = False      # True = n'entre que si BTC > EMA200 D1
    ema_period: int = 200

    # ── 2️⃣ Filtre compression ATR (squeeze avant breakout) ──
    use_atr_squeeze: bool = False     # True = ATR doit avoir été compressé récemment
    atr_squeeze_lookback: int = 10    # cherche la compression dans les N dernières barres
    atr_squeeze_ratio: float = 0.90   # ATR min récent < 90% de ATR moy → squeeze

    # ── 3️⃣ Stop adaptatif ──
    adaptive_trailing: bool = False   # True = trailing step progressif basé sur profit
    trail_step_1_pct: float = 0.02   # à +2% de profit, trailing = entry + 0.2%
    trail_step_2_pct: float = 0.05   # à +5%, trailing = entry + 2%
    trail_lock_1_pct: float = 0.002  # verrouiller 0.2% de profit au step 1
    trail_lock_2_pct: float = 0.02   # verrouiller 2% au step 2

    # ── 4️⃣ Kill-switch mensuel ──
    use_kill_switch: bool = False     # True = stopper trading si mois > -X%
    kill_switch_pct: float = -0.10   # seuil : -10% mensuel

    # ── Money management ──
    initial_balance: float = 1000.0
    risk_percent: float = 0.02        # 2% du capital par trade (base)
    max_positions: int = 5            # positions simultanées max
    fee_pct: float = 0.00075          # 0.075% par côté (Binance spot)
    slippage_pct: float = 0.001       # 0.1% slippage

    # ── 5️⃣ Sizing dynamique (volatility-based) ──
    dynamic_sizing: bool = False       # True = ajuster risk% selon volatilité
    base_atr_pct: float = 0.03        # ATR/prix "normal" (3%) → risk_percent normal
    min_risk_percent: float = 0.005   # plancher : 0.5% si ATR très élevé
    max_risk_percent: float = 0.03    # plafond : 3% si ATR très bas

    # ── 6️⃣ Plafond risque par trade en USD ──
    risk_cap_usd: float = 0.0         # 0 = pas de plafond. Ex: 50 = max $50/trade

    # ── 7️⃣ Exposition max simultanée ──
    max_exposure_pct: float = 0.0     # 0 = pas de limite. Ex: 0.5 = max 50% du capital en positions

    # ── Cooldown ──
    cooldown_bars: int = 3            # bougies d'attente après fermeture

    # ── Compounding ──
    compound: bool = False            # sizing sur capital courant vs initial

    def to_breakout_config(self) -> BreakoutConfig:
        """Convertit en config pour le détecteur."""
        return BreakoutConfig(
            bb_period=self.bb_period,
            bb_std=self.bb_std,
            bb_width_expansion=self.bb_width_expansion,
            donchian_period=self.donchian_period,
            adx_period=self.adx_period,
            adx_threshold=self.adx_threshold,
            vol_avg_period=self.vol_avg_period,
            vol_multiplier=self.vol_multiplier,
            sl_atr_mult=self.sl_atr_mult,
            atr_period=self.atr_period,
            allow_short=self.allow_short,
        )


# ── Moteur ─────────────────────────────────────────────────────────────────────


class BreakoutEngine:
    """Simule la stratégie breakout bar-par-bar."""

    def __init__(
        self,
        candles_by_symbol: dict[str, list[Candle]],
        config: BreakoutSimConfig,
        btc_d1_candles: Optional[list[Candle]] = None,
    ) -> None:
        self.candles = candles_by_symbol
        self.cfg = config
        self.pairs = list(candles_by_symbol.keys())

        # État financier
        self.cash: float = config.initial_balance
        self.positions: list[BreakoutPosition] = []
        self.closed_trades: list[BreakoutTrade] = []
        self.equity_curve: list[EquityPoint] = []

        # Cooldown par paire : ts avant lequel on ne trade pas
        self.cooldown_until: dict[str, int] = {p: 0 for p in self.pairs}

        # ── 1️⃣ EMA200 Daily filter ──
        self._ema_mode: dict[int, bool] = {}     # ts_d1 → BTC > EMA200
        self._ema_ts_sorted: list[int] = []
        self._current_ema_bullish: bool = True    # fallback si pas de données
        if config.use_ema_filter and btc_d1_candles:
            self._build_ema_lookup(btc_d1_candles)

        # ── 4️⃣ Kill-switch mensuel ──
        self._month_start_equity: float = config.initial_balance
        self._current_month: str = ""
        self._killed_months: set[str] = set()     # mois où le kill-switch a été activé

        # Pré-calcul des indicateurs par paire
        self._indicators: dict[str, dict] = {}
        self._precompute_indicators()

    def _precompute_indicators(self) -> None:
        """Pré-calcule BB, Donchian, ADX, ATR, Volume SMA pour chaque paire."""
        cfg = self.cfg
        for sym, candles in self.candles.items():
            closes = [c.close for c in candles]
            volumes = [c.volume for c in candles]

            bb_upper, bb_mid, bb_lower = bollinger_bands(closes, cfg.bb_period, cfg.bb_std)
            bbw = calc_bb_width(bb_upper, bb_mid, bb_lower)
            bbw_avg = sma([x if x is not None else 0.0 for x in bbw], cfg.bb_period)
            dc_high, dc_low = donchian_channel(candles, cfg.donchian_period)
            adx_values = calc_adx(candles, cfg.adx_period)
            atr_values = atr(candles, cfg.atr_period)
            vol_avg = sma(volumes, cfg.vol_avg_period)

            self._indicators[sym] = {
                "bb_upper": bb_upper,
                "bb_mid": bb_mid,
                "bb_lower": bb_lower,
                "bbw": bbw,
                "bbw_avg": bbw_avg,
                "dc_high": dc_high,
                "dc_low": dc_low,
                "adx": adx_values,
                "atr": atr_values,
                "vol_avg": vol_avg,
            }

    # ── 1️⃣ EMA200 Daily ──────────────────────────────────────────────────

    def _build_ema_lookup(self, d1_candles: list[Candle]) -> None:
        """Pré-calcule l'EMA200 sur BTC D1."""
        period = self.cfg.ema_period
        if len(d1_candles) < period:
            logger.warning(
                "⚠️ Seulement %d bougies D1 (< EMA%d) — filtre EMA désactivé",
                len(d1_candles), period,
            )
            return

        closes = [c.close for c in d1_candles]
        # SMA initiale
        ema = sum(closes[:period]) / period
        k = 2.0 / (period + 1)

        for i in range(period, len(closes)):
            ema = closes[i] * k + ema * (1 - k)
            ts = d1_candles[i].timestamp
            self._ema_mode[ts] = closes[i] > ema

        self._ema_ts_sorted = sorted(self._ema_mode.keys())
        n_bull = sum(1 for v in self._ema_mode.values() if v)
        n_bear = len(self._ema_mode) - n_bull
        logger.info(
            "📊 EMA%d D1 : %d jours BULLISH, %d jours BEARISH",
            period, n_bull, n_bear,
        )

    def _is_ema_bullish(self, ts_h4: int) -> bool:
        """Vérifie si BTC est au-dessus de l'EMA200 D1 (sans lookahead)."""
        if not self._ema_mode:
            return True  # pas de filtre → toujours OK
        import bisect
        idx = bisect.bisect_right(self._ema_ts_sorted, ts_h4) - 1
        if idx >= 0:
            last_d1_ts = self._ema_ts_sorted[idx]
            return self._ema_mode[last_d1_ts]
        return True  # pas encore de données → OK

    # ── 2️⃣ Filtre ATR Squeeze ─────────────────────────────────────────────

    def _had_atr_squeeze(self, sym: str, idx: int) -> bool:
        """Vérifie si l'ATR était compressé récemment (squeeze avant expansion)."""
        ind = self._indicators.get(sym)
        if not ind:
            return False
        atr_values = ind["atr"]
        lookback = self.cfg.atr_squeeze_lookback

        # On cherche dans [idx-lookback, idx) si l'ATR était < ratio * ATR moyen
        if idx < lookback + 20:
            return True  # pas assez de données → on laisse passer

        # ATR moyen sur les 20 barres précédant la fenêtre
        atr_window = []
        for i in range(idx - lookback - 20, idx - lookback):
            if 0 <= i < len(atr_values) and atr_values[i] is not None:
                atr_window.append(atr_values[i])
        if not atr_window:
            return True
        atr_avg = sum(atr_window) / len(atr_window)
        if atr_avg <= 0:
            return True

        # Chercher si au moins une barre récente avait un ATR compressé
        for i in range(max(0, idx - lookback), idx):
            if i < len(atr_values) and atr_values[i] is not None:
                if atr_values[i] < self.cfg.atr_squeeze_ratio * atr_avg:
                    return True
        return False

    # ── 4️⃣ Kill-switch mensuel ─────────────────────────────────────────────

    def _update_kill_switch(self, ts: int, equity: float) -> None:
        """Met à jour le kill-switch mensuel."""
        dt = datetime.fromtimestamp(ts / 1000, tz=timezone.utc)
        month_key = f"{dt.year}-{dt.month:02d}"

        if month_key != self._current_month:
            # Nouveau mois → reset
            self._current_month = month_key
            self._month_start_equity = equity

        # Vérifier si le mois est déjà killed
        if month_key in self._killed_months:
            return

        # Vérifier le drawdown mensuel
        if self._month_start_equity > 0:
            monthly_return = (equity - self._month_start_equity) / self._month_start_equity
            if monthly_return <= self.cfg.kill_switch_pct:
                self._killed_months.add(month_key)
                logger.info(
                    "🛑 KILL-SWITCH %s : %.1f%% (seuil %.0f%%)",
                    month_key, monthly_return * 100, self.cfg.kill_switch_pct * 100,
                )

    def _is_month_killed(self, ts: int) -> bool:
        """Vérifie si le mois courant est en kill-switch."""
        dt = datetime.fromtimestamp(ts / 1000, tz=timezone.utc)
        return f"{dt.year}-{dt.month:02d}" in self._killed_months

    # ── Point d'entrée ─────────────────────────────────────────────────────

    def run(self) -> BreakoutResult:
        """Lance le backtest."""
        # Couper les logs verbeux
        logging.getLogger("src.core.breakout_detector").setLevel(logging.WARNING)

        timeline = self._build_timeline()
        total = len(timeline)
        logger.info(
            "📊 Breakout Backtest : %d barres H4, %d paires, capital $%.0f",
            total, len(self.pairs), self.cfg.initial_balance,
        )

        warmup = max(
            self.cfg.bb_period,
            self.cfg.donchian_period,
            2 * self.cfg.adx_period,
        ) + 5

        for i_ts, ts in enumerate(timeline):
            # ── Snapshot equity ──
            equity = self._mark_to_market(ts)
            self.equity_curve.append(EquityPoint(timestamp=ts, equity=equity))

            # ── Mise à jour EMA200 ──
            if self.cfg.use_ema_filter:
                self._current_ema_bullish = self._is_ema_bullish(ts)

            # ── Mise à jour kill-switch ──
            if self.cfg.use_kill_switch:
                self._update_kill_switch(ts, equity)

            # ── Pour chaque paire, traiter la bougie ──
            for sym in self.pairs:
                candle_idx = self._ts_to_idx(sym, ts)
                if candle_idx is None or candle_idx < warmup:
                    continue

                candle = self.candles[sym][candle_idx]

                # 1) Gérer les positions ouvertes (SL, trailing)
                self._manage_positions(sym, candle, candle_idx)

                # 2) Chercher de nouveaux signaux
                self._check_entry(sym, candle, candle_idx)

            # Progress
            if (i_ts + 1) % 2000 == 0 or i_ts == total - 1:
                pct = (i_ts + 1) / total * 100
                logger.info(
                    "  ⏳ %d/%d (%.0f%%) — Equity: $%.2f — Trades: %d — Positions: %d",
                    i_ts + 1, total, pct, equity,
                    len(self.closed_trades), len(self.positions),
                )

        # Fermer les positions restantes
        self._close_all_remaining(timeline[-1] if timeline else 0)

        final_equity = self.cash + self._unrealized_pnl(timeline[-1] if timeline else 0)
        start_dt = datetime.fromtimestamp(timeline[0] / 1000, tz=timezone.utc) if timeline else datetime.now(tz=timezone.utc)
        end_dt = datetime.fromtimestamp(timeline[-1] / 1000, tz=timezone.utc) if timeline else datetime.now(tz=timezone.utc)

        logger.info(
            "✅ Terminé : %d trades, equity $%.2f → $%.2f",
            len(self.closed_trades), self.cfg.initial_balance, final_equity,
        )

        return BreakoutResult(
            trades=self.closed_trades,
            equity_curve=self.equity_curve,
            initial_balance=self.cfg.initial_balance,
            final_equity=final_equity,
            start_date=start_dt,
            end_date=end_dt,
            pairs=self.pairs,
        )

    # ── Construction de la timeline ────────────────────────────────────────

    def _build_timeline(self) -> list[int]:
        """Timeline unifiée de tous les timestamps H4."""
        ts_set: set[int] = set()
        for candles in self.candles.values():
            for c in candles:
                ts_set.add(c.timestamp)
        return sorted(ts_set)

    def _ts_to_idx(self, sym: str, ts: int) -> Optional[int]:
        """Trouve l'index de la bougie pour un symbole et timestamp."""
        candles = self.candles[sym]
        # Recherche binaire
        lo, hi = 0, len(candles) - 1
        while lo <= hi:
            mid = (lo + hi) // 2
            if candles[mid].timestamp == ts:
                return mid
            elif candles[mid].timestamp < ts:
                lo = mid + 1
            else:
                hi = mid - 1
        return None

    # ── Gestion des positions ──────────────────────────────────────────────

    def _manage_positions(self, sym: str, candle: Candle, idx: int) -> None:
        """Vérifie SL / trailing stop pour les positions de cette paire."""
        to_close: list[tuple[BreakoutPosition, float, str]] = []

        for pos in self.positions:
            if pos.symbol != sym:
                continue

            if pos.direction == BreakoutDirection.LONG:
                # Mise à jour du plus haut
                if candle.high > pos.highest_since_entry:
                    pos.highest_since_entry = candle.high

                # SL touché ?
                if candle.low <= pos.sl_price:
                    reason = "TRAIL_SL" if pos.sl_price > pos.trailing_sl else "SL"
                    to_close.append((pos, pos.sl_price, reason))
                    continue

                move_pct = (pos.highest_since_entry - pos.entry_price) / pos.entry_price

                # ── 3️⃣ Stop adaptatif par paliers ──
                if self.cfg.adaptive_trailing:
                    new_sl = pos.sl_price
                    if move_pct >= self.cfg.trail_step_2_pct:
                        # Palier 2 : verrouiller trail_lock_2
                        lock = pos.entry_price * (1 + self.cfg.trail_lock_2_pct)
                        new_sl = max(new_sl, lock)
                    elif move_pct >= self.cfg.trail_step_1_pct:
                        # Palier 1 : verrouiller trail_lock_1 (breakeven+)
                        lock = pos.entry_price * (1 + self.cfg.trail_lock_1_pct)
                        new_sl = max(new_sl, lock)

                    # + trailing ATR classique par-dessus
                    if move_pct >= self.cfg.activate_trailing_pct:
                        ind = self._indicators.get(sym)
                        current_atr = ind["atr"][idx] if ind and ind["atr"][idx] else 0
                        atr_trail = pos.highest_since_entry - self.cfg.trailing_atr_mult * current_atr
                        new_sl = max(new_sl, atr_trail)

                    if new_sl > pos.sl_price:
                        pos.sl_price = new_sl
                else:
                    # Trailing classique
                    if move_pct >= self.cfg.activate_trailing_pct:
                        ind = self._indicators.get(sym)
                        current_atr = ind["atr"][idx] if ind and ind["atr"][idx] else 0
                        new_trail = pos.highest_since_entry - self.cfg.trailing_atr_mult * current_atr
                        if new_trail > pos.trailing_sl:
                            pos.trailing_sl = new_trail
                            if pos.trailing_sl > pos.sl_price:
                                pos.sl_price = pos.trailing_sl

            elif pos.direction == BreakoutDirection.SHORT:
                # Mise à jour du plus bas
                if candle.low < pos.lowest_since_entry:
                    pos.lowest_since_entry = candle.low

                # SL touché ?
                if candle.high >= pos.sl_price:
                    reason = "TRAIL_SL" if pos.sl_price < pos.trailing_sl else "SL"
                    to_close.append((pos, pos.sl_price, reason))
                    continue

                move_pct = (pos.entry_price - pos.lowest_since_entry) / pos.entry_price

                # ── 3️⃣ Stop adaptatif par paliers ──
                if self.cfg.adaptive_trailing:
                    new_sl = pos.sl_price
                    if move_pct >= self.cfg.trail_step_2_pct:
                        lock = pos.entry_price * (1 - self.cfg.trail_lock_2_pct)
                        new_sl = min(new_sl, lock)
                    elif move_pct >= self.cfg.trail_step_1_pct:
                        lock = pos.entry_price * (1 - self.cfg.trail_lock_1_pct)
                        new_sl = min(new_sl, lock)

                    if move_pct >= self.cfg.activate_trailing_pct:
                        ind = self._indicators.get(sym)
                        current_atr = ind["atr"][idx] if ind and ind["atr"][idx] else 0
                        atr_trail = pos.lowest_since_entry + self.cfg.trailing_atr_mult * current_atr
                        new_sl = min(new_sl, atr_trail)

                    if new_sl < pos.sl_price:
                        pos.sl_price = new_sl
                else:
                    if move_pct >= self.cfg.activate_trailing_pct:
                        ind = self._indicators.get(sym)
                        current_atr = ind["atr"][idx] if ind and ind["atr"][idx] else 0
                        new_trail = pos.lowest_since_entry + self.cfg.trailing_atr_mult * current_atr
                        if new_trail < pos.trailing_sl:
                            pos.trailing_sl = new_trail
                            if pos.trailing_sl < pos.sl_price:
                                pos.sl_price = pos.trailing_sl

        # Fermer les positions marquées
        for pos, exit_price, reason in to_close:
            self._close_position(pos, exit_price, candle.timestamp, reason)

    # ── 5️⃣6️⃣7️⃣ Sizing dynamique + plafond + exposition ──────────────────

    def _current_exposure(self, ts: int) -> float:
        """Notionnel total des positions ouvertes (mark-to-market)."""
        total = 0.0
        for pos in self.positions:
            candles = self.candles.get(pos.symbol, [])
            if not candles:
                continue
            last_price = candles[-1].close
            for c in candles:
                if c.timestamp <= ts:
                    last_price = c.close
                else:
                    break
            total += pos.size * last_price
        return total

    def _compute_dynamic_risk(self, sym: str, idx: int) -> float:
        """Calcule le risk% ajusté à la volatilité de la paire.

        Logique : si ATR/prix est élevé (paire très volatile),
        on réduit le risk% pour garder la taille de position raisonnable.
        Inversement, faible vol → on augmente.
        """
        if not self.cfg.dynamic_sizing:
            return self.cfg.risk_percent

        ind = self._indicators.get(sym)
        if not ind or ind["atr"][idx] is None:
            return self.cfg.risk_percent

        candle = self.candles[sym][idx]
        if candle.close <= 0:
            return self.cfg.risk_percent

        atr_pct = ind["atr"][idx] / candle.close  # ATR normalisé
        base = self.cfg.base_atr_pct

        # ratio = base_atr / actual_atr → >1 si vol basse, <1 si vol haute
        if atr_pct > 0:
            ratio = base / atr_pct
        else:
            ratio = 1.0

        adjusted = self.cfg.risk_percent * ratio
        return max(self.cfg.min_risk_percent, min(adjusted, self.cfg.max_risk_percent))

    def _check_entry(self, sym: str, candle: Candle, idx: int) -> None:
        """Vérifie si un nouveau signal breakout est émis à cette bougie."""
        # Déjà en position sur cette paire ?
        if any(p.symbol == sym for p in self.positions):
            return

        # Cooldown
        if candle.timestamp < self.cooldown_until.get(sym, 0):
            return

        # Max positions atteint ?
        if len(self.positions) >= self.cfg.max_positions:
            return

        # ── 1️⃣ Filtre EMA200 HTF ──
        if self.cfg.use_ema_filter and not self._current_ema_bullish:
            return

        # ── 4️⃣ Kill-switch mensuel ──
        if self.cfg.use_kill_switch and self._is_month_killed(candle.timestamp):
            return

        ind = self._indicators.get(sym)
        if not ind:
            return

        # Vérifier les 4 filtres
        bbw_val = ind["bbw"][idx]
        bbw_avg_val = ind["bbw_avg"][idx]
        dc_high = ind["dc_high"][idx]
        dc_low = ind["dc_low"][idx]
        adx_val = ind["adx"][idx]
        vol_avg_val = ind["vol_avg"][idx]
        atr_val = ind["atr"][idx]

        if any(v is None for v in (bbw_val, bbw_avg_val, dc_high, dc_low, adx_val, vol_avg_val, atr_val)):
            return

        # Filtres
        bbw_expanding = bbw_val > self.cfg.bb_width_expansion * bbw_avg_val
        adx_strong = adx_val > self.cfg.adx_threshold
        vol_above = candle.volume > self.cfg.vol_multiplier * vol_avg_val if vol_avg_val > 0 else False

        if not (bbw_expanding and adx_strong and vol_above):
            return

        # ── 2️⃣ Filtre compression ATR (squeeze avant expansion) ──
        if self.cfg.use_atr_squeeze and not self._had_atr_squeeze(sym, idx):
            return

        vol_ratio = candle.volume / vol_avg_val if vol_avg_val > 0 else 0

        direction: Optional[BreakoutDirection] = None
        sl_price: float = 0.0

        # LONG breakout
        if candle.close > dc_high:
            direction = BreakoutDirection.LONG
            sl_price = candle.close - self.cfg.sl_atr_mult * atr_val

        # SHORT breakout
        elif self.cfg.allow_short and candle.close < dc_low:
            direction = BreakoutDirection.SHORT
            sl_price = candle.close + self.cfg.sl_atr_mult * atr_val

        if direction is None:
            return

        # ── Sizing ──
        balance = self.cash if not self.cfg.compound else self._current_equity(candle.timestamp)

        # 5️⃣ Risk% dynamique (volatility-based)
        risk_pct = self._compute_dynamic_risk(sym, idx)
        risk_amount = balance * risk_pct

        # 6️⃣ Plafond risque en USD
        if self.cfg.risk_cap_usd > 0:
            risk_amount = min(risk_amount, self.cfg.risk_cap_usd)

        sl_distance = abs(candle.close - sl_price)
        if sl_distance <= 0:
            return

        size = risk_amount / sl_distance
        entry_price = candle.close

        # Appliquer slippage
        if direction == BreakoutDirection.LONG:
            entry_price *= (1 + self.cfg.slippage_pct)
        else:
            entry_price *= (1 - self.cfg.slippage_pct)

        # Frais d'entrée
        cost = size * entry_price * self.cfg.fee_pct

        # Vérifier qu'on a assez de cash
        notional = size * entry_price
        if notional + cost > self.cash:
            return

        # 7️⃣ Exposition max simultanée
        if self.cfg.max_exposure_pct > 0:
            current_expo = self._current_exposure(candle.timestamp)
            equity = self._current_equity(candle.timestamp)
            max_expo = equity * self.cfg.max_exposure_pct
            if current_expo + notional > max_expo:
                # Réduire la taille pour respecter le plafond
                remaining = max_expo - current_expo
                if remaining <= 0:
                    return
                size = remaining / entry_price
                notional = size * entry_price
                cost = size * entry_price * self.cfg.fee_pct
                if notional + cost > self.cash:
                    return

        self.cash -= cost  # frais d'entrée

        pos = BreakoutPosition(
            symbol=sym,
            direction=direction,
            entry_price=entry_price,
            size=size,
            entry_time=candle.timestamp,
            sl_price=sl_price,
            trailing_sl=sl_price,  # trailing = SL initial au départ
            highest_since_entry=candle.high,
            lowest_since_entry=candle.low,
            adx_at_entry=adx_val,
            bb_width_at_entry=bbw_val,
            volume_ratio_at_entry=vol_ratio,
        )
        self.positions.append(pos)

    def _close_position(
        self, pos: BreakoutPosition, exit_price: float, exit_ts: int, reason: str,
    ) -> None:
        """Ferme une position et enregistre le trade."""
        # Slippage à la sortie
        if pos.direction == BreakoutDirection.LONG:
            exit_price *= (1 - self.cfg.slippage_pct)
        else:
            exit_price *= (1 + self.cfg.slippage_pct)

        # PnL
        if pos.direction == BreakoutDirection.LONG:
            pnl_raw = (exit_price - pos.entry_price) * pos.size
        else:
            pnl_raw = (pos.entry_price - exit_price) * pos.size

        # Frais de sortie
        fee = pos.size * exit_price * self.cfg.fee_pct
        pnl = pnl_raw - fee

        pnl_pct = pnl / (pos.size * pos.entry_price) if pos.size * pos.entry_price > 0 else 0

        self.cash += pnl  # PnL net (frais déjà déduits)

        trade = BreakoutTrade(
            symbol=pos.symbol,
            direction=pos.direction,
            entry_price=pos.entry_price,
            exit_price=exit_price,
            size=pos.size,
            entry_time=pos.entry_time,
            exit_time=exit_ts,
            pnl_usd=pnl,
            pnl_pct=pnl_pct,
            exit_reason=reason,
            sl_price=pos.sl_price,
            adx_at_entry=pos.adx_at_entry,
            bb_width_at_entry=pos.bb_width_at_entry,
            volume_ratio_at_entry=pos.volume_ratio_at_entry,
        )
        self.closed_trades.append(trade)
        self.positions = [p for p in self.positions if p is not pos]

        # Cooldown
        # Estimer combien de ms représentent cooldown_bars bougies H4 (4h = 14_400_000 ms)
        cooldown_ms = self.cfg.cooldown_bars * 4 * 3600 * 1000
        self.cooldown_until[pos.symbol] = exit_ts + cooldown_ms

    def _close_all_remaining(self, ts: int) -> None:
        """Ferme toutes les positions restantes à la fin du backtest."""
        for pos in list(self.positions):
            # Trouver le dernier prix connu
            candles = self.candles.get(pos.symbol, [])
            last_price = candles[-1].close if candles else pos.entry_price
            self._close_position(pos, last_price, ts, "END")

    # ── Equity helpers ─────────────────────────────────────────────────────

    def _unrealized_pnl(self, ts: int) -> float:
        """PnL non réalisé de toutes les positions ouvertes."""
        total = 0.0
        for pos in self.positions:
            # Dernier prix = dernière bougie connue
            candles = self.candles.get(pos.symbol, [])
            if not candles:
                continue
            # Trouver la bougie au timestamp ts ou la plus proche avant
            last_price = candles[-1].close
            for c in candles:
                if c.timestamp <= ts:
                    last_price = c.close
                else:
                    break
            if pos.direction == BreakoutDirection.LONG:
                total += (last_price - pos.entry_price) * pos.size
            else:
                total += (pos.entry_price - last_price) * pos.size
        return total

    def _mark_to_market(self, ts: int) -> float:
        """Equity = cash + unrealized PnL."""
        return self.cash + self._unrealized_pnl(ts)

    def _current_equity(self, ts: int) -> float:
        return self._mark_to_market(ts)
