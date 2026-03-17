"""
Bot DCA RSI — Revolut X, maker-only.

Stratégie : Achat quotidien de BTC (80%) et ETH (20%) selon le RSI daily du BTC.
  - RSI > 70      → $0 (skip)
  - 55 < RSI ≤ 70 → $12
  - 45 ≤ RSI ≤ 55 → $24
  - RSI < 45      → $36
  + Crash reserve : achats bonus BTC quand le prix chute -15%/-25%/-35% du 90-day high.

Capital : DCA_CAPITAL_PCT du solde Revolut X, réparti 80% DCA actif / 20% crash reserve.
         Budget calculé dynamiquement au démarrage du bot.
Exchange : Revolut X (maker 0%, taker 0.09%).
Exécution : 1× par jour à l'heure configurée (défaut 10h UTC), maker-only.

Usage :
    python -m src.bot_dca              # Production
    python -m src.bot_dca --dry-run    # Log les ordres sans les exécuter
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import signal
import sys
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Optional
from zoneinfo import ZoneInfo

from src import config
from src.core.models import (
    Balance,
    Candle,
    OrderRequest,
    OrderSide,
    StrategyType,
    TickerData,
)
from src.core.dca_engine import (
    DCAConfig,
    DCAState,
    DCADecision,
    RSIBracket,
    MarketRegime,
    classify_rsi,
    compute_daily_amount,
    compute_mvrv_multiplier,
    classify_regime,
    compute_regime_allocation,
    reset_period_counters,
    split_allocation,
    compute_crash_anchor,
    check_crash_triggers,
    reset_crash_levels_if_recovered,
    remaining_dca_budget,
    remaining_crash_budget,
    is_budget_exhausted,
    compute_buy_size,
    format_summary,
)
from src.core.indicators import rsi_series, sma
from src.core.onchain import fetch_mvrv
from src.exchange.revolut_client import RevolutXClient
from src.exchange.data_provider import DataProvider
from src.notifications.telegram import TelegramNotifier
from src.firebase.trade_logger import (
    log_trade_opened,
    log_trade_closed,
    log_event as fb_log_event,
    log_daily_snapshot as fb_log_daily_snapshot,
    cleanup_old_events as fb_cleanup_events,
)
from src.runtime_overrides import (
    get_heartbeat_override_seconds,
)

# ── Logging ────────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=getattr(logging, config.LOG_LEVEL, logging.INFO),
    format="[%(asctime)s] %(levelname)s %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("tradex.dca_bot")

logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)


# ── Config ─────────────────────────────────────────────────────────────────────

DCA_POLLING_SECONDS: int = config.DCA_POLLING_SECONDS
DCA_HEARTBEAT_SECONDS: int = config.DCA_HEARTBEAT_SECONDS
DCA_MAKER_WAIT_SECONDS: int = config.DCA_MAKER_WAIT_SECONDS
DCA_EXECUTION_HOUR_UTC: int = config.DCA_EXECUTION_HOUR_UTC

DAILY_INTERVAL = 1440  # 1 day in minutes
H4_INTERVAL = 240  # 4 hours in minutes — fallback if daily not available


def _fmt(price: float) -> str:
    """Format a price for display."""
    if price >= 1000:
        return f"{price:,.2f}"
    elif price >= 1:
        return f"{price:.4f}"
    elif price >= 0.0001:
        return f"{price:.6f}"
    else:
        decimals = 6
        temp = price
        while temp < 0.01 and decimals < 10:
            temp *= 10
            decimals += 1
        return f"{price:.{decimals}f}"


# ── State store ────────────────────────────────────────────────────────────────


class DCAStateStore:
    """Persistance atomique de l'état DCA en JSON."""

    def __init__(self, state_file: str) -> None:
        self._path = Path(state_file).resolve()

    def save(self, state: DCAState) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self._path.with_suffix(".json.tmp")
        try:
            with open(tmp, "w") as f:
                json.dump(state.to_dict(), f, indent=2)
            tmp.replace(self._path)
        except Exception as e:
            logger.error("❌ Save failed (%s): %s", self._path.name, e)
            if tmp.exists():
                try:
                    tmp.unlink()
                except OSError:
                    pass

    def load(self) -> Optional[DCAState]:
        if not self._path.exists():
            logger.info("📂 Pas de state DCA — démarrage à vide")
            return None
        try:
            with open(self._path, "r") as f:
                data = json.load(f)
            return DCAState.from_dict(data)
        except Exception as e:
            logger.error("❌ Load state DCA échoué: %s", e)
            return None


# ── Bot principal ──────────────────────────────────────────────────────────────


class DCABot:
    """Bot DCA RSI-based — Revolut X, maker-only, 1 achat/jour."""

    def __init__(self, dry_run: bool = False) -> None:
        self.dry_run = dry_run
        self._running = False

        # ── Services (avant tout, on a besoin du client pour fetch le solde) ──
        self._client = RevolutXClient(
            api_key=config.REVOLUT_X_API_KEY,
            private_key_path=config.REVOLUT_X_PRIVATE_KEY_PATH,
        )
        self._data = DataProvider(self._client)
        self._telegram = TelegramNotifier(
            bot_token=config.TELEGRAM_BOT_TOKEN,
            chat_id=config.TELEGRAM_CHAT_ID,
            silent=self.dry_run,
        )

        # ── Budget dynamique : fetch solde Revolut X → calcul des budgets ──
        usd_balance = self._fetch_usd_balance()
        dca_capital = usd_balance * config.DCA_CAPITAL_PCT
        active_budget = dca_capital * config.DCA_ACTIVE_PCT
        crash_reserve = dca_capital * config.DCA_CRASH_PCT

        logger.info(
            "💰 Solde Revolut X: $%.2f → DCA capital (%.0f%%): $%.2f "
            "(actif %.0f%%: $%.2f | crash %.0f%%: $%.2f)",
            usd_balance, config.DCA_CAPITAL_PCT * 100, dca_capital,
            config.DCA_ACTIVE_PCT * 100, active_budget,
            config.DCA_CRASH_PCT * 100, crash_reserve,
        )

        # Build config from computed budgets
        self._cfg = DCAConfig(
            total_capital=dca_capital,
            active_budget=active_budget,
            crash_reserve=crash_reserve,
            base_daily_amount=config.DCA_BASE_DAILY_AMOUNT,
            max_daily_buy=config.DCA_MAX_DAILY_BUY,
            btc_alloc=config.DCA_BTC_ALLOC,
            eth_alloc=config.DCA_ETH_ALLOC,
            regime_alloc={
                "NORMAL": (config.DCA_ALLOC_NORMAL_BTC, config.DCA_ALLOC_NORMAL_ETH),
                "WEAK": (config.DCA_ALLOC_WEAK_BTC, config.DCA_ALLOC_WEAK_ETH),
                "CAPITULATION": (config.DCA_ALLOC_CAPIT_BTC, config.DCA_ALLOC_CAPIT_ETH),
            },
            rsi_overbought=config.DCA_RSI_OVERBOUGHT,
            rsi_warm=config.DCA_RSI_WARM,
            rsi_neutral_low=config.DCA_RSI_NEUTRAL_LOW,
            crash_levels=[
                (config.DCA_CRASH_DROP_1, config.DCA_CRASH_PCT_1),
                (config.DCA_CRASH_DROP_2, config.DCA_CRASH_PCT_2),
                (config.DCA_CRASH_DROP_3, config.DCA_CRASH_PCT_3),
            ],
            crash_lookback_days=config.DCA_CRASH_LOOKBACK_DAYS,
            crash_anchor_long_days=config.DCA_CRASH_ANCHOR_LONG_DAYS,
            execution_hour_utc=DCA_EXECUTION_HOUR_UTC,
            maker_wait_seconds=DCA_MAKER_WAIT_SECONDS,
            mvrv_enabled=config.DCA_MVRV_ENABLED,
            mvrv_threshold=config.DCA_MVRV_THRESHOLD,
            mvrv_deep_threshold=config.DCA_MVRV_DEEP_THRESHOLD,
            mvrv_mult_low=config.DCA_MVRV_MULT_LOW,
            mvrv_mult_deep=config.DCA_MVRV_MULT_DEEP,
            crash_btc_only=config.DCA_CRASH_BTC_ONLY,
            monthly_cap=config.DCA_MONTHLY_CAP,
            weekly_cap=config.DCA_WEEKLY_CAP,
            boost_cooldown_hours=config.DCA_BOOST_COOLDOWN_HOURS,
            boost_threshold=config.DCA_BOOST_THRESHOLD,
            regime_filter_enabled=config.DCA_REGIME_FILTER_ENABLED,
            capitulation_threshold=config.DCA_CAPITULATION_THRESHOLD,
        )

        # MVRV cache
        self._last_mvrv: float | None = None
        # MA200 cache
        self._ma200: float = 0.0
        self._regime: MarketRegime = MarketRegime.NORMAL

        # State
        data_dir = os.path.join(os.path.dirname(__file__), "..", "data")
        state_file = os.path.join(data_dir, "state_dca.json")
        self._store = DCAStateStore(state_file)
        self._state = self._store.load() or DCAState()

        # Set start date if new
        if not self._state.start_date:
            self._state.start_date = datetime.now(timezone.utc).strftime("%Y-%m-%d")

        # Daily candle highs for rolling high calculation
        self._daily_highs: list[float] = []
        # Daily closes for MA200
        self._daily_closes: list[float] = []

        # Heartbeat
        self._last_heartbeat: float = 0.0
        self._heartbeat_seconds: int = DCA_HEARTBEAT_SECONDS
        self._tick_count: int = 0

        # Track whether we already bought today
        self._last_execution_date: str = self._state.last_buy_date

        # Daily cleanup
        self._last_cleanup_date: str = ""

        if dry_run:
            logger.info("🔧 Mode DRY-RUN — aucun ordre ne sera exécuté")

    # ── Run ────────────────────────────────────────────────────────────────────

    def run(self) -> None:
        self._running = True

        logger.info("═" * 60)
        logger.info("📈 DCA Bot v2 démarré — RSI + MVRV + Regime + Caps")
        logger.info("   Capital  : $%.0f total ($%.0f DCA + $%.0f crash)",
                     self._cfg.total_capital, self._cfg.active_budget, self._cfg.crash_reserve)
        logger.info("   Base     : $%.0f/jour | Max: $%.0f | Alloc: BTC %.0f%% / ETH %.0f%%",
                     self._cfg.base_daily_amount, self._cfg.max_daily_buy,
                     self._cfg.btc_alloc * 100, self._cfg.eth_alloc * 100)
        logger.info("   RSI bands: >%.0f→$0 | %.0f-%.0f→$%.0f | %.0f-%.0f→$%.0f | <%.0f→$%.0f",
                     self._cfg.rsi_overbought,
                     self._cfg.rsi_warm, self._cfg.rsi_overbought,
                     self._cfg.base_daily_amount * 1,
                     self._cfg.rsi_neutral_low, self._cfg.rsi_warm,
                     self._cfg.base_daily_amount * 2,
                     self._cfg.rsi_neutral_low,
                     self._cfg.base_daily_amount * 3)
        logger.info("   MVRV     : %s | <%.2f→×%.1f | <%.2f→×%.1f",
                     "ON" if self._cfg.mvrv_enabled else "OFF",
                     self._cfg.mvrv_threshold, self._cfg.mvrv_mult_low,
                     self._cfg.mvrv_deep_threshold, self._cfg.mvrv_mult_deep)
        logger.info("   Regime   : %s | capit_threshold=%.2f",
                     "ON" if self._cfg.regime_filter_enabled else "OFF",
                     self._cfg.capitulation_threshold)
        logger.info("   Crash    : -15%%→%.0f%% | -25%%→%.0f%% | -35%%→%.0f%% of reserve (anchor %d/%dd)",
                     self._cfg.crash_levels[0][1] * 100,
                     self._cfg.crash_levels[1][1] * 100,
                     self._cfg.crash_levels[2][1] * 100,
                     self._cfg.crash_lookback_days,
                     self._cfg.crash_anchor_long_days)
        logger.info("   Caps     : monthly=$%.0f | weekly=$%.0f | cooldown=%.0fh (seuil $%.0f)",
                     self._cfg.monthly_cap, self._cfg.weekly_cap,
                     self._cfg.boost_cooldown_hours, self._cfg.boost_threshold)
        logger.info("   Exec     : %02d:00 UTC | Polling: %ds | Maker wait: %ds",
                     DCA_EXECUTION_HOUR_UTC, DCA_POLLING_SECONDS, DCA_MAKER_WAIT_SECONDS)
        logger.info("   Spent    : DCA $%.2f / $%.0f | Crash $%.2f / $%.0f",
                     self._state.total_spent_dca, self._cfg.active_budget,
                     self._state.total_spent_crash, self._cfg.crash_reserve)
        logger.info("   Accum    : BTC %.8f | ETH %.8f | Buys: %d | Days: %d",
                     self._state.total_btc_bought, self._state.total_eth_bought,
                     self._state.buy_count, self._state.total_days_active)
        if self.dry_run:
            logger.info("   ⚠️  DRY-RUN actif")
        logger.info("═" * 60)

        self._initialize()

        try:
            while self._running:
                self._tick()
                time.sleep(DCA_POLLING_SECONDS)
        except KeyboardInterrupt:
            logger.info("Arrêt demandé par l'utilisateur")
        finally:
            self._shutdown()

    def stop(self) -> None:
        self._running = False

    # ── Init ───────────────────────────────────────────────────────────────────

    def _fetch_usd_balance(self) -> float:
        """Récupère le solde USD total depuis Revolut X.

        Utilisé au démarrage pour calculer le budget DCA dynamiquement.
        Retourne le solde total (available + reserved) en USD.
        """
        try:
            balances = self._client.get_balances()
            usd = next((b for b in balances if b.currency == "USD"), None)
            if usd is None:
                logger.warning("⚠️ Pas de solde USD trouvé — fallback $0")
                return 0.0
            logger.info("💵 Solde USD: total=$%.2f (available=$%.2f, reserved=$%.2f)",
                        usd.total, usd.available, usd.reserved)
            return usd.total
        except Exception as e:
            logger.error("❌ Impossible de récupérer le solde USD: %s", e)
            return 0.0

    def _initialize(self) -> None:
        """Charge les bougies daily pour le RSI, rolling high et MA200."""
        try:
            candles = self._fetch_daily_candles(self._cfg.btc_symbol)
            if candles:
                self._daily_highs = [c.high for c in candles]
                self._daily_closes = [c.close for c in candles]
                # Crash anchor = max(90j, 180j)
                self._state.rolling_high_price = compute_crash_anchor(
                    self._daily_highs,
                    self._cfg.crash_lookback_days,
                    self._cfg.crash_anchor_long_days,
                )
                # MA200
                if len(self._daily_closes) >= 200:
                    ma200_vals = sma(self._daily_closes, 200)
                    self._ma200 = ma200_vals[-1]
                    self._regime = classify_regime(
                        self._daily_closes[-1], self._ma200, self._cfg
                    )
                else:
                    self._ma200 = 0.0
                    self._regime = MarketRegime.NORMAL
                logger.info(
                    "📊 %d bougies chargées | Crash anchor: %s | MA200: %s | Regime: %s",
                    len(candles), _fmt(self._state.rolling_high_price),
                    _fmt(self._ma200) if self._ma200 > 0 else "N/A",
                    self._regime.value,
                )
            else:
                logger.warning("⚠️ Aucune bougie BTC reçue à l'init")
        except Exception as e:
            logger.error("❌ Erreur chargement bougies init: %s", e)

        logger.info("── Init DCA v2 terminée ──")

    def _fetch_daily_candles(self, symbol: str) -> list[Candle]:
        """Récupère les bougies daily depuis Revolut X.

        Revolut X may not support daily candles directly, so we
        fetch H4 candles and aggregate them to daily ourselves.
        """
        try:
            # Try daily interval first
            candles = self._client.get_candles(symbol, interval=DAILY_INTERVAL)
            candles.sort(key=lambda c: c.timestamp)
            if len(candles) >= 20:
                return candles
        except Exception:
            pass

        # Fallback: fetch H4 candles and aggregate to daily
        logger.info("📊 Daily non dispo, agrégation depuis H4...")
        try:
            # Fetch ~220 days of H4 (6 candles/day × 220 days = 1320) for MA200
            h4_candles = self._client.get_candles(symbol, interval=H4_INTERVAL)
            h4_candles.sort(key=lambda c: c.timestamp)
            if not h4_candles:
                return []
            return self._aggregate_h4_to_daily(h4_candles)
        except Exception as e:
            logger.error("❌ Erreur fetch H4 pour agrégation: %s", e)
            return []

    @staticmethod
    def _aggregate_h4_to_daily(h4_candles: list[Candle]) -> list[Candle]:
        """Agrège les bougies H4 en bougies daily."""
        days: dict[str, list[Candle]] = {}
        for c in h4_candles:
            dt = datetime.fromtimestamp(c.timestamp / 1000, tz=timezone.utc)
            day_key = dt.strftime("%Y-%m-%d")
            days.setdefault(day_key, []).append(c)

        daily: list[Candle] = []
        for day_key in sorted(days.keys()):
            group = days[day_key]
            if not group:
                continue
            daily.append(Candle(
                timestamp=group[0].timestamp,
                open=group[0].open,
                high=max(c.high for c in group),
                low=min(c.low for c in group),
                close=group[-1].close,
                volume=sum(c.volume for c in group),
            ))
        return daily

    # ── Tick ───────────────────────────────────────────────────────────────────

    def _tick(self) -> None:
        """Un cycle de polling."""
        self._tick_count += 1

        now_utc = datetime.now(timezone.utc)
        today_str = now_utc.strftime("%Y-%m-%d")

        # 0. Reset period counters (month/week) if needed
        current_month = now_utc.strftime("%Y-%m")
        current_week = now_utc.strftime("%G-W%V")
        reset_period_counters(self._state, current_month, current_week)

        # 1. Check crash triggers (every tick)
        self._check_crash_reserve()

        # 2. Check daily DCA execution
        if self._should_execute_daily(now_utc, today_str):
            self._execute_daily_dca(today_str)

        # 3. Heartbeat
        self._maybe_heartbeat()

        # 4. Daily cleanup
        self._maybe_daily_tasks(today_str)

    def _should_execute_daily(self, now_utc: datetime, today_str: str) -> bool:
        """Vérifie si c'est l'heure d'exécuter le DCA quotidien."""
        # Already executed today?
        if self._last_execution_date == today_str:
            return False

        # Is it the right hour?
        if now_utc.hour < DCA_EXECUTION_HOUR_UTC:
            return False

        # Budget remaining?
        if remaining_dca_budget(self._state, self._cfg) <= 0:
            return False

        return True

    # ── Daily DCA execution ────────────────────────────────────────────────────

    def _refresh_budget(self) -> None:
        """Recalcule le budget DCA à partir du solde Revolut X actuel.

        Appelé avant chaque exécution quotidienne pour que le budget
        se mette à jour automatiquement si le capital change.
        """
        usd_balance = self._fetch_usd_balance()
        if usd_balance <= 0:
            logger.warning("⚠️ Solde USD = $0 — budget inchangé")
            return

        old_total = self._cfg.total_capital
        dca_capital = usd_balance * config.DCA_CAPITAL_PCT
        self._cfg.total_capital = dca_capital
        self._cfg.active_budget = dca_capital * config.DCA_ACTIVE_PCT
        self._cfg.crash_reserve = dca_capital * config.DCA_CRASH_PCT

        if abs(dca_capital - old_total) > 1.0:
            logger.info(
                "🔄 Budget recalculé: $%.2f → $%.2f (actif $%.2f | crash $%.2f)",
                old_total, dca_capital, self._cfg.active_budget, self._cfg.crash_reserve,
            )

    def _execute_daily_dca(self, today_str: str) -> None:
        """Exécute l'achat DCA quotidien v2 : RSI → MVRV mult → caps → regime → execute."""
        logger.info("═" * 40)
        logger.info("📅 DCA quotidien v2 — %s", today_str)

        now_ts = time.time()

        # 0. Mark today as executed BEFORE placing orders (restart-safe)
        self._last_execution_date = today_str
        self._state.last_buy_date = today_str
        self._save_state()

        # 0b. Refresh budget from live balance
        self._refresh_budget()

        # 0c. Refresh MA200 and regime
        try:
            candles = self._fetch_daily_candles(self._cfg.btc_symbol)
            if candles:
                self._daily_highs = [c.high for c in candles]
                self._daily_closes = [c.close for c in candles]
                self._state.rolling_high_price = compute_crash_anchor(
                    self._daily_highs,
                    self._cfg.crash_lookback_days,
                    self._cfg.crash_anchor_long_days,
                )
                if len(self._daily_closes) >= 200:
                    ma200_vals = sma(self._daily_closes, 200)
                    self._ma200 = ma200_vals[-1]
                    self._regime = classify_regime(
                        self._daily_closes[-1], self._ma200, self._cfg
                    )
        except Exception as e:
            logger.warning("⚠️ Erreur refresh candles: %s", e)

        # 1. Fetch MVRV (if enabled)
        mvrv = None
        if self._cfg.mvrv_enabled:
            mvrv = fetch_mvrv()
            self._last_mvrv = mvrv
            if mvrv is not None:
                logger.info("   📊 MVRV BTC: %.4f", mvrv)
            else:
                logger.warning("   ⚠️ MVRV indisponible — fallback RSI seul")

        # 2. Fetch RSI
        rsi = self._get_btc_daily_rsi()
        bracket = classify_rsi(rsi, self._cfg)
        logger.info("   RSI BTC daily: %.1f → bracket %s", rsi, bracket.value)

        # 3. Compute amount v2 (RSI → MVRV mult → caps → cooldown)
        total_amount, reason, mvrv_mult = compute_daily_amount(
            rsi, self._cfg, mvrv=mvrv, state=self._state, now_ts=now_ts
        )

        # Cap to remaining DCA budget
        remaining = remaining_dca_budget(self._state, self._cfg)
        if total_amount > remaining:
            total_amount = remaining
            reason += f" | BUDGET_CAP (${remaining:.0f})"

        logger.info("   Décision: $%.2f | %s", total_amount, reason)
        logger.info("   Regime: %s | MA200: %s | MVRV mult: ×%.1f",
                     self._regime.value,
                     _fmt(self._ma200) if self._ma200 > 0 else "N/A",
                     mvrv_mult)

        # Build DCADecision for observability
        btc_pct, eth_pct = compute_regime_allocation(self._regime, self._cfg)
        decision = DCADecision(
            date=today_str,
            rsi=rsi,
            bracket=bracket.value,
            mvrv=mvrv,
            mvrv_mult=mvrv_mult,
            regime=self._regime.value,
            base_amount=self._cfg.base_daily_amount * self._cfg.rsi_multipliers.get(bracket.value, 0),
            mvrv_amount=self._cfg.base_daily_amount * self._cfg.rsi_multipliers.get(bracket.value, 0) * mvrv_mult,
            capped_amount=total_amount,
            reason=reason,
            monthly_spent=self._state.monthly_spent,
            weekly_spent=self._state.weekly_spent,
            monthly_cap=self._cfg.monthly_cap,
            weekly_cap=self._cfg.weekly_cap,
            cap_limited="CAP" in reason,
            cooldown_active="COOLDOWN" in reason,
            btc_alloc=btc_pct,
            eth_alloc=eth_pct,
            skipped=(total_amount <= 0),
        )

        # Log decision to Firebase
        try:
            fb_log_event("DCA_DECISION", decision.to_dict(), exchange="revolut-dca")
        except Exception:
            pass

        if total_amount <= 0:
            logger.info("   ⏭️ Pas d'achat aujourd'hui: %s", reason)
            self._state.last_buy_rsi = rsi
            self._state.last_buy_bracket = bracket.value
            self._state.total_days_active += 1
            self._save_state()
            return

        # 4. Split BTC/ETH with regime-based allocation
        allocations = split_allocation(total_amount, self._cfg, regime=self._regime)
        logger.info("   Montant: $%.2f → BTC $%.2f (%.0f%%) / ETH $%.2f (%.0f%%)",
                     total_amount,
                     allocations[self._cfg.btc_symbol], btc_pct * 100,
                     allocations[self._cfg.eth_symbol], eth_pct * 100)

        # 5. Execute buys
        bought_any = False
        for symbol, amount_usd in allocations.items():
            if amount_usd < 1.0:
                continue
            success = self._execute_buy(symbol, amount_usd, reason, rsi, bracket)
            if success:
                bought_any = True

        # 6. Update state
        if bought_any:
            self._state.total_spent_dca += total_amount
            self._state.buy_count += 1
            self._state.monthly_spent += total_amount
            self._state.weekly_spent += total_amount
            # Track boost timestamp
            if total_amount >= self._cfg.boost_threshold:
                self._state.last_boost_ts = now_ts

        self._state.last_buy_rsi = rsi
        self._state.last_buy_bracket = bracket.value
        self._state.total_days_active += 1
        self._save_state()

        logger.info("   ✅ DCA jour terminé | Dépensé: $%.2f | Mois: $%.0f/$%.0f | Sem: $%.0f/$%.0f",
                     total_amount,
                     self._state.monthly_spent, self._cfg.monthly_cap,
                     self._state.weekly_spent, self._cfg.weekly_cap)

        # Telegram notification
        summary = format_summary(self._state, self._cfg)
        self._telegram.notify_dca_buy(
            rsi=rsi,
            bracket=bracket.value,
            amount_usd=total_amount,
            btc_amount=allocations.get(self._cfg.btc_symbol, 0),
            eth_amount=allocations.get(self._cfg.eth_symbol, 0),
            total_spent=summary["total_spent"],
            remaining=summary["remaining"],
            btc_accumulated=self._state.total_btc_bought,
            eth_accumulated=self._state.total_eth_bought,
            buy_count=self._state.buy_count,
            mvrv=mvrv,
            mvrv_mult=mvrv_mult,
            regime=self._regime.value,
            reason=reason,
            monthly_spent=self._state.monthly_spent,
            monthly_cap=self._cfg.monthly_cap,
            weekly_spent=self._state.weekly_spent,
            weekly_cap=self._cfg.weekly_cap,
        )

    # ── Crash reserve ──────────────────────────────────────────────────────────

    def _check_crash_reserve(self) -> None:
        """Vérifie et exécute les achats crash reserve."""
        if remaining_crash_budget(self._state, self._cfg) <= 0:
            return

        # Get current BTC price
        try:
            ticker = self._data.get_ticker(self._cfg.btc_symbol)
            if not ticker:
                return
        except Exception:
            return

        price = ticker.last_price

        # Update rolling high periodically (every ~100 ticks to avoid too many API calls)
        if self._tick_count % 100 == 0:
            try:
                candles = self._fetch_daily_candles(self._cfg.btc_symbol)
                if candles:
                    self._daily_highs = [c.high for c in candles]
                    self._daily_closes = [c.close for c in candles]
                    self._state.rolling_high_price = compute_crash_anchor(
                        self._daily_highs,
                        self._cfg.crash_lookback_days,
                        self._cfg.crash_anchor_long_days,
                    )
            except Exception:
                pass

        rolling_high = self._state.rolling_high_price
        if rolling_high <= 0:
            return

        # Check for crash level resets (price recovered)
        reset = reset_crash_levels_if_recovered(price, rolling_high, self._state, self._cfg)
        if reset:
            logger.info("📈 Crash levels reset (recovery): %s", reset)
            self._save_state()

        # Check for new triggers
        triggers = check_crash_triggers(price, rolling_high, self._state, self._cfg)
        if not triggers:
            return

        for drop_pct, amount_usd in triggers:
            level_name = f"LEVEL_{int(drop_pct * 100)}"
            drop_display = f"-{drop_pct*100:.0f}%"
            logger.info("🚨 CRASH RESERVE déclenchée : %s (prix %s, high %s) → $%.0f BTC",
                         drop_display, _fmt(price), _fmt(rolling_high), amount_usd)

            # ── Persist level BEFORE placing order ──
            # Prevents duplicate buys if bot is killed during maker wait.
            self._state.crash_levels_triggered.append(level_name)
            self._save_state()

            # Buy BTC only for crash reserve (100% BTC)
            rsi = self._get_btc_daily_rsi()
            bracket = classify_rsi(rsi, self._cfg)
            success = self._execute_buy(
                self._cfg.btc_symbol, amount_usd, f"CRASH_{drop_display}", rsi, bracket
            )

            if success:
                self._state.total_spent_crash += amount_usd
                self._state.crash_buy_count += 1
                self._save_state()

                # Telegram
                self._telegram.notify_dca_crash_buy(
                    drop_pct=drop_pct,
                    price=price,
                    rolling_high=rolling_high,
                    amount_usd=amount_usd,
                    crash_spent=self._state.total_spent_crash,
                    crash_remaining=remaining_crash_budget(self._state, self._cfg),
                    levels_triggered=self._state.crash_levels_triggered,
                )
            else:
                # Rollback: remove level if order failed (not filled)
                if level_name in self._state.crash_levels_triggered:
                    self._state.crash_levels_triggered.remove(level_name)
                    self._save_state()
                    logger.warning("↩️ Crash level %s rollback (ordre échoué)", level_name)

    # ── Order execution ────────────────────────────────────────────────────────

    def _execute_buy(
        self,
        symbol: str,
        amount_usd: float,
        reason: str,
        rsi: float,
        bracket: RSIBracket,
    ) -> bool:
        """Exécute un achat maker-only pour un symbole.

        Args:
            symbol: Paire (ex: "BTC-USD").
            amount_usd: Montant USD à investir.
            reason: Raison de l'achat ("DCA" ou "CRASH_-15%").
            rsi: RSI courant.
            bracket: RSI bracket.

        Returns:
            True si l'achat a été exécuté.
        """
        try:
            ticker = self._data.get_ticker(symbol)
            if not ticker:
                logger.warning("[%s] Ticker non disponible — skip", symbol)
                return False
        except Exception as e:
            logger.error("[%s] Erreur ticker: %s", symbol, e)
            return False

        price = ticker.bid  # Use bid for passive buy order
        if price <= 0:
            logger.warning("[%s] Prix bid invalide: %s", symbol, price)
            return False

        size = compute_buy_size(amount_usd, price)
        if size <= 0:
            return False

        # Determine precision for base size
        base = symbol.split("-")[0] if "-" in symbol else symbol
        if base == "BTC":
            size_str = f"{size:.8f}"
        elif base == "ETH":
            size_str = f"{size:.6f}"
        else:
            size_str = f"{size:.8f}"

        # Determine price precision
        if price >= 1000:
            price_str = f"{price:.2f}"
        elif price >= 1:
            price_str = f"{price:.4f}"
        else:
            price_str = f"{price:.6f}"

        logger.info(
            "   💰 %s BUY %s | $%.2f @ %s (%s %s) | RSI %.1f [%s]",
            reason, symbol, amount_usd, price_str, size_str, base, rsi, bracket.value,
        )

        if self.dry_run:
            logger.info("   🔧 DRY-RUN — ordre non exécuté")
            # Still track for simulation
            if base == "BTC":
                self._state.total_btc_bought += float(size_str)
            elif base == "ETH":
                self._state.total_eth_bought += float(size_str)
            return True

        # Place maker-only order
        order = OrderRequest(
            symbol=symbol,
            side=OrderSide.BUY,
            base_size=size_str,
            price=price_str,
        )

        try:
            result = self._place_maker_only_order(order)
            fill_type = result.get("fill_type", "unknown")
            venue_order_id = result.get("venue_order_id", "unknown")
            actual_price = float(result.get("actual_price", price))

            # Update accumulated amounts
            actual_size = float(result.get("filled_size", size_str))
            actual_cost = actual_size * actual_price

            if base == "BTC":
                self._state.total_btc_bought += actual_size
            elif base == "ETH":
                self._state.total_eth_bought += actual_size

            logger.info(
                "   ✅ Fill %s | %s %s @ %s | cost $%.2f",
                fill_type, size_str, base, _fmt(actual_price), actual_cost,
            )

            # Firebase trade log
            try:
                fb_log_event(
                    "DCA_BUY",
                    {
                        "symbol": symbol,
                        "reason": reason,
                        "amount_usd": amount_usd,
                        "price": actual_price,
                        "size": actual_size,
                        "rsi": rsi,
                        "bracket": bracket.value,
                        "fill_type": fill_type,
                        "venue_order_id": venue_order_id,
                    },
                    exchange="revolut-dca",
                )
            except Exception:
                pass

            return True

        except Exception as e:
            logger.error("   ❌ Échec achat %s: %s", symbol, e)
            return False

    def _place_maker_only_order(self, order: OrderRequest) -> dict:
        """Place un ordre limit passif (maker 0%). Si pas rempli → annule et retry taker."""
        logger.info(
            "💰 MAKER-ONLY | %s %s @ %s (attente %ds)",
            order.side.value.upper(), order.symbol, order.price, DCA_MAKER_WAIT_SECONDS,
        )

        try:
            resp = self._client.place_order(order)
        except Exception as e:
            logger.error("💰 MAKER-ONLY | Placement échoué: %s", e)
            raise

        data = resp.get("data", {})
        if isinstance(data, dict):
            venue_order_id = data.get("venue_order_id", "unknown")
        elif isinstance(data, list) and data:
            venue_order_id = data[0].get("venue_order_id", "unknown")
        else:
            venue_order_id = "unknown"

        if venue_order_id == "unknown":
            return {"venue_order_id": venue_order_id, "fill_type": "maker",
                    "actual_price": float(order.price)}

        # Check instant fill
        initial_state = ""
        if isinstance(data, dict):
            initial_state = (data.get("state") or data.get("status") or "").upper()
        if initial_state == "FILLED":
            logger.info("💰 MAKER-ONLY | ✅ Fill instantané")
            return {"venue_order_id": venue_order_id, "fill_type": "maker",
                    "actual_price": float(order.price)}

        # Wait for fill
        logger.info("💰 MAKER-ONLY | ⏳ Attente %ds…", DCA_MAKER_WAIT_SECONDS)
        time.sleep(DCA_MAKER_WAIT_SECONDS)

        # Check status
        try:
            order_status = self._client.get_order(venue_order_id)
        except Exception as e:
            logger.warning("💰 MAKER-ONLY | Status check failed: %s → assume filled", e)
            return {"venue_order_id": venue_order_id, "fill_type": "maker",
                    "actual_price": float(order.price)}

        order_data_raw = order_status.get("data", order_status) if isinstance(order_status, dict) else order_status
        if isinstance(order_data_raw, list) and order_data_raw:
            order_data_raw = order_data_raw[0]
        od: dict = order_data_raw if isinstance(order_data_raw, dict) else {}
        status = (od.get("status") or od.get("state") or "").upper()
        filled_size = float(od.get("filled_size", "0") or "0")
        total_size = float(order.base_size)

        if status == "FILLED" or (filled_size > 0 and filled_size >= total_size * 0.99):
            logger.info("💰 MAKER-ONLY | ✅ Rempli — fee 0%%")
            return {"venue_order_id": venue_order_id, "fill_type": "maker",
                    "actual_price": float(order.price)}

        if filled_size > 0:
            logger.info("💰 MAKER-ONLY | ⚡ Fill partiel (%.8f/%.8f)", filled_size, total_size)
            try:
                self._client.cancel_order(venue_order_id)
            except Exception:
                pass
            return {
                "venue_order_id": venue_order_id,
                "fill_type": "partial_maker",
                "actual_price": float(order.price),
                "filled_size": filled_size,
            }

        # Not filled → cancel and try taker (aggressive limit)
        logger.info("💰 MAKER-ONLY | ❌ Pas de fill → bascule taker")
        try:
            self._client.cancel_order(venue_order_id)
        except Exception as e:
            if "409" in str(e) or "conflict" in str(e).lower():
                try:
                    recheck = self._client.get_order(venue_order_id)
                    rd_raw = recheck.get("data", recheck) if isinstance(recheck, dict) else recheck
                    if isinstance(rd_raw, list) and rd_raw:
                        rd_raw = rd_raw[0]
                    rd: dict = rd_raw if isinstance(rd_raw, dict) else {}
                    if (rd.get("status") or rd.get("state") or "").upper() == "FILLED":
                        logger.info("💰 MAKER-ONLY | ✅ Rempli entre-temps!")
                        return {"venue_order_id": venue_order_id, "fill_type": "maker",
                                "actual_price": float(order.price)}
                except Exception:
                    pass

        # Place taker order (cross the spread)
        try:
            ticker = self._data.get_ticker(order.symbol)
            if ticker:
                taker_price = ticker.ask * 1.001  # Slightly above ask to ensure fill
                if taker_price >= 1000:
                    taker_price_str = f"{taker_price:.2f}"
                elif taker_price >= 1:
                    taker_price_str = f"{taker_price:.4f}"
                else:
                    taker_price_str = f"{taker_price:.6f}"

                taker_order = OrderRequest(
                    symbol=order.symbol,
                    side=order.side,
                    base_size=order.base_size,
                    price=taker_price_str,
                )
                resp2 = self._client.place_order(taker_order)
                data2 = resp2.get("data", {})
                taker_id = data2.get("venue_order_id", "unknown") if isinstance(data2, dict) else "unknown"
                logger.info("💰 TAKER | Ordre placé @ %s → fee 0.09%%", taker_price_str)

                # Wait briefly for taker fill
                time.sleep(5)
                return {
                    "venue_order_id": taker_id,
                    "fill_type": "taker",
                    "actual_price": taker_price,
                }
        except Exception as e2:
            logger.error("💰 TAKER | Échec: %s", e2)
            raise

        return {"venue_order_id": "unknown", "fill_type": "failed",
                "actual_price": float(order.price)}

    # ── RSI calculation ────────────────────────────────────────────────────────

    def _get_btc_daily_rsi(self, period: int = 14) -> float:
        """Calcule le RSI daily du BTC."""
        try:
            candles = self._fetch_daily_candles(self._cfg.btc_symbol)
            if len(candles) < period + 1:
                logger.warning("Pas assez de bougies pour le RSI (%d)", len(candles))
                return 50.0
            rsi_vals = rsi_series(candles, period)
            return rsi_vals[-1]
        except Exception as e:
            logger.error("❌ Erreur calcul RSI: %s", e)
            return 50.0

    # ── State persistence ──────────────────────────────────────────────────────

    def _save_state(self) -> None:
        """Persiste l'état DCA en JSON."""
        self._store.save(self._state)

    # ── Heartbeat ──────────────────────────────────────────────────────────────

    def _maybe_heartbeat(self) -> None:
        now = time.time()
        heartbeat_seconds = get_heartbeat_override_seconds("dca", DCA_HEARTBEAT_SECONDS)
        if heartbeat_seconds != self._heartbeat_seconds:
            self._heartbeat_seconds = heartbeat_seconds
            logger.info("💓 [DCA] Heartbeat runtime override: %ss", self._heartbeat_seconds)

        if now - self._last_heartbeat < self._heartbeat_seconds:
            return
        self._last_heartbeat = now

        # Get current prices
        btc_price = 0.0
        eth_price = 0.0
        try:
            btc_ticker = self._data.get_ticker(self._cfg.btc_symbol)
            if btc_ticker:
                btc_price = btc_ticker.last_price
        except Exception:
            pass
        try:
            eth_ticker = self._data.get_ticker(self._cfg.eth_symbol)
            if eth_ticker:
                eth_price = eth_ticker.last_price
        except Exception:
            pass

        # Portfolio value (BTC + ETH holdings)
        btc_value = self._state.total_btc_bought * btc_price
        eth_value = self._state.total_eth_bought * eth_price
        portfolio_value = btc_value + eth_value
        total_spent = self._state.total_spent_dca + self._state.total_spent_crash
        pnl = portfolio_value - total_spent
        pnl_pct = (pnl / total_spent * 100) if total_spent > 0 else 0

        # RSI + MVRV
        rsi = self._get_btc_daily_rsi()
        mvrv = None
        mvrv_mult = 1.0
        if self._cfg.mvrv_enabled:
            mvrv = fetch_mvrv()
            self._last_mvrv = mvrv
            mvrv_mult = compute_mvrv_multiplier(mvrv, self._cfg)
        bracket = classify_rsi(rsi, self._cfg)

        # Rolling high & crash info
        rolling_high = self._state.rolling_high_price
        drop_pct = ((rolling_high - btc_price) / rolling_high * 100) if rolling_high > 0 else 0

        summary = format_summary(self._state, self._cfg)

        # Console heartbeat
        logger.info("═" * 40)
        logger.info("💓 DCA Heartbeat v2 | tick=%d", self._tick_count)
        mvrv_str = f"{mvrv:.4f}" if mvrv is not None else "N/A"
        logger.info("   RSI: %.1f [%s] | MVRV: %s (×%.1f) | Regime: %s",
                     rsi, bracket.value, mvrv_str, mvrv_mult, self._regime.value)
        logger.info("   BTC: %s | ETH: %s | MA200: %s",
                     _fmt(btc_price), _fmt(eth_price),
                     _fmt(self._ma200) if self._ma200 > 0 else "N/A")
        logger.info("   Portfolio: $%.2f (BTC $%.2f + ETH $%.2f)",
                     portfolio_value, btc_value, eth_value)
        logger.info("   Dépensé: $%.2f | PnL: $%+.2f (%+.1f%%)",
                     total_spent, pnl, pnl_pct)
        logger.info("   Budget DCA: $%.0f restant | Crash: $%.0f restant",
                     summary["dca_remaining"], summary["crash_remaining"])
        logger.info("   Caps: mois $%.0f/$%.0f | sem $%.0f/$%.0f",
                     self._state.monthly_spent, self._cfg.monthly_cap,
                     self._state.weekly_spent, self._cfg.weekly_cap)
        logger.info("   Rolling high: %s | Drop: %.1f%%",
                     _fmt(rolling_high), drop_pct)
        logger.info("   Crash levels triggered: %s",
                     ", ".join(self._state.crash_levels_triggered) or "aucun")
        logger.info("   Accum: BTC %.8f | ETH %.8f",
                     self._state.total_btc_bought, self._state.total_eth_bought)
        logger.info("═" * 40)

        # Telegram heartbeat
        self._telegram.notify_dca_heartbeat(
            rsi=rsi,
            bracket=bracket.value,
            btc_price=btc_price,
            eth_price=eth_price,
            portfolio_value=portfolio_value,
            total_spent=total_spent,
            pnl=pnl,
            pnl_pct=pnl_pct,
            dca_remaining=summary["dca_remaining"],
            crash_remaining=summary["crash_remaining"],
            btc_accumulated=self._state.total_btc_bought,
            eth_accumulated=self._state.total_eth_bought,
            buy_count=self._state.buy_count,
            crash_buy_count=self._state.crash_buy_count,
            rolling_high=rolling_high,
            drop_pct=drop_pct,
            crash_levels_triggered=self._state.crash_levels_triggered,
            days_active=self._state.total_days_active,
            mvrv=mvrv,
            mvrv_mult=mvrv_mult,
            regime=self._regime.value,
            ma200=self._ma200,
            monthly_spent=self._state.monthly_spent,
            monthly_cap=self._cfg.monthly_cap,
            weekly_spent=self._state.weekly_spent,
            weekly_cap=self._cfg.weekly_cap,
        )

        # Firebase heartbeat (use log_event directly — DCA has different data shape)
        try:
            fb_log_event(
                "DCA_HEARTBEAT",
                {
                    "rsi": rsi,
                    "bracket": bracket.value,
                    "total_spent": total_spent,
                    "pnl": pnl,
                    "pnl_pct": pnl_pct,
                    "btc_accumulated": self._state.total_btc_bought,
                    "eth_accumulated": self._state.total_eth_bought,
                    "dca_remaining": summary["dca_remaining"],
                    "crash_remaining": summary["crash_remaining"],
                    "mvrv": mvrv,
                    "mvrv_mult": mvrv_mult,
                    "regime": self._regime.value,
                    "ma200": self._ma200,
                    "equity": portfolio_value,
                    "rolling_high": rolling_high,
                    "drop_pct": drop_pct,
                    "crash_levels_triggered": self._state.crash_levels_triggered,
                    "buy_count": self._state.buy_count,
                    "crash_buy_count": self._state.crash_buy_count,
                    "days_active": self._state.total_days_active,
                    "monthly_spent": self._state.monthly_spent,
                    "monthly_cap": self._cfg.monthly_cap,
                    "weekly_spent": self._state.weekly_spent,
                    "weekly_cap": self._cfg.weekly_cap,
                },
                symbol="DCA",
                exchange="revolut-dca",
            )
        except Exception:
            pass

    # ── Daily tasks ────────────────────────────────────────────────────────────

    def _maybe_daily_tasks(self, today_str: str) -> None:
        """Cleanup events + daily snapshot."""
        if self._last_cleanup_date == today_str:
            return
        self._last_cleanup_date = today_str

        try:
            fb_cleanup_events()
        except Exception:
            pass

        # Daily snapshot
        try:
            btc_ticker = self._data.get_ticker(self._cfg.btc_symbol)
            eth_ticker = self._data.get_ticker(self._cfg.eth_symbol)
            btc_price = btc_ticker.last_price if btc_ticker else 0
            eth_price = eth_ticker.last_price if eth_ticker else 0
            portfolio = self._state.total_btc_bought * btc_price + self._state.total_eth_bought * eth_price

            fb_log_daily_snapshot(
                equity=portfolio,
                positions=[],
                daily_pnl=portfolio - (self._state.total_spent_dca + self._state.total_spent_crash),
                trades_today=self._state.buy_count + self._state.crash_buy_count,
                exchange="revolut-dca",
            )
        except Exception:
            pass

    # ── Shutdown ───────────────────────────────────────────────────────────────

    def _shutdown(self) -> None:
        logger.info("🛑 Arrêt DCA Bot...")
        self._save_state()
        logger.info("💾 État final sauvegardé")
        self._client.close()
        self._telegram.close()
        logger.info("DCA Bot arrêté proprement")


# ── Point d'entrée ─────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="TradeX – DCA Bot (RSI-based daily buying)")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Mode simulation (log les ordres sans les exécuter)",
    )
    args = parser.parse_args()

    bot = DCABot(dry_run=args.dry_run)
    signal.signal(signal.SIGTERM, lambda *_: bot.stop())
    bot.run()


if __name__ == "__main__":
    main()
