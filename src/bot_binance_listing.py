"""
Boucle principale du bot TradeX — Binance Listing Event.

Stratégie : détecte les nouveaux listings de paires USDC sur Binance Spot,
vérifie un filtre momentum (≥30% dans la 1ère minute), et entre en MARKET BUY.
Gestion par OCO dynamique à 2 paliers (SL -8%, TP +30%, puis re-arm à +100%).
Force close après 7 jours.

Architecture :
  - Même BinanceClient que les autres bots (API keys partagées)
  - État séparé (state_binance_listing.json)
  - Capital alloué dynamiquement via l'allocator (30% fixe du total Binance)
  - Détection via exchangeInfo polling + listing_detector.py (core)
  - Exit via OCO natif Binance (SL + TP)

Usage :
    python -m src.bot_binance_listing              # Production
    python -m src.bot_binance_listing --dry-run    # Simulation
"""

from __future__ import annotations

import argparse
import json as _json
import logging
import os
import signal
import sys
import time
from datetime import datetime, timezone
from typing import Optional

from src import config
from src.core.models import (
    Balance,
    OrderSide,
    Position,
    PositionStatus,
    StrategyType,
)
from src.core.listing_detector import (
    check_momentum,
    compute_oco_levels,
    compute_position_size,
    compute_rearm_oco_levels,
    detect_new_symbols,
    should_force_close,
    should_rearm_oco,
)
from src.core.position_store import PositionStore
from src.core.allocator import compute_allocation, compute_profit_factor
from src.exchange.binance_client import BinanceClient
from src.notifications.telegram import TelegramNotifier
from src.firebase.trade_logger import (
    log_trade_opened,
    log_trade_closed,
    log_event as fb_log_event,
    log_heartbeat as fb_log_heartbeat,
    log_daily_snapshot as fb_log_daily_snapshot,
    cleanup_old_events as fb_cleanup_events,
    log_allocation as fb_log_allocation,
    get_trail_range_pnl_list as fb_get_trail_range_pnl_list,
)


# ── Logging ────────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=getattr(logging, config.LOG_LEVEL, logging.INFO),
    format="[%(asctime)s] %(levelname)s %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("tradex.bot.binance.listing")

logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)

EXCHANGE_NAME = "binance-listing"

_STATE_FILE = os.environ.get(
    "TRADEX_BINANCE_LISTING_STATE_FILE",
    os.path.join(os.path.dirname(__file__), "..", "data", "state_binance_listing.json"),
)

# Fichier persistant pour les symboles déjà connus (évite de re-traiter les anciens)
_KNOWN_SYMBOLS_FILE = os.path.join(
    os.path.dirname(__file__), "..", "data", "listing_known_symbols.json",
)

# Fichier persistant pour les symboles skippés (momentum insuffisant)
_SKIPPED_SYMBOLS_FILE = os.path.join(
    os.path.dirname(__file__), "..", "data", "listing_skipped_symbols.json",
)


def _fmt(price: float) -> str:
    """Formate un prix de façon lisible."""
    if price >= 1000:
        return f"{price:,.4f}"
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


def _pct(a: float, b: float) -> str:
    if b == 0:
        return "N/A"
    return f"{((a - b) / b) * 100:+.2f}%"


class TradeXBinanceListingBot:
    """Bot Listing Event pour Binance — achète les nouveaux listings USDC."""

    def __init__(self, dry_run: bool = False) -> None:
        self.dry_run = dry_run
        self._running = False

        # Services Binance
        self._client = BinanceClient(
            api_key=config.BINANCE_API_KEY,
            secret_key=config.BINANCE_SECRET_KEY,
            base_url=config.BINANCE_BASE_URL,
        )
        self._telegram = TelegramNotifier(
            bot_token=config.TELEGRAM_BOT_TOKEN,
            chat_id=config.TELEGRAM_CHAT_ID,
            silent=self.dry_run,
        )

        # Persistance
        self._store = PositionStore(state_file=_STATE_FILE)

        # Positions ouvertes : symbol → state dict
        self._positions: dict[str, dict] = {}

        # Symboles USDC connus (pour détecter les nouveaux)
        self._known_symbols: set[str] = set()

        # Symboles skippés (déjà évalués, momentum insuffisant ou déjà tradés)
        self._skipped_symbols: set[str] = set()

        # Cache exchangeInfo
        self._last_exchange_info_fetch: float = 0.0
        self._cached_usdc_symbols: set[str] = set()

        # OCO tracking
        self._oco_orders: dict[str, dict] = {}  # symbol → {"order_list_id": int, "tp": float, ...}

        # Dynamic allocation
        self._allocated_balance: float = config.LISTING_ALLOCATED_BALANCE  # fallback
        self._allocation_pct: float = 0.30
        self._last_allocation_date: str = ""

        # Heartbeat
        self._last_heartbeat_time: float = 0.0
        self._cycle_count: int = 0

        # Stats
        self._listings_detected: int = 0
        self._listings_traded: int = 0
        self._listings_skipped_momentum: int = 0

    # ═══════════════════════════════════════════════════════════════════════════
    #  LIFECYCLE
    # ═══════════════════════════════════════════════════════════════════════════

    def start(self) -> None:
        """Démarrage du bot listing."""
        self._running = True
        signal.signal(signal.SIGTERM, self._handle_shutdown)
        signal.signal(signal.SIGINT, self._handle_shutdown)

        logger.info("=" * 60)
        logger.info("🆕 ListingBot Binance démarré (dry_run=%s)", self.dry_run)
        logger.info(
            "Config: SL=%.0f%% TP=%.0f%% momentum≥%.0f%% slots=%d horizon=%dj cap=$%s",
            config.LISTING_SL_INIT_PCT * 100,
            config.LISTING_TP_INIT_PCT * 100,
            config.LISTING_MOMENTUM_PCT * 100,
            config.LISTING_MAX_SLOTS,
            config.LISTING_HORIZON_DAYS,
            f"{config.LISTING_MAX_ALLOC_USD:,.0f}",
        )
        logger.info("=" * 60)

        # Charger état persisté
        self._load_known_symbols()
        self._load_skipped_symbols()
        self._load_positions()

        # Allocation dynamique
        self._init_allocation()

        # Initialiser known_symbols avec les paires USDC actuelles
        if not self._known_symbols:
            logger.info("Première exécution : chargement de tous les symboles USDC existants...")
            current = self._fetch_usdc_symbols()
            self._known_symbols = current
            self._save_known_symbols()
            logger.info("  → %d symboles USDC initialisés comme connus", len(current))

        self._telegram.send_raw(
            f"🆕 *ListingBot Binance démarré*\n"
            f"  Mode: {'DRY RUN 🧪' if self.dry_run else 'PRODUCTION 🔴'}\n"
            f"  Capital: `${self._allocated_balance:,.0f}` ({self._allocation_pct*100:.0f}% du total)\n"
            f"  Slots: {config.LISTING_MAX_SLOTS} | Cap: `${config.LISTING_MAX_ALLOC_USD:,.0f}`\n"
            f"  SL: `-{config.LISTING_SL_INIT_PCT*100:.0f}%` | TP: `+{config.LISTING_TP_INIT_PCT*100:.0f}%`\n"
            f"  Momentum: `≥{config.LISTING_MOMENTUM_PCT*100:.0f}%` en {config.LISTING_MOMENTUM_WINDOW_MIN}min\n"
            f"  Symboles connus: {len(self._known_symbols)} | Positions open: {len(self._positions)}"
        )

        self._main_loop()

    def _handle_shutdown(self, signum, frame) -> None:
        logger.info("Signal %s reçu → arrêt gracieux", signum)
        self._running = False

    def _main_loop(self) -> None:
        """Boucle principale — poll toutes les N secondes."""
        poll_interval = config.LISTING_POLL_INTERVAL_SECONDS

        while self._running:
            try:
                self._cycle_count += 1
                self._run_cycle()
            except Exception:
                logger.exception("Erreur dans le cycle #%d", self._cycle_count)
                try:
                    self._telegram.notify_error(
                        f"ListingBot cycle #{self._cycle_count} — erreur (voir logs)"
                    )
                except Exception:
                    pass

            # Heartbeat
            now = time.time()
            if now - self._last_heartbeat_time >= config.LISTING_HEARTBEAT_SECONDS:
                self._send_heartbeat()
                self._last_heartbeat_time = now

            time.sleep(poll_interval)

        logger.info("Bot arrêté après %d cycles.", self._cycle_count)
        self._telegram.send_raw("🛑 *ListingBot arrêté*")

    # ═══════════════════════════════════════════════════════════════════════════
    #  CYCLE PRINCIPAL
    # ═══════════════════════════════════════════════════════════════════════════

    def _run_cycle(self) -> None:
        """Un cycle = recompute allocation + scan + manage positions."""
        # 0. Recompute allocation si nouveau jour UTC
        self._maybe_recompute_allocation()

        # 1. Scanner les nouveaux listings
        self._scan_new_listings()

        # 2. Gérer les positions ouvertes (OCO status, re-arm, force close)
        self._manage_open_positions()

    # ═══════════════════════════════════════════════════════════════════════════
    #  1. SCAN — DÉTECTION DE NOUVEAUX LISTINGS
    # ═══════════════════════════════════════════════════════════════════════════

    def _fetch_usdc_symbols(self) -> set[str]:
        """Récupère tous les symboles USDC en TRADING depuis exchangeInfo (avec cache)."""
        now = time.time()
        cache_ttl = config.LISTING_EXCHANGEINFO_CACHE_SECONDS

        if now - self._last_exchange_info_fetch < cache_ttl and self._cached_usdc_symbols:
            return self._cached_usdc_symbols

        try:
            info = self._client.get_exchange_info()
            symbols = set()
            for sym in info.get("symbols", []):
                if (
                    sym.get("quoteAsset") == "USDC"
                    and sym.get("status") == "TRADING"
                    and sym.get("isSpotTradingAllowed", False)
                ):
                    symbols.add(sym["symbol"])

            self._cached_usdc_symbols = symbols
            self._last_exchange_info_fetch = now
            return symbols
        except Exception as e:
            logger.warning("Erreur exchangeInfo: %s", e)
            return self._cached_usdc_symbols

    def _scan_new_listings(self) -> None:
        """Détecte les nouveaux symboles USDC et tente d'entrer."""
        current_symbols = self._fetch_usdc_symbols()
        if not current_symbols:
            return

        new_symbols = detect_new_symbols(current_symbols, self._known_symbols)
        if not new_symbols:
            return

        for symbol in new_symbols:
            # Marquer comme connu immédiatement (même si on skip)
            self._known_symbols.add(symbol)

            # Skip si déjà traité
            if symbol in self._skipped_symbols:
                continue
            if symbol in self._positions:
                continue

            self._listings_detected += 1
            logger.info("🆕 NOUVEAU LISTING DÉTECTÉ: %s", symbol)

            # Notification Telegram immédiate — nouveau listing détecté
            self._telegram.send_raw(
                f"🔔 *Nouveau listing détecté* — `{symbol}`\n"
                f"  Vérification momentum ≥{config.LISTING_MOMENTUM_PCT*100:.0f}%..."
            )

            # Vérifier le momentum
            try:
                self._process_new_listing(symbol)
            except Exception:
                logger.exception("Erreur traitement listing %s", symbol)
                self._skipped_symbols.add(symbol)

        # Persister les sets mis à jour
        self._save_known_symbols()
        self._save_skipped_symbols()

    def _process_new_listing(self, symbol: str) -> None:
        """Traite un nouveau listing : check momentum → entrée si OK."""
        # Récupérer les premières bougies 1m
        window = config.LISTING_MOMENTUM_WINDOW_MIN
        candles_1m = self._fetch_1m_candles(symbol, limit=max(window, 5))

        if not candles_1m:
            logger.warning("%s: pas de données 1m disponibles, skip", symbol)
            self._skipped_symbols.add(symbol)
            self._telegram.send_raw(
                f"⏭️ *{symbol}* — skip (pas de données 1m)"
            )
            return

        # Check momentum
        signal_result = check_momentum(
            candles_1m=candles_1m,
            momentum_threshold=config.LISTING_MOMENTUM_PCT,
            window_minutes=window,
        )

        if signal_result is None or not signal_result.momentum_ok:
            pump = signal_result.momentum_pct * 100 if signal_result else 0
            logger.info(
                "%s: momentum insuffisant (%.1f%% < %.0f%%) → skip",
                symbol, pump, config.LISTING_MOMENTUM_PCT * 100,
            )
            self._listings_skipped_momentum += 1
            self._skipped_symbols.add(symbol)
            self._telegram.send_raw(
                f"⏭️ *{symbol}* — skip (momentum `{pump:+.1f}%` "
                f"< `{config.LISTING_MOMENTUM_PCT*100:.0f}%`)"
            )
            return

        logger.info(
            "✅ %s: momentum OK (+%.1f%%) — préparation entrée",
            symbol, signal_result.momentum_pct * 100,
        )

        # Vérifier les slots disponibles
        if len(self._positions) >= config.LISTING_MAX_SLOTS:
            logger.warning(
                "%s: tous les slots occupés (%d/%d) → skip",
                symbol, len(self._positions), config.LISTING_MAX_SLOTS,
            )
            self._skipped_symbols.add(symbol)
            self._telegram.send_raw(
                f"⏭️ *{symbol}* — skip (slots pleins "
                f"{len(self._positions)}/{config.LISTING_MAX_SLOTS})"
            )
            return

        # Exécuter l'entrée
        self._execute_entry(symbol, signal_result.momentum_pct)

    def _fetch_1m_candles(self, symbol: str, limit: int = 5) -> list[dict]:
        """Récupère les premières bougies 1m d'un symbole."""
        try:
            candles = self._client.get_candles(
                symbol=symbol,
                interval=1,  # 1 minute
                limit=limit,
            )
            return [
                {
                    "symbol": symbol,
                    "open": c.open,
                    "high": c.high,
                    "low": c.low,
                    "close": c.close,
                    "volume": c.volume,
                    "timestamp": c.timestamp,
                }
                for c in candles
            ]
        except Exception as e:
            logger.warning("%s: erreur récupération 1m: %s", symbol, e)
            return []

    # ═══════════════════════════════════════════════════════════════════════════
    #  ENTRÉE — MARKET BUY + OCO
    # ═══════════════════════════════════════════════════════════════════════════

    def _execute_entry(self, symbol: str, momentum_pct: float) -> None:
        """Place un market buy et un OCO pour un nouveau listing."""
        # Sizing
        equity = self._get_listing_equity()
        cash = self._get_usdc_balance()
        alloc_usd = compute_position_size(
            equity=equity,
            cash=cash,
            max_slots=config.LISTING_MAX_SLOTS,
            max_alloc_usd=config.LISTING_MAX_ALLOC_USD,
        )

        if alloc_usd < 10:
            logger.warning("%s: allocation trop faible ($%.2f) → skip", symbol, alloc_usd)
            self._skipped_symbols.add(symbol)
            return

        # Obtenir le prix courant
        try:
            ask_price = self._client.get_ticker_price(symbol)
        except Exception as e:
            logger.error("%s: impossible d'obtenir le prix: %s", symbol, e)
            return

        if ask_price <= 0:
            logger.error("%s: prix invalide (%.8f)", symbol, ask_price)
            return

        # Quantité
        raw_qty = alloc_usd / ask_price
        qty_str = self._client.format_quantity(symbol, raw_qty, market=True)
        qty_float = float(qty_str)

        if qty_float <= 0:
            logger.error("%s: quantité nulle après formatage", symbol)
            return

        # Vérifier min notional
        if not self._client.check_min_notional(symbol, qty_float, ask_price):
            logger.warning("%s: min notional non respecté → skip", symbol)
            self._skipped_symbols.add(symbol)
            return

        logger.info(
            "🛒 %s: MARKET BUY qty=%s (~$%.0f) @ ~%s",
            symbol, qty_str, alloc_usd, _fmt(ask_price),
        )

        if self.dry_run:
            entry_price = ask_price
            order_id = f"DRY-{symbol}-{int(time.time())}"
            logger.info("  [DRY RUN] Ordre simulé: %s", order_id)
        else:
            try:
                resp = self._client.place_market_order(
                    symbol=symbol,
                    side="BUY",
                    quantity=qty_str,
                )
                order_id = str(resp.get("orderId", "unknown"))
                # Calculer le prix moyen de remplissage
                fills = resp.get("fills", [])
                if fills:
                    total_qty = sum(float(f["qty"]) for f in fills)
                    total_cost = sum(float(f["qty"]) * float(f["price"]) for f in fills)
                    entry_price = total_cost / total_qty if total_qty > 0 else ask_price
                    qty_float = total_qty
                    qty_str = self._client.format_quantity(symbol, qty_float)
                else:
                    entry_price = ask_price

                logger.info(
                    "✅ %s: FILL @ %s (qty=%s) orderId=%s",
                    symbol, _fmt(entry_price), qty_str, order_id,
                )
            except Exception as e:
                logger.error("❌ %s: MARKET BUY échoué: %s", symbol, e)
                self._telegram.notify_error(f"ListingBot: BUY {symbol} échoué — {e}")
                return

        # Calculer les niveaux OCO
        oco = compute_oco_levels(
            entry_price=entry_price,
            sl_pct=config.LISTING_SL_INIT_PCT,
            tp_pct=config.LISTING_TP_INIT_PCT,
        )

        # Placer l'OCO SELL
        oco_list_id = self._place_oco_sell(symbol, qty_str, entry_price, oco.sl_price, oco.tp_price)

        # Enregistrer la position
        entry_ts = int(time.time() * 1000)
        pos_data = {
            "symbol": symbol,
            "entry_price": entry_price,
            "entry_ts": entry_ts,
            "size": qty_float,
            "size_str": qty_str,
            "alloc_usd": alloc_usd,
            "sl_price": oco.sl_price,
            "tp_price": oco.tp_price,
            "tp1_price": oco.tp_price,  # Garder le TP initial pour le re-arm
            "oco_list_id": oco_list_id,
            "oco_rearmed": False,
            "order_id": order_id,
            "momentum_pct": momentum_pct,
        }
        self._positions[symbol] = pos_data
        self._save_positions()

        self._listings_traded += 1

        # Firebase
        try:
            position = Position(
                symbol=symbol,
                side=OrderSide.BUY,
                entry_price=entry_price,
                sl_price=oco.sl_price,
                size=qty_float,
                venue_order_id=order_id,
                status=PositionStatus.OPEN,
                strategy=StrategyType.LISTING,
                tp_price=oco.tp_price,
            )
            fb_id = log_trade_opened(
                position=position,
                exchange=EXCHANGE_NAME,
                account_equity=equity,
                dry_run=self.dry_run,
            )
            pos_data["firebase_trade_id"] = fb_id
            self._save_positions()
        except Exception:
            logger.exception("Firebase log_trade_opened échoué pour %s", symbol)

        # Notification Telegram — ordre placé
        base = symbol.replace("USDC", "")
        self._telegram.send_raw(
            f"🆕🛒 *LISTING BUY — {symbol}*\n"
            f"  Momentum: `+{momentum_pct*100:.1f}%` ✅\n"
            f"  Entrée: `{_fmt(entry_price)}` | Size: `{qty_str} {base}` (`${alloc_usd:.0f}`)\n"
            f"  🎯 TP1: `{_fmt(oco.tp_price)}` (`+{config.LISTING_TP_INIT_PCT*100:.0f}%`)\n"
            f"  🛑 SL: `{_fmt(oco.sl_price)}` (`-{config.LISTING_SL_INIT_PCT*100:.0f}%`)\n"
            f"  Horizon: {config.LISTING_HORIZON_DAYS}j | "
            f"Slots: {len(self._positions)}/{config.LISTING_MAX_SLOTS}"
        )

    def _place_oco_sell(
        self,
        symbol: str,
        qty_str: str,
        entry_price: float,
        sl_price: float,
        tp_price: float,
    ) -> int | None:
        """Place un OCO SELL (SL + TP) pour protéger une position."""
        if self.dry_run:
            logger.info(
                "  [DRY RUN] OCO SELL: SL=%s TP=%s", _fmt(sl_price), _fmt(tp_price),
            )
            return None

        try:
            sl_stop_str = self._client.format_price(symbol, sl_price)
            # SL limit légèrement en-dessous du stop pour garantir le fill
            sl_limit = sl_price * (1 - config.BINANCE_SL_LIMIT_OFFSET_PCT)
            sl_limit_str = self._client.format_price(symbol, sl_limit)
            tp_str = self._client.format_price(symbol, tp_price)

            resp = self._client.place_oco_order(
                symbol=symbol,
                side="SELL",
                quantity=qty_str,
                tp_price=tp_str,
                sl_stop_price=sl_stop_str,
                sl_limit_price=sl_limit_str,
            )
            oco_list_id = resp.get("orderListId")
            logger.info(
                "🎯 %s: OCO placé — SL=%s TP=%s (listId=%s)",
                symbol, sl_stop_str, tp_str, oco_list_id,
            )
            return oco_list_id
        except Exception as e:
            logger.error("❌ %s: OCO échoué: %s", symbol, e)
            self._telegram.notify_error(f"ListingBot: OCO {symbol} échoué — {e}")
            return None

    # ═══════════════════════════════════════════════════════════════════════════
    #  2. GESTION DES POSITIONS OUVERTES
    # ═══════════════════════════════════════════════════════════════════════════

    def _manage_open_positions(self) -> None:
        """Vérifie OCO status, re-arm, force close pour chaque position."""
        if not self._positions:
            return

        closed_symbols: list[str] = []
        now_ms = int(time.time() * 1000)

        for symbol, pos in list(self._positions.items()):
            try:
                self._manage_single_position(symbol, pos, now_ms, closed_symbols)
            except Exception:
                logger.exception("Erreur gestion position %s", symbol)

        # Nettoyer les positions fermées
        for sym in closed_symbols:
            self._positions.pop(sym, None)

        if closed_symbols:
            self._save_positions()

    def _manage_single_position(
        self,
        symbol: str,
        pos: dict,
        now_ms: int,
        closed_symbols: list[str],
    ) -> None:
        """Gère une seule position (check OCO, re-arm, horizon)."""

        oco_list_id = pos.get("oco_list_id")

        # ── 1. Vérifier si l'OCO a été rempli ──
        if oco_list_id is not None and not self.dry_run:
            try:
                oco_info = self._client.get_order_list(oco_list_id)
                list_status = oco_info.get("listOrderStatus", "")

                if list_status == "ALL_DONE":
                    # Un des deux ordres a été rempli
                    self._handle_oco_filled(symbol, pos, oco_info, closed_symbols)
                    return

            except Exception as e:
                logger.warning("%s: erreur check OCO (listId=%s): %s", symbol, oco_list_id, e)

        # ── 2. Check re-arm OCO (si pas encore re-armed) ──
        if not pos.get("oco_rearmed", False):
            try:
                current_price = self._client.get_ticker_price(symbol)
                tp1 = pos.get("tp1_price", pos.get("tp_price", 0))

                if tp1 > 0 and should_rearm_oco(
                    current_price=current_price,
                    tp_price=tp1,
                    tp_near_ratio=config.LISTING_TP_NEAR_RATIO,
                ):
                    self._rearm_oco(symbol, pos)
            except Exception as e:
                logger.warning("%s: erreur check re-arm: %s", symbol, e)

        # ── 3. Force close si horizon dépassé ──
        entry_ts = pos.get("entry_ts", 0)
        if entry_ts > 0 and should_force_close(
            entry_ts_ms=entry_ts,
            current_ts_ms=now_ms,
            horizon_days=config.LISTING_HORIZON_DAYS,
        ):
            logger.info(
                "⏰ %s: horizon de %d jours dépassé → force close",
                symbol, config.LISTING_HORIZON_DAYS,
            )
            self._force_close_position(symbol, pos, "HORIZON", closed_symbols)

    def _handle_oco_filled(
        self,
        symbol: str,
        pos: dict,
        oco_info: dict,
        closed_symbols: list[str],
    ) -> None:
        """Traite un OCO rempli (SL ou TP hit)."""
        orders = oco_info.get("orders", [])
        filled_order = None
        exit_reason = "UNKNOWN"

        for order in orders:
            status = order.get("status", "")
            if status == "FILLED":
                filled_order = order
                order_type = order.get("type", "")
                if order_type in ("STOP_LOSS_LIMIT", "STOP_LOSS"):
                    exit_reason = "SL"
                elif order_type in ("LIMIT_MAKER", "LIMIT", "TAKE_PROFIT_LIMIT"):
                    exit_reason = "TP"
                break

        # Récupérer le prix de sortie
        if filled_order:
            exit_price = float(filled_order.get("price", 0))
            if exit_price == 0:
                # Fallback: prix courant
                try:
                    exit_price = self._client.get_ticker_price(symbol)
                except Exception:
                    exit_price = pos.get("entry_price", 0)
        else:
            exit_price = pos.get("entry_price", 0)

        entry_price = pos.get("entry_price", 0)
        size = pos.get("size", 0)
        pnl_pct = (exit_price - entry_price) / entry_price * 100 if entry_price > 0 else 0
        pnl_usd = (exit_price - entry_price) * size

        oco_label = "OCO2" if pos.get("oco_rearmed") else "OCO1"
        emoji = "💰" if pnl_usd >= 0 else "💸"

        logger.info(
            "%s %s: %s hit @ %s | PnL: %+.2f USD (%+.1f%%) [%s]",
            emoji, symbol, exit_reason, _fmt(exit_price), pnl_usd, pnl_pct, oco_label,
        )

        # Firebase
        try:
            position = Position(
                symbol=symbol,
                side=OrderSide.BUY,
                entry_price=entry_price,
                sl_price=pos.get("sl_price", 0),
                size=size,
                venue_order_id=pos.get("order_id", ""),
                status=PositionStatus.CLOSED,
                strategy=StrategyType.LISTING,
                tp_price=pos.get("tp_price", 0),
                pnl=pnl_usd,
                firebase_trade_id=pos.get("firebase_trade_id"),
            )
            log_trade_closed(
                position=position,
                exit_price=exit_price,
                reason=exit_reason,
                exchange=EXCHANGE_NAME,
            )
        except Exception:
            logger.exception("Firebase log_trade_closed échoué pour %s", symbol)

        # Telegram
        base = symbol.replace("USDC", "")
        self._telegram.send_raw(
            f"{emoji} *LISTING {exit_reason} — {symbol}* [{oco_label}]\n"
            f"  Entrée: `{_fmt(entry_price)}` → Sortie: `{_fmt(exit_price)}`\n"
            f"  P&L: `{pnl_usd:+.2f} USD` (`{pnl_pct:+.1f}%`)\n"
            f"  Size: `{size:.8f} {base}`"
        )

        closed_symbols.append(symbol)

    def _rearm_oco(self, symbol: str, pos: dict) -> None:
        """Cancel l'OCO actuel et place un nouvel OCO avec SL2/TP2."""
        entry_price = pos.get("entry_price", 0)
        tp1_price = pos.get("tp1_price", 0)

        if tp1_price <= 0 or entry_price <= 0:
            return

        rearm = compute_rearm_oco_levels(
            entry_price=entry_price,
            tp1_price=tp1_price,
            sl2_tp1_mult=config.LISTING_SL2_TP1_MULT,
            tp2_tp1_mult=config.LISTING_TP2_TP1_MULT,
        )

        logger.info(
            "🔄 %s: RE-ARM OCO → SL2=%s (%+.1f%% vs entry) TP2=%s (%+.1f%% vs entry)",
            symbol,
            _fmt(rearm.sl_price), rearm.sl_pct_vs_entry * 100,
            _fmt(rearm.tp_price), rearm.tp_pct_vs_entry * 100,
        )

        # Cancel l'OCO existant
        old_oco_id = pos.get("oco_list_id")
        if old_oco_id is not None and not self.dry_run:
            try:
                self._client.cancel_order_list(symbol=symbol, order_list_id=old_oco_id)
                logger.info("  OCO %s annulé", old_oco_id)
            except Exception as e:
                logger.warning("  Erreur annulation OCO %s: %s", old_oco_id, e)

        # Placer le nouvel OCO
        qty_str = pos.get("size_str", str(pos.get("size", 0)))
        new_oco_id = self._place_oco_sell(
            symbol=symbol,
            qty_str=qty_str,
            entry_price=entry_price,
            sl_price=rearm.sl_price,
            tp_price=rearm.tp_price,
        )

        # Mettre à jour la position
        pos["sl_price"] = rearm.sl_price
        pos["tp_price"] = rearm.tp_price
        pos["oco_list_id"] = new_oco_id
        pos["oco_rearmed"] = True
        self._save_positions()

        # Telegram
        self._telegram.send_raw(
            f"🔄 *OCO RE-ARM — {symbol}*\n"
            f"  SL2: `{_fmt(rearm.sl_price)}` (`{rearm.sl_pct_vs_entry*100:+.1f}%` vs entry)\n"
            f"  TP2: `{_fmt(rearm.tp_price)}` (`{rearm.tp_pct_vs_entry*100:+.1f}%` vs entry)\n"
            f"  Profit verrouillé ≥ `{rearm.sl_pct_vs_entry*100:+.1f}%` 🔒"
        )

    def _force_close_position(
        self,
        symbol: str,
        pos: dict,
        reason: str,
        closed_symbols: list[str],
    ) -> None:
        """Force close une position (horizon dépassé)."""
        # Cancel OCO existant
        oco_id = pos.get("oco_list_id")
        if oco_id is not None and not self.dry_run:
            try:
                self._client.cancel_order_list(symbol=symbol, order_list_id=oco_id)
            except Exception:
                pass

        # Market sell
        qty_str = pos.get("size_str", str(pos.get("size", 0)))
        entry_price = pos.get("entry_price", 0)

        if self.dry_run:
            exit_price = entry_price
            logger.info("  [DRY RUN] Force close %s", symbol)
        else:
            try:
                resp = self._client.place_market_order(
                    symbol=symbol,
                    side="SELL",
                    quantity=qty_str,
                )
                fills = resp.get("fills", [])
                if fills:
                    total_qty = sum(float(f["qty"]) for f in fills)
                    total_cost = sum(float(f["qty"]) * float(f["price"]) for f in fills)
                    exit_price = total_cost / total_qty if total_qty > 0 else entry_price
                else:
                    exit_price = entry_price
            except Exception as e:
                logger.error("❌ %s: force close échoué: %s", symbol, e)
                self._telegram.notify_error(f"ListingBot: force close {symbol} échoué — {e}")
                return

        size = pos.get("size", 0)
        pnl_usd = (exit_price - entry_price) * size
        pnl_pct = (exit_price - entry_price) / entry_price * 100 if entry_price > 0 else 0
        emoji = "💰" if pnl_usd >= 0 else "💸"

        logger.info(
            "%s %s: FORCE CLOSE (%s) @ %s | PnL: %+.2f USD (%+.1f%%)",
            emoji, symbol, reason, _fmt(exit_price), pnl_usd, pnl_pct,
        )

        # Firebase
        try:
            position = Position(
                symbol=symbol,
                side=OrderSide.BUY,
                entry_price=entry_price,
                sl_price=pos.get("sl_price", 0),
                size=size,
                venue_order_id=pos.get("order_id", ""),
                status=PositionStatus.CLOSED,
                strategy=StrategyType.LISTING,
                tp_price=pos.get("tp_price", 0),
                pnl=pnl_usd,
                firebase_trade_id=pos.get("firebase_trade_id"),
            )
            log_trade_closed(
                position=position,
                exit_price=exit_price,
                reason=reason,
                exchange=EXCHANGE_NAME,
            )
        except Exception:
            logger.exception("Firebase log_trade_closed échoué pour %s", symbol)

        # Telegram
        base = symbol.replace("USDC", "")
        self._telegram.send_raw(
            f"{emoji} *LISTING {reason} — {symbol}*\n"
            f"  Entrée: `{_fmt(entry_price)}` → Sortie: `{_fmt(exit_price)}`\n"
            f"  P&L: `{pnl_usd:+.2f} USD` (`{pnl_pct:+.1f}%`)\n"
            f"  Size: `{size:.8f} {base}`\n"
            f"  ⏰ Force close après {config.LISTING_HORIZON_DAYS}j"
        )

        closed_symbols.append(symbol)

    # ═══════════════════════════════════════════════════════════════════════════
    #  HEARTBEAT
    # ═══════════════════════════════════════════════════════════════════════════

    def _send_heartbeat(self) -> None:
        """Envoie un heartbeat Telegram + Firebase."""
        equity = self._get_listing_equity()
        n_positions = len(self._positions)
        now_utc = datetime.now(timezone.utc).strftime("%H:%M UTC")

        # Positions détail
        pos_lines = []
        total_unrealized = 0.0
        for symbol, pos in self._positions.items():
            entry = pos.get("entry_price", 0)
            try:
                current = self._client.get_ticker_price(symbol)
            except Exception:
                current = entry
            pnl_pct = (current - entry) / entry * 100 if entry > 0 else 0
            unrealized = (current - entry) * pos.get("size", 0)
            total_unrealized += unrealized
            emoji = "🟢" if pnl_pct >= 0 else "🔴"
            oco_label = "OCO2" if pos.get("oco_rearmed") else "OCO1"
            age_h = (int(time.time() * 1000) - pos.get("entry_ts", 0)) / 3600000
            pos_lines.append(
                f"  {emoji} `{symbol}` {oco_label} (`{pnl_pct:+.1f}%`) — {age_h:.0f}h"
            )

        lines = [
            f"💓 *LISTING Heartbeat* (cycle #{self._cycle_count})",
            f"  💰 Equity: `${equity:,.0f}`",
            f"  📊 Positions: {n_positions}/{config.LISTING_MAX_SLOTS}",
            f"  🆕 Détectés: {self._listings_detected} | "
            f"Tradés: {self._listings_traded} | "
            f"Skip: {self._listings_skipped_momentum}",
        ]

        if pos_lines:
            lines.append(f"  💵 PnL open: `${total_unrealized:+.2f}`")
            lines.extend(pos_lines)

        lines.append(f"  🕐 `{now_utc}`")
        self._telegram.send_raw("\n".join(lines))

        # Firebase heartbeat
        try:
            fb_log_heartbeat(
                exchange=EXCHANGE_NAME,
                total_equity=equity,
                total_risk_pct=0.0,
                pairs_count=len(self._known_symbols),
                open_positions=n_positions,
                dry_run=self.dry_run,
            )
        except Exception:
            logger.exception("Firebase heartbeat échoué")

    # ═══════════════════════════════════════════════════════════════════════════
    #  CAPITAL & BALANCE
    # ═══════════════════════════════════════════════════════════════════════════

    def _get_usdc_balance(self) -> float:
        """Retourne le solde USDC disponible."""
        try:
            balances = self._client.get_balances()
            for b in balances:
                if b.currency == "USDC":
                    return b.available
        except Exception as e:
            logger.warning("Erreur récupération balance: %s", e)
        return 0.0

    def _get_listing_equity(self) -> float:
        """Retourne l'equity allouée au listing bot via l'allocator dynamique.

        En mode dynamique : equity Binance live × listing_pct.
        En fallback : base allouée + PnL ouvert.
        """
        if config.DYNAMIC_ALLOCATION_ENABLED:
            try:
                total_equity = self._calculate_equity()
                pct = min(max(self._allocation_pct, 0.0), 1.0)
                return max(total_equity * pct, 0.0)
            except Exception:
                pass
        # Fallback : capital alloué statique + PnL ouvert
        base = self._allocated_balance
        for pos in self._positions.values():
            entry = pos.get("entry_price", 0)
            size = pos.get("size", 0)
            try:
                current = self._client.get_ticker_price(pos.get("symbol", ""))
                pnl = (current - entry) * size
                base += pnl
            except Exception:
                pass
        return base

    def _calculate_equity(self) -> float:
        """Calcule l'equity totale Binance (USDC + valeur des positions de TOUS les bots)."""
        try:
            balances = self._client.get_balances()
            total = 0.0
            for b in balances:
                if b.currency == "USDC":
                    total += b.available + b.reserved
            # Ajouter la valeur des positions listing en cours
            for pos in self._positions.values():
                size = pos.get("size", 0)
                try:
                    price = self._client.get_ticker_price(pos.get("symbol", ""))
                    total += size * price
                except Exception:
                    entry = pos.get("entry_price", 0)
                    total += size * entry
            return total
        except Exception as e:
            logger.warning("Erreur calcul equity: %s", e)
            return self._allocated_balance

    # ---- Dynamic allocation ------------------------------------------------

    def _init_allocation(self) -> None:
        """Calcule l'allocation dynamique Crash/Trail/Listing au démarrage.

        Si DYNAMIC_ALLOCATION_ENABLED=false → garde le fallback statique.
        Sinon → requête Firebase pour le PF Trail Range 90j → compute_allocation().
        """
        if not config.DYNAMIC_ALLOCATION_ENABLED:
            logger.info(
                "📊 Allocation dynamique désactivée — fallback statique $%.0f",
                self._allocated_balance,
            )
            self._last_allocation_date = datetime.now(timezone.utc).strftime("%Y-%m-%d")
            return

        try:
            # Solde réel Binance (USDC + valeur des positions)
            total_balance = self._calculate_equity()
            logger.info("💰 Solde total Binance: $%.2f", total_balance)

            pnl_list = fb_get_trail_range_pnl_list(days=90)
            trail_pf = compute_profit_factor(pnl_list)
            trail_trades = len(pnl_list)

            result = compute_allocation(
                total_balance=total_balance,
                trail_pf=trail_pf,
                trail_trade_count=trail_trades,
            )

            self._allocated_balance = result.listing_balance
            self._allocation_pct = result.listing_pct
            self._last_allocation_date = datetime.now(timezone.utc).strftime("%Y-%m-%d")

            # Persister dans Firebase pour le dashboard
            fb_log_allocation(
                regime=result.regime.value,
                crash_pct=result.crash_pct,
                trail_pct=result.trail_pct,
                listing_pct=result.listing_pct,
                crash_balance=result.crash_balance,
                trail_balance=result.trail_balance,
                listing_balance=result.listing_balance,
                total_balance=total_balance,
                trail_pf=trail_pf,
                trail_trades=trail_trades,
                reason=result.reason,
            )

            logger.info("═" * 50)
            logger.info("📊 ALLOCATION DYNAMIQUE — %s", result.regime.value.upper())
            logger.info("   %s", result.reason)
            logger.info(
                "   Total: $%.0f | Listing: %.0f%% → $%.0f | Crash: %.0f%% → $%.0f | Trail: %.0f%% → $%.0f",
                total_balance,
                result.listing_pct * 100,
                result.listing_balance,
                result.crash_pct * 100,
                result.crash_balance,
                result.trail_pct * 100,
                result.trail_balance,
            )
            logger.info("═" * 50)

        except Exception as e:
            logger.warning(
                "⚠️ Allocation dynamique échouée: %s — fallback $%.0f",
                e,
                self._allocated_balance,
            )
            self._last_allocation_date = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    def _maybe_recompute_allocation(self) -> None:
        """Recalcule l'allocation 1×/jour (changement de date UTC)."""
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        if today == self._last_allocation_date:
            return
        logger.info("🔄 Nouveau jour UTC — recalcul allocation dynamique")
        self._init_allocation()

    # ═══════════════════════════════════════════════════════════════════════════
    #  PERSISTANCE
    # ═══════════════════════════════════════════════════════════════════════════

    def _load_known_symbols(self) -> None:
        """Charge les symboles connus depuis le fichier JSON."""
        try:
            if os.path.exists(_KNOWN_SYMBOLS_FILE):
                with open(_KNOWN_SYMBOLS_FILE) as f:
                    data = _json.load(f)
                self._known_symbols = set(data.get("symbols", []))
                logger.info("Symboles connus chargés: %d", len(self._known_symbols))
        except Exception:
            logger.exception("Erreur chargement known_symbols")
            self._known_symbols = set()

    def _save_known_symbols(self) -> None:
        """Persiste les symboles connus."""
        try:
            os.makedirs(os.path.dirname(_KNOWN_SYMBOLS_FILE), exist_ok=True)
            with open(_KNOWN_SYMBOLS_FILE, "w") as f:
                _json.dump({"symbols": sorted(self._known_symbols)}, f, indent=2)
        except Exception:
            logger.exception("Erreur sauvegarde known_symbols")

    def _load_skipped_symbols(self) -> None:
        """Charge les symboles skippés depuis le fichier JSON."""
        try:
            if os.path.exists(_SKIPPED_SYMBOLS_FILE):
                with open(_SKIPPED_SYMBOLS_FILE) as f:
                    data = _json.load(f)
                self._skipped_symbols = set(data.get("symbols", []))
                logger.info("Symboles skippés chargés: %d", len(self._skipped_symbols))
        except Exception:
            logger.exception("Erreur chargement skipped_symbols")
            self._skipped_symbols = set()

    def _save_skipped_symbols(self) -> None:
        """Persiste les symboles skippés."""
        try:
            os.makedirs(os.path.dirname(_SKIPPED_SYMBOLS_FILE), exist_ok=True)
            with open(_SKIPPED_SYMBOLS_FILE, "w") as f:
                _json.dump({"symbols": sorted(self._skipped_symbols)}, f, indent=2)
        except Exception:
            logger.exception("Erreur sauvegarde skipped_symbols")

    def _load_positions(self) -> None:
        """Charge les positions ouvertes depuis le fichier JSON d'état."""
        try:
            state_path = self._store._path
            if not state_path.exists():
                logger.info("Pas de fichier d'état listing — démarrage à vide")
                self._positions = {}
                return
            with open(state_path, "r") as f:
                state = _json.load(f)
            self._positions = state.get("positions", {})
            logger.info("Positions chargées: %d", len(self._positions))
        except Exception:
            logger.exception("Erreur chargement positions")
            self._positions = {}

    def _save_positions(self) -> None:
        """Persiste les positions ouvertes."""
        try:
            self._store.save({"positions": self._positions})
        except Exception:
            logger.exception("Erreur sauvegarde positions")


# ═══════════════════════════════════════════════════════════════════════════════
#  CLI
# ═══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    parser = argparse.ArgumentParser(description="TradeX Binance ListingBot")
    parser.add_argument("--dry-run", action="store_true", help="Mode simulation")
    args = parser.parse_args()

    bot = TradeXBinanceListingBot(dry_run=args.dry_run)
    bot.start()


if __name__ == "__main__":
    main()
