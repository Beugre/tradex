"""Tests pour la réconciliation des positions au démarrage (bot._reconcile_positions)."""

from __future__ import annotations

import json
import tempfile
from unittest.mock import MagicMock, patch

import pytest

from src.core.models import (
    Balance,
    OrderSide,
    Position,
    PositionStatus,
    RangeState,
    StrategyType,
    TickerData,
)
from src.core.position_store import PositionStore


# ── Helpers ────────────────────────────────────────────────────────────────────


def _make_position(
    symbol: str = "BTC-USD",
    side: OrderSide = OrderSide.BUY,
    entry_price: float = 95000.0,
    sl_price: float = 92000.0,
    size: float = 0.001,
    venue_order_id: str = "abc-123",
    status: PositionStatus = PositionStatus.OPEN,
    strategy: StrategyType = StrategyType.TREND,
) -> Position:
    return Position(
        symbol=symbol,
        side=side,
        entry_price=entry_price,
        sl_price=sl_price,
        size=size,
        venue_order_id=venue_order_id,
        status=status,
        strategy=strategy,
    )


def _make_balance(currency: str, available: float, reserved: float = 0.0) -> Balance:
    return Balance(
        currency=currency,
        available=available,
        reserved=reserved,
        total=available + reserved,
    )


def _make_range(symbol: str = "XRP-USD") -> RangeState:
    return RangeState(symbol=symbol, range_high=0.65, range_low=0.55)


def _build_bot(state_file: str, positions: dict | None = None, ranges: dict | None = None):
    """Construit un TradeXBot mocké avec un state.json pré-rempli."""
    # Sauvegarder l'état local initial
    store = PositionStore(state_file=state_file)
    store.save(positions or {}, ranges or {})

    # Patcher tout l'I/O pour isoler la logique de réconciliation
    with patch("src.bot.RevolutXClient") as MockClient, \
         patch("src.bot.DataProvider"), \
         patch("src.bot.TelegramNotifier") as MockTelegram, \
         patch("src.bot.config") as mock_config:

        # Config minimale
        mock_config.REVOLUT_X_API_KEY = "fake"
        mock_config.REVOLUT_X_PRIVATE_KEY_PATH = "/tmp/fake.pem"
        mock_config.TELEGRAM_BOT_TOKEN = ""
        mock_config.TELEGRAM_CHAT_ID = ""
        mock_config.TRADING_PAIRS = ["BTC-USD", "SOL-USD", "XRP-USD"]
        mock_config.LOG_LEVEL = "WARNING"

        from src.bot import TradeXBot

        bot = TradeXBot.__new__(TradeXBot)
        bot.dry_run = False
        bot._running = False
        bot._client = MockClient.return_value
        bot._data = MagicMock()
        bot._data.get_ticker.return_value = None  # Pas de reconstruction par défaut
        bot._telegram = MockTelegram.return_value
        bot._store = store
        bot._trends = {}
        bot._ranges = {}
        bot._positions = {}
        bot._last_heartbeat_time = 0.0
        bot._cycle_count = 0
        bot._prev_state = {}
        bot._prev_ignored = {}

    return bot


# ── Tests ──────────────────────────────────────────────────────────────────────


class TestReconcileBuyPosition:
    """Réconciliation de positions BUY (on devrait détenir le crypto)."""

    def test_buy_confirmed_by_balance(self, tmp_path):
        """BUY position + solde suffisant → position confirmée."""
        pos = _make_position(size=0.001)
        state_file = str(tmp_path / "state.json")
        bot = _build_bot(state_file, positions={"BTC-USD": pos})

        bot._client.get_balances.return_value = [
            _make_balance("BTC", 0.001),
            _make_balance("USD", 1000.0),
        ]

        bot._reconcile_positions()

        assert "BTC-USD" in bot._positions
        assert bot._positions["BTC-USD"].entry_price == 95000.0
        assert bot._positions["BTC-USD"].status == PositionStatus.OPEN

    def test_buy_confirmed_with_fees_tolerance(self, tmp_path):
        """BUY position + solde un peu inférieur (frais) → confirmée (tolérance 10%)."""
        pos = _make_position(size=0.001)
        state_file = str(tmp_path / "state.json")
        bot = _build_bot(state_file, positions={"BTC-USD": pos})

        # Solde = 91% de la taille → au-dessus du seuil de 90%
        bot._client.get_balances.return_value = [
            _make_balance("BTC", 0.00091),
            _make_balance("USD", 1000.0),
        ]

        bot._reconcile_positions()

        assert "BTC-USD" in bot._positions

    def test_buy_removed_insufficient_balance(self, tmp_path):
        """BUY position mais solde trop faible → retirée."""
        pos = _make_position(size=0.001)
        state_file = str(tmp_path / "state.json")
        bot = _build_bot(state_file, positions={"BTC-USD": pos})

        # Solde = 50% de la taille → en dessous du seuil
        bot._client.get_balances.return_value = [
            _make_balance("BTC", 0.0005),
            _make_balance("USD", 1000.0),
        ]

        bot._reconcile_positions()

        assert "BTC-USD" not in bot._positions

    def test_buy_removed_zero_balance(self, tmp_path):
        """BUY position mais solde zéro → retirée (vendue externement)."""
        pos = _make_position(size=0.001)
        state_file = str(tmp_path / "state.json")
        bot = _build_bot(state_file, positions={"BTC-USD": pos})

        bot._client.get_balances.return_value = [
            _make_balance("USD", 1000.0),
        ]

        bot._reconcile_positions()

        assert "BTC-USD" not in bot._positions

    def test_buy_reserved_counted(self, tmp_path):
        """BUY position + solde réparti entre available et reserved → confirmée."""
        pos = _make_position(size=0.001)
        state_file = str(tmp_path / "state.json")
        bot = _build_bot(state_file, positions={"BTC-USD": pos})

        bot._client.get_balances.return_value = [
            _make_balance("BTC", available=0.0005, reserved=0.0005),
            _make_balance("USD", 1000.0),
        ]

        bot._reconcile_positions()

        assert "BTC-USD" in bot._positions


class TestReconcileSellPosition:
    """Réconciliation de positions SELL (on a vendu, solde base devrait être faible)."""

    def test_sell_confirmed_low_balance(self, tmp_path):
        """SELL position + solde base faible → confirmée."""
        pos = _make_position(
            symbol="SOL-USD",
            side=OrderSide.SELL,
            entry_price=150.0,
            sl_price=160.0,
            size=10.0,
        )
        state_file = str(tmp_path / "state.json")
        bot = _build_bot(state_file, positions={"SOL-USD": pos})

        bot._client.get_balances.return_value = [
            _make_balance("SOL", 0.0),  # Vendu
            _make_balance("USD", 2500.0),
        ]

        bot._reconcile_positions()

        assert "SOL-USD" in bot._positions
        assert bot._positions["SOL-USD"].side == OrderSide.SELL

    def test_sell_removed_high_balance(self, tmp_path):
        """SELL position mais solde base élevé → retirée (déjà rachetée)."""
        pos = _make_position(
            symbol="SOL-USD",
            side=OrderSide.SELL,
            entry_price=150.0,
            sl_price=160.0,
            size=10.0,
        )
        state_file = str(tmp_path / "state.json")
        bot = _build_bot(state_file, positions={"SOL-USD": pos})

        bot._client.get_balances.return_value = [
            _make_balance("SOL", 10.0),  # On détient à nouveau → racheté
            _make_balance("USD", 1000.0),
        ]

        bot._reconcile_positions()

        assert "SOL-USD" not in bot._positions


class TestReconcileClosedPositions:
    """Les positions CLOSED dans state.json sont ignorées."""

    def test_closed_position_not_restored(self, tmp_path):
        """Position CLOSED dans state.json → pas restaurée même si on a du solde."""
        pos = _make_position(status=PositionStatus.CLOSED)
        state_file = str(tmp_path / "state.json")
        bot = _build_bot(state_file, positions={"BTC-USD": pos})

        bot._client.get_balances.return_value = [
            _make_balance("BTC", 0.001),
            _make_balance("USD", 1000.0),
        ]

        bot._reconcile_positions()

        assert "BTC-USD" not in bot._positions


class TestReconcileOrphans:
    """Reconstruction des positions orphelines (crypto détenu sans position trackée)."""

    def test_orphan_reconstructed_as_position(self, tmp_path):
        """Crypto détenu sans position locale → position reconstruite avec prix actuel."""
        state_file = str(tmp_path / "state.json")
        bot = _build_bot(state_file, positions={})

        bot._client.get_balances.return_value = [
            _make_balance("BTC", 0.002),
            _make_balance("USD", 1000.0),
        ]
        bot._data.get_ticker.return_value = TickerData(
            symbol="BTC-USD", bid=94000.0, ask=94100.0, mid=94050.0, last_price=95000.0,
        )

        bot._reconcile_positions()

        assert "BTC-USD" in bot._positions
        pos = bot._positions["BTC-USD"]
        assert pos.side == OrderSide.BUY
        assert pos.entry_price == 95000.0
        assert pos.sl_price == pytest.approx(95000.0 * 0.95, rel=1e-6)
        assert pos.size == 0.002
        assert pos.venue_order_id == "recovered"
        assert pos.status == PositionStatus.OPEN
        assert pos.strategy == StrategyType.TREND

    def test_orphan_not_reconstructed_no_ticker(self, tmp_path):
        """Crypto détenu mais ticker indisponible → reste orphelin (pas de position)."""
        state_file = str(tmp_path / "state.json")
        bot = _build_bot(state_file, positions={})

        bot._client.get_balances.return_value = [
            _make_balance("BTC", 0.002),
            _make_balance("USD", 1000.0),
        ]
        bot._data.get_ticker.return_value = None

        bot._reconcile_positions()

        assert "BTC-USD" not in bot._positions
        # Orphelin signalé dans la notification
        call_args = bot._telegram.notify_reconciliation.call_args[0]
        orphans = call_args[2]
        assert len(orphans) == 1
        assert "BTC" in orphans[0]

    def test_orphan_not_reconstructed_api_error(self, tmp_path):
        """Ticker lève une exception → orphelin logué avec l'erreur."""
        state_file = str(tmp_path / "state.json")
        bot = _build_bot(state_file, positions={})

        bot._client.get_balances.return_value = [
            _make_balance("BTC", 0.002),
            _make_balance("USD", 1000.0),
        ]
        bot._data.get_ticker.side_effect = Exception("API timeout")

        bot._reconcile_positions()

        assert "BTC-USD" not in bot._positions
        call_args = bot._telegram.notify_reconciliation.call_args[0]
        orphans = call_args[2]
        assert len(orphans) == 1
        assert "erreur" in orphans[0]

    def test_orphan_recovered_notification_includes_position(self, tmp_path):
        """La notification contient la position reconstruite."""
        state_file = str(tmp_path / "state.json")
        bot = _build_bot(state_file, positions={})

        bot._client.get_balances.return_value = [
            _make_balance("SOL", 15.5),
            _make_balance("USD", 1000.0),
        ]
        bot._data.get_ticker.return_value = TickerData(
            symbol="SOL-USD", bid=149.0, ask=151.0, mid=150.0, last_price=150.0,
        )

        bot._reconcile_positions()

        call_args = bot._telegram.notify_reconciliation.call_args[0]
        recovered = call_args[3]
        assert len(recovered) == 1
        assert recovered[0].symbol == "SOL-USD"
        assert recovered[0].size == 15.5

    def test_no_orphan_for_non_trading_pair(self, tmp_path):
        """Crypto détenu mais pas dans TRADING_PAIRS → pas d'orphelin."""
        state_file = str(tmp_path / "state.json")
        bot = _build_bot(state_file, positions={})

        bot._client.get_balances.return_value = [
            _make_balance("DOGE", 1000.0),  # DOGE pas dans TRADING_PAIRS
            _make_balance("USD", 1000.0),
        ]

        bot._reconcile_positions()

        call_args = bot._telegram.notify_reconciliation.call_args[0]
        orphans = call_args[2]
        recovered = call_args[3]
        assert len(orphans) == 0
        assert len(recovered) == 0

    def test_no_orphan_for_fiat(self, tmp_path):
        """USD/EUR ne sont pas des orphelins."""
        state_file = str(tmp_path / "state.json")
        bot = _build_bot(state_file, positions={})

        bot._client.get_balances.return_value = [
            _make_balance("USD", 1000.0),
            _make_balance("EUR", 500.0),
        ]

        bot._reconcile_positions()

        call_args = bot._telegram.notify_reconciliation.call_args[0]
        orphans = call_args[2]
        recovered = call_args[3]
        assert len(orphans) == 0
        assert len(recovered) == 0


class TestReconcileAPIFailure:
    """Repli sur l'état local si l'exchange est injoignable."""

    def test_fallback_to_local_on_api_error(self, tmp_path):
        """API KO → positions restaurées depuis le fichier local."""
        pos = _make_position(size=0.001)
        state_file = str(tmp_path / "state.json")
        bot = _build_bot(state_file, positions={"BTC-USD": pos})

        bot._client.get_balances.side_effect = Exception("Connection refused")

        bot._reconcile_positions()

        # Position quand même restaurée (fallback local)
        assert "BTC-USD" in bot._positions
        assert bot._positions["BTC-USD"].entry_price == 95000.0


class TestReconcileRanges:
    """Les ranges sont restaurés normalement."""

    def test_ranges_restored(self, tmp_path):
        """Les ranges du state.json sont chargés indépendamment de l'API."""
        rs = _make_range("XRP-USD")
        state_file = str(tmp_path / "state.json")
        bot = _build_bot(state_file, ranges={"XRP-USD": rs})

        bot._client.get_balances.return_value = [
            _make_balance("USD", 1000.0),
        ]

        bot._reconcile_positions()

        assert "XRP-USD" in bot._ranges
        assert bot._ranges["XRP-USD"].range_high == 0.65


class TestReconcileMultiplePositions:
    """Réconciliation avec plusieurs positions simultanées."""

    def test_mixed_confirmed_and_removed(self, tmp_path):
        """2 positions : une confirmée, une retirée."""
        pos_btc = _make_position(symbol="BTC-USD", size=0.001)
        pos_sol = _make_position(
            symbol="SOL-USD",
            side=OrderSide.BUY,
            entry_price=150.0,
            sl_price=140.0,
            size=10.0,
        )
        state_file = str(tmp_path / "state.json")
        bot = _build_bot(
            state_file,
            positions={"BTC-USD": pos_btc, "SOL-USD": pos_sol},
        )

        bot._client.get_balances.return_value = [
            _make_balance("BTC", 0.001),   # Confirmée
            _make_balance("SOL", 0.0),     # Plus rien → retirée
            _make_balance("USD", 1000.0),
        ]

        bot._reconcile_positions()

        assert "BTC-USD" in bot._positions
        assert "SOL-USD" not in bot._positions

    def test_zero_risk_position_reconciled(self, tmp_path):
        """Position ZERO_RISK est aussi réconciliée (pas seulement OPEN)."""
        pos = _make_position(status=PositionStatus.ZERO_RISK, size=0.001)
        state_file = str(tmp_path / "state.json")
        bot = _build_bot(state_file, positions={"BTC-USD": pos})

        bot._client.get_balances.return_value = [
            _make_balance("BTC", 0.001),
            _make_balance("USD", 1000.0),
        ]

        bot._reconcile_positions()

        assert "BTC-USD" in bot._positions
        assert bot._positions["BTC-USD"].status == PositionStatus.ZERO_RISK


class TestReconcileEmptyState:
    """Réconciliation avec un state.json vide."""

    def test_empty_state_no_crash(self, tmp_path):
        """Aucune position locale + aucun solde → pas de crash."""
        state_file = str(tmp_path / "state.json")
        bot = _build_bot(state_file)

        bot._client.get_balances.return_value = [
            _make_balance("USD", 1000.0),
        ]

        bot._reconcile_positions()

        assert len(bot._positions) == 0

    def test_no_state_file(self, tmp_path):
        """Pas de fichier state.json → pas de crash."""
        state_file = str(tmp_path / "nonexistent" / "state.json")
        bot = _build_bot.__wrapped__(state_file) if hasattr(_build_bot, '__wrapped__') else None

        # Utiliser directement le builder
        store = PositionStore(state_file=state_file)
        with patch("src.bot.RevolutXClient") as MockClient, \
             patch("src.bot.DataProvider"), \
             patch("src.bot.TelegramNotifier") as MockTelegram, \
             patch("src.bot.config") as mock_config:

            mock_config.TRADING_PAIRS = ["BTC-USD", "SOL-USD", "XRP-USD"]

            from src.bot import TradeXBot

            bot = TradeXBot.__new__(TradeXBot)
            bot.dry_run = False
            bot._running = False
            bot._client = MockClient.return_value
            bot._data = MagicMock()
            bot._data.get_ticker.return_value = None
            bot._telegram = MockTelegram.return_value
            bot._store = store
            bot._trends = {}
            bot._ranges = {}
            bot._positions = {}
            bot._last_heartbeat_time = 0.0
            bot._cycle_count = 0
            bot._prev_state = {}
            bot._prev_ignored = {}

        bot._client.get_balances.return_value = [
            _make_balance("USD", 1000.0),
        ]

        bot._reconcile_positions()

        assert len(bot._positions) == 0


class TestReconcileTelegramNotification:
    """La réconciliation envoie une notification Telegram."""

    def test_notification_sent_with_correct_counts(self, tmp_path):
        """notify_reconciliation appelée avec les bons compteurs."""
        pos = _make_position(size=0.001)
        state_file = str(tmp_path / "state.json")
        bot = _build_bot(state_file, positions={"BTC-USD": pos})

        bot._client.get_balances.return_value = [
            _make_balance("BTC", 0.001),
            _make_balance("USD", 1000.0),
        ]

        bot._reconcile_positions()

        bot._telegram.notify_reconciliation.assert_called_once_with(1, 0, [], [])

    def test_notification_with_removed_and_recovered(self, tmp_path):
        """Notification correcte quand position retirée + orphelin reconstruit."""
        pos = _make_position(symbol="SOL-USD", size=10.0)
        state_file = str(tmp_path / "state.json")
        bot = _build_bot(state_file, positions={"SOL-USD": pos})

        bot._client.get_balances.return_value = [
            _make_balance("SOL", 0.0),     # Position retirée (pas de solde)
            _make_balance("BTC", 0.005),   # Orphelin → reconstruit
            _make_balance("USD", 1000.0),
        ]
        bot._data.get_ticker.return_value = TickerData(
            symbol="BTC-USD", bid=94000.0, ask=94100.0, mid=94050.0, last_price=95000.0,
        )

        bot._reconcile_positions()

        call_args = bot._telegram.notify_reconciliation.call_args[0]
        confirmed, removed, orphans, recovered = call_args
        assert confirmed == 0
        assert removed == 1
        assert len(orphans) == 0
        assert len(recovered) == 1
        assert recovered[0].symbol == "BTC-USD"
        assert recovered[0].size == 0.005
