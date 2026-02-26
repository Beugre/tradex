"""Tests pour strategy_trend.py — signaux d'entrée, SL, et construction d'ordres."""

import pytest

from src.core.models import (
    OrderSide,
    Position,
    PositionStatus,
    SwingLevel,
    SwingPoint,
    SwingType,
    TickerData,
    TrendDirection,
    TrendState,
)
from src.core.strategy_trend import (
    build_entry_order,
    build_exit_order,
    check_entry_signal,
    check_sl_hit,
    process_trailing_stop,
    process_zero_risk,
)


def _ticker(symbol: str, price: float) -> TickerData:
    return TickerData(
        symbol=symbol, bid=price - 1, ask=price + 1, mid=price, last_price=price
    )


def _bullish_trend(symbol: str = "BTC-USD") -> TrendState:
    """Crée un état de tendance bullish avec HH=110, HL=95."""
    return TrendState(
        symbol=symbol,
        direction=TrendDirection.BULLISH,
        last_high=SwingPoint(
            index=2, price=110, level=SwingLevel.HIGH,
            timestamp=2000, swing_type=SwingType.HH,
        ),
        last_low=SwingPoint(
            index=3, price=95, level=SwingLevel.LOW,
            timestamp=3000, swing_type=SwingType.HL,
        ),
    )


def _bearish_trend(symbol: str = "BTC-USD") -> TrendState:
    """Crée un état de tendance bearish avec LH=105, LL=90."""
    return TrendState(
        symbol=symbol,
        direction=TrendDirection.BEARISH,
        last_high=SwingPoint(
            index=2, price=105, level=SwingLevel.HIGH,
            timestamp=2000, swing_type=SwingType.LH,
        ),
        last_low=SwingPoint(
            index=3, price=90, level=SwingLevel.LOW,
            timestamp=3000, swing_type=SwingType.LL,
        ),
    )


class TestCheckEntrySignal:
    """Tests des signaux d'entrée."""

    def test_bullish_signal_triggered(self):
        """Prix >= HH * (1 + buffer%) → signal BUY."""
        trend = _bullish_trend()  # HH=110, HL=95
        ticker = _ticker("BTC-USD", 115)  # > 110 * 1.02 = 112.2
        signal = check_entry_signal(trend, ticker, entry_buffer_pct=0.02)

        assert signal is not None
        assert signal["side"] == OrderSide.BUY

    def test_bullish_signal_not_triggered(self):
        """Prix < HH * (1 + buffer%) → pas de signal."""
        trend = _bullish_trend()  # HH=110
        ticker = _ticker("BTC-USD", 108)  # < 110 * 1.02 = 112.2
        signal = check_entry_signal(trend, ticker, entry_buffer_pct=0.02)

        assert signal is None

    def test_bearish_signal_triggered(self):
        """Prix <= LL * (1 - buffer%) → signal SELL."""
        trend = _bearish_trend()  # LH=105, LL=90
        ticker = _ticker("BTC-USD", 87)  # < 90 * 0.98 = 88.2
        signal = check_entry_signal(trend, ticker, entry_buffer_pct=0.02)

        assert signal is not None
        assert signal["side"] == OrderSide.SELL

    def test_neutral_trend_no_signal(self):
        trend = TrendState(symbol="BTC-USD", direction=TrendDirection.NEUTRAL)
        ticker = _ticker("BTC-USD", 100)
        signal = check_entry_signal(trend, ticker, entry_buffer_pct=0.02)

        assert signal is None


class TestCheckSlHit:
    """Tests de détection du stop loss."""

    def test_buy_sl_hit(self):
        """Long : prix descend sous le SL → True."""
        pos = Position(
            symbol="BTC-USD", side=OrderSide.BUY,
            entry_price=100, sl_price=95, size=0.01,
            venue_order_id="test", status=PositionStatus.OPEN,
        )
        ticker = _ticker("BTC-USD", 93)  # < 95 - 95*0.01 = 94.05
        assert check_sl_hit(pos, ticker, sl_buffer_pct=0.01)

    def test_buy_sl_not_hit(self):
        """Long : prix au-dessus du SL → False."""
        pos = Position(
            symbol="BTC-USD", side=OrderSide.BUY,
            entry_price=100, sl_price=95, size=0.01,
            venue_order_id="test", status=PositionStatus.OPEN,
        )
        ticker = _ticker("BTC-USD", 96)
        assert not check_sl_hit(pos, ticker, sl_buffer_pct=0.01)

    def test_sell_sl_hit(self):
        """Short : prix monte au-dessus du SL → True."""
        pos = Position(
            symbol="BTC-USD", side=OrderSide.SELL,
            entry_price=100, sl_price=105, size=0.01,
            venue_order_id="test", status=PositionStatus.OPEN,
        )
        ticker = _ticker("BTC-USD", 107)  # > 105 + 105*0.01 = 106.05
        assert check_sl_hit(pos, ticker, sl_buffer_pct=0.01)

    def test_zero_risk_sl_used_when_applied(self):
        """Quand zero-risk est appliqué, utiliser le SL ajusté."""
        pos = Position(
            symbol="BTC-USD", side=OrderSide.BUY,
            entry_price=100, sl_price=95, size=0.01,
            venue_order_id="test", status=PositionStatus.ZERO_RISK,
            is_zero_risk_applied=True, zero_risk_sl=100.5,
        )
        # Prix à 99 : sous le zero-risk SL (100.5 - 100.5*0.01 = 99.495)
        ticker = _ticker("BTC-USD", 99)
        assert check_sl_hit(pos, ticker, sl_buffer_pct=0.01)


class TestBuildOrders:
    """Tests de construction d'ordres."""

    def test_build_entry_order(self):
        order = build_entry_order("BTC-USD", OrderSide.BUY, 95000.0, 0.001)
        payload = order.to_api_payload()

        assert payload["symbol"] == "BTC-USD"
        assert payload["side"] == "buy"
        assert payload["order_configuration"]["limit"]["base_size"] == "0.00100000"
        assert payload["order_configuration"]["limit"]["price"] == "95000.00"

    def test_build_exit_order_reverses_side(self):
        pos = Position(
            symbol="BTC-USD", side=OrderSide.BUY,
            entry_price=95000, sl_price=93000, size=0.001,
            venue_order_id="test", status=PositionStatus.OPEN,
        )
        order = build_exit_order(pos, current_price=93000)
        payload = order.to_api_payload()

        # Exit d'un BUY → SELL
        assert payload["side"] == "sell"
        assert payload["order_configuration"]["limit"]["base_size"] == "0.00100000"


class TestProcessZeroRisk:
    """Tests du passage zero-risk."""

    def test_zero_risk_applied(self):
        pos = Position(
            symbol="BTC-USD", side=OrderSide.BUY,
            entry_price=100, sl_price=95, size=0.01,
            venue_order_id="test", status=PositionStatus.OPEN,
        )
        # Prix à 103 → 3% gain, trigger à 2%
        new_sl = process_zero_risk(pos, current_price=103, trigger_percent=0.02, lock_percent=0.005)

        assert new_sl is not None
        assert new_sl == pytest.approx(100.5)
        assert pos.is_zero_risk_applied is True
        assert pos.status == PositionStatus.ZERO_RISK

    def test_zero_risk_not_applied_insufficient_move(self):
        pos = Position(
            symbol="BTC-USD", side=OrderSide.BUY,
            entry_price=100, sl_price=95, size=0.01,
            venue_order_id="test", status=PositionStatus.OPEN,
        )
        new_sl = process_zero_risk(pos, current_price=101, trigger_percent=0.02, lock_percent=0.005)

        assert new_sl is None
        assert pos.is_zero_risk_applied is False


class TestProcessTrailingStop:
    """Tests du trailing stop via order_manager."""

    def test_trailing_stop_updates_sl(self):
        """Trailing stop met à jour le SL quand le prix progresse."""
        pos = Position(
            symbol="BTC-USD", side=OrderSide.BUY,
            entry_price=100, sl_price=95, size=0.01,
            venue_order_id="test", status=PositionStatus.ZERO_RISK,
            is_zero_risk_applied=True, zero_risk_sl=100.5,
        )
        new_sl = process_trailing_stop(pos, current_price=110, trailing_percent=0.02)

        assert new_sl is not None
        assert new_sl == pytest.approx(107.80)

    def test_trailing_stop_none_without_zero_risk(self):
        """Trailing stop retourne None si zero-risk pas activé."""
        pos = Position(
            symbol="BTC-USD", side=OrderSide.BUY,
            entry_price=100, sl_price=95, size=0.01,
            venue_order_id="test", status=PositionStatus.OPEN,
        )
        result = process_trailing_stop(pos, current_price=110, trailing_percent=0.02)
        assert result is None
