"""Tests pour strategy_mean_rev.py — signaux range, TP, SL, cooldown."""

import time
from unittest.mock import patch

import pytest

from src.core.models import (
    OrderSide,
    Position,
    PositionStatus,
    RangeState,
    StrategyType,
    SwingLevel,
    SwingPoint,
    SwingType,
    TickerData,
    TrendDirection,
    TrendState,
)
from src.core.strategy_mean_rev import (
    activate_cooldown,
    build_range_from_trend,
    check_range_entry_signal,
    check_range_sl_hit,
    check_range_tp_hit,
    is_in_cooldown,
)


def _ticker(symbol: str, price: float) -> TickerData:
    return TickerData(
        symbol=symbol, bid=price - 1, ask=price + 1, mid=price, last_price=price
    )


def _neutral_trend(symbol: str = "LINK-USD", high: float = 110.0, low: float = 90.0) -> TrendState:
    """Crée une tendance NEUTRAL avec des swings high/low définis."""
    return TrendState(
        symbol=symbol,
        direction=TrendDirection.NEUTRAL,
        last_high=SwingPoint(
            index=2, price=high, level=SwingLevel.HIGH,
            timestamp=2000, swing_type=SwingType.HH,
        ),
        last_low=SwingPoint(
            index=3, price=low, level=SwingLevel.LOW,
            timestamp=3000, swing_type=SwingType.LL,
        ),
    )


def _range(symbol: str = "LINK-USD", high: float = 110.0, low: float = 90.0) -> RangeState:
    return RangeState(symbol=symbol, range_high=high, range_low=low)


def _range_position(
    symbol: str = "LINK-USD",
    side: OrderSide = OrderSide.BUY,
    entry: float = 90.5,
    sl: float = 89.8,
    tp: float = 100.0,
) -> Position:
    return Position(
        symbol=symbol,
        side=side,
        entry_price=entry,
        sl_price=sl,
        size=1.0,
        venue_order_id="test-range",
        status=PositionStatus.OPEN,
        strategy=StrategyType.RANGE,
        tp_price=tp,
    )


# ── build_range_from_trend ─────────────────────────────────────────────────────

class TestBuildRangeFromTrend:

    def test_valid_range_from_neutral_trend(self):
        """Tendance NEUTRAL avec swings assez larges → range valide."""
        trend = _neutral_trend(high=110, low=90)
        rs = build_range_from_trend(trend, min_width_pct=0.02)

        assert rs is not None
        assert rs.range_high == 110
        assert rs.range_low == 90
        assert rs.range_mid == pytest.approx(100.0)
        assert rs.range_width_pct == pytest.approx(0.20)  # 20/100

    def test_returns_none_for_bullish_trend(self):
        """Tendance BULLISH → pas de range."""
        trend = _neutral_trend()
        trend.direction = TrendDirection.BULLISH
        assert build_range_from_trend(trend, min_width_pct=0.02) is None

    def test_returns_none_for_bearish_trend(self):
        """Tendance BEARISH → pas de range."""
        trend = _neutral_trend()
        trend.direction = TrendDirection.BEARISH
        assert build_range_from_trend(trend, min_width_pct=0.02) is None

    def test_returns_none_if_range_too_narrow(self):
        """Range de 1% quand le min est 2% → None."""
        trend = _neutral_trend(high=101, low=100)  # ~1%
        rs = build_range_from_trend(trend, min_width_pct=0.02)
        assert rs is None

    def test_returns_none_if_no_swings(self):
        """Pas de swings → None."""
        trend = TrendState(symbol="TEST", direction=TrendDirection.NEUTRAL)
        assert build_range_from_trend(trend, min_width_pct=0.02) is None

    def test_returns_none_if_high_equals_low(self):
        """High == Low → range invalide."""
        trend = _neutral_trend(high=100, low=100)
        assert build_range_from_trend(trend, min_width_pct=0.02) is None


# ── check_range_entry_signal ───────────────────────────────────────────────────

class TestCheckRangeEntrySignal:

    def test_buy_signal_at_range_low(self):
        """Prix au bas du range → signal BUY."""
        rs = _range(high=110, low=90)
        ticker = _ticker("LINK-USD", 90.0)
        signal = check_range_entry_signal(rs, ticker, entry_buffer_pct=0.002)

        assert signal is not None
        assert signal["side"] == OrderSide.BUY
        assert signal["entry_price"] == 90.0
        assert signal["tp_price"] == pytest.approx(100.0)

    def test_sell_signal_at_range_high(self):
        """Prix au haut du range → signal SELL."""
        rs = _range(high=110, low=90)
        ticker = _ticker("LINK-USD", 110.0)
        signal = check_range_entry_signal(rs, ticker, entry_buffer_pct=0.002)

        assert signal is not None
        assert signal["side"] == OrderSide.SELL
        assert signal["entry_price"] == 110.0
        assert signal["tp_price"] == pytest.approx(100.0)

    def test_no_signal_in_middle_of_range(self):
        """Prix au milieu du range → pas de signal."""
        rs = _range(high=110, low=90)
        ticker = _ticker("LINK-USD", 100.0)
        signal = check_range_entry_signal(rs, ticker, entry_buffer_pct=0.002)
        assert signal is None

    def test_no_signal_when_invalid_range(self):
        """Range invalide → pas de signal."""
        rs = _range(high=110, low=90)
        rs.is_valid = False
        ticker = _ticker("LINK-USD", 90.0)
        assert check_range_entry_signal(rs, ticker, entry_buffer_pct=0.002) is None

    def test_no_signal_during_cooldown(self):
        """En cooldown → pas de signal."""
        rs = _range(high=110, low=90)
        rs.cooldown_until = int(time.time()) + 3600  # 1h dans le futur
        ticker = _ticker("LINK-USD", 90.0)
        assert check_range_entry_signal(rs, ticker, entry_buffer_pct=0.002) is None

    def test_no_buy_signal_on_breakout_below(self):
        """Prix bien en dessous du range → breakout, pas de signal BUY.
        
        Cas réel : range_low=0.9511, prix=0.905 → SL (0.949) > entry (0.905).
        """
        rs = _range(high=1.0042, low=0.9511)
        ticker = _ticker("SUI-USD", 0.905)
        signal = check_range_entry_signal(rs, ticker, entry_buffer_pct=0.002)
        assert signal is None

    def test_no_sell_signal_on_breakout_above(self):
        """Prix bien au-dessus du range → breakout, pas de signal SELL."""
        rs = _range(high=110, low=90)
        ticker = _ticker("LINK-USD", 115.0)
        signal = check_range_entry_signal(rs, ticker, entry_buffer_pct=0.002)
        assert signal is None


# ── check_range_tp_hit ─────────────────────────────────────────────────────────

class TestCheckRangeTpHit:

    def test_buy_tp_hit(self):
        """BUY : prix atteint le TP (range_mid) → True."""
        pos = _range_position(side=OrderSide.BUY, tp=100.0)
        ticker = _ticker("LINK-USD", 100.5)
        assert check_range_tp_hit(pos, ticker) is True

    def test_buy_tp_not_hit(self):
        """BUY : prix sous le TP → False."""
        pos = _range_position(side=OrderSide.BUY, tp=100.0)
        ticker = _ticker("LINK-USD", 95.0)
        assert check_range_tp_hit(pos, ticker) is False

    def test_sell_tp_hit(self):
        """SELL : prix descend au TP (range_mid) → True."""
        pos = _range_position(side=OrderSide.SELL, entry=109.5, sl=110.2, tp=100.0)
        ticker = _ticker("LINK-USD", 99.5)
        assert check_range_tp_hit(pos, ticker) is True

    def test_sell_tp_not_hit(self):
        """SELL : prix au-dessus du TP → False."""
        pos = _range_position(side=OrderSide.SELL, entry=109.5, sl=110.2, tp=100.0)
        ticker = _ticker("LINK-USD", 105.0)
        assert check_range_tp_hit(pos, ticker) is False

    def test_tp_ignored_for_trend_position(self):
        """Position TREND → TP range pas vérifié."""
        pos = _range_position(tp=100.0)
        pos.strategy = StrategyType.TREND
        ticker = _ticker("LINK-USD", 101.0)
        assert check_range_tp_hit(pos, ticker) is False

    def test_tp_none_returns_false(self):
        """Pas de tp_price → False."""
        pos = _range_position(tp=100.0)
        pos.tp_price = None
        ticker = _ticker("LINK-USD", 101.0)
        assert check_range_tp_hit(pos, ticker) is False


# ── check_range_sl_hit ─────────────────────────────────────────────────────────

class TestCheckRangeSlHit:

    def test_buy_sl_hit_breakout_below(self):
        """BUY : cassure sous le SL → True."""
        pos = _range_position(side=OrderSide.BUY, sl=89.8)
        ticker = _ticker("LINK-USD", 88.0)  # < 89.8 - 89.8*0.003 = 89.53
        assert check_range_sl_hit(pos, ticker, sl_buffer_pct=0.003) is True

    def test_buy_sl_not_hit(self):
        """BUY : prix au-dessus du SL → False."""
        pos = _range_position(side=OrderSide.BUY, sl=89.8)
        ticker = _ticker("LINK-USD", 90.0)
        assert check_range_sl_hit(pos, ticker, sl_buffer_pct=0.003) is False

    def test_sell_sl_hit_breakout_above(self):
        """SELL : cassure au-dessus du SL → True."""
        pos = _range_position(side=OrderSide.SELL, entry=109.5, sl=110.2, tp=100.0)
        ticker = _ticker("LINK-USD", 111.0)  # > 110.2 + 110.2*0.003 ≈ 110.53
        assert check_range_sl_hit(pos, ticker, sl_buffer_pct=0.003) is True

    def test_sl_ignored_for_trend_position(self):
        """Position TREND → SL range pas vérifié."""
        pos = _range_position(sl=89.8)
        pos.strategy = StrategyType.TREND
        ticker = _ticker("LINK-USD", 85.0)
        assert check_range_sl_hit(pos, ticker, sl_buffer_pct=0.003) is False


# ── Cooldown ───────────────────────────────────────────────────────────────────

class TestCooldown:

    def test_no_cooldown_when_zero(self):
        rs = _range()
        rs.cooldown_until = 0
        assert is_in_cooldown(rs) is False

    def test_in_cooldown_when_future(self):
        rs = _range()
        rs.cooldown_until = int(time.time()) + 3600
        assert is_in_cooldown(rs) is True

    def test_not_in_cooldown_when_past(self):
        rs = _range()
        rs.cooldown_until = int(time.time()) - 1
        assert is_in_cooldown(rs) is False

    def test_activate_cooldown_sets_timestamp(self):
        rs = _range()
        before = int(time.time())
        activate_cooldown(rs, cooldown_bars=3, bar_duration_seconds=14400)
        after = int(time.time())

        expected_min = before + 3 * 14400
        expected_max = after + 3 * 14400

        assert rs.cooldown_until >= expected_min
        assert rs.cooldown_until <= expected_max
