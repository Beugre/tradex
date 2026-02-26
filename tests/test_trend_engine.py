"""Tests pour trend_engine.py — classification HH/HL/LH/LL et tendance."""

import pytest

from src.core.models import (
    SwingLevel,
    SwingPoint,
    SwingType,
    TrendDirection,
)
from src.core.trend_engine import (
    check_trend_invalidation,
    classify_swings,
    determine_trend,
)


def _high(index: int, price: float) -> SwingPoint:
    """Helper : crée un swing high."""
    return SwingPoint(
        index=index,
        price=price,
        level=SwingLevel.HIGH,
        timestamp=index * 14400000,
    )


def _low(index: int, price: float) -> SwingPoint:
    """Helper : crée un swing low."""
    return SwingPoint(
        index=index,
        price=price,
        level=SwingLevel.LOW,
        timestamp=index * 14400000,
    )


class TestClassifySwings:
    """Tests de classification HH/HL/LH/LL."""

    def test_bullish_sequence(self):
        """High croissants + Low croissants → HH + HL."""
        swings = [
            _high(0, 100),
            _low(1, 90),
            _high(2, 110),   # HH
            _low(3, 95),     # HL
        ]
        classified = classify_swings(swings)

        assert classified[2].swing_type == SwingType.HH
        assert classified[3].swing_type == SwingType.HL

    def test_bearish_sequence(self):
        """High décroissants + Low décroissants → LH + LL."""
        swings = [
            _high(0, 110),
            _low(1, 95),
            _high(2, 105),   # LH
            _low(3, 90),     # LL
        ]
        classified = classify_swings(swings)

        assert classified[2].swing_type == SwingType.LH
        assert classified[3].swing_type == SwingType.LL

    def test_first_swing_of_each_type_has_no_classification(self):
        """Le tout premier high et le tout premier low ne peuvent pas être classifiés."""
        swings = [
            _high(0, 100),
            _low(1, 90),
        ]
        classified = classify_swings(swings)

        assert classified[0].swing_type is None
        assert classified[1].swing_type is None

    def test_mixed_sequence(self):
        """Séquence mixte : HH puis LH."""
        swings = [
            _high(0, 100),
            _low(1, 90),
            _high(2, 110),   # HH
            _low(3, 85),     # LL (90 → 85)
            _high(4, 105),   # LH (110 → 105)
        ]
        classified = classify_swings(swings)

        assert classified[2].swing_type == SwingType.HH
        assert classified[3].swing_type == SwingType.LL
        assert classified[4].swing_type == SwingType.LH


class TestDetermineTrend:
    """Tests de détermination de la tendance."""

    def test_bullish_trend(self):
        swings = [
            _high(0, 100),
            _low(1, 90),
            _high(2, 110),  # HH
            _low(3, 95),    # HL
        ]
        trend = determine_trend(swings, "BTC-USD")

        assert trend.direction == TrendDirection.BULLISH
        assert trend.symbol == "BTC-USD"

    def test_bearish_trend(self):
        swings = [
            _high(0, 110),
            _low(1, 95),
            _high(2, 105),  # LH
            _low(3, 90),    # LL
        ]
        trend = determine_trend(swings, "ETH-USD")

        assert trend.direction == TrendDirection.BEARISH

    def test_neutral_on_mixed_signals(self):
        """HH + LL → ni bullish ni bearish → NEUTRAL."""
        swings = [
            _high(0, 100),
            _low(1, 90),
            _high(2, 110),  # HH
            _low(3, 85),    # LL
        ]
        trend = determine_trend(swings, "SOL-USD")

        assert trend.direction == TrendDirection.NEUTRAL

    def test_neutral_with_insufficient_swings(self):
        swings = [_high(0, 100)]
        trend = determine_trend(swings, "XRP-USD")

        assert trend.direction == TrendDirection.NEUTRAL


class TestTrendInvalidation:
    """Tests d'invalidation de tendance."""

    def test_bearish_invalidated_by_price_above_lh(self):
        swings = [
            _high(0, 110),
            _low(1, 95),
            _high(2, 105),  # LH
            _low(3, 90),    # LL
        ]
        trend = determine_trend(swings, "BTC-USD")
        assert trend.direction == TrendDirection.BEARISH

        # Prix casse le LH (105)
        updated = check_trend_invalidation(trend, current_price=106)
        assert updated.direction == TrendDirection.NEUTRAL

    def test_bullish_invalidated_by_price_below_hl(self):
        swings = [
            _high(0, 100),
            _low(1, 90),
            _high(2, 110),  # HH
            _low(3, 95),    # HL
        ]
        trend = determine_trend(swings, "BTC-USD")
        assert trend.direction == TrendDirection.BULLISH

        # Prix casse le HL (95)
        updated = check_trend_invalidation(trend, current_price=94)
        assert updated.direction == TrendDirection.NEUTRAL

    def test_no_invalidation_when_price_within_range(self):
        swings = [
            _high(0, 110),
            _low(1, 95),
            _high(2, 105),  # LH
            _low(3, 90),    # LL
        ]
        trend = determine_trend(swings, "BTC-USD")
        assert trend.direction == TrendDirection.BEARISH

        # Prix reste sous le LH
        updated = check_trend_invalidation(trend, current_price=100)
        assert updated.direction == TrendDirection.BEARISH
