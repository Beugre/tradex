"""Tests pour swing_detector.py — détection des swings highs et lows."""

import pytest

from src.core.models import Candle, SwingLevel
from src.core.swing_detector import detect_swings, get_latest_swings


def _make_candle(
    index: int,
    high: float,
    low: float,
    open_: float = 0,
    close: float = 0,
    volume: float = 1.0,
) -> Candle:
    """Helper : crée une bougie avec high/low spécifiés."""
    return Candle(
        timestamp=1000000 + index * 14400000,
        open=open_ or (high + low) / 2,
        high=high,
        low=low,
        close=close or (high + low) / 2,
        volume=volume,
    )


class TestDetectSwings:
    """Tests de détection des swing highs et lows."""

    def test_empty_candles_returns_empty(self):
        assert detect_swings([], lookback=3) == []

    def test_too_few_candles_returns_empty(self):
        candles = [_make_candle(i, 100 + i, 90 + i) for i in range(5)]
        assert detect_swings(candles, lookback=3) == []

    def test_detects_single_swing_high(self):
        """Bougie au milieu avec le high le plus élevé → swing high."""
        #         Index:  0    1    2    3    4    5    6
        highs =         [100, 102, 105, 110, 105, 102, 100]
        lows =          [95,  97,  100, 105, 100, 97,  95]

        candles = [_make_candle(i, h, l) for i, (h, l) in enumerate(zip(highs, lows))]
        swings = detect_swings(candles, lookback=3)

        swing_highs = [s for s in swings if s.level == SwingLevel.HIGH]
        assert len(swing_highs) == 1
        assert swing_highs[0].index == 3
        assert swing_highs[0].price == 110

    def test_detects_single_swing_low(self):
        """Bougie au milieu avec le low le plus bas → swing low."""
        highs = [110, 108, 105, 100, 105, 108, 110]
        lows =  [105, 103, 100, 90,  100, 103, 105]

        candles = [_make_candle(i, h, l) for i, (h, l) in enumerate(zip(highs, lows))]
        swings = detect_swings(candles, lookback=3)

        swing_lows = [s for s in swings if s.level == SwingLevel.LOW]
        assert len(swing_lows) == 1
        assert swing_lows[0].index == 3
        assert swing_lows[0].price == 90

    def test_detects_multiple_swings(self):
        """Séquence en zigzag → plusieurs swings détectés."""
        #              0    1    2    3    4    5    6    7    8    9   10   11   12
        highs =      [100, 102, 105, 110, 108, 103, 100, 105, 115, 112, 108, 105, 100]
        lows =       [95,  97,  100, 105, 103, 98,  90,  100, 110, 107, 103, 100, 95]

        candles = [_make_candle(i, h, l) for i, (h, l) in enumerate(zip(highs, lows))]
        swings = detect_swings(candles, lookback=3)

        assert len(swings) >= 2  # Au moins un high et un low

    def test_equal_highs_no_swing(self):
        """Si les voisines ont le même high, pas de swing high."""
        highs = [100, 100, 100, 100, 100, 100, 100]
        lows =  [90,  90,  90,  90,  90,  90,  90]

        candles = [_make_candle(i, h, l) for i, (h, l) in enumerate(zip(highs, lows))]
        swings = detect_swings(candles, lookback=3)

        assert len(swings) == 0

    def test_lookback_1(self):
        """Lookback=1 : chaque bougie n'a besoin que d'un voisin de chaque côté."""
        highs = [100, 110, 100]
        lows =  [95,  105, 95]

        candles = [_make_candle(i, h, l) for i, (h, l) in enumerate(zip(highs, lows))]
        swings = detect_swings(candles, lookback=1)

        swing_highs = [s for s in swings if s.level == SwingLevel.HIGH]
        assert len(swing_highs) == 1
        assert swing_highs[0].index == 1


class TestGetLatestSwings:
    """Tests de get_latest_swings."""

    def test_returns_last_n(self):
        from src.core.models import SwingPoint

        swings = [
            SwingPoint(index=i, price=100 + i, level=SwingLevel.HIGH, timestamp=i * 1000)
            for i in range(10)
        ]
        latest = get_latest_swings(swings, count=4)
        assert len(latest) == 4
        assert latest[0].index == 6

    def test_returns_all_if_less_than_count(self):
        from src.core.models import SwingPoint

        swings = [
            SwingPoint(index=0, price=100, level=SwingLevel.HIGH, timestamp=0),
            SwingPoint(index=1, price=90, level=SwingLevel.LOW, timestamp=1000),
        ]
        latest = get_latest_swings(swings, count=4)
        assert len(latest) == 2
