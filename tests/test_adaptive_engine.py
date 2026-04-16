"""Tests pour adaptive_engine.py — régime indicateur, pas Dow Theory."""

from unittest.mock import patch

from src.core.adaptive_engine import (
    Regime,
    detect_regime,
    detect_regime_debug,
)
from src.core.models import Candle


def _trend_candles(
    count: int = 260,
    start_price: float = 100.0,
    step_pct: float = 0.003,
) -> list[Candle]:
    """Crée une série OHLCV monotone pour exercer les indicateurs Adaptive."""
    candles: list[Candle] = []
    price = start_price

    for idx in range(count):
        open_price = price
        close_price = price * (1.0 + step_pct)
        high_price = max(open_price, close_price) * 1.002
        low_price = min(open_price, close_price) * 0.998
        candles.append(
            Candle(
                timestamp=idx * 3600000,
                open=open_price,
                high=high_price,
                low=low_price,
                close=close_price,
                volume=1000.0,
            )
        )
        price = close_price

    return candles


class TestDetectRegime:
    """Tests de classification de régime pour Adaptive."""

    def test_bull_regime_on_strong_uptrend(self):
        """Une forte hausse continue valide le score indicateur haussier."""
        candles = _trend_candles(step_pct=0.003)

        debug = detect_regime_debug(candles)

        assert detect_regime(candles) == Regime.BULL
        assert debug.regime == Regime.BULL
        assert debug.bull_score >= 4
        assert debug.bear_score <= 1

    def test_bear_regime_on_strong_downtrend(self):
        """Une forte baisse continue valide le score indicateur baissier."""
        candles = _trend_candles(start_price=200.0, step_pct=-0.003)

        debug = detect_regime_debug(candles)

        assert detect_regime(candles) == Regime.BEAR
        assert debug.regime == Regime.BEAR
        assert debug.bear_score >= 4
        assert debug.bull_score <= 1

    def test_range_can_happen_even_with_higher_highs_and_higher_lows(self):
        """
        Adaptive n'utilise pas Dow Theory.

        Meme sur une serie compatible HH/HL, le regime reste RANGE si le score
        indicateur ne passe pas et que l'ADX est faible.
        """
        candles = _trend_candles(step_pct=0.003)
        size = len(candles)

        with patch("src.core.adaptive_engine._adx", return_value=[10.0] * size), \
             patch("src.core.adaptive_engine._rsi", return_value=[50.0] * size), \
             patch("src.core.adaptive_engine._atr", return_value=[10.0] * size), \
             patch("src.core.adaptive_engine._sma", return_value=[10.0] * size), \
             patch("src.core.adaptive_engine._bollinger_width", return_value=[0.03] * size):
            debug = detect_regime_debug(candles)
            regime = detect_regime(candles)

        assert regime == Regime.RANGE
        assert debug.regime == Regime.RANGE
        assert debug.stagnation is False
        assert debug.bull_score == 3
        assert debug.bear_score == 0
        assert debug.adx == 10.0
        assert debug.rsi_1h == 50.0
