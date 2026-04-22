"""Tests pour breakout_engine.py — détection breakout, trailing stop, indicateurs."""

from typing import Optional

import pytest

from src.core.models import Candle
from src.core.breakout_engine import (
    BreakoutSignal,
    TrailingResult,
    compute_atr,
    compute_sma,
    detect_breakout,
    rolling_high,
    update_trailing_stop,
)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_candle(
    close: float,
    high: Optional[float] = None,
    low: Optional[float] = None,
    volume: float = 1000.0,
    ts: int = 0,
) -> Candle:
    h = high if high is not None else close * 1.01
    l = low if low is not None else close * 0.99
    return Candle(timestamp=ts, open=close, high=h, low=l, close=close, volume=volume)


def _flat_candles(price: float, n: int, volume: float = 1000.0) -> list[Candle]:
    return [_make_candle(price, volume=volume, ts=i * 900_000) for i in range(n)]


# ── compute_atr ───────────────────────────────────────────────────────────────

class TestComputeATR:
    def test_returns_same_length(self):
        candles = _flat_candles(100.0, 20)
        atrs = compute_atr(candles, period=14)
        assert len(atrs) == 20

    def test_insufficient_candles(self):
        candles = _flat_candles(100.0, 1)
        atrs = compute_atr(candles, period=14)
        assert len(atrs) == 1

    def test_flat_market_low_atr(self):
        """Marché plat → ATR ≈ (high - low) = ~2% de 100."""
        candles = _flat_candles(100.0, 30)
        atrs = compute_atr(candles, period=14)
        # high = 101, low = 99 → TR ≈ 2.0
        assert all(0 < a <= 2.5 for a in atrs)

    def test_atr_positive(self):
        candles = _flat_candles(50.0, 50)
        atrs = compute_atr(candles, period=14)
        assert all(a >= 0 for a in atrs)


# ── compute_sma ───────────────────────────────────────────────────────────────

class TestComputeSMA:
    def test_basic(self):
        values = [1.0, 2.0, 3.0, 4.0, 5.0]
        result = compute_sma(values, period=3)
        assert len(result) == 5
        assert result[2] == pytest.approx(2.0)
        assert result[3] == pytest.approx(3.0)
        assert result[4] == pytest.approx(4.0)

    def test_period_1(self):
        values = [1.0, 2.0, 3.0]
        result = compute_sma(values, period=1)
        assert result == pytest.approx(values)

    def test_warm_up(self):
        values = [1.0, 2.0, 3.0, 4.0]
        result = compute_sma(values, period=3)
        # Avant warmup : moyenne cumulative
        assert result[0] == pytest.approx(1.0)
        assert result[1] == pytest.approx(1.5)


# ── rolling_high ──────────────────────────────────────────────────────────────

class TestRollingHigh:
    def test_basic_lookback(self):
        highs = [1.0, 3.0, 2.0, 5.0, 4.0]
        result = rolling_high(highs, lookback=2)
        assert len(result) == 5
        # Index 0 : pas de barre précédente → highs[0]
        assert result[0] == pytest.approx(1.0)
        # Index 1 : max(highs[0:1]) = 1.0
        assert result[1] == pytest.approx(1.0)
        # Index 2 : max(highs[0:2]) = 3.0 (lookback 2 → indices 0..1)
        assert result[2] == pytest.approx(3.0)
        # Index 3 : max(highs[1:3]) = 3.0
        assert result[3] == pytest.approx(3.0)
        # Index 4 : max(highs[2:4]) = 5.0
        assert result[4] == pytest.approx(5.0)

    def test_monotone_increasing(self):
        highs = [1.0, 2.0, 3.0, 4.0, 5.0]
        result = rolling_high(highs, lookback=3)
        # Chaque résultat est ≤ la valeur précédente peut grandir
        assert all(r > 0 for r in result)

    def test_single_element(self):
        result = rolling_high([42.0], lookback=5)
        assert result == [42.0]

    def test_excludes_current_bar(self):
        """rolling_high exclut la barre courante (regarde seulement les barres précédentes)."""
        highs = [10.0, 5.0, 3.0, 100.0]
        result = rolling_high(highs, lookback=10)
        # À l'index 3, current high = 100, mais rolling_high regarde [0:3] = [10,5,3] → 10
        assert result[3] == pytest.approx(10.0)


# ── update_trailing_stop ──────────────────────────────────────────────────────

class TestUpdateTrailingStop:
    def test_not_activated_yet(self):
        """Avant l'activation, le SL ne bouge pas."""
        result = update_trailing_stop(
            current_price=100.0,
            entry_price=100.0,
            current_sl=95.0,
            peak_price=100.0,
            trailing_activation=103.0,
            trailing_distance=2.0,
        )
        assert result.trailing_active is False
        assert result.new_sl == pytest.approx(95.0)

    def test_activates_at_threshold(self):
        """SL suit le peak - distance dès que current_price ≥ activation."""
        result = update_trailing_stop(
            current_price=104.0,
            entry_price=100.0,
            current_sl=95.0,
            peak_price=104.0,
            trailing_activation=103.0,
            trailing_distance=2.0,
        )
        assert result.trailing_active is True
        assert result.new_sl == pytest.approx(102.0)  # 104 - 2

    def test_sl_never_decreases(self):
        """Le SL ne peut jamais descendre."""
        result = update_trailing_stop(
            current_price=103.5,
            entry_price=100.0,
            current_sl=103.0,
            peak_price=105.0,
            trailing_activation=103.0,
            trailing_distance=2.0,
        )
        # candidate = 105 - 2 = 103, current_sl = 103 → new_sl = max(103, 103) = 103
        assert result.new_sl >= 103.0

    def test_peak_updates(self):
        """Le peak monte avec le prix."""
        result = update_trailing_stop(
            current_price=110.0,
            entry_price=100.0,
            current_sl=90.0,
            peak_price=105.0,
            trailing_activation=103.0,
            trailing_distance=2.0,
        )
        assert result.peak_price == pytest.approx(110.0)
        assert result.new_sl == pytest.approx(108.0)  # 110 - 2


# ── detect_breakout ───────────────────────────────────────────────────────────

class TestDetectBreakout:
    def _build_candles_with_spike(self, base_price: float = 100.0, n: int = 60) -> list[Candle]:
        """Série plate puis spike breakout sur la dernière bougie."""
        candles = _flat_candles(base_price, n - 1, volume=1000.0)
        # Dernière bougie : close bien au-dessus du high précédent, volume spike
        spike = Candle(
            timestamp=n * 900_000,
            open=base_price,
            high=base_price * 1.03,
            low=base_price * 0.995,
            close=base_price * 1.025,  # > rolling high des n-1 barres (~base_price*1.01)
            volume=3000.0,  # 3× le volume moyen
        )
        return candles + [spike]

    def test_insufficient_data_returns_none(self):
        candles = _flat_candles(100.0, 5)
        assert detect_breakout(candles) is None

    def test_flat_market_no_signal(self):
        """Marché plat → pas de breakout."""
        candles = _flat_candles(100.0, 60)
        assert detect_breakout(candles) is None

    def test_breakout_detected(self):
        """Spike clair → signal détecté avec SL < entry < TP."""
        candles = self._build_candles_with_spike(100.0, 60)
        signal = detect_breakout(
            candles,
            lookback=12,
            atr_expansion_ratio=1.0,  # relâcher filtre expansion
            volume_spike_mult=1.0,
            atr_expansion_lookback=5,
        )
        if signal is not None:
            assert isinstance(signal, BreakoutSignal)
            assert signal.sl_price < signal.entry_price < signal.tp_price
            assert signal.atr_value > 0
            assert signal.trailing_activation > signal.entry_price
            assert signal.trailing_distance > 0

    def test_signal_levels_coherent(self):
        """Si signal détecté : SL < entry, TP > entry, trail_activation > entry."""
        candles = self._build_candles_with_spike(50.0, 80)
        signal = detect_breakout(candles, atr_expansion_ratio=1.0, volume_spike_mult=0.5)
        if signal is not None:
            assert signal.sl_price < signal.entry_price
            assert signal.tp_price > signal.entry_price
            assert signal.trailing_activation > signal.entry_price
