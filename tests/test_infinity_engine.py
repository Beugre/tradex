"""Tests pour infinity_engine.py — logique DCA inversé sans I/O."""

import pytest

from src.core.infinity_engine import (
    InfinityConfig,
    InfinityCycle,
    InfinityPhase,
    check_buy_conditions,
    check_first_entry,
    check_override_sell,
    check_sell_conditions,
    check_stop_loss,
    compute_buy_size,
    rsi_series,
    sma_series,
)
from src.core.models import Candle


# ── Helpers ───────────────────────────────────────────────────────────────────

def _closes(prices: list[float]) -> list[Candle]:
    return [
        Candle(timestamp=i * 14400_000, open=p, high=p * 1.005,
               low=p * 0.995, close=p, volume=500.0)
        for i, p in enumerate(prices)
    ]


# ── rsi_series ────────────────────────────────────────────────────────────────

class TestRsiSeries:
    def test_length(self):
        closes = list(range(1, 31, 1))  # 30 valeurs
        result = rsi_series([float(c) for c in closes], period=14)
        assert len(result) == 30

    def test_default_value_short_series(self):
        """Série trop courte → 50.0 par défaut."""
        result = rsi_series([100.0, 101.0], period=14)
        assert all(v == pytest.approx(50.0) for v in result)

    def test_all_up_gives_high_rsi(self):
        """Série strictement haussière → RSI élevé (>= 70)."""
        closes = [float(i * 2 + 100) for i in range(30)]
        result = rsi_series(closes, period=14)
        assert result[-1] >= 70.0

    def test_all_down_gives_low_rsi(self):
        """Série strictement baissière → RSI bas (<= 30)."""
        closes = [float(200 - i * 2) for i in range(30)]
        result = rsi_series(closes, period=14)
        assert result[-1] <= 30.0

    def test_range_0_to_100(self):
        closes = [100.0 + (i % 5) for i in range(30)]
        result = rsi_series(closes, period=14)
        assert all(0.0 <= v <= 100.0 for v in result)


# ── sma_series ────────────────────────────────────────────────────────────────

class TestSmaSeries:
    def test_basic(self):
        values = [1.0, 2.0, 3.0, 4.0, 5.0]
        result = sma_series(values, period=3)
        assert len(result) == 5
        assert result[2] == pytest.approx(2.0)
        assert result[3] == pytest.approx(3.0)
        assert result[4] == pytest.approx(4.0)

    def test_too_short(self):
        """Série plus courte que period → retourne des zéros."""
        result = sma_series([1.0, 2.0], period=5)
        assert all(v == 0.0 for v in result)

    def test_sliding_window(self):
        values = [2.0, 4.0, 6.0, 8.0]
        result = sma_series(values, period=2)
        assert result[1] == pytest.approx(3.0)
        assert result[2] == pytest.approx(5.0)
        assert result[3] == pytest.approx(7.0)


# ── check_first_entry ────────────────────────────────────────────────────────

class TestCheckFirstEntry:
    def test_valid_entry(self):
        """Drop ≥ 5% depuis le high, RSI < 50 → entrée valide."""
        assert check_first_entry(
            close=0.94,
            trailing_high=1.0,
            entry_drop_pct=0.05,
            rsi=40.0,
            rsi_max=50.0,
            volume=1200.0,
            volume_ma=1000.0,
            require_volume=False,
        ) is True

    def test_drop_insufficient(self):
        """Drop < 5% → pas d'entrée."""
        assert check_first_entry(
            close=0.97,
            trailing_high=1.0,
            entry_drop_pct=0.05,
            rsi=40.0,
            rsi_max=50.0,
            volume=1200.0,
            volume_ma=1000.0,
            require_volume=False,
        ) is False

    def test_rsi_too_high(self):
        """RSI > rsi_max → pas d'entrée."""
        assert check_first_entry(
            close=0.90,
            trailing_high=1.0,
            entry_drop_pct=0.05,
            rsi=60.0,
            rsi_max=50.0,
            volume=1200.0,
            volume_ma=1000.0,
            require_volume=False,
        ) is False

    def test_volume_filter_enabled(self):
        """Volume insuffisant avec require_volume=True → pas d'entrée."""
        assert check_first_entry(
            close=0.90,
            trailing_high=1.0,
            entry_drop_pct=0.05,
            rsi=40.0,
            rsi_max=50.0,
            volume=500.0,
            volume_ma=1000.0,
            require_volume=True,
        ) is False

    def test_trailing_high_zero_safe(self):
        """trailing_high=0 → False sans ZeroDivisionError."""
        assert check_first_entry(
            close=100.0,
            trailing_high=0.0,
            entry_drop_pct=0.05,
            rsi=30.0,
            rsi_max=50.0,
            volume=1000.0,
            volume_ma=800.0,
            require_volume=False,
        ) is False


# ── compute_buy_size ──────────────────────────────────────────────────────────

class TestComputeBuySize:
    def test_full_buy_low_rsi(self):
        """RSI < rsi_full → tranche complète."""
        size = compute_buy_size(
            rsi=20.0,
            rsi_full=30.0,
            rsi_half=50.0,
            target_amount=100.0,
            cash_available=500.0,
            max_invested=300.0,
            already_invested=0.0,
        )
        assert size == pytest.approx(100.0)

    def test_half_buy_neutral_rsi(self):
        """30 ≤ RSI < 50 → moitié de la tranche."""
        size = compute_buy_size(
            rsi=40.0,
            rsi_full=30.0,
            rsi_half=50.0,
            target_amount=100.0,
            cash_available=500.0,
            max_invested=300.0,
            already_invested=0.0,
        )
        assert size == pytest.approx(50.0)

    def test_no_buy_high_rsi(self):
        """RSI > rsi_half → 0."""
        size = compute_buy_size(
            rsi=60.0,
            rsi_full=30.0,
            rsi_half=50.0,
            target_amount=100.0,
            cash_available=500.0,
            max_invested=300.0,
            already_invested=0.0,
        )
        assert size == pytest.approx(0.0)

    def test_capped_by_cash(self):
        """Montant limité au cash disponible."""
        size = compute_buy_size(
            rsi=20.0,
            rsi_full=30.0,
            rsi_half=50.0,
            target_amount=100.0,
            cash_available=40.0,
            max_invested=300.0,
            already_invested=0.0,
        )
        assert size == pytest.approx(40.0)

    def test_max_invested_cap(self):
        """Plafonné au budget restant du cycle."""
        size = compute_buy_size(
            rsi=20.0,
            rsi_full=30.0,
            rsi_half=50.0,
            target_amount=100.0,
            cash_available=500.0,
            max_invested=250.0,
            already_invested=200.0,
        )
        assert size == pytest.approx(50.0)  # 250 - 200 = 50 restant

    def test_max_invested_exhausted(self):
        """Budget déjà épuisé → 0."""
        size = compute_buy_size(
            rsi=20.0,
            rsi_full=30.0,
            rsi_half=50.0,
            target_amount=100.0,
            cash_available=500.0,
            max_invested=200.0,
            already_invested=200.0,
        )
        assert size == pytest.approx(0.0)


# ── check_sell_conditions ────────────────────────────────────────────────────

class TestCheckSellConditions:
    def test_sell_level_reached(self):
        assert check_sell_conditions(
            close=1.016,
            pmp=1.0,
            sell_level_pct=0.015,
            rsi=60.0,
            rsi_sell_min=0.0,
        ) is True

    def test_below_sell_level(self):
        assert check_sell_conditions(
            close=1.005,
            pmp=1.0,
            sell_level_pct=0.015,
            rsi=60.0,
            rsi_sell_min=0.0,
        ) is False

    def test_pmp_zero_safe(self):
        """PMP = 0 → False sans erreur."""
        assert check_sell_conditions(
            close=1.0,
            pmp=0.0,
            sell_level_pct=0.015,
            rsi=60.0,
            rsi_sell_min=0.0,
        ) is False


# ── check_override_sell ───────────────────────────────────────────────────────

class TestCheckOverrideSell:
    def test_override_triggered(self):
        """Prix ≥ PMP * 1.20 → override sell."""
        assert check_override_sell(close=1.21, pmp=1.0, override_pct=0.20) is True

    def test_override_not_triggered(self):
        assert check_override_sell(close=1.10, pmp=1.0, override_pct=0.20) is False

    def test_pmp_zero_safe(self):
        assert check_override_sell(close=100.0, pmp=0.0, override_pct=0.20) is False


# ── check_stop_loss ───────────────────────────────────────────────────────────

class TestCheckStopLoss:
    def test_stop_triggered(self):
        """Prix ≤ PMP - 15% → stop-loss."""
        assert check_stop_loss(close=0.84, pmp=1.0, stop_pct=0.15) is True

    def test_stop_not_triggered(self):
        assert check_stop_loss(close=0.90, pmp=1.0, stop_pct=0.15) is False

    def test_pmp_zero_safe(self):
        assert check_stop_loss(close=50.0, pmp=0.0, stop_pct=0.15) is False


# ── InfinityCycle.recalc_pmp ──────────────────────────────────────────────────

class TestInfinityCycle:
    def test_pmp_recalculation(self):
        cycle = InfinityCycle()
        cycle.total_cost = 1000.0
        cycle.total_size = 0.1
        cycle.recalc_pmp()
        assert cycle.pmp == pytest.approx(10000.0)

    def test_pmp_zero_size_safe(self):
        cycle = InfinityCycle()
        cycle.total_cost = 100.0
        cycle.total_size = 0.0
        cycle.recalc_pmp()
        # total_size = 0 → pmp ne doit pas changer (rester 0)
        assert cycle.pmp == pytest.approx(0.0)
