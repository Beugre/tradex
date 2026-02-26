"""Tests pour risk_manager.py — calcul de taille de position et zero-risk."""

import pytest

from src.core.models import Balance, OrderSide, Position, PositionStatus
from src.core.risk_manager import (
    calculate_position_size,
    calculate_zero_risk_sl,
    get_fiat_balance,
    should_apply_zero_risk,
    update_trailing_stop,
)


class TestCalculatePositionSize:
    """Tests du calcul de taille de position."""

    def test_basic_calculation(self):
        """1000 USD, 5% risque, entry=100, sl=95 → size = 50/5 = 10."""
        size = calculate_position_size(
            account_balance=1000.0,
            risk_percent=0.05,
            entry_price=100.0,
            sl_price=95.0,
        )
        assert size == pytest.approx(10.0)

    def test_btc_example(self):
        """1000 USD, 5% risque, entry=95000, sl=93000.

        Idéal : 50/2000 = 0.025 BTC (coût 2375$) > 1000$ capital
        Cappé : 1000/95000 = 0.01053 BTC
        """
        size = calculate_position_size(
            account_balance=1000.0,
            risk_percent=0.05,
            entry_price=95000.0,
            sl_price=93000.0,
        )
        assert size == pytest.approx(1000.0 / 95000.0)

    def test_zero_balance_returns_zero(self):
        size = calculate_position_size(0, 0.05, 100, 95)
        assert size == 0.0

    def test_negative_balance_returns_zero(self):
        size = calculate_position_size(-100, 0.05, 100, 95)
        assert size == 0.0

    def test_same_entry_and_sl_returns_zero(self):
        size = calculate_position_size(1000, 0.05, 100, 100)
        assert size == 0.0

    def test_sell_position_sizing(self):
        """Short : entry=100, sl=105 → distance=5 → 50/5 = 10."""
        size = calculate_position_size(
            account_balance=1000.0,
            risk_percent=0.05,
            entry_price=100.0,
            sl_price=105.0,
        )
        assert size == pytest.approx(10.0)

    def test_position_capped_to_balance_btc(self):
        """1050 USD, 5% risque, BTC à 66000, SL à 63800.

        Idéal : 52.5 / 2200 = 0.02386 BTC (coût ~1575$) > 1050$
        Cappé : 1050 / 66000 = 0.01591 BTC
        """
        size = calculate_position_size(
            account_balance=1050.0,
            risk_percent=0.05,
            entry_price=66000.0,
            sl_price=63800.0,
        )
        expected = 1050.0 / 66000.0  # cappé au capital
        assert size == pytest.approx(expected)

    def test_position_capped_to_balance_sol(self):
        """1050 USD, 5% risque, SOL à 81, SL à 78.

        Idéal : 52.5 / 3 = 17.5 SOL (coût 1417$) > 1050$
        Cappé : 1050 / 81 = 12.963 SOL
        """
        size = calculate_position_size(
            account_balance=1050.0,
            risk_percent=0.05,
            entry_price=81.0,
            sl_price=78.0,
        )
        expected = 1050.0 / 81.0
        assert size == pytest.approx(expected)

    def test_no_cap_when_position_affordable(self):
        """1050 USD, 5% risque, XRP à 1.42, SL à 1.34.

        Idéal : 52.5 / 0.08 = 656.25 XRP (coût 931$) < 1050$ → pas de cap.
        """
        size = calculate_position_size(
            account_balance=1050.0,
            risk_percent=0.05,
            entry_price=1.42,
            sl_price=1.34,
        )
        expected = 52.5 / 0.08
        assert size == pytest.approx(expected)

    # --- Tests avec max_position_percent ---

    def test_cap_at_20_percent_btc(self):
        """1050 USD, max 20%, BTC à 95000.

        Budget max = 1050 * 0.20 = 210$
        Idéal (5% risque, SL=93000) : 52.5/2000 = 0.02625 BTC (coût 2494$) > 210$
        Cappé : 210/95000 ≈ 0.002211 BTC
        """
        size = calculate_position_size(
            account_balance=1050.0,
            risk_percent=0.05,
            entry_price=95000.0,
            sl_price=93000.0,
            max_position_percent=0.20,
        )
        expected = (1050.0 * 0.20) / 95000.0
        assert size == pytest.approx(expected)

    def test_cap_at_20_percent_sol(self):
        """1050 USD, max 20%, SOL à 81, SL=78.

        Budget max = 210$
        Idéal : 52.5/3 = 17.5 SOL (coût 1417$) > 210$
        Cappé : 210/81 ≈ 2.593 SOL
        """
        size = calculate_position_size(
            account_balance=1050.0,
            risk_percent=0.05,
            entry_price=81.0,
            sl_price=78.0,
            max_position_percent=0.20,
        )
        expected = (1050.0 * 0.20) / 81.0
        assert size == pytest.approx(expected)

    def test_cap_at_20_percent_xrp_no_cap(self):
        """1050 USD, max 20%, XRP à 1.42, SL=1.34.

        Budget max = 210$
        Idéal : 52.5/0.08 = 656.25 XRP (coût 931$) > 210$
        Cappé : 210/1.42 ≈ 147.89 XRP
        """
        size = calculate_position_size(
            account_balance=1050.0,
            risk_percent=0.05,
            entry_price=1.42,
            sl_price=1.34,
            max_position_percent=0.20,
        )
        expected = (1050.0 * 0.20) / 1.42
        assert size == pytest.approx(expected)

    def test_no_cap_at_20_percent_small_position(self):
        """1050 USD, max 20%, entry=0.50, SL=0.47.

        Budget max = 210$
        Idéal : 52.5/0.03 = 1750 unités (coût 875$) > 210$
        → Cappé à 210/0.50 = 420 unités
        """
        size = calculate_position_size(
            account_balance=1050.0,
            risk_percent=0.05,
            entry_price=0.50,
            sl_price=0.47,
            max_position_percent=0.20,
        )
        expected = (1050.0 * 0.20) / 0.50
        assert size == pytest.approx(expected)

    def test_affordable_within_20_percent(self):
        """10000 USD, max 20%, entry=1.00, SL=0.95.

        Budget max = 2000$
        Idéal : 500/0.05 = 10000 unités (coût 10000$) > 2000$
        Cappé : 2000/1.00 = 2000 unités
        """
        size = calculate_position_size(
            account_balance=10000.0,
            risk_percent=0.05,
            entry_price=1.0,
            sl_price=0.95,
            max_position_percent=0.20,
        )
        expected = 2000.0 / 1.0
        assert size == pytest.approx(expected)

    def test_small_risk_fits_within_20_percent(self):
        """10000 USD, max 20%, entry=10.0, SL=9.90.

        Budget max = 2000$
        Idéal : 500/0.10 = 5000 (coût 50000$) > 2000$
        Cappé : 2000/10 = 200 unités
        """
        size = calculate_position_size(
            account_balance=10000.0,
            risk_percent=0.05,
            entry_price=10.0,
            sl_price=9.90,
            max_position_percent=0.20,
        )
        expected = 2000.0 / 10.0
        assert size == pytest.approx(expected)


class TestGetFiatBalance:
    """Tests d'extraction du solde fiat (USD, EUR, GBP)."""

    def test_finds_usd(self):
        balances = [
            Balance(currency="BTC", available=0.5, reserved=0.0, total=0.5),
            Balance(currency="USD", available=1234.56, reserved=100.0, total=1334.56),
        ]
        amount, currency = get_fiat_balance(balances)
        assert amount == 1234.56
        assert currency == "USD"

    def test_finds_eur_converts_to_usd(self):
        balances = [
            Balance(currency="BTC", available=0.5, reserved=0.0, total=0.5),
            Balance(currency="EUR", available=100.0, reserved=0.0, total=100.0),
        ]
        amount, currency = get_fiat_balance(balances)
        assert amount == pytest.approx(105.0)  # 100 EUR * 1.05
        assert currency == "EUR"

    def test_usd_preferred_over_eur(self):
        balances = [
            Balance(currency="EUR", available=100.0, reserved=0.0, total=100.0),
            Balance(currency="USD", available=50.0, reserved=0.0, total=50.0),
        ]
        amount, currency = get_fiat_balance(balances)
        assert amount == 50.0
        assert currency == "USD"

    def test_no_fiat_returns_zero(self):
        balances = [
            Balance(currency="BTC", available=0.5, reserved=0.0, total=0.5),
        ]
        amount, currency = get_fiat_balance(balances)
        assert amount == 0.0
        assert currency == ""

    def test_empty_balances_returns_zero(self):
        amount, currency = get_fiat_balance([])
        assert amount == 0.0
        assert currency == ""


class TestZeroRisk:
    """Tests de la logique zero-risk."""

    def _make_position(self, side: OrderSide, entry: float, sl: float) -> Position:
        return Position(
            symbol="BTC-USD",
            side=side,
            entry_price=entry,
            sl_price=sl,
            size=0.01,
            venue_order_id="test-123",
            status=PositionStatus.OPEN,
        )

    def test_should_apply_zero_risk_buy(self):
        """Long, entry=100, prix=103 (3% gain) → trigger à 2% → True."""
        pos = self._make_position(OrderSide.BUY, 100, 95)
        assert should_apply_zero_risk(pos, current_price=103, trigger_percent=0.02)

    def test_should_not_apply_zero_risk_insufficient_move(self):
        """Long, entry=100, prix=101 (1% gain) → trigger à 2% → False."""
        pos = self._make_position(OrderSide.BUY, 100, 95)
        assert not should_apply_zero_risk(pos, current_price=101, trigger_percent=0.02)

    def test_should_apply_zero_risk_sell(self):
        """Short, entry=100, prix=97 (3% gain) → trigger à 2% → True."""
        pos = self._make_position(OrderSide.SELL, 100, 105)
        assert should_apply_zero_risk(pos, current_price=97, trigger_percent=0.02)

    def test_already_applied_returns_false(self):
        pos = self._make_position(OrderSide.BUY, 100, 95)
        pos.is_zero_risk_applied = True
        assert not should_apply_zero_risk(pos, current_price=110, trigger_percent=0.02)

    def test_calculate_zero_risk_sl_buy(self):
        """Long, entry=100, lock=0.5% → SL=100.50."""
        pos = self._make_position(OrderSide.BUY, 100, 95)
        new_sl = calculate_zero_risk_sl(pos, lock_percent=0.005)
        assert new_sl == pytest.approx(100.50)

    def test_calculate_zero_risk_sl_sell(self):
        """Short, entry=100, lock=0.5% → SL=99.50."""
        pos = self._make_position(OrderSide.SELL, 100, 105)
        new_sl = calculate_zero_risk_sl(pos, lock_percent=0.005)
        assert new_sl == pytest.approx(99.50)


class TestTrailingStop:
    """Tests du trailing stop après activation du zero-risk."""

    def _make_zr_position(
        self, side: OrderSide, entry: float, sl: float, zero_risk_sl: float
    ) -> Position:
        return Position(
            symbol="BTC-USD",
            side=side,
            entry_price=entry,
            sl_price=sl,
            size=0.01,
            venue_order_id="test-123",
            status=PositionStatus.ZERO_RISK,
            is_zero_risk_applied=True,
            zero_risk_sl=zero_risk_sl,
        )

    def test_not_applied_without_zero_risk(self):
        """Trailing stop ne s'active pas si zero-risk pas encore appliqué."""
        pos = Position(
            symbol="BTC-USD",
            side=OrderSide.BUY,
            entry_price=100,
            sl_price=95,
            size=0.01,
            venue_order_id="test",
            status=PositionStatus.OPEN,
        )
        result = update_trailing_stop(pos, current_price=110, trailing_percent=0.02)
        assert result is None

    def test_buy_trailing_stop_moves_up(self):
        """Long : prix monte à 110, trailing 2% → SL = 110 * 0.98 = 107.80."""
        pos = self._make_zr_position(OrderSide.BUY, 100, 95, 100.5)
        result = update_trailing_stop(pos, current_price=110, trailing_percent=0.02)

        assert result is not None
        assert result == pytest.approx(107.80)
        assert pos.zero_risk_sl == pytest.approx(107.80)
        assert pos.peak_price == 110

    def test_buy_trailing_stop_no_move_down(self):
        """Long : prix descend → SL ne descend pas."""
        pos = self._make_zr_position(OrderSide.BUY, 100, 95, 100.5)
        pos.peak_price = 110
        pos.zero_risk_sl = 107.80

        # Prix redescend à 105, le SL doit rester à 107.80
        result = update_trailing_stop(pos, current_price=105, trailing_percent=0.02)
        assert result is None
        assert pos.zero_risk_sl == 107.80
        assert pos.peak_price == 110  # peak inchangé

    def test_buy_trailing_stop_progressive(self):
        """Long : prix monte progressivement, SL suit."""
        pos = self._make_zr_position(OrderSide.BUY, 100, 95, 100.5)

        # Prix monte à 108 → SL = 108 * 0.98 = 105.84 (> 100.5)
        r1 = update_trailing_stop(pos, current_price=108, trailing_percent=0.02)
        assert r1 == pytest.approx(105.84)

        # Prix monte à 115 → SL = 115 * 0.98 = 112.70 (> 105.84)
        r2 = update_trailing_stop(pos, current_price=115, trailing_percent=0.02)
        assert r2 == pytest.approx(112.70)

        # Prix redescend à 113 → SL inchangé
        r3 = update_trailing_stop(pos, current_price=113, trailing_percent=0.02)
        assert r3 is None
        assert pos.zero_risk_sl == pytest.approx(112.70)

    def test_sell_trailing_stop_moves_down(self):
        """Short : prix descend à 90, trailing 2% → SL = 90 * 1.02 = 91.80."""
        pos = self._make_zr_position(OrderSide.SELL, 100, 105, 99.5)
        result = update_trailing_stop(pos, current_price=90, trailing_percent=0.02)

        assert result is not None
        assert result == pytest.approx(91.80)
        assert pos.zero_risk_sl == pytest.approx(91.80)
        assert pos.peak_price == 90

    def test_sell_trailing_stop_no_move_up(self):
        """Short : prix remonte → SL ne remonte pas."""
        pos = self._make_zr_position(OrderSide.SELL, 100, 105, 99.5)
        pos.peak_price = 90
        pos.zero_risk_sl = 91.80

        # Prix remonte à 95 → SL reste à 91.80
        result = update_trailing_stop(pos, current_price=95, trailing_percent=0.02)
        assert result is None
        assert pos.zero_risk_sl == 91.80

    def test_sell_trailing_stop_progressive(self):
        """Short : prix descend progressivement, SL suit."""
        pos = self._make_zr_position(OrderSide.SELL, 100, 105, 99.5)

        # Prix descend à 95 → SL = 95 * 1.02 = 96.90 (< 99.5)
        r1 = update_trailing_stop(pos, current_price=95, trailing_percent=0.02)
        assert r1 == pytest.approx(96.90)

        # Prix descend à 88 → SL = 88 * 1.02 = 89.76 (< 96.90)
        r2 = update_trailing_stop(pos, current_price=88, trailing_percent=0.02)
        assert r2 == pytest.approx(89.76)

        # Prix remonte à 92 → SL inchangé
        r3 = update_trailing_stop(pos, current_price=92, trailing_percent=0.02)
        assert r3 is None
        assert pos.zero_risk_sl == pytest.approx(89.76)

    def test_no_zero_risk_sl_returns_none(self):
        """Si zero_risk_sl est None, pas de trailing."""
        pos = Position(
            symbol="BTC-USD",
            side=OrderSide.BUY,
            entry_price=100,
            sl_price=95,
            size=0.01,
            venue_order_id="test",
            status=PositionStatus.ZERO_RISK,
            is_zero_risk_applied=True,
            zero_risk_sl=None,
        )
        result = update_trailing_stop(pos, current_price=110, trailing_percent=0.02)
        assert result is None
