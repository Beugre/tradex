"""Tests for src.core.dca_engine — pure logic, no I/O."""

import pytest

from src.core.dca_engine import (
    CrashLevel,
    DCAConfig,
    DCAState,
    RSIBracket,
    check_crash_triggers,
    classify_rsi,
    compute_buy_size,
    compute_daily_amount,
    compute_rolling_high,
    format_summary,
    is_budget_exhausted,
    remaining_crash_budget,
    remaining_dca_budget,
    reset_crash_levels_if_recovered,
    split_allocation,
)


# ── Fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture
def cfg() -> DCAConfig:
    """Default DCA config for tests."""
    return DCAConfig()


@pytest.fixture
def state() -> DCAState:
    """Fresh DCA state for tests."""
    return DCAState()


# ── classify_rsi ──────────────────────────────────────────────────────────────


class TestClassifyRSI:
    def test_overbought(self, cfg: DCAConfig):
        assert classify_rsi(75.0, cfg) == RSIBracket.OVERBOUGHT
        assert classify_rsi(70.1, cfg) == RSIBracket.OVERBOUGHT

    def test_warm(self, cfg: DCAConfig):
        assert classify_rsi(70.0, cfg) == RSIBracket.WARM  # boundary: <= 70
        assert classify_rsi(60.0, cfg) == RSIBracket.WARM
        assert classify_rsi(55.1, cfg) == RSIBracket.WARM

    def test_neutral(self, cfg: DCAConfig):
        assert classify_rsi(55.0, cfg) == RSIBracket.NEUTRAL  # boundary: <= 55
        assert classify_rsi(50.0, cfg) == RSIBracket.NEUTRAL
        assert classify_rsi(45.0, cfg) == RSIBracket.NEUTRAL  # boundary: >= 45

    def test_oversold(self, cfg: DCAConfig):
        assert classify_rsi(44.9, cfg) == RSIBracket.OVERSOLD
        assert classify_rsi(30.0, cfg) == RSIBracket.OVERSOLD
        assert classify_rsi(0.0, cfg) == RSIBracket.OVERSOLD

    def test_boundary_70(self, cfg: DCAConfig):
        """RSI exactly at 70 → WARM (not overbought: > 70 required)."""
        assert classify_rsi(70.0, cfg) == RSIBracket.WARM

    def test_boundary_45(self, cfg: DCAConfig):
        """RSI exactly at 45 → NEUTRAL (>= 45 required)."""
        assert classify_rsi(45.0, cfg) == RSIBracket.NEUTRAL


# ── compute_daily_amount ──────────────────────────────────────────────────────


class TestComputeDailyAmount:
    def test_overbought_zero(self, cfg: DCAConfig):
        assert compute_daily_amount(75.0, cfg) == 0.0

    def test_warm_1x(self, cfg: DCAConfig):
        assert compute_daily_amount(60.0, cfg) == 12.0  # base × 1

    def test_neutral_2x(self, cfg: DCAConfig):
        assert compute_daily_amount(50.0, cfg) == 24.0  # base × 2

    def test_oversold_3x(self, cfg: DCAConfig):
        assert compute_daily_amount(30.0, cfg) == 36.0  # base × 3


# ── split_allocation ─────────────────────────────────────────────────────────


class TestSplitAllocation:
    def test_default_split(self, cfg: DCAConfig):
        result = split_allocation(24.0, cfg)
        assert result["BTC-USD"] == 19.20  # 80%
        assert result["ETH-USD"] == 4.80   # 20%

    def test_zero_amount(self, cfg: DCAConfig):
        result = split_allocation(0.0, cfg)
        assert result["BTC-USD"] == 0.0
        assert result["ETH-USD"] == 0.0

    def test_small_amount(self, cfg: DCAConfig):
        result = split_allocation(12.0, cfg)
        assert result["BTC-USD"] == 9.60
        assert result["ETH-USD"] == 2.40

    def test_allocations_sum(self, cfg: DCAConfig):
        """Total of BTC + ETH should equal input amount."""
        result = split_allocation(36.0, cfg)
        assert result["BTC-USD"] + result["ETH-USD"] == pytest.approx(36.0, abs=0.01)


# ── compute_rolling_high ─────────────────────────────────────────────────────


class TestComputeRollingHigh:
    def test_basic(self):
        highs = [100.0, 110.0, 105.0, 108.0]
        assert compute_rolling_high(highs, 4) == 110.0

    def test_window_smaller_than_list(self):
        highs = [100.0, 110.0, 105.0, 108.0, 95.0]
        assert compute_rolling_high(highs, 3) == 108.0  # last 3: 105, 108, 95

    def test_window_larger_than_list(self):
        highs = [100.0, 110.0]
        assert compute_rolling_high(highs, 90) == 110.0

    def test_empty_list(self):
        assert compute_rolling_high([], 90) == 0.0

    def test_single_element(self):
        assert compute_rolling_high([42.0], 90) == 42.0


# ── check_crash_triggers ─────────────────────────────────────────────────────


class TestCheckCrashTriggers:
    def test_no_trigger_small_drop(self, cfg: DCAConfig, state: DCAState):
        """Drop of 10% should not trigger any level (min is 15%)."""
        triggers = check_crash_triggers(90.0, 100.0, state, cfg)
        assert triggers == []

    def test_trigger_level_15(self, cfg: DCAConfig, state: DCAState):
        """Drop of 16% should trigger only LEVEL_15."""
        triggers = check_crash_triggers(84.0, 100.0, state, cfg)
        assert len(triggers) == 1
        assert triggers[0] == (0.15, 150.0)

    def test_trigger_level_25(self, cfg: DCAConfig, state: DCAState):
        """Drop of 26% should trigger LEVEL_15 and LEVEL_25."""
        triggers = check_crash_triggers(74.0, 100.0, state, cfg)
        assert len(triggers) == 2
        drop_pcts = [t[0] for t in triggers]
        assert 0.15 in drop_pcts
        assert 0.25 in drop_pcts

    def test_trigger_all_levels(self, cfg: DCAConfig, state: DCAState):
        """Drop of 36% should trigger all 3 levels."""
        triggers = check_crash_triggers(64.0, 100.0, state, cfg)
        assert len(triggers) == 3

    def test_already_triggered_skipped(self, cfg: DCAConfig, state: DCAState):
        """Levels already triggered in state should not re-trigger."""
        state.crash_levels_triggered = ["LEVEL_15"]
        triggers = check_crash_triggers(84.0, 100.0, state, cfg)
        assert triggers == []

    def test_partial_already_triggered(self, cfg: DCAConfig, state: DCAState):
        """Only un-triggered levels should fire."""
        state.crash_levels_triggered = ["LEVEL_15"]
        triggers = check_crash_triggers(74.0, 100.0, state, cfg)
        assert len(triggers) == 1
        assert triggers[0][0] == 0.25  # Only LEVEL_25

    def test_rolling_high_zero(self, cfg: DCAConfig, state: DCAState):
        """Rolling high of 0 should return no triggers (guard clause)."""
        triggers = check_crash_triggers(50.0, 0.0, state, cfg)
        assert triggers == []

    def test_capped_by_remaining_reserve(self, cfg: DCAConfig, state: DCAState):
        """If crash reserve is almost empty, amount should be capped."""
        state.total_spent_crash = 1050.0  # Only $50 left of $1100
        triggers = check_crash_triggers(84.0, 100.0, state, cfg)
        assert len(triggers) == 1
        assert triggers[0][1] == 50.0  # Capped at remaining $50

    def test_no_trigger_if_reserve_empty(self, cfg: DCAConfig, state: DCAState):
        """No triggers if crash reserve is fully spent."""
        state.total_spent_crash = 1100.0  # $0 remaining
        triggers = check_crash_triggers(64.0, 100.0, state, cfg)
        assert triggers == []


# ── reset_crash_levels_if_recovered ──────────────────────────────────────────


class TestResetCrashLevels:
    def test_reset_when_recovered(self, cfg: DCAConfig, state: DCAState):
        """Price back above -10% → all levels reset."""
        state.crash_levels_triggered = ["LEVEL_15", "LEVEL_25"]
        reset = reset_crash_levels_if_recovered(95.0, 100.0, state, cfg)
        assert reset == ["LEVEL_15", "LEVEL_25"]
        assert state.crash_levels_triggered == []

    def test_no_reset_if_still_down(self, cfg: DCAConfig, state: DCAState):
        """Price still at -12% → no reset."""
        state.crash_levels_triggered = ["LEVEL_15"]
        reset = reset_crash_levels_if_recovered(88.0, 100.0, state, cfg)
        assert reset == []
        assert state.crash_levels_triggered == ["LEVEL_15"]

    def test_no_reset_if_nothing_triggered(self, cfg: DCAConfig, state: DCAState):
        """No triggered levels → nothing to reset."""
        reset = reset_crash_levels_if_recovered(95.0, 100.0, state, cfg)
        assert reset == []

    def test_no_reset_rolling_high_zero(self, cfg: DCAConfig, state: DCAState):
        """Rolling high 0 → guard clause, no crash."""
        state.crash_levels_triggered = ["LEVEL_15"]
        reset = reset_crash_levels_if_recovered(50.0, 0.0, state, cfg)
        assert reset == []

    def test_boundary_exactly_10_pct(self, cfg: DCAConfig, state: DCAState):
        """Price exactly at -10% → drop == 0.10 ≥ threshold → NO reset."""
        state.crash_levels_triggered = ["LEVEL_15"]
        reset = reset_crash_levels_if_recovered(90.0, 100.0, state, cfg)
        assert reset == []
        assert state.crash_levels_triggered == ["LEVEL_15"]


# ── Budget functions ─────────────────────────────────────────────────────────


class TestBudget:
    def test_remaining_dca_fresh(self, state: DCAState, cfg: DCAConfig):
        assert remaining_dca_budget(state, cfg) == 4200.0

    def test_remaining_dca_after_spending(self, state: DCAState, cfg: DCAConfig):
        state.total_spent_dca = 1000.0
        assert remaining_dca_budget(state, cfg) == 3200.0

    def test_remaining_dca_overspent(self, state: DCAState, cfg: DCAConfig):
        state.total_spent_dca = 5000.0
        assert remaining_dca_budget(state, cfg) == 0.0

    def test_remaining_crash_fresh(self, state: DCAState, cfg: DCAConfig):
        assert remaining_crash_budget(state, cfg) == 1100.0

    def test_remaining_crash_after_spending(self, state: DCAState, cfg: DCAConfig):
        state.total_spent_crash = 400.0
        assert remaining_crash_budget(state, cfg) == 700.0

    def test_not_exhausted_fresh(self, state: DCAState, cfg: DCAConfig):
        assert not is_budget_exhausted(state, cfg)

    def test_exhausted_dca_only(self, state: DCAState, cfg: DCAConfig):
        """DCA exhausted but crash still available → NOT exhausted."""
        state.total_spent_dca = 4200.0
        assert not is_budget_exhausted(state, cfg)

    def test_exhausted_fully(self, state: DCAState, cfg: DCAConfig):
        """Both budgets exhausted → IS exhausted."""
        state.total_spent_dca = 4200.0
        state.total_spent_crash = 1100.0
        assert is_budget_exhausted(state, cfg)


# ── compute_buy_size ─────────────────────────────────────────────────────────


class TestComputeBuySize:
    def test_normal_btc(self):
        size = compute_buy_size(19.20, 67500.0)
        assert size == pytest.approx(0.00028444, abs=1e-7)

    def test_normal_eth(self):
        size = compute_buy_size(4.80, 3400.0)
        assert size == pytest.approx(0.00141176, abs=1e-7)

    def test_zero_price(self):
        assert compute_buy_size(100.0, 0.0) == 0.0

    def test_zero_amount(self):
        assert compute_buy_size(0.0, 67500.0) == 0.0

    def test_negative_price(self):
        assert compute_buy_size(100.0, -1.0) == 0.0


# ── DCAState serialization ──────────────────────────────────────────────────


class TestDCAStateSerialization:
    def test_roundtrip(self):
        s = DCAState(
            total_spent_dca=500.0,
            total_spent_crash=150.0,
            total_btc_bought=0.01,
            total_eth_bought=0.05,
            last_buy_date="2025-01-15",
            last_buy_rsi=48.0,
            last_buy_bracket="NEUTRAL",
            crash_levels_triggered=["LEVEL_15"],
            rolling_high_price=72000.0,
            buy_count=30,
            crash_buy_count=1,
            total_days_active=45,
            start_date="2025-01-01",
        )
        d = s.to_dict()
        restored = DCAState.from_dict(d)

        assert restored.total_spent_dca == 500.0
        assert restored.total_spent_crash == 150.0
        assert restored.total_btc_bought == 0.01
        assert restored.total_eth_bought == 0.05
        assert restored.last_buy_date == "2025-01-15"
        assert restored.last_buy_rsi == 48.0
        assert restored.last_buy_bracket == "NEUTRAL"
        assert restored.crash_levels_triggered == ["LEVEL_15"]
        assert restored.rolling_high_price == 72000.0
        assert restored.buy_count == 30
        assert restored.crash_buy_count == 1
        assert restored.total_days_active == 45
        assert restored.start_date == "2025-01-01"

    def test_from_empty_dict(self):
        s = DCAState.from_dict({})
        assert s.total_spent_dca == 0.0
        assert s.crash_levels_triggered == []
        assert s.buy_count == 0

    def test_from_partial_dict(self):
        s = DCAState.from_dict({"total_spent_dca": 100.0, "buy_count": 5})
        assert s.total_spent_dca == 100.0
        assert s.buy_count == 5
        assert s.total_eth_bought == 0.0  # default


# ── format_summary ───────────────────────────────────────────────────────────


class TestFormatSummary:
    def test_fresh_state(self, state: DCAState, cfg: DCAConfig):
        summary = format_summary(state, cfg)
        assert summary["total_capital"] == 5300.0
        assert summary["total_spent"] == 0.0
        assert summary["remaining"] == 5300.0
        assert summary["dca_remaining"] == 4200.0
        assert summary["crash_remaining"] == 1100.0
        assert summary["btc_accumulated"] == 0.0
        assert summary["eth_accumulated"] == 0.0

    def test_after_spending(self, cfg: DCAConfig):
        state = DCAState(
            total_spent_dca=1200.0,
            total_spent_crash=150.0,
            total_btc_bought=0.005,
            total_eth_bought=0.02,
            buy_count=50,
            crash_buy_count=1,
        )
        summary = format_summary(state, cfg)
        assert summary["total_spent"] == 1350.0
        assert summary["remaining"] == 3950.0
        assert summary["dca_remaining"] == 3000.0
        assert summary["crash_remaining"] == 950.0
        assert summary["btc_accumulated"] == 0.005
        assert summary["buy_count"] == 50
        assert summary["crash_buy_count"] == 1
