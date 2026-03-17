"""Tests for src.core.dca_engine v2 — pure logic, no I/O."""

import pytest

from src.core.dca_engine import (
    DCAConfig,
    DCADecision,
    DCAState,
    MarketRegime,
    RSIBracket,
    check_boost_cooldown,
    check_crash_triggers,
    check_spending_caps,
    classify_regime,
    classify_rsi,
    compute_buy_size,
    compute_crash_anchor,
    compute_daily_amount,
    compute_mvrv_multiplier,
    compute_regime_allocation,
    compute_rolling_high,
    format_summary,
    is_budget_exhausted,
    remaining_crash_budget,
    remaining_dca_budget,
    reset_crash_levels_if_recovered,
    reset_period_counters,
    split_allocation,
)


# ── Fixtures ──


@pytest.fixture
def cfg() -> DCAConfig:
    """Default DCA config for tests (v2 defaults)."""
    return DCAConfig()


@pytest.fixture
def state() -> DCAState:
    """Fresh DCA state for tests."""
    return DCAState()


# ── RSIBracket enum ──


class TestRSIBracketEnum:
    def test_no_deep_value(self):
        """v2: DEEP_VALUE bracket removed."""
        brackets = [b.value for b in RSIBracket]
        assert "DEEP_VALUE" not in brackets

    def test_expected_brackets(self):
        brackets = {b.value for b in RSIBracket}
        assert brackets == {"OVERBOUGHT", "WARM", "NEUTRAL", "OVERSOLD"}


# ── MarketRegime enum ──


class TestMarketRegimeEnum:
    def test_values(self):
        assert MarketRegime.NORMAL.value == "NORMAL"
        assert MarketRegime.WEAK.value == "WEAK"
        assert MarketRegime.CAPITULATION.value == "CAPITULATION"


# ── classify_rsi ──


class TestClassifyRSI:
    def test_overbought(self, cfg: DCAConfig):
        assert classify_rsi(75.0, cfg) == RSIBracket.OVERBOUGHT
        assert classify_rsi(70.1, cfg) == RSIBracket.OVERBOUGHT

    def test_warm(self, cfg: DCAConfig):
        assert classify_rsi(70.0, cfg) == RSIBracket.WARM
        assert classify_rsi(60.0, cfg) == RSIBracket.WARM
        assert classify_rsi(55.1, cfg) == RSIBracket.WARM

    def test_neutral(self, cfg: DCAConfig):
        assert classify_rsi(55.0, cfg) == RSIBracket.NEUTRAL
        assert classify_rsi(50.0, cfg) == RSIBracket.NEUTRAL
        assert classify_rsi(45.0, cfg) == RSIBracket.NEUTRAL

    def test_oversold(self, cfg: DCAConfig):
        assert classify_rsi(44.9, cfg) == RSIBracket.OVERSOLD
        assert classify_rsi(30.0, cfg) == RSIBracket.OVERSOLD
        assert classify_rsi(0.0, cfg) == RSIBracket.OVERSOLD

    def test_no_mvrv_param(self, cfg: DCAConfig):
        """v2: classify_rsi takes NO mvrv parameter."""
        import inspect
        sig = inspect.signature(classify_rsi)
        params = list(sig.parameters.keys())
        assert "mvrv" not in params


# ── compute_mvrv_multiplier ──


class TestMvrvMultiplier:
    def test_disabled(self, cfg: DCAConfig):
        cfg.mvrv_enabled = False
        assert compute_mvrv_multiplier(0.5, cfg) == 1.0

    def test_none_mvrv(self, cfg: DCAConfig):
        assert compute_mvrv_multiplier(None, cfg) == 1.0

    def test_above_threshold(self, cfg: DCAConfig):
        assert compute_mvrv_multiplier(1.5, cfg) == 1.0

    def test_exact_threshold(self, cfg: DCAConfig):
        assert compute_mvrv_multiplier(1.0, cfg) == 1.0

    def test_below_threshold(self, cfg: DCAConfig):
        assert compute_mvrv_multiplier(0.95, cfg) == 1.5
        assert compute_mvrv_multiplier(0.90, cfg) == 1.5
        assert compute_mvrv_multiplier(0.85, cfg) == 1.5

    def test_deep_value(self, cfg: DCAConfig):
        assert compute_mvrv_multiplier(0.80, cfg) == 2.0
        assert compute_mvrv_multiplier(0.50, cfg) == 2.0


# ── classify_regime ──


class TestClassifyRegime:
    def test_above_ma200(self, cfg: DCAConfig):
        assert classify_regime(70000, 65000, cfg) == MarketRegime.NORMAL

    def test_below_ma200(self, cfg: DCAConfig):
        assert classify_regime(60000, 65000, cfg) == MarketRegime.WEAK

    def test_capitulation(self, cfg: DCAConfig):
        assert classify_regime(50000, 65000, cfg) == MarketRegime.CAPITULATION
        assert classify_regime(55249, 65000, cfg) == MarketRegime.CAPITULATION

    def test_boundary_capitulation(self, cfg: DCAConfig):
        assert classify_regime(55250, 65000, cfg) == MarketRegime.WEAK

    def test_disabled(self, cfg: DCAConfig):
        cfg.regime_filter_enabled = False
        assert classify_regime(50000, 65000, cfg) == MarketRegime.NORMAL

    def test_ma200_zero(self, cfg: DCAConfig):
        assert classify_regime(50000, 0, cfg) == MarketRegime.NORMAL

    def test_ma200_negative(self, cfg: DCAConfig):
        assert classify_regime(50000, -1, cfg) == MarketRegime.NORMAL


# ── compute_regime_allocation ──


class TestRegimeAllocation:
    def test_normal(self, cfg: DCAConfig):
        btc, eth = compute_regime_allocation(MarketRegime.NORMAL, cfg)
        assert btc == 0.90
        assert eth == 0.10

    def test_weak(self, cfg: DCAConfig):
        btc, eth = compute_regime_allocation(MarketRegime.WEAK, cfg)
        assert btc == 0.95
        assert eth == 0.05

    def test_capitulation(self, cfg: DCAConfig):
        btc, eth = compute_regime_allocation(MarketRegime.CAPITULATION, cfg)
        assert btc == 1.00
        assert eth == 0.00

    def test_unknown_regime_fallback(self, cfg: DCAConfig):
        cfg.regime_alloc = {}
        btc, eth = compute_regime_allocation(MarketRegime.NORMAL, cfg)
        assert btc == cfg.btc_alloc
        assert eth == cfg.eth_alloc


# ── reset_period_counters ──


class TestResetCounters:
    def test_same_period(self, state: DCAState):
        state.current_month = "2025-04"
        state.current_week = "2025-W14"
        state.monthly_spent = 500
        state.weekly_spent = 200
        reset_period_counters(state, "2025-04", "2025-W14")
        assert state.monthly_spent == 500
        assert state.weekly_spent == 200

    def test_new_month(self, state: DCAState):
        state.current_month = "2025-03"
        state.monthly_spent = 500
        state.current_week = "2025-W14"
        state.weekly_spent = 200
        reset_period_counters(state, "2025-04", "2025-W14")
        assert state.monthly_spent == 0.0
        assert state.current_month == "2025-04"
        assert state.weekly_spent == 200

    def test_new_week(self, state: DCAState):
        state.current_month = "2025-04"
        state.monthly_spent = 500
        state.current_week = "2025-W13"
        state.weekly_spent = 200
        reset_period_counters(state, "2025-04", "2025-W14")
        assert state.weekly_spent == 0.0
        assert state.current_week == "2025-W14"
        assert state.monthly_spent == 500

    def test_both_new(self, state: DCAState):
        state.current_month = "2025-03"
        state.monthly_spent = 500
        state.current_week = "2025-W13"
        state.weekly_spent = 200
        reset_period_counters(state, "2025-04", "2025-W14")
        assert state.monthly_spent == 0.0
        assert state.weekly_spent == 0.0


# ── check_spending_caps ──


class TestSpendingCaps:
    def test_under_caps(self, state: DCAState, cfg: DCAConfig):
        amount, capped = check_spending_caps(100, state, cfg)
        assert amount == 100
        assert not capped

    def test_monthly_cap(self, state: DCAState, cfg: DCAConfig):
        state.monthly_spent = 1450
        amount, capped = check_spending_caps(100, state, cfg)
        assert amount == 50
        assert capped

    def test_weekly_cap(self, state: DCAState, cfg: DCAConfig):
        state.weekly_spent = 380
        amount, capped = check_spending_caps(100, state, cfg)
        assert amount == 20
        assert capped

    def test_both_caps_weekly_tighter(self, state: DCAState, cfg: DCAConfig):
        state.monthly_spent = 1200
        state.weekly_spent = 390
        amount, capped = check_spending_caps(100, state, cfg)
        assert amount == 10
        assert capped

    def test_zero_remaining(self, state: DCAState, cfg: DCAConfig):
        state.monthly_spent = 1500
        amount, capped = check_spending_caps(100, state, cfg)
        assert amount == 0
        assert capped

    def test_exact_cap(self, state: DCAState, cfg: DCAConfig):
        state.monthly_spent = 1400
        amount, capped = check_spending_caps(100, state, cfg)
        assert amount == 100
        assert not capped


# ── check_boost_cooldown ──


class TestBoostCooldown:
    def test_below_threshold(self, state: DCAState, cfg: DCAConfig):
        assert not check_boost_cooldown(50, state, cfg, now_ts=1000)

    def test_no_previous_boost(self, state: DCAState, cfg: DCAConfig):
        assert not check_boost_cooldown(150, state, cfg, now_ts=1000)

    def test_cooldown_active(self, state: DCAState, cfg: DCAConfig):
        state.last_boost_ts = 1000
        assert check_boost_cooldown(150, state, cfg, now_ts=1000 + 12 * 3600)

    def test_cooldown_expired(self, state: DCAState, cfg: DCAConfig):
        state.last_boost_ts = 1000
        assert not check_boost_cooldown(150, state, cfg, now_ts=1000 + 25 * 3600)

    def test_cooldown_exact_boundary(self, state: DCAState, cfg: DCAConfig):
        state.last_boost_ts = 1000
        assert not check_boost_cooldown(
            150, state, cfg, now_ts=1000 + 24 * 3600
        )


# ── compute_daily_amount ──


class TestComputeDailyAmount:
    def test_returns_tuple(self, cfg: DCAConfig):
        result = compute_daily_amount(60, cfg)
        assert isinstance(result, tuple)
        assert len(result) == 3

    def test_overbought_skip(self, cfg: DCAConfig):
        amount, reason, mult = compute_daily_amount(75, cfg)
        assert amount == 0
        assert "OVERBOUGHT" in reason
        assert mult == 1.0

    def test_warm_1x(self, cfg: DCAConfig):
        amount, reason, mult = compute_daily_amount(60, cfg)
        assert amount == 30

    def test_neutral_2x(self, cfg: DCAConfig):
        amount, reason, mult = compute_daily_amount(50, cfg)
        assert amount == 60

    def test_oversold_3x(self, cfg: DCAConfig):
        amount, reason, mult = compute_daily_amount(30, cfg)
        assert amount == 90

    def test_mvrv_boost_low(self, cfg: DCAConfig):
        amount, reason, mult = compute_daily_amount(30, cfg, mvrv=0.95)
        assert amount == 135
        assert mult == 1.5
        assert "MVRV" in reason

    def test_mvrv_boost_deep(self, cfg: DCAConfig):
        amount, reason, mult = compute_daily_amount(30, cfg, mvrv=0.80)
        assert amount == 150
        assert mult == 2.0

    def test_max_daily_cap(self, cfg: DCAConfig):
        cfg.max_daily_buy = 100
        amount, reason, mult = compute_daily_amount(30, cfg, mvrv=0.80)
        assert amount == 100

    def test_with_spending_caps(self, cfg: DCAConfig):
        st = DCAState(weekly_spent=390)
        amount, reason, mult = compute_daily_amount(50, cfg, state=st)
        assert amount == 10
        assert "CAP" in reason

    def test_boost_cooldown(self, cfg: DCAConfig):
        st = DCAState(last_boost_ts=1000)
        amount, reason, mult = compute_daily_amount(
            30, cfg, mvrv=0.95, state=st, now_ts=1000 + 12 * 3600
        )
        assert amount == 30
        assert "COOLDOWN" in reason

    def test_no_mvrv_no_boost(self, cfg: DCAConfig):
        amount, reason, mult = compute_daily_amount(50, cfg, mvrv=None)
        assert amount == 60
        assert mult == 1.0


# ── compute_crash_anchor ──


class TestCrashAnchor:
    def test_max_of_windows(self):
        highs = [100] * 90 + [80] * 90
        anchor = compute_crash_anchor(highs, 90, 180)
        assert anchor == 100

    def test_short_window_higher(self):
        highs = [80] * 90 + [100] * 90
        anchor = compute_crash_anchor(highs, 90, 180)
        assert anchor == 100

    def test_empty(self):
        assert compute_crash_anchor([], 90, 180) == 0.0

    def test_single_element(self):
        assert compute_crash_anchor([42.0], 90, 180) == 42.0


# ── compute_rolling_high ──


class TestComputeRollingHigh:
    def test_basic(self):
        highs = [100.0, 110.0, 105.0, 108.0]
        assert compute_rolling_high(highs, 4) == 110.0

    def test_window_smaller_than_list(self):
        highs = [100.0, 110.0, 105.0, 108.0, 95.0]
        assert compute_rolling_high(highs, 3) == 108.0

    def test_window_larger_than_list(self):
        highs = [100.0, 110.0]
        assert compute_rolling_high(highs, 90) == 110.0

    def test_empty_list(self):
        assert compute_rolling_high([], 90) == 0.0

    def test_single_element(self):
        assert compute_rolling_high([42.0], 90) == 42.0


# ── split_allocation ──


class TestSplitAllocation:
    def test_default_split(self, cfg: DCAConfig):
        result = split_allocation(100.0, cfg)
        assert result["BTC-USD"] == 90.0
        assert result["ETH-USD"] == 10.0

    def test_zero_amount(self, cfg: DCAConfig):
        result = split_allocation(0.0, cfg)
        assert result["BTC-USD"] == 0.0
        assert result["ETH-USD"] == 0.0

    def test_allocations_sum(self, cfg: DCAConfig):
        result = split_allocation(36.0, cfg)
        assert result["BTC-USD"] + result["ETH-USD"] == pytest.approx(36.0, abs=0.01)

    def test_regime_normal(self, cfg: DCAConfig):
        result = split_allocation(100.0, cfg, regime=MarketRegime.NORMAL)
        assert result["BTC-USD"] == 90.0
        assert result["ETH-USD"] == 10.0

    def test_regime_weak(self, cfg: DCAConfig):
        result = split_allocation(100.0, cfg, regime=MarketRegime.WEAK)
        assert result["BTC-USD"] == 95.0
        assert result["ETH-USD"] == 5.0

    def test_regime_capitulation(self, cfg: DCAConfig):
        result = split_allocation(100.0, cfg, regime=MarketRegime.CAPITULATION)
        assert result["BTC-USD"] == 100.0
        assert result["ETH-USD"] == 0.0


# ── check_crash_triggers ──


class TestCheckCrashTriggers:
    def test_no_trigger_small_drop(self, cfg: DCAConfig, state: DCAState):
        triggers = check_crash_triggers(90.0, 100.0, state, cfg)
        assert triggers == []

    def test_trigger_level_15(self, cfg: DCAConfig, state: DCAState):
        triggers = check_crash_triggers(84.0, 100.0, state, cfg)
        assert len(triggers) == 1
        assert triggers[0][0] == 0.15
        assert triggers[0][1] == pytest.approx(275.0)

    def test_trigger_two_levels(self, cfg: DCAConfig, state: DCAState):
        triggers = check_crash_triggers(74.0, 100.0, state, cfg)
        assert len(triggers) == 2
        drop_pcts = [t[0] for t in triggers]
        assert 0.15 in drop_pcts
        assert 0.25 in drop_pcts

    def test_trigger_all_levels(self, cfg: DCAConfig, state: DCAState):
        triggers = check_crash_triggers(64.0, 100.0, state, cfg)
        assert len(triggers) == 3

    def test_proportional_amounts(self, cfg: DCAConfig, state: DCAState):
        triggers = check_crash_triggers(60.0, 100.0, state, cfg)
        amounts = {t[0]: t[1] for t in triggers}
        assert amounts[0.15] == pytest.approx(0.25 * 1100)
        assert amounts[0.25] == pytest.approx(0.35 * 1100)
        assert amounts[0.35] == pytest.approx(0.40 * 1100)

    def test_already_triggered_skipped(self, cfg: DCAConfig, state: DCAState):
        state.crash_levels_triggered = ["LEVEL_15"]
        triggers = check_crash_triggers(84.0, 100.0, state, cfg)
        assert triggers == []

    def test_partial_already_triggered(self, cfg: DCAConfig, state: DCAState):
        state.crash_levels_triggered = ["LEVEL_15"]
        triggers = check_crash_triggers(74.0, 100.0, state, cfg)
        assert len(triggers) == 1
        assert triggers[0][0] == 0.25

    def test_rolling_high_zero(self, cfg: DCAConfig, state: DCAState):
        triggers = check_crash_triggers(50.0, 0.0, state, cfg)
        assert triggers == []

    def test_capped_by_remaining_reserve(self, cfg: DCAConfig, state: DCAState):
        state.total_spent_crash = 1050.0
        triggers = check_crash_triggers(84.0, 100.0, state, cfg)
        assert len(triggers) == 1
        assert triggers[0][1] == 50.0

    def test_no_trigger_if_reserve_empty(self, cfg: DCAConfig, state: DCAState):
        state.total_spent_crash = 1100.0
        triggers = check_crash_triggers(64.0, 100.0, state, cfg)
        assert triggers == []


# ── reset_crash_levels_if_recovered ──


class TestResetCrashLevels:
    def test_reset_when_recovered(self, cfg: DCAConfig, state: DCAState):
        state.crash_levels_triggered = ["LEVEL_15", "LEVEL_25"]
        reset = reset_crash_levels_if_recovered(95.0, 100.0, state, cfg)
        assert reset == ["LEVEL_15", "LEVEL_25"]
        assert state.crash_levels_triggered == []

    def test_no_reset_if_still_down(self, cfg: DCAConfig, state: DCAState):
        state.crash_levels_triggered = ["LEVEL_15"]
        reset = reset_crash_levels_if_recovered(88.0, 100.0, state, cfg)
        assert reset == []
        assert state.crash_levels_triggered == ["LEVEL_15"]

    def test_no_reset_if_nothing_triggered(self, cfg: DCAConfig, state: DCAState):
        reset = reset_crash_levels_if_recovered(95.0, 100.0, state, cfg)
        assert reset == []

    def test_no_reset_rolling_high_zero(self, cfg: DCAConfig, state: DCAState):
        state.crash_levels_triggered = ["LEVEL_15"]
        reset = reset_crash_levels_if_recovered(50.0, 0.0, state, cfg)
        assert reset == []

    def test_boundary_exactly_10_pct(self, cfg: DCAConfig, state: DCAState):
        state.crash_levels_triggered = ["LEVEL_15"]
        reset = reset_crash_levels_if_recovered(90.0, 100.0, state, cfg)
        assert reset == []
        assert state.crash_levels_triggered == ["LEVEL_15"]


# ── Budget functions ──


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
        state.total_spent_dca = 4200.0
        assert not is_budget_exhausted(state, cfg)

    def test_exhausted_fully(self, state: DCAState, cfg: DCAConfig):
        state.total_spent_dca = 4200.0
        state.total_spent_crash = 1100.0
        assert is_budget_exhausted(state, cfg)


# ── compute_buy_size ──


class TestComputeBuySize:
    def test_normal_btc(self):
        size = compute_buy_size(90.0, 67500.0)
        assert size == pytest.approx(0.00133333, abs=1e-7)

    def test_normal_eth(self):
        size = compute_buy_size(10.0, 3400.0)
        assert size == pytest.approx(0.00294118, abs=1e-7)

    def test_zero_price(self):
        assert compute_buy_size(100.0, 0.0) == 0.0

    def test_zero_amount(self):
        assert compute_buy_size(0.0, 67500.0) == 0.0

    def test_negative_price(self):
        assert compute_buy_size(100.0, -1.0) == 0.0


# ── DCAState serialization ──


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
            monthly_spent=200.0,
            current_month="2025-04",
            weekly_spent=80.0,
            current_week="2025-W14",
            last_boost_ts=99999.0,
        )
        d = s.to_dict()
        restored = DCAState.from_dict(d)

        assert restored.total_spent_dca == 500.0
        assert restored.total_spent_crash == 150.0
        assert restored.total_btc_bought == 0.01
        assert restored.total_eth_bought == 0.05
        assert restored.last_buy_date == "2025-01-15"
        assert restored.crash_levels_triggered == ["LEVEL_15"]
        assert restored.monthly_spent == 200.0
        assert restored.current_month == "2025-04"
        assert restored.weekly_spent == 80.0
        assert restored.current_week == "2025-W14"
        assert restored.last_boost_ts == 99999.0

    def test_from_empty_dict(self):
        s = DCAState.from_dict({})
        assert s.total_spent_dca == 0.0
        assert s.crash_levels_triggered == []
        assert s.buy_count == 0

    def test_backwards_compatible(self):
        """v1 state dict (no v2 fields) loads with defaults."""
        d = {"total_spent_dca": 100.0, "buy_count": 5, "crash_levels_triggered": ["LEVEL_15"]}
        s = DCAState.from_dict(d)
        assert s.total_spent_dca == 100.0
        assert s.buy_count == 5
        assert s.monthly_spent == 0.0
        assert s.current_month == ""
        assert s.weekly_spent == 0.0
        assert s.last_boost_ts == 0.0


# ── DCADecision ──


class TestDCADecision:
    def test_to_dict(self):
        dec = DCADecision(
            date="2025-04-01",
            rsi=50.0,
            bracket="NEUTRAL",
            mvrv=0.95,
            mvrv_mult=1.5,
            regime="WEAK",
            base_amount=60,
            mvrv_amount=90,
            capped_amount=90,
            reason="RSI 50 -> NEUTRAL | MVRV 0.95 -> x1.5",
        )
        d = dec.to_dict()
        assert d["date"] == "2025-04-01"
        assert d["rsi"] == 50.0
        assert d["mvrv_mult"] == 1.5
        assert d["regime"] == "WEAK"
        assert d["cap_limited"] is False
        assert d["skipped"] is False

    def test_default_values(self):
        dec = DCADecision()
        d = dec.to_dict()
        assert d["mvrv_mult"] == 1.0
        assert d["regime"] == "NORMAL"
        assert d["skipped"] is False


# ── format_summary ──


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
        st = DCAState(
            total_spent_dca=1200.0,
            total_spent_crash=150.0,
            total_btc_bought=0.005,
            total_eth_bought=0.02,
            buy_count=50,
            crash_buy_count=1,
        )
        summary = format_summary(st, cfg)
        assert summary["total_spent"] == 1350.0
        assert summary["remaining"] == 3950.0
        assert summary["dca_remaining"] == 3000.0
        assert summary["crash_remaining"] == 950.0

    def test_includes_caps(self, cfg: DCAConfig):
        st = DCAState(monthly_spent=200, weekly_spent=50)
        summary = format_summary(st, cfg)
        assert summary["monthly_spent"] == 200
        assert summary["weekly_spent"] == 50
        assert summary["monthly_cap"] == 1500.0
        assert summary["weekly_cap"] == 400.0
