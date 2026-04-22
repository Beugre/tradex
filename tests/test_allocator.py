"""Tests pour allocator.py — répartition Crash/Trail/Listing basée sur le PF."""

import pytest

from src.core.allocator import (
    AllocationRegime,
    AllocationResult,
    compute_allocation,
    compute_profit_factor,
)


class TestComputeProfitFactor:
    def test_basic_pf(self):
        pnl = [10.0, 20.0, -5.0, -10.0]
        # gains = 30, losses = 15 → PF = 2.0
        assert compute_profit_factor(pnl) == pytest.approx(2.0)

    def test_no_losses(self):
        pnl = [10.0, 20.0]
        assert compute_profit_factor(pnl) == float("inf")

    def test_no_gains(self):
        pnl = [-10.0, -5.0]
        assert compute_profit_factor(pnl) == 0.0

    def test_empty_list(self):
        assert compute_profit_factor([]) == 0.0

    def test_all_zeros(self):
        assert compute_profit_factor([0.0, 0.0]) == 0.0


class TestComputeAllocation:
    """Tests de la logique d'allocation Trail/Crash/Listing."""

    # ── Régime DEFENSIVE ──────────────────────────────────────────────────────

    def test_defensive_low_pf(self):
        """PF < 0.9 → régime DEFENSIVE : Trail 5%, Crash 65%, Listing 30%."""
        result = compute_allocation(
            total_balance=1000.0,
            trail_pf=0.8,
            trail_trade_count=25,
        )
        assert result.regime == AllocationRegime.DEFENSIVE
        assert result.trail_pct == pytest.approx(0.05)
        assert result.crash_pct == pytest.approx(0.65)
        assert result.listing_pct == pytest.approx(0.30)

    def test_defensive_few_trades(self):
        """< 20 trades → DEFENSIVE quelle que soit la valeur du PF."""
        result = compute_allocation(
            total_balance=1000.0,
            trail_pf=1.5,
            trail_trade_count=5,
        )
        assert result.regime == AllocationRegime.DEFENSIVE
        assert result.trail_pct == pytest.approx(0.05)

    def test_defensive_exact_boundary(self):
        """PF == 0.9 → NEUTRAL (borne inclusive)."""
        result = compute_allocation(
            total_balance=1000.0,
            trail_pf=0.9,
            trail_trade_count=20,
        )
        assert result.regime == AllocationRegime.NEUTRAL

    # ── Régime NEUTRAL ────────────────────────────────────────────────────────

    def test_neutral_regime(self):
        """0.9 ≤ PF ≤ 1.1 avec ≥ 20 trades → NEUTRAL : Trail 10%, Crash 60%."""
        result = compute_allocation(
            total_balance=2000.0,
            trail_pf=1.0,
            trail_trade_count=30,
        )
        assert result.regime == AllocationRegime.NEUTRAL
        assert result.trail_pct == pytest.approx(0.10)
        assert result.crash_pct == pytest.approx(0.60)
        assert result.trail_balance == pytest.approx(200.0)
        assert result.crash_balance == pytest.approx(1200.0)
        assert result.listing_balance == pytest.approx(600.0)

    def test_neutral_upper_boundary(self):
        """PF == 1.1 → NEUTRAL (borne haute inclusive)."""
        result = compute_allocation(
            total_balance=1000.0,
            trail_pf=1.1,
            trail_trade_count=20,
        )
        assert result.regime == AllocationRegime.NEUTRAL

    # ── Régime AGGRESSIVE ────────────────────────────────────────────────────

    def test_aggressive_regime(self):
        """PF > 1.1 avec ≥ 20 trades → AGGRESSIVE : Trail 20%, Crash 50%."""
        result = compute_allocation(
            total_balance=1000.0,
            trail_pf=1.5,
            trail_trade_count=50,
        )
        assert result.regime == AllocationRegime.AGGRESSIVE
        assert result.trail_pct == pytest.approx(0.20)
        assert result.crash_pct == pytest.approx(0.50)

    # ── Propriétés invariantes ────────────────────────────────────────────────

    def test_allocations_sum_to_one(self):
        """Trail + Crash + Listing == 1.0 dans tous les régimes."""
        for pf, count in [(0.5, 30), (1.0, 25), (1.5, 40)]:
            result = compute_allocation(1000.0, pf, count)
            total = result.trail_pct + result.crash_pct + result.listing_pct
            assert total == pytest.approx(1.0), f"PF={pf}: total={total}"

    def test_balances_sum_to_total(self):
        """trail_balance + crash_balance + listing_balance == total_balance."""
        result = compute_allocation(5000.0, 1.2, 50)
        total = result.trail_balance + result.crash_balance + result.listing_balance
        assert total == pytest.approx(5000.0)

    def test_listing_always_30pct(self):
        """Le Listing Bot reçoit toujours 30% par défaut."""
        for pf, count in [(0.5, 30), (1.0, 25), (1.5, 40)]:
            result = compute_allocation(1000.0, pf, count)
            assert result.listing_pct == pytest.approx(0.30)

    # ── Override trail_pct ────────────────────────────────────────────────────

    def test_trail_pct_override(self):
        """trail_pct_override=0.0 → Trail à 0 (dry-run), Crash absorbe le reste."""
        result = compute_allocation(
            total_balance=1000.0,
            trail_pf=1.5,
            trail_trade_count=50,
            trail_pct_override=0.0,
        )
        assert result.trail_pct == pytest.approx(0.0)
        assert result.crash_pct == pytest.approx(0.70)
        assert result.trail_balance == pytest.approx(0.0)

    # ── Custom listing_pct ────────────────────────────────────────────────────

    def test_custom_listing_pct(self):
        """listing_pct personnalisé est respecté."""
        result = compute_allocation(
            total_balance=1000.0,
            trail_pf=1.5,
            trail_trade_count=50,
            listing_pct=0.40,
        )
        assert result.listing_pct == pytest.approx(0.40)
        total = result.trail_pct + result.crash_pct + result.listing_pct
        assert total == pytest.approx(1.0)
