"""
Tests unitaires pour src/core/listing_detector.py

Couvre :
- detect_new_symbols
- check_momentum
- compute_oco_levels
- should_rearm_oco
- compute_rearm_oco_levels
- should_force_close
- compute_position_size
"""

import pytest

from src.core.listing_detector import (
    ListingSignal,
    OCOLevels,
    RearmOCOLevels,
    check_momentum,
    compute_oco_levels,
    compute_position_size,
    compute_rearm_oco_levels,
    detect_new_symbols,
    should_force_close,
    should_rearm_oco,
)


# ═══════════════════════════════════════════════════════════════════════════════
#  detect_new_symbols
# ═══════════════════════════════════════════════════════════════════════════════

class TestDetectNewSymbols:
    def test_no_new_symbols(self):
        current = {"BTCUSDC", "ETHUSDC"}
        known = {"BTCUSDC", "ETHUSDC"}
        assert detect_new_symbols(current, known) == []

    def test_one_new_symbol(self):
        current = {"BTCUSDC", "ETHUSDC", "NEWUSDC"}
        known = {"BTCUSDC", "ETHUSDC"}
        assert detect_new_symbols(current, known) == ["NEWUSDC"]

    def test_multiple_new_symbols_sorted(self):
        current = {"BTCUSDC", "ETHUSDC", "ZZUSDC", "AAUSDC"}
        known = {"BTCUSDC", "ETHUSDC"}
        result = detect_new_symbols(current, known)
        assert result == ["AAUSDC", "ZZUSDC"]

    def test_empty_known(self):
        current = {"BTCUSDC", "ETHUSDC"}
        known: set[str] = set()
        result = detect_new_symbols(current, known)
        assert len(result) == 2

    def test_empty_current(self):
        known = {"BTCUSDC"}
        result = detect_new_symbols(set(), known)
        assert result == []


# ═══════════════════════════════════════════════════════════════════════════════
#  check_momentum
# ═══════════════════════════════════════════════════════════════════════════════

class TestCheckMomentum:
    def _make_candles(self, open_price: float, highs: list[float]) -> list[dict]:
        """Helper : génère des bougies 1m fictives."""
        candles = []
        for i, h in enumerate(highs):
            candles.append({
                "symbol": "TESTUSDC",
                "open": open_price if i == 0 else open_price * 1.01,
                "high": h,
                "low": open_price * 0.99,
                "close": h * 0.99,
                "volume": 100000,
            })
        return candles

    def test_strong_momentum_passes(self):
        """Pump de +50% dans la 1ère minute → OK."""
        candles = self._make_candles(1.0, [1.50])
        result = check_momentum(candles, momentum_threshold=0.30, window_minutes=1)
        assert result is not None
        assert result.momentum_ok is True
        assert result.momentum_pct >= 0.30
        assert result.listing_price == 1.0

    def test_weak_momentum_fails(self):
        """Pump de +10% seulement → KO."""
        candles = self._make_candles(1.0, [1.10])
        result = check_momentum(candles, momentum_threshold=0.30, window_minutes=1)
        assert result is not None
        assert result.momentum_ok is False
        assert result.momentum_pct == pytest.approx(0.10, abs=0.01)

    def test_momentum_exact_threshold(self):
        """Pump de exactement +30% → OK."""
        candles = self._make_candles(1.0, [1.30])
        result = check_momentum(candles, momentum_threshold=0.30, window_minutes=1)
        assert result is not None
        assert result.momentum_ok is True

    def test_momentum_multi_window(self):
        """Pump de +30% dans la 2ème minute (window=2) → OK."""
        candles = self._make_candles(1.0, [1.10, 1.35])
        result = check_momentum(candles, momentum_threshold=0.30, window_minutes=2)
        assert result is not None
        assert result.momentum_ok is True

    def test_momentum_multi_window_too_late(self):
        """Pump de +30% dans la 3ème minute mais window=2 → KO."""
        candles = self._make_candles(1.0, [1.05, 1.10, 1.35])
        result = check_momentum(candles, momentum_threshold=0.30, window_minutes=2)
        assert result is not None
        assert result.momentum_ok is False

    def test_empty_candles(self):
        result = check_momentum([], momentum_threshold=0.30)
        assert result is None

    def test_zero_open_price(self):
        candles = [{"symbol": "X", "open": 0, "high": 0, "low": 0, "close": 0, "volume": 0}]
        result = check_momentum(candles, momentum_threshold=0.30)
        assert result is None


# ═══════════════════════════════════════════════════════════════════════════════
#  compute_oco_levels
# ═══════════════════════════════════════════════════════════════════════════════

class TestComputeOCOLevels:
    def test_default_levels(self):
        oco = compute_oco_levels(entry_price=100.0, sl_pct=0.08, tp_pct=0.30)
        assert oco.sl_price == pytest.approx(92.0)
        assert oco.tp_price == pytest.approx(130.0)
        assert oco.sl_pct == pytest.approx(-0.08)
        assert oco.tp_pct == pytest.approx(0.30)

    def test_small_price(self):
        oco = compute_oco_levels(entry_price=0.001234, sl_pct=0.08, tp_pct=0.30)
        assert oco.sl_price == pytest.approx(0.001234 * 0.92, rel=1e-6)
        assert oco.tp_price == pytest.approx(0.001234 * 1.30, rel=1e-6)


# ═══════════════════════════════════════════════════════════════════════════════
#  should_rearm_oco
# ═══════════════════════════════════════════════════════════════════════════════

class TestShouldRearmOCO:
    def test_price_above_threshold(self):
        # TP = 130, seuil = 130 * 0.98 = 127.4
        assert should_rearm_oco(128.0, 130.0, 0.98) is True

    def test_price_exactly_at_threshold(self):
        assert should_rearm_oco(127.4, 130.0, 0.98) is True

    def test_price_below_threshold(self):
        assert should_rearm_oco(120.0, 130.0, 0.98) is False


# ═══════════════════════════════════════════════════════════════════════════════
#  compute_rearm_oco_levels
# ═══════════════════════════════════════════════════════════════════════════════

class TestComputeRearmOCOLevels:
    def test_default_rearm(self):
        # Entry = 100, TP1 = 130
        # SL2 = 130 * 0.769 = 99.97 ≈ +0% vs entry (break-even)
        # TP2 = 130 * 1.538 = 199.94 ≈ +100% vs entry
        rearm = compute_rearm_oco_levels(
            entry_price=100.0,
            tp1_price=130.0,
            sl2_tp1_mult=0.769,
            tp2_tp1_mult=1.538,
        )
        assert rearm.sl_price == pytest.approx(99.97, abs=0.1)
        assert rearm.tp_price == pytest.approx(199.94, abs=0.1)
        # SL2 verrouille quasiment breakeven
        assert rearm.sl_pct_vs_entry == pytest.approx(0.0, abs=0.01)
        # TP2 ≈ +100%
        assert rearm.tp_pct_vs_entry == pytest.approx(1.0, abs=0.01)

    def test_rearm_locks_profit(self):
        """SL2 doit être >= entry (verrouille du profit)."""
        rearm = compute_rearm_oco_levels(100.0, 130.0, 0.80, 1.538)
        assert rearm.sl_price >= 100.0  # 130 * 0.80 = 104

    def test_rearm_tp2_above_tp1(self):
        """TP2 doit être > TP1."""
        rearm = compute_rearm_oco_levels(100.0, 130.0, 0.769, 1.538)
        assert rearm.tp_price > 130.0


# ═══════════════════════════════════════════════════════════════════════════════
#  should_force_close
# ═══════════════════════════════════════════════════════════════════════════════

class TestShouldForceClose:
    def test_within_horizon(self):
        entry = 1000000
        current = entry + 6 * 24 * 3600 * 1000  # 6 jours
        assert should_force_close(entry, current, 7) is False

    def test_at_horizon(self):
        entry = 1000000
        current = entry + 7 * 24 * 3600 * 1000  # exactement 7 jours
        assert should_force_close(entry, current, 7) is True

    def test_past_horizon(self):
        entry = 1000000
        current = entry + 10 * 24 * 3600 * 1000
        assert should_force_close(entry, current, 7) is True


# ═══════════════════════════════════════════════════════════════════════════════
#  compute_position_size
# ═══════════════════════════════════════════════════════════════════════════════

class TestComputePositionSize:
    def test_basic_sizing(self):
        # Equity = 1500, cash = 1500, 3 slots → 500 par slot, cap 5000
        size = compute_position_size(1500, 1500, 3, 5000)
        assert size == pytest.approx(500.0)

    def test_cash_limited(self):
        # Equity = 1500, cash = 200 → capped by cash
        size = compute_position_size(1500, 200, 3, 5000)
        assert size == pytest.approx(200.0)

    def test_max_alloc_cap(self):
        # Equity = 30000, cash = 30000, 3 slots = 10000, mais cap $5000
        size = compute_position_size(30000, 30000, 3, 5000)
        assert size == pytest.approx(5000.0)

    def test_no_cap(self):
        # max_alloc=0 → pas de plafond
        size = compute_position_size(30000, 30000, 3, 0)
        assert size == pytest.approx(10000.0)

    def test_zero_equity(self):
        size = compute_position_size(0, 0, 3, 5000)
        assert size == 0.0
