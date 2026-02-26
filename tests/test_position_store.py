"""Tests pour la persistance des positions (position_store.py)."""

from __future__ import annotations

import json
import os
import tempfile

import pytest

from src.core.models import (
    OrderSide,
    Position,
    PositionStatus,
    RangeState,
    StrategyType,
)
from src.core.position_store import PositionStore


@pytest.fixture
def tmp_state_file(tmp_path):
    """Fichier d'état temporaire pour les tests."""
    return str(tmp_path / "state.json")


@pytest.fixture
def sample_position():
    """Position d'exemple."""
    return Position(
        symbol="BTC-USD",
        side=OrderSide.BUY,
        entry_price=95000.0,
        sl_price=92000.0,
        size=0.001,
        venue_order_id="abc-123",
        status=PositionStatus.OPEN,
        strategy=StrategyType.TREND,
        is_zero_risk_applied=False,
        zero_risk_sl=None,
        peak_price=96000.0,
        tp_price=None,
        pnl=None,
    )


@pytest.fixture
def sample_range():
    """Range d'exemple."""
    return RangeState(
        symbol="XRP-USD",
        range_high=0.65,
        range_low=0.55,
        is_valid=True,
        cooldown_until=1700000000,
    )


# ── Tests to_dict / from_dict ─────────────────────────────────────────────────


class TestPositionSerialization:
    """Tests de sérialisation/désérialisation de Position."""

    def test_to_dict_basic(self, sample_position):
        d = sample_position.to_dict()
        assert d["symbol"] == "BTC-USD"
        assert d["side"] == "buy"
        assert d["entry_price"] == 95000.0
        assert d["sl_price"] == 92000.0
        assert d["size"] == 0.001
        assert d["venue_order_id"] == "abc-123"
        assert d["status"] == "OPEN"
        assert d["strategy"] == "TREND"
        assert d["is_zero_risk_applied"] is False
        assert d["peak_price"] == 96000.0
        assert d["tp_price"] is None

    def test_from_dict_roundtrip(self, sample_position):
        d = sample_position.to_dict()
        restored = Position.from_dict(d)
        assert restored.symbol == sample_position.symbol
        assert restored.side == sample_position.side
        assert restored.entry_price == sample_position.entry_price
        assert restored.sl_price == sample_position.sl_price
        assert restored.size == sample_position.size
        assert restored.venue_order_id == sample_position.venue_order_id
        assert restored.status == sample_position.status
        assert restored.strategy == sample_position.strategy
        assert restored.is_zero_risk_applied == sample_position.is_zero_risk_applied
        assert restored.peak_price == sample_position.peak_price

    def test_from_dict_with_zero_risk(self):
        d = {
            "symbol": "SOL-USD",
            "side": "sell",
            "entry_price": 150.0,
            "sl_price": 160.0,
            "size": 10.0,
            "venue_order_id": "xyz-789",
            "status": "ZERO_RISK",
            "strategy": "RANGE",
            "is_zero_risk_applied": True,
            "zero_risk_sl": 148.0,
            "peak_price": 145.0,
            "tp_price": 155.0,
            "pnl": None,
        }
        pos = Position.from_dict(d)
        assert pos.status == PositionStatus.ZERO_RISK
        assert pos.strategy == StrategyType.RANGE
        assert pos.is_zero_risk_applied is True
        assert pos.zero_risk_sl == 148.0
        assert pos.tp_price == 155.0

    def test_from_dict_missing_optional_fields(self):
        """from_dict doit tolérer l'absence de champs optionnels (rétrocompatibilité)."""
        d = {
            "symbol": "BTC-USD",
            "side": "buy",
            "entry_price": 90000.0,
            "sl_price": 88000.0,
            "size": 0.002,
            "venue_order_id": "old-order",
            "status": "OPEN",
            # Pas de strategy, is_zero_risk_applied, etc.
        }
        pos = Position.from_dict(d)
        assert pos.strategy == StrategyType.TREND  # Défaut
        assert pos.is_zero_risk_applied is False
        assert pos.zero_risk_sl is None
        assert pos.tp_price is None


class TestRangeStateSerialization:
    """Tests de sérialisation/désérialisation de RangeState."""

    def test_to_dict(self, sample_range):
        d = sample_range.to_dict()
        assert d["symbol"] == "XRP-USD"
        assert d["range_high"] == 0.65
        assert d["range_low"] == 0.55
        assert d["is_valid"] is True
        assert d["cooldown_until"] == 1700000000

    def test_from_dict_roundtrip(self, sample_range):
        d = sample_range.to_dict()
        restored = RangeState.from_dict(d)
        assert restored.symbol == sample_range.symbol
        assert restored.range_high == sample_range.range_high
        assert restored.range_low == sample_range.range_low
        assert restored.is_valid == sample_range.is_valid
        assert restored.cooldown_until == sample_range.cooldown_until


# ── Tests PositionStore ────────────────────────────────────────────────────────


class TestPositionStore:
    """Tests de sauvegarde et chargement."""

    def test_save_and_load_positions(self, tmp_state_file, sample_position):
        store = PositionStore(tmp_state_file)
        positions = {"BTC-USD": sample_position}
        ranges: dict[str, RangeState] = {}

        store.save(positions, ranges)
        loaded_pos, loaded_rng = store.load()

        assert "BTC-USD" in loaded_pos
        assert loaded_pos["BTC-USD"].entry_price == 95000.0
        assert loaded_pos["BTC-USD"].side == OrderSide.BUY
        assert len(loaded_rng) == 0

    def test_save_and_load_ranges(self, tmp_state_file, sample_range):
        store = PositionStore(tmp_state_file)
        positions: dict[str, Position] = {}
        ranges = {"XRP-USD": sample_range}

        store.save(positions, ranges)
        loaded_pos, loaded_rng = store.load()

        assert len(loaded_pos) == 0
        assert "XRP-USD" in loaded_rng
        assert loaded_rng["XRP-USD"].range_high == 0.65
        assert loaded_rng["XRP-USD"].cooldown_until == 1700000000

    def test_save_and_load_both(self, tmp_state_file, sample_position, sample_range):
        store = PositionStore(tmp_state_file)
        positions = {"BTC-USD": sample_position}
        ranges = {"XRP-USD": sample_range}

        store.save(positions, ranges)
        loaded_pos, loaded_rng = store.load()

        assert len(loaded_pos) == 1
        assert len(loaded_rng) == 1

    def test_load_no_file(self, tmp_state_file):
        """Sans fichier, load() doit retourner des dicts vides."""
        store = PositionStore(tmp_state_file)
        loaded_pos, loaded_rng = store.load()
        assert loaded_pos == {}
        assert loaded_rng == {}

    def test_load_corrupted_json(self, tmp_state_file):
        """Un fichier corrompu ne doit pas crasher, juste retourner des dicts vides."""
        with open(tmp_state_file, "w") as f:
            f.write("{invalid json!!!}")
        store = PositionStore(tmp_state_file)
        loaded_pos, loaded_rng = store.load()
        assert loaded_pos == {}
        assert loaded_rng == {}

    def test_load_corrupted_position(self, tmp_state_file):
        """Une position corrompue est ignorée, les autres sont chargées."""
        state = {
            "positions": {
                "BTC-USD": {"symbol": "BTC-USD"},  # Manque des champs obligatoires
                "SOL-USD": {
                    "symbol": "SOL-USD",
                    "side": "buy",
                    "entry_price": 150.0,
                    "sl_price": 140.0,
                    "size": 5.0,
                    "venue_order_id": "ok-123",
                    "status": "OPEN",
                },
            },
            "ranges": {},
        }
        with open(tmp_state_file, "w") as f:
            json.dump(state, f)
        store = PositionStore(tmp_state_file)
        loaded_pos, _ = store.load()
        # BTC-USD corrompu → ignoré, SOL-USD valide → chargé
        assert "BTC-USD" not in loaded_pos
        assert "SOL-USD" in loaded_pos

    def test_overwrite_state(self, tmp_state_file, sample_position):
        """Une sauvegarde écrase la précédente."""
        store = PositionStore(tmp_state_file)
        store.save({"BTC-USD": sample_position}, {})

        # Modifier et resauvegarder
        sample_position.status = PositionStatus.ZERO_RISK
        sample_position.zero_risk_sl = 93500.0
        store.save({"BTC-USD": sample_position}, {})

        loaded_pos, _ = store.load()
        assert loaded_pos["BTC-USD"].status == PositionStatus.ZERO_RISK
        assert loaded_pos["BTC-USD"].zero_risk_sl == 93500.0

    def test_atomic_write(self, tmp_state_file, sample_position):
        """Le fichier temporaire ne doit pas rester après une sauvegarde réussie."""
        store = PositionStore(tmp_state_file)
        store.save({"BTC-USD": sample_position}, {})
        tmp_file = tmp_state_file + ".tmp"
        assert not os.path.exists(tmp_file)
        assert os.path.exists(tmp_state_file)

    def test_multiple_positions(self, tmp_state_file):
        """Sauvegarder et charger plusieurs positions."""
        pos1 = Position(
            symbol="BTC-USD", side=OrderSide.BUY, entry_price=95000.0,
            sl_price=92000.0, size=0.001, venue_order_id="id1",
            status=PositionStatus.OPEN, strategy=StrategyType.TREND,
        )
        pos2 = Position(
            symbol="SOL-USD", side=OrderSide.SELL, entry_price=150.0,
            sl_price=160.0, size=10.0, venue_order_id="id2",
            status=PositionStatus.ZERO_RISK, strategy=StrategyType.RANGE,
            tp_price=155.0,
        )
        store = PositionStore(tmp_state_file)
        store.save({"BTC-USD": pos1, "SOL-USD": pos2}, {})
        loaded_pos, _ = store.load()
        assert len(loaded_pos) == 2
        assert loaded_pos["BTC-USD"].side == OrderSide.BUY
        assert loaded_pos["SOL-USD"].side == OrderSide.SELL
        assert loaded_pos["SOL-USD"].tp_price == 155.0
