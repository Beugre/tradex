from __future__ import annotations

from unittest.mock import patch

from src.core.models import OrderSide, Position, PositionStatus, StrategyType
from src.firebase.trade_logger import get_latest_heartbeat, log_heartbeat, log_trade_closed


def test_log_trade_closed_uses_legacy_maker_or_taker_field_for_entry_fees():
    position = Position(
        symbol="BTC-USD",
        side=OrderSide.BUY,
        entry_price=100.0,
        sl_price=90.0,
        size=1.0,
        venue_order_id="order-1",
        status=PositionStatus.OPEN,
        strategy=StrategyType.TREND,
    )

    with patch("src.firebase.trade_logger.get_documents", return_value=[{
        "trade_id": "trade-1",
        "exchange": "revolut",
        "maker_or_taker": "taker",
        "opened_at": "2026-04-14T08:00:00+00:00",
    }]), patch("src.firebase.trade_logger.update_document") as update_document:
        update_document.return_value = True

        ok = log_trade_closed(
            trade_id="trade-1",
            position=position,
            exit_price=110.0,
            reason="TP",
            fill_type="maker",
            equity_after=1010.0,
        )

    assert ok is True
    updates = update_document.call_args[0][2]
    assert updates["fees_entry"] == 0.09
    assert updates["fees_exit"] == 0.0
    assert updates["fees_total"] == 0.09


def test_log_heartbeat_writes_current_status_doc_without_event_archive():
    with patch("src.firebase.trade_logger.add_document") as add_document, \
         patch("src.firebase.trade_logger.log_event") as log_event, \
         patch("src.firebase.trade_logger.config") as mock_config:
        mock_config.FIREBASE_HEARTBEAT_STATUS_ENABLED = True
        mock_config.FIREBASE_HEARTBEAT_EVENT_ARCHIVE_ENABLED = False
        add_document.return_value = "binance"

        doc_id = log_heartbeat(
            open_positions=2,
            total_equity=1234.5,
            total_risk_pct=0.031,
            pairs_count=12,
            exchange="binance",
        )

    assert doc_id == "binance"
    assert add_document.call_args[0][0] == "bot_status"
    assert add_document.call_args[1]["doc_id"] == "binance"
    log_event.assert_not_called()


def test_get_latest_heartbeat_prefers_current_status_doc():
    with patch("src.firebase.trade_logger.get_document", return_value={
        "exchange": "binance",
        "timestamp": "2026-04-14T10:00:00+00:00",
        "data": {"open_positions": 1},
    }), patch("src.firebase.trade_logger.get_documents") as get_documents:
        row = get_latest_heartbeat("binance", max_age_hours=72)

    assert row is not None
    assert row["exchange"] == "binance"
    get_documents.assert_not_called()
