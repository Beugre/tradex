from __future__ import annotations

from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock

from src.telegram_command_bot import TelegramCommandBot


def _make_bot() -> TelegramCommandBot:
    bot = TelegramCommandBot.__new__(TelegramCommandBot)
    bot._allowed_chat = "42"
    bot._allowed_user_ids = set()
    bot._pending_confirms = {}
    bot._dispatch = MagicMock(return_value="ok")
    bot._send = MagicMock()
    return bot


def test_handle_update_accepts_private_chat_without_user_allowlist():
    bot = _make_bot()

    bot._handle_update({
        "message": {
            "text": "/help",
            "chat": {"id": "42", "type": "private"},
            "from": {"id": 7},
        }
    })

    bot._dispatch.assert_called_once_with("/help")
    bot._send.assert_called_once_with("42", "ok")


def test_handle_update_rejects_group_chat_without_allowed_user():
    bot = _make_bot()

    bot._handle_update({
        "message": {
            "text": "/close now range all",
            "chat": {"id": "42", "type": "group"},
            "from": {"id": 7},
        }
    })

    bot._dispatch.assert_not_called()
    bot._send.assert_not_called()


def test_handle_update_accepts_group_chat_when_user_is_allowlisted():
    bot = _make_bot()
    bot._allowed_user_ids = {"7"}

    bot._handle_update({
        "message": {
            "text": "/health all",
            "chat": {"id": "42", "type": "supergroup"},
            "from": {"id": 7},
        }
    })

    bot._dispatch.assert_called_once_with("/health all")
    bot._send.assert_called_once_with("42", "ok")


def test_confirm_enqueues_close_all_runtime_action():
    bot = _make_bot()
    bot._queue_runtime_action = MagicMock(return_value="action-123")
    bot._pending_confirms["ABC123"] = {
        "action": "close",
        "bot": "range",
        "symbol": "ALL",
        "value": None,
        "expires_at": datetime.now(timezone.utc) + timedelta(seconds=120),
    }

    msg = bot._cmd_confirm(["abc123"])

    bot._queue_runtime_action.assert_called_once_with("range", "close", "ALL", None)
    assert "action-123" in msg
    assert "ABC123" not in bot._pending_confirms
