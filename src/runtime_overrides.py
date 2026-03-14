from __future__ import annotations

import time
from typing import Any
from datetime import datetime, timezone

from src.firebase.client import get_document, get_documents, update_document

_HB_CACHE_TTL_SECONDS = 20.0
_hb_cache_doc: dict[str, Any] = {}
_hb_cache_fetched_at: float = 0.0


def _normalize_seconds(value: Any) -> int | None:
    try:
        seconds = int(value)
    except Exception:
        return None
    if seconds <= 0:
        return None
    return seconds


def _get_heartbeat_doc() -> dict[str, Any]:
    global _hb_cache_doc, _hb_cache_fetched_at
    now = time.time()
    if now - _hb_cache_fetched_at < _HB_CACHE_TTL_SECONDS:
        return _hb_cache_doc

    doc = get_document("runtime_overrides", "heartbeat") or {}
    _hb_cache_doc = doc
    _hb_cache_fetched_at = now
    return _hb_cache_doc


def get_heartbeat_override_seconds(bot_key: str, default_seconds: int) -> int:
    seconds = _normalize_seconds(default_seconds) or 600
    try:
        doc = _get_heartbeat_doc()
        values = doc.get("values") if isinstance(doc, dict) else None
        if not isinstance(values, dict):
            return seconds
        override = _normalize_seconds(values.get(bot_key))
        if override is None:
            return seconds
        return override
    except Exception:
        return seconds


def get_all_heartbeat_overrides() -> dict[str, int]:
    try:
        doc = _get_heartbeat_doc()
        values = doc.get("values") if isinstance(doc, dict) else None
        if not isinstance(values, dict):
            return {}
        out: dict[str, int] = {}
        for key, value in values.items():
            sec = _normalize_seconds(value)
            if sec is not None:
                out[str(key)] = sec
        return out
    except Exception:
        return {}


def get_pending_runtime_actions(bot_key: str) -> list[dict[str, Any]]:
    try:
        rows = get_documents(
            "runtime_actions",
            filters=[("status", "==", "pending")],
        )
        out: list[dict[str, Any]] = []
        for row in rows:
            if str(row.get("bot", "")).lower().strip() != bot_key:
                continue
            out.append(row)
        return out
    except Exception:
        return []


def mark_runtime_action_status(action_id: str, status: str, message: str) -> bool:
    try:
        return update_document("runtime_actions", action_id, {
            "status": status,
            "result_message": message,
            "processed_at": datetime.now(timezone.utc).isoformat(),
        })
    except Exception:
        return False
