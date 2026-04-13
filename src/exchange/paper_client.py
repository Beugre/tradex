"""
Paper Trading Exchange Clients.

Wrappers autour des vrais clients exchange qui :
  - Délèguent toutes les opérations de LECTURE (candles, tickers, exchange info)
  - Simulent les opérations d'ÉCRITURE (ordres) avec des fills virtuels
  - Gèrent un solde virtuel persisté sur disque
  - Simulent les OCO orders (Binance) avec vérification de prix

Usage :
    real_client = BinanceClient(api_key=..., secret_key=...)
    paper = PaperBinanceClient(real_client, initial_balance=1000.0, bot_name="range")
    # paper a la même API que BinanceClient
"""

from __future__ import annotations

import json
import logging
import os
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from src.core.models import Balance

logger = logging.getLogger("tradex.paper")

_DATA_DIR = Path(os.getenv("TRADEX_DATA_DIR", "data"))


# ═══════════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════════

def _to_float(v: Any) -> float:
    """Convertit str ou float en float."""
    return float(v) if isinstance(v, str) else float(v)


# ═══════════════════════════════════════════════════════════════════════════════
# Paper Binance Client
# ═══════════════════════════════════════════════════════════════════════════════

class PaperBinanceClient:
    """
    Wrapper paper trading autour de BinanceClient.

    - Lecture (candles, tickers, exchange info, filters) → délégué au vrai client
    - Écriture (ordres, OCO, annulations) → simulé localement
    - Balance → virtuelle, persistée dans data/paper_state_{bot}.json
    """

    def __init__(
        self,
        real_client: Any,
        initial_balance: float = 1000.0,
        bot_name: str = "unknown",
    ) -> None:
        self._real = real_client
        self._bot_name = bot_name
        self._state_path = _DATA_DIR / f"paper_state_{bot_name}.json"

        # Virtual state
        self._balances: dict[str, float] = {}
        self._virtual_ocos: dict[int, dict] = {}
        self._next_order_id: int = 100_000
        self._next_oco_id: int = 200_000

        # Charger l'état persisté ou initialiser
        if self._state_path.exists():
            self._load_state()
            logger.info(
                "📄 [PAPER/%s] State restauré — USDC=%.2f",
                bot_name, self._balances.get("USDC", 0),
            )
        else:
            self._balances = {"USDC": initial_balance}
            self._save_state()
            logger.info(
                "📄 [PAPER/%s] Initialisé — USDC=%.2f",
                bot_name, initial_balance,
            )

    # ── State persistence ─────────────────────────────────────────────────

    def _save_state(self) -> None:
        _DATA_DIR.mkdir(parents=True, exist_ok=True)
        state = {
            "balances": self._balances,
            "virtual_ocos": {str(k): v for k, v in self._virtual_ocos.items()},
            "next_order_id": self._next_order_id,
            "next_oco_id": self._next_oco_id,
        }
        self._state_path.write_text(json.dumps(state, indent=2))

    def _load_state(self) -> None:
        raw = json.loads(self._state_path.read_text())
        self._balances = raw.get("balances", {"USDC": 0})
        self._virtual_ocos = {int(k): v for k, v in raw.get("virtual_ocos", {}).items()}
        self._next_order_id = raw.get("next_order_id", 100_000)
        self._next_oco_id = raw.get("next_oco_id", 200_000)

    # ── Méthodes LECTURE → délégation au vrai client ──────────────────────

    def get_exchange_info(self, symbol: Optional[str] = None) -> dict:
        return self._real.get_exchange_info(symbol)

    def get_symbol_filters(self, symbol: str) -> dict:
        return self._real.get_symbol_filters(symbol)

    def get_all_usdc_pairs(self) -> list[str]:
        return self._real.get_all_usdc_pairs()

    def get_tickers(self, symbols: Optional[list[str]] = None) -> list:
        return self._real.get_tickers(symbols)

    def get_ticker_price(self, symbol: str) -> float:
        return self._real.get_ticker_price(symbol)

    def get_candles(
        self, symbol: str, interval: Optional[int] = None,
        since: Optional[int] = None, limit: int = 200,
    ) -> list:
        return self._real.get_candles(symbol, interval, since, limit)

    def format_quantity(self, symbol: str, quantity: float, *, market: bool = False) -> str:
        return self._real.format_quantity(symbol, quantity, market=market)

    def format_price(self, symbol: str, price: float) -> str:
        return self._real.format_price(symbol, price)

    def check_min_notional(self, symbol: str, quantity: float, price: float) -> bool:
        return self._real.check_min_notional(symbol, quantity, price)

    def close(self) -> None:
        pass  # Ne pas fermer le vrai client

    # ── Méthodes ÉCRITURE → simulation ────────────────────────────────────

    def get_balances(self) -> list[Balance]:
        """Retourne les soldes virtuels."""
        return [
            Balance(currency=asset, available=amount, reserved=0.0, total=amount)
            for asset, amount in self._balances.items()
            if amount > 0.001
        ]

    def place_market_order(
        self, symbol: str, side: str, quantity: str,
        client_order_id: Optional[str] = None,
    ) -> dict:
        """Simule un MARKET order au prix courant."""
        price = self._real.get_ticker_price(symbol)
        order_id = self._next_order_id
        self._next_order_id += 1

        qty = _to_float(quantity)
        quote_qty = qty * price
        base_asset = symbol.replace("USDC", "")

        if side.upper() == "BUY":
            self._balances["USDC"] = self._balances.get("USDC", 0) - quote_qty
            self._balances[base_asset] = self._balances.get(base_asset, 0) + qty
        else:
            self._balances["USDC"] = self._balances.get("USDC", 0) + quote_qty
            self._balances[base_asset] = max(0, self._balances.get(base_asset, 0) - qty)

        self._save_state()

        logger.info(
            "📄 [PAPER/%s] %s %s %s @ %.6f (%.2f USDC)",
            self._bot_name, side.upper(), qty, symbol, price, quote_qty,
        )

        return {
            "symbol": symbol,
            "orderId": order_id,
            "orderListId": -1,
            "clientOrderId": client_order_id or f"paper_{order_id}",
            "transactTime": int(time.time() * 1000),
            "price": "0.00000000",
            "origQty": str(qty),
            "executedQty": str(qty),
            "cummulativeQuoteQty": f"{quote_qty:.8f}",
            "status": "FILLED",
            "timeInForce": "GTC",
            "type": "MARKET",
            "side": side.upper(),
            "fills": [{
                "price": f"{price:.8f}",
                "qty": str(qty),
                "commission": "0.00000000",
                "commissionAsset": "USDC",
            }],
        }

    def place_order(
        self, symbol: str, side: str, quantity: str, price: str,
        order_type: str = "LIMIT", time_in_force: str = "GTC",
        client_order_id: Optional[str] = None,
    ) -> dict:
        """Simule un LIMIT order (fill immédiat au prix spécifié)."""
        order_id = self._next_order_id
        self._next_order_id += 1

        qty = _to_float(quantity)
        p = _to_float(price)
        base_asset = symbol.replace("USDC", "")

        if side.upper() == "BUY":
            self._balances["USDC"] = self._balances.get("USDC", 0) - qty * p
            self._balances[base_asset] = self._balances.get(base_asset, 0) + qty
        else:
            self._balances["USDC"] = self._balances.get("USDC", 0) + qty * p
            self._balances[base_asset] = max(0, self._balances.get(base_asset, 0) - qty)

        self._save_state()

        return {
            "symbol": symbol,
            "orderId": order_id,
            "clientOrderId": client_order_id or f"paper_{order_id}",
            "transactTime": int(time.time() * 1000),
            "price": str(p),
            "origQty": str(qty),
            "executedQty": str(qty),
            "cummulativeQuoteQty": f"{qty * p:.8f}",
            "status": "FILLED",
            "type": order_type,
            "side": side.upper(),
            "timeInForce": time_in_force,
        }

    def place_oco_order(
        self, symbol: str, side: str, quantity: str,
        tp_price: str, sl_stop_price: str, sl_limit_price: str,
        list_client_order_id: Optional[str] = None,
    ) -> dict:
        """Simule un OCO order (stocké virtuellement, vérifié par get_order_list)."""
        oco_id = self._next_oco_id
        self._next_oco_id += 1
        tp_oid = self._next_order_id
        self._next_order_id += 1
        sl_oid = self._next_order_id
        self._next_order_id += 1

        qty = _to_float(quantity)

        self._virtual_ocos[oco_id] = {
            "symbol": symbol,
            "side": side.upper(),
            "quantity": qty,
            "tp_price": _to_float(tp_price),
            "sl_stop_price": _to_float(sl_stop_price),
            "sl_limit_price": _to_float(sl_limit_price),
            "tp_order_id": tp_oid,
            "sl_order_id": sl_oid,
            "filled": False,
            "filled_side": None,
        }
        self._save_state()

        logger.info(
            "📄 [PAPER/%s] OCO %s qty=%.6f TP=%.6f SL=%.6f",
            self._bot_name, symbol, qty,
            _to_float(tp_price), _to_float(sl_stop_price),
        )

        return {
            "orderListId": oco_id,
            "contingencyType": "OCO",
            "listStatusType": "EXEC_STARTED",
            "listOrderStatus": "EXECUTING",
            "symbol": symbol,
            "transactionTime": int(time.time() * 1000),
            "orders": [
                {"symbol": symbol, "orderId": tp_oid, "clientOrderId": f"paper_tp_{oco_id}"},
                {"symbol": symbol, "orderId": sl_oid, "clientOrderId": f"paper_sl_{oco_id}"},
            ],
            "orderReports": [
                {
                    "symbol": symbol, "orderId": tp_oid,
                    "side": side.upper(), "type": "LIMIT_MAKER",
                    "status": "NEW", "price": str(tp_price),
                    "origQty": str(qty), "executedQty": "0.00000000",
                },
                {
                    "symbol": symbol, "orderId": sl_oid,
                    "side": side.upper(), "type": "STOP_LOSS_LIMIT",
                    "status": "NEW", "price": str(sl_limit_price),
                    "stopPrice": str(sl_stop_price),
                    "origQty": str(qty), "executedQty": "0.00000000",
                },
            ],
        }

    def get_order_list(self, order_list_id: int) -> dict:
        """
        Vérifie un OCO virtuel : compare le prix courant aux seuils TP/SL.
        Retourne ALL_DONE si TP ou SL touché, sinon EXEC_STARTED.
        """
        oco = self._virtual_ocos.get(order_list_id)
        if not oco:
            return {"orderListId": order_list_id, "listStatusType": "ALL_DONE"}

        if oco["filled"]:
            return self._oco_done_response(oco, order_list_id)

        # Vérifier le prix courant
        try:
            price = self._real.get_ticker_price(oco["symbol"])
        except Exception:
            return self._oco_active_response(oco, order_list_id)

        if oco["side"] == "SELL":
            if price >= oco["tp_price"]:
                oco["filled"] = True
                oco["filled_side"] = "TP"
                fill_price = oco["tp_price"]
            elif price <= oco["sl_stop_price"]:
                oco["filled"] = True
                oco["filled_side"] = "SL"
                fill_price = oco["sl_limit_price"]
            else:
                return self._oco_active_response(oco, order_list_id)
        else:  # BUY OCO (rare)
            if price <= oco["tp_price"]:
                oco["filled"] = True
                oco["filled_side"] = "TP"
                fill_price = oco["tp_price"]
            elif price >= oco["sl_stop_price"]:
                oco["filled"] = True
                oco["filled_side"] = "SL"
                fill_price = oco["sl_limit_price"]
            else:
                return self._oco_active_response(oco, order_list_id)

        # OCO filled — mettre à jour le solde
        base_asset = oco["symbol"].replace("USDC", "")
        if oco["side"] == "SELL":
            self._balances["USDC"] = self._balances.get("USDC", 0) + oco["quantity"] * fill_price
            self._balances[base_asset] = max(0, self._balances.get(base_asset, 0) - oco["quantity"])
        else:
            self._balances["USDC"] = self._balances.get("USDC", 0) - oco["quantity"] * fill_price
            self._balances[base_asset] = self._balances.get(base_asset, 0) + oco["quantity"]

        self._save_state()

        logger.info(
            "📄 [PAPER/%s] OCO %s %s @ %.6f (%s)",
            self._bot_name, oco["filled_side"], oco["symbol"],
            fill_price, "TP" if oco["filled_side"] == "TP" else "SL",
        )

        return self._oco_done_response(oco, order_list_id)

    def cancel_order(
        self, symbol: str,
        order_id: Optional[int] = None,
        client_order_id: Optional[str] = None,
    ) -> dict:
        return {"symbol": symbol, "orderId": order_id or 0, "status": "CANCELED"}

    def cancel_order_list(
        self, symbol: str,
        order_list_id: Optional[int] = None,
        list_client_order_id: Optional[str] = None,
    ) -> dict:
        if order_list_id and order_list_id in self._virtual_ocos:
            del self._virtual_ocos[order_list_id]
            self._save_state()
        return {
            "orderListId": order_list_id or 0,
            "contingencyType": "OCO",
            "listStatusType": "ALL_DONE",
            "listOrderStatus": "ALL_DONE",
            "symbol": symbol,
        }

    def get_order(
        self, symbol: str,
        order_id: Optional[int] = None,
        client_order_id: Optional[str] = None,
    ) -> dict:
        """Retourne le statut d'un ordre individuel (potentiellement leg d'un OCO)."""
        for oco in self._virtual_ocos.values():
            if order_id == oco["tp_order_id"]:
                filled = oco["filled"] and oco["filled_side"] == "TP"
                canceled = oco["filled"] and oco["filled_side"] == "SL"
                return {
                    "orderId": order_id, "symbol": symbol,
                    "side": oco["side"], "type": "LIMIT_MAKER",
                    "status": "FILLED" if filled else ("CANCELED" if canceled else "NEW"),
                    "price": f"{oco['tp_price']:.8f}",
                    "origQty": f"{oco['quantity']:.8f}",
                    "executedQty": f"{oco['quantity']:.8f}" if filled else "0.00000000",
                }
            if order_id == oco["sl_order_id"]:
                filled = oco["filled"] and oco["filled_side"] == "SL"
                canceled = oco["filled"] and oco["filled_side"] == "TP"
                return {
                    "orderId": order_id, "symbol": symbol,
                    "side": oco["side"], "type": "STOP_LOSS_LIMIT",
                    "status": "FILLED" if filled else ("CANCELED" if canceled else "NEW"),
                    "price": f"{oco['sl_limit_price']:.8f}",
                    "stopPrice": f"{oco['sl_stop_price']:.8f}",
                    "origQty": f"{oco['quantity']:.8f}",
                    "executedQty": f"{oco['quantity']:.8f}" if filled else "0.00000000",
                }
        return {"orderId": order_id or 0, "symbol": symbol, "status": "CANCELED"}

    def get_active_orders(self, symbol: Optional[str] = None) -> list[dict]:
        return []

    def get_active_order_lists(self) -> list[dict]:
        return [
            {
                "orderListId": oco_id,
                "contingencyType": "OCO",
                "listStatusType": "EXEC_STARTED",
                "symbol": oco["symbol"],
            }
            for oco_id, oco in self._virtual_ocos.items()
            if not oco["filled"]
        ]

    # ── Helpers OCO response ──────────────────────────────────────────────

    def _oco_active_response(self, oco: dict, oco_id: int) -> dict:
        return {
            "orderListId": oco_id,
            "contingencyType": "OCO",
            "listStatusType": "EXEC_STARTED",
            "listOrderStatus": "EXECUTING",
            "symbol": oco["symbol"],
            "orders": [
                {"symbol": oco["symbol"], "orderId": oco["tp_order_id"], "clientOrderId": f"paper_tp_{oco_id}"},
                {"symbol": oco["symbol"], "orderId": oco["sl_order_id"], "clientOrderId": f"paper_sl_{oco_id}"},
            ],
        }

    def _oco_done_response(self, oco: dict, oco_id: int) -> dict:
        is_tp = oco["filled_side"] == "TP"
        tp_status = "FILLED" if is_tp else "CANCELED"
        sl_status = "CANCELED" if is_tp else "FILLED"
        tp_exec = f"{oco['quantity']:.8f}" if is_tp else "0.00000000"
        sl_exec = "0.00000000" if is_tp else f"{oco['quantity']:.8f}"

        return {
            "orderListId": oco_id,
            "contingencyType": "OCO",
            "listStatusType": "ALL_DONE",
            "listOrderStatus": "ALL_DONE",
            "symbol": oco["symbol"],
            "orders": [
                {"symbol": oco["symbol"], "orderId": oco["tp_order_id"], "clientOrderId": f"paper_tp_{oco_id}"},
                {"symbol": oco["symbol"], "orderId": oco["sl_order_id"], "clientOrderId": f"paper_sl_{oco_id}"},
            ],
            "orderReports": [
                {
                    "symbol": oco["symbol"], "orderId": oco["tp_order_id"],
                    "side": oco["side"], "type": "LIMIT_MAKER",
                    "status": tp_status,
                    "price": f"{oco['tp_price']:.8f}",
                    "origQty": f"{oco['quantity']:.8f}",
                    "executedQty": tp_exec,
                },
                {
                    "symbol": oco["symbol"], "orderId": oco["sl_order_id"],
                    "side": oco["side"], "type": "STOP_LOSS_LIMIT",
                    "status": sl_status,
                    "price": f"{oco['sl_limit_price']:.8f}",
                    "stopPrice": f"{oco['sl_stop_price']:.8f}",
                    "origQty": f"{oco['quantity']:.8f}",
                    "executedQty": sl_exec,
                },
            ],
        }


# ═══════════════════════════════════════════════════════════════════════════════
# Paper Revolut Client
# ═══════════════════════════════════════════════════════════════════════════════

class PaperRevolutClient:
    """
    Wrapper paper trading autour de RevolutXClient.

    - Lecture (candles, tickers) → délégué au vrai client
    - Écriture (ordres) → simulé localement
    - Balance → virtuelle, persistée dans data/paper_state_{bot}.json
    """

    def __init__(
        self,
        real_client: Any,
        initial_balance: float = 1000.0,
        bot_name: str = "unknown",
    ) -> None:
        self._real = real_client
        self._bot_name = bot_name
        self._state_path = _DATA_DIR / f"paper_state_{bot_name}.json"

        self._balances: dict[str, float] = {}
        self._next_order_id: int = 1

        if self._state_path.exists():
            self._load_state()
            logger.info(
                "📄 [PAPER/%s] State restauré — USD=%.2f",
                bot_name, self._balances.get("USD", 0),
            )
        else:
            self._balances = {"USD": initial_balance}
            self._save_state()
            logger.info(
                "📄 [PAPER/%s] Initialisé — USD=%.2f",
                bot_name, initial_balance,
            )

    # ── State persistence ─────────────────────────────────────────────────

    def _save_state(self) -> None:
        _DATA_DIR.mkdir(parents=True, exist_ok=True)
        state = {
            "balances": self._balances,
            "next_order_id": self._next_order_id,
        }
        self._state_path.write_text(json.dumps(state, indent=2))

    def _load_state(self) -> None:
        raw = json.loads(self._state_path.read_text())
        self._balances = raw.get("balances", {"USD": 0})
        self._next_order_id = raw.get("next_order_id", 1)

    # ── Méthodes LECTURE → délégation ─────────────────────────────────────

    def get_tickers(self, symbols: Optional[list[str]] = None) -> list:
        return self._real.get_tickers(symbols)

    def get_candles(
        self, symbol: str, interval: Optional[int] = None,
        since: Optional[int] = None,
    ) -> list:
        return self._real.get_candles(symbol, interval, since)

    def get_order_fills(self, venue_order_id: str) -> list[dict]:
        return []

    def close(self) -> None:
        pass

    # ── Méthodes ÉCRITURE → simulation ────────────────────────────────────

    def get_balances(self) -> list[Balance]:
        return [
            Balance(currency=asset, available=amount, reserved=0.0, total=amount)
            for asset, amount in self._balances.items()
            if amount > 0.001
        ]

    def place_order(self, order: Any) -> dict:
        """Simule un ordre limit (fill immédiat au prix spécifié)."""
        venue_order_id = f"paper_{self._next_order_id}"
        self._next_order_id += 1

        price = _to_float(order.price)
        size = _to_float(order.base_size)
        base, quote = order.symbol.split("-")

        if order.side.value == "buy":
            self._balances[quote] = self._balances.get(quote, 0) - size * price
            self._balances[base] = self._balances.get(base, 0) + size
        else:
            self._balances[quote] = self._balances.get(quote, 0) + size * price
            self._balances[base] = max(0, self._balances.get(base, 0) - size)

        self._save_state()

        logger.info(
            "📄 [PAPER/%s] %s %s %s @ %.6f",
            self._bot_name, order.side.value.upper(), size, order.symbol, price,
        )

        return {
            "data": {
                "venue_order_id": venue_order_id,
                "state": "FILLED",
                "created_at": int(time.time() * 1000),
                "symbol": order.symbol,
                "side": order.side.value,
                "type": "LIMIT",
                "executed_base_amount": str(size),
                "executed_quote_amount": f"{size * price:.8f}",
            }
        }

    def place_maker_first_order(
        self, order: Any, wait_seconds: int = 30,
    ) -> dict:
        """Simule un maker-first order (fill instantané, type=maker)."""
        result = self.place_order(order)
        return {
            "venue_order_id": result["data"]["venue_order_id"],
            "fill_type": "maker",
            "actual_price": _to_float(order.price),
            "response": result,
            "filled_size": _to_float(order.base_size),
        }

    def cancel_order(self, venue_order_id: str) -> None:
        pass

    def get_active_orders(self, symbols: Optional[list[str]] = None) -> list[dict]:
        return []

    def get_order(self, venue_order_id: str) -> dict:
        return {"data": {"venue_order_id": venue_order_id, "state": "FILLED"}}
