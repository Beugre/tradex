"""
Client HTTP pour l'API Binance Spot.

Gère l'authentification HMAC-SHA256 et expose des méthodes typées
pour chaque endpoint utilisé par le bot.

Supporte les ordres OCO natifs (SL + TP en un seul appel) — avantage
majeur par rapport à Revolut X qui nécessite une simulation.

Doc API : https://developers.binance.com/docs/binance-spot-api-docs/rest-api
"""

from __future__ import annotations

import hashlib
import hmac
import logging
import time
import uuid
from typing import Any, Optional
from urllib.parse import urlencode

import httpx

from src.core.models import Balance, Candle, OrderSide, TickerData

logger = logging.getLogger(__name__)

# Intervalles Binance pour les klines
_INTERVAL_MAP = {
    240: "4h",   # H4
    60: "1h",
    15: "15m",
    5: "5m",
    1: "1m",
    1440: "1d",
}


class BinanceClient:
    """Client pour l'API REST Binance Spot avec authentification HMAC-SHA256."""

    def __init__(
        self,
        api_key: str,
        secret_key: str,
        base_url: str = "https://api.binance.com",
    ) -> None:
        self._api_key = api_key
        self._secret_key = secret_key
        self._base_url = base_url.rstrip("/")
        self._client = httpx.Client(timeout=30.0)

        # Cache exchange info (filters, lot size, tick size, etc.)
        self._symbol_filters: dict[str, dict] = {}

    # ═══════════════════════════════════════════════════════════════════════════
    # ENDPOINTS PUBLICS (pas de signature requise)
    # ═══════════════════════════════════════════════════════════════════════════

    def get_exchange_info(self, symbol: Optional[str] = None) -> dict:
        """GET /api/v3/exchangeInfo → infos de la paire (filtres, précisions)."""
        params = {}
        if symbol:
            params["symbol"] = symbol
        return self._request("GET", "/api/v3/exchangeInfo", params=params, signed=False)

    def get_symbol_filters(self, symbol: str) -> dict:
        """Retourne les filtres (LOT_SIZE, PRICE_FILTER, MIN_NOTIONAL) pour un symbole."""
        if symbol in self._symbol_filters:
            return self._symbol_filters[symbol]

        info = self.get_exchange_info(symbol=symbol)
        for sym_info in info.get("symbols", []):
            if sym_info["symbol"] == symbol:
                filters = {}
                for f in sym_info.get("filters", []):
                    filters[f["filterType"]] = f
                filters["_baseAssetPrecision"] = sym_info.get("baseAssetPrecision", 8)
                filters["_quoteAssetPrecision"] = sym_info.get("quoteAssetPrecision", 8)
                filters["_baseAsset"] = sym_info.get("baseAsset", "")
                filters["_quoteAsset"] = sym_info.get("quoteAsset", "")
                filters["_status"] = sym_info.get("status", "TRADING")
                self._symbol_filters[symbol] = filters
                return filters

        logger.warning("Symbol %s introuvable dans exchangeInfo", symbol)
        return {}

    def get_all_usdc_pairs(self) -> list[str]:
        """Retourne toutes les paires USDC en état TRADING sur Binance."""
        info = self.get_exchange_info()
        pairs = []
        for sym in info.get("symbols", []):
            if (
                sym.get("quoteAsset") == "USDC"
                and sym.get("status") == "TRADING"
                and sym.get("isSpotTradingAllowed", False)
            ):
                pairs.append(sym["symbol"])
        pairs.sort()
        logger.info("Binance: %d paires USDC TRADING trouvées", len(pairs))
        return pairs

    def get_tickers(
        self, symbols: Optional[list[str]] = None
    ) -> list[TickerData]:
        """GET /api/v3/ticker/bookTicker → prix temps réel (bid/ask)."""
        params: dict[str, Any] = {}
        if symbols:
            if len(symbols) == 1:
                params["symbol"] = symbols[0]
            else:
                # Binance attend un JSON array pour multiple symbols
                params["symbols"] = str(symbols).replace("'", '"')

        resp = self._request("GET", "/api/v3/ticker/bookTicker", params=params, signed=False)

        # Normaliser en liste
        if isinstance(resp, dict):
            resp = [resp]

        tickers = []
        for t in resp:
            bid = float(t.get("bidPrice", 0))
            ask = float(t.get("askPrice", 0))
            mid = (bid + ask) / 2 if bid and ask else 0
            tickers.append(TickerData(
                symbol=t["symbol"],
                bid=bid,
                ask=ask,
                mid=mid,
                last_price=mid,  # bookTicker n'a pas de lastPrice → mid
            ))
        return tickers

    def get_ticker_price(self, symbol: str) -> float:
        """GET /api/v3/ticker/price → dernier prix d'un seul symbole."""
        resp = self._request(
            "GET", "/api/v3/ticker/price",
            params={"symbol": symbol}, signed=False,
        )
        return float(resp.get("price", 0))

    def get_candles(
        self,
        symbol: str,
        interval: Optional[int] = None,
        since: Optional[int] = None,
        limit: int = 200,
    ) -> list[Candle]:
        """
        GET /api/v3/klines → bougies OHLCV.

        Args:
            symbol: Paire (ex: "BTCUSDC").
            interval: Durée en minutes (ex: 240 pour H4). Défaut: 240.
            since: Timestamp Unix ms à partir duquel récupérer les données.
            limit: Nombre max de bougies (1-1000, défaut: 200).
        """
        interval_min = interval or 240
        interval_str = _INTERVAL_MAP.get(interval_min, "4h")

        params: dict[str, Any] = {
            "symbol": symbol,
            "interval": interval_str,
            "limit": limit,
        }
        if since is not None:
            params["startTime"] = since

        resp = self._request("GET", "/api/v3/klines", params=params, signed=False)

        candles = []
        for k in resp:
            # Format Binance kline: [open_time, O, H, L, C, volume, close_time, ...]
            candles.append(Candle(
                timestamp=int(k[0]),
                open=float(k[1]),
                high=float(k[2]),
                low=float(k[3]),
                close=float(k[4]),
                volume=float(k[5]),
            ))
        return candles

    # ═══════════════════════════════════════════════════════════════════════════
    # ENDPOINTS SIGNÉS (authentifiés)
    # ═══════════════════════════════════════════════════════════════════════════

    def get_balances(self) -> list[Balance]:
        """GET /api/v3/account → liste des soldes."""
        data = self._request("GET", "/api/v3/account", signed=True)
        balances = []
        for b in data.get("balances", []):
            free = float(b["free"])
            locked = float(b["locked"])
            total = free + locked
            if total > 0:
                balances.append(Balance(
                    currency=b["asset"],
                    available=free,
                    reserved=locked,
                    total=total,
                ))
        return balances

    def place_order(
        self,
        symbol: str,
        side: str,
        quantity: str,
        price: str,
        order_type: str = "LIMIT",
        time_in_force: str = "GTC",
        client_order_id: Optional[str] = None,
    ) -> dict:
        """
        POST /api/v3/order → placer un ordre.

        Args:
            symbol: ex "BTCUSDC"
            side: "BUY" ou "SELL"
            quantity: taille en base asset
            price: prix limit
            order_type: LIMIT, MARKET, STOP_LOSS_LIMIT, TAKE_PROFIT_LIMIT
            time_in_force: GTC, IOC, FOK
            client_order_id: ID client optionnel
        """
        params: dict[str, Any] = {
            "symbol": symbol,
            "side": side.upper(),
            "type": order_type,
            "quantity": quantity,
            "newOrderRespType": "FULL",
        }

        if order_type in ("LIMIT", "STOP_LOSS_LIMIT", "TAKE_PROFIT_LIMIT"):
            params["price"] = price
            params["timeInForce"] = time_in_force

        if client_order_id:
            params["newClientOrderId"] = client_order_id

        resp = self._request("POST", "/api/v3/order", params=params, signed=True)
        logger.info(
            "Ordre placé: %s %s %s @ %s → orderId=%s status=%s",
            side, symbol, quantity, price,
            resp.get("orderId"), resp.get("status"),
        )
        return resp

    def place_market_order(
        self,
        symbol: str,
        side: str,
        quantity: str,
        client_order_id: Optional[str] = None,
    ) -> dict:
        """Ordre MARKET (exécution immédiate au prix du marché)."""
        params: dict[str, Any] = {
            "symbol": symbol,
            "side": side.upper(),
            "type": "MARKET",
            "quantity": quantity,
            "newOrderRespType": "FULL",
        }
        if client_order_id:
            params["newClientOrderId"] = client_order_id

        resp = self._request("POST", "/api/v3/order", params=params, signed=True)
        logger.info(
            "Ordre MARKET placé: %s %s %s → orderId=%s status=%s",
            side, symbol, quantity,
            resp.get("orderId"), resp.get("status"),
        )
        return resp

    def place_oco_order(
        self,
        symbol: str,
        side: str,
        quantity: str,
        tp_price: str,
        sl_stop_price: str,
        sl_limit_price: str,
        list_client_order_id: Optional[str] = None,
    ) -> dict:
        """
        POST /api/v3/orderList/oco → Ordre OCO natif (SL + TP en un appel).

        Pour un SELL OCO (après un BUY d'entrée) :
          - above = TAKE_PROFIT_LIMIT (TP) → quand le prix MONTE au-dessus
          - below = STOP_LOSS_LIMIT (SL) → quand le prix DESCEND en-dessous

        Pour un BUY OCO (après un SELL d'entrée) :
          - above = STOP_LOSS_LIMIT (SL) → quand le prix MONTE au-dessus
          - below = TAKE_PROFIT_LIMIT (TP) → quand le prix DESCEND en-dessous

        Quand un des deux ordres est déclenché, l'autre est annulé automatiquement.
        La quantité est identique pour les deux jambes → PAS de dust.

        Args:
            symbol: ex "BTCUSDC"
            side: "SELL" (pour fermer un BUY) ou "BUY" (pour fermer un SELL)
            quantity: taille en base asset (même pour les deux jambes)
            tp_price: prix Take Profit
            sl_stop_price: prix de déclenchement du SL
            sl_limit_price: prix limit du SL (légèrement au-delà du stop)
            list_client_order_id: ID optionnel pour l'OCO list
        """
        params: dict[str, Any] = {
            "symbol": symbol,
            "side": side.upper(),
            "quantity": quantity,
            "newOrderRespType": "RESULT",
        }

        if list_client_order_id:
            params["listClientOrderId"] = list_client_order_id

        if side.upper() == "SELL":
            # Fermer un BUY → TP au-dessus, SL en-dessous
            # above = LIMIT_MAKER (TP), below = STOP_LOSS_LIMIT (SL)
            params["aboveType"] = "LIMIT_MAKER"
            params["abovePrice"] = tp_price
            params["belowType"] = "STOP_LOSS_LIMIT"
            params["belowPrice"] = sl_limit_price
            params["belowStopPrice"] = sl_stop_price
            params["belowTimeInForce"] = "GTC"
        else:
            # Fermer un SELL → SL au-dessus, TP en-dessous
            # above = STOP_LOSS_LIMIT (SL), below = LIMIT_MAKER (TP)
            params["aboveType"] = "STOP_LOSS_LIMIT"
            params["abovePrice"] = sl_limit_price
            params["aboveStopPrice"] = sl_stop_price
            params["aboveTimeInForce"] = "GTC"
            params["belowType"] = "LIMIT_MAKER"
            params["belowPrice"] = tp_price

        resp = self._request("POST", "/api/v3/orderList/oco", params=params, signed=True)
        order_list_id = resp.get("orderListId", "unknown")

        logger.info(
            "🎯 OCO placé: %s %s %s | TP=%s | SL_stop=%s SL_limit=%s → orderListId=%s",
            side, symbol, quantity, tp_price, sl_stop_price, sl_limit_price,
            order_list_id,
        )
        return resp

    def cancel_order(
        self,
        symbol: str,
        order_id: Optional[int] = None,
        client_order_id: Optional[str] = None,
    ) -> dict:
        """DELETE /api/v3/order → annuler un ordre."""
        params: dict[str, Any] = {"symbol": symbol}
        if order_id:
            params["orderId"] = order_id
        if client_order_id:
            params["origClientOrderId"] = client_order_id

        resp = self._request("DELETE", "/api/v3/order", params=params, signed=True)
        logger.info("Ordre annulé: %s %s", symbol, order_id or client_order_id)
        return resp

    def cancel_order_list(
        self,
        symbol: str,
        order_list_id: Optional[int] = None,
        list_client_order_id: Optional[str] = None,
    ) -> dict:
        """DELETE /api/v3/orderList → annuler un OCO complet."""
        params: dict[str, Any] = {"symbol": symbol}
        if order_list_id:
            params["orderListId"] = order_list_id
        if list_client_order_id:
            params["listClientOrderId"] = list_client_order_id

        resp = self._request("DELETE", "/api/v3/orderList", params=params, signed=True)
        logger.info("OCO annulé: %s (listId=%s)", symbol, order_list_id or list_client_order_id)
        return resp

    def get_order(
        self,
        symbol: str,
        order_id: Optional[int] = None,
        client_order_id: Optional[str] = None,
    ) -> dict:
        """GET /api/v3/order → détails d'un ordre."""
        params: dict[str, Any] = {"symbol": symbol}
        if order_id:
            params["orderId"] = order_id
        if client_order_id:
            params["origClientOrderId"] = client_order_id
        return self._request("GET", "/api/v3/order", params=params, signed=True)

    def get_order_list(self, order_list_id: int) -> dict:
        """GET /api/v3/orderList → détails d'un OCO."""
        return self._request(
            "GET", "/api/v3/orderList",
            params={"orderListId": order_list_id}, signed=True,
        )

    def get_active_orders(
        self, symbol: Optional[str] = None
    ) -> list[dict]:
        """GET /api/v3/openOrders → ordres en cours."""
        params: dict[str, Any] = {}
        if symbol:
            params["symbol"] = symbol
        resp = self._request("GET", "/api/v3/openOrders", params=params, signed=True)
        return resp if isinstance(resp, list) else []

    def get_active_order_lists(self) -> list[dict]:
        """GET /api/v3/openOrderList → OCOs en cours."""
        resp = self._request("GET", "/api/v3/openOrderList", signed=True)
        return resp if isinstance(resp, list) else []

    # ═══════════════════════════════════════════════════════════════════════════
    # HELPERS : formatage des quantités / prix selon les filtres Binance
    # ═══════════════════════════════════════════════════════════════════════════

    def format_quantity(self, symbol: str, quantity: float) -> str:
        """Arrondit la quantité selon le filtre LOT_SIZE du symbole."""
        filters = self.get_symbol_filters(symbol)
        lot_size = filters.get("LOT_SIZE", {})
        step_size = float(lot_size.get("stepSize", "0.00000001"))
        min_qty = float(lot_size.get("minQty", "0.00000001"))

        # Arrondir au step_size inférieur
        precision = self._step_to_precision(step_size)
        qty = max(min_qty, self._truncate(quantity, precision))
        return f"{qty:.{precision}f}"

    def format_price(self, symbol: str, price: float) -> str:
        """Arrondit le prix selon le filtre PRICE_FILTER du symbole."""
        filters = self.get_symbol_filters(symbol)
        price_filter = filters.get("PRICE_FILTER", {})
        tick_size = float(price_filter.get("tickSize", "0.01"))

        precision = self._step_to_precision(tick_size)
        p = self._truncate(price, precision)
        return f"{p:.{precision}f}"

    def check_min_notional(self, symbol: str, quantity: float, price: float) -> bool:
        """Vérifie que le notionnel (qty × price) respecte le MIN_NOTIONAL."""
        filters = self.get_symbol_filters(symbol)
        notional_filter = filters.get("NOTIONAL", filters.get("MIN_NOTIONAL", {}))
        min_notional = float(notional_filter.get("minNotional", "10.0"))
        notional = quantity * price
        if notional < min_notional:
            logger.warning(
                "%s: notionnel %.4f < min %.4f (qty=%.8f × price=%.4f)",
                symbol, notional, min_notional, quantity, price,
            )
            return False
        return True

    # ═══════════════════════════════════════════════════════════════════════════
    # AUTH & HTTP INTERNE
    # ═══════════════════════════════════════════════════════════════════════════

    def _request(
        self,
        method: str,
        path: str,
        params: Optional[dict] = None,
        signed: bool = False,
    ) -> Any:
        """Effectue une requête vers l'API Binance."""
        url = f"{self._base_url}{path}"
        query_params = dict(params or {})

        if signed:
            query_params["timestamp"] = str(int(time.time() * 1000))
            query_params["recvWindow"] = "5000"
            query_string = urlencode(query_params)
            signature = self._sign(query_string)
            query_params["signature"] = signature

        headers = {
            "X-MBX-APIKEY": self._api_key,
            "Content-Type": "application/x-www-form-urlencoded",
        }

        logger.debug("→ %s %s (signed=%s)", method, path, signed)

        if method.upper() in ("POST", "PUT", "DELETE"):
            response = self._client.request(
                method=method,
                url=url,
                headers=headers,
                data=query_params,
            )
        else:
            response = self._client.request(
                method=method,
                url=url,
                headers=headers,
                params=query_params,
            )

        if response.status_code >= 400:
            resp_text = response.text
            logger.error(
                "API error %d: %s %s → %s",
                response.status_code, method, path, resp_text,
            )
            raise RuntimeError(
                f"Binance API {response.status_code} {method} {path} | {resp_text[:500]}"
            )

        return response.json()

    def _sign(self, query_string: str) -> str:
        """Signe un query string avec HMAC-SHA256."""
        return hmac.new(
            self._secret_key.encode("utf-8"),
            query_string.encode("utf-8"),
            hashlib.sha256,
        ).hexdigest()

    @staticmethod
    def _step_to_precision(step: float) -> int:
        """Convertit un stepSize en nombre de décimales."""
        if step >= 1:
            return 0
        s = f"{step:.10f}".rstrip("0")
        if "." in s:
            return len(s.split(".")[1])
        return 0

    @staticmethod
    def _truncate(value: float, precision: int) -> float:
        """Tronque une valeur flottante à la précision donnée (sans arrondi)."""
        factor = 10 ** precision
        return int(value * factor) / factor

    def close(self) -> None:
        """Ferme le client HTTP."""
        self._client.close()
