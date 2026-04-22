"""
Client HTTP pour l'API Revolut X Crypto Exchange.

Gère l'authentification Ed25519 (signature par requête) et expose
des méthodes typées pour chaque endpoint utilisé par le bot.

Maker-First Execution :
    Revolut X : maker = 0% / taker = 0.09%
    → place_maker_first_order() place un limit passif (maker),
      attend MAKER_WAIT_SECONDS, puis bascule en taker si non rempli.

Doc API : https://developer.revolut.com/docs/x-api/revolut-x-crypto-exchange-rest-api
"""

from __future__ import annotations

import base64
import json
import logging
import time
from pathlib import Path
from typing import Any, Optional

import httpx
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from cryptography.hazmat.primitives.serialization import (
    Encoding,
    NoEncryption,
    PrivateFormat,
    load_pem_private_key,
)

from src.core.models import Balance, Candle, OrderRequest, OrderSide, TickerData

logger = logging.getLogger(__name__)


# ── Exceptions typées ─────────────────────────────────────────────────────────

class RevolutAPIError(RuntimeError):
    """Erreur API Revolut X — porte le status_code pour faciliter le filtrage."""
    def __init__(self, status_code: int, method: str, path: str, body: str) -> None:
        super().__init__(f"API {status_code} {method} {path} | {body[:500]}")
        self.status_code = status_code


class RevolutAuthError(RevolutAPIError):
    """401 — clé invalide ou signature incorrecte. Le bot doit s'arrêter."""


class RevolutRateLimitError(RevolutAPIError):
    """429 — rate limit atteint, backoff requis."""


class RevolutServerError(RevolutAPIError):
    """5xx — erreur serveur temporaire, retry possible."""


class RevolutXClient:
    """Client pour l'API REST Revolut X avec authentification Ed25519."""

    def __init__(
        self,
        api_key: str,
        private_key_path: Path,
        base_url: str = "https://revx.revolut.com/api/1.0",
    ) -> None:
        self._api_key = api_key
        self._base_url = base_url.rstrip("/")
        self._private_key = self._load_private_key(private_key_path)
        self._client = httpx.Client(timeout=30.0)

    # ── Endpoints publics ──────────────────────────────────────────────────────

    def get_balances(self) -> list[Balance]:
        """GET /balances → liste des soldes par devise."""
        data = self._request("GET", "/balances")
        return [
            Balance(
                currency=b["currency"],
                available=float(b["available"]),
                reserved=float(b["reserved"]),
                total=float(b["total"]),
            )
            for b in data
        ]

    def get_tickers(
        self, symbols: Optional[list[str]] = None
    ) -> list[TickerData]:
        """GET /tickers → prix temps réel pour toutes les paires (ou filtrées)."""
        params = {}
        if symbols:
            params["symbols"] = ",".join(symbols)
        resp = self._request("GET", "/tickers", params=params)
        tickers_data = resp.get("data", []) if isinstance(resp, dict) else resp
        return [
            TickerData(
                symbol=t["symbol"],
                bid=float(t["bid"]),
                ask=float(t["ask"]),
                mid=float(t["mid"]),
                last_price=float(t["last_price"]),
            )
            for t in tickers_data
        ]

    def get_candles(
        self,
        symbol: str,
        interval: Optional[int] = None,
        since: Optional[int] = None,
    ) -> list[Candle]:
        """
        GET /candles/{symbol} → bougies OHLCV.

        Args:
            symbol: Paire (ex: "BTC-USD").
            interval: Durée de chaque bougie en minutes (ex: 240 pour H4).
            since: Timestamp Unix ms à partir duquel récupérer les données.
        """
        params: dict[str, Any] = {}
        if interval is not None:
            params["interval"] = interval
        if since is not None:
            params["since"] = since

        resp = self._request("GET", f"/candles/{symbol}", params=params)
        candles_data = resp.get("data", []) if isinstance(resp, dict) else resp
        return [
            Candle(
                timestamp=c["start"],
                open=float(c["open"]),
                high=float(c["high"]),
                low=float(c["low"]),
                close=float(c["close"]),
                volume=float(c["volume"]),
            )
            for c in candles_data
        ]

    def place_order(self, order: OrderRequest) -> dict:
        """POST /orders → placer un ordre limit."""
        payload = order.to_api_payload()
        resp = self._request("POST", "/orders", json_body=payload)
        logger.info(
            "Ordre placé: %s %s %s @ %s → %s",
            order.side.value,
            order.symbol,
            order.base_size,
            order.price,
            resp,
        )
        return resp

    def get_active_orders(
        self, symbols: Optional[list[str]] = None
    ) -> list[dict]:
        """GET /orders/active → ordres en cours."""
        params = {}
        if symbols:
            params["symbols"] = ",".join(symbols)
        resp = self._request("GET", "/orders/active", params=params)
        return resp.get("data", []) if isinstance(resp, dict) else resp

    def cancel_order(self, venue_order_id: str) -> None:
        """DELETE /orders/{venue_order_id} → annuler un ordre."""
        self._request("DELETE", f"/orders/{venue_order_id}")
        logger.info("Ordre annulé: %s", venue_order_id)

    def get_order(self, venue_order_id: str) -> dict:
        """GET /orders/{venue_order_id} → détails d'un ordre."""
        return self._request("GET", f"/orders/{venue_order_id}")

    def get_order_fills(self, venue_order_id: str) -> list[dict]:
        """GET /orders/{venue_order_id}/fills → exécutions d'un ordre."""
        resp = self._request("GET", f"/orders/{venue_order_id}/fills")
        return resp.get("data", []) if isinstance(resp, dict) else resp

    # ── Maker-First Execution ──────────────────────────────────────────────────

    def place_maker_first_order(
        self,
        order: OrderRequest,
        wait_seconds: int = 30,
    ) -> dict:
        """Place un ordre limit passif (maker 0%), attend, puis bascule en taker si non rempli.

        Flow :
            1. Place un limit au prix exact de la borne (passif → maker fee = 0%)
            2. Attend `wait_seconds` secondes
            3. Vérifie le statut :
               - FILLED → retourne (0% fee payé ✅)
               - PARTIALLY_FILLED → annule le reste, retourne le fill partiel
               - Pas rempli → annule, place un limit agressif (cross le spread → taker 0.09%)

        Args:
            order: L'ordre limit à placer (prix = borne du range).
            wait_seconds: Temps d'attente pour le fill maker (défaut 30s).

        Returns:
            dict avec les clés: venue_order_id, fill_type ('maker', 'taker', 'partial_maker')
        """
        # ── 1. Placer l'ordre passif (maker) ──
        logger.info(
            "💰 MAKER-FIRST | %s %s | limit passif @ %s (attente %ds)",
            order.side.value.upper(), order.symbol, order.price, wait_seconds,
        )
        try:
            resp = self.place_order(order)
        except Exception as e:
            logger.error("💰 MAKER-FIRST | Échec du placement maker: %s", e)
            raise

        data = resp.get("data", {})
        if isinstance(data, dict):
            venue_order_id = data.get("venue_order_id", "unknown")
        elif isinstance(data, list) and data:
            venue_order_id = data[0].get("venue_order_id", "unknown")
        else:
            venue_order_id = "unknown"

        if venue_order_id == "unknown":
            logger.warning("💰 MAKER-FIRST | Impossible de récupérer venue_order_id → retour direct")
            return {"venue_order_id": venue_order_id, "fill_type": "maker", "actual_price": float(order.price), "response": resp}

        # ── 1b. Vérifier si l'ordre a été rempli instantanément ──
        initial_state = ""
        if isinstance(data, dict):
            initial_state = (data.get("state") or data.get("status") or "").upper()
        if initial_state == "FILLED":
            logger.info("💰 MAKER-FIRST | ✅ FILL INSTANTANÉ — order already filled at placement")
            return {"venue_order_id": venue_order_id, "fill_type": "maker", "actual_price": float(order.price), "response": resp}

        # ── 2. Attendre ──
        logger.info("💰 MAKER-FIRST | ⏳ Attente %ds pour fill maker…", wait_seconds)
        time.sleep(wait_seconds)

        # ── 3. Vérifier le statut ──
        try:
            order_status = self.get_order(venue_order_id)
        except Exception as e:
            logger.warning("💰 MAKER-FIRST | Impossible de vérifier le statut: %s → assume filled", e)
            return {"venue_order_id": venue_order_id, "fill_type": "maker", "actual_price": float(order.price), "response": resp}

        # Handle both 'status' and 'state' field names, and nested 'data' wrapper
        order_data = order_status.get("data", order_status) if isinstance(order_status, dict) else order_status
        if isinstance(order_data, list) and order_data:
            order_data = order_data[0]
        status = (order_data.get("status") or order_data.get("state") or "").upper()
        filled_size = float(order_data.get("filled_size", "0") or "0")
        total_size = float(order.base_size)

        logger.info(
            "💰 MAKER-FIRST | Statut: %s | Rempli: %.8f / %.8f",
            status, filled_size, total_size,
        )

        # ── FILLED → parfait, 0% fee ──
        if status == "FILLED" or (filled_size > 0 and filled_size >= total_size * 0.99):
            logger.info("💰 MAKER-FIRST | ✅ FILL MAKER complet — fee 0%%")
            return {"venue_order_id": venue_order_id, "fill_type": "maker", "actual_price": float(order.price), "response": order_status}

        # ── PARTIALLY FILLED → garder le fill, annuler le reste ──
        if filled_size > 0:
            logger.info(
                "💰 MAKER-FIRST | ⚡ Fill partiel maker (%.8f/%.8f) — annulation du reste",
                filled_size, total_size,
            )
            try:
                self.cancel_order(venue_order_id)
            except Exception as e:
                logger.warning("💰 MAKER-FIRST | Cancel partiel échoué: %s", e)
            return {
                "venue_order_id": venue_order_id,
                "fill_type": "partial_maker",
                "actual_price": float(order.price),
                "filled_size": filled_size,
                "response": order_status,
            }

        # ── PAS REMPLI → annuler et passer en taker ──
        logger.info("💰 MAKER-FIRST | ❌ Pas de fill maker — bascule en TAKER")
        try:
            self.cancel_order(venue_order_id)
        except Exception as e:
            # 409 Conflict = l'ordre est déjà rempli ou annulé → vérifier
            is_conflict = "409" in str(e) or "conflict" in str(e).lower()
            logger.warning(
                "💰 MAKER-FIRST | Cancel échoué%s: %s — re-vérification",
                " (409 Conflict)" if is_conflict else "", e,
            )
            try:
                recheck = self.get_order(venue_order_id)
                recheck_data = recheck.get("data", recheck) if isinstance(recheck, dict) else recheck
                if isinstance(recheck_data, list) and recheck_data:
                    recheck_data = recheck_data[0]
                recheck_status = (recheck_data.get("status") or recheck_data.get("state") or "").upper()
                if recheck_status == "FILLED":
                    logger.info("💰 MAKER-FIRST | ✅ Rempli entre-temps — fee 0%%")
                    return {"venue_order_id": venue_order_id, "fill_type": "maker", "actual_price": float(order.price), "response": recheck}
            except Exception as e_recheck:
                logger.debug("💰 MAKER-FIRST | Re-check statut échoué: %s", e_recheck)
                pass
            if is_conflict:
                logger.warning("💰 MAKER-FIRST | ⚠️ 409 Conflict + re-check échoué → assume filled")
                return {"venue_order_id": venue_order_id, "fill_type": "maker", "actual_price": float(order.price), "response": resp}
            raise

        # ── 4. Placer le taker (limit agressif qui cross le spread) ──
        try:
            tickers = self.get_tickers(symbols=[order.symbol])
            if tickers:
                t = tickers[0]
                if order.side == OrderSide.BUY:
                    # Acheter → placer au ask (garantit le fill immédiat)
                    taker_price = t.ask
                else:
                    # Vendre → placer au bid
                    taker_price = t.bid
            else:
                # Fallback : utiliser le prix original
                taker_price = float(order.price)
        except Exception as e:
            logger.warning("💰 MAKER-FIRST | Ticker indisponible: %s — utilisation du prix original", e)
            taker_price = float(order.price)

        taker_order = OrderRequest(
            symbol=order.symbol,
            side=order.side,
            base_size=order.base_size,
            price=f"{taker_price:.8f}".rstrip("0").rstrip("."),
        )

        logger.info(
            "💰 MAKER-FIRST | 🔄 TAKER | %s %s @ %s (fee 0.09%%)",
            order.side.value.upper(), order.symbol, taker_order.price,
        )
        try:
            taker_resp = self.place_order(taker_order)
        except Exception as e:
            logger.error("💰 MAKER-FIRST | Échec du placement taker: %s", e)
            raise

        taker_data = taker_resp.get("data", {})
        if isinstance(taker_data, dict):
            taker_id = taker_data.get("venue_order_id", venue_order_id)
        elif isinstance(taker_data, list) and taker_data:
            taker_id = taker_data[0].get("venue_order_id", venue_order_id)
        else:
            taker_id = venue_order_id

        return {"venue_order_id": taker_id, "fill_type": "taker", "actual_price": taker_price, "response": taker_resp}

    # ── Auth & HTTP interne ────────────────────────────────────────────────────

    def _request(
        self,
        method: str,
        path: str,
        params: Optional[dict] = None,
        json_body: Optional[dict] = None,
    ) -> Any:
        """Effectue une requête signée vers l'API Revolut X."""
        url = f"{self._base_url}{path}"
        timestamp = str(int(time.time() * 1000))

        # Construire la query string
        query_string = ""
        if params:
            query_string = "&".join(f"{k}={v}" for k, v in sorted(params.items()))

        # Construire le body
        body_str = ""
        if json_body is not None:
            body_str = json.dumps(json_body, separators=(",", ":"))

        # Construire le message à signer : {timestamp}{METHOD}{/api/1.0/path}{query}{body}
        api_path = f"/api/1.0{path}"
        message = f"{timestamp}{method.upper()}{api_path}{query_string}{body_str}"

        # Signer avec Ed25519
        signature = self._sign(message)

        headers = {
            "X-Revx-API-Key": self._api_key,
            "X-Revx-Timestamp": timestamp,
            "X-Revx-Signature": signature,
            "Content-Type": "application/json",
            "Accept": "application/json",
        }

        logger.debug("→ %s %s (params=%s)", method, url, params)

        response = self._client.request(
            method=method,
            url=url,
            headers=headers,
            params=params,
            content=body_str.encode() if body_str else None,
        )

        if response.status_code == 204:
            return {}

        if response.status_code >= 400:
            resp_text = response.text
            logger.error(
                "API error %d: %s %s → %s",
                response.status_code,
                method,
                path,
                resp_text,
            )
            sc = response.status_code
            if sc == 401:
                raise RevolutAuthError(sc, method, path, resp_text)
            if sc == 429:
                raise RevolutRateLimitError(sc, method, path, resp_text)
            if sc >= 500:
                raise RevolutServerError(sc, method, path, resp_text)
            raise RevolutAPIError(sc, method, path, resp_text)

        return response.json()

    def _sign(self, message: str) -> str:
        """Signe un message avec la clé privée Ed25519 et retourne le base64."""
        signature_bytes = self._private_key.sign(message.encode("utf-8"))
        return base64.b64encode(signature_bytes).decode("utf-8")

    @staticmethod
    def _load_private_key(path: Path) -> Ed25519PrivateKey:
        """Charge la clé privée Ed25519 depuis un fichier PEM."""
        pem_data = path.read_bytes()
        key = load_pem_private_key(pem_data, password=None)
        if not isinstance(key, Ed25519PrivateKey):
            raise TypeError(
                f"La clé dans {path} n'est pas une clé Ed25519. "
                f"Générez-la avec: openssl genpkey -algorithm ed25519 -out private.pem"
            )
        return key

    def close(self) -> None:
        """Ferme le client HTTP."""
        self._client.close()
