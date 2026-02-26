"""Script de diagnostic pour tester l'authentification Revolut X."""

import base64
import time

import httpx
from cryptography.hazmat.primitives.serialization import load_pem_private_key

API_KEY = "4fiN7hPyMdirdTo1U4Xo1idypVDkNkxdnGbQHsvS7uKGmrdUZiuGuDtQi7MSakDV"
PRIVATE_KEY_PATH = "private.pem"
BASE_URL = "https://revx.revolut.com/api/1.0"


def main():
    # 1. Vérifier l'IP publique
    ip_resp = httpx.get("https://api.ipify.org?format=json")
    my_ip = ip_resp.json()["ip"]
    print(f"🌐 IP publique: {my_ip}")
    print(f"📝 IP whitelistée: 90.83.11.90")
    print(f"{'✅' if my_ip == '90.83.11.90' else '❌'} Match: {my_ip == '90.83.11.90'}")
    print()

    # 2. Charger la clé privée
    with open(PRIVATE_KEY_PATH, "rb") as f:
        private_key = load_pem_private_key(f.read(), password=None)
    print("🔑 Clé privée chargée OK")

    # 3. Test sans auth (pour voir si l'API répond)
    print("\n── Test 1: GET /tickers SANS auth ──")
    r = httpx.get(f"{BASE_URL}/tickers")
    print(f"Status: {r.status_code}")
    print(f"Response: {r.text[:300]}")

    # 4. Test avec auth
    print("\n── Test 2: GET /tickers AVEC auth ──")
    timestamp = str(int(time.time() * 1000))
    method = "GET"
    path = "/api/1.0/tickers"
    message = f"{timestamp}{method}{path}"

    signature = base64.b64encode(private_key.sign(message.encode())).decode()

    print(f"Timestamp: {timestamp}")
    print(f"Message: {message}")
    print(f"Signature: {signature[:50]}...")

    r = httpx.get(
        f"{BASE_URL}/tickers",
        headers={
            "X-Revx-API-Key": API_KEY,
            "X-Revx-Timestamp": timestamp,
            "X-Revx-Signature": signature,
            "Content-Type": "application/json",
        },
    )
    print(f"Status: {r.status_code}")
    print(f"Response: {r.text[:500]}")

    # 5. Test avec auth sur /balances (endpoint authentifié)
    print("\n── Test 3: GET /balances AVEC auth ──")
    timestamp = str(int(time.time() * 1000))
    path = "/api/1.0/balances"
    message = f"{timestamp}GET{path}"
    signature = base64.b64encode(private_key.sign(message.encode())).decode()

    r = httpx.get(
        f"{BASE_URL}/balances",
        headers={
            "X-Revx-API-Key": API_KEY,
            "X-Revx-Timestamp": timestamp,
            "X-Revx-Signature": signature,
            "Content-Type": "application/json",
        },
    )
    print(f"Status: {r.status_code}")
    print(f"Response: {r.text[:500]}")


if __name__ == "__main__":
    main()
