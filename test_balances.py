"""Affiche TOUS les soldes du compte Revolut X."""
import httpx, base64, time
from cryptography.hazmat.primitives.serialization import load_pem_private_key

API_KEY = "4fiN7hPyMdirdTo1U4Xo1idypVDkNkxdnGbQHsvS7uKGmrdUZiuGuDtQi7MSakDV"
BASE_URL = "https://revx.revolut.com/api/1.0"

with open("private.pem", "rb") as f:
    private_key = load_pem_private_key(f.read(), password=None)

ts = str(int(time.time() * 1000))
msg = f"{ts}GET/api/1.0/balances"
sig = base64.b64encode(private_key.sign(msg.encode())).decode()

r = httpx.get(
    f"{BASE_URL}/balances",
    headers={
        "X-Revx-API-Key": API_KEY,
        "X-Revx-Timestamp": ts,
        "X-Revx-Signature": sig,
    },
)

print("Tous les soldes Revolut X :")
print("-" * 50)
for b in r.json():
    print(f"  {b['currency']:>6s}  available={b['available']:>15s}  reserved={b['reserved']:>15s}  total={b['total']:>15s}")
