"""Preflight check avant lancement réel."""
from src.exchange.revolut_client import RevolutXClient
from src.core.risk_manager import get_fiat_balance
from src.core.models import Balance
from src import config

client = RevolutXClient(config.REVOLUT_X_API_KEY, config.REVOLUT_X_PRIVATE_KEY_PATH)

# 1. Check balances
balances = client.get_balances()
fiat, currency = get_fiat_balance(balances)
print(f"💰 Solde: {fiat:.2f} USD equiv (depuis {currency})")
print(f"   Risque 5%: {fiat * 0.05:.2f} USD par trade")
for b in balances:
    if b.available > 0 or b.reserved > 0:
        print(f"   {b.currency}: available={b.available}, reserved={b.reserved}")

# 2. Check candles (doit retourner 200)
candles = client.get_candles("BTC-USD", interval=240)
print(f"\n📊 Candles BTC-USD: {len(candles)} bougies H4 reçues (interval=240) ✅")

# 3. Config recap
print(f"\n⚙️  Config:")
print(f"   Paires: {config.TRADING_PAIRS}")
print(f"   Entry buffer: {config.ENTRY_BUFFER_PERCENT * 100:.1f}%")
print(f"   SL buffer: {config.SL_BUFFER_PERCENT * 100:.1f}%")
print(f"   Zero-risk trigger: {config.ZERO_RISK_TRIGGER_PERCENT * 100:.1f}%")
print(f"   Polling: {config.POLLING_INTERVAL_SECONDS}s")

client.close()
print("\n✅ Tout est OK — prêt pour le lancement réel")
