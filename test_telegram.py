"""Test rapide du token Telegram."""
import httpx

TOKEN = "8117770827:AAHitB3fjxC0WIK4_BonLmXzmPa1vHCDZh4"
CHAT_ID = "1181024836"

# Test 1: getMe — vérifie si le token est valide
r = httpx.get(f"https://api.telegram.org/bot{TOKEN}/getMe")
print(f"getMe → Status: {r.status_code}")
print(f"Response: {r.text}")

if r.status_code == 200:
    # Test 2: envoyer un message
    r2 = httpx.post(
        f"https://api.telegram.org/bot{TOKEN}/sendMessage",
        json={"chat_id": CHAT_ID, "text": "🤖 TradeX test — connexion OK !"},
    )
    print(f"\nsendMessage → Status: {r2.status_code}")
    print(f"Response: {r2.text}")
