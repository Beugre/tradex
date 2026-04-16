"""
Script one-shot : re-pose les OCO manquants pour les positions Adaptive Bull
dont oco_order_list_id est null dans state_adaptive.json.

Utilisation : python scripts/repair_oco_adaptive.py
"""
import json
import os
import sys
import uuid
from pathlib import Path

# Ajouter le répertoire racine au path
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv
load_dotenv(ROOT / ".env")

from src.exchange.binance_client import BinanceClient

STATE_FILE = ROOT / "data" / "state_adaptive.json"


def _format_qty(client: BinanceClient, symbol: str, qty: float) -> str:
    return client.format_quantity(symbol, qty)


def _format_price(client: BinanceClient, symbol: str, price: float) -> str:
    return client.format_price(symbol, price)


def main():
    state = json.loads(STATE_FILE.read_text())
    positions = state.get("positions", {})

    client = BinanceClient(
        api_key=os.environ["BINANCE_API_KEY"],
        api_secret=os.environ["BINANCE_SECRET_KEY"],
    )

    # Récupérer les balances réelles
    balances = {b.currency: b.available for b in client.get_balances()}
    print(f"Balances disponibles : { {k: v for k, v in balances.items() if v > 0} }")

    modified = False

    for symbol, pos in positions.items():
        if pos.get("oco_order_list_id") is not None:
            print(f"[{symbol}] OCO déjà présent (listId={pos['oco_order_list_id']}) — skip")
            continue

        base = symbol.replace("USDC", "").replace("USDT", "")
        available = balances.get(base, 0.0)
        stored_size = float(pos["size"])

        if available <= 0:
            print(f"[{symbol}] ⚠️  Balance {base} = 0 — position peut-être déjà clôturée ? skip")
            continue

        # Utiliser min(stocké, disponible) arrondi au step_size
        raw_qty = min(stored_size, available)
        qty_str = _format_qty(client, symbol, raw_qty)

        sl_stop   = _format_price(client, symbol, float(pos["sl_price"]))
        sl_limit  = _format_price(client, symbol, float(pos["sl_price"]) * 0.995)
        tp_str    = _format_price(client, symbol, float(pos["tp_price"]))
        list_cid  = f"adt-{base.lower()}-{str(uuid.uuid4())[:6]}"

        print(f"[{symbol}] Pose OCO | qty={qty_str} (dispo={available:.6f}) | SL={sl_stop} | TP={tp_str}")

        try:
            resp = client.place_oco_order(
                symbol=symbol,
                side="SELL",
                quantity=qty_str,
                tp_price=tp_str,
                sl_stop_price=sl_stop,
                sl_limit_price=sl_limit,
                list_client_order_id=list_cid,
            )
            order_list_id = resp.get("orderListId")
            pos["oco_order_list_id"] = order_list_id
            pos["oco_list_client_id"] = list_cid
            # Mettre à jour la taille stockée avec la quantité nette réelle
            pos["size"] = float(qty_str)
            print(f"[{symbol}] ✅ OCO posé | listId={order_list_id}")
            modified = True
        except Exception as e:
            print(f"[{symbol}] ❌ Échec OCO : {e}")

    if modified:
        STATE_FILE.write_text(json.dumps(state, indent=2))
        print(f"\n✅ State mis à jour : {STATE_FILE}")
    else:
        print("\nAucune modification apportée.")


if __name__ == "__main__":
    main()
