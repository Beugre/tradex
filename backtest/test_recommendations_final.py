#!/usr/bin/env python3
"""
Test des 5 recommandations — Version simple & directe

Utilise les paramètres existants de _run_adaptive_pair pour tester les variations.
"""

from __future__ import annotations

import sys
sys.path.insert(0, '.')

from datetime import datetime, timedelta, timezone
from backtest.data_loader import download_candles
from backtest.run_backtest_adaptive import _run_adaptive_pair

end = datetime(2026, 5, 4, tzinfo=timezone.utc)
start = end - timedelta(days=180)

pairs = ["ETH-USD", "BTC-USD", "SOL-USD"]

scenarios = {
    "01_BASELINE": {
        "desc": "Original (SL -1.5%)",
        "params": {}
    },
    "02_SL_2PCT": {
        "desc": "SL -2.0% (Recommandation #1)",
        "params": {'bull_sl_pct': 0.020}
    },
    "03_ADX_15": {
        "desc": "Filtre ADX trend confirmation (v2)",
        "params": {'use_progressive_cooldown': True}
    },
    "04_COOLDOWN_PROG": {
        "desc": "Cooldown progressif après losses",
        "params": {'use_progressive_cooldown': True}
    },
    "05_VOL_FILTER": {
        "desc": "Volume filter pour entrée BULL",
        "params": {'use_volume_filter': True, 'vol_spike_mult': 1.2}
    },
    "06_COND_PYRAMID": {
        "desc": "Pyramiding conditionnel (>50% to TP)",
        "params": {'use_conditional_pyramid': True}
    },
    "07_ALL_V2": {
        "desc": "🎯 Combinaison optimale : SL +Cooldown +Vol +PyramidCond",
        "params": {
            'bull_sl_pct': 0.020,
            'use_progressive_cooldown': True,
            'use_volume_filter': True,
            'vol_spike_mult': 1.2,
            'use_conditional_pyramid': True,
        }
    },
}

print("\n" + "="*120)
print("  🎯 TEST RECOMMANDATIONS — ADAPTIVE BULL")
print("="*120)

results = {s: [] for s in scenarios}

for pair in pairs:
    print(f"\n📊 {pair} ({start.date()} → {end.date()})")
    print("-"*120)
    
    c15 = download_candles(pair, start, end, interval="15m")
    c1h = download_candles(pair, start, end, interval="1h")
    
    if not c15 or not c1h:
        print(f"  ❌ Données manquantes")
        continue
    
    for scenario_name, scenario in scenarios.items():
        desc = scenario['desc']
        params = scenario['params']
        
        try:
            bal, trades, eq = _run_adaptive_pair(c15, c1h, 1000.0, **params)
            
            wins = sum(1 for t in trades if t.is_win)
            wr = wins / len(trades) if trades else 0
            pnl = bal - 1000.0
            
            results[scenario_name].append({'pnl': pnl, 'trades': len(trades), 'wr': wr})
            
            mark = "📈" if pnl > 0 else "📉" if pnl < 0 else "⏸️ "
            print(f"  {mark} {scenario_name:<12s} | {len(trades):>3d} trades | WR {wr:>5.1%} | PnL ${pnl:>+8.2f}")
        except Exception as e:
            print(f"  ❌ {scenario_name:<12s} | ERROR: {e}")

# Summary
print("\n" + "="*120)
print("  📊 RÉSUMÉ (moyenne sur 3 paires)")
print("="*120)

for scenario_name in scenarios:
    if not results[scenario_name]:
        continue
    
    r = results[scenario_name]
    avg_pnl = sum(x['pnl'] for x in r) / len(r)
    avg_trades = sum(x['trades'] for x in r) / len(r)
    avg_wr = sum(x['wr'] for x in r) / len(r)
    
    vs_baseline = avg_pnl - results["01_BASELINE"][0]['pnl'] if results["01_BASELINE"] else 0
    delta_pct = (vs_baseline / abs(results["01_BASELINE"][0]['pnl'])) * 100 if results["01_BASELINE"] and results["01_BASELINE"][0]['pnl'] != 0 else 0
    
    mark = "🟢" if avg_pnl > 0 else "🔴"
    delta_str = f"{vs_baseline:+.2f}" if scenario_name != "01_BASELINE" else "baseline"
    print(f"  {mark} {scenario_name:<12s} | Trades {avg_trades:>5.0f} | WR {avg_wr:>5.1%} | PnL ${avg_pnl:>+7.2f} | vs baseline {delta_str}")

print("\n" + "="*120)
print("  RECOMMANDATIONS POUR PROD")
print("="*120)
print("""
💡 Analyse :
  - Comparer 01_BASELINE vs 07_ALL_V2 pour la meilleure amélioration
  - Si 02_SL_2PCT seul aide → augmenter SL dans bot_adaptive.py
  - Si 04_COOLDOWN_PROG aide → activer use_progressive_cooldown
  - Si 05_VOL_FILTER aide → activer use_volume_filter
  - Si 06_COND_PYRAMID aide → activer use_conditional_pyramid

🚀 Déploiement :
  1. Tester la meilleure combinaison en walk-forward
  2. Si OOS PF >= 1.15 → déployer en production
  3. Monitorer heartbeat & trades live pendant 1 semaine
  4. Ajuster si needed
""")
