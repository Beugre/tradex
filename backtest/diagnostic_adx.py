#!/usr/bin/env python3
"""
Test PROGRESSIF des 5 recommandations — Diagnostic

Version 1 : Tester chaque changement individuellement pour voir lequel tue les trades
"""

from __future__ import annotations

import sys
sys.path.insert(0, '.')

from datetime import datetime, timedelta, timezone
from backtest.data_loader import download_candles
from backtest.run_backtest_adaptive import _run_adaptive_pair, _ema, _rsi, _atr, _sma, _adx, _bollinger_width, Regime, _Position, AdaptiveTrade, _detect_regime
from src.core.models import Candle

def _run_adaptive_pair_v1_baseline(c15, c1h, bal_init):
    """Baseline original."""
    return _run_adaptive_pair(c15, c1h, bal_init)

def _run_adaptive_pair_v2_sl_2pct(c15, c1h, bal_init):
    """Amélioration #1 : SL -2.0% au lieu de -1.5%"""
    # Même logique que baseline mais SL = -2%
    bull_sl_pct = 0.020
    # ... (implémentation dupliquée avec changement)
    # Pour l'instant, utiliser le même pour éviter bugs
    return _run_adaptive_pair(c15, c1h, bal_init)

def diagnostic_adx():
    """Vérifier pourquoi ADX tue les trades."""
    end = datetime(2026, 5, 4, tzinfo=timezone.utc)
    start = end - timedelta(days=180)
    
    print("\n🔍 DIAGNOSTIC ADX")
    print("="*80)
    
    pair = "ETH-USD"
    c1h = download_candles(pair, start, end, interval="1h")
    
    if not c1h:
        print(f"❌ Pas de données")
        return
    
    closes = [c.close for c in c1h]
    adx_vals = _adx(c1h, 14)
    
    print(f"\n📊 {pair} - ADX(14) analysis")
    print(f"  Total candles 1H: {len(c1h)}")
    print(f"  ADX values calculés: {len([v for v in adx_vals if v > 0])}")
    
    # Stats ADX
    adx_nonzero = [v for v in adx_vals if v > 0]
    if adx_nonzero:
        print(f"  ADX min: {min(adx_nonzero):.1f}")
        print(f"  ADX max: {max(adx_nonzero):.1f}")
        print(f"  ADX mean: {sum(adx_nonzero)/len(adx_nonzero):.1f}")
        print(f"  ADX > 20: {len([v for v in adx_nonzero if v > 20])}/{len(adx_nonzero)} ({len([v for v in adx_nonzero if v > 20])/len(adx_nonzero)*100:.1f}%)")
        print(f"  ADX > 25: {len([v for v in adx_nonzero if v > 25])}/{len(adx_nonzero)} ({len([v for v in adx_nonzero if v > 25])/len(adx_nonzero)*100:.1f}%)")
        print(f"  ADX > 30: {len([v for v in adx_nonzero if v > 30])}/{len(adx_nonzero)} ({len([v for v in adx_nonzero if v > 30])/len(adx_nonzero)*100:.1f}%)")
    
    # Dernières valeurs
    print(f"\n  Dernières 10 ADX (1H):")
    for i in range(max(0, len(adx_vals)-10), len(adx_vals)):
        print(f"    [{i}] ADX={adx_vals[i]:.1f} close={closes[i]:.2f}")

if __name__ == "__main__":
    diagnostic_adx()
    
    print("\n\n📋 RÉSUMÉ FINDINGS")
    print("="*80)
    print("""
Si ADX > 25 = 0% des candles 1H :
  → Le filtre est TOO STRICT
  → Solution : Baisser seuil à ADX > 15 ou le rendre conditionnel
  
Si ADX > 25 = ~10-30% des candles 1H :
  → Acceptable mais va réduire trades
  → Solution : Combiner avec autre filtre ou réduire seuil progressivement
    
Recommandation : Commencer avec ADX > 15 et valider walk-forward
""")
