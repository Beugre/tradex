#!/usr/bin/env python3
"""
Test ISOLÉ des 5 recommandations — Meilleure approche

Tester chaque recommandation individuellement sur 6 mois ETH/BTC/SOL
"""

from __future__ import annotations

import sys
sys.path.insert(0, '.')

from datetime import datetime, timedelta, timezone
from collections import defaultdict
from backtest.data_loader import download_candles
from backtest.run_backtest_adaptive import (
    _run_adaptive_pair, _ema, _rsi, _atr, _sma, _adx, _bollinger_width,
    Regime, _Position, AdaptiveTrade, _detect_regime
)
from src.core.models import Candle

# ═══════════════════════════════════════════════════════════════════════════
# Version modifiée : changements ciblés
# ═══════════════════════════════════════════════════════════════════════════

def _run_adaptive_with_config(
    candles_15m: list[Candle],
    candles_1h: list[Candle],
    initial_balance: float,
    config: dict,  # {'sl_pct': 0.02, 'use_adx': True, 'adx_threshold': 25, ...}
) -> tuple[float, list[AdaptiveTrade], list[float]]:
    """Version générique avec paramètres configurables."""
    
    # Config default
    bull_sl_pct = config.get('sl_pct', 0.015)
    bull_tp_pct = config.get('tp_pct', 0.080)
    use_adx_filter = config.get('use_adx', False)
    adx_threshold = config.get('adx_threshold', 25)
    use_dynamic_tp = config.get('use_dynamic_tp', False)
    use_progressive_cooldown = config.get('use_progressive_cooldown', False)
    
    bull_rsi_min, bull_rsi_max = 50, 65
    bull_slope_bars = 3
    bull_slope_min_pct = 0.005
    bull_pullback_bars = 60
    alloc_pct = 0.33
    bull_alloc_pct = 0.33
    bull_pyramid_alloc = 0.15
    max_positions = 3
    cooldown_bars = 2
    slippage_pct, entry_fee, exit_fee = 0.0005, 0.001, 0.001
    
    n = len(candles_15m)
    balance = initial_balance
    positions: list[_Position] = []
    trades: list[AdaptiveTrade] = []
    equity: list[float] = [balance] * n
    cooldown = 0
    consecutive_losses = 0
    
    # Pré-calcul 1H
    closes_1h = [c.close for c in candles_1h]
    ema50_h = _ema(closes_1h, 50)
    ema200_h = _ema(closes_1h, 200)
    ema20_h = _ema(closes_1h, 20)
    adx_h = _adx(candles_1h, 14) if use_adx_filter else [0]*len(candles_1h)
    rsi_h = _rsi(closes_1h, 14)
    atr_h = _atr(candles_1h, 14)
    atr_ma50_h = _sma(atr_h, 50)
    bb_width_h = _bollinger_width(closes_1h, 20, 2.0)
    
    # Pré-calcul 15m
    closes_15m = [c.close for c in candles_15m]
    lows_15m = [c.low for c in candles_15m]
    vol_15m = [c.volume for c in candles_15m]
    ema50_15 = _ema(closes_15m, 50)
    ema200_15 = _ema(closes_15m, 200)
    rsi_15 = _rsi(closes_15m, 14)
    atr_15 = _atr(candles_15m, 14)
    vol_ma20_15 = _sma(vol_15m, 20)
    
    # Loop 15m
    for i in range(len(candles_15m)):
        c = candles_15m[i]
        price = c.close
        vol = c.volume
        idx_1h = min(i // 4, len(candles_1h) - 1)
        
        # Détection régime 1H
        if idx_1h < len(adx_h) - 1:
            regime = _detect_regime(
                candles_1h, idx_1h, ema20_h, ema50_h, ema200_h,
                adx_h, rsi_h, atr_h, atr_ma50_h, bb_width_h
            )
        else:
            regime = Regime.UNKNOWN
        
        daily_dd = 0.0
        
        # ── Fermeture positions ──────────────────────────────────
        still_open = []
        for pos in positions:
            exit_price = None
            reason = ""
            
            if pos.tp > 0 and price >= pos.tp:
                exit_price = pos.tp
                reason = "TP"
            elif price <= pos.sl:
                exit_price = pos.sl
                reason = "SL"
                consecutive_losses += 1
            elif pos.regime == Regime.BULL and i > 0:
                if ema50_15[i] > 0 and ema200_15[i] > 0 and ema50_15[i] < ema200_15[i]:
                    exit_price = price
                    reason = "TREND_BREAK"
                    consecutive_losses += 1
            
            if exit_price is not None:
                effective_exit = exit_price * (1.0 - slippage_pct)
                net = pos.size * effective_exit * (1.0 - exit_fee)
                pnl = net - pos.cost
                is_win = pnl > 0
                balance += net
                trades.append(AdaptiveTrade(
                    pos.entry, exit_price,
                    (exit_price - pos.entry) / pos.entry,
                    pnl, is_win, pos.regime, reason,
                ))
                if is_win:
                    consecutive_losses = 0
            else:
                still_open.append(pos)
        
        positions = still_open
        
        # ── Cooldown progressif ──────────────────────────────────
        if cooldown > 0:
            cooldown -= 1
        elif use_progressive_cooldown and consecutive_losses >= 2:
            cooldown = 3
            consecutive_losses = 0
        
        # ── Condition d'entrée ───────────────────────────────────
        can_enter = (
            cooldown == 0
            and len(positions) < max_positions
            and balance > 10.0
            and regime == Regime.BULL
            and daily_dd > -0.05
        )
        
        if not can_enter:
            equity[i] = balance + sum(p.size * price for p in positions)
            continue
        
        rsi = rsi_15[i]
        e50 = ema50_15[i]
        e200 = ema200_15[i]
        atr_v = atr_15[i]
        
        # ── BULL entry ───────────────────────────────────────────
        if regime == Regime.BULL and e50 > 0 and e200 > 0:
            trend_ok = e50 > e200
            price_ok = price > e50
            rsi_ok = bull_rsi_min <= rsi <= bull_rsi_max
            rsi_up = i >= 2 and rsi_15[i] > rsi_15[i - 1] > rsi_15[i - 2]
            
            slope_ok = False
            if i >= bull_slope_bars:
                ref = ema50_15[i - bull_slope_bars]
                slope_ok = ref > 0 and (e50 - ref) / ref >= bull_slope_min_pct
            
            pb_start = max(60, i - bull_pullback_bars)
            pullback_ok = any(
                lows_15m[j] <= ema50_15[j] * 1.012
                for j in range(pb_start, i) if ema50_15[j] > 0
            )
            
            bull_candle = c.close > c.open and c.close > closes_15m[i - 1] if i > 0 else False
            bull_open = sum(1 for p in positions if p.regime == Regime.BULL)
            vol_ok = vol_ma20_15[i] > 0 and vol >= 1.0 * vol_ma20_15[i]
            
            # ADX filter
            adx_ok = True
            if use_adx_filter:
                adx_ok = idx_1h < len(adx_h) and adx_h[idx_1h] > adx_threshold
            
            if (trend_ok and price_ok and rsi_ok and rsi_up and slope_ok
                and pullback_ok and bull_candle and bull_open == 0 and vol_ok
                and adx_ok):
                
                cost = balance * bull_alloc_pct
                if cost > 1.0:
                    actual_entry = c.close * (1.0 + slippage_pct)
                    fee_in = cost * entry_fee
                    size = (cost - fee_in) / actual_entry
                    balance -= cost
                    
                    sl_price = actual_entry * (1.0 - bull_sl_pct)
                    
                    # TP dynamique
                    if use_dynamic_tp and atr_v > 0:
                        atr_pct = atr_v / actual_entry
                        if atr_pct < 0.005:
                            tp_pct = 0.12
                        elif atr_pct < 0.015:
                            tp_pct = 0.10
                        else:
                            tp_pct = 0.08
                    else:
                        tp_pct = bull_tp_pct
                    
                    positions.append(_Position(
                        entry=actual_entry, size=size, cost=cost,
                        sl=sl_price,
                        tp=actual_entry * (1.0 + tp_pct),
                        peak=actual_entry, regime=Regime.BULL,
                    ))
        
        equity[i] = balance + sum(p.size * price for p in positions)
    
    return balance, trades, equity

# ═══════════════════════════════════════════════════════════════════════════
# Runner
# ═══════════════════════════════════════════════════════════════════════════

def test_scenarios():
    """Tester 5 scénarios sur ETH/BTC/SOL."""
    
    print("\n" + "="*100)
    print("  COMPARAISON 5 RECOMMANDATIONS — ADAPTIVE BULL")
    print("="*100)
    
    end = datetime(2026, 5, 4, tzinfo=timezone.utc)
    start = end - timedelta(days=180)
    
    pairs = ["ETH-USD", "BTC-USD", "SOL-USD"]
    
    scenarios = {
        "01_BASELINE": {
            'description': "Original (SL -1.5%)",
            'config': {}
        },
        "02_SL_2PCT": {
            'description': "SL -2.0% (Recommandation #1)",
            'config': {'sl_pct': 0.020}
        },
        "03_ADX_FILTER": {
            'description': "+ ADX > 25 BULL filter (Recommandation #2)",
            'config': {'use_adx': True, 'adx_threshold': 25}
        },
        "04_PROGRESSIVE_CD": {
            'description': "+ Cooldown progressif 2 pertes (Recommandation #3)",
            'config': {'use_progressive_cooldown': True}
        },
        "05_DYNAMIC_TP": {
            'description': "+ TP dynamique par volatilité (Recommandation #4)",
            'config': {'use_dynamic_tp': True}
        },
        "06_ALL_TOGETHER": {
            'description': "🎯 TOUTES recommandations (1+2+3+4)",
            'config': {
                'sl_pct': 0.020,
                'use_adx': True,
                'adx_threshold': 25,
                'use_progressive_cooldown': True,
                'use_dynamic_tp': True
            }
        },
    }
    
    results = defaultdict(list)
    
    for pair in pairs:
        print(f"\n📊 {pair} ({start.date()} → {end.date()})")
        print("-"*100)
        
        c15 = download_candles(pair, start, end, interval="15m")
        c1h = download_candles(pair, start, end, interval="1h")
        
        if not c15 or not c1h:
            print(f"  ❌ Données manquantes")
            continue
        
        for scenario_name, scenario_data in scenarios.items():
            desc = scenario_data['description']
            config = scenario_data['config']
            
            bal, trades, eq = _run_adaptive_with_config(c15, c1h, 1000.0, config)
            
            wins = sum(1 for t in trades if t.is_win)
            wr = wins / len(trades) if trades else 0
            pnl = bal - 1000.0
            
            results[scenario_name].append({
                'pair': pair,
                'trades': len(trades),
                'wr': wr,
                'pnl': pnl,
                'bal': bal
            })
            
            mark = "📈" if pnl > 0 else "📉"
            print(f"  {mark} {scenario_name:<12s} | {len(trades):>3d} trades | WR {wr:>5.1%} | PnL ${pnl:>+8.2f}")
    
    # Summary par scenario
    print("\n" + "="*100)
    print("  RÉSUMÉ PAR SCÉNARIO (moyenne 3 paires)")
    print("="*100)
    
    for scenario_name, scenario_data in scenarios.items():
        desc = scenario_data['description']
        res_list = results[scenario_name]
        
        if not res_list:
            continue
        
        avg_trades = sum(r['trades'] for r in res_list) / len(res_list)
        avg_wr = sum(r['wr'] for r in res_list) / len(res_list)
        avg_pnl = sum(r['pnl'] for r in res_list) / len(res_list)
        
        mark = "🟢" if avg_pnl > 0 else "🔴"
        print(f"  {mark} {scenario_name:<12s} | {desc:<50s} | Trades {avg_trades:>5.0f} | WR {avg_wr:>5.1%} | Avg PnL ${avg_pnl:>+7.2f}")

if __name__ == "__main__":
    test_scenarios()
