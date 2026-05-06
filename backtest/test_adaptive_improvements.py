#!/usr/bin/env python3
"""
Test des 5 recommandations pour Adaptive Bull

Objectif : Comparer les performances avec les améliorations proposées :
  1. Augmenter SL : -1.5% → -2.0%
  2. Filtre régime : Exiger ADX > 25 en BULL
  3. Cooldown progressif : 1 bar → 2-3 bars après 2 pertes
  4. TP% dynamique : +8% fixe → +8–12% selon regime/volatilité
  5. Valider sur BTC/SOL : Paires moins correlées

Usage :
    python3 -m backtest.test_adaptive_improvements
"""

from __future__ import annotations

import sys
sys.path.insert(0, '.')

from datetime import datetime, timedelta, timezone
from backtest.data_loader import download_candles
from backtest.run_backtest_adaptive import (
    _run_adaptive_pair,
    _compute_metrics,
    PAIRS_BIG5,
)
from src.core.models import Candle
from backtest.run_backtest_adaptive import (
    _ema, _rsi, _atr, _sma, _adx, _bollinger_width,
    Regime, _Position, AdaptiveTrade, _detect_regime
)

# ═══════════════════════════════════════════════════════════════════════════
# Version AMÉLIORÉE avec les 5 recommandations
# ═══════════════════════════════════════════════════════════════════════════

def _run_adaptive_pair_improved(
    candles_15m: list[Candle],
    candles_1h: list[Candle],
    initial_balance: float,
) -> tuple[float, list[AdaptiveTrade], list[float]]:
    """Version améliorée du backtest adaptive avec les 5 recommandations."""
    
    # Config améliorée
    bull_sl_pct = 0.020  # 💡 #1 : SL -2.0% au lieu de -1.5%
    bull_tp_pct = 0.080  # TP fixe +8% (baseline, sera ajusté avec #4)
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
    consecutive_losses = 0  # 💡 #3 : Tracker les pertes consécutives
    
    # Pré-calcul indicateurs 1H
    closes_1h = [c.close for c in candles_1h]
    ema50_h = _ema(closes_1h, 50)
    ema200_h = _ema(closes_1h, 200)
    ema20_h = _ema(closes_1h, 20)
    adx_h = _adx(candles_1h, 14)  # 💡 #2 : Utiliser ADX
    rsi_h = _rsi(closes_1h, 14)
    atr_h = _atr(candles_1h, 14)
    atr_ma50_h = _sma(atr_h, 50)
    bb_width_h = _bollinger_width(closes_1h, 20, 2.0)
    
    # Pré-calcul indicateurs 15m
    closes_15m = [c.close for c in candles_15m]
    lows_15m = [c.low for c in candles_15m]
    vol_15m = [c.volume for c in candles_15m]
    ema20_15 = _ema(closes_15m, 20)
    ema50_15 = _ema(closes_15m, 50)
    ema200_15 = _ema(closes_15m, 200)
    rsi_15 = _rsi(closes_15m, 14)
    atr_15 = _atr(candles_15m, 14)
    vol_ma20_15 = _sma(vol_15m, 20)
    
    # Interpolation simple 1H → 15m (4× par bougie 1H)
    ema50_15m_aligned = []
    ema200_15m_aligned = []
    atr_15m_aligned = []
    for i in range(n):
        idx_1h = i // 4
        if idx_1h < len(ema50_h):
            ema50_15m_aligned.append(ema50_h[idx_1h])
            ema200_15m_aligned.append(ema200_h[idx_1h])
            atr_15m_aligned.append(atr_h[idx_1h])
        else:
            ema50_15m_aligned.append(ema50_15m_aligned[-1] if ema50_15m_aligned else 0)
            ema200_15m_aligned.append(ema200_15m_aligned[-1] if ema200_15m_aligned else 0)
            atr_15m_aligned.append(atr_15m_aligned[-1] if atr_15m_aligned else 0)
    
    # Loop backtest 15m
    for i in range(len(candles_15m)):
        c = candles_15m[i]
        price = c.close
        high, low = c.high, c.low
        vol = c.volume
        
        # Index 1H correspondant
        idx_1h = min(i // 4, len(candles_1h) - 1)
        
        # Détection régime 1H
        if idx_1h < len(adx_h) - 1:
            regime = _detect_regime(
                candles_1h, idx_1h, ema20_h, ema50_h, ema200_h,
                adx_h, rsi_h, atr_h, atr_ma50_h, bb_width_h, bb_width_low_pct=0.015
            )
        else:
            regime = Regime.UNKNOWN
        
        daily_dd = 0.0  # Simplifié pour ce test
        
        # Fermeture positions
        still_open = []
        for pos in positions:
            exit_price = None
            reason = ""
            
            # TP
            if pos.tp > 0 and price >= pos.tp:
                exit_price = pos.tp
                reason = "TP"
            # SL
            elif price <= pos.sl:
                exit_price = pos.sl
                reason = "SL"
                consecutive_losses += 1  # 💡 #3 : Incrémenter losess
            # Trend break BULL
            elif pos.regime == Regime.BULL:
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
                # 💡 #3 : Reset loss counter si win
                if is_win:
                    consecutive_losses = 0
            else:
                still_open.append(pos)
        
        positions = still_open
        
        # Cooldown avec progression 💡 #3
        if cooldown > 0:
            cooldown -= 1
        else:
            # Appliquer cooldown progressif après N pertes consécutives
            if consecutive_losses >= 2:
                cooldown = 3  # 3 barres = ~45 min en 15m
                consecutive_losses = 0
        
        # Conditions d'entrée
        can_enter = (
            cooldown == 0
            and len(positions) < max_positions
            and balance > 10.0
            and regime == Regime.BULL  # BULL only
            and daily_dd > -0.05
        )
        
        if not can_enter:
            equity[i] = balance + sum(p.size * price for p in positions)
            continue
        
        rsi = rsi_15[i]
        e50 = ema50_15[i]
        e200 = ema200_15[i]
        atr_v = atr_15[i]
        
        # BULL entry avec améliorations
        if regime == Regime.BULL and e50 > 0 and e200 > 0:
            trend_ok = e50 > e200
            price_ok = price > e50
            rsi_ok = bull_rsi_min <= rsi <= bull_rsi_max
            rsi_up = i >= 2 and rsi_15[i] > rsi_15[i - 1] > rsi_15[i - 2]
            
            # Slope check
            slope_ok = False
            if i >= bull_slope_bars:
                ref = ema50_15[i - bull_slope_bars]
                slope_ok = ref > 0 and (e50 - ref) / ref >= bull_slope_min_pct
            
            # Pullback
            pb_start = max(60, i - bull_pullback_bars)
            pullback_ok = any(
                lows_15m[j] <= ema50_15[j] * 1.012
                for j in range(pb_start, i)
                if ema50_15[j] > 0
            )
            
            bull_candle = c.close > c.open and c.close > closes_15m[i - 1]
            bull_open = sum(1 for p in positions if p.regime == Regime.BULL)
            vol_ok = vol_ma20_15[i] > 0 and vol >= 1.0 * vol_ma20_15[i]
            
            # 💡 #2 : Filtre ADX > 25 pour confirmer trend
            adx_ok = idx_1h < len(adx_h) and adx_h[idx_1h] > 25
            
            if (trend_ok and price_ok and rsi_ok and rsi_up and slope_ok
                and pullback_ok and bull_candle and bull_open == 0 and vol_ok
                and adx_ok):  # 💡 #2 : Ajouter ADX filter
                
                cost = balance * bull_alloc_pct
                if cost > 1.0:
                    actual_entry = c.close * (1.0 + slippage_pct)
                    fee_in = cost * entry_fee
                    size = (cost - fee_in) / actual_entry
                    balance -= cost
                    
                    # SL avec la nouvelle valeur 💡 #1
                    sl_price = actual_entry * (1.0 - bull_sl_pct)
                    
                    # 💡 #4 : TP dynamique selon volatilité (ATR)
                    if atr_v > 0:
                        # TP% augmente si volatilité basse
                        atr_pct = atr_v / actual_entry
                        if atr_pct < 0.005:  # Basse volatilité
                            tp_pct = 0.12  # +12%
                        elif atr_pct < 0.015:  # Moyenne
                            tp_pct = 0.10  # +10%
                        else:  # Haute
                            tp_pct = 0.08  # +8%
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
# Comparaison Baseline vs Amélioré
# ═══════════════════════════════════════════════════════════════════════════

def compare_improvements():
    """Comparer baseline vs amélioré sur ETH 6 mois et BTC/SOL."""
    
    print("\n" + "="*80)
    print("  🚀 TEST DES 5 RECOMMANDATIONS — ADAPTIVE BULL")
    print("="*80)
    
    end = datetime(2026, 5, 4, tzinfo=timezone.utc)
    start = end - timedelta(days=180)
    
    pairs_test = ["ETH-USD", "BTC-USD", "SOL-USD"]
    
    for pair in pairs_test:
        print(f"\n📊 {pair} ({start.date()} → {end.date()})")
        print("-"*80)
        
        c15 = download_candles(pair, start, end, interval="15m")
        c1h = download_candles(pair, start, end, interval="1h")
        
        if not c15 or not c1h:
            print(f"  ❌ Données manquantes")
            continue
        
        # Baseline
        bal_base, trades_base, eq_base = _run_adaptive_pair(c15, c1h, 1000.0)
        
        # Improved
        bal_imp, trades_imp, eq_imp = _run_adaptive_pair_improved(c15, c1h, 1000.0)
        
        # Stats baseline
        wins_base = sum(1 for t in trades_base if t.is_win)
        wr_base = wins_base / len(trades_base) if trades_base else 0
        pnl_base = bal_base - 1000.0
        
        # Stats amélioré
        wins_imp = sum(1 for t in trades_imp if t.is_win)
        wr_imp = wins_imp / len(trades_imp) if trades_imp else 0
        pnl_imp = bal_imp - 1000.0
        
        # Affichage
        print(f"  Baseline    | {len(trades_base):>3d} trades | WR {wr_base:>5.1%} | PnL ${pnl_base:>+8.2f} | Bal ${bal_base:>8.2f}")
        print(f"  Amélioré    | {len(trades_imp):>3d} trades | WR {wr_imp:>5.1%} | PnL ${pnl_imp:>+8.2f} | Bal ${bal_imp:>8.2f}")
        
        pnl_delta = pnl_imp - pnl_base
        delta_pct = (pnl_delta / abs(pnl_base)) * 100 if pnl_base != 0 else 0
        
        if pnl_delta > 0:
            print(f"  📈 GAIN : +${pnl_delta:+.2f} ({delta_pct:+.1f}%)")
        else:
            print(f"  📉 PERTE : ${pnl_delta:+.2f} ({delta_pct:+.1f}%)")
    
    print("\n" + "="*80)

if __name__ == "__main__":
    compare_improvements()
