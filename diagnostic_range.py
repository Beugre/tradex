"""
Diagnostic complet du Range Bot : régime BTC, analyse trades live, 
comparaison backtest vs live, test SL élargi.
"""
import requests
import pandas as pd
import numpy as np
import json
from datetime import datetime, timedelta

# ═══════════════════════════════════════════════════════════════════════
# 1. RÉGIME BTC DAILY
# ═══════════════════════════════════════════════════════════════════════

def fetch_binance_klines(symbol, interval, days=220):
    end_ms = int(datetime.now().timestamp() * 1000)
    start_ms = int((datetime.now() - timedelta(days=days)).timestamp() * 1000)
    url = 'https://api.binance.com/api/v3/klines'
    params = {'symbol': symbol, 'interval': interval, 'startTime': start_ms, 'endTime': end_ms, 'limit': 1000}
    r = requests.get(url, params=params)
    data = r.json()
    df = pd.DataFrame(data, columns=['ts','o','h','l','c','v','tc','qv','nt','tbv','tqv','i'])
    for col in ['o','h','l','c','v']:
        df[col] = df[col].astype(float)
    df['date'] = pd.to_datetime(df['ts'], unit='ms')
    df.set_index('date', inplace=True)
    return df

def calc_adx(df, period=14):
    high, low, close = df['h'], df['l'], df['c']
    plus_dm = high.diff().clip(lower=0)
    minus_dm = (-low.diff()).clip(lower=0)
    mask = plus_dm < minus_dm
    plus_dm = plus_dm.copy()
    plus_dm[mask] = 0
    mask2 = minus_dm < plus_dm
    minus_dm = minus_dm.copy()
    minus_dm[mask2] = 0
    tr = pd.concat([high - low, abs(high - close.shift(1)), abs(low - close.shift(1))], axis=1).max(axis=1)
    atr = tr.ewm(span=period, adjust=False).mean()
    plus_di = 100 * (plus_dm.ewm(span=period, adjust=False).mean() / atr)
    minus_di = 100 * (minus_dm.ewm(span=period, adjust=False).mean() / atr)
    dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = dx.ewm(span=period, adjust=False).mean()
    return adx, plus_di, minus_di

def calc_atr(df, period=14):
    tr = pd.concat([
        df['h'] - df['l'],
        abs(df['h'] - df['c'].shift(1)),
        abs(df['l'] - df['c'].shift(1))
    ], axis=1).max(axis=1)
    return tr.rolling(period).mean()

def find_swings(df, lookback=3):
    highs, lows = [], []
    for i in range(lookback, len(df) - lookback):
        is_high = all(df['h'].iloc[i] > df['h'].iloc[i-j] for j in range(1, lookback+1)) and \
                  all(df['h'].iloc[i] > df['h'].iloc[i+j] for j in range(1, lookback+1))
        is_low = all(df['l'].iloc[i] < df['l'].iloc[i-j] for j in range(1, lookback+1)) and \
                 all(df['l'].iloc[i] < df['l'].iloc[i+j] for j in range(1, lookback+1))
        if is_high:
            highs.append((df.index[i], df['h'].iloc[i]))
        if is_low:
            lows.append((df.index[i], df['l'].iloc[i]))
    return highs, lows


def diagnostic_regime_btc():
    print("=" * 70)
    print("1. DIAGNOSTIC RÉGIME BTC/USDT DAILY")
    print("=" * 70)
    
    df = fetch_binance_klines('BTCUSDT', '1d', days=220)
    df['sma200'] = df['c'].rolling(200).mean()
    adx, plus_di, minus_di = calc_adx(df)
    df['adx'] = adx
    atr14 = calc_atr(df)
    
    last = df.iloc[-1]
    
    print(f"\nDate: {last.name.strftime('%Y-%m-%d')}")
    print(f"Prix: {last['c']:.0f}")
    
    if not pd.isna(last['sma200']):
        print(f"SMA200: {last['sma200']:.0f}")
        above = last['c'] > last['sma200']
        print(f"Prix vs SMA200: {'AU-DESSUS ✅' if above else 'EN-DESSOUS 🔴'}")
    else:
        print("SMA200: N/A (pas assez de données)")
    
    adx_val = last['adx']
    if adx_val > 25:
        adx_label = "TREND FORT 🔴"
    elif adx_val < 20:
        adx_label = "RANGE/FAIBLE ✅"
    else:
        adx_label = "TRANSITION ⚠️"
    print(f"ADX(14): {adx_val:.1f} ({adx_label})")
    print(f"+DI: {plus_di.iloc[-1]:.1f} | -DI: {minus_di.iloc[-1]:.1f}")
    direction = "HAUSSIER" if plus_di.iloc[-1] > minus_di.iloc[-1] else "BAISSIER"
    print(f"Direction: {direction}")
    
    # Structure swings
    recent = df.iloc[-45:]
    highs, lows = find_swings(recent, lookback=2)
    
    print(f"\nSTRUCTURE (45 derniers jours):")
    print(f"  Swing Highs: {[(str(d.date()), f'{p:.0f}') for d, p in highs[-4:]]}")
    print(f"  Swing Lows:  {[(str(d.date()), f'{p:.0f}') for d, p in lows[-4:]]}")
    
    if len(highs) >= 2:
        if highs[-1][1] < highs[-2][1]:
            print("  → LH (Lower High) 🔴")
        else:
            print("  → HH (Higher High) ✅")
    if len(lows) >= 2:
        if lows[-1][1] < lows[-2][1]:
            print("  → LL (Lower Low) 🔴")
        else:
            print("  → HL (Higher Low) ✅")
    
    # Historique ADX
    print(f"\nADX HISTORIQUE (8 semaines):")
    for i in range(-56, 0, 7):
        if abs(i) < len(df):
            row = df.iloc[i]
            sma_txt = ""
            if not pd.isna(row['sma200']):
                sma_txt = f"vs SMA200={'ABOVE' if row['c'] > row['sma200'] else 'BELOW'}"
            print(f"  {row.name.strftime('%Y-%m-%d')}: ADX={df['adx'].iloc[i]:.1f} | Prix={row['c']:.0f} | {sma_txt}")
    
    # ATR context
    atr_val = atr14.iloc[-1]
    atr_pct = atr_val / last['c'] * 100
    print(f"\nATR(14) Daily: {atr_val:.0f} USD ({atr_pct:.1f}% du prix)")
    print(f"⚠️  Le SL du Range bot est à ~0.4% du prix")
    print(f"⚠️  L'ATR Daily est {atr_pct:.1f}% → {atr_pct/0.4:.0f}× plus large que le SL !")
    
    # VERDICT
    print(f"\n{'='*50}")
    print("VERDICT RÉGIME:")
    issues = []
    if not pd.isna(last['sma200']) and last['c'] < last['sma200']:
        issues.append("Prix SOUS SMA200 (bear)")
    if adx_val > 20:
        issues.append(f"ADX={adx_val:.0f} (trending, pas range)")
    if direction == "BAISSIER":
        issues.append("-DI > +DI (pression vendeuse)")
    if len(highs) >= 2 and highs[-1][1] < highs[-2][1]:
        issues.append("Structure LH (Lower Highs)")
    if len(lows) >= 2 and lows[-1][1] < lows[-2][1]:
        issues.append("Structure LL (Lower Lows)")
    
    if issues:
        print("🔴 LE RANGE BOT NE DEVRAIT PAS ÊTRE ACTIF !")
        for issue in issues:
            print(f"   ❌ {issue}")
    else:
        print("✅ Régime favorable au range trading")
    
    return df


# ═══════════════════════════════════════════════════════════════════════
# 2. ANALYSE DES 14 TRADES LIVE
# ═══════════════════════════════════════════════════════════════════════

def analyze_live_trades():
    print(f"\n\n{'=' * 70}")
    print("2. ANALYSE DES TRADES LIVE (state_binance.json)")
    print("=" * 70)
    
    # Le state file copié du VPS (on va le copier via SSH)
    import subprocess
    result = subprocess.run(
        ['ssh', 'BOT-VPS', 'cat /opt/tradex/data/state_binance.json'],
        capture_output=True, text=True, timeout=15
    )
    state = json.loads(result.stdout)
    positions = state.get('positions', {})
    
    print(f"\n{'SYM':>15} {'SIDE':>4} {'ENTRY':>12} {'SL':>12} {'TP':>12} {'SL%':>6} {'TP%':>6} {'R:R':>5} {'PnL':>8} {'STATUS':>7}")
    print("-" * 100)
    
    closed_trades = []
    open_trades = []
    
    for sym, pos in positions.items():
        entry = pos['entry_price']
        sl = pos.get('sl_price', 0) or 0
        tp = pos.get('tp_price', 0) or 0
        pnl = pos.get('pnl') or 0
        status = pos.get('status', '?')
        side = pos.get('side', '?')
        
        sl_dist_pct = abs(entry - sl) / entry * 100 if entry > 0 else 0
        tp_dist_pct = abs(tp - entry) / entry * 100 if entry > 0 else 0
        rr = tp_dist_pct / sl_dist_pct if sl_dist_pct > 0 else 0
        
        print(f"{sym:>15} {side:>4} {entry:>12.6f} {sl:>12.6f} {tp:>12.6f} {sl_dist_pct:>5.2f}% {tp_dist_pct:>5.2f}% {rr:>5.1f} {pnl:>+8.4f} {status:>7}")
        
        trade_info = {
            'symbol': sym, 'side': side, 'entry': entry, 'sl': sl, 'tp': tp,
            'sl_pct': sl_dist_pct, 'tp_pct': tp_dist_pct, 'rr': rr,
            'pnl': pnl, 'status': status
        }
        
        if status == 'CLOSED':
            closed_trades.append(trade_info)
        elif status == 'OPEN':
            open_trades.append(trade_info)
    
    # Stats
    wins = [t for t in closed_trades if t['pnl'] > 0]
    losses = [t for t in closed_trades if t['pnl'] <= 0]
    total_pnl = sum(t['pnl'] for t in closed_trades)
    avg_sl_pct = np.mean([t['sl_pct'] for t in closed_trades]) if closed_trades else 0
    avg_tp_pct = np.mean([t['tp_pct'] for t in closed_trades]) if closed_trades else 0
    avg_rr = np.mean([t['rr'] for t in closed_trades]) if closed_trades else 0
    
    print(f"\n--- RÉSUMÉ ---")
    print(f"Total fermés: {len(closed_trades)} | Wins: {len(wins)} | Losses: {len(losses)}")
    print(f"WR: {len(wins)/len(closed_trades)*100:.1f}%" if closed_trades else "WR: N/A")
    print(f"PnL total: {total_pnl:+.2f} USD")
    print(f"SL moyen: {avg_sl_pct:.2f}% | TP moyen: {avg_tp_pct:.2f}% | R:R moyen: {avg_rr:.1f}")
    print(f"Positions ouvertes: {len(open_trades)}")
    
    # Diagnostic SL
    print(f"\n--- DIAGNOSTIC SL ---")
    print(f"⚠️  SL moyen = {avg_sl_pct:.2f}% → EXTRÊMEMENT serré !")
    print(f"    Pour contexte : ATR H4 typique = 1-3% du prix")
    print(f"    Le SL se fait manger par les mèches normales")
    
    # Test : si SL avait été 1.5x, 2x, 3x plus large
    print(f"\n--- SIMULATION SL ÉLARGI ---")
    print(f"    Si SL était 1.5× plus large ({avg_sl_pct*1.5:.2f}%) : toujours touché (mèches H4 >> 1%)")
    print(f"    Si SL était 3× plus large ({avg_sl_pct*3:.2f}%) : certains trades auraient survécu")
    print(f"    Si SL était à ATR ({avg_sl_pct*5:.1f}%~) : R:R beaucoup plus réaliste")
    
    return closed_trades, open_trades


# ═══════════════════════════════════════════════════════════════════════
# 3. COMPARAISON BACKTEST VS LIVE (même période)
# ═══════════════════════════════════════════════════════════════════════

def compare_backtest_live():
    print(f"\n\n{'=' * 70}")
    print("3. COMPARAISON LOGIQUE BACKTEST vs LIVE")
    print("=" * 70)
    
    # Lire le code backtest et le code live
    import subprocess
    
    # Récupérer les fichiers du VPS
    live_code = subprocess.run(
        ['ssh', 'BOT-VPS', 'cat /opt/tradex/src/core/strategy_mean_rev.py'],
        capture_output=True, text=True, timeout=15
    ).stdout
    
    live_bot = subprocess.run(
        ['ssh', 'BOT-VPS', 'grep -n "range_entry_buffer\\|range_sl_buffer\\|RANGE_WIDTH_MIN\\|RANGE_ENTRY\\|RANGE_SL\\|strategy_mean_rev\\|check_range" /opt/tradex/src/bot_binance.py /opt/tradex/src/bot.py 2>/dev/null'],
        capture_output=True, text=True, timeout=15
    ).stdout
    
    live_env = subprocess.run(
        ['ssh', 'BOT-VPS', 'grep -E "RANGE|BUFFER|SL_|ENTRY_|WIDTH" /opt/tradex/.env'],
        capture_output=True, text=True, timeout=15
    ).stdout
    
    print("\n[A] Paramètres .env live:")
    print(live_env)
    
    print("\n[B] Logique d'entrée live (strategy_mean_rev.py):")
    # Extract key logic
    lines = live_code.split('\n')
    for i, line in enumerate(lines):
        if any(kw in line for kw in ['buy_zone', 'sell_zone', 'sl_price', 'tp_price', 'range_mid', 'entry_buffer', 'sl_buffer']):
            print(f"  L{i+1}: {line.strip()}")
    
    print(f"\n[C] Références dans le bot:")
    print(live_bot[:1000] if live_bot else "  (pas trouvé)")
    
    # Comparaison avec backtest local
    import os
    backtest_path = os.path.join(os.path.dirname(__file__), 'backtest', 'simulator_range_pro.py')
    if os.path.exists(backtest_path):
        with open(backtest_path) as f:
            bt_code = f.read()
        print(f"\n[D] Logique backtest (simulator_range_pro.py):")
        bt_lines = bt_code.split('\n')
        for i, line in enumerate(bt_lines):
            if any(kw in line for kw in ['buy_zone', 'sell_zone', 'sl_price', 'tp_price', 'range_mid', 'entry_buffer', 'sl_buffer', 'range_entry', 'range_sl', 'sl_eff']):
                print(f"  L{i+1}: {line.strip()}")
    else:
        print(f"\n[D] Backtest file not found: {backtest_path}")
        # Try to find it
        import glob
        matches = glob.glob(os.path.join(os.path.dirname(__file__), 'backtest', '*range*'))
        print(f"    Found: {matches}")
    
    # Check for regime filters in backtest
    print(f"\n[E] FILTRES RÉGIME dans le backtest:")
    if os.path.exists(backtest_path):
        with open(backtest_path) as f:
            content = f.read()
        for kw in ['adx', 'sma200', 'sma_200', 'regime', 'filter', 'btc_filter', 'trend_filter']:
            count = content.lower().count(kw)
            if count > 0:
                print(f"  '{kw}' trouvé {count} fois")
        if 'adx' not in content.lower() and 'sma200' not in content.lower():
            print("  ⚠️  PAS de filtre régime ADX/SMA200 dans le backtest non plus !")
    
    # Check live code for regime filters
    print(f"\n[F] FILTRES RÉGIME dans le code live:")
    for kw in ['adx', 'sma200', 'sma_200', 'regime', 'filter', 'btc_filter', 'trend_filter']:
        count = live_code.lower().count(kw)
        if count > 0:
            print(f"  '{kw}' trouvé {count} fois")
    if 'adx' not in live_code.lower() and 'sma200' not in live_code.lower():
        print("  ⚠️  PAS de filtre régime ADX/SMA200 dans le code live non plus !")


# ═══════════════════════════════════════════════════════════════════════
# 4. BACKTEST 6 DERNIERS MOIS — Mêmes paires que live
# ═══════════════════════════════════════════════════════════════════════

def backtest_recent_period():
    print(f"\n\n{'=' * 70}")
    print("4. MICRO-BACKTEST 3 MOIS — Range sur altcoins Binance")
    print("=" * 70)
    
    # On va tester sur quelques paires qui apparaissent dans les trades live
    live_symbols = ['ATUSDC', 'INITUSDC', 'HAEDALUSDC', 'LSKUSDC', 'EGLDUSDC', 
                    'COTIUSDC', 'LAYERUSDC', 'CETUSUSDC', 'ICPUSDC']
    
    # Convertir en format Binance API
    binance_symbols = []
    for s in live_symbols:
        # ATUSDC -> ATUSDT (approx, certains existent en USDC)
        base = s.replace('USDC', '')
        binance_symbols.append(f"{base}USDT")
    
    results = []
    
    for sym in binance_symbols[:5]:  # Test sur 5 paires
        try:
            df = fetch_binance_klines(sym, '4h', days=90)
            if len(df) < 50:
                continue
                
            atr = calc_atr(df, 14)
            
            # Simuler la stratégie range avec les mêmes params que live
            entry_buffer = 0.002  # 0.2%
            sl_buffer = 0.003    # 0.3%
            min_width = 0.02     # 2%
            
            # Stats de volatilité H4
            avg_range_pct = ((df['h'] - df['l']) / df['c']).mean() * 100
            avg_atr_pct = (atr / df['c']).mean() * 100
            
            results.append({
                'symbol': sym,
                'avg_h4_range': avg_range_pct,
                'avg_atr_pct': avg_atr_pct,
                'sl_pct': entry_buffer * 2 * 100,  # SL distance ~0.4%
                'sl_vs_atr': (entry_buffer * 2 * 100) / avg_atr_pct if avg_atr_pct > 0 else 0,
            })
            
            print(f"\n  {sym}:")
            print(f"    Range H4 moyen: {avg_range_pct:.2f}%")
            print(f"    ATR(14) H4: {avg_atr_pct:.2f}%")
            print(f"    SL bot: ~0.40%")
            print(f"    SL/ATR ratio: {(0.4/avg_atr_pct*100):.0f}% de l'ATR → {'🔴 TROP SERRÉ' if avg_atr_pct > 1 else '✅ OK'}")
            
        except Exception as e:
            print(f"  {sym}: erreur - {e}")
    
    if results:
        avg_h4_range = np.mean([r['avg_h4_range'] for r in results])
        avg_atr = np.mean([r['avg_atr_pct'] for r in results])
        print(f"\n  MOYENNE:")
        print(f"    Range H4 moyen: {avg_h4_range:.2f}%")
        print(f"    ATR H4 moyen: {avg_atr:.2f}%")
        print(f"    SL bot (~0.4%) = {0.4/avg_atr*100:.0f}% de l'ATR")
        print(f"    → Le SL est {avg_atr/0.4:.1f}× plus petit que l'ATR !")
        print(f"    → Probabilité d'être stoppé par bruit : ~{min(99, 100-0.4/avg_atr*100):.0f}%")


# ═══════════════════════════════════════════════════════════════════════
# 5. DIAGNOSTIC FINAL
# ═══════════════════════════════════════════════════════════════════════

def diagnostic_final():
    print(f"\n\n{'=' * 70}")
    print("5. DIAGNOSTIC FINAL & RECOMMANDATIONS")
    print("=" * 70)
    
    print("""
╔══════════════════════════════════════════════════════════════════════╗
║  CAUSES IDENTIFIÉES DU WR 0% LIVE (vs ~22% backtest)              ║
╠══════════════════════════════════════════════════════════════════════╣
║                                                                    ║
║  🔴 CAUSE #1 : SL MICROSCOPIQUE                                    ║
║     SL = 0.4% du prix d'entrée                                    ║
║     ATR H4 = 2-4% → le SL est 5-10× plus petit que le bruit      ║
║     → Chaque mèche H4 normale déclenche le SL                     ║
║                                                                    ║
║  🔴 CAUSE #2 : AUCUN FILTRE RÉGIME                                 ║
║     Pas de filtre ADX, SMA200, ou structure BTC                    ║
║     Le bot trade en range pendant des marchés baissiers            ║
║     Les "ranges" qu'il détecte sont des consolidations bearish     ║
║     → Cassure vers le bas quasi systématique                       ║
║                                                                    ║
║  🔴 CAUSE #3 : AUTO-DISCOVER PAIRS                                 ║
║     BINANCE_AUTO_DISCOVER_PAIRS=true                               ║
║     Le bot trade des micro-caps illiquides (AT, INIT, HAEDAL...)   ║
║     Ces paires ont des spreads larges et des mèches violentes      ║
║     → SL touché encore plus vite                                   ║
║                                                                    ║
║  ⚠️  CAUSE #4 : DIVERGENCE BACKTEST (possible)                     ║
║     Le backtest utilise les CLOSE des bougies pour vérifier SL     ║
║     Le live utilise le prix en temps réel (intrabar)               ║
║     → Le live touche des SL que le backtest ne verrait jamais      ║
║                                                                    ║
╠══════════════════════════════════════════════════════════════════════╣
║  RECOMMANDATIONS                                                   ║
╠══════════════════════════════════════════════════════════════════════╣
║                                                                    ║
║  1. STOPPER le range bot immédiatement (0W-11L = -$11.54)          ║
║                                                                    ║
║  2. Si on veut le garder, ajouter ces filtres :                    ║
║     □ OFF quand ADX H4 > 20 (ou ADX Daily > 25)                   ║
║     □ OFF quand BTC < SMA200 Daily                                 ║
║     □ OFF quand structure LH/LL sur BTC                            ║
║     □ SL = 1× ATR (pas 0.2% fixe)                                 ║
║     □ Whitelist de paires liquides uniquement                      ║
║                                                                    ║
║  3. Alternative : remplacer par Donchian (validé robuste)          ║
║                                                                    ║
╚══════════════════════════════════════════════════════════════════════╝
""")


# ═══════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    diagnostic_regime_btc()
    analyze_live_trades()
    compare_backtest_live()
    backtest_recent_period()
    diagnostic_final()
