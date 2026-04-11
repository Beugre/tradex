#!/usr/bin/env python3
"""Debug rapide stratégie adaptative."""
import sys
sys.path.insert(0, '.')  # noqa

from collections import Counter
from datetime import datetime, timedelta, timezone

from backtest.data_loader import download_candles
from backtest.run_backtest_adaptive import _run_adaptive_pair

end   = datetime(2025, 1, 1, tzinfo=timezone.utc)
start = end - timedelta(days=180)
c15 = download_candles("ETH-USD", start, end, interval="15m")
c1h = download_candles("ETH-USD", start, end, interval="1h")
print(f"1H: {len(c1h)}, 15m: {len(c15)}")

bal, trades, eq = _run_adaptive_pair(c15, c1h, 200.0)
wins   = [t for t in trades if t.is_win]
losses = [t for t in trades if not t.is_win]
print(f"Total: {len(trades)} | Wins: {len(wins)} | Losses: {len(losses)}")
print(f"WR: {len(wins)/max(len(trades),1):.1%}")
if wins:
    print(f"Avg win PnL: {sum(t.pnl_abs for t in wins)/len(wins):.4f}")
if losses:
    print(f"Avg loss PnL: {sum(t.pnl_abs for t in losses)/len(losses):.4f}")
print()
print("Exit reasons:")
for r, n in sorted(Counter(t.exit_reason for t in trades).items(), key=lambda x: -x[1]):
    wc = sum(1 for t in trades if t.exit_reason == r and t.is_win)
    print(f"  {r}: {n} (wins: {wc})")
print()
print("Last 10 trades:")
for t in trades[-10:]:
    print(f"  {t.regime.value:12} entry={t.entry_price:.2f} exit={t.exit_price:.2f} pnl={t.pnl_abs:+.4f} win={t.is_win} reason={t.exit_reason}")
print(f"\nFinal balance: {bal:.2f}")
