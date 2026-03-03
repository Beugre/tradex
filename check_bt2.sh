#!/bin/bash
sleep 30
echo "=== LOG ($(date)) ==="
tail -40 /tmp/momentum_backtest.log
echo "=== PROCESS ==="
ps aux | grep run_backtest_momentum | grep -v grep
