#!/bin/bash
sleep 60
echo "=== LOG ($(date)) ==="
tail -50 /tmp/momentum_backtest.log
echo "=== PROCESS ==="
ps aux | grep run_backtest_momentum | grep -v grep
