#!/bin/bash
sleep 120
echo "=== LOG ($(date)) ==="
tail -80 /tmp/momentum_backtest.log
echo "=== PROCESS ==="
ps aux | grep run_backtest_momentum | grep -v grep || echo "PROCESS TERMINE"
