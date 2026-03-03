#!/bin/bash
sleep 180
echo "=== LOG TAIL ($(date)) ==="
tail -80 /tmp/momentum_backtest.log
echo "=== PROCESS ==="
ps aux | grep momentum | grep python | grep -v grep || echo "PROCESS TERMINE"
