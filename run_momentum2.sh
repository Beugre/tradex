#!/bin/bash
cd "/Users/yoannbeugre/Documents/Documents - MacBook Pro de Yoann/DEV/tradeX"
PYTHONPATH=. .venv/bin/python -u backtest/run_backtest_momentum.py --months 72 --no-show 2>&1 | tee /tmp/momentum_backtest.log
echo "EXIT CODE: $?"
