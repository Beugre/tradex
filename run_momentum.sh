#!/bin/bash
cd "/Users/yoannbeugre/Documents/Documents - MacBook Pro de Yoann/DEV/tradeX"
PYTHONPATH=. .venv/bin/python backtest/run_backtest_momentum.py --months 72 --no-show
