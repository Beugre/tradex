#!/usr/bin/env python
"""
Point d'entrée du backtest TradeX.

Usage :
    python -m backtest.run_backtest                   # défaut : 2 ans, 5 paires, $1000
    python -m backtest.run_backtest --balance 5000     # capital custom
    python -m backtest.run_backtest --years 1          # 1 an seulement
    python -m backtest.run_backtest --start 2022-02-20 --end 2026-02-20
    python -m backtest.run_backtest --ema              # activer le filtre EMA200 D1
    python -m backtest.run_backtest --no-show          # ne pas ouvrir le graphique
"""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import datetime, timedelta, timezone

from dotenv import load_dotenv

# Charger .env pour les paramètres de trading (pas besoin des clés Binance pour les klines publiques)
load_dotenv()

from src import config
from backtest.data_loader import download_all_pairs, download_all_pairs_d1
from backtest.simulator import BacktestConfig, BacktestEngine
from backtest.metrics import compute_metrics
from backtest.report import generate_report

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("backtest")


def main() -> None:
    parser = argparse.ArgumentParser(description="TradeX Backtest")
    parser.add_argument("--balance", type=float, default=1000.0, help="Capital initial ($)")
    parser.add_argument("--years", type=float, default=2.0, help="Nombre d'années (si pas --start/--end)")
    parser.add_argument("--start", type=str, default=None, help="Date de début (YYYY-MM-DD)")
    parser.add_argument("--end", type=str, default=None, help="Date de fin (YYYY-MM-DD)")
    parser.add_argument("--ema", action="store_true", help="Activer le filtre BTC EMA200 D1")
    parser.add_argument("--pullback", action="store_true", help="TREND pullback D1 (au lieu de breakout H4)")
    parser.add_argument("--trend-only", action="store_true", help="TREND uniquement")
    parser.add_argument("--range-only", action="store_true", help="RANGE uniquement")
    parser.add_argument("--short", action="store_true", help="Autoriser les SELL (short) pour TREND")
    parser.add_argument("--no-show", action="store_true", help="Ne pas afficher le graphique")
    args = parser.parse_args()

    # Résolution des dates
    if args.start and args.end:
        start = datetime.strptime(args.start, "%Y-%m-%d").replace(tzinfo=timezone.utc)
        end = datetime.strptime(args.end, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    else:
        end = datetime.now(tz=timezone.utc)
        start = end - timedelta(days=int(args.years * 365.25))

    pairs = config.TRADING_PAIRS
    logger.info("🚀 TradeX Backtest — %s → %s", start.date(), end.date())
    logger.info("   Paires : %s", ", ".join(pairs))

    # ── 1. Téléchargement des données ──────────────────────────────────────
    candles = download_all_pairs(pairs, start, end, interval="4h")

    # D1 candles si EMA ou pullback
    d1_candles = None
    if args.ema or args.pullback:
        d1_candles = download_all_pairs_d1(pairs, start, end)

    # ── 2. Configuration ───────────────────────────────────────────────────
    cfg = BacktestConfig(
        initial_balance=args.balance,
        risk_percent_trend=config.RISK_PERCENT_TREND,
        risk_percent_range=config.RISK_PERCENT_RANGE,
        entry_buffer_pct=config.ENTRY_BUFFER_PERCENT,
        sl_buffer_pct=config.SL_BUFFER_PERCENT,
        zero_risk_trigger_pct=config.ZERO_RISK_TRIGGER_PERCENT,
        zero_risk_lock_pct=config.ZERO_RISK_LOCK_PERCENT,
        trailing_stop_pct=config.TRAILING_STOP_PERCENT,
        max_position_pct=config.MAX_POSITION_PERCENT,
        max_simultaneous_positions=config.MAX_SIMULTANEOUS_POSITIONS,
        swing_lookback=config.SWING_LOOKBACK,
        range_width_min=config.RANGE_WIDTH_MIN,
        range_entry_buffer_pct=config.RANGE_ENTRY_BUFFER_PERCENT,
        range_sl_buffer_pct=config.RANGE_SL_BUFFER_PERCENT,
        range_cooldown_bars=config.RANGE_COOLDOWN_BARS,
        max_total_risk_pct=config.MAX_TOTAL_RISK_PERCENT,
        use_ema_filter=args.ema,
        use_d1_pullback=args.pullback,
        enable_trend=not args.range_only,
        enable_range=not args.trend_only,
        allow_short=args.short,
    )

    # ── 3. Simulation ─────────────────────────────────────────────────────
    engine = BacktestEngine(candles, cfg, d1_candles_by_symbol=d1_candles)
    result = engine.run()

    # ── 4. Métriques ───────────────────────────────────────────────────────
    metrics = compute_metrics(result)

    # ── 5. Rapport ─────────────────────────────────────────────────────────
    generate_report(result, metrics, show=not args.no_show)


if __name__ == "__main__":
    main()
