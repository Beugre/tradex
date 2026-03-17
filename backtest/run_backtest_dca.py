"""
Backtest DCA RSI v2 — simulation jour par jour.

Utilise :
  - Bougies D1 BTC/ETH via Binance public API (data_loader)
  - MVRV historique via CoinMetrics Community API
  - Logique pure de src/core/dca_engine (v2)
  - Indicateurs RSI / SMA de src/core/indicators

Usage :
    python -m backtest.run_backtest_dca [--start 2022-01-01] [--end 2025-03-01] [--no-mvrv]
"""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import requests

# Local imports
from backtest.data_loader import download_candles, SYMBOL_MAP
from src.core.dca_engine import (
    DCAConfig,
    DCADecision,
    DCAState,
    MarketRegime,
    check_crash_triggers,
    classify_regime,
    classify_rsi,
    compute_crash_anchor,
    compute_daily_amount,
    compute_mvrv_multiplier,
    compute_regime_allocation,
    compute_rolling_high,
    format_summary,
    remaining_crash_budget,
    remaining_dca_budget,
    reset_crash_levels_if_recovered,
    reset_period_counters,
    split_allocation,
)
from src.core.indicators import rsi_series, sma
from src.core.models import Candle

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger("backtest.dca")


# ── MVRV historical fetch ─────────────────────────────────────────────────────


def fetch_mvrv_history(
    start: datetime,
    end: datetime,
    asset: str = "btc",
) -> dict[str, float]:
    """Fetch historical daily MVRV from CoinMetrics Community API.

    Returns dict mapping date string "YYYY-MM-DD" to MVRV float.
    """
    url = "https://community-api.coinmetrics.io/v4/timeseries/asset-metrics"
    all_data: dict[str, float] = {}

    page_size = 1000
    start_str = start.strftime("%Y-%m-%dT00:00:00Z")
    end_str = end.strftime("%Y-%m-%dT00:00:00Z")

    logger.info("Fetching MVRV history %s to %s...", start.date(), end.date())

    next_page = None
    while True:
        params = {
            "assets": asset,
            "metrics": "CapMVRVCur",
            "frequency": "1d",
            "start_time": start_str,
            "end_time": end_str,
            "page_size": page_size,
        }
        if next_page:
            params["next_page_token"] = next_page

        try:
            resp = requests.get(url, params=params, timeout=30)
            resp.raise_for_status()
            body = resp.json()
        except Exception as e:
            logger.warning("MVRV fetch error: %s", e)
            break

        data = body.get("data", [])
        for entry in data:
            date_str = entry.get("time", "")[:10]  # "2022-01-01T00:00:00.000000000Z"
            mvrv_val = entry.get("CapMVRVCur")
            if date_str and mvrv_val is not None:
                try:
                    all_data[date_str] = float(mvrv_val)
                except (ValueError, TypeError):
                    pass

        next_page = body.get("next_page_token")
        if not next_page or not data:
            break

    logger.info("Fetched %d MVRV data points", len(all_data))
    return all_data


# ── Candle helpers ─────────────────────────────────────────────────────────────


def download_daily_candles(
    symbol: str,
    start: datetime,
    end: datetime,
) -> list[Candle]:
    """Download daily candles via Binance public API."""
    warmup_days = 250  # For MA200 + RSI warmup
    real_start = start - timedelta(days=warmup_days)
    logger.info("Downloading %s D1 (%s to %s, warmup %dd)...",
                symbol, real_start.date(), end.date(), warmup_days)
    candles = download_candles(symbol, real_start, end, interval="1d", use_cache=True)
    logger.info("  %s: %d daily candles", symbol, len(candles))
    return candles


# ── Backtest engine ────────────────────────────────────────────────────────────


def run_backtest(
    start: datetime,
    end: datetime,
    cfg: DCAConfig,
    use_mvrv: bool = True,
) -> dict:
    """Run the DCA v2 backtest and return performance summary."""

    # ── Download data ──────────────────────────────────────────────────────
    btc_candles = download_daily_candles("BTC-USD", start, end)
    eth_candles = download_daily_candles("ETH-USD", start, end)

    if len(btc_candles) < 215:
        logger.error("Not enough BTC candles (%d < 215)", len(btc_candles))
        return {}

    # Build price maps {date_str: close_price}
    btc_prices: dict[str, float] = {}
    eth_prices: dict[str, float] = {}
    for c in btc_candles:
        d = datetime.fromtimestamp(c.timestamp / 1000, tz=timezone.utc).strftime("%Y-%m-%d")
        btc_prices[d] = c.close
    for c in eth_candles:
        d = datetime.fromtimestamp(c.timestamp / 1000, tz=timezone.utc).strftime("%Y-%m-%d")
        eth_prices[d] = c.close

    # Compute RSI series for BTC
    btc_closes = [c.close for c in btc_candles]
    btc_rsi_vals = rsi_series(btc_candles, period=14)
    btc_sma200 = sma(btc_closes, 200)

    # Build date-indexed maps
    btc_date_idx: dict[str, int] = {}
    for i, c in enumerate(btc_candles):
        d = datetime.fromtimestamp(c.timestamp / 1000, tz=timezone.utc).strftime("%Y-%m-%d")
        btc_date_idx[d] = i

    # MVRV history
    mvrv_data: dict[str, float] = {}
    if use_mvrv:
        mvrv_warmup = start - timedelta(days=30)
        mvrv_data = fetch_mvrv_history(mvrv_warmup, end)

    # ── State init ─────────────────────────────────────────────────────────
    state = DCAState(start_date=start.strftime("%Y-%m-%d"))

    # ── Day-by-day simulation ──────────────────────────────────────────────
    current_date = start
    total_days = 0
    decisions: list[dict] = []
    equity_curve: list[dict] = []
    crash_buys: list[dict] = []

    while current_date <= end:
        date_str = current_date.strftime("%Y-%m-%d")
        month_str = current_date.strftime("%Y-%m")
        week_str = current_date.strftime("%Y-W%W")

        # Skip if no price data for this day
        if date_str not in btc_date_idx:
            current_date += timedelta(days=1)
            continue

        idx = btc_date_idx[date_str]
        btc_price = btc_closes[idx]
        btc_rsi = btc_rsi_vals[idx]
        btc_ma200 = btc_sma200[idx]
        eth_price = eth_prices.get(date_str, 0)

        # Reset period counters
        reset_period_counters(state, month_str, week_str)

        # Market regime
        regime = classify_regime(btc_price, btc_ma200, cfg)

        # MVRV
        mvrv = mvrv_data.get(date_str) if use_mvrv else None

        # Daily highs for crash anchor
        highs_for_anchor = btc_closes[max(0, idx - cfg.crash_anchor_long_days):idx + 1]

        # Crash anchor
        crash_anchor = compute_crash_anchor(
            highs_for_anchor,
            cfg.crash_lookback_days,
            cfg.crash_anchor_long_days,
        )

        # ── 1. Regular DCA buy ─────────────────────────────────────────
        now_ts = current_date.timestamp()
        amount, reason, mvrv_mult = compute_daily_amount(
            btc_rsi, cfg, mvrv=mvrv, state=state, now_ts=now_ts,
        )

        # Build decision record
        bracket = classify_rsi(btc_rsi, cfg)
        dec = DCADecision(
            date=date_str,
            rsi=btc_rsi,
            bracket=bracket.value,
            mvrv=mvrv,
            mvrv_mult=mvrv_mult,
            regime=regime.value,
            base_amount=cfg.base_daily_amount * cfg.rsi_multipliers.get(bracket.value, 0),
            mvrv_amount=amount,
            capped_amount=amount,
            reason=reason,
            monthly_spent=state.monthly_spent,
            weekly_spent=state.weekly_spent,
            monthly_cap=cfg.monthly_cap,
            weekly_cap=cfg.weekly_cap,
            skipped=amount <= 0,
        )

        if amount > 0 and remaining_dca_budget(state, cfg) > 0:
            actual = min(amount, remaining_dca_budget(state, cfg))
            alloc = split_allocation(actual, cfg, regime=regime)

            btc_buy = alloc.get("BTC-USD", 0)
            eth_buy = alloc.get("ETH-USD", 0)

            if btc_price > 0 and btc_buy > 0:
                state.total_btc_bought += btc_buy / btc_price
            if eth_price > 0 and eth_buy > 0:
                state.total_eth_bought += eth_buy / eth_price

            state.total_spent_dca += actual
            state.monthly_spent += actual
            state.weekly_spent += actual
            state.buy_count += 1
            state.last_buy_date = date_str
            state.last_buy_rsi = btc_rsi
            state.last_buy_bracket = bracket.value

            if amount >= cfg.boost_threshold:
                state.last_boost_ts = now_ts

        decisions.append(dec.to_dict())

        # ── 2. Crash reserve check ─────────────────────────────────────
        crash_triggers = check_crash_triggers(btc_price, crash_anchor, state, cfg)
        for drop_pct, crash_amount in crash_triggers:
            level_name = f"LEVEL_{int(drop_pct * 100)}"
            actual_crash = min(crash_amount, remaining_crash_budget(state, cfg))
            if actual_crash > 0 and btc_price > 0:
                state.total_btc_bought += actual_crash / btc_price
                state.total_spent_crash += actual_crash
                state.crash_buy_count += 1
                state.crash_levels_triggered.append(level_name)
                crash_buys.append({
                    "date": date_str,
                    "level": level_name,
                    "amount": actual_crash,
                    "price": btc_price,
                    "btc_size": actual_crash / btc_price,
                    "drop_pct": drop_pct,
                })

        # ── 3. Crash recovery reset ────────────────────────────────────
        reset_crash_levels_if_recovered(btc_price, crash_anchor, state, cfg)

        # ── 4. Track equity ────────────────────────────────────────────
        portfolio_value = (
            state.total_btc_bought * btc_price
            + state.total_eth_bought * eth_price
        )
        total_invested = state.total_spent_dca + state.total_spent_crash
        equity_curve.append({
            "date": date_str,
            "invested": total_invested,
            "value": portfolio_value,
            "pnl": portfolio_value - total_invested,
            "pnl_pct": ((portfolio_value / total_invested - 1) * 100) if total_invested > 0 else 0,
            "btc_price": btc_price,
            "btc_held": state.total_btc_bought,
            "eth_held": state.total_eth_bought,
            "regime": regime.value,
            "mvrv": mvrv,
        })

        total_days += 1
        state.total_days_active = total_days
        current_date += timedelta(days=1)

    # ── Final metrics ──────────────────────────────────────────────────────
    if not equity_curve:
        logger.error("No equity data — backtest produced no results")
        return {}

    final = equity_curve[-1]
    total_invested = final["invested"]
    final_value = final["value"]
    total_return = ((final_value / total_invested - 1) * 100) if total_invested > 0 else 0

    # Max drawdown on portfolio value
    peak = 0.0
    max_dd = 0.0
    for point in equity_curve:
        val = point["value"]
        if val > peak:
            peak = val
        if peak > 0:
            dd = (val - peak) / peak
            max_dd = min(max_dd, dd)

    # MVRV boost stats
    boosted_days = sum(1 for d in decisions if d.get("mvrv_mult", 1.0) > 1.0)
    capped_days = sum(1 for d in decisions if d.get("cap_limited", False))
    skip_days = sum(1 for d in decisions if d.get("skipped", False))

    # Regime distribution
    regime_counts = {}
    for d in decisions:
        r = d.get("regime", "NORMAL")
        regime_counts[r] = regime_counts.get(r, 0) + 1

    years = total_days / 365.25 if total_days > 0 else 1
    monthly_avg = total_invested / (years * 12) if years > 0 else 0

    results = {
        "period": f"{start.date()} → {end.date()}",
        "total_days": total_days,
        "total_invested": round(total_invested, 2),
        "final_value": round(final_value, 2),
        "pnl": round(final_value - total_invested, 2),
        "total_return_pct": round(total_return, 2),
        "max_drawdown_pct": round(max_dd * 100, 2),
        "btc_accumulated": round(state.total_btc_bought, 8),
        "eth_accumulated": round(state.total_eth_bought, 8),
        "avg_btc_cost": round(state.total_spent_dca / state.total_btc_bought, 2) if state.total_btc_bought > 0 else 0,
        "buy_count": state.buy_count,
        "crash_buy_count": state.crash_buy_count,
        "crash_total_spent": round(state.total_spent_crash, 2),
        "skip_days": skip_days,
        "boosted_days": boosted_days,
        "capped_days": capped_days,
        "monthly_avg_spend": round(monthly_avg, 2),
        "regime_distribution": regime_counts,
        "decisions": decisions,
        "equity_curve": equity_curve,
        "crash_buys": crash_buys,
    }

    return results


# ── Report ─────────────────────────────────────────────────────────────────────


def print_report(results: dict) -> None:
    """Print backtest results in a readable format."""
    if not results:
        print("No results to display.")
        return

    print("\n" + "=" * 70)
    print("  DCA RSI v2 — BACKTEST REPORT")
    print("=" * 70)
    print(f"  Period          : {results['period']}")
    print(f"  Trading days    : {results['total_days']}")
    print(f"  Buy days        : {results['buy_count']} ({results['buy_count']/max(results['total_days'],1)*100:.0f}%)")
    print(f"  Skip days       : {results['skip_days']} (RSI overbought)")
    print()

    print("  PERFORMANCE")
    print(f"  Total invested  : ${results['total_invested']:,.2f}")
    print(f"  Final value     : ${results['final_value']:,.2f}")
    pnl = results['pnl']
    pnl_emoji = "✅" if pnl >= 0 else "❌"
    print(f"  PnL             : {pnl_emoji} ${pnl:+,.2f} ({results['total_return_pct']:+.1f}%)")
    print(f"  Max drawdown    : {results['max_drawdown_pct']:.1f}%")
    print(f"  Monthly avg     : ${results['monthly_avg_spend']:,.0f}/month")
    print()

    print("  ACCUMULATION")
    print(f"  BTC accumulated : {results['btc_accumulated']:.8f} BTC")
    print(f"  ETH accumulated : {results['eth_accumulated']:.8f} ETH")
    if results.get('avg_btc_cost', 0) > 0:
        print(f"  Avg BTC cost    : ${results['avg_btc_cost']:,.0f}")
    print()

    print("  CRASH RESERVE")
    print(f"  Crash buys      : {results['crash_buy_count']}")
    print(f"  Crash spent     : ${results['crash_total_spent']:,.2f}")
    if results.get('crash_buys'):
        for cb in results['crash_buys']:
            print(f"    {cb['date']}: {cb['level']} — ${cb['amount']:,.0f} @ ${cb['price']:,.0f} ({cb['btc_size']:.8f} BTC)")
    print()

    print("  v2 FEATURES")
    print(f"  MVRV boosted    : {results['boosted_days']} days")
    print(f"  Cap limited     : {results['capped_days']} days")
    regimes = results.get('regime_distribution', {})
    for r, count in sorted(regimes.items()):
        pct = count / max(results['total_days'], 1) * 100
        print(f"  Regime {r:15s}: {count:4d} days ({pct:.0f}%)")
    print("=" * 70)


# ── Comparison: v2 vs simple DCA ───────────────────────────────────────────────


def run_comparison(
    start: datetime,
    end: datetime,
) -> None:
    """Run v2 vs simple DCA comparison."""
    print("\n🔬 Running DCA v2 vs Simple DCA comparison...\n")

    # v2: full strategy
    cfg_v2 = DCAConfig()
    r_v2 = run_backtest(start, end, cfg_v2, use_mvrv=True)

    # Simple: flat $30/day, no RSI, no MVRV, no regime
    cfg_simple = DCAConfig(
        base_daily_amount=30.0,
        max_daily_buy=30.0,
        rsi_multipliers={
            "OVERBOUGHT": 1.0,  # buy even when overbought
            "WARM": 1.0,
            "NEUTRAL": 1.0,
            "OVERSOLD": 1.0,
        },
        mvrv_enabled=False,
        regime_filter_enabled=False,
        monthly_cap=999999,
        weekly_cap=999999,
        crash_levels=[],  # no crash reserve
        active_budget=999999,
        crash_reserve=0,
        total_capital=999999,
    )
    r_simple = run_backtest(start, end, cfg_simple, use_mvrv=False)

    # v2 no MVRV: RSI + regime but no MVRV
    cfg_no_mvrv = DCAConfig(mvrv_enabled=False)
    r_no_mvrv = run_backtest(start, end, cfg_no_mvrv, use_mvrv=False)

    print("\n" + "=" * 70)
    print("  COMPARISON: v2 vs Simple DCA vs v2-noMVRV")
    print("=" * 70)

    headers = ["Metric", "Simple DCA", "v2 (no MVRV)", "v2 (full)"]
    rows = [
        ["Total invested", f"${r_simple.get('total_invested', 0):,.0f}",
         f"${r_no_mvrv.get('total_invested', 0):,.0f}",
         f"${r_v2.get('total_invested', 0):,.0f}"],
        ["Final value", f"${r_simple.get('final_value', 0):,.0f}",
         f"${r_no_mvrv.get('final_value', 0):,.0f}",
         f"${r_v2.get('final_value', 0):,.0f}"],
        ["PnL", f"${r_simple.get('pnl', 0):+,.0f}",
         f"${r_no_mvrv.get('pnl', 0):+,.0f}",
         f"${r_v2.get('pnl', 0):+,.0f}"],
        ["Return %", f"{r_simple.get('total_return_pct', 0):+.1f}%",
         f"{r_no_mvrv.get('total_return_pct', 0):+.1f}%",
         f"{r_v2.get('total_return_pct', 0):+.1f}%"],
        ["Max DD", f"{r_simple.get('max_drawdown_pct', 0):.1f}%",
         f"{r_no_mvrv.get('max_drawdown_pct', 0):.1f}%",
         f"{r_v2.get('max_drawdown_pct', 0):.1f}%"],
        ["BTC accumulated", f"{r_simple.get('btc_accumulated', 0):.6f}",
         f"{r_no_mvrv.get('btc_accumulated', 0):.6f}",
         f"{r_v2.get('btc_accumulated', 0):.6f}"],
        ["Buy days", f"{r_simple.get('buy_count', 0)}",
         f"{r_no_mvrv.get('buy_count', 0)}",
         f"{r_v2.get('buy_count', 0)}"],
        ["Monthly avg", f"${r_simple.get('monthly_avg_spend', 0):,.0f}",
         f"${r_no_mvrv.get('monthly_avg_spend', 0):,.0f}",
         f"${r_v2.get('monthly_avg_spend', 0):,.0f}"],
    ]

    # Print table
    col_widths = [max(len(str(row[i])) for row in [headers] + rows) for i in range(4)]
    fmt = "  {:<{w0}}  {:<{w1}}  {:<{w2}}  {:<{w3}}"
    print(fmt.format(*headers, w0=col_widths[0], w1=col_widths[1], w2=col_widths[2], w3=col_widths[3]))
    print("  " + "─" * sum(col_widths) + "──────")
    for row in rows:
        print(fmt.format(*row, w0=col_widths[0], w1=col_widths[1], w2=col_widths[2], w3=col_widths[3]))
    print("=" * 70)


# ── CLI ────────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(description="DCA RSI v2 backtest")
    parser.add_argument("--start", default="2022-01-01", help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", default="2025-03-01", help="End date (YYYY-MM-DD)")
    parser.add_argument("--no-mvrv", action="store_true", help="Disable MVRV")
    parser.add_argument("--compare", action="store_true", help="Run v2 vs simple DCA comparison")
    args = parser.parse_args()

    start = datetime.strptime(args.start, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    end = datetime.strptime(args.end, "%Y-%m-%d").replace(tzinfo=timezone.utc)

    if args.compare:
        run_comparison(start, end)
    else:
        cfg = DCAConfig()
        results = run_backtest(start, end, cfg, use_mvrv=not args.no_mvrv)
        print_report(results)


if __name__ == "__main__":
    main()
