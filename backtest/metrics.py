"""
Calcul des métriques de performance du backtest.
"""

from __future__ import annotations

import math
from collections import defaultdict
from datetime import datetime, timezone

from backtest.simulator import BacktestResult, Trade

# H4 = 6 barres / jour  →  365.25 jours / an
_BARS_PER_YEAR = 6 * 365.25


def compute_metrics(result: BacktestResult) -> dict:
    """Renvoie un dictionnaire complet de KPIs."""
    trades = result.trades
    eq = result.equity_curve
    init = result.initial_balance
    final = result.final_equity
    years = max((result.end_date - result.start_date).days / 365.25, 0.01)

    # ── Rendement global ──
    total_return = (final - init) / init
    cagr = (final / init) ** (1 / years) - 1 if final > 0 else 0

    # ── Drawdown ──
    peak = init
    max_dd = 0.0
    dd_curve: list[float] = []
    for pt in eq:
        peak = max(peak, pt.equity)
        dd = (pt.equity - peak) / peak if peak else 0
        dd_curve.append(dd)
        max_dd = min(max_dd, dd)

    # ── Sharpe (sur rendements par barre H4) ──
    returns: list[float] = []
    for i in range(1, len(eq)):
        prev = eq[i - 1].equity
        if prev > 0:
            returns.append((eq[i].equity - prev) / prev)
    sharpe = _sharpe(returns)
    sortino = _sortino(returns)

    # ── Stats sur les trades ──
    n = len(trades)
    if n:
        wins = [t for t in trades if t.pnl_usd > 0]
        losses = [t for t in trades if t.pnl_usd <= 0]
        win_rate = len(wins) / n
        gross_profit = sum(t.pnl_usd for t in wins) or 0
        gross_loss = abs(sum(t.pnl_usd for t in losses)) or 1e-9
        profit_factor = gross_profit / gross_loss
        avg_pnl = sum(t.pnl_usd for t in trades) / n
        avg_pnl_pct = sum(t.pnl_pct for t in trades) / n
        best = max(trades, key=lambda t: t.pnl_usd)
        worst = min(trades, key=lambda t: t.pnl_usd)
    else:
        win_rate = profit_factor = avg_pnl = avg_pnl_pct = 0
        best = worst = None

    # ── Par stratégie ──
    by_strat = _group_trades(trades, lambda t: t.strategy.value)

    # ── Par paire ──
    by_pair = _group_trades(trades, lambda t: t.symbol)

    # ── Par motif de sortie ──
    by_exit = _group_trades(trades, lambda t: t.exit_reason)

    # ── Rendements mensuels ──
    monthly = _monthly_returns(eq, init)

    return {
        "total_return": total_return,
        "cagr": cagr,
        "max_drawdown": max_dd,
        "sharpe": sharpe,
        "sortino": sortino,
        "n_trades": n,
        "win_rate": win_rate,
        "profit_factor": profit_factor,
        "avg_pnl_usd": avg_pnl,
        "avg_pnl_pct": avg_pnl_pct,
        "best_trade": best,
        "worst_trade": worst,
        "by_strategy": by_strat,
        "by_pair": by_pair,
        "by_exit": by_exit,
        "monthly_returns": monthly,
        "dd_curve": dd_curve,
        "years": years,
        "final_equity": final,
    }


# ── Helpers ────────────────────────────────────────────────────────────────────


def _sharpe(returns: list[float]) -> float:
    if len(returns) < 2:
        return 0.0
    mu = sum(returns) / len(returns)
    var = sum((r - mu) ** 2 for r in returns) / (len(returns) - 1)
    std = math.sqrt(var) if var > 0 else 1e-9
    return (mu / std) * math.sqrt(_BARS_PER_YEAR)


def _sortino(returns: list[float]) -> float:
    if len(returns) < 2:
        return 0.0
    mu = sum(returns) / len(returns)
    neg = [r for r in returns if r < 0]
    if not neg:
        return 99.0  # pas de perte
    down_var = sum(r ** 2 for r in neg) / len(neg)
    down_std = math.sqrt(down_var) if down_var > 0 else 1e-9
    return (mu / down_std) * math.sqrt(_BARS_PER_YEAR)


def _group_trades(trades: list[Trade], key_fn) -> dict[str, dict]:
    groups: dict[str, list[Trade]] = defaultdict(list)
    for t in trades:
        groups[key_fn(t)].append(t)
    out: dict[str, dict] = {}
    for k, tlist in sorted(groups.items()):
        n = len(tlist)
        wins = sum(1 for t in tlist if t.pnl_usd > 0)
        pnl = sum(t.pnl_usd for t in tlist)
        gp = sum(t.pnl_usd for t in tlist if t.pnl_usd > 0) or 0
        gl = abs(sum(t.pnl_usd for t in tlist if t.pnl_usd <= 0)) or 1e-9
        out[k] = {
            "n": n,
            "wins": wins,
            "wr": wins / n if n else 0,
            "pnl": pnl,
            "pf": gp / gl,
            "avg_pct": sum(t.pnl_pct for t in tlist) / n if n else 0,
        }
    return out


def _monthly_returns(eq: list, init: float) -> list[tuple[str, float]]:
    """Retourne [(YYYY-MM, return%), ...]."""
    if not eq:
        return []

    # Grouper les equity par mois → prendre la dernière valeur du mois
    by_month: dict[str, float] = {}
    for pt in eq:
        dt = datetime.fromtimestamp(pt.timestamp / 1000, tz=timezone.utc)
        key = f"{dt.year}-{dt.month:02d}"
        by_month[key] = pt.equity

    months = sorted(by_month.keys())
    result: list[tuple[str, float]] = []
    prev_eq = init
    for m in months:
        cur = by_month[m]
        ret = (cur - prev_eq) / prev_eq if prev_eq else 0
        result.append((m, ret))
        prev_eq = cur
    return result
