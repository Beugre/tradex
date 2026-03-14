#!/usr/bin/env python3
"""
Backtest stratégie "Achat au listing" — vraie logique Listing Event Binance.

Règles:
- Univers: paires Binance Spot quote USDC avec onboardDate réel (`exchangeInfo`).
- Entrée LONG sur la première bougie H4 disponible après la date de listing.
- OCO initial: SL = -10%, TP = +8% (référencés au prix d'entrée).
- Trigger re-arm: si le prix atteint 98% du niveau TP initial (price >= 0.98 * TP1).
- Nouvel OCO (activé sur la bougie suivante):
    - SL2 = 6% sous le TP initial (SL2 = TP1 * 0.94)
    - TP2 = +10% depuis le prix d'entrée (TP2 = entry * 1.10)

Hypothèses d'exécution:
- Déclenchements évalués sur OHLC de chaque bougie.
- Si SL et TP sont touchés dans la même bougie, priorité au SL (prudence).
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import logging
import time as _time
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path

import httpx

logging.basicConfig(level=logging.WARNING, format="[%(asctime)s] %(message)s")
logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).parent / "data"
LISTING_CACHE_DIR = DATA_DIR / "listing_event"
LISTING_META_CACHE = LISTING_CACHE_DIR / "listing_meta.csv"
BINANCE_URLS = [
    "https://api.binance.com",
    "https://api1.binance.com",
    "https://api2.binance.com",
    "https://api3.binance.com",
]
KLINES_LIMIT = 1000


@dataclass
class ListingSymbol:
    symbol: str
    onboard_date_ms: int


@dataclass
class Candle:
    timestamp: int
    open: float
    high: float
    low: float
    close: float
    volume: float


@dataclass
class ListingTrade:
    symbol: str
    entry_ts: int
    entry_price: float
    exit_ts: int
    exit_price: float
    exit_reason: str
    pnl_pct: float
    oco_rearmed: bool
    onboard_date_ms: int


@dataclass
class PortfolioPosition:
    symbol: str
    entry_ts: int
    exit_ts: int
    entry_price: float
    exit_price: float
    allocated_capital: float
    pnl_pct: float
    pnl_usd: float
    exit_reason: str
    oco_rearmed: bool


@dataclass
class PortfolioResult:
    period_label: str
    lookback_months: int
    initial_capital: float
    final_capital: float
    return_pct: float
    n_listings: int
    n_trades_taken: int
    n_trades_skipped: int
    n_wins: int
    win_rate: float
    max_drawdown_pct: float
    best_trade_pct: float
    worst_trade_pct: float
    avg_trade_pnl_pct: float
    profit_factor: float
    n_not_filled: int = 0
    positions: list[PortfolioPosition] = field(default_factory=list)
    equity_snapshots: list[tuple[int, float]] = field(default_factory=list)


@dataclass
class Summary:
    n_trades: int
    win_rate: float
    avg_pnl_pct: float
    total_pnl_pct: float
    best_pct: float
    worst_pct: float


def load_candles_from_csv(path: Path) -> list[Candle]:
    candles: list[Candle] = []
    with open(path) as f:
        for row in csv.DictReader(f):
            candles.append(
                Candle(
                    timestamp=int(row["timestamp"]),
                    open=float(row["open"]),
                    high=float(row["high"]),
                    low=float(row["low"]),
                    close=float(row["close"]),
                    volume=float(row["volume"]),
                )
            )
    return candles


def save_candles_to_csv(path: Path, candles: list[Candle]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["timestamp", "open", "high", "low", "close", "volume"])
        for c in candles:
            w.writerow([c.timestamp, c.open, c.high, c.low, c.close, c.volume])


def fetch_usdc_listing_symbols() -> list[ListingSymbol]:
    payload = _http_get_json("/api/v3/exchangeInfo", params=None)
    out: list[ListingSymbol] = []
    for item in payload.get("symbols", []):
        if item.get("status") != "TRADING":
            continue
        if item.get("quoteAsset") != "USDC":
            continue
        onboard_ms = int(item.get("onboardDate") or 0)
        out.append(ListingSymbol(symbol=str(item["symbol"]).upper(), onboard_date_ms=onboard_ms))
    out.sort(key=lambda x: x.onboard_date_ms)
    return out


def _http_get_json(path: str, params: dict | None = None):
    last_exc: Exception | None = None
    for base in BINANCE_URLS:
        try:
            with httpx.Client(timeout=30) as client:
                resp = client.get(f"{base}{path}", params=params)
                resp.raise_for_status()
                return resp.json()
        except Exception as exc:  # noqa: BLE001
            last_exc = exc
            logger.warning("API %s KO: %s", base, exc)
    raise RuntimeError(f"Impossible de récupérer {path}: {last_exc}")


def load_listing_meta_cache() -> dict[str, int]:
    if not LISTING_META_CACHE.exists():
        return {}
    out: dict[str, int] = {}
    with open(LISTING_META_CACHE) as f:
        for row in csv.DictReader(f):
            symbol = str(row.get("symbol", "")).upper().strip()
            listing_ms = int(float(row.get("listing_ms", 0) or 0))
            if symbol and listing_ms > 0:
                out[symbol] = listing_ms
    return out


def save_listing_meta_cache(data: dict[str, int]) -> None:
    LISTING_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    with open(LISTING_META_CACHE, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["symbol", "listing_ms"])
        for symbol in sorted(data.keys()):
            w.writerow([symbol, data[symbol]])


def fetch_first_kline_timestamp(symbol: str, interval: str = "1d") -> int:
    rows = _http_get_json(
        "/api/v3/klines",
        params={
            "symbol": symbol,
            "interval": interval,
            "startTime": 0,
            "limit": 1,
        },
    )
    if not rows:
        return 0
    return int(rows[0][0])


def resolve_listing_timestamp(symbol: str, onboard_date_ms: int, meta_cache: dict[str, int]) -> int:
    if onboard_date_ms > 0:
        return onboard_date_ms
    if symbol in meta_cache:
        return meta_cache[symbol]
    listing_ms = fetch_first_kline_timestamp(symbol, interval="1d")
    if listing_ms > 0:
        meta_cache[symbol] = listing_ms
    return listing_ms


def download_listing_candles(symbol: str, onboard_date_ms: int, interval: str, horizon_days: int, use_cache: bool) -> list[Candle]:
    end_dt = datetime.now(timezone.utc)
    end_ms = int(end_dt.timestamp() * 1000)
    start_ms = onboard_date_ms
    if horizon_days > 0:
        end_ms = min(end_ms, start_ms + horizon_days * 24 * 60 * 60 * 1000)

    cache = LISTING_CACHE_DIR / f"{symbol}_{interval}_{start_ms}_{end_ms}.csv"
    if use_cache and cache.exists():
        return load_candles_from_csv(cache)

    candles: list[Candle] = []
    cursor = start_ms
    while cursor < end_ms:
        rows = _http_get_json(
            "/api/v3/klines",
            params={
                "symbol": symbol,
                "interval": interval,
                "startTime": cursor,
                "endTime": end_ms,
                "limit": KLINES_LIMIT,
            },
        )
        if not rows:
            break
        for k in rows:
            candles.append(
                Candle(
                    timestamp=int(k[0]),
                    open=float(k[1]),
                    high=float(k[2]),
                    low=float(k[3]),
                    close=float(k[4]),
                    volume=float(k[5]),
                )
            )
        cursor = int(rows[-1][0]) + 1
        _time.sleep(0.10)

    if candles:
        save_candles_to_csv(cache, candles)
    return candles


def download_listing_1m_candles(
    symbol: str, listing_ms: int, minutes: int = 60, use_cache: bool = True,
) -> list[Candle]:
    """Download first N minutes of 1m klines for a listing event (cached).

    Uses startTime=0 to get the actual first available 1m data,
    which corresponds to the real listing moment (not the onboard midnight).
    """
    cache_path = LISTING_CACHE_DIR / f"{symbol}_1m_first{minutes}.csv"
    if use_cache and cache_path.exists():
        return load_candles_from_csv(cache_path)

    candles: list[Candle] = []
    rows = _http_get_json(
        "/api/v3/klines",
        params={
            "symbol": symbol,
            "interval": "1m",
            "startTime": 0,
            "limit": min(minutes, 1000),
        },
    )
    if not rows:
        return []
    for k in rows:
        candles.append(
            Candle(
                timestamp=int(k[0]),
                open=float(k[1]),
                high=float(k[2]),
                low=float(k[3]),
                close=float(k[4]),
                volume=float(k[5]),
            )
        )
    if candles:
        save_candles_to_csv(cache_path, candles)
    _time.sleep(0.05)
    return candles


def check_listing_entry(
    candles_1m: list[Candle],
    entry_delay_minutes: int = 1,
    momentum_pct: float = 0.0,
    momentum_window: int = 1,
) -> tuple[float | None, str]:
    """Analyse les premières minutes du listing.

    Returns:
        (entry_price, skip_reason):
        - entry_price = prix d'entrée brut (avant slippage/spread) si filtres OK,
          None si le listing est rejeté.
        - skip_reason = raison du skip (\"no_momentum\", \"no_1m_data\", \"\").
    """
    if not candles_1m:
        return None, "no_1m_data"

    listing_price = candles_1m[0].open

    # ── filtre momentum : le prix doit monter de +X% dans les N premières minutes ──
    if momentum_pct > 0:
        window = max(momentum_window, 1)
        target = listing_price * (1 + momentum_pct)
        passed = False
        for i in range(min(window, len(candles_1m))):
            if candles_1m[i].high >= target:
                passed = True
                break
        if not passed:
            return None, "no_momentum"

    # ── prix d'entrée retardé (close de la bougie 1m à t + delay) ──
    if entry_delay_minutes > 0 and len(candles_1m) >= entry_delay_minutes:
        entry_price = candles_1m[entry_delay_minutes - 1].close
    else:
        entry_price = listing_price

    return entry_price, ""


def run_listing_trade(
    symbol: str,
    onboard_date_ms: int,
    candles: list[Candle],
    sl_init_pct: float,
    tp_init_pct: float,
    tp_near_ratio: float,
    sl2_tp1_mult: float,
    tp2_tp1_mult: float,
    # ── frictions de marché ──
    slippage_entry: float = 0.0,
    slippage_stop: float = 0.0,
    slippage_tp: float = 0.0,
    spread: float = 0.0,
    fees_roundtrip: float = 0.0,
    min_volume_1h: float = 0.0,
    # ── mode pullback ──
    pullback_pct: float = 0.0,
    pullback_window: int = 6,
    # ── entrée retardée (1m data) ──
    override_entry_price: float = 0.0,
    # ── spread listing additionnel ──
    spread_listing: float = 0.0,
) -> ListingTrade | None:
    if len(candles) < 2:
        return None

    # ── filtre de liquidité : volume de la 1ère bougie (proxy volume 1h) ──
    if min_volume_1h > 0:
        vol_usd = candles[0].volume * candles[0].close
        if vol_usd < min_volume_1h:
            return None

    # ── entrée : standard ou pullback ──
    if pullback_pct > 0:
        # Chercher un pullback de -pullback_pct% sous l'open de listing
        reference = candles[0].open
        target_entry = reference * (1 - pullback_pct)
        entry = 0.0
        entry_ts = 0
        entry_idx = -1
        for idx in range(1, min(len(candles), 1 + pullback_window)):
            c = candles[idx]
            if c.low <= target_entry:
                # Limit order rempli au prix target (pas de slippage, juste spread)
                entry = target_entry * (1 + spread / 2)
                entry_ts = c.timestamp
                entry_idx = idx
                break
        if entry_idx < 0:
            return None  # pas de pullback dans la fenêtre → skip
    else:
        # Entrée standard : open de la 1ère bougie post-listing (+ frictions)
        base_price = override_entry_price if override_entry_price > 0 else candles[0].open
        total_spread = spread + spread_listing  # spread normal + spread listing
        entry = base_price * (1 + slippage_entry) * (1 + total_spread / 2)
        entry_ts = candles[0].timestamp
        entry_idx = 0

    # ── niveaux OCO initiaux ──
    sl = entry * (1 - sl_init_pct)
    tp_initial = entry * (1 + tp_init_pct)
    tp = tp_initial

    trigger_price = tp_initial * tp_near_ratio
    pending_rearm = False
    oco_rearmed = False

    for idx in range(entry_idx + 1, len(candles)):
        c = candles[idx]

        # Active le nouvel OCO en début de bougie suivante
        if pending_rearm:
            sl = tp_initial * sl2_tp1_mult
            tp = tp_initial * tp2_tp1_mult
            pending_rearm = False
            oco_rearmed = True

        sl_hit = c.low <= sl
        tp_hit = c.high >= tp

        if sl_hit and tp_hit:
            exit_price = sl * (1 - slippage_stop) * (1 - spread / 2)
            reason = "SL_SAME_BAR"
            pnl = (exit_price / entry) - 1 - fees_roundtrip
            return ListingTrade(symbol, entry_ts, entry, c.timestamp, exit_price, reason, pnl, oco_rearmed, onboard_date_ms)

        if sl_hit:
            exit_price = sl * (1 - slippage_stop) * (1 - spread / 2)
            reason = "SL"
            pnl = (exit_price / entry) - 1 - fees_roundtrip
            return ListingTrade(symbol, entry_ts, entry, c.timestamp, exit_price, reason, pnl, oco_rearmed, onboard_date_ms)

        if tp_hit:
            exit_price = tp * (1 - slippage_tp) * (1 - spread / 2)
            reason = "TP"
            pnl = (exit_price / entry) - 1 - fees_roundtrip
            return ListingTrade(symbol, entry_ts, entry, c.timestamp, exit_price, reason, pnl, oco_rearmed, onboard_date_ms)

        # Déclenchement re-arm OCO quand prix atteint 98% du TP initial
        if not oco_rearmed and not pending_rearm and c.high >= trigger_price:
            pending_rearm = True

    last = candles[-1]
    exit_price = last.close * (1 - spread / 2)
    pnl = (exit_price / entry) - 1 - fees_roundtrip
    return ListingTrade(symbol, entry_ts, entry, last.timestamp, exit_price, "EOD", pnl, oco_rearmed, onboard_date_ms)


# ─────────────────────────────────────────────────────────────────────
#  Portfolio simulation – vrai capital, slots, PnL composé
# ─────────────────────────────────────────────────────────────────────

def _equity(cash: float, open_positions: list[dict]) -> float:
    """Total equity = cash + capital réservé dans les positions ouvertes (at cost)."""
    return cash + sum(p["allocated"] for p in open_positions)


def run_portfolio_simulation(
    all_trades: list[ListingTrade],
    initial_capital: float,
    max_slots: int,
    lookback_start_ms: int,
    lookback_end_ms: int,
    period_label: str,
    lookback_months: int,
    fill_rate: float = 1.0,
    max_alloc_usd: float = 0.0,
) -> PortfolioResult:
    """Simulate a real portfolio with slot-based capital allocation.

    - Only listings whose *onboard_date_ms* falls inside
      [lookback_start_ms, lookback_end_ms] are considered.
    - Each position gets ``equity / max_slots`` (capped at available cash).
    - Capital is returned (with P&L) when a position closes.
    - Equity curve tracked at every open/close event.
    - fill_rate: probability of fill (0-1). Deterministic via symbol hash.
    - max_alloc_usd: hard cap in USD per position (0 = no cap).
    """
    eligible = [
        t for t in all_trades
        if lookback_start_ms <= t.onboard_date_ms <= lookback_end_ms
    ]
    eligible.sort(key=lambda t: t.entry_ts)

    cash = initial_capital
    open_slots: list[dict] = []
    closed_positions: list[PortfolioPosition] = []
    equity_snapshots: list[tuple[int, float]] = [(lookback_start_ms, initial_capital)]
    skipped = 0
    not_filled = 0

    for trade in eligible:
        # ── simulation non-fill (déterministe par symbole) ──
        if fill_rate < 1.0:
            h = int(hashlib.md5(trade.symbol.encode()).hexdigest(), 16)
            if (h % 10000) / 10000.0 >= fill_rate:
                not_filled += 1
                continue

        # ── séparer positions expirées vs encore ouvertes ──
        closing: list[dict] = []
        still_open: list[dict] = []
        for slot in open_slots:
            if slot["exit_ts"] <= trade.entry_ts:
                closing.append(slot)
            else:
                still_open.append(slot)

        # ── fermer les positions expirées (still_open est déjà complet) ──
        closing_sorted = sorted(closing, key=lambda s: s["exit_ts"])
        for i, slot in enumerate(closing_sorted):
            pnl_usd = slot["allocated"] * slot["pnl_pct"]
            cash += slot["allocated"] + pnl_usd
            closed_positions.append(PortfolioPosition(
                symbol=slot["symbol"],
                entry_ts=slot["entry_ts"],
                exit_ts=slot["exit_ts"],
                entry_price=slot["entry_price"],
                exit_price=slot["exit_price"],
                allocated_capital=slot["allocated"],
                pnl_pct=slot["pnl_pct"],
                pnl_usd=pnl_usd,
                exit_reason=slot["exit_reason"],
                oco_rearmed=slot["oco_rearmed"],
            ))
            # capital encore bloqué dans les positions closing pas encore traitées
            remaining_alloc = sum(s["allocated"] for s in closing_sorted[i + 1:])
            equity_snapshots.append(
                (slot["exit_ts"], _equity(cash, still_open) + remaining_alloc)
            )
        open_slots = still_open

        # ── vérifier la dispo d'un slot ──
        if len(open_slots) >= max_slots:
            skipped += 1
            continue

        # ── allocation : equity / max_slots, plafonnée au cash dispo ──
        eq = _equity(cash, open_slots)
        alloc = min(cash, eq / max_slots)
        if max_alloc_usd > 0:
            alloc = min(alloc, max_alloc_usd)
        if alloc < 1.0:
            skipped += 1
            continue

        cash -= alloc
        open_slots.append({
            "symbol": trade.symbol,
            "entry_ts": trade.entry_ts,
            "exit_ts": trade.exit_ts,
            "entry_price": trade.entry_price,
            "exit_price": trade.exit_price,
            "allocated": alloc,
            "pnl_pct": trade.pnl_pct,
            "exit_reason": trade.exit_reason,
            "oco_rearmed": trade.oco_rearmed,
        })
        equity_snapshots.append((trade.entry_ts, _equity(cash, open_slots)))

    # ── clôturer les positions restantes ──
    remaining_sorted = sorted(open_slots, key=lambda s: s["exit_ts"])
    for i, slot in enumerate(remaining_sorted):
        pnl_usd = slot["allocated"] * slot["pnl_pct"]
        cash += slot["allocated"] + pnl_usd
        closed_positions.append(PortfolioPosition(
            symbol=slot["symbol"],
            entry_ts=slot["entry_ts"],
            exit_ts=slot["exit_ts"],
            entry_price=slot["entry_price"],
            exit_price=slot["exit_price"],
            allocated_capital=slot["allocated"],
            pnl_pct=slot["pnl_pct"],
            pnl_usd=pnl_usd,
            exit_reason=slot["exit_reason"],
            oco_rearmed=slot["oco_rearmed"],
        ))
        remaining_alloc = sum(s["allocated"] for s in remaining_sorted[i + 1:])
        equity_snapshots.append((slot["exit_ts"], cash + remaining_alloc))

    # ── métriques ──
    final_capital = cash
    return_pct = (final_capital / initial_capital) - 1.0 if initial_capital > 0 else 0.0

    pnls = [p.pnl_pct for p in closed_positions]
    wins = [p for p in pnls if p > 0]

    # max drawdown
    peak = initial_capital
    max_dd = 0.0
    for _, eq in equity_snapshots:
        if eq > peak:
            peak = eq
        dd = (peak - eq) / peak if peak > 0 else 0.0
        if dd > max_dd:
            max_dd = dd

    # profit factor
    gross_profit = sum(p.pnl_usd for p in closed_positions if p.pnl_usd > 0)
    gross_loss = abs(sum(p.pnl_usd for p in closed_positions if p.pnl_usd < 0))
    pf = gross_profit / gross_loss if gross_loss > 0 else float("inf")

    return PortfolioResult(
        period_label=period_label,
        lookback_months=lookback_months,
        initial_capital=initial_capital,
        final_capital=final_capital,
        return_pct=return_pct,
        n_listings=len(eligible),
        n_trades_taken=len(closed_positions),
        n_trades_skipped=skipped,
        n_wins=len(wins),
        n_not_filled=not_filled,
        win_rate=len(wins) / len(closed_positions) if closed_positions else 0.0,
        max_drawdown_pct=max_dd,
        best_trade_pct=max(pnls) if pnls else 0.0,
        worst_trade_pct=min(pnls) if pnls else 0.0,
        avg_trade_pnl_pct=sum(pnls) / len(pnls) if pnls else 0.0,
        profit_factor=pf,
        positions=closed_positions,
        equity_snapshots=equity_snapshots,
    )


def print_portfolio_results(results: list[PortfolioResult], args: argparse.Namespace) -> None:
    """Pretty-print portfolio simulation results."""
    print("\n" + "═" * 78)
    print("  BACKTEST PORTFOLIO — Listing OCO (capital réel, slots, PnL composé)")
    print("═" * 78)
    print(f"  Capital initial     : {args.capital:,.2f} USDC")
    print(f"  Max positions       : {args.max_slots} slots ({1/args.max_slots:.0%} par paire)")
    print(f"  Horizon par trade   : {args.horizon_days} jours")
    print(f"  Config OCO          : SL -{args.sl_init:.0%} / TP +{args.tp_init:.0%}")
    print(f"  Re-arm              : trigger {args.tp_near_ratio:.0%} de TP1")
    print(f"  OCO2                : SL2 = {args.sl2_tp1_mult:.2f}×TP1 | TP2 = {args.tp2_tp1_mult:.2f}×TP1")
    print(f"  Frictions           : {_format_friction_summary(args)}")
    print("═" * 78)

    # ── tableau comparatif ──
    header = f"{'Période':<10} {'Listings':>8} {'Trades':>7} {'Skip':>5} {'NF':>4} {'WR':>7} {'PF':>6} {'Avg%':>8} {'Return':>10} {'Final$':>10} {'MaxDD':>8}"
    print(f"\n{header}")
    print("─" * len(header))
    for r in results:
        print(
            f"{r.period_label:<10} {r.n_listings:>8} {r.n_trades_taken:>7} "
            f"{r.n_trades_skipped:>5} {r.n_not_filled:>4} {r.win_rate:>6.1%} {r.profit_factor:>6.2f} "
            f"{r.avg_trade_pnl_pct:>+7.2%} {r.return_pct:>+9.2%} "
            f"{r.final_capital:>9,.2f} {r.max_drawdown_pct:>7.1%}"
        )
    print("─" * len(header))

    # ── détail par période ──
    for r in results:
        print(f"\n{'─' * 78}")
        print(f"  📊 {r.period_label.upper()} — {r.n_listings} listings disponibles")
        print(f"{'─' * 78}")
        print(f"  Trades exécutés  : {r.n_trades_taken} / {r.n_listings}  (skippés: {r.n_trades_skipped})")
        print(f"  Capital          : {r.initial_capital:,.2f} → {r.final_capital:,.2f} USDC  ({r.return_pct:+.2%})")
        print(f"  Win Rate         : {r.win_rate:.1%}  ({r.n_wins}W / {r.n_trades_taken - r.n_wins}L)")
        print(f"  Profit Factor    : {r.profit_factor:.2f}")
        print(f"  Max Drawdown     : {r.max_drawdown_pct:.1%}")
        print(f"  Best / Worst     : {r.best_trade_pct:+.2%} / {r.worst_trade_pct:+.2%}")
        print(f"  Avg PnL / trade  : {r.avg_trade_pnl_pct:+.2%}")

        # top trades
        if r.positions:
            sorted_pos = sorted(r.positions, key=lambda p: p.pnl_usd, reverse=True)
            print(f"\n  Top 5 gains:")
            for p in sorted_pos[:5]:
                dt_in = datetime.fromtimestamp(p.entry_ts / 1000, tz=timezone.utc).strftime("%Y-%m-%d")
                dt_out = datetime.fromtimestamp(p.exit_ts / 1000, tz=timezone.utc).strftime("%Y-%m-%d")
                print(
                    f"    {p.symbol:14s} {dt_in}→{dt_out}  alloc ${p.allocated_capital:7.2f}  "
                    f"PnL {p.pnl_pct:+.2%} = ${p.pnl_usd:+.2f}  [{p.exit_reason}]"
                )
            print(f"  Top 5 pertes:")
            for p in sorted_pos[-5:]:
                dt_in = datetime.fromtimestamp(p.entry_ts / 1000, tz=timezone.utc).strftime("%Y-%m-%d")
                dt_out = datetime.fromtimestamp(p.exit_ts / 1000, tz=timezone.utc).strftime("%Y-%m-%d")
                print(
                    f"    {p.symbol:14s} {dt_in}→{dt_out}  alloc ${p.allocated_capital:7.2f}  "
                    f"PnL {p.pnl_pct:+.2%} = ${p.pnl_usd:+.2f}  [{p.exit_reason}]"
                )


def summarize(trades: list[ListingTrade]) -> Summary:
    if not trades:
        return Summary(0, 0.0, 0.0, 0.0, 0.0, 0.0)

    pnls = [t.pnl_pct for t in trades]
    wins = [p for p in pnls if p > 0]
    return Summary(
        n_trades=len(trades),
        win_rate=len(wins) / len(trades),
        avg_pnl_pct=sum(pnls) / len(pnls),
        total_pnl_pct=sum(pnls),
        best_pct=max(pnls),
        worst_pct=min(pnls),
    )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Backtest listing OCO strategy (real Binance listing events)")
    p.add_argument("--pairs", default="", help="Liste de symboles Binance (ex: BTCUSDC,ETHUSDC)")
    p.add_argument("--max-symbols", type=int, default=0, help="Limiter le nombre de symboles testés")
    p.add_argument("--interval", default="4h", help="Intervalle kline Binance (défaut: 4h)")
    p.add_argument("--horizon-days", type=int, default=120, help="Durée max post-listing à simuler (0 = jusqu'à now)")
    p.add_argument("--no-cache", action="store_true", help="Désactive le cache CSV local")

    p.add_argument("--sl-init", type=float, default=0.10, help="SL initial (ex: 0.10 = 10%%)")
    p.add_argument("--tp-init", type=float, default=0.08, help="TP initial (ex: 0.08 = 8%%)")
    p.add_argument("--tp-near-ratio", type=float, default=0.98, help="Seuil d'activation du re-arm sur trajet vers TP")
    p.add_argument("--sl2-tp1-mult", type=float, default=0.94, help="SL2 exprimé en multiple de TP1 (ex: 0.80)")
    p.add_argument("--tp2-tp1-mult", type=float, default=1.20, help="TP2 exprimé en multiple de TP1 (ex: 1.20)")

    # ── Portfolio mode ──
    p.add_argument("--portfolio", action="store_true", help="Mode portfolio : capital réel, slots, PnL composé")
    p.add_argument("--capital", type=float, default=1000.0, help="Capital initial en USDC (défaut: 1000)")
    p.add_argument("--max-slots", type=int, default=4, help="Nb max de positions simultanées (défaut: 4 → 25%% par paire)")

    # ── Frictions de marché ──
    p.add_argument("--slippage-entry", type=float, default=0.0, help="Slippage à l'entrée (ex: 0.03 = 3%%)")
    p.add_argument("--slippage-stop", type=float, default=0.0, help="Slippage sur le stop (ex: 0.01 = 1%%)")
    p.add_argument("--slippage-tp", type=float, default=0.0, help="Slippage sur le TP (ex: 0.005 = 0.5%%)")
    p.add_argument("--spread", type=float, default=0.0, help="Spread buy/sell (ex: 0.01 = 1%%)")
    p.add_argument("--fees", type=float, default=0.0, help="Frais aller-retour (ex: 0.002 = 0.2%%)")
    p.add_argument("--min-volume", type=float, default=0.0, help="Volume min 1ère bougie en USD (ex: 5000000)")

    # ── Mode pullback ──
    p.add_argument("--pullback", type=float, default=0.0, help="Attendre un pullback de X%% sous le pump (ex: 0.03 = 3%%)")
    p.add_argument("--pullback-window", type=int, default=6, help="Nb de bougies max pour attendre le pullback (défaut: 6)")

    # ── Entrée réaliste & filtre momentum ──
    p.add_argument("--entry-delay", type=int, default=0, help="Minutes après listing pour entrer (0 = H4 open, 1 = close 1m après)")
    p.add_argument("--momentum", type=float, default=0.0, help="Filtre: hausse min dans les 1ères minutes (ex: 0.025 = +2.5%%%%)")
    p.add_argument("--momentum-window", type=int, default=1, help="Fenêtre momentum en minutes (défaut: 1)")

    # ── Réalisme avancé ──
    p.add_argument("--fill-rate", type=float, default=1.0, help="Probabilité de fill (0.0-1.0, ex: 0.70 = 70%% des ordres remplis)")
    p.add_argument("--max-alloc", type=float, default=0.0, help="Cap USD absolu par position (ex: 5000 = max 5000$ par trade, 0 = pas de cap)")
    p.add_argument("--spread-listing", type=float, default=0.0, help="Spread spécifique au listing en plus du spread normal (ex: 0.05 = 5%%)")

    # ── Presets ──
    p.add_argument("--realistic", action="store_true", help="Preset réaliste (slip 3%%/1%%/0.5%%, spread 1%%, fees 0.2%%)")

    return p.parse_args()


def _fetch_all_trades(args: argparse.Namespace) -> list[ListingTrade]:
    """Fetch listing symbols, download candles, compute individual trade results."""
    listing_symbols = fetch_usdc_listing_symbols()
    if not listing_symbols:
        raise RuntimeError("Aucune paire USDC TRADING trouvée via exchangeInfo.")

    selected_pairs: set[str] = set()
    if args.pairs.strip():
        selected_pairs = {x.strip().upper() for x in args.pairs.split(",") if x.strip()}

    meta_cache = load_listing_meta_cache()
    trades: list[ListingTrade] = []
    count_tested = 0

    for item in listing_symbols:
        sym = item.symbol
        if selected_pairs and sym not in selected_pairs:
            continue

        listing_ms = resolve_listing_timestamp(sym, item.onboard_date_ms, meta_cache)
        if listing_ms <= 0:
            continue

        candles = download_listing_candles(
            symbol=sym,
            onboard_date_ms=listing_ms,
            interval=args.interval,
            horizon_days=args.horizon_days,
            use_cache=not args.no_cache,
        )
        if len(candles) < 2:
            continue

        # ── analyse 1m pour entrée réaliste / filtre momentum ──
        override_entry = 0.0
        if args.entry_delay > 0 or args.momentum > 0:
            needed_minutes = max(args.entry_delay, args.momentum_window, 10)
            candles_1m = download_listing_1m_candles(
                symbol=sym,
                listing_ms=listing_ms,
                minutes=needed_minutes,
                use_cache=not args.no_cache,
            )
            entry_price_1m, skip_reason = check_listing_entry(
                candles_1m=candles_1m,
                entry_delay_minutes=args.entry_delay,
                momentum_pct=args.momentum,
                momentum_window=args.momentum_window,
            )
            if entry_price_1m is None:
                logger.debug("%s skipped: %s", sym, skip_reason)
                continue
            if args.entry_delay > 0:
                override_entry = entry_price_1m

        trade = run_listing_trade(
            symbol=sym,
            onboard_date_ms=listing_ms,
            candles=candles,
            sl_init_pct=args.sl_init,
            tp_init_pct=args.tp_init,
            tp_near_ratio=args.tp_near_ratio,
            sl2_tp1_mult=args.sl2_tp1_mult,
            tp2_tp1_mult=args.tp2_tp1_mult,
            slippage_entry=args.slippage_entry,
            slippage_stop=args.slippage_stop,
            slippage_tp=args.slippage_tp,
            spread=args.spread,
            fees_roundtrip=args.fees,
            min_volume_1h=args.min_volume,
            pullback_pct=args.pullback,
            pullback_window=args.pullback_window,
            override_entry_price=override_entry,
            spread_listing=getattr(args, 'spread_listing', 0.0),
        )
        if trade:
            trades.append(trade)
            count_tested += 1

        if args.max_symbols > 0 and count_tested >= args.max_symbols:
            break

    save_listing_meta_cache(meta_cache)
    return trades


def _apply_presets(args: argparse.Namespace) -> None:
    """Applique les presets de frictions si --realistic est activé."""
    if args.realistic:
        if args.slippage_entry == 0.0:
            args.slippage_entry = 0.03
        if args.slippage_stop == 0.0:
            args.slippage_stop = 0.01
        if args.slippage_tp == 0.0:
            args.slippage_tp = 0.005
        if args.spread == 0.0:
            args.spread = 0.01
        if args.fees == 0.0:
            args.fees = 0.002


def _format_friction_summary(args: argparse.Namespace) -> str:
    """Résumé des frictions actives."""
    parts = []
    if args.slippage_entry > 0:
        parts.append(f"slip_entry={args.slippage_entry:.1%}")
    if args.slippage_stop > 0:
        parts.append(f"slip_SL={args.slippage_stop:.1%}")
    if args.slippage_tp > 0:
        parts.append(f"slip_TP={args.slippage_tp:.1%}")
    if args.spread > 0:
        parts.append(f"spread={args.spread:.1%}")
    if args.fees > 0:
        parts.append(f"fees={args.fees:.1%}")
    if args.min_volume > 0:
        parts.append(f"min_vol=${args.min_volume:,.0f}")
    if args.pullback > 0:
        parts.append(f"pullback={args.pullback:.1%} ({args.pullback_window} bougies)")
    if getattr(args, 'entry_delay', 0) > 0:
        parts.append(f"entry_delay={args.entry_delay}min")
    if getattr(args, 'momentum', 0) > 0:
        parts.append(f"momentum≥+{args.momentum:.1%}/{args.momentum_window}min")
    if getattr(args, 'fill_rate', 1.0) < 1.0:
        parts.append(f"fill_rate={args.fill_rate:.0%}")
    if getattr(args, 'max_alloc', 0) > 0:
        parts.append(f"max_alloc=${args.max_alloc:,.0f}")
    if getattr(args, 'spread_listing', 0) > 0:
        parts.append(f"spread_listing={args.spread_listing:.1%}")
    return " | ".join(parts) if parts else "aucune (idéal)"


def main() -> int:
    args = parse_args()
    _apply_presets(args)

    # ── mode portfolio ──
    if args.portfolio:
        return main_portfolio(args)

    # ── mode classique (per-symbol) ──
    trades = _fetch_all_trades(args)
    if not trades:
        logger.error("Aucun trade généré.")
        return 1

    summary = summarize(trades)

    print("\n════════════════════════════════════════════════════════════════════")
    print("  Backtest Listing OCO (Real Binance Listing Event)")
    print("════════════════════════════════════════════════════════════════════")
    print(f"Symboles testés       : {len(trades)}")
    print(f"Intervalle            : {args.interval}")
    print(f"Horizon post-listing  : {args.horizon_days} jours" if args.horizon_days > 0 else "Horizon post-listing  : jusqu'à maintenant")
    print(f"SL/TP init            : -{args.sl_init:.1%} / +{args.tp_init:.1%}")
    print(f"Trigger re-arm        : {args.tp_near_ratio:.0%} de TP initial (price >= {args.tp_near_ratio:.2f} * TP1)")
    print(f"SL2/TP2               : SL2 = {args.sl2_tp1_mult:.3f} * TP1 | TP2 = {args.tp2_tp1_mult:.3f} * TP1")
    print("────────────────────────────────────────────────────────────────────")
    print(f"Trades                : {summary.n_trades}")
    print(f"Win rate              : {summary.win_rate:.1%}")
    print(f"PnL moyen / trade     : {summary.avg_pnl_pct:+.2%}")
    print(f"PnL cumulé (non composé): {summary.total_pnl_pct:+.2%}")
    print(f"Meilleur trade        : {summary.best_pct:+.2%}")
    print(f"Pire trade            : {summary.worst_pct:+.2%}")

    rearmed = sum(1 for t in trades if t.oco_rearmed)
    print(f"Re-arm OCO déclenché  : {rearmed}/{summary.n_trades} ({rearmed/summary.n_trades:.1%})")

    print("\nTop 10 récents:")
    for t in trades[-10:]:
        dt_list = datetime.fromtimestamp(t.onboard_date_ms / 1000, tz=timezone.utc).strftime("%Y-%m-%d")
        dt_in = datetime.fromtimestamp(t.entry_ts / 1000, tz=timezone.utc).strftime("%Y-%m-%d")
        dt_out = datetime.fromtimestamp(t.exit_ts / 1000, tz=timezone.utc).strftime("%Y-%m-%d")
        print(
            f"- {t.symbol:14s} list {dt_list} | {dt_in} -> {dt_out} | {t.exit_reason:11s} | "
            f"PnL {t.pnl_pct:+.2%} | rearm={'Y' if t.oco_rearmed else 'N'}"
        )

    return 0


def main_portfolio(args: argparse.Namespace) -> int:
    """Mode portfolio — vrai capital, slots, PnL composé sur 6m / 1a / 2a."""
    print("\n⏳ Récupération des données de listing Binance…")
    trades = _fetch_all_trades(args)
    if not trades:
        logger.error("Aucun trade généré.")
        return 1

    print(f"   {len(trades)} trades pré-calculés sur l'ensemble des listings USDC.")

    now = datetime.now(timezone.utc)
    now_ms = int(now.timestamp() * 1000)

    periods = [
        (6, "6 mois"),
        (12, "1 an"),
        (24, "2 ans"),
    ]

    results: list[PortfolioResult] = []
    for months, label in periods:
        start_dt = now - timedelta(days=months * 30)
        start_ms = int(start_dt.timestamp() * 1000)
        result = run_portfolio_simulation(
            all_trades=trades,
            initial_capital=args.capital,
            max_slots=args.max_slots,
            lookback_start_ms=start_ms,
            lookback_end_ms=now_ms,
            period_label=label,
            lookback_months=months,
            fill_rate=getattr(args, 'fill_rate', 1.0),
            max_alloc_usd=getattr(args, 'max_alloc', 0.0),
        )
        results.append(result)

    print_portfolio_results(results, args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
