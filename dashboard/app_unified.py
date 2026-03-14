"""
TradeX Unified Dashboard — Overview + 4 onglets (Binance Range · Binance CrashBot · Revolut Infinity · Revolut London).
Un seul processus Streamlit, port 8502.
Lit les données depuis Firebase Firestore.
"""

from __future__ import annotations

import os
from datetime import datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import streamlit as st
from google.cloud import firestore
from google.oauth2 import service_account

# Allocator (pure logic — pas d'I/O)
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.core.allocator import AllocationRegime
from src import config as app_config

# ── Config ─────────────────────────────────────────────────────────────────────

_BASE = Path(__file__).resolve().parent.parent
_CRED_PATH = os.getenv(
    "FIREBASE_CREDENTIALS_PATH",
    str(_BASE / "firebase-credentials.json"),
)

st.set_page_config(
    page_title="TradeX Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

DASHBOARD_TZ = ZoneInfo(os.getenv("DASHBOARD_TZ", "Europe/Paris"))


def _to_display_datetime(series: pd.Series, normalize_day: bool = False) -> pd.Series:
    """Convertit une série datetime en timezone dashboard puis retire la timezone.

    Objectif: éviter les décalages UTC/local et les mélanges tz-aware/tz-naive
    dans les courbes Plotly.
    """
    dt = pd.to_datetime(series, errors="coerce", utc=True)
    dt = dt.dt.tz_convert(DASHBOARD_TZ).dt.tz_localize(None)
    if normalize_day:
        dt = dt.dt.normalize()
    return dt


# ══════════════════════════════════════════════════════════════════════════════
#  Firebase & Data Fetching
# ══════════════════════════════════════════════════════════════════════════════

@st.cache_resource
def _get_db() -> firestore.Client:
    cred = service_account.Credentials.from_service_account_file(_CRED_PATH)
    return firestore.Client(project=cred.project_id, credentials=cred)


def _fetch_trades(days: int = 90, exchange: str | None = None) -> pd.DataFrame:
    db = _get_db()
    since = datetime.now(timezone.utc) - timedelta(days=days)
    query = db.collection("trades")
    if exchange:
        query = query.where("exchange", "==", exchange)
    docs = (
        query
        .where("created_at", ">=", since.isoformat())
        .order_by("created_at", direction=firestore.Query.DESCENDING)
        .stream()
    )
    rows = [doc.to_dict() | {"_id": doc.id} for doc in docs]
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    for col in ("opened_at", "closed_at", "created_at", "updated_at"):
        if col in df.columns:
            df[col] = _to_display_datetime(df[col])
    return df


def _fetch_daily_snapshots(days: int = 90, exchange: str | None = None) -> pd.DataFrame:
    db = _get_db()
    since = (datetime.now(timezone.utc) - timedelta(days=days)).strftime("%Y-%m-%d")
    query = db.collection("daily_snapshots")
    if exchange:
        query = query.where("exchange", "==", exchange)
    docs = query.where("date", ">=", since).order_by("date").stream()
    rows = [doc.to_dict() for doc in docs]
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    df["date"] = _to_display_datetime(df["date"], normalize_day=True)
    return df


def _fetch_open_positions(exchange: str | None = None) -> pd.DataFrame:
    db = _get_db()
    query = db.collection("trades")
    if exchange:
        query = query.where("exchange", "==", exchange)
    docs = query.where("status", "in", ["OPEN", "ZERO_RISK", "TRAILING"]).stream()
    rows = [doc.to_dict() | {"_id": doc.id} for doc in docs]
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows)


def _fetch_close_failures(hours: int = 24, exchange: str | None = None) -> pd.DataFrame:
    db = _get_db()
    cutoff = (datetime.now(timezone.utc) - timedelta(hours=hours)).isoformat()
    query = db.collection("events")
    if exchange:
        query = query.where("exchange", "==", exchange)
    docs = (
        query
        .where("event_type", "==", "CLOSE_FAILURE")
        .where("timestamp", ">=", cutoff)
        .order_by("timestamp", direction=firestore.Query.DESCENDING)
        .limit(50)
        .stream()
    )
    rows = []
    for doc in docs:
        d = doc.to_dict()
        data = d.get("data", {})
        rows.append({
            "symbol": d.get("symbol", "?"),
            "attempt": data.get("attempt", 0),
            "error": data.get("error", "?"),
            "next_retry_s": data.get("next_retry_seconds", 0),
            "timestamp": d.get("timestamp", ""),
        })
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows)


def _fetch_events(exchange: str, event_type: str | None = None, hours: int = 48) -> pd.DataFrame:
    db = _get_db()
    cutoff = (datetime.now(timezone.utc) - timedelta(hours=hours)).isoformat()
    query = (
        db.collection("events")
        .where("exchange", "==", exchange)
        .where("timestamp", ">=", cutoff)
        .order_by("timestamp", direction=firestore.Query.DESCENDING)
        .limit(100)
    )
    if event_type:
        query = query.where("event_type", "==", event_type)
    docs = query.stream()
    rows = [doc.to_dict() for doc in docs]
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows)


def _fetch_last_heartbeat(exchange: str, hours: int = 72) -> dict:
    """Retourne le dernier heartbeat Firebase pour un exchange donné."""
    db = _get_db()
    cutoff = (datetime.now(timezone.utc) - timedelta(hours=hours)).isoformat()
    docs = (
        db.collection("events")
        .where("exchange", "==", exchange)
        .where("event_type", "==", "HEARTBEAT")
        .where("timestamp", ">=", cutoff)
        .order_by("timestamp", direction=firestore.Query.DESCENDING)
        .limit(1)
        .stream()
    )
    for doc in docs:
        return doc.to_dict() or {}
    return {}


# ══════════════════════════════════════════════════════════════════════════════
#  Helpers
# ══════════════════════════════════════════════════════════════════════════════

def _color_pnl(val):
    if pd.isna(val):
        return ""
    return "color: #00c853" if val >= 0 else "color: #ff1744"


def _fetch_current_allocation() -> dict:
    """Lit l'allocation courante depuis Firebase (doc allocation/current).

    Ce document est écrit par les bots à chaque recalcul d'allocation.
    """
    try:
        db = _get_db()
        doc = db.collection("allocation").document("current").get()
        if not doc.exists:
            return {}
        data = doc.to_dict()
        # Convertir le regime string en enum pour la cohérence
        regime_str = data.get("regime", "defensive")
        regime_map = {
            "defensive": AllocationRegime.DEFENSIVE,
            "neutral": AllocationRegime.NEUTRAL,
            "aggressive": AllocationRegime.AGGRESSIVE,
        }
        data["regime"] = regime_map.get(regime_str, AllocationRegime.DEFENSIVE)
        return data
    except Exception:
        return {}


def _fmt_price(p):
    if pd.isna(p) or p is None:
        return "—"
    if p >= 1000:
        return f"{p:,.4f}"
    if p >= 1:
        return f"{p:.4f}"
    if p >= 0.0001:
        return f"{p:.6f}"
    d = 6
    t = p
    while t < 0.01 and d < 10:
        t *= 10
        d += 1
    return f"{p:.{d}f}"


def _apply_mobile_chart_theme(fig: go.Figure, show_legend: bool = False) -> go.Figure:
    """Applique un style visuel type app mobile trading (dark + cyan)."""
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="#05080d",
        plot_bgcolor="#05080d",
        font=dict(color="#d9e7f0"),
        hovermode="x unified",
        hoverlabel=dict(bgcolor="#101722", font_color="#d9e7f0"),
        margin=dict(l=0, r=0, t=10, b=0),
        showlegend=show_legend,
    )
    fig.update_xaxes(
        showgrid=True,
        gridcolor="rgba(255,255,255,0.08)",
        zeroline=False,
        linecolor="rgba(255,255,255,0.18)",
    )
    fig.update_yaxes(
        showgrid=True,
        gridcolor="rgba(255,255,255,0.08)",
        zeroline=False,
        linecolor="rgba(255,255,255,0.18)",
    )
    return fig


@st.cache_data(ttl=15)
def _fetch_binance_prices() -> dict[str, float]:
    """Récupère tous les prix actuels depuis l'API publique Binance."""
    try:
        resp = requests.get(
            "https://api.binance.com/api/v3/ticker/price",
            timeout=5,
        )
        resp.raise_for_status()
        return {t["symbol"]: float(t["price"]) for t in resp.json()}
    except Exception:
        return {}


@st.cache_data(ttl=15)
def _fetch_revolut_prices() -> dict[str, float]:
    """Récupère les prix actuels depuis l'API publique Revolut X (tickers)."""
    try:
        resp = requests.get(
            "https://revx.revolut.com/api/1.0/tickers",
            timeout=5,
        )
        resp.raise_for_status()
        return {t["symbol"]: float(t["last"]) for t in resp.json() if "last" in t}
    except Exception:
        return {}


def _enrich_with_prices(df: pd.DataFrame, exchange: str = "binance") -> pd.DataFrame:
    """Ajoute current_price et pnl_latent (%) aux positions ouvertes."""
    if df.empty or "symbol" not in df.columns:
        return df
    df = df.copy()
    if exchange == "revolut":
        prices = _fetch_revolut_prices()
    else:
        prices = _fetch_binance_prices()
    df["current_price"] = df["symbol"].map(prices)

    def _calc_pnl(row):
        cp = row.get("current_price")
        entry = row.get("entry_filled")
        side = row.get("side", "buy")
        if pd.isna(cp) or pd.isna(entry) or not entry:
            return None
        if str(side).lower() in ("sell", "short"):
            return (entry - cp) / entry * 100
        return (cp - entry) / entry * 100

    df["pnl_latent"] = df.apply(_calc_pnl, axis=1)

    def _calc_pnl_usd(row):
        cp = row.get("current_price")
        entry = row.get("entry_filled")
        size = row.get("size")
        side = row.get("side", "buy")
        if pd.isna(cp) or pd.isna(entry) or not entry or pd.isna(size) or not size:
            return None
        if str(side).lower() in ("sell", "short"):
            return (entry - cp) * size
        return (cp - entry) * size

    df["pnl_latent_usd"] = df.apply(_calc_pnl_usd, axis=1)
    return df


def _fmt_price_with_dist(price, current_price) -> str:
    """Formate un prix + distance % par rapport au prix actuel.

    Ex: '95,000.0000 (-1.52%)'
    """
    if pd.isna(price) or price is None:
        return "—"
    formatted = _fmt_price(price)
    if pd.isna(current_price) or current_price is None or current_price == 0:
        return formatted
    dist = (price - current_price) / current_price * 100
    return f"{formatted} ({dist:+.1f}%)"


def _format_positions_display(df: pd.DataFrame) -> pd.DataFrame:
    """Formate les colonnes Entrée, SL, TP avec prix + distance % du prix actuel."""
    if df.empty:
        return df
    df = df.copy()

    for col in ("entry_filled", "sl_price", "tp_price"):
        if col in df.columns and "current_price" in df.columns:
            df[col] = df.apply(
                lambda r: _fmt_price_with_dist(r.get(col), r.get("current_price")),
                axis=1,
            )

    if "current_price" in df.columns:
        df["current_price"] = df["current_price"].apply(
            lambda x: _fmt_price(x) if pd.notna(x) else "—"
        )

    if "pnl_latent" in df.columns:
        def _fmt_pnl(row):
            pct = row.get("pnl_latent")
            usd = row.get("pnl_latent_usd")
            if pd.isna(pct):
                return "—"
            if pd.notna(usd):
                return f"${usd:+.2f} ({pct:+.2f}%)"
            return f"{pct:+.2f}%"
        df["pnl_latent"] = df.apply(_fmt_pnl, axis=1)

    # Supprimer la colonne intermédiaire
    if "pnl_latent_usd" in df.columns:
        df = df.drop(columns=["pnl_latent_usd"])

    return df


def _compute_stats(closed: pd.DataFrame) -> dict:
    """Calcule les KPIs à partir d'un DataFrame de trades fermés."""
    if closed.empty or "pnl_usd" not in closed.columns:
        return {}
    total_pnl = closed["pnl_usd"].sum()
    win_trades = closed[closed["pnl_usd"] > 0]
    loss_trades = closed[closed["pnl_usd"] <= 0]
    n = len(closed)
    win_rate = len(win_trades) / n * 100 if n > 0 else 0
    pf = (
        abs(win_trades["pnl_usd"].sum() / loss_trades["pnl_usd"].sum())
        if len(loss_trades) > 0 and loss_trades["pnl_usd"].sum() != 0
        else float("inf")
    )
    equity = None
    if "equity_after" in closed.columns:
        last = closed.sort_values("closed_at").iloc[-1].get("equity_after", None)
        if pd.notna(last):
            equity = last
    # Max drawdown
    cumulative = closed.sort_values("closed_at")["pnl_usd"].cumsum()
    running_max = cumulative.cummax()
    max_dd = (cumulative - running_max).min()
    return {
        "total_pnl": total_pnl,
        "n_trades": n,
        "n_wins": len(win_trades),
        "n_losses": len(loss_trades),
        "win_rate": win_rate,
        "profit_factor": pf,
        "equity": equity,
        "max_dd": max_dd,
        "avg_win": win_trades["pnl_usd"].mean() if len(win_trades) > 0 else 0,
        "avg_loss": loss_trades["pnl_usd"].mean() if len(loss_trades) > 0 else 0,
    }


# ══════════════════════════════════════════════════════════════════════════════
#  Sidebar (global)
# ══════════════════════════════════════════════════════════════════════════════

st.sidebar.image("https://img.icons8.com/color/96/combo-chart.png", width=60)
st.sidebar.title("TradeX")
st.sidebar.caption("Dashboard unifié — 4 bots")

days_filter = st.sidebar.slider("Période (jours)", 1, 90, 30)

if st.sidebar.button("🔄 Rafraîchir les données"):
    st.cache_resource.clear()
    st.rerun()

auto_refresh = st.sidebar.checkbox("Auto-refresh (60s)", value=False)
if auto_refresh:
    st.markdown(
        '<meta http-equiv="refresh" content="60">',
        unsafe_allow_html=True,
    )


# ══════════════════════════════════════════════════════════════════════════════
#  Load all data
# ══════════════════════════════════════════════════════════════════════════════

BOTS = {
    "binance": {"label": "Binance Range", "icon": "🟡", "exchange": "binance", "color": "#f0b90b", "max_pos": 3},
    "crashbot": {"label": "Binance CrashBot", "icon": "💥", "exchange": "binance-crashbot", "color": "#7c4dff", "max_pos": 5},
    "listing": {"label": "Binance Listing", "icon": "🆕", "exchange": "binance-listing", "color": "#00e676", "max_pos": 3},
    "infinity": {"label": "Revolut Infinity", "icon": "♾️", "exchange": "revolut-infinity", "color": "#ff6d00", "max_pos": 6},
    "london": {"label": "Revolut London", "icon": "🇬🇧", "exchange": "revolut-london", "color": "#2196f3", "max_pos": 1},
}


@st.cache_data(ttl=55)
def _load_all(days: int):
    """Charge les données des 3 bots en un seul appel caché."""
    data = {}
    for key, cfg in BOTS.items():
        ex = cfg["exchange"]
        trades = _fetch_trades(days=days, exchange=ex)
        closed = trades[trades["closed_at"].notna()].copy() if not trades.empty else pd.DataFrame()
        data[key] = {
            "trades": trades,
            "closed": closed,
            "snapshots": _fetch_daily_snapshots(days=days, exchange=ex),
            "open": _fetch_open_positions(exchange=ex),
            "stats": _compute_stats(closed),
            "heartbeat": _fetch_last_heartbeat(ex),
        }
    return data


INF_PAIRS = ["BTC-USD", "AAVE-USD", "XLM-USD", "ADA-USD", "DOT-USD", "LTC-USD"]


@st.cache_data(ttl=55)
def _fetch_infinity_cycles() -> dict[str, dict]:
    """Lit les cycles Infinity par paire depuis Firebase (infinity_cycles/{symbol})."""
    cycles = {}
    try:
        db = _get_db()
        for symbol in INF_PAIRS:
            doc = db.collection("infinity_cycles").document(symbol).get()
            if doc.exists:
                cycles[symbol] = doc.to_dict()
    except Exception:
        pass
    # Fallback: try legacy "current" doc
    if not cycles:
        try:
            db = _get_db()
            doc = db.collection("infinity_cycles").document("current").get()
            if doc.exists:
                data = doc.to_dict()
                sym = data.get("symbol", "BTC-USD")
                cycles[sym] = data
        except Exception:
            pass
    return cycles


all_data = _load_all(days_filter)


# ══════════════════════════════════════════════════════════════════════════════
#  Reusable rendering functions
# ══════════════════════════════════════════════════════════════════════════════

def _render_kpis(stats: dict, n_open: int, max_pos: int):
    """Affiche les 6 KPIs en colonnes."""
    col1, col2, col3, col4, col5, col6 = st.columns(6)
    if stats:
        pf = stats["profit_factor"]
        col1.metric("💰 P&L Total", f"${stats['total_pnl']:+,.2f}")
        col2.metric("📈 Trades", f"{stats['n_trades']}", f"{stats['n_wins']}W / {stats['n_losses']}L")
        col3.metric("🎯 Win Rate", f"{stats['win_rate']:.1f}%")
        col4.metric("⚖️ Profit Factor", f"{pf:.2f}" if pf != float("inf") else "∞")
        col5.metric("🏦 Equity", f"${stats['equity']:,.2f}" if stats['equity'] else "—")
        col6.metric("📂 Positions", f"{n_open}/{max_pos}")
    else:
        col1.metric("💰 P&L Total", "—")
        col2.metric("📈 Trades", "0")
        col3.metric("🎯 Win Rate", "—")
        col4.metric("⚖️ Profit Factor", "—")
        col5.metric("🏦 Equity", "—")
        col6.metric("📂 Positions", f"{n_open}/{max_pos}")


def _render_last_heartbeat_cockpit(bot_key: str, d: dict, max_pos: int):
    """Affiche le dernier heartbeat dans un format cockpit uniforme."""
    hb = d.get("heartbeat") or {}
    if not hb:
        st.info("💓 Aucun heartbeat Firebase récent.")
        return

    data = hb.get("data") or {}
    hb_ts_raw = hb.get("timestamp")
    hb_ts = pd.to_datetime(hb_ts_raw, errors="coerce", utc=True)
    if pd.isna(hb_ts):
        ts_label = "—"
        stale_min = None
    else:
        hb_local = hb_ts.tz_convert(DASHBOARD_TZ)
        ts_label = hb_local.strftime("%d/%m %H:%M:%S")
        stale_min = (datetime.now(timezone.utc) - hb_ts.to_pydatetime()).total_seconds() / 60

    open_positions = int(data.get("open_positions", len(d.get("open", []))))
    pairs_count = int(data.get("pairs_count", 0))
    total_equity = float(data.get("total_equity", 0.0))
    risk_pct = float(data.get("total_risk_pct", 0.0)) * 100

    open_df = d.get("open") if isinstance(d, dict) else None
    exposure_usd = 0.0
    if isinstance(open_df, pd.DataFrame) and not open_df.empty and "size_usd" in open_df.columns:
        exposure_usd = float(pd.to_numeric(open_df["size_usd"], errors="coerce").fillna(0).sum())

    if stale_min is not None and stale_min > 20:
        sys_emoji, sys_label = "🔴", "stale"
    elif risk_pct > 6:
        sys_emoji, sys_label = "🟡", "watch"
    else:
        sys_emoji, sys_label = "🟢", "ok"

    lines = [
        f"{sys_emoji} **Dernier heartbeat** · {sys_label.upper()} · {ts_label}",
        f"💰 Equity `${total_equity:,.0f}` · 📂 Pos `{open_positions}/{max_pos}` · ⚠️ Risque `{risk_pct:.1f}%` · 📦 Expo `${exposure_usd:,.0f}`",
    ]

    if bot_key == "infinity":
        slots = max(1, int(app_config.INF_CAPITAL_ACTIVE_SLOTS))
        slots_used = min(open_positions, slots)
        slots_free = max(0, slots - slots_used)
        lines.append(f"🧩 Slots `{slots_used}/{slots}` · libres `{slots_free}` · paires suivies `{pairs_count}`")
    elif bot_key == "crashbot":
        lines.append(f"📡 Universe `{pairs_count}` paires · Signaux/état via Telegram heartbeat")
    elif bot_key == "london":
        lines.append(f"🇬🇧 Session breakout · paires suivies `{pairs_count}`")
    else:
        lines.append(f"🟡 Range bot · paires suivies `{pairs_count}`")

    st.markdown("\n\n".join(lines))


def _render_positions(open_df: pd.DataFrame, bot_type: str = "range", exchange: str = "binance"):
    """Affiche les positions ouvertes."""
    # Détecter si certaines positions ont le trailing actif
    has_trailing = (
        not open_df.empty
        and "trailing_active" in open_df.columns
        and open_df["trailing_active"].any()
    )

    if bot_type == "crashbot":
        st.subheader("🟢 Positions ouvertes — Step Trailing")
    elif has_trailing:
        st.subheader("🟢 Positions ouvertes — 🔄 Trailing")
    else:
        st.subheader("🟢 Positions ouvertes")

    if not open_df.empty:
        # Enrichir avec les prix actuels
        open_df = _enrich_with_prices(open_df, exchange=exchange)

        if bot_type == "crashbot":
            display_cols = [
                "symbol", "side", "entry_filled", "sl_price",
                "current_price", "tp_price",
                "trail_steps", "peak_price", "trail_gain_pct",
                "size", "size_usd", "risk_usd",
                "pnl_latent", "pnl_latent_usd", "status", "opened_at",
            ]
        else:
            display_cols = [
                "symbol", "side", "entry_filled", "sl_price",
                "current_price", "tp_price",
                "size", "size_usd", "risk_usd", "strategy",
                "pnl_latent", "pnl_latent_usd",
            ]
            # Ajouter les colonnes trailing si elles existent
            if has_trailing:
                display_cols.insert(6, "trailing_steps")
            display_cols += ["status", "opened_at"]

        available_cols = [c for c in display_cols if c in open_df.columns]
        show_df = open_df[available_cols].copy()

        # Formater prix avec distance % du prix actuel
        show_df = _format_positions_display(show_df)

        # Adapter le label SL selon le type de bot
        sl_label = "Trail SL" if bot_type == "crashbot" else ("SL 🔄" if has_trailing else "SL")
        tp_label = "TP Cible" if bot_type == "crashbot" else "TP"
        rename_map = {
            "symbol": "Paire", "side": "Side", "entry_filled": "Entrée",
            "sl_price": sl_label,
            "current_price": "💲 Prix actuel",
            "tp_price": tp_label,
            "peak_price": "Peak",
            "trail_gain_pct": "Gain %", "size": "Taille",
            "trailing_steps": "Trail Step", "trail_steps": "Steps",
            "size_usd": "Notionnel $", "risk_usd": "Risque $",
            "pnl_latent": "P&L latent",
            "strategy": "Stratégie", "status": "Statut", "opened_at": "Ouvert le",
        }
        show_df = show_df.rename(columns={k: v for k, v in rename_map.items() if k in show_df.columns})

        if "Gain %" in show_df.columns:
            show_df["Gain %"] = show_df["Gain %"].apply(
                lambda x: f"{x * 100:+.2f}%" if pd.notna(x) else "—"
            )

        # Mettre en forme le statut avec emoji
        if "Statut" in show_df.columns and has_trailing:
            show_df["Statut"] = show_df["Statut"].apply(
                lambda s: f"🔄 {s}" if s == "TRAILING" else s
            )

        st.dataframe(show_df, width='stretch', hide_index=True)
    else:
        st.info("Aucune position ouverte")


def _render_alerts(open_df: pd.DataFrame, exchange: str | None):
    """Affiche les alertes (close bloquées)."""
    blocked_trades = pd.DataFrame()
    if not open_df.empty and "close_blocked" in open_df.columns:
        blocked_trades = open_df[open_df["close_blocked"] == True]

    close_failures_df = _fetch_close_failures(hours=24, exchange=exchange)

    has_alerts = (not blocked_trades.empty) or (not close_failures_df.empty)
    if not has_alerts:
        return

    st.subheader("⚠️ Alertes")

    if not blocked_trades.empty:
        for _, row in blocked_trades.iterrows():
            sym = row.get("symbol", "?")
            attempts = row.get("close_blocked_attempts", "?")
            err = row.get("close_blocked_error", "Erreur inconnue")
            blocked_at = row.get("close_blocked_at", "")
            st.error(
                f"🛑 **{sym}** — Clôture bloquée (×{attempts})\n\n"
                f"`{err}`\n\n"
                f"Dernière tentative : {blocked_at[:19] if blocked_at else '?'}",
                icon="🚨",
            )

    if not close_failures_df.empty:
        with st.expander(f"📄 Historique des échecs de clôture (24h) — {len(close_failures_df)} événements"):
            st.dataframe(
                close_failures_df.rename(columns={
                    "symbol": "Paire", "attempt": "Tentative",
                    "error": "Erreur", "next_retry_s": "Retry (s)",
                    "timestamp": "Date",
                }),
                width='stretch', hide_index=True,
            )


def _render_equity_curve(snapshots_df: pd.DataFrame, closed: pd.DataFrame, color: str):
    """Affiche la courbe d'equity."""
    st.subheader("📈 Courbe d'equity")

    line_color = "#14d8c4"
    pos_color = "#14d8c4"
    neg_color = "#ff5c7a"
    neu_color = "#9aa7b3"

    if not snapshots_df.empty and "equity" in snapshots_df.columns:
        eq_df = snapshots_df.sort_values("date").copy()
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=eq_df["date"],
            y=eq_df["equity"],
            mode="lines",
            name="Equity",
            line=dict(color=line_color, width=3),
            connectgaps=True,
        ))

        if not closed.empty and "closed_at" in closed.columns:
            markers = closed.dropna(subset=["closed_at"]).copy()
            if not markers.empty:
                y_ref = eq_df.set_index("date")["equity"]
                marker_y = y_ref.reindex(markers["closed_at"], method="nearest").values
                marker_colors = [
                    pos_color if (v or 0) > 0 else neg_color if (v or 0) < 0 else neu_color
                    for v in markers.get("pnl_usd", pd.Series([0] * len(markers)))
                ]
                fig.add_trace(go.Scatter(
                    x=markers["closed_at"],
                    y=marker_y,
                    mode="markers",
                    name="Trades",
                    marker=dict(size=7, color=marker_colors, line=dict(width=1, color="#111")),
                    opacity=0.95,
                ))

        fig.update_layout(
            height=350,
            yaxis_tickformat="$,.0f",
        )
        _apply_mobile_chart_theme(fig, show_legend=False)
        st.plotly_chart(fig, width='stretch')
    elif not closed.empty and "pnl_usd" in closed.columns:
        eq_from_field = (
            closed.dropna(subset=["equity_after"]).sort_values("closed_at")
            if "equity_after" in closed.columns else pd.DataFrame()
        )

        fig = go.Figure()
        if not eq_from_field.empty:
            eq_df = eq_from_field
            x_col, y_col = "closed_at", "equity_after"
        else:
            eq_df = closed.sort_values("closed_at").copy()
            initial_equity = eq_df.iloc[0].get("equity_at_entry", 1000) or 1000
            eq_df["equity_curve"] = initial_equity + eq_df["pnl_usd"].cumsum()

            x_col, y_col = "closed_at", "equity_curve"

        fig.add_trace(go.Scatter(
            x=eq_df[x_col],
            y=eq_df[y_col],
            mode="lines",
            name="Equity",
            line=dict(color=line_color, width=3),
            connectgaps=True,
        ))

        marker_colors = [
            pos_color if (v or 0) > 0 else neg_color if (v or 0) < 0 else neu_color
            for v in eq_df.get("pnl_usd", pd.Series([0] * len(eq_df)))
        ]
        fig.add_trace(go.Scatter(
            x=eq_df[x_col],
            y=eq_df[y_col],
            mode="markers",
            name="Trades",
            marker=dict(size=7, color=marker_colors, line=dict(width=1, color="#111")),
            opacity=0.95,
        ))

        fig.update_layout(
            height=350,
            yaxis_tickformat="$,.0f",
        )
        _apply_mobile_chart_theme(fig, show_legend=False)
        st.plotly_chart(fig, width='stretch')
    else:
        st.info("Pas de données d'equity disponibles")


def _render_cumulative_pnl(closed: pd.DataFrame, color: str):
    """Affiche le P&L cumulé."""
    if closed.empty or "pnl_usd" not in closed.columns:
        return
    st.subheader("💵 P&L cumulé")
    pnl_df = closed.sort_values("closed_at").copy()
    pnl_df["cumulative_pnl"] = pnl_df["pnl_usd"].cumsum()

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=pnl_df["closed_at"], y=pnl_df["cumulative_pnl"],
        mode="lines+markers", name="P&L cumulé",
        line=dict(color="#14d8c4", width=3),
        marker=dict(size=5, color="#14d8c4"),
    ))
    fig.add_hline(y=0, line_dash="dot", line_color="#9aa7b3", opacity=0.6)
    fig.update_layout(
        height=300,
        yaxis_tickformat="$,.2f",
    )
    _apply_mobile_chart_theme(fig, show_legend=False)
    st.plotly_chart(fig, width='stretch')


def _render_daily_pnl(closed: pd.DataFrame):
    """Affiche le P&L journalier."""
    if closed.empty or "pnl_usd" not in closed.columns:
        return
    st.subheader("📅 P&L journalier")

    daily_pnl = closed.copy()
    daily_pnl["day"] = daily_pnl["closed_at"].dt.date
    daily_agg = daily_pnl.groupby("day").agg(
        pnl=("pnl_usd", "sum"),
        trades=("pnl_usd", "count"),
        wins=("pnl_usd", lambda x: (x > 0).sum()),
    ).reset_index()

    colors = ["#00c853" if v >= 0 else "#ff1744" for v in daily_agg["pnl"]]

    fig = go.Figure(go.Bar(
        x=daily_agg["day"], y=daily_agg["pnl"],
        marker_color=colors,
        text=[f"${v:+.2f}" for v in daily_agg["pnl"]],
        textposition="outside",
        hovertemplate="<b>%{x}</b><br>P&L: $%{y:+.2f}<br>Trades: %{customdata[0]}<br>Wins: %{customdata[1]}<extra></extra>",
        customdata=daily_agg[["trades", "wins"]].values,
    ))
    fig.update_layout(
        height=300, margin=dict(l=0, r=0, t=10, b=0),
        yaxis_tickformat="$,.2f",
    )
    st.plotly_chart(fig, width='stretch')


def _render_pair_performance(closed: pd.DataFrame):
    """Affiche la performance par paire."""
    if closed.empty or "pnl_usd" not in closed.columns:
        return
    st.subheader("🪙 Performance par paire")

    col_left, col_right = st.columns(2)

    pair_agg = closed.groupby("symbol").agg(
        pnl=("pnl_usd", "sum"),
        trades=("pnl_usd", "count"),
        wins=("pnl_usd", lambda x: (x > 0).sum()),
        avg_pnl=("pnl_usd", "mean"),
    ).reset_index().sort_values("pnl", ascending=True)

    with col_left:
        colors_pair = ["#00c853" if v >= 0 else "#ff1744" for v in pair_agg["pnl"]]
        fig = go.Figure(go.Bar(
            y=pair_agg["symbol"], x=pair_agg["pnl"],
            orientation="h", marker_color=colors_pair,
            text=[f"${v:+.2f}" for v in pair_agg["pnl"]],
            textposition="outside",
            hovertemplate="<b>%{y}</b><br>P&L: $%{x:+.2f}<br>Trades: %{customdata[0]}<br>Win Rate: %{customdata[1]:.0f}%<extra></extra>",
            customdata=list(zip(
                pair_agg["trades"],
                pair_agg["wins"] / pair_agg["trades"] * 100,
            )),
        ))
        fig.update_layout(
            height=max(300, len(pair_agg) * 28),
            margin=dict(l=0, r=0, t=10, b=0),
            xaxis_tickformat="$,.2f",
        )
        st.plotly_chart(fig, width='stretch')

    with col_right:
        st.dataframe(
            pair_agg.rename(columns={
                "symbol": "Paire", "pnl": "P&L ($)", "trades": "Trades",
                "wins": "Wins", "avg_pnl": "Moy ($)",
            }).sort_values("P&L ($)", ascending=False).style.map(
                _color_pnl, subset=["P&L ($)", "Moy ($)"]
            ).format({"P&L ($)": "${:+,.2f}", "Moy ($)": "${:+,.2f}"}),
            width='stretch', hide_index=True,
        )


def _render_last_trades(closed: pd.DataFrame):
    """Affiche les 30 derniers trades."""
    st.subheader("📋 Derniers trades")

    if closed.empty:
        st.info("Aucun trade fermé sur cette période")
        return

    show_trades = closed.sort_values("closed_at", ascending=False).head(30).copy()

    # Colonnes de base + colonnes optionnelles présentes
    cols = [
        "symbol", "side", "strategy", "entry_filled", "exit_price",
        "peak_price", "trail_steps",
        "pnl_usd", "pnl_pct", "exit_reason", "holding_time_hours",
        "maker_or_taker", "exit_fill_type", "fees_total", "closed_at",
    ]

    display = show_trades[[c for c in cols if c in show_trades.columns]].copy()

    rename = {
        "symbol": "Paire", "side": "Side", "strategy": "Strat",
        "entry_filled": "Entrée", "exit_price": "Sortie",
        "peak_price": "Peak", "trail_steps": "Steps",
        "pnl_usd": "P&L ($)", "pnl_pct": "P&L (%)",
        "exit_reason": "Raison", "holding_time_hours": "Durée (h)",
        "maker_or_taker": "Fill entrée", "exit_fill_type": "Fill sortie",
        "fees_total": "Fees ($)", "closed_at": "Clôturé le",
    }
    display = display.rename(columns={k: v for k, v in rename.items() if k in display.columns})

    if "P&L (%)" in display.columns:
        display["P&L (%)"] = display["P&L (%)"] * 100

    styled = display.style
    if "P&L ($)" in display.columns:
        styled = styled.map(_color_pnl, subset=["P&L ($)"])
        styled = styled.format({"P&L ($)": "${:+,.2f}"})
    if "P&L (%)" in display.columns:
        styled = styled.map(_color_pnl, subset=["P&L (%)"])
        styled = styled.format({"P&L (%)": "{:+.2f}%"})
    if "Durée (h)" in display.columns:
        styled = styled.format({"Durée (h)": "{:.1f}h"})
    if "Fees ($)" in display.columns:
        styled = styled.format({"Fees ($)": "${:.4f}"})

    st.dataframe(styled, width='stretch', hide_index=True)


def _render_pnl_distribution(closed: pd.DataFrame, color: str, group_col: str = "strategy"):
    """Affiche la distribution des P&L."""
    if closed.empty or "pnl_usd" not in closed.columns:
        return
    st.subheader("📊 Distribution des P&L")

    col_hist, col_box = st.columns(2)

    with col_hist:
        fig_hist = px.histogram(
            closed, x="pnl_usd", nbins=30,
            labels={"pnl_usd": "P&L ($)"},
            color_discrete_sequence=[color],
        )
        fig_hist.add_vline(x=0, line_dash="dash", line_color="white", opacity=0.5)
        fig_hist.update_layout(
            height=300, margin=dict(l=0, r=0, t=10, b=0), showlegend=False,
        )
        st.plotly_chart(fig_hist, width='stretch')

    with col_box:
        if group_col in closed.columns:
            fig_box = px.box(
                closed, x=group_col, y="pnl_usd",
                labels={group_col: group_col.title(), "pnl_usd": "P&L ($)"},
                color=group_col,
            )
        else:
            fig_box = px.box(closed, y="pnl_usd", labels={"pnl_usd": "P&L ($)"})
        fig_box.update_layout(
            height=300, margin=dict(l=0, r=0, t=10, b=0), showlegend=False,
        )
        st.plotly_chart(fig_box, width='stretch')


# ── CrashBot-specific sections ─────────────────────────────────────────────────

def _render_kill_switch(exchange: str):
    kill_events = _fetch_events(exchange, "KILL_SWITCH", hours=720)
    if kill_events.empty:
        return
    st.subheader("🚨 Kill-Switch")
    for _, row in kill_events.iterrows():
        data = row.get("data", {})
        st.error(
            f"**Kill-Switch activé** — Perf mois: {data.get('month_return_pct', '?')}% "
            f"(seuil: {data.get('threshold_pct', '?')}%)\n\n"
            f"Equity: ${data.get('equity', 0):,.2f} | "
            f"Début mois: ${data.get('month_start_equity', 0):,.2f}",
            icon="🚨",
        )


def _render_exit_reasons(closed: pd.DataFrame):
    if closed.empty or "exit_reason" not in closed.columns:
        return
    st.subheader("🏁 Raisons de sortie")

    col_pie, col_table = st.columns(2)

    reason_agg = closed.groupby("exit_reason").agg(
        count=("pnl_usd", "count"),
        total_pnl=("pnl_usd", "sum"),
        avg_pnl=("pnl_usd", "mean"),
    ).reset_index().sort_values("count", ascending=False)

    with col_pie:
        fig = px.pie(
            reason_agg, names="exit_reason", values="count",
            color_discrete_sequence=px.colors.qualitative.Set2, hole=0.4,
        )
        fig.update_layout(height=300, margin=dict(l=0, r=0, t=10, b=0))
        st.plotly_chart(fig, width='stretch')

    with col_table:
        st.dataframe(
            reason_agg.rename(columns={
                "exit_reason": "Raison", "count": "Trades",
                "total_pnl": "P&L Total ($)", "avg_pnl": "P&L Moy ($)",
            }).style.map(
                _color_pnl, subset=["P&L Total ($)", "P&L Moy ($)"]
            ).format({"P&L Total ($)": "${:+,.2f}", "P&L Moy ($)": "${:+,.2f}"}),
            width='stretch', hide_index=True,
        )


def _render_trailing_stats(closed: pd.DataFrame):
    if closed.empty:
        return
    if "peak_price" not in closed.columns or "entry_filled" not in closed.columns:
        return

    trail_df = closed.copy()
    trail_df["peak_gain_pct"] = (trail_df["peak_price"] - trail_df["entry_filled"]) / trail_df["entry_filled"] * 100
    trail_df["exit_gain_pct"] = (trail_df["exit_price"] - trail_df["entry_filled"]) / trail_df["entry_filled"] * 100

    trail_valid = trail_df.dropna(subset=["peak_gain_pct", "exit_gain_pct"])
    if trail_valid.empty:
        return

    st.subheader("📊 Performance du Trailing Stop")

    col_t1, col_t2, col_t3 = st.columns(3)
    avg_peak = trail_valid["peak_gain_pct"].mean()
    avg_exit = trail_valid["exit_gain_pct"].mean()
    avg_capture = (
        (trail_valid["exit_gain_pct"] / trail_valid["peak_gain_pct"])
        .replace([float("inf"), -float("inf")], 0).mean() * 100
    )
    col_t1.metric("🏔️ Gain peak moyen", f"{avg_peak:+.2f}%")
    col_t2.metric("🎯 Gain sortie moyen", f"{avg_exit:+.2f}%")
    col_t3.metric("📐 Capture ratio", f"{avg_capture:.0f}%",
                   help="% du gain peak capturé à la sortie")

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=trail_valid["peak_gain_pct"], y=trail_valid["exit_gain_pct"],
        mode="markers",
        marker=dict(
            color=trail_valid["pnl_usd"], colorscale="RdYlGn",
            size=8, showscale=True, colorbar=dict(title="P&L $"),
        ),
        text=trail_valid["symbol"],
        hovertemplate="<b>%{text}</b><br>Peak: %{x:.1f}%<br>Exit: %{y:.1f}%<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=[0, trail_valid["peak_gain_pct"].max()],
        y=[0, trail_valid["peak_gain_pct"].max()],
        mode="lines", name="Capture 100%",
        line=dict(dash="dash", color="gray", width=1),
    ))
    fig.update_layout(
        xaxis_title="Gain au peak (%)", yaxis_title="Gain à la sortie (%)",
        height=350, margin=dict(l=0, r=0, t=10, b=0), showlegend=False,
    )
    st.plotly_chart(fig, width='stretch')


def _render_advanced_stats(closed: pd.DataFrame):
    if closed.empty or "pnl_usd" not in closed.columns:
        return
    st.subheader("📐 Statistiques avancées")

    wins = closed[closed["pnl_usd"] > 0]
    losses = closed[closed["pnl_usd"] <= 0]

    avg_win_val = wins["pnl_usd"].mean() if len(wins) > 0 else 0
    avg_loss_val = abs(losses["pnl_usd"].mean()) if len(losses) > 0 else 0
    payoff_ratio = avg_win_val / avg_loss_val if avg_loss_val > 0 else float("inf")

    cumulative = closed.sort_values("closed_at")["pnl_usd"].cumsum()
    max_dd = (cumulative - cumulative.cummax()).min()

    avg_holding = closed["holding_time_hours"].mean() if "holding_time_hours" in closed.columns else 0

    col_s1, col_s2, col_s3, col_s4 = st.columns(4)
    col_s1.metric("💵 Gain moyen", f"${avg_win_val:+.2f}")
    col_s2.metric("💸 Perte moyenne", f"${-avg_loss_val:-.2f}")
    col_s3.metric("📊 Payoff Ratio", f"{payoff_ratio:.2f}" if payoff_ratio != float("inf") else "∞")
    col_s4.metric("📉 Max Drawdown", f"${max_dd:+.2f}")

    streak_list = [1 if pnl > 0 else -1 for pnl in closed.sort_values("closed_at")["pnl_usd"]]
    max_win_streak = max_loss_streak = current_streak = 0
    for s in streak_list:
        if s > 0:
            current_streak = current_streak + 1 if current_streak > 0 else 1
            max_win_streak = max(max_win_streak, current_streak)
        else:
            current_streak = current_streak - 1 if current_streak < 0 else -1
            max_loss_streak = min(max_loss_streak, current_streak)

    total_fees = closed["fees_total"].sum() if "fees_total" in closed.columns else 0

    col_s5, col_s6, col_s7, col_s8 = st.columns(4)
    col_s5.metric("⏱️ Durée moyenne", f"{avg_holding:.1f}h")
    col_s6.metric("🔥 Série max wins", f"{max_win_streak}")
    col_s7.metric("❄️ Série max pertes", f"{abs(max_loss_streak)}")
    col_s8.metric("💳 Fees total", f"${total_fees:.2f}")


# ══════════════════════════════════════════════════════════════════════════════
#  TAB: Overview
# ══════════════════════════════════════════════════════════════════════════════

def render_overview():
    st.title("🏠 Overview")
    st.caption("Vue consolidée des 5 bots de trading")

    # ── Allocation (source de vérité pour l'equity par bot) ──────────────
    alloc = _fetch_current_allocation()

    # Mapping allocation → equity par bot
    alloc_equity = {}
    if alloc:
        alloc_equity["binance"] = alloc.get("trail_balance")
        alloc_equity["crashbot"] = alloc.get("crash_balance")
        alloc_equity["listing"] = alloc.get("listing_balance")
        alloc_equity["infinity"] = None  # Infinity est sur Revolut, pas dans l'allocation Binance
        alloc_equity["london"] = None    # London est sur Revolut, pas dans l'allocation Binance

    # ── Summary cards ──────────────────────────────────────────────────────
    cols = st.columns(len(BOTS))
    total_equity = 0.0
    total_pnl = 0.0
    total_trades = 0
    total_wins = 0
    total_open = 0

    for i, (key, cfg) in enumerate(BOTS.items()):
        d = all_data[key]
        stats = d["stats"]
        n_open = len(d["open"])
        total_open += n_open

        # Equity : allocation > snapshot > stats
        eq = alloc_equity.get(key)
        if eq is None:
            snap = d["snapshots"]
            if not snap.empty and "equity" in snap.columns:
                last_eq = snap.sort_values("date").iloc[-1].get("equity", None)
                if pd.notna(last_eq):
                    eq = float(last_eq)
        if eq is None and stats:
            eq = stats.get("equity")

        total_equity += eq or 0

        with cols[i]:
            st.markdown(f"### {cfg['icon']} {cfg['label']}")
            if stats:
                pnl = stats["total_pnl"]
                wr = stats["win_rate"]
                total_pnl += pnl
                total_trades += stats["n_trades"]
                total_wins += stats["n_wins"]

                st.metric("Equity", f"${eq:,.2f}" if eq else "—")
                st.metric("P&L", f"${pnl:+,.2f}")
                st.metric("Win Rate", f"{wr:.1f}%", f"{stats['n_wins']}W / {stats['n_losses']}L")
                st.metric("Positions", f"{n_open}/{cfg['max_pos']}")
            else:
                st.metric("Equity", f"${eq:,.2f}" if eq else "—")
                st.metric("P&L", "$0.00")
                st.metric("Win Rate", "— (0 trades)")
                st.metric("Positions", f"{n_open}/{cfg['max_pos']}")

    st.divider()

    # ── Allocation dynamique ───────────────────────────────────────────────
    if alloc:
        st.subheader("⚖️ Allocation dynamique")

        regime = alloc["regime"]
        regime_emoji = {
            AllocationRegime.DEFENSIVE: "🛡️",
            AllocationRegime.NEUTRAL: "⚖️",
            AllocationRegime.AGGRESSIVE: "🚀",
        }
        regime_color = {
            AllocationRegime.DEFENSIVE: "#ff9800",
            AllocationRegime.NEUTRAL: "#2196f3",
            AllocationRegime.AGGRESSIVE: "#4caf50",
        }

        a1, a2, a3, a4, a5, a6 = st.columns(6)
        a1.metric(
            "Régime",
            f"{regime_emoji.get(regime, '')} {regime.value.upper()}",
        )
        a2.metric("PF Trail Range (90j)", f"{alloc['trail_pf']:.2f}" if alloc['trail_pf'] != float('inf') else "∞")
        a3.metric("Trades Trail (90j)", f"{alloc['trail_trades']}")
        a4.metric(
            "💥 CrashBot",
            f"${alloc['crash_balance']:,.0f}",
            f"{alloc['crash_pct']*100:.0f}%",
        )
        a5.metric(
            "🟡 Trail Range",
            f"${alloc['trail_balance']:,.0f}",
            f"{alloc['trail_pct']*100:.0f}%",
        )
        listing_bal = alloc.get('listing_balance', 0) or 0
        listing_pct_val = alloc.get('listing_pct', 0.30) or 0.30
        a6.metric(
            "🆕 Listing",
            f"${listing_bal:,.0f}",
            f"{listing_pct_val*100:.0f}%",
        )

        # Gauge visuelle 3 segments
        crash_pct_val = alloc.get('crash_pct', 0.65) or 0.65
        trail_pct_val = alloc.get('trail_pct', 0.05) or 0.05
        fig_alloc = go.Figure(go.Bar(
            x=[crash_pct_val * 100, trail_pct_val * 100, listing_pct_val * 100],
            y=["Allocation", "Allocation", "Allocation"],
            orientation="h",
            marker_color=["#7c4dff", "#f0b90b", "#00e676"],
            text=[
                f"CrashBot {crash_pct_val*100:.0f}% (${alloc['crash_balance']:,.0f})",
                f"Trail {trail_pct_val*100:.0f}% (${alloc['trail_balance']:,.0f})",
                f"Listing {listing_pct_val*100:.0f}% (${listing_bal:,.0f})",
            ],
            textposition="inside",
            textfont=dict(color="white", size=14),
        ))
        fig_alloc.update_layout(
            barmode="stack",
            height=70,
            margin=dict(l=0, r=0, t=0, b=0),
            xaxis=dict(visible=False, range=[0, 100]),
            yaxis=dict(visible=False),
            showlegend=False,
        )
        st.plotly_chart(fig_alloc, width='stretch')

        st.caption(f"ℹ️ {alloc['reason']} — Total: ${alloc['total_balance']:,.0f}")

    st.divider()

    # ── Global KPIs ────────────────────────────────────────────────────────
    st.subheader("📊 Totaux consolidés")
    g1, g2, g3, g4, g5 = st.columns(5)
    global_wr = (total_wins / total_trades * 100) if total_trades > 0 else 0
    g1.metric("🏦 Equity totale", f"${total_equity:,.2f}" if total_equity > 0 else "—")
    g2.metric("💰 P&L global", f"${total_pnl:+,.2f}")
    g3.metric("📈 Trades total", f"{total_trades}")
    g4.metric("🎯 Win Rate global", f"{global_wr:.1f}%")
    g5.metric("📂 Positions actives", f"{total_open}")

    st.divider()

    # ── Equity curves comparison ───────────────────────────────────────────
    st.subheader("📈 Courbes d'equity comparées")

    fig_eq = go.Figure()
    for key, cfg in BOTS.items():
        snap = all_data[key]["snapshots"]
        if not snap.empty and "equity" in snap.columns:
            fig_eq.add_trace(go.Scatter(
                x=snap["date"], y=snap["equity"],
                mode="lines", name=cfg["label"],
                line=dict(color=cfg["color"], width=2),
            ))

    if fig_eq.data:
        fig_eq.update_layout(
            height=400,
            yaxis_tickformat="$,.0f",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        )
        _apply_mobile_chart_theme(fig_eq, show_legend=True)
        st.plotly_chart(fig_eq, width='stretch')
    else:
        st.info("Pas de données d'equity disponibles")

    # ── Daily P&L comparison ───────────────────────────────────────────────
    st.subheader("📅 P&L journalier combiné")

    all_daily = []
    for key, cfg in BOTS.items():
        closed = all_data[key]["closed"]
        if not closed.empty and "pnl_usd" in closed.columns:
            daily = closed.copy()
            daily["day"] = daily["closed_at"].dt.date
            daily_agg = daily.groupby("day")["pnl_usd"].sum().reset_index()
            daily_agg.columns = ["day", "pnl"]
            daily_agg["bot"] = cfg["label"]
            daily_agg["color"] = cfg["color"]
            all_daily.append(daily_agg)

    if all_daily:
        combined = pd.concat(all_daily, ignore_index=True)
        # P&L total par jour
        total_daily = combined.groupby("day")["pnl"].sum().reset_index()
        total_colors = ["#00c853" if v >= 0 else "#ff1744" for v in total_daily["pnl"]]

        fig_daily = go.Figure(go.Bar(
            x=total_daily["day"], y=total_daily["pnl"],
            marker_color=total_colors,
            text=[f"${v:+.2f}" for v in total_daily["pnl"]],
            textposition="outside",
        ))
        fig_daily.update_layout(
            height=300,
            yaxis_tickformat="$,.2f",
        )
        _apply_mobile_chart_theme(fig_daily, show_legend=False)
        st.plotly_chart(fig_daily, width='stretch')
    else:
        st.info("Pas de trades sur cette période")

    # ── All open positions ─────────────────────────────────────────────────
    st.subheader(f"🟢 Toutes les positions ouvertes ({total_open})")
    all_open = []
    for key, cfg in BOTS.items():
        o = all_data[key]["open"]
        if not o.empty:
            o = o.copy()
            o["bot"] = cfg["label"]
            # Enrichir avec les prix de la bonne source
            ex = "revolut" if cfg["exchange"] == "revolut" else "binance"
            o = _enrich_with_prices(o, exchange=ex)
            all_open.append(o)

    if all_open:
        combined_open = pd.concat(all_open, ignore_index=True)
        display_cols = ["bot", "symbol", "side", "entry_filled", "sl_price",
                        "current_price", "tp_price", "size_usd", "risk_usd",
                        "pnl_latent", "pnl_latent_usd", "status"]
        available = [c for c in display_cols if c in combined_open.columns]
        show = combined_open[available].copy()
        show = _format_positions_display(show)
        show = show.rename(columns={
            "bot": "Bot", "symbol": "Paire", "side": "Side",
            "entry_filled": "Entrée", "sl_price": "SL",
            "current_price": "💲 Prix actuel", "tp_price": "TP",
            "size_usd": "Notionnel $", "risk_usd": "Risque $",
            "pnl_latent": "P&L latent", "status": "Statut",
        })
        st.dataframe(show, width='stretch', hide_index=True)
    else:
        st.info("Aucune position ouverte sur les 4 bots")


# ══════════════════════════════════════════════════════════════════════════════
#  TAB: Binance Range
# ══════════════════════════════════════════════════════════════════════════════

def render_binance_range():
    d = all_data["binance"]
    cfg = BOTS["binance"]

    st.title(f"{cfg['icon']} Binance Range")
    _render_last_heartbeat_cockpit("binance", d, cfg["max_pos"])
    st.divider()
    _render_kpis(d["stats"], len(d["open"]), cfg["max_pos"])
    st.divider()

    _render_positions(d["open"])
    _render_alerts(d["open"], cfg["exchange"])
    st.divider()

    _render_equity_curve(d["snapshots"], d["closed"], cfg["color"])
    _render_cumulative_pnl(d["closed"], cfg["color"])
    st.divider()

    _render_daily_pnl(d["closed"])
    _render_pair_performance(d["closed"])
    st.divider()

    _render_last_trades(d["closed"])
    _render_pnl_distribution(d["closed"], cfg["color"], "strategy")


# ══════════════════════════════════════════════════════════════════════════════
#  TAB: Binance CrashBot
# ══════════════════════════════════════════════════════════════════════════════

def render_binance_crashbot():
    d = all_data["crashbot"]
    cfg = BOTS["crashbot"]

    st.title(f"{cfg['icon']} Binance CrashBot")
    _render_last_heartbeat_cockpit("crashbot", d, cfg["max_pos"])
    st.divider()
    _render_kpis(d["stats"], len(d["open"]), cfg["max_pos"])
    st.divider()

    _render_positions(d["open"], bot_type="crashbot")
    _render_kill_switch(cfg["exchange"])
    _render_alerts(d["open"], cfg["exchange"])
    st.divider()

    _render_equity_curve(d["snapshots"], d["closed"], cfg["color"])
    _render_cumulative_pnl(d["closed"], cfg["color"])
    st.divider()

    _render_daily_pnl(d["closed"])
    _render_pair_performance(d["closed"])
    st.divider()

    _render_exit_reasons(d["closed"])
    _render_trailing_stats(d["closed"])
    st.divider()

    _render_last_trades(d["closed"])
    _render_pnl_distribution(d["closed"], cfg["color"], "exit_reason")
    st.divider()

    _render_advanced_stats(d["closed"])


# ══════════════════════════════════════════════════════════════════════════════
#  TAB: Binance Listing Event
# ══════════════════════════════════════════════════════════════════════════════

def render_binance_listing():
    d = all_data["listing"]
    cfg = BOTS["listing"]

    st.title(f"{cfg['icon']} Binance Listing Event")
    st.caption(
        "Achat automatique des nouveaux listings USDC — momentum ≥30%, OCO dynamique, horizon 7j"
    )
    _render_last_heartbeat_cockpit("listing", d, cfg["max_pos"])
    st.divider()
    _render_kpis(d["stats"], len(d["open"]), cfg["max_pos"])
    st.divider()

    _render_positions(d["open"], bot_type="listing")
    _render_alerts(d["open"], cfg["exchange"])
    st.divider()

    _render_equity_curve(d["snapshots"], d["closed"], cfg["color"])
    _render_cumulative_pnl(d["closed"], cfg["color"])
    st.divider()

    _render_daily_pnl(d["closed"])
    _render_pair_performance(d["closed"])
    st.divider()

    _render_exit_reasons(d["closed"])
    st.divider()

    _render_last_trades(d["closed"])
    _render_pnl_distribution(d["closed"], cfg["color"], "exit_reason")
    st.divider()

    _render_advanced_stats(d["closed"])


# ══════════════════════════════════════════════════════════════════════════════
#  TAB: Revolut London Breakout
# ══════════════════════════════════════════════════════════════════════════════

def render_revolut_london():
    d = all_data["london"]
    cfg = BOTS["london"]

    st.title(f"{cfg['icon']} Revolut London Breakout")
    st.caption(
        f"Session breakout (08-16 UTC) — H4, maker-only, {app_config.LON_CAPITAL_PCT * 100:.0f}% du capital Revolut X"
    )
    _render_last_heartbeat_cockpit("london", d, cfg["max_pos"])
    st.divider()
    _render_kpis(d["stats"], len(d["open"]), cfg["max_pos"])
    st.divider()

    _render_positions(d["open"], bot_type="london", exchange="revolut")
    _render_alerts(d["open"], cfg["exchange"])
    st.divider()

    _render_equity_curve(d["snapshots"], d["closed"], cfg["color"])
    _render_cumulative_pnl(d["closed"], cfg["color"])
    st.divider()

    _render_daily_pnl(d["closed"])
    _render_pair_performance(d["closed"])
    st.divider()

    _render_exit_reasons(d["closed"])
    st.divider()

    _render_last_trades(d["closed"])
    _render_pnl_distribution(d["closed"], cfg["color"], "exit_reason")
    st.divider()

    _render_advanced_stats(d["closed"])


# ══════════════════════════════════════════════════════════════════════════════
#  Infinity V-Curve Visualization
# ══════════════════════════════════════════════════════════════════════════════

def _pfmt(v: float) -> str:
    """Format adaptatif pour les prix (XLM ~$0.15 vs BTC ~$68k)."""
    if v < 1:
        return f"${v:,.4f}"
    if v < 10:
        return f"${v:,.2f}"
    if v < 1000:
        return f"${v:,.2f}"
    return f"${v:,.0f}"

def _render_infinity_vcurve(cycle: dict | None):
    """Affiche la courbe en V du cycle DCA Infinity avec paliers d'achat/vente."""

    if not cycle or cycle.get("phase") == "WAITING":
        st.info("♾️ **Cycle en attente** — Le bot attend un drop de "
                f"{cycle.get('entry_drop_pct', 0.05) * 100:.0f}% depuis le trailing high "
                f"pour démarrer un nouveau cycle." if cycle else
                "♾️ **Aucun cycle actif** — En attente de données Firebase.")

        # Même en WAITING, on peut afficher les niveaux théoriques
        if cycle and cycle.get("trailing_high", 0) > 0:
            _render_vcurve_theoretical(cycle)
        return

    st.subheader(f"📊 Cycle actif — {cycle.get('symbol', '?')}")

    ref_price = cycle.get("reference_price", 0)
    pmp = cycle.get("pmp", 0)
    buy_levels = cycle.get("buy_levels", [-0.05, -0.10, -0.15, -0.20, -0.25])
    buy_pcts = cycle.get("buy_pcts", [0.25, 0.20, 0.15, 0.10, 0.00])
    sell_levels = cycle.get("sell_levels", [0.008, 0.015, 0.022, 0.030, 0.040])
    buys = cycle.get("buys", [])
    sells = cycle.get("sells", [])
    current_price = cycle.get("current_price", 0)
    stop_loss_pct = cycle.get("stop_loss_pct", 0.15)

    if ref_price <= 0:
        return

    # ── Construire les points de la courbe V ──
    fig = go.Figure()

    # ━━ Branche gauche : paliers d'achat (descente) ━━
    buy_x = []
    buy_y = []
    buy_texts = []
    bought_levels = {b["level"] for b in buys}

    for i, drop in enumerate(buy_levels):
        price_at_level = ref_price * (1 + drop)
        pct_label = f"{buy_pcts[i] * 100:.0f}%" if i < len(buy_pcts) else "?"
        buy_x.append(f"L{i + 1} ({drop * 100:+.0f}%)")
        buy_y.append(price_at_level)
        buy_texts.append(f"Buy L{i + 1}\n{_pfmt(price_at_level)}\n{pct_label} du capital")

    # Ligne des niveaux d'achat (gris pointillé)
    fig.add_trace(go.Scatter(
        x=buy_x, y=buy_y,
        mode="lines+markers",
        line=dict(color="rgba(150,150,150,0.4)", width=2, dash="dot"),
        marker=dict(size=12, color="rgba(150,150,150,0.3)", symbol="circle"),
        text=buy_texts,
        hoverinfo="text",
        name="Paliers d'achat (cibles)",
        showlegend=True,
    ))

    # Points d'achat exécutés (rouges/oranges)
    for b in buys:
        lvl = b.get("level", 0)
        x_label = f"L{lvl + 1} ({buy_levels[lvl] * 100:+.0f}%)" if lvl < len(buy_levels) else f"L{lvl + 1}"
        fig.add_trace(go.Scatter(
            x=[x_label], y=[b["price"]],
            mode="markers+text",
            marker=dict(size=18, color="#ff1744", symbol="circle",
                        line=dict(color="white", width=2)),
            text=[_pfmt(b['price'])],
            textposition="bottom center",
            textfont=dict(size=11, color="#ff1744"),
            hovertext=f"🔴 ACHAT L{lvl + 1}<br>{_pfmt(b['price'])}<br>"
                       f"Size: {b.get('size', 0):.6f} {cycle.get('symbol', '?').split('-')[0]}<br>"
                       f"Coût: ${b.get('cost', 0):,.2f}",
            hoverinfo="text",
            name=f"🔴 Achat L{lvl + 1}",
            showlegend=True,
        ))

    # ━━ Point central : PMP ━━
    if pmp > 0:
        fig.add_trace(go.Scatter(
            x=["PMP"], y=[pmp],
            mode="markers+text",
            marker=dict(size=22, color="#ff6d00", symbol="diamond",
                        line=dict(color="white", width=2)),
            text=[f"PMP\n{_pfmt(pmp)}"],
            textposition="top center",
            textfont=dict(size=12, color="#ff6d00", family="Arial Black"),
            hovertext=f"💎 PMP (Prix Moyen Pondéré)<br>${pmp:,.2f}<br>"
                       f"Total investi: ${cycle.get('total_cost', 0):,.2f}<br>"
                       f"{cycle.get('symbol', '?').split('-')[0]} restant: {cycle.get('size_remaining', 0):.6f}",
            hoverinfo="text",
            name="💎 PMP",
            showlegend=True,
        ))

    # ━━ Branche droite : paliers de vente (remontée) ━━
    sell_x = []
    sell_y = []
    sell_texts = []
    sold_levels = set(cycle.get("sell_levels_hit", []))

    if pmp > 0:
        for i, gain in enumerate(sell_levels):
            price_at_level = pmp * (1 + gain)
            sell_x.append(f"TP{i + 1} (+{gain * 100:.1f}%)")
            sell_y.append(price_at_level)
            sell_texts.append(f"Sell TP{i + 1}\n{_pfmt(price_at_level)}\n+{gain * 100:.1f}%")

        # Ligne des niveaux de vente (gris pointillé)
        fig.add_trace(go.Scatter(
            x=sell_x, y=sell_y,
            mode="lines+markers",
            line=dict(color="rgba(150,150,150,0.4)", width=2, dash="dot"),
            marker=dict(size=12, color="rgba(150,150,150,0.3)", symbol="circle"),
            text=sell_texts,
            hoverinfo="text",
            name="Paliers de vente (cibles)",
            showlegend=True,
        ))

    # Points de vente exécutés (verts)
    for s in sells:
        lvl = s.get("level", 0)
        x_label = f"TP{lvl + 1} (+{sell_levels[lvl] * 100:.1f}%)" if lvl < len(sell_levels) else f"TP{lvl + 1}"
        fig.add_trace(go.Scatter(
            x=[x_label], y=[s["price"]],
            mode="markers+text",
            marker=dict(size=18, color="#00e676", symbol="circle",
                        line=dict(color="white", width=2)),
            text=[_pfmt(s['price'])],
            textposition="top center",
            textfont=dict(size=11, color="#00e676"),
            hovertext=f"🟢 VENTE TP{lvl + 1}<br>{_pfmt(s['price'])}<br>"
                       f"Size: {s.get('size', 0):.6f} {cycle.get('symbol', '?').split('-')[0]}<br>"
                       f"Revenus: ${s.get('proceeds', 0):,.2f}",
            hoverinfo="text",
            name=f"🟢 Vente TP{lvl + 1}",
            showlegend=True,
        ))

    # ━━ Prix actuel (ligne horizontale) ━━
    if current_price > 0:
        all_x = buy_x + (["PMP"] if pmp > 0 else []) + sell_x
        fig.add_hline(
            y=current_price,
            line_dash="dash", line_color="#2196f3", line_width=1.5,
            annotation_text=f"Prix actuel: {_pfmt(current_price)}",
            annotation_position="top right",
            annotation_font_color="#2196f3",
        )

    # ━━ Stop-loss (ligne rouge) ━━
    if pmp > 0:
        sl_price = pmp * (1 - stop_loss_pct)
        fig.add_hline(
            y=sl_price,
            line_dash="dot", line_color="#ff1744", line_width=1,
            annotation_text=f"SL: {_pfmt(sl_price)} (-{stop_loss_pct * 100:.0f}%)",
            annotation_position="bottom right",
            annotation_font_color="#ff1744",
        )

    # ━━ Breakeven (ligne orange) ━━
    if cycle.get("breakeven_active") and pmp > 0:
        fig.add_hline(
            y=pmp,
            line_dash="dash", line_color="#ff6d00", line_width=1,
            annotation_text="🔒 Breakeven actif",
            annotation_position="top left",
            annotation_font_color="#ff6d00",
        )

    # ━━ Trailing high (ligne haute) ━━
    trailing_high = cycle.get("trailing_high", 0)
    if trailing_high > 0:
        fig.add_hline(
            y=trailing_high,
            line_dash="dot", line_color="rgba(255,255,255,0.3)", line_width=1,
            annotation_text=f"Trail High: {_pfmt(trailing_high)}",
            annotation_position="top right",
            annotation_font_color="rgba(255,255,255,0.5)",
        )

    # ━━ Layout ━━
    fig.update_layout(
        title=dict(
            text=f"♾️ {cycle.get('symbol', '?')} — Cycle #{cycle.get('cycle_count', 0)} — {cycle.get('phase', '?')}",
            font=dict(size=18),
        ),
        xaxis=dict(
            title="Paliers",
            showgrid=False,
            categoryorder="array",
            categoryarray=buy_x + (["PMP"] if pmp > 0 else []) + sell_x,
        ),
        yaxis=dict(
            title=f"Prix {cycle.get('symbol', '').split('-')[0]} ($)",
            showgrid=True,
            gridcolor="rgba(255,255,255,0.1)",
            tickformat="$,.2f" if (pmp > 0 and pmp < 10) else "$,.0f",
        ),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        height=500,
        legend=dict(
            orientation="h",
            yanchor="bottom", y=1.02,
            xanchor="right", x=1,
            font=dict(size=10),
        ),
        margin=dict(l=60, r=20, t=80, b=60),
    )

    st.plotly_chart(fig, width='stretch')

    # ── Métriques du cycle ──
    if pmp > 0:
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Buys", f"{len(buys)}/5")
        c2.metric("Sells", f"{len(sells)}/5")
        pmp_fmt = f"${pmp:,.4f}" if pmp < 10 else f"${pmp:,.2f}" if pmp < 1000 else f"${pmp:,.0f}"
        c3.metric("PMP", pmp_fmt)
        invested = cycle.get("total_cost", 0)
        c4.metric("Investi", f"${invested:,.2f}")
        if current_price > 0 and invested > 0:
            latent = (current_price - pmp) / pmp * 100
            c5.metric("P&L latent", f"{latent:+.1f}%")
        else:
            c5.metric("P&L latent", "—")


def _render_vcurve_theoretical(cycle: dict):
    """Affiche les niveaux théoriques quand le cycle est en attente."""
    trailing_high = cycle.get("trailing_high", 0)
    entry_drop = cycle.get("entry_drop_pct", 0.05)
    buy_levels = cycle.get("buy_levels", [-0.05, -0.10, -0.15, -0.20, -0.25])
    sell_levels = cycle.get("sell_levels", [0.008, 0.015, 0.022, 0.030, 0.040])
    current_price = cycle.get("current_price", 0)
    target = cycle.get("target_entry", trailing_high * (1 - entry_drop))

    st.subheader("📐 Niveaux théoriques (prochain cycle)")

    col1, col2, col3 = st.columns(3)
    col1.metric("Trail High", _pfmt(trailing_high))
    col2.metric("🎯 Cible entrée", _pfmt(target), f"-{entry_drop * 100:.0f}%")
    if current_price > 0:
        gap = (current_price - target) / current_price * 100
        col3.metric("Prix actuel", _pfmt(current_price), f"{gap:+.1f}% de la cible")
    else:
        col3.metric("Prix actuel", "—")

    # Table des niveaux
    rows = []
    for i, drop in enumerate(buy_levels):
        price = trailing_high * (1 + drop)
        rows.append({
            "Palier": f"L{i + 1}",
            "Drop": f"{drop * 100:+.0f}%",
            "Prix": _pfmt(price),
        })
    st.dataframe(pd.DataFrame(rows), width='stretch', hide_index=True)


# ══════════════════════════════════════════════════════════════════════════════
#  TAB: Revolut Infinity
# ══════════════════════════════════════════════════════════════════════════════

def render_revolut_infinity():
    d = all_data["infinity"]
    cfg = BOTS["infinity"]

    st.title(f"{cfg['icon']} Revolut Infinity")
    st.caption(
        f"DCA inversé multi-paires (BTC, AAVE, XLM, ADA, DOT, LTC) — H4, maker-only, {app_config.INF_CAPITAL_PCT * 100:.0f}% du capital Revolut X"
    )

    _render_last_heartbeat_cockpit("infinity", d, cfg["max_pos"])
    st.divider()

    _render_kpis(d["stats"], len(d["open"]), cfg["max_pos"])
    st.divider()

    # ── V-Curves per pair ──
    cycles = _fetch_infinity_cycles()
    if cycles:
        pair_tabs = st.tabs([f"{sym}" for sym in cycles.keys()])
        for tab, (sym, cycle) in zip(pair_tabs, cycles.items()):
            with tab:
                _render_infinity_vcurve(cycle)
    else:
        st.info("♾️ Aucun cycle actif — En attente de données Firebase.")
    st.divider()

    # ── Cycle actif (positions ouvertes) ──
    _render_positions(d["open"], bot_type="infinity", exchange="revolut")
    _render_alerts(d["open"], cfg["exchange"])
    st.divider()

    _render_equity_curve(d["snapshots"], d["closed"], cfg["color"])
    _render_cumulative_pnl(d["closed"], cfg["color"])
    st.divider()

    _render_daily_pnl(d["closed"])
    st.divider()

    _render_exit_reasons(d["closed"])
    st.divider()

    _render_last_trades(d["closed"])
    _render_pnl_distribution(d["closed"], cfg["color"], "exit_reason")
    st.divider()

    _render_advanced_stats(d["closed"])


# ══════════════════════════════════════════════════════════════════════════════
#  Main — Tabs
# ══════════════════════════════════════════════════════════════════════════════

tab_overview, tab_binance, tab_crashbot, tab_listing, tab_infinity, tab_london = st.tabs([
    "🏠 Overview",
    "🟡 Binance Range",
    "💥 Binance CrashBot",
    "🆕 Binance Listing",
    "♾️ Revolut Infinity",
    "🇬🇧 Revolut London",
])

with tab_overview:
    render_overview()

with tab_binance:
    render_binance_range()

with tab_crashbot:
    render_binance_crashbot()

with tab_listing:
    render_binance_listing()

with tab_infinity:
    render_revolut_infinity()

with tab_london:
    render_revolut_london()

# ── Footer ─────────────────────────────────────────────────────────────────────

st.divider()
st.caption(
    f"TradeX Unified Dashboard • Firebase (satochi-d38ec) • "
    f"Dernière mise à jour : {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}"
)
