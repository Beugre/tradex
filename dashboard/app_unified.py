"""
TradeX Unified Dashboard — Overview + 3 onglets (Revolut · Binance Range · Binance CrashBot).
Un seul processus Streamlit, port 8502.
Lit les données depuis Firebase Firestore.
"""

from __future__ import annotations

import os
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from google.cloud import firestore
from google.oauth2 import service_account

# Allocator (pure logic — pas d'I/O)
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.core.allocator import (
    AllocationRegime,
    compute_allocation,
    compute_profit_factor,
)

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
            df[col] = pd.to_datetime(df[col], errors="coerce", utc=True)
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
    df["date"] = pd.to_datetime(df["date"])
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


# ══════════════════════════════════════════════════════════════════════════════
#  Helpers
# ══════════════════════════════════════════════════════════════════════════════

def _color_pnl(val):
    if pd.isna(val):
        return ""
    return "color: #00c853" if val >= 0 else "color: #ff1744"


def _compute_current_allocation() -> dict:
    """Calcule l'allocation courante à afficher dans le dashboard."""
    try:
        db = _get_db()
        since = datetime.now(timezone.utc) - timedelta(days=90)
        docs = (
            db.collection("trades")
            .where("exchange", "==", "binance")
            .where("status", "==", "CLOSED")
            .where("closed_at", ">=", since.isoformat())
            .stream()
        )
        pnl_list = []
        for doc in docs:
            d = doc.to_dict()
            pnl = d.get("pnl_net_usd") or d.get("pnl_usd") or 0
            if isinstance(pnl, (int, float)):
                pnl_list.append(float(pnl))

        trail_pf = compute_profit_factor(pnl_list)
        trail_trades = len(pnl_list)

        # On utilise la somme des equities des snapshots les plus récents comme proxy
        # du total balance, sinon on utilise les stats
        total_balance = 0.0
        for key in ("binance", "crashbot"):
            snap = all_data[key]["snapshots"]
            if not snap.empty and "equity" in snap.columns:
                last_eq = snap.sort_values("date").iloc[-1]["equity"]
                if pd.notna(last_eq):
                    total_balance += float(last_eq)

        if total_balance <= 0:
            total_balance = 2350.0  # fallback

        result = compute_allocation(total_balance, trail_pf, trail_trades)
        return {
            "regime": result.regime,
            "crash_pct": result.crash_pct,
            "trail_pct": result.trail_pct,
            "crash_balance": result.crash_balance,
            "trail_balance": result.trail_balance,
            "trail_pf": result.trail_pf,
            "trail_trades": result.trail_trades,
            "reason": result.reason,
            "total_balance": total_balance,
        }
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
st.sidebar.caption("Dashboard unifié — 3 bots")

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
    "revolut": {"label": "Revolut Range", "icon": "⚫", "exchange": "revolut", "color": "#2196f3", "max_pos": 3},
    "binance": {"label": "Binance Range", "icon": "🟡", "exchange": "binance", "color": "#f0b90b", "max_pos": 3},
    "crashbot": {"label": "Binance CrashBot", "icon": "💥", "exchange": "binance-crashbot", "color": "#7c4dff", "max_pos": 5},
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
        }
    return data


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


def _render_positions(open_df: pd.DataFrame, bot_type: str = "range"):
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
        if bot_type == "crashbot":
            display_cols = [
                "symbol", "side", "entry_filled", "sl_price", "trail_tp",
                "trail_steps", "peak_price", "trail_gain_pct",
                "size", "size_usd", "risk_usd",
                "status", "opened_at",
            ]
        else:
            display_cols = [
                "symbol", "side", "entry_filled", "sl_price", "tp_price",
                "size", "size_usd", "risk_usd", "strategy",
            ]
            # Ajouter les colonnes trailing si elles existent
            if has_trailing:
                display_cols.insert(5, "trailing_steps")
            display_cols += ["status", "opened_at"]

        available_cols = [c for c in display_cols if c in open_df.columns]
        show_df = open_df[available_cols].copy()

        # Adapter le label SL selon le type de bot
        sl_label = "Trail SL" if bot_type == "crashbot" else ("SL 🔄" if has_trailing else "SL")
        rename_map = {
            "symbol": "Paire", "side": "Side", "entry_filled": "Entrée",
            "sl_price": sl_label,
            "tp_price": "TP", "trail_tp": "TP Cible",
            "peak_price": "Peak",
            "trail_gain_pct": "Gain %", "size": "Taille",
            "trailing_steps": "Trail Step", "trail_steps": "Steps",
            "size_usd": "Notionnel $", "risk_usd": "Risque $",
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

        st.dataframe(show_df, use_container_width=True, hide_index=True)
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
                use_container_width=True, hide_index=True,
            )


def _render_equity_curve(snapshots_df: pd.DataFrame, closed: pd.DataFrame, color: str):
    """Affiche la courbe d'equity."""
    st.subheader("📈 Courbe d'equity")

    if not snapshots_df.empty and "equity" in snapshots_df.columns:
        fig = px.area(
            snapshots_df, x="date", y="equity",
            labels={"date": "Date", "equity": "Equity ($)"},
            color_discrete_sequence=[color],
        )
        fig.update_layout(
            height=350, margin=dict(l=0, r=0, t=10, b=0),
            yaxis_tickformat="$,.0f", hovermode="x unified",
        )
        st.plotly_chart(fig, use_container_width=True)
    elif not closed.empty and "pnl_usd" in closed.columns:
        eq_from_field = (
            closed.dropna(subset=["equity_after"]).sort_values("closed_at")
            if "equity_after" in closed.columns else pd.DataFrame()
        )
        if not eq_from_field.empty:
            fig = px.area(
                eq_from_field, x="closed_at", y="equity_after",
                labels={"closed_at": "Date", "equity_after": "Equity ($)"},
                color_discrete_sequence=[color],
            )
        else:
            eq_df = closed.sort_values("closed_at").copy()
            initial_equity = eq_df.iloc[0].get("equity_at_entry", 1000) or 1000
            eq_df["equity_curve"] = initial_equity + eq_df["pnl_usd"].cumsum()
            fig = px.area(
                eq_df, x="closed_at", y="equity_curve",
                labels={"closed_at": "Date", "equity_curve": "Equity ($)"},
                color_discrete_sequence=[color],
            )
        fig.update_layout(
            height=350, margin=dict(l=0, r=0, t=10, b=0),
            yaxis_tickformat="$,.0f", hovermode="x unified",
        )
        st.plotly_chart(fig, use_container_width=True)
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
        mode="lines", name="P&L cumulé",
        line=dict(color=color, width=2),
        fill="tozeroy",
        fillcolor=f"rgba({int(color[1:3],16)},{int(color[3:5],16)},{int(color[5:7],16)},0.1)",
    ))
    fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)
    fig.update_layout(
        height=300, margin=dict(l=0, r=0, t=10, b=0),
        yaxis_tickformat="$,.2f", hovermode="x unified",
    )
    st.plotly_chart(fig, use_container_width=True)


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
    st.plotly_chart(fig, use_container_width=True)


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
        st.plotly_chart(fig, use_container_width=True)

    with col_right:
        st.dataframe(
            pair_agg.rename(columns={
                "symbol": "Paire", "pnl": "P&L ($)", "trades": "Trades",
                "wins": "Wins", "avg_pnl": "Moy ($)",
            }).sort_values("P&L ($)", ascending=False).style.map(
                _color_pnl, subset=["P&L ($)", "Moy ($)"]
            ).format({"P&L ($)": "${:+,.2f}", "Moy ($)": "${:+,.2f}"}),
            use_container_width=True, hide_index=True,
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

    st.dataframe(styled, use_container_width=True, hide_index=True)


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
        st.plotly_chart(fig_hist, use_container_width=True)

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
        st.plotly_chart(fig_box, use_container_width=True)


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
        st.plotly_chart(fig, use_container_width=True)

    with col_table:
        st.dataframe(
            reason_agg.rename(columns={
                "exit_reason": "Raison", "count": "Trades",
                "total_pnl": "P&L Total ($)", "avg_pnl": "P&L Moy ($)",
            }).style.map(
                _color_pnl, subset=["P&L Total ($)", "P&L Moy ($)"]
            ).format({"P&L Total ($)": "${:+,.2f}", "P&L Moy ($)": "${:+,.2f}"}),
            use_container_width=True, hide_index=True,
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
    st.plotly_chart(fig, use_container_width=True)


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
    st.caption("Vue consolidée des 3 bots de trading")

    # ── Summary cards ──────────────────────────────────────────────────────
    cols = st.columns(3)
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

        with cols[i]:
            st.markdown(f"### {cfg['icon']} {cfg['label']}")
            if stats:
                eq = stats["equity"]
                pnl = stats["total_pnl"]
                wr = stats["win_rate"]
                total_equity += eq or 0
                total_pnl += pnl
                total_trades += stats["n_trades"]
                total_wins += stats["n_wins"]

                st.metric("Equity", f"${eq:,.2f}" if eq else "—")
                st.metric("P&L", f"${pnl:+,.2f}")
                st.metric("Win Rate", f"{wr:.1f}%", f"{stats['n_wins']}W / {stats['n_losses']}L")
                st.metric("Positions", f"{n_open}/{cfg['max_pos']}")
            else:
                st.metric("Equity", "—")
                st.metric("P&L", "—")
                st.metric("Win Rate", "—")
                st.metric("Positions", f"{n_open}/{cfg['max_pos']}")

    st.divider()

    # ── Allocation dynamique ───────────────────────────────────────────────
    alloc = _compute_current_allocation()
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

        a1, a2, a3, a4, a5 = st.columns(5)
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

        # Gauge visuelle
        fig_alloc = go.Figure(go.Bar(
            x=[alloc["crash_pct"] * 100, alloc["trail_pct"] * 100],
            y=["Allocation", "Allocation"],
            orientation="h",
            marker_color=["#7c4dff", "#f0b90b"],
            text=[
                f"CrashBot {alloc['crash_pct']*100:.0f}% (${alloc['crash_balance']:,.0f})",
                f"Trail Range {alloc['trail_pct']*100:.0f}% (${alloc['trail_balance']:,.0f})",
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
        st.plotly_chart(fig_alloc, use_container_width=True)

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
            height=400, margin=dict(l=0, r=0, t=10, b=0),
            yaxis_tickformat="$,.0f", hovermode="x unified",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        )
        st.plotly_chart(fig_eq, use_container_width=True)
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
            height=300, margin=dict(l=0, r=0, t=10, b=0),
            yaxis_tickformat="$,.2f",
        )
        st.plotly_chart(fig_daily, use_container_width=True)
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
            all_open.append(o)

    if all_open:
        combined_open = pd.concat(all_open, ignore_index=True)
        display_cols = ["bot", "symbol", "side", "entry_filled", "sl_price",
                        "tp_price", "size_usd", "risk_usd", "status"]
        available = [c for c in display_cols if c in combined_open.columns]
        show = combined_open[available].rename(columns={
            "bot": "Bot", "symbol": "Paire", "side": "Side",
            "entry_filled": "Entrée", "sl_price": "SL", "tp_price": "TP",
            "size_usd": "Notionnel $", "risk_usd": "Risque $", "status": "Statut",
        })
        st.dataframe(show, use_container_width=True, hide_index=True)
    else:
        st.info("Aucune position ouverte sur les 3 bots")


# ══════════════════════════════════════════════════════════════════════════════
#  TAB: Revolut Range
# ══════════════════════════════════════════════════════════════════════════════

def render_revolut():
    d = all_data["revolut"]
    cfg = BOTS["revolut"]

    st.title(f"{cfg['icon']} Revolut Range")
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
#  TAB: Binance Range
# ══════════════════════════════════════════════════════════════════════════════

def render_binance_range():
    d = all_data["binance"]
    cfg = BOTS["binance"]

    st.title(f"{cfg['icon']} Binance Range")
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
#  Main — Tabs
# ══════════════════════════════════════════════════════════════════════════════

tab_overview, tab_revolut, tab_binance, tab_crashbot = st.tabs([
    "🏠 Overview",
    "⚫ Revolut Range",
    "🟡 Binance Range",
    "💥 Binance CrashBot",
])

with tab_overview:
    render_overview()

with tab_revolut:
    render_revolut()

with tab_binance:
    render_binance_range()

with tab_crashbot:
    render_binance_crashbot()

# ── Footer ─────────────────────────────────────────────────────────────────────

st.divider()
st.caption(
    f"TradeX Unified Dashboard • Firebase (satochi-d38ec) • "
    f"Dernière mise à jour : {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}"
)
