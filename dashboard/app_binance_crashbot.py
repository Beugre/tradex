"""
TradeX Binance CRASHBOT Dashboard — Monitoring du bot CrashBot (Dip Buy).
Lit les données depuis Firebase Firestore avec filtre exchange="binance-crashbot".
Port: 8504 (remplace le dashboard Breakout décommissionné).
"""

from __future__ import annotations

import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from google.cloud import firestore
from google.oauth2 import service_account

# ── Config ─────────────────────────────────────────────────────────────────────

_BASE = Path(__file__).resolve().parent.parent
_CRED_PATH = os.getenv(
    "FIREBASE_CREDENTIALS_PATH",
    str(_BASE / "firebase-credentials.json"),
)

EXCHANGE = "binance-crashbot"

st.set_page_config(
    page_title="TradeX CrashBot Dashboard",
    page_icon="💥",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ── Firebase connection (cached) ───────────────────────────────────────────────

@st.cache_resource
def _get_db() -> firestore.Client:
    cred = service_account.Credentials.from_service_account_file(_CRED_PATH)
    return firestore.Client(project=cred.project_id, credentials=cred)


def _fetch_trades(days: int = 90) -> pd.DataFrame:
    """Récupère les trades CrashBot depuis Firestore."""
    db = _get_db()
    since = datetime.now(timezone.utc) - timedelta(days=days)
    docs = (
        db.collection("trades")
        .where("exchange", "==", EXCHANGE)
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


def _fetch_daily_snapshots(days: int = 90) -> pd.DataFrame:
    """Récupère les snapshots journaliers CrashBot."""
    db = _get_db()
    since = (datetime.now(timezone.utc) - timedelta(days=days)).strftime("%Y-%m-%d")
    docs = (
        db.collection("daily_snapshots")
        .where("exchange", "==", EXCHANGE)
        .where("date", ">=", since)
        .order_by("date")
        .stream()
    )
    rows = [doc.to_dict() for doc in docs]
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"])
    return df


def _fetch_open_positions() -> pd.DataFrame:
    """Récupère les trades CrashBot ouverts."""
    db = _get_db()
    docs = (
        db.collection("trades")
        .where("exchange", "==", EXCHANGE)
        .where("status", "in", ["OPEN", "ZERO_RISK"])
        .stream()
    )
    rows = [doc.to_dict() | {"_id": doc.id} for doc in docs]
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows)


def _fetch_events(event_type: str | None = None, hours: int = 48) -> pd.DataFrame:
    """Récupère les événements CrashBot récents."""
    db = _get_db()
    cutoff = (datetime.utcnow() - timedelta(hours=hours)).isoformat()
    query = (
        db.collection("events")
        .where("exchange", "==", EXCHANGE)
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


# ── Helpers ────────────────────────────────────────────────────────────────────

def _color_pnl(val):
    if pd.isna(val):
        return ""
    return "color: #00c853" if val >= 0 else "color: #ff1744"


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


# ── Sidebar ────────────────────────────────────────────────────────────────────

st.sidebar.image("https://img.icons8.com/color/96/flash-bang.png", width=60)
st.sidebar.title("TradeX · CrashBot")
st.sidebar.caption("Bot USDC — Dip Buy (crash -20%) · Long Only · Step Trail")

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


# ── Load data ──────────────────────────────────────────────────────────────────

trades_df = _fetch_trades(days=days_filter)
snapshots_df = _fetch_daily_snapshots(days=days_filter)
open_df = _fetch_open_positions()


# ── Header KPIs ────────────────────────────────────────────────────────────────

st.title("💥 TradeX CrashBot Dashboard")

closed = trades_df[trades_df["closed_at"].notna()].copy() if not trades_df.empty else pd.DataFrame()

col1, col2, col3, col4, col5, col6 = st.columns(6)

if not closed.empty and "pnl_usd" in closed.columns:
    total_pnl = closed["pnl_usd"].sum()
    win_trades = closed[closed["pnl_usd"] > 0]
    loss_trades = closed[closed["pnl_usd"] <= 0]
    win_rate = len(win_trades) / len(closed) * 100 if len(closed) > 0 else 0
    avg_win = win_trades["pnl_usd"].mean() if len(win_trades) > 0 else 0
    avg_loss = loss_trades["pnl_usd"].mean() if len(loss_trades) > 0 else 0
    profit_factor = abs(win_trades["pnl_usd"].sum() / loss_trades["pnl_usd"].sum()) if len(loss_trades) > 0 and loss_trades["pnl_usd"].sum() != 0 else float("inf")

    equity = closed.sort_values("closed_at").iloc[-1].get("equity_after", None)
    if pd.isna(equity):
        equity = None

    col1.metric("💰 P&L Total", f"${total_pnl:+,.2f}")
    col2.metric("📈 Trades", f"{len(closed)}", f"{len(win_trades)}W / {len(loss_trades)}L")
    col3.metric("🎯 Win Rate", f"{win_rate:.1f}%")
    col4.metric("⚖️ Profit Factor", f"{profit_factor:.2f}" if profit_factor != float("inf") else "∞")
    col5.metric("🏦 Equity", f"${equity:,.2f}" if equity else "—")
    col6.metric("📂 Positions", f"{len(open_df)}")
else:
    col1.metric("💰 P&L Total", "—")
    col2.metric("📈 Trades", "0")
    col3.metric("🎯 Win Rate", "—")
    col4.metric("⚖️ Profit Factor", "—")
    col5.metric("🏦 Equity", "—")
    col6.metric("📂 Positions", f"{len(open_df)}")

st.divider()


# ── Positions ouvertes (CrashBot — avec step trailing info) ────────────────────

st.subheader("🟢 Positions ouvertes — Step Trailing")

if not open_df.empty:
    display_cols = [
        "symbol", "side", "entry_filled", "sl_price", "trail_tp",
        "trail_steps", "peak_price", "trail_gain_pct",
        "size", "size_usd", "risk_usd", "status", "opened_at",
    ]
    available_cols = [c for c in display_cols if c in open_df.columns]
    show_df = open_df[available_cols].copy()

    rename_map = {
        "symbol": "Paire", "side": "Side", "entry_filled": "Entrée",
        "sl_price": "Trail SL", "trail_tp": "TP Cible",
        "trail_steps": "Steps", "peak_price": "Peak",
        "trail_gain_pct": "Gain %", "size": "Taille",
        "size_usd": "Notionnel $", "risk_usd": "Risque $",
        "status": "Statut", "opened_at": "Ouvert le",
    }
    show_df = show_df.rename(columns={k: v for k, v in rename_map.items() if k in show_df.columns})

    if "Gain %" in show_df.columns:
        show_df["Gain %"] = show_df["Gain %"].apply(
            lambda x: f"{x * 100:+.2f}%" if pd.notna(x) else "—"
        )

    st.dataframe(show_df, use_container_width=True, hide_index=True)
else:
    st.info("Aucune position CrashBot ouverte")


# ── Kill-switch status ─────────────────────────────────────────────────────────

kill_events = _fetch_events("KILL_SWITCH", hours=720)
if not kill_events.empty:
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

st.divider()


# ── Equity curve ───────────────────────────────────────────────────────────────

st.subheader("📈 Courbe d'equity")

if not snapshots_df.empty and "equity" in snapshots_df.columns:
    fig_eq = px.area(
        snapshots_df, x="date", y="equity",
        labels={"date": "Date", "equity": "Equity ($)"},
        color_discrete_sequence=["#7c4dff"],  # Violet crashbot
    )
    fig_eq.update_layout(
        height=350, margin=dict(l=0, r=0, t=10, b=0),
        yaxis_tickformat="$,.0f",
        hovermode="x unified",
    )
    st.plotly_chart(fig_eq, use_container_width=True)
elif not closed.empty and "pnl_usd" in closed.columns:
    eq_from_field = closed.dropna(subset=["equity_after"]).sort_values("closed_at") if "equity_after" in closed.columns else pd.DataFrame()
    if not eq_from_field.empty:
        fig_eq = px.area(
            eq_from_field, x="closed_at", y="equity_after",
            labels={"closed_at": "Date", "equity_after": "Equity ($)"},
            color_discrete_sequence=["#7c4dff"],
        )
        fig_eq.update_layout(
            height=350, margin=dict(l=0, r=0, t=10, b=0),
            yaxis_tickformat="$,.0f",
            hovermode="x unified",
        )
        st.plotly_chart(fig_eq, use_container_width=True)
    else:
        eq_df = closed.sort_values("closed_at").copy()
        initial_equity = eq_df.iloc[0].get("equity_at_entry", 1000) or 1000
        eq_df["equity_curve"] = initial_equity + eq_df["pnl_usd"].cumsum()
        fig_eq = px.area(
            eq_df, x="closed_at", y="equity_curve",
            labels={"closed_at": "Date", "equity_curve": "Equity ($)"},
            color_discrete_sequence=["#7c4dff"],
        )
        fig_eq.update_layout(
            height=350, margin=dict(l=0, r=0, t=10, b=0),
            yaxis_tickformat="$,.0f",
            hovermode="x unified",
        )
        st.plotly_chart(fig_eq, use_container_width=True)
else:
    st.info("Pas de données d'equity disponibles")


# ── P&L cumulé ─────────────────────────────────────────────────────────────────

if not closed.empty and "pnl_usd" in closed.columns:
    st.subheader("💵 P&L cumulé")
    pnl_df = closed.sort_values("closed_at").copy()
    pnl_df["cumulative_pnl"] = pnl_df["pnl_usd"].cumsum()

    fig_pnl = go.Figure()
    fig_pnl.add_trace(go.Scatter(
        x=pnl_df["closed_at"], y=pnl_df["cumulative_pnl"],
        mode="lines", name="P&L cumulé",
        line=dict(color="#7c4dff", width=2),
        fill="tozeroy",
        fillcolor="rgba(124,77,255,0.1)",
    ))
    fig_pnl.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)
    fig_pnl.update_layout(
        height=300, margin=dict(l=0, r=0, t=10, b=0),
        yaxis_tickformat="$,.2f",
        hovermode="x unified",
    )
    st.plotly_chart(fig_pnl, use_container_width=True)

st.divider()


# ── Performance par jour ───────────────────────────────────────────────────────

if not closed.empty and "pnl_usd" in closed.columns:
    st.subheader("📅 P&L journalier")

    daily_pnl = closed.copy()
    daily_pnl["day"] = daily_pnl["closed_at"].dt.date
    daily_agg = daily_pnl.groupby("day").agg(
        pnl=("pnl_usd", "sum"),
        trades=("pnl_usd", "count"),
        wins=("pnl_usd", lambda x: (x > 0).sum()),
    ).reset_index()

    colors = ["#00c853" if v >= 0 else "#ff1744" for v in daily_agg["pnl"]]

    fig_daily = go.Figure(go.Bar(
        x=daily_agg["day"], y=daily_agg["pnl"],
        marker_color=colors,
        text=[f"${v:+.2f}" for v in daily_agg["pnl"]],
        textposition="outside",
        hovertemplate="<b>%{x}</b><br>P&L: $%{y:+.2f}<br>Trades: %{customdata[0]}<br>Wins: %{customdata[1]}<extra></extra>",
        customdata=daily_agg[["trades", "wins"]].values,
    ))
    fig_daily.update_layout(
        height=300, margin=dict(l=0, r=0, t=10, b=0),
        yaxis_tickformat="$,.2f",
    )
    st.plotly_chart(fig_daily, use_container_width=True)


# ── Performance par paire ──────────────────────────────────────────────────────

if not closed.empty and "pnl_usd" in closed.columns:
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
        fig_pair = go.Figure(go.Bar(
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
        fig_pair.update_layout(
            height=max(300, len(pair_agg) * 28),
            margin=dict(l=0, r=0, t=10, b=0),
            xaxis_tickformat="$,.2f",
        )
        st.plotly_chart(fig_pair, use_container_width=True)

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

st.divider()


# ── Exit reasons breakdown ─────────────────────────────────────────────────────

if not closed.empty and "exit_reason" in closed.columns:
    st.subheader("🏁 Raisons de sortie")

    col_pie, col_table = st.columns(2)

    reason_agg = closed.groupby("exit_reason").agg(
        count=("pnl_usd", "count"),
        total_pnl=("pnl_usd", "sum"),
        avg_pnl=("pnl_usd", "mean"),
    ).reset_index().sort_values("count", ascending=False)

    with col_pie:
        fig_pie = px.pie(
            reason_agg, names="exit_reason", values="count",
            color_discrete_sequence=px.colors.qualitative.Set2,
            hole=0.4,
        )
        fig_pie.update_layout(
            height=300, margin=dict(l=0, r=0, t=10, b=0),
        )
        st.plotly_chart(fig_pie, use_container_width=True)

    with col_table:
        st.dataframe(
            reason_agg.rename(columns={
                "exit_reason": "Raison", "count": "Trades",
                "total_pnl": "P&L Total ($)", "avg_pnl": "P&L Moy ($)",
            }).style.map(
                _color_pnl, subset=["P&L Total ($)", "P&L Moy ($)"]
            ).format({
                "P&L Total ($)": "${:+,.2f}",
                "P&L Moy ($)": "${:+,.2f}",
            }),
            use_container_width=True, hide_index=True,
        )

st.divider()


# ── Step Trailing stats ───────────────────────────────────────────────────────

if not closed.empty and "peak_price" in closed.columns and "entry_filled" in closed.columns:
    st.subheader("📊 Performance du Step Trailing")

    trail_df = closed.copy()
    trail_df["peak_gain_pct"] = (trail_df["peak_price"] - trail_df["entry_filled"]) / trail_df["entry_filled"] * 100
    trail_df["exit_gain_pct"] = (trail_df["exit_price"] - trail_df["entry_filled"]) / trail_df["entry_filled"] * 100

    trail_valid = trail_df.dropna(subset=["peak_gain_pct", "exit_gain_pct"])

    if not trail_valid.empty:
        col_t1, col_t2, col_t3, col_t4 = st.columns(4)

        avg_peak = trail_valid["peak_gain_pct"].mean()
        avg_exit = trail_valid["exit_gain_pct"].mean()
        avg_capture = (trail_valid["exit_gain_pct"] / trail_valid["peak_gain_pct"]).replace([float("inf"), -float("inf")], 0).mean() * 100
        avg_steps = trail_valid["trail_steps"].mean() if "trail_steps" in trail_valid.columns else 0

        col_t1.metric("🏔️ Gain peak moyen", f"{avg_peak:+.2f}%")
        col_t2.metric("🎯 Gain sortie moyen", f"{avg_exit:+.2f}%")
        col_t3.metric("📐 Capture ratio", f"{avg_capture:.0f}%",
                       help="% du gain peak capturé à la sortie")
        col_t4.metric("🔗 Steps moyen", f"{avg_steps:.1f}",
                       help="Nombre moyen de steps trailing franchis")

        fig_scatter = go.Figure()
        fig_scatter.add_trace(go.Scatter(
            x=trail_valid["peak_gain_pct"],
            y=trail_valid["exit_gain_pct"],
            mode="markers",
            marker=dict(
                color=trail_valid["pnl_usd"],
                colorscale="RdYlGn",
                size=8,
                showscale=True,
                colorbar=dict(title="P&L $"),
            ),
            text=trail_valid["symbol"],
            hovertemplate="<b>%{text}</b><br>Peak: %{x:.1f}%<br>Exit: %{y:.1f}%<extra></extra>",
        ))
        fig_scatter.add_trace(go.Scatter(
            x=[0, trail_valid["peak_gain_pct"].max()],
            y=[0, trail_valid["peak_gain_pct"].max()],
            mode="lines", name="Capture 100%",
            line=dict(dash="dash", color="gray", width=1),
        ))
        fig_scatter.update_layout(
            xaxis_title="Gain au peak (%)",
            yaxis_title="Gain à la sortie (%)",
            height=350, margin=dict(l=0, r=0, t=10, b=0),
            showlegend=False,
        )
        st.plotly_chart(fig_scatter, use_container_width=True)

    # Distribution des steps trailing
    if "exit_reason" in closed.columns:
        trail_exits = closed[closed["exit_reason"].str.startswith("TRAIL_SL", na=False)]
        if not trail_exits.empty and "trail_steps" in trail_exits.columns:
            st.subheader("🔗 Distribution des Trail Steps")
            fig_steps = px.histogram(
                trail_exits, x="trail_steps", nbins=20,
                labels={"trail_steps": "Nombre de steps"},
                color_discrete_sequence=["#7c4dff"],
            )
            fig_steps.update_layout(
                height=250, margin=dict(l=0, r=0, t=10, b=0),
                showlegend=False,
            )
            st.plotly_chart(fig_steps, use_container_width=True)

st.divider()


# ── Derniers trades ────────────────────────────────────────────────────────────

st.subheader("📋 Derniers trades")

if not closed.empty:
    show_trades = closed.sort_values("closed_at", ascending=False).head(30).copy()

    display = show_trades[[
        c for c in [
            "symbol", "side", "entry_filled", "exit_price", "peak_price",
            "pnl_usd", "pnl_pct", "exit_reason", "trail_steps",
            "holding_time_hours", "fees_total", "closed_at",
        ] if c in show_trades.columns
    ]].copy()

    rename = {
        "symbol": "Paire", "side": "Side",
        "entry_filled": "Entrée", "exit_price": "Sortie",
        "peak_price": "Peak",
        "pnl_usd": "P&L ($)", "pnl_pct": "P&L (%)",
        "exit_reason": "Raison", "trail_steps": "Steps",
        "holding_time_hours": "Durée (h)",
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
else:
    st.info("Aucun trade CrashBot fermé sur cette période")


# ── Distribution des P&L ──────────────────────────────────────────────────────

if not closed.empty and "pnl_usd" in closed.columns:
    st.subheader("📊 Distribution des P&L")

    col_hist, col_box = st.columns(2)

    with col_hist:
        fig_hist = px.histogram(
            closed, x="pnl_usd", nbins=30,
            labels={"pnl_usd": "P&L ($)"},
            color_discrete_sequence=["#7c4dff"],
        )
        fig_hist.add_vline(x=0, line_dash="dash", line_color="white", opacity=0.5)
        fig_hist.update_layout(
            height=300, margin=dict(l=0, r=0, t=10, b=0),
            showlegend=False,
        )
        st.plotly_chart(fig_hist, use_container_width=True)

    with col_box:
        if "exit_reason" in closed.columns:
            fig_box = px.box(
                closed, x="exit_reason", y="pnl_usd",
                labels={"exit_reason": "Raison sortie", "pnl_usd": "P&L ($)"},
                color="exit_reason",
            )
        else:
            fig_box = px.box(
                closed, y="pnl_usd",
                labels={"pnl_usd": "P&L ($)"},
            )
        fig_box.update_layout(
            height=300, margin=dict(l=0, r=0, t=10, b=0),
            showlegend=False,
        )
        st.plotly_chart(fig_box, use_container_width=True)


# ── Statistiques avancées ──────────────────────────────────────────────────────

if not closed.empty and "pnl_usd" in closed.columns:
    st.divider()
    st.subheader("📐 Statistiques avancées")

    col_s1, col_s2, col_s3, col_s4 = st.columns(4)

    wins = closed[closed["pnl_usd"] > 0]
    losses = closed[closed["pnl_usd"] <= 0]

    avg_win_val = wins["pnl_usd"].mean() if len(wins) > 0 else 0
    avg_loss_val = abs(losses["pnl_usd"].mean()) if len(losses) > 0 else 0
    payoff_ratio = avg_win_val / avg_loss_val if avg_loss_val > 0 else float("inf")

    cumulative = closed.sort_values("closed_at")["pnl_usd"].cumsum()
    running_max = cumulative.cummax()
    drawdown = cumulative - running_max
    max_dd = drawdown.min()

    avg_holding = closed["holding_time_hours"].mean() if "holding_time_hours" in closed.columns else 0

    col_s1.metric("💵 Gain moyen", f"${avg_win_val:+.2f}")
    col_s2.metric("💸 Perte moyenne", f"${-avg_loss_val:-.2f}")
    col_s3.metric("📊 Payoff Ratio", f"{payoff_ratio:.2f}" if payoff_ratio != float("inf") else "∞")
    col_s4.metric("📉 Max Drawdown", f"${max_dd:+.2f}")

    col_s5, col_s6, col_s7, col_s8 = st.columns(4)

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

    col_s5.metric("⏱️ Durée moyenne", f"{avg_holding:.1f}h")
    col_s6.metric("🔥 Série max wins", f"{max_win_streak}")
    col_s7.metric("❄️ Série max pertes", f"{abs(max_loss_streak)}")
    col_s8.metric("💳 Fees total", f"${total_fees:.2f}")


# ── Footer ─────────────────────────────────────────────────────────────────────

st.divider()
st.caption(
    f"TradeX CrashBot Dashboard • Données Firebase (exchange={EXCHANGE}) • "
    f"Stratégie: Dip Buy (-20% en 48h) → Long + Step Trail + Kill-Switch • "
    f"Risk: 5% par trade • Max 5 positions simultanées • "
    f"Dernière mise à jour : {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}"
)
