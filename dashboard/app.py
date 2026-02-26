"""
TradeX Dashboard — Monitoring temps réel du bot de trading.
Lit les données depuis Firebase Firestore (collections: trades, events, daily_snapshots).
Port: 8502 (le 8501 est déjà pris par betX).
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

# Résoudre le chemin des credentials Firebase
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


# ── Firebase connection (cached) ───────────────────────────────────────────────

@st.cache_resource
def _get_db() -> firestore.Client:
    cred = service_account.Credentials.from_service_account_file(_CRED_PATH)
    return firestore.Client(project=cred.project_id, credentials=cred)


def _fetch_trades(days: int = 90) -> pd.DataFrame:
    """Récupère les trades depuis Firestore."""
    db = _get_db()
    since = datetime.now(timezone.utc) - timedelta(days=days)
    docs = (
        db.collection("trades")
        .where("created_at", ">=", since.isoformat())
        .order_by("created_at", direction=firestore.Query.DESCENDING)
        .stream()
    )
    rows = [doc.to_dict() | {"_id": doc.id} for doc in docs]
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    # Parse dates
    for col in ("opened_at", "closed_at", "created_at", "updated_at"):
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce", utc=True)
    return df


def _fetch_daily_snapshots(days: int = 90) -> pd.DataFrame:
    """Récupère les snapshots journaliers."""
    db = _get_db()
    since = (datetime.now(timezone.utc) - timedelta(days=days)).strftime("%Y-%m-%d")
    docs = (
        db.collection("daily_snapshots")
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
    """Récupère les trades ouverts (pas de closed_at)."""
    db = _get_db()
    docs = (
        db.collection("trades")
        .where("status", "in", ["OPEN", "ZERO_RISK"])
        .stream()
    )
    rows = [doc.to_dict() | {"_id": doc.id} for doc in docs]
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows)


def _fetch_close_failures(hours: int = 24) -> pd.DataFrame:
    """Récupère les événements CLOSE_FAILURE récents."""
    db = _get_db()
    cutoff = (datetime.utcnow() - timedelta(hours=hours)).isoformat()
    docs = (
        db.collection("events")
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


# ── Helpers ────────────────────────────────────────────────────────────────────

def _color_pnl(val):
    """Colorise le P&L."""
    if pd.isna(val):
        return ""
    return "color: #00c853" if val >= 0 else "color: #ff1744"


def _fmt_price(p):
    """Formatage intelligent des prix."""
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

st.sidebar.image("https://img.icons8.com/color/96/combo-chart.png", width=60)
st.sidebar.title("TradeX")
st.sidebar.caption("Bot de trading crypto — Range Only")

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

st.title("📊 TradeX Dashboard")

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

    # Equity actuelle
    equity = closed.sort_values("closed_at").iloc[-1].get("equity_after", None)
    if pd.isna(equity):
        equity = None

    col1.metric("💰 P&L Total", f"${total_pnl:+,.2f}")
    col2.metric("📈 Trades", f"{len(closed)}", f"{len(win_trades)}W / {len(loss_trades)}L")
    col3.metric("🎯 Win Rate", f"{win_rate:.1f}%")
    col4.metric("⚖️ Profit Factor", f"{profit_factor:.2f}" if profit_factor != float("inf") else "∞")
    col5.metric("🏦 Equity", f"${equity:,.2f}" if equity else "—")
    col6.metric("📂 Positions", f"{len(open_df)}/3")
else:
    col1.metric("💰 P&L Total", "—")
    col2.metric("📈 Trades", "0")
    col3.metric("🎯 Win Rate", "—")
    col4.metric("⚖️ Profit Factor", "—")
    col5.metric("🏦 Equity", "—")
    col6.metric("📂 Positions", f"{len(open_df)}/3")

st.divider()

# ── Positions ouvertes ─────────────────────────────────────────────────────────

st.subheader("🟢 Positions ouvertes")

if not open_df.empty:
    display_cols = ["symbol", "side", "entry_filled", "sl_price", "tp_price",
                    "size", "size_usd", "risk_usd", "strategy", "status", "opened_at"]
    available_cols = [c for c in display_cols if c in open_df.columns]
    show_df = open_df[available_cols].copy()

    rename_map = {
        "symbol": "Paire", "side": "Side", "entry_filled": "Entrée",
        "sl_price": "SL", "tp_price": "TP", "size": "Taille",
        "size_usd": "Notionnel $", "risk_usd": "Risque $",
        "strategy": "Stratégie", "status": "Statut", "opened_at": "Ouvert le",
    }
    show_df = show_df.rename(columns={k: v for k, v in rename_map.items() if k in show_df.columns})
    st.dataframe(show_df, width='stretch', hide_index=True)
else:
    st.info("Aucune position ouverte")

# ── Alertes (close bloquées) ───────────────────────────────────────────────────────────

# Trades marqués close_blocked dans Firebase
blocked_trades = []
if not open_df.empty and "close_blocked" in open_df.columns:
    blocked_trades = open_df[open_df["close_blocked"] == True]

# Événements CLOSE_FAILURE récents
close_failures_df = _fetch_close_failures(hours=24)

if not blocked_trades.empty if isinstance(blocked_trades, pd.DataFrame) else len(blocked_trades) > 0 or not close_failures_df.empty:
    st.subheader("⚠️ Alertes")

    if isinstance(blocked_trades, pd.DataFrame) and not blocked_trades.empty:
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

st.divider()

# ── Equity curve ───────────────────────────────────────────────────────────────

st.subheader("📈 Courbe d'equity")

if not snapshots_df.empty and "equity" in snapshots_df.columns:
    fig_eq = px.area(
        snapshots_df, x="date", y="equity",
        labels={"date": "Date", "equity": "Equity ($)"},
        color_discrete_sequence=["#00c853"],
    )
    fig_eq.update_layout(
        height=350, margin=dict(l=0, r=0, t=10, b=0),
        yaxis_tickformat="$,.0f",
        hovermode="x unified",
    )
    st.plotly_chart(fig_eq, width='stretch')
elif not closed.empty and "pnl_usd" in closed.columns:
    # Fallback 1: construire à partir de equity_after si disponible
    eq_from_field = closed.dropna(subset=["equity_after"]).sort_values("closed_at") if "equity_after" in closed.columns else pd.DataFrame()
    if not eq_from_field.empty:
        fig_eq = px.area(
            eq_from_field, x="closed_at", y="equity_after",
            labels={"closed_at": "Date", "equity_after": "Equity ($)"},
            color_discrete_sequence=["#00c853"],
        )
        fig_eq.update_layout(
            height=350, margin=dict(l=0, r=0, t=10, b=0),
            yaxis_tickformat="$,.0f",
            hovermode="x unified",
        )
        st.plotly_chart(fig_eq, width='stretch')
    else:
        # Fallback 2: construire equity à partir du P&L cumulé
        eq_df = closed.sort_values("closed_at").copy()
        initial_equity = eq_df.iloc[0].get("equity_at_entry", 1000) or 1000
        eq_df["equity_curve"] = initial_equity + eq_df["pnl_usd"].cumsum()
        fig_eq = px.area(
            eq_df, x="closed_at", y="equity_curve",
            labels={"closed_at": "Date", "equity_curve": "Equity ($)"},
            color_discrete_sequence=["#00c853"],
        )
        fig_eq.update_layout(
            height=350, margin=dict(l=0, r=0, t=10, b=0),
            yaxis_tickformat="$,.0f",
            hovermode="x unified",
        )
        st.plotly_chart(fig_eq, width='stretch')
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
        line=dict(color="#2196f3", width=2),
        fill="tozeroy",
        fillcolor="rgba(33,150,243,0.1)",
    ))
    fig_pnl.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)
    fig_pnl.update_layout(
        height=300, margin=dict(l=0, r=0, t=10, b=0),
        yaxis_tickformat="$,.2f",
        hovermode="x unified",
    )
    st.plotly_chart(fig_pnl, width='stretch')

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
    st.plotly_chart(fig_daily, width='stretch')

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
        st.plotly_chart(fig_pair, width='stretch')

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

st.divider()

# ── Derniers trades ────────────────────────────────────────────────────────────

st.subheader("📋 Derniers trades")

if not closed.empty:
    show_trades = closed.sort_values("closed_at", ascending=False).head(30).copy()

    display = show_trades[[
        c for c in [
            "symbol", "side", "strategy", "entry_filled", "exit_price",
            "pnl_usd", "pnl_pct", "exit_reason", "holding_time_hours",
            "maker_or_taker", "exit_fill_type", "fees_total", "closed_at",
        ] if c in show_trades.columns
    ]].copy()

    rename = {
        "symbol": "Paire", "side": "Side", "strategy": "Strat",
        "entry_filled": "Entrée", "exit_price": "Sortie",
        "pnl_usd": "P&L ($)", "pnl_pct": "P&L (%)",
        "exit_reason": "Raison", "holding_time_hours": "Durée (h)",
        "maker_or_taker": "Fill entrée", "exit_fill_type": "Fill sortie",
        "fees_total": "Fees ($)", "closed_at": "Clôturé le",
    }
    display = display.rename(columns={k: v for k, v in rename.items() if k in display.columns})

    # Convertir pnl_pct (décimal) en pourcentage
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
else:
    st.info("Aucun trade fermé sur cette période")

# ── Distribution des P&L ──────────────────────────────────────────────────────

if not closed.empty and "pnl_usd" in closed.columns:
    st.subheader("📊 Distribution des P&L")

    col_hist, col_box = st.columns(2)

    with col_hist:
        fig_hist = px.histogram(
            closed, x="pnl_usd", nbins=30,
            labels={"pnl_usd": "P&L ($)"},
            color_discrete_sequence=["#2196f3"],
        )
        fig_hist.add_vline(x=0, line_dash="dash", line_color="white", opacity=0.5)
        fig_hist.update_layout(
            height=300, margin=dict(l=0, r=0, t=10, b=0),
            showlegend=False,
        )
        st.plotly_chart(fig_hist, width='stretch')

    with col_box:
        if "strategy" in closed.columns:
            fig_box = px.box(
                closed, x="strategy", y="pnl_usd",
                labels={"strategy": "Stratégie", "pnl_usd": "P&L ($)"},
                color="strategy",
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
        st.plotly_chart(fig_box, width='stretch')

# ── Footer ─────────────────────────────────────────────────────────────────────

st.divider()
st.caption(
    f"TradeX Dashboard • Données Firebase (projet satochi-d38ec) • "
    f"Dernière mise à jour : {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}"
)
