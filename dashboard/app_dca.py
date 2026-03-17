"""
TradeX DCA Dashboard — Standalone premium dashboard for the DCA RSI bot.
Runs on port 8503, dark theme, 3 tabs (Dashboard · DCA · Performance).

Usage:
    streamlit run dashboard/app_dca.py --server.port 8503
"""

from __future__ import annotations

import os
from datetime import datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st
from google.cloud import firestore
from google.oauth2 import service_account
from plotly.subplots import make_subplots

# ── Config ─────────────────────────────────────────────────────────────────────

_BASE = Path(__file__).resolve().parent.parent
_CRED_PATH = os.getenv(
    "FIREBASE_CREDENTIALS_PATH",
    str(_BASE / "firebase-credentials.json"),
)

st.set_page_config(
    page_title="TradeX · DCA Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="collapsed",
)

DASHBOARD_TZ = ZoneInfo(os.getenv("DASHBOARD_TZ", "Europe/Paris"))

# ── Palette ────────────────────────────────────────────────────────────────────

C_BG = "#0a0e14"
C_CARD = "#1e293b"
C_CYAN = "#00d4ff"
C_GREEN = "#22c55e"
C_RED = "#ef4444"
C_ORANGE = "#f97316"
C_BLUE = "#3b82f6"
C_VIOLET = "#a855f7"
C_GOLD = "#fbbf24"
C_GRAY = "#6b7280"
C_TEXT = "#e2e8f0"
C_MUTED = "#94a3b8"

# bracket → colour
_BRACKET_COLOR = {
    "WARM": C_BLUE,
    "NEUTRAL": C_ORANGE,
    "OVERSOLD": C_RED,
    "DEEP_VALUE": C_GOLD,
}


def _buy_color(reason: str, bracket: str) -> str:
    if "CRASH" in str(reason).upper():
        return C_VIOLET
    return _BRACKET_COLOR.get(bracket, C_BLUE)


def _buy_label(reason: str, bracket: str) -> str:
    if "CRASH" in str(reason).upper():
        for lvl in ("15", "25", "35"):
            if lvl in str(reason):
                return f"Crash -{lvl}%"
        return "Crash"
    return {"WARM": "×1", "NEUTRAL": "×2", "OVERSOLD": "×3", "DEEP_VALUE": "×5 MVRV"}.get(bracket, "DCA")


def _buy_emoji(reason: str, bracket: str) -> str:
    if "CRASH" in str(reason).upper():
        return "🟣"
    return {"OVERSOLD": "🔴", "NEUTRAL": "🟠", "WARM": "🔵", "DEEP_VALUE": "🟡"}.get(bracket, "⚪")


# ══════════════════════════════════════════════════════════════════════════════
#  Firebase helpers
# ══════════════════════════════════════════════════════════════════════════════

@st.cache_resource
def _get_db() -> firestore.Client:
    cred = service_account.Credentials.from_service_account_file(_CRED_PATH)
    return firestore.Client(project=cred.project_id, credentials=cred)


@st.cache_data(ttl=120)
def _fetch_dca_buys(days: int = 365) -> list[dict]:
    db = _get_db()
    since = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()
    try:
        docs = (
            db.collection("events")
            .where("event_type", "==", "DCA_BUY")
            .where("exchange", "==", "revolut-dca")
            .where("timestamp", ">=", since)
            .order_by("timestamp", direction=firestore.Query.DESCENDING)
            .limit(2000)
            .stream()
        )
        return [doc.to_dict() for doc in docs]
    except Exception:
        return []


@st.cache_data(ttl=120)
def _fetch_dca_heartbeat() -> dict:
    db = _get_db()
    try:
        docs = (
            db.collection("events")
            .where("event_type", "==", "DCA_HEARTBEAT")
            .where("exchange", "==", "revolut-dca")
            .order_by("timestamp", direction=firestore.Query.DESCENDING)
            .limit(1)
            .stream()
        )
        for doc in docs:
            return doc.to_dict()
    except Exception:
        pass
    return {}


# ══════════════════════════════════════════════════════════════════════════════
#  On-chain: MVRV (CoinMetrics Community API — free, no auth)
# ══════════════════════════════════════════════════════════════════════════════

@st.cache_data(ttl=3600)
def _fetch_mvrv() -> float | None:
    """Fetch latest BTC MVRV ratio from CoinMetrics Community API."""
    try:
        r = requests.get(
            "https://community-api.coinmetrics.io/v4/timeseries/asset-metrics",
            params={"assets": "btc", "metrics": "CapMVRVCur", "frequency": "1d", "page_size": 2},
            timeout=10,
        )
        r.raise_for_status()
        rows = r.json().get("data", [])
        if rows:
            return float(rows[-1]["CapMVRVCur"])
    except Exception:
        pass
    return None


# ══════════════════════════════════════════════════════════════════════════════
#  Price APIs (public, no auth)
# ══════════════════════════════════════════════════════════════════════════════

@st.cache_data(ttl=30)
def _fetch_current_prices() -> dict[str, float]:
    """Live BTC / ETH prices via Binance public ticker (no auth required)."""
    mapping = {"BTCUSDT": "BTC-USD", "ETHUSDT": "ETH-USD"}
    try:
        r = requests.get(
            "https://api.binance.com/api/v3/ticker/price",
            params={"symbols": '["BTCUSDT","ETHUSDT"]'},
            timeout=5,
        )
        r.raise_for_status()
        return {
            mapping.get(t["symbol"], t["symbol"]): float(t["price"])
            for t in r.json()
            if t["symbol"] in mapping
        }
    except Exception:
        return {}


@st.cache_data(ttl=3600)
def _fetch_daily_klines(symbol: str = "BTCUSDT", days: int = 365) -> pd.DataFrame:
    """Daily OHLC from Binance public klines endpoint."""
    try:
        r = requests.get(
            "https://api.binance.com/api/v3/klines",
            params={"symbol": symbol, "interval": "1d", "limit": min(days, 1000)},
            timeout=10,
        )
        r.raise_for_status()
        rows = [
            {"date": pd.Timestamp(k[0], unit="ms", tz="UTC").tz_localize(None), "close": float(k[4])}
            for k in r.json()
        ]
        return pd.DataFrame(rows).sort_values("date").reset_index(drop=True)
    except Exception:
        return pd.DataFrame()


# ══════════════════════════════════════════════════════════════════════════════
#  Chart helpers
# ══════════════════════════════════════════════════════════════════════════════

def _dark_layout(
    fig: go.Figure,
    height: int = 420,
    show_legend: bool = True,
) -> go.Figure:
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor=C_BG,
        plot_bgcolor=C_BG,
        font=dict(color=C_TEXT, size=13),
        hovermode="x unified",
        hoverlabel=dict(bgcolor="#1e293b", font_color=C_TEXT, font_size=12),
        margin=dict(l=10, r=10, t=30, b=10),
        height=height,
        showlegend=show_legend,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            font=dict(size=11),
        ),
    )
    for ax_fn in (fig.update_xaxes, fig.update_yaxes):
        ax_fn(
            showgrid=True,
            gridcolor="rgba(255,255,255,0.06)",
            zeroline=False,
            linecolor="rgba(255,255,255,0.12)",
        )
    return fig


# ══════════════════════════════════════════════════════════════════════════════
#  Data processing
# ══════════════════════════════════════════════════════════════════════════════

def _parse_buys(events: list[dict]) -> pd.DataFrame:
    rows: list[dict] = []
    for ev in events:
        d = ev.get("data", {})
        ts_raw = ev.get("timestamp", "")
        if hasattr(ts_raw, "isoformat"):
            ts_raw = ts_raw.isoformat()
        try:
            ts = pd.Timestamp(str(ts_raw))
            # Ensure tz-naive for merge compatibility with Binance klines
            if ts.tzinfo is not None:
                ts = ts.tz_convert("UTC").tz_localize(None)
        except Exception:
            ts = pd.NaT

        reason = d.get("reason", "")
        bracket = d.get("bracket", "")
        rows.append(
            {
                "timestamp": ts,
                "date": ts.normalize() if pd.notna(ts) else pd.NaT,
                "symbol": d.get("symbol", ev.get("symbol", "")),
                "reason": reason,
                "bracket": bracket,
                "amount_usd": float(d.get("amount_usd", 0)),
                "price": float(d.get("price", 0)),
                "size": float(d.get("size", 0)),
                "rsi": float(d.get("rsi", 0)),
                "fill_type": d.get("fill_type", ""),
                "color": _buy_color(reason, bracket),
                "label": _buy_label(reason, bracket),
                "emoji": _buy_emoji(reason, bracket),
            }
        )

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    df = df.sort_values("timestamp").reset_index(drop=True)

    # Tag BTC / ETH
    df["is_btc"] = df["symbol"].str.startswith("BTC")
    df["btc_size"] = df["size"].where(df["is_btc"], 0)
    df["eth_size"] = df["size"].where(~df["is_btc"], 0)

    df["cum_usd"] = df["amount_usd"].cumsum()
    df["cum_btc"] = df["btc_size"].cumsum()
    df["cum_eth"] = df["eth_size"].cumsum()

    return df


def _portfolio_timeseries(
    df_buys: pd.DataFrame,
    df_btc: pd.DataFrame,
    df_eth: pd.DataFrame,
) -> pd.DataFrame:
    """Merge buy data with daily prices → portfolio + PnL timeseries."""
    if df_buys.empty or df_btc.empty:
        return pd.DataFrame()

    # Daily aggregated buys — ensure tz-naive dates
    buys = df_buys.copy()
    buys["date"] = pd.to_datetime(buys["date"]).dt.tz_localize(None)
    daily = (
        buys.groupby("date")
        .agg(btc=("btc_size", "sum"), eth=("eth_size", "sum"), usd=("amount_usd", "sum"))
        .sort_index()
        .reset_index()
    )
    daily["cum_btc"] = daily["btc"].cumsum()
    daily["cum_eth"] = daily["eth"].cumsum()
    daily["cum_usd"] = daily["usd"].cumsum()

    # Merge prices — ensure tz-naive
    df_btc = df_btc.copy()
    df_btc["date"] = pd.to_datetime(df_btc["date"]).dt.tz_localize(None)
    df_btc = df_btc.rename(columns={"close": "btc_close"}).sort_values("date")
    if not df_eth.empty:
        df_eth = df_eth.copy()
        df_eth["date"] = pd.to_datetime(df_eth["date"]).dt.tz_localize(None)
        df_eth = df_eth.rename(columns={"close": "eth_close"}).sort_values("date")
        prices = pd.merge_asof(df_btc, df_eth[["date", "eth_close"]], on="date", direction="backward")
    else:
        prices = df_btc.copy()
        prices["eth_close"] = 0.0

    # Merge cumulative buys on price grid
    merged = pd.merge_asof(
        prices.sort_values("date"),
        daily[["date", "cum_btc", "cum_eth", "cum_usd"]].sort_values("date"),
        on="date",
        direction="backward",
    )
    for c in ("cum_btc", "cum_eth", "cum_usd"):
        merged[c] = merged[c].fillna(0)

    # Portfolio & PnL
    merged["portfolio"] = merged["cum_btc"] * merged["btc_close"] + merged["cum_eth"] * merged["eth_close"]
    merged["pnl"] = merged["portfolio"] - merged["cum_usd"]
    merged["pnl_pct"] = merged["pnl"] / merged["cum_usd"].replace(0, np.nan) * 100

    # Restrict to period starting from first buy
    first_buy = pd.Timestamp(buys["date"].min())
    if first_buy.tzinfo is not None:
        first_buy = first_buy.tz_localize(None)
    merged = merged[merged["date"] >= first_buy].copy().reset_index(drop=True)

    # Drawdown
    merged["peak"] = merged["portfolio"].cummax()
    merged["dd"] = (merged["portfolio"] - merged["peak"]) / merged["peak"].replace(0, np.nan) * 100

    return merged


# ══════════════════════════════════════════════════════════════════════════════
#  Custom CSS
# ══════════════════════════════════════════════════════════════════════════════

st.markdown(
    """
<style>
    .stApp { background-color: #0e1117; }
    [data-testid="stMetric"] {
        background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 12px;
        padding: 16px 20px;
    }
    [data-testid="stMetricValue"] { font-size: 1.4rem; font-weight: 700; }
    [data-testid="stMetricDelta"]  { font-size: 0.9rem; }
    .stTabs [data-baseweb="tab-list"] {
        gap: 0; background: #1e293b; border-radius: 12px; padding: 4px;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px; padding: 8px 24px; font-weight: 600;
    }
    .stTabs [aria-selected="true"] {
        background: #0ea5e9 !important; color: white !important;
    }
    div[data-testid="stDataFrame"] {
        border: 1px solid rgba(255,255,255,0.08); border-radius: 8px;
    }
    .stDivider { margin: 1.5rem 0; }
</style>
""",
    unsafe_allow_html=True,
)


# ══════════════════════════════════════════════════════════════════════════════
#  TAB 1 — Dashboard
# ══════════════════════════════════════════════════════════════════════════════


def render_dashboard(
    df_buys: pd.DataFrame,
    hb: dict,
    prices: dict[str, float],
    df_btc: pd.DataFrame,
    df_eth: pd.DataFrame,
    mvrv: float | None = None,
):
    hb_data = hb.get("data", {})
    btc_price = prices.get("BTC-USD", 0)
    eth_price = prices.get("ETH-USD", 0)

    # ── Totals ─────────────────────────────────────────────────────────────
    if not df_buys.empty:
        total_btc = df_buys["btc_size"].sum()
        total_eth = df_buys["eth_size"].sum()
        total_invested = df_buys["amount_usd"].sum()
    else:
        total_btc = total_eth = total_invested = 0.0

    portfolio_value = total_btc * btc_price + total_eth * eth_price
    pnl = portfolio_value - total_invested
    pnl_pct = (pnl / total_invested * 100) if total_invested > 0 else 0

    btc_spent = df_buys.loc[df_buys["is_btc"], "amount_usd"].sum() if not df_buys.empty else 0
    avg_btc = (btc_spent / total_btc) if total_btc > 0 else 0
    avg_vs_spot = ((btc_price - avg_btc) / avg_btc * 100) if avg_btc > 0 else 0

    # ── KPI Cards ──────────────────────────────────────────────────────────
    k1, k2, k3, k4 = st.columns(4)
    with k1:
        n_buys = len(df_buys) if not df_buys.empty else 0
        st.metric("Capital investi", f"${total_invested:,.0f}", f"{n_buys} achats")
    with k2:
        st.metric("Valeur actuelle", f"${portfolio_value:,.0f}")
    with k3:
        dc = "normal" if pnl >= 0 else "inverse"
        st.metric("PnL", f"${pnl:+,.2f}", f"{pnl_pct:+.1f}%", delta_color=dc)
    with k4:
        st.metric("PMP Bitcoin", f"${avg_btc:,.0f}", f"{avg_vs_spot:+.1f}% vs spot")

    # ── MVRV / RSI row ─────────────────────────────────────────────────────
    m1, m2, m3, m4 = st.columns(4)
    hb_rsi = hb_data.get("rsi", 0)
    hb_bracket = hb_data.get("bracket", "")
    with m1:
        if mvrv is not None:
            mvrv_label = "🟢 Deep Value" if mvrv < 1.0 else ("Normal" if mvrv < 3.0 else "🔴 Surchauffe")
            st.metric("MVRV BTC", f"{mvrv:.3f}", mvrv_label)
        else:
            st.metric("MVRV BTC", "N/A")
    with m2:
        st.metric("RSI BTC", f"{hb_rsi:.1f}", hb_bracket)
    with m3:
        hb_mvrv = hb_data.get("mvrv")
        mvrv_mult = "×5" if (mvrv is not None and mvrv < 1.0) else "×1"
        st.metric("Multiplicateur MVRV", mvrv_mult)
    with m4:
        alloc = hb_data.get("alloc", "90/10 BTC/ETH")
        st.metric("Allocation", "90 % BTC / 10 % ETH")

    st.markdown("")

    # ── Hero chart — dual-axis ─────────────────────────────────────────────
    ts = _portfolio_timeseries(df_buys, df_btc, df_eth)

    if not ts.empty:
        fig = make_subplots(specs=[[{"secondary_y": True}]])

        # BTC price (left y-axis)
        fig.add_trace(
            go.Scatter(
                x=ts["date"],
                y=ts["btc_close"],
                name="Prix BTC",
                mode="lines",
                line=dict(color=C_CYAN, width=2),
                hovertemplate="BTC: $%{y:,.0f}<extra></extra>",
            ),
            secondary_y=False,
        )

        # Invested (right y-axis, dashed gray)
        fig.add_trace(
            go.Scatter(
                x=ts["date"],
                y=ts["cum_usd"],
                name="Investi",
                mode="lines",
                line=dict(color=C_GRAY, width=1.5, dash="dot"),
                hovertemplate="Investi: $%{y:,.0f}<extra></extra>",
            ),
            secondary_y=True,
        )

        # Portfolio value (right y-axis, solid green + fill)
        fig.add_trace(
            go.Scatter(
                x=ts["date"],
                y=ts["portfolio"],
                name="Portfolio",
                mode="lines",
                line=dict(color=C_GREEN, width=2.5),
                fill="tonexty",
                fillcolor="rgba(34,197,94,0.08)",
                hovertemplate="Portfolio: $%{y:,.0f}<extra></extra>",
            ),
            secondary_y=True,
        )

        # Buy markers on BTC price axis
        if not df_buys.empty:
            btc_buys = df_buys[df_buys["is_btc"]].copy()
            if not btc_buys.empty:
                fig.add_trace(
                    go.Scatter(
                        x=btc_buys["timestamp"],
                        y=btc_buys["price"],
                        mode="markers",
                        name="Achats",
                        marker=dict(
                            size=[max(8, min(20, a / 12)) for a in btc_buys["amount_usd"]],
                            color=btc_buys["color"].tolist(),
                            line=dict(width=1.5, color="#0a0e14"),
                            symbol="circle",
                        ),
                        text=[
                            f"<b>{r['label']}</b><br>"
                            f"${r['amount_usd']:.0f} @ ${r['price']:,.0f}<br>"
                            f"RSI {r['rsi']:.0f}"
                            for _, r in btc_buys.iterrows()
                        ],
                        hovertemplate="%{text}<extra></extra>",
                    ),
                    secondary_y=False,
                )

        fig.update_yaxes(
            title_text="Prix BTC ($)", tickformat="$,.0f", secondary_y=False
        )
        fig.update_yaxes(
            title_text="Valeur ($)", tickformat="$,.0f", secondary_y=True
        )
        _dark_layout(fig, height=480)
        fig.update_layout(margin=dict(l=10, r=10, t=40, b=10))
        st.plotly_chart(fig, use_container_width=True, key="hero")

    elif not df_buys.empty:
        st.info("Impossible de charger les prix historiques BTC pour le graphique principal.")

    # ── Bottom row: donut + recent buys ────────────────────────────────────
    col_donut, col_recent = st.columns([1, 2])

    with col_donut:
        st.markdown("##### Allocation")
        btc_val = total_btc * btc_price
        eth_val = total_eth * eth_price
        dca_rem = hb_data.get("dca_remaining", 0)
        crash_rem = hb_data.get("crash_remaining", 0)
        cash = dca_rem + crash_rem

        labels, values, colors = ["BTC", "ETH"], [btc_val, eth_val], [C_CYAN, C_BLUE]
        if cash > 0:
            labels.append("USD restant")
            values.append(cash)
            colors.append(C_GRAY)

        fig_d = go.Figure(
            go.Pie(
                labels=labels,
                values=values,
                hole=0.65,
                marker=dict(colors=colors, line=dict(width=2, color=C_BG)),
                textinfo="label+percent",
                textfont=dict(size=12, color=C_TEXT),
                hovertemplate="<b>%{label}</b><br>$%{value:,.0f}<br>%{percent}<extra></extra>",
            )
        )
        _dark_layout(fig_d, height=320, show_legend=False)
        fig_d.update_layout(margin=dict(l=0, r=0, t=10, b=10))
        st.plotly_chart(fig_d, use_container_width=True, key="donut")

        # Budget sub-metrics
        if dca_rem or crash_rem:
            b1, b2 = st.columns(2)
            with b1:
                st.metric("Budget DCA", f"${dca_rem:,.0f}")
            with b2:
                st.metric("Crash reserve", f"${crash_rem:,.0f}")

    with col_recent:
        st.markdown("##### 10 derniers achats")
        if not df_buys.empty:
            recent = df_buys.sort_values("timestamp", ascending=False).head(10)
            rows_disp = [
                {
                    "Date": r["timestamp"].strftime("%d/%m %H:%M") if pd.notna(r["timestamp"]) else "—",
                    "": r["emoji"],
                    "Type": r["label"],
                    "Pair": r["symbol"],
                    "Montant": f"${r['amount_usd']:.0f}",
                    "Prix": f"${r['price']:,.0f}",
                    "RSI": f"{r['rsi']:.0f}",
                }
                for _, r in recent.iterrows()
            ]
            st.dataframe(rows_disp, use_container_width=True, hide_index=True)
        else:
            st.info("Aucun achat enregistré.")


# ══════════════════════════════════════════════════════════════════════════════
#  TAB 2 — DCA detail
# ══════════════════════════════════════════════════════════════════════════════


def render_dca_detail(df_buys: pd.DataFrame, prices: dict[str, float]):
    btc_price = prices.get("BTC-USD", 0)

    if df_buys.empty:
        st.info("Aucune donnée d'achat disponible.")
        return

    # ── Filters ────────────────────────────────────────────────────────────
    st.markdown("##### Historique des achats")
    f1, f2, f3 = st.columns(3)
    with f1:
        syms = ["Tous"] + sorted(df_buys["symbol"].unique().tolist())
        sel_sym = st.selectbox("Symbole", syms, key="f_sym")
    with f2:
        sel_type = st.selectbox("Type", ["Tous", "DCA régulier", "Crash"], key="f_type")
    with f3:
        period_map = {"30j": 30, "90j": 90, "180j": 180, "Tout": 9999}
        sel_per = st.selectbox("Période", list(period_map.keys()), index=3, key="f_per")

    df_f = df_buys.copy()
    if sel_sym != "Tous":
        df_f = df_f[df_f["symbol"] == sel_sym]
    if sel_type == "DCA régulier":
        df_f = df_f[~df_f["reason"].str.contains("CRASH", na=False)]
    elif sel_type == "Crash":
        df_f = df_f[df_f["reason"].str.contains("CRASH", na=False)]
    d = period_map[sel_per]
    if d < 9999:
        df_f = df_f[df_f["timestamp"] >= pd.Timestamp.now() - pd.Timedelta(days=d)]

    rows_disp = [
        {
            "Date": r["timestamp"].strftime("%Y-%m-%d %H:%M") if pd.notna(r["timestamp"]) else "—",
            "": r["emoji"],
            "Type": r["label"],
            "Pair": r["symbol"],
            "Montant": f"${r['amount_usd']:.2f}",
            "Prix": f"${r['price']:,.2f}",
            "Taille": f"{r['size']:.8f}",
            "RSI": f"{r['rsi']:.1f}",
            "Fill": r["fill_type"],
        }
        for _, r in df_f.sort_values("timestamp", ascending=False).iterrows()
    ]
    st.dataframe(rows_disp, use_container_width=True, hide_index=True, height=400)

    st.markdown("")

    # ── Two columns: PMP evolution + distribution ──────────────────────────
    col_avg, col_dist = st.columns(2)

    with col_avg:
        st.markdown("##### Évolution du PMP Bitcoin")
        btc_only = df_buys[df_buys["is_btc"]].sort_values("timestamp").copy()
        if not btc_only.empty:
            btc_only["cum_cost"] = btc_only["amount_usd"].cumsum()
            btc_only["cum_size"] = btc_only["size"].cumsum()
            btc_only["avg_price"] = btc_only["cum_cost"] / btc_only["cum_size"]

            fig_a = go.Figure()
            fig_a.add_trace(
                go.Scatter(
                    x=btc_only["timestamp"],
                    y=btc_only["avg_price"],
                    mode="lines+markers",
                    name="PMP",
                    line=dict(color=C_ORANGE, width=2),
                    marker=dict(size=4, color=C_ORANGE),
                    hovertemplate="PMP: $%{y:,.0f}<extra></extra>",
                )
            )
            if btc_price:
                fig_a.add_hline(
                    y=btc_price,
                    line_dash="dash",
                    line_color=C_CYAN,
                    opacity=0.5,
                    annotation_text=f"Spot ${btc_price:,.0f}",
                    annotation_position="top right",
                    annotation_font_color=C_CYAN,
                )
            _dark_layout(fig_a, height=320, show_legend=False)
            fig_a.update_layout(yaxis_tickformat="$,.0f", yaxis_title="PMP ($)")
            st.plotly_chart(fig_a, use_container_width=True, key="avg_price")

    with col_dist:
        st.markdown("##### Répartition par type")
        grp = (
            df_buys.groupby("label")
            .agg(total=("amount_usd", "sum"), count=("amount_usd", "count"))
            .reset_index()
        )
        if not grp.empty:
            cmap = {"×1": C_BLUE, "×2": C_ORANGE, "×3": C_RED}
            for lbl in grp["label"]:
                if "Crash" in lbl:
                    cmap[lbl] = C_VIOLET
            fig_b = go.Figure(
                go.Bar(
                    x=grp["label"],
                    y=grp["total"],
                    marker_color=[cmap.get(l, C_GRAY) for l in grp["label"]],
                    text=[f"${v:,.0f}\n({c}×)" for v, c in zip(grp["total"], grp["count"])],
                    textposition="auto",
                    textfont=dict(size=11),
                    hovertemplate="<b>%{x}</b><br>$%{y:,.0f}<extra></extra>",
                )
            )
            _dark_layout(fig_b, height=320, show_legend=False)
            fig_b.update_layout(yaxis_tickformat="$,.0f", yaxis_title="Montant ($)")
            st.plotly_chart(fig_b, use_container_width=True, key="dist")

    # ── RSI scatter ────────────────────────────────────────────────────────
    st.markdown("##### RSI au moment de chaque achat")

    fig_r = go.Figure()

    # RSI bands
    fig_r.add_hrect(
        y0=70, y1=100, fillcolor="rgba(239,68,68,0.08)", line_width=0,
        annotation_text="Suracheté", annotation_position="top left",
        annotation_font_color=C_RED, annotation_font_size=10,
    )
    fig_r.add_hrect(y0=55, y1=70, fillcolor="rgba(249,115,22,0.06)", line_width=0)
    fig_r.add_hrect(y0=45, y1=55, fillcolor="rgba(59,130,246,0.06)", line_width=0)
    fig_r.add_hrect(
        y0=0, y1=45, fillcolor="rgba(34,197,94,0.06)", line_width=0,
        annotation_text="Survendu", annotation_position="bottom left",
        annotation_font_color=C_GREEN, annotation_font_size=10,
    )

    fig_r.add_trace(
        go.Scatter(
            x=df_buys["timestamp"],
            y=df_buys["rsi"],
            mode="markers",
            marker=dict(
                size=[max(6, min(16, a / 12)) for a in df_buys["amount_usd"]],
                color=df_buys["color"].tolist(),
                line=dict(width=1, color="rgba(0,0,0,0.4)"),
            ),
            text=[
                f"<b>{r['label']}</b> — {r['symbol']}<br>"
                f"${r['amount_usd']:.0f} @ ${r['price']:,.0f}"
                for _, r in df_buys.iterrows()
            ],
            hovertemplate="%{text}<br>RSI: %{y:.1f}<extra></extra>",
        )
    )

    for val, clr in [(70, C_RED), (55, C_ORANGE), (45, C_GREEN)]:
        fig_r.add_hline(y=val, line_dash="dot", line_color=clr, opacity=0.3)

    _dark_layout(fig_r, height=300, show_legend=False)
    fig_r.update_layout(yaxis_title="RSI", yaxis_range=[20, 85])
    st.plotly_chart(fig_r, use_container_width=True, key="rsi")


# ══════════════════════════════════════════════════════════════════════════════
#  TAB 3 — Performance
# ══════════════════════════════════════════════════════════════════════════════


def render_performance(
    df_buys: pd.DataFrame,
    prices: dict[str, float],
    df_btc: pd.DataFrame,
    df_eth: pd.DataFrame,
):
    btc_price = prices.get("BTC-USD", 0)

    if df_buys.empty or not btc_price:
        st.info("Pas assez de données pour l'analyse de performance.")
        return

    ts = _portfolio_timeseries(df_buys, df_btc, df_eth)
    if ts.empty:
        st.warning("Impossible de calculer la série temporelle du portfolio.")
        return

    total_invested = ts["cum_usd"].iloc[-1]
    portfolio_now = ts["portfolio"].iloc[-1]
    total_pnl = portfolio_now - total_invested
    total_pnl_pct = (total_pnl / total_invested * 100) if total_invested > 0 else 0
    max_dd = ts["dd"].min()

    # Win rate (days with positive daily PnL change)
    ts["pnl_change"] = ts["pnl"].diff()
    n_pos = (ts["pnl_change"] > 0).sum()
    n_days = ts["pnl_change"].notna().sum()
    wr = (n_pos / n_days * 100) if n_days > 0 else 0
    days_active = (ts["date"].max() - ts["date"].min()).days

    # ── KPIs ───────────────────────────────────────────────────────────────
    k1, k2, k3, k4 = st.columns(4)
    with k1:
        dc = "normal" if total_pnl >= 0 else "inverse"
        st.metric("PnL total", f"${total_pnl:+,.2f}", f"{total_pnl_pct:+.1f}%", delta_color=dc)
    with k2:
        st.metric("Max Drawdown", f"{max_dd:.1f}%")
    with k3:
        st.metric("Jours positifs", f"{wr:.0f}%", f"{n_pos}/{n_days}")
    with k4:
        st.metric("Durée", f"{days_active}j")

    st.markdown("")

    # ── PnL + Drawdown ────────────────────────────────────────────────────
    st.markdown("##### Évolution du PnL")

    fig_p = make_subplots(
        rows=2, cols=1, shared_xaxes=True, row_heights=[0.7, 0.3], vertical_spacing=0.05
    )

    # PnL
    pnl_color = C_GREEN if total_pnl >= 0 else C_RED
    pnl_fill = "rgba(34,197,94,0.12)" if total_pnl >= 0 else "rgba(239,68,68,0.12)"
    fig_p.add_trace(
        go.Scatter(
            x=ts["date"], y=ts["pnl"], mode="lines", name="PnL ($)",
            line=dict(color=pnl_color, width=2),
            fill="tozeroy", fillcolor=pnl_fill,
            hovertemplate="PnL: $%{y:+,.2f}<extra></extra>",
        ),
        row=1, col=1,
    )
    fig_p.add_hline(y=0, line_dash="dot", line_color=C_GRAY, opacity=0.5, row=1, col=1)

    # Drawdown
    fig_p.add_trace(
        go.Scatter(
            x=ts["date"], y=ts["dd"], mode="lines", name="Drawdown",
            line=dict(color=C_RED, width=1.5),
            fill="tozeroy", fillcolor="rgba(239,68,68,0.15)",
            hovertemplate="DD: %{y:.1f}%<extra></extra>",
        ),
        row=2, col=1,
    )

    fig_p.update_yaxes(title_text="PnL ($)", tickformat="$,.0f", row=1, col=1)
    fig_p.update_yaxes(title_text="DD (%)", tickformat=".1f", row=2, col=1)
    _dark_layout(fig_p, height=500)
    fig_p.update_layout(margin=dict(l=10, r=10, t=40, b=10))
    st.plotly_chart(fig_p, use_container_width=True, key="pnl_curve")

    # ── Benchmarks ─────────────────────────────────────────────────────────
    st.markdown("##### Comparaison avec benchmarks")
    st.caption("Stratégie RSI vs DCA fixe quotidien vs Lump Sum (buy & hold)")

    # Lump sum — all-in on day 1, 100% BTC
    first_price = ts["btc_close"].iloc[0]
    if first_price > 0 and total_invested > 0:
        lump_btc = total_invested / first_price
        ts["lump_sum"] = lump_btc * ts["btc_close"]
        ts["lump_pct"] = (ts["lump_sum"] / total_invested - 1) * 100
    else:
        ts["lump_sum"] = 0.0
        ts["lump_pct"] = 0.0

    # Fixed DCA — same total, spread equally across every calendar day, 80/20
    n_cal_days = max(1, (ts["date"].max() - ts["date"].min()).days)
    daily_fixed = total_invested / n_cal_days
    fixed_cum_btc = 0.0
    fixed_cum_eth = 0.0
    fixed_rows: list[dict] = []
    for _, row in ts.iterrows():
        bp = row["btc_close"]
        ep = row.get("eth_close", 0)
        if bp > 0:
            fixed_cum_btc += (daily_fixed * 0.80) / bp
        if ep > 0:
            fixed_cum_eth += (daily_fixed * 0.20) / ep
        fixed_rows.append({"date": row["date"], "f_btc": fixed_cum_btc, "f_eth": fixed_cum_eth})

    df_fixed = pd.DataFrame(fixed_rows)
    ts = ts.merge(df_fixed, on="date", how="left")
    ts["fixed_dca"] = ts["f_btc"] * ts["btc_close"] + ts["f_eth"].fillna(0) * ts.get("eth_close", 0)
    ts["fixed_cum_usd"] = (ts.index + 1) * daily_fixed  # progressive invested
    ts["fixed_pct"] = (ts["fixed_dca"] / ts["fixed_cum_usd"].replace(0, np.nan) - 1) * 100

    # RSI strategy % return
    ts["rsi_pct"] = ts["pnl_pct"]

    fig_bm = go.Figure()
    fig_bm.add_trace(
        go.Scatter(
            x=ts["date"], y=ts["rsi_pct"], mode="lines", name="DCA RSI (vous)",
            line=dict(color=C_GREEN, width=2.5),
            hovertemplate="RSI: %{y:+.1f}%<extra></extra>",
        )
    )
    fig_bm.add_trace(
        go.Scatter(
            x=ts["date"], y=ts["fixed_pct"], mode="lines", name="DCA fixe",
            line=dict(color=C_ORANGE, width=2, dash="dash"),
            hovertemplate="Fixe: %{y:+.1f}%<extra></extra>",
        )
    )
    fig_bm.add_trace(
        go.Scatter(
            x=ts["date"], y=ts["lump_pct"], mode="lines", name="Lump sum",
            line=dict(color=C_CYAN, width=2, dash="dot"),
            hovertemplate="Lump: %{y:+.1f}%<extra></extra>",
        )
    )
    fig_bm.add_hline(y=0, line_dash="dot", line_color=C_GRAY, opacity=0.3)

    _dark_layout(fig_bm, height=400)
    fig_bm.update_layout(yaxis_title="Rendement (%)", yaxis_tickformat="+.1f")
    st.plotly_chart(fig_bm, use_container_width=True, key="benchmark")

    # Summary table
    rsi_f = ts["rsi_pct"].iloc[-1] if pd.notna(ts["rsi_pct"].iloc[-1]) else 0
    fixed_f = ts["fixed_pct"].iloc[-1] if pd.notna(ts["fixed_pct"].iloc[-1]) else 0
    lump_f = ts["lump_pct"].iloc[-1] if pd.notna(ts["lump_pct"].iloc[-1]) else 0
    fixed_pnl = ts["fixed_dca"].iloc[-1] - total_invested if total_invested else 0
    lump_pnl = ts["lump_sum"].iloc[-1] - total_invested if total_invested else 0

    bench = [
        {"Stratégie": "🧠 DCA RSI (vous)", "Rendement": f"{rsi_f:+.1f}%", "PnL": f"${total_pnl:+,.2f}", "Max DD": f"{max_dd:.1f}%"},
        {"Stratégie": "📅 DCA fixe", "Rendement": f"{fixed_f:+.1f}%", "PnL": f"${fixed_pnl:+,.2f}", "Max DD": "—"},
        {"Stratégie": "💰 Lump sum", "Rendement": f"{lump_f:+.1f}%", "PnL": f"${lump_pnl:+,.2f}", "Max DD": "—"},
    ]
    st.dataframe(bench, use_container_width=True, hide_index=True)


# ══════════════════════════════════════════════════════════════════════════════
#  Main
# ══════════════════════════════════════════════════════════════════════════════

# Header
st.markdown(
    """
<div style="text-align:center; padding:1rem 0 .5rem 0">
    <h1 style="margin:0; font-weight:800; letter-spacing:-1px">
        <span style="color:#00d4ff">📈</span> DCA Dashboard
    </h1>
    <p style="color:#94a3b8; margin:4px 0 0 0; font-size:.95rem">
        RSI-based daily buying · BTC 90 % + ETH 10 % · MVRV deep-value ×5 · Revolut X maker-only
    </p>
</div>
""",
    unsafe_allow_html=True,
)
st.markdown("")

# ── Load all data once ─────────────────────────────────────────────────────────

raw_buys = _fetch_dca_buys(365)
df_buys = _parse_buys(raw_buys)
hb = _fetch_dca_heartbeat()
prices = _fetch_current_prices()
mvrv = _fetch_mvrv()
df_btc = _fetch_daily_klines("BTCUSDT", 365)
df_eth = _fetch_daily_klines("ETHUSDT", 365)

# ── Tabs ───────────────────────────────────────────────────────────────────────

tab_dash, tab_dca, tab_perf = st.tabs(
    ["📊 Dashboard", "🛒 DCA", "📈 Performance"]
)

with tab_dash:
    render_dashboard(df_buys, hb, prices, df_btc, df_eth, mvrv=mvrv)

with tab_dca:
    render_dca_detail(df_buys, prices)

with tab_perf:
    render_performance(df_buys, prices, df_btc, df_eth)

# ── Footer ─────────────────────────────────────────────────────────────────────

st.divider()
st.caption(
    f"TradeX DCA Dashboard · Firebase (satochi-d38ec) · "
    f"Dernière mise à jour : {datetime.now(DASHBOARD_TZ).strftime('%d/%m/%Y %H:%M:%S')}"
)
