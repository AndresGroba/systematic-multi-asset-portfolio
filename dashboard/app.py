"""Dashboard Streamlit de informes de cartera: streamlit run app.py."""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf

warnings.filterwarnings("ignore")

st.set_page_config(
    page_title="Gestión Cuantitativa — Grupo 4",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown(
    """
    <style>
	    :root {
	        --bg: #F7F9FC;
	        --surface: #FFFFFF;
	        --surface-soft: #F1F5F9;
	        --border: #E2E8F0;
	        --border-strong: #CBD5E1;
	        --text: #0F172A;
	        --muted: #64748B;
	        --dim: #94A3B8;
	        --blue: #2563EB;
	        --blue-soft: #EFF6FF;
	        --green: #15803D;
	        --green-soft: #F0FDF4;
	        --red: #B91C1C;
	        --red-soft: #FEF2F2;
	        --amber: #B45309;
	        --amber-soft: #FFFBEB;
	        --teal: #0F766E;
	        --shadow: 0 18px 45px rgba(15, 23, 42, 0.07);
	    }

	    html, body, [data-testid="stAppViewContainer"], .stApp {
	        background: radial-gradient(circle at top left, rgba(37,99,235,0.08), transparent 30rem),
	                    linear-gradient(180deg, #FFFFFF 0%, var(--bg) 28rem);
	        color: var(--text);
	    }

	    [data-testid="stHeader"] { background: transparent; height: 0; }
	    [data-testid="stToolbar"], [data-testid="stDecoration"], #MainMenu {
	        visibility: hidden;
	        display: none;
	    }

	    /* Main container */
	    .block-container {
	        max-width: 1440px;
	        padding: 1.45rem 2rem 2.25rem 2rem;
	    }

	    /* Page hero */
	    .page-hero {
	        display: flex;
	        align-items: center;
	        gap: 0.9rem;
	        margin: 0 0 0.8rem 0;
	    }
	    .brand-mark {
	        width: 3.25rem;
	        height: 3.25rem;
	        border-radius: 16px;
	        display: inline-flex;
	        align-items: center;
	        justify-content: center;
	        background: linear-gradient(180deg, #FFFFFF 0%, #EFF6FF 100%);
	        border: 1px solid #DDE7F5;
	        box-shadow: 0 12px 32px rgba(37, 99, 235, 0.10);
	        color: #0F766E;
	        font-size: 1.75rem;
	        font-weight: 900;
	    }
		    .page-hero__title {
		        margin: 0;
		        color: var(--text);
		        font-size: clamp(1.75rem, 2.55vw, 2.55rem);
		        line-height: 1.08;
		        font-weight: 760;
		        letter-spacing: -0.035em;
	    }
	    .page-hero__subtitle {
	        margin: 0.45rem 0 0 0;
	        color: var(--muted);
		        font-size: 0.94rem;
		        line-height: 1.45;
		    }
	    .dashboard-controlbar {
	        display: flex;
	        align-items: end;
	        justify-content: space-between;
	        gap: 1rem;
	        margin: 0.1rem 0 1rem 0;
	    }
	    .dashboard-controlbar__title {
	        margin: 0;
	        color: var(--text);
	        font-size: 1.05rem;
	        font-weight: 760;
	    }
	    .dashboard-controlbar__sub {
	        margin-top: 0.18rem;
	        color: var(--muted);
	        font-size: 0.82rem;
	    }
	    .dashboard-context {
	        margin: 0.15rem 0 0.4rem 0;
	    }
	    .dashboard-context__title {
	        color: var(--text);
	        font-size: 1rem;
	        font-weight: 780;
	        letter-spacing: -0.015em;
	    }
	    .dashboard-context__sub {
	        color: var(--muted);
	        font-size: 0.8rem;
	        line-height: 1.35;
	        margin-top: 0.15rem;
	        max-width: 34rem;
	    }
	    .date-pill {
	        display: inline-flex;
	        align-items: center;
	        justify-content: center;
	        min-height: 2.45rem;
	        padding: 0.45rem 0.8rem;
	        border: 1px solid var(--border);
	        border-radius: 12px;
	        background: white;
	        color: #475569;
	        font-size: 0.82rem;
	        font-weight: 650;
	        white-space: nowrap;
	    }

	    /* KPI cards */
		    .kpi-card {
		        background: rgba(255,255,255,0.94);
		        border: 1px solid var(--border);
		        border-left: 4px solid var(--border-strong);
		        border-radius: 14px;
		        padding: 0.95rem 1rem;
		        margin-bottom: 0.45rem;
		        box-shadow: 0 10px 30px rgba(15, 23, 42, 0.045);
		        min-height: 6.85rem;
		    }
	    .kpi-head {
	        display: flex;
	        align-items: center;
	        justify-content: space-between;
	        gap: 0.7rem;
	    }
	    .kpi-icon {
	        color: #0891B2;
	        font-size: 1.35rem;
	        line-height: 1;
	        opacity: 0.95;
	    }
		    .kpi-label {
		        font-size: 0.68rem;
	        text-transform: uppercase;
	        letter-spacing: 0.095em;
	        color: var(--muted);
	        margin-bottom: 0.34rem;
	        font-weight: 750;
	    }
		    .kpi-value {
		        font-size: 1.58rem;
	        font-weight: 760;
	        color: var(--text);
	        line-height: 1.1;
	        font-variant-numeric: tabular-nums;
	        letter-spacing: -0.025em;
	    }
	    .kpi-value.positive { color: var(--green); }
	    .kpi-value.negative { color: var(--red); }
	    .kpi-value.muted    { color: var(--muted); }
	    .kpi-sub {
	        font-size: 0.74rem;
	        color: var(--muted);
	        margin-top: 0.34rem;
	        line-height: 1.35;
	    }

	    /* Section headers */
		    .section-header {
	        font-size: 0.78rem;
	        text-transform: uppercase;
	        letter-spacing: 0.115em;
	        color: #1E40AF;
	        border-bottom: 1px solid var(--border);
	        padding-bottom: 0.55rem;
		        margin: 1.1rem 0 0.75rem 0;
		        font-weight: 800;
		    }

	    /* Alert banners */
	    .alert-warning {
	        background: var(--amber-soft);
	        border: 1px solid #FCD34D;
	        border-radius: 14px;
	        padding: 0.7rem 1rem;
	        color: var(--amber);
	        font-size: 0.85rem;
	        margin-bottom: 0.8rem;
	    }

	    /* Tab styling */
	    .stTabs [data-baseweb="tab-list"] {
	        gap: 6px;
	        background: rgba(255,255,255,0.86);
	        border: 1px solid var(--border);
	        border-radius: 16px;
	        display: inline-flex;
	        width: fit-content;
	        max-width: 100%;
	        padding: 5px;
	        box-shadow: 0 8px 24px rgba(15, 23, 42, 0.04);
	    }
	    .stTabs [data-baseweb="tab"] {
	        background: transparent;
	        border-radius: 12px;
	        min-height: 2rem;
	        padding: 0.42rem 0.7rem;
	        color: var(--muted);
	        font-size: 0.85rem;
	        font-weight: 650;
	    }
	    .stTabs [data-baseweb="tab"] p { margin: 0; }
	    .stTabs [aria-selected="true"] {
	        background: var(--blue);
	        color: white;
	    }

	    /* Dataframe */
	    .stDataFrame {
	        border: 1px solid var(--border);
	        border-radius: 16px;
	        overflow: hidden;
	        box-shadow: 0 10px 30px rgba(15, 23, 42, 0.04);
	    }

	    [data-testid="stPlotlyChart"] {
	        background: white;
	        border: 1px solid var(--border);
	        border-radius: 14px;
	        padding: 0.35rem;
	        box-shadow: 0 12px 35px rgba(15, 23, 42, 0.055);
	    }

	    .benchmark-panel {
	        background: linear-gradient(135deg, #FFFFFF 0%, #F8FAFC 100%);
	        border: 1px solid var(--border);
	        border-radius: 20px;
	        padding: 0.95rem 1rem 0.85rem 1rem;
	        box-shadow: 0 14px 40px rgba(15, 23, 42, 0.06);
	    }
	    .benchmark-panel__label {
	        color: var(--muted);
	        font-size: 0.69rem;
	        font-weight: 800;
	        letter-spacing: 0.105em;
	        text-transform: uppercase;
	        margin-bottom: 0.35rem;
	    }
	    .compare-card {
	        background: white;
	        border: 1px solid var(--border);
	        border-radius: 14px;
	        padding: 1rem 1.05rem;
	        min-height: 24.3rem;
	        box-shadow: 0 12px 35px rgba(15,23,42,0.055);
	    }
	    .compare-title {
	        color: var(--text);
	        font-weight: 780;
	        font-size: 1rem;
	        margin-bottom: 0.55rem;
	    }
	    .compare-sub {
	        color: var(--muted);
	        font-size: 0.76rem;
	        margin-bottom: 0.65rem;
	        line-height: 1.35;
	    }
	    .mini-table {
	        width: 100%;
	        border-collapse: collapse;
	        font-size: 0.82rem;
	    }
	    .mini-table th {
	        color: #2563EB;
	        font-size: 0.68rem;
	        text-transform: uppercase;
	        letter-spacing: 0.08em;
	        text-align: right;
	        padding: 0.45rem 0.25rem;
	        border-bottom: 1px solid var(--border);
	    }
	    .mini-table th:first-child { text-align: left; color: var(--muted); }
	    .mini-table td {
	        padding: 0.55rem 0.25rem;
	        border-bottom: 1px solid var(--border);
	        text-align: right;
	        color: var(--text);
	        font-variant-numeric: tabular-nums;
	    }
	    .mini-table td:first-child {
	        text-align: left;
	        color: #334155;
	        font-variant-numeric: normal;
	    }
	    .positive-text { color: var(--green) !important; }
	    .negative-text { color: var(--red) !important; }
	    .muted-text { color: var(--muted) !important; }
	    .benchmark-panel__help {
	        margin-top: 0.45rem;
	        color: var(--muted);
	        font-size: 0.74rem;
	        line-height: 1.35;
	    }

	    /* Selector */
	    .selected-ticker-banner {
	        background: var(--blue-soft);
	        border: 1px solid #BFDBFE;
	        border-radius: 14px;
	        padding: 0.5rem 1rem;
	        color: #1D4ED8;
	        font-size: 0.85rem;
	        margin-bottom: 0.8rem;
	    }

	    div[data-testid="stRadio"] label, div[data-testid="stSelectbox"] label {
	        color: var(--muted) !important;
	        font-weight: 700;
	    }
	    [data-baseweb="select"] > div {
	        border-color: var(--border) !important;
	        background: white !important;
	        border-radius: 12px !important;
	    }
	    button[kind="secondary"], button[kind="primary"] {
	        border-radius: 12px !important;
	    }
	    </style>
    """,
    unsafe_allow_html=True,
)


# Helpers

def _kpi(
    label: str,
    value: str,
    sub: str = "",
    cls: str = "",
    accent: str = "#CBD5E1",
    icon: str = "",
) -> str:
    sub_html = f'<div class="kpi-sub">{sub}</div>' if sub else ""
    icon_html = f'<div class="kpi-icon">{icon}</div>' if icon else ""
    return (
        f'<div class="kpi-card" style="border-left-color:{accent};">'
        f'<div class="kpi-head"><div class="kpi-label">{label}</div>{icon_html}</div>'
        f'<div class="kpi-value {cls}">{value}</div>'
        f'{sub_html}</div>'
    )


def _bm_kpi(
    label: str,
    strat_val: str,
    bm_val: str,
    benchmark_short: str,
    delta: float | None,
    delta_str: str,
    higher_is_better: bool = True,
) -> str:
    """Tarjeta KPI estrategia vs benchmark; delta verde si favorable, rojo si no."""
    if strat_val.startswith("+"):
        strat_color = "#15803D"
    elif strat_val.startswith("-"):
        strat_color = "#B91C1C"
    else:
        strat_color = "#0F172A"

    if delta is not None and not (isinstance(delta, float) and np.isnan(delta)):
        good = (delta > 0 and higher_is_better) or (delta < 0 and not higher_is_better)
        d_color = "#15803D" if good else ("#B91C1C" if not good and delta != 0 else "#64748B")
        arrow   = " ▲" if delta > 0 else (" ▼" if delta < 0 else "")
        delta_block = (
            f'<div style="border-top:1px solid #E2E8F0;margin-top:0.45rem;'
            f'padding-top:0.22rem;display:flex;justify-content:space-between;'
            f'align-items:center;">'
            f'<span style="font-size:0.65rem;color:#64748B;text-transform:uppercase;'
            f'letter-spacing:0.04em;">Diferencia</span>'
            f'<span style="font-size:0.78rem;font-weight:700;color:{d_color};'
            f'font-variant-numeric:tabular-nums;">{delta_str}{arrow}</span>'
            f'</div>'
        )
    else:
        delta_block = ""

    return (
        f'<div class="kpi-card" style="border-left-color:#64748B;">'
        f'<div class="kpi-label">{label}</div>'
        f'<div style="display:flex;justify-content:space-between;align-items:flex-end;'
        f'margin-top:0.25rem;gap:0.5rem;">'
        # Columna estrategia
        f'<div style="flex:1;">'
        f'<div style="font-size:0.62rem;color:#64748B;margin-bottom:3px;'
        f'text-transform:uppercase;letter-spacing:0.04em;">Estrategia</div>'
        f'<div style="font-size:1.1rem;font-weight:700;color:{strat_color};'
        f'font-variant-numeric:tabular-nums;line-height:1.1;">{strat_val}</div>'
        f'</div>'
        # Columna benchmark
        f'<div style="flex:1;text-align:right;">'
        f'<div style="font-size:0.62rem;color:#64748B;margin-bottom:3px;'
        f'text-transform:uppercase;letter-spacing:0.04em;">{benchmark_short}</div>'
        f'<div style="font-size:1.1rem;font-weight:650;color:#475569;'
        f'font-variant-numeric:tabular-nums;line-height:1.1;">{bm_val}</div>'
        f'</div>'
        f'</div>'
        f'{delta_block}'
        f'</div>'
    )


def _color_cls(val: float) -> str:
    if pd.isna(val):
        return "muted"
    return "positive" if val >= 0 else "negative"


def _pnl_breakdown_sub(ret: float, gross: float, costs: float) -> str:
    """Sub-bloque HTML: retorno % + desglose bruto/costes."""
    ret_str   = fmt_pct(ret)
    gross_str = fmt_eur(gross, compact=True)
    cost_str  = fmt_eur(costs, compact=True)
    gross_c   = "#15803D" if gross >= 0 else "#B91C1C"
    # Los costes siempre restan: rojo salvo si son nulos
    cost_c    = "#B91C1C" if costs > 0.005 else "#64748B"
    divider   = "1px solid #E2E8F0"
    return (
        f'<div style="margin-bottom:0.15rem;">Retorno:&nbsp;<b>{ret_str}</b></div>'
        f'<div style="border-top:{divider};margin-top:0.28rem;padding-top:0.26rem;">'
        f'<div style="display:flex;justify-content:space-between;margin-bottom:0.12rem;">'
        f'<span style="color:#64748B;">Bruto</span>'
        f'<span style="color:{gross_c};font-weight:600;">{gross_str}</span>'
        f'</div>'
        f'<div style="display:flex;justify-content:space-between;">'
        f'<span style="color:#64748B;">Costes</span>'
        f'<span style="color:{cost_c};font-weight:600;">−{cost_str}</span>'
        f'</div>'
        f'</div>'
    )


def fmt_eur(v, decimals=2, compact=True):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "N/A"
    sign = "-" if v < 0 else ""
    av = abs(v)
    if compact and av >= 1_000_000:
        return f"{sign}€{av/1_000_000:.{decimals}f}M"
    if compact and av >= 1_000:
        return f"{sign}€{av/1_000:.{decimals}f}K"
    return f"{sign}€{av:,.{decimals}f}"


def fmt_eur_no_neg_zero(v, decimals=2, compact=True):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "N/A"
    if abs(float(v)) < 0.5 * 10 ** (-decimals):
        v = 0.0
    return fmt_eur(v, decimals=decimals, compact=compact)


def fmt_pct(v, decimals=2):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "N/A"
    return f"{v*100:+.{decimals}f}%"


def fmt_n(v, decimals=2):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "N/A"
    return f"{v:,.{decimals}f}"


# Config plotly: barra visible + PNG de alta resolución (escala 3) con nombre propio.
# Cada gráfico se descarga con su nombre para que la memoria no acumule newplot.png.
def _export_cfg(filename: str) -> dict:
    safe = "".join(c if (c.isalnum() or c in "-_") else "_" for c in filename)
    return {
        "displaylogo": False,
        "displayModeBar": True,
        "modeBarButtonsToRemove": ["lasso2d", "select2d"],
        "toImageButtonOptions": {"format": "png", "scale": 3, "filename": safe},
    }


# Corte temporal de la cartera viva (toggle del Panel)
LIVE_END_TODAY_LABEL = "Hasta hoy"
LIVE_END_LABEL = "Hasta 14-may-26"


# Carga de datos (cacheada)

@st.cache_data(ttl=3600, show_spinner=False)
def load_data(end_date_str: str | None = None):
    from src.data_loader import (
        load_historial, load_operativa, load_operativa_fase3, build_operations_table,
    )
    from src.portfolio_engine import build_portfolio

    historial_df = load_historial()
    operativa_df = load_operativa()
    operativa_fase3_df = load_operativa_fase3()
    ops_table = build_operations_table(historial_df, operativa_df, operativa_fase3_df)
    portfolio_df, weights_long, failed_tickers = build_portfolio(
        historial_df, operativa_df, operativa_fase3_df, end_date=end_date_str
    )
    return historial_df, operativa_df, ops_table, portfolio_df, weights_long, failed_tickers


@st.cache_data(ttl=3600, show_spinner=False)
def load_prices_for_tab(tickers: tuple, start: str, end: str):
    from src.portfolio_engine import download_prices
    prices, failed = download_prices(list(tickers), start, end)
    return prices, failed


@st.cache_data(ttl=3600, show_spinner=False)
def load_benchmark_nav(ticker: str, start: str, end: str, portfolio_index: tuple) -> "pd.Series":
    """NAV buy&hold del benchmark alineado a las fechas de la cartera."""
    from src.portfolio_engine import download_prices
    from src.utils import CAPITAL_INICIAL

    prices, _ = download_prices([ticker], start, end)
    if ticker not in prices.columns:
        return pd.Series(dtype=float)

    # ffill sobre la serie completa (el caller descarga con colchón previo) y luego
    # restringir al índice de la cartera: el día-0 toma el último precio real <= su
    # fecha. Sin bfill — back-fillar metería un precio futuro como base (look-ahead).
    full = prices[ticker].dropna()
    if full.empty:
        return pd.Series(dtype=float)
    idx = pd.DatetimeIndex(list(portfolio_index))
    price = full.reindex(full.index.union(idx)).ffill().reindex(idx)
    price_start = float(price.iloc[0])
    if not np.isfinite(price_start) or price_start == 0:
        return pd.Series(dtype=float)

    return (CAPITAL_INICIAL * price / price_start).rename(f"{ticker}_nav")


def main():
    st.markdown(
	        """
	        <div class="page-hero">
	          <div class="brand-mark">↗</div>
	          <div>
	            <h1 class="page-hero__title">Gestión Cuantitativa</h1>
	            <p class="page-hero__subtitle">
	              Panel de gestión de activos
	            </p>
	          </div>
	        </div>
	        """,
	        unsafe_allow_html=True,
	    )

    # Corte temporal de la cartera viva (lo fija el toggle del Panel vía session_state):
    # "hoy" = última sesión disponible; "live_end" = 2026-05-14, fin del periodo en vivo
    # que usa el contrafactual (permite comparar real vs contrafactual en la misma ventana).
    LIVE_END = "2026-05-14"
    end_date_str = LIVE_END if st.session_state.get("live_end_choice") == LIVE_END_LABEL else None

    with st.spinner("Cargando datos de cartera y precios de mercado..."):
        try:
            (
                historial_df, operativa_df, ops_table,
                portfolio_df, weights_long, failed_tickers,
            ) = load_data(end_date_str)
            data_ok = True
        except Exception as e:
            st.error(f"Error al cargar los datos: {e}")
            st.stop()

    from src.metrics import compute_metrics, compute_drawdown_series
    metrics = compute_metrics(portfolio_df)

    # EURUSD=X es interno (conversión a EUR), no una posición: no se avisa de él aquí
    # (si falla puntualmente cae a un fallback de FX). Solo se avisa de holdings fallidos.
    holdings_failed = [t for t in failed_tickers if t != "EURUSD=X"]
    if holdings_failed:
        st.markdown(
            f'<div class="alert-warning">⚠ Los siguientes tickers no pudieron obtenerse de '
            f'yfinance y se mantienen al precio de ejecución: '
            f'<b>{", ".join(holdings_failed)}</b>. '
            f'Las métricas pueden ser aproximadas.</div>',
            unsafe_allow_html=True,
        )

    # Live (Panel/Operaciones/Precios) + Estrategia (Backtest/Validación) + Parámetros
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(
        ["Panel", "Operaciones", "Precios históricos", "Backtest", "Validación", "Parámetros"]
    )

    with tab1:
        _render_dashboard(portfolio_df, weights_long, metrics)

    with tab2:
        _render_operations(ops_table)

    with tab3:
        _render_prices(portfolio_df, weights_long)

    with tab4:
        _render_backtest()

    with tab5:
        _render_validacion()

    with tab6:
        _render_parametros()


def _render_dashboard(portfolio_df, weights_long, metrics):
    from src.utils import (
        CAPITAL_INICIAL,
        OPERATIVA_DATE,
        RISK_FREE_RATE_ANNUAL,
        TRADING_DAYS_PER_YEAR,
        daily_risk_free_rate,
        ticker_category,
    )
    from src.benchmarks import benchmark_label_options, get_benchmark_config
    from src.plots import (
        plot_equity_curve,
        plot_daily_pnl,
        plot_drawdown,
        plot_allocation_donut,
        plot_etf_pnl_attribution,
        plot_composition_treemap,
        plot_holdings_bar,
        plot_weights_area,
        plot_cumulative_costs,
    )
    from src.metrics import compute_drawdown_series

    _nav     = metrics.get("nav_current")
    _pnl     = metrics.get("pnl_cum", 0.0) or 0.0
    _dp      = metrics.get("daily_pnl", 0.0) or 0.0
    _ret     = metrics.get("ret_cum", 0.0) or 0.0
    _cagr    = metrics.get("cagr")
    _vol     = metrics.get("vol_ann")
    _sharpe  = metrics.get("sharpe")
    _sortino = metrics.get("sortino")
    _downside_vol = metrics.get("downside_vol_ann")
    _mdd     = metrics.get("max_dd", 0.0) or 0.0
    _calmar  = metrics.get("calmar")
    _n_etfs  = metrics.get("n_etfs", 0)
    _cash    = metrics.get("cash_current", 0.0) or 0.0
    _costs   = metrics.get("total_costs", 0.0) or 0.0
    _drag    = metrics.get("cost_drag_pct", 0.0) or 0.0
    _rf      = metrics.get("risk_free_rate_annual", RISK_FREE_RATE_ANNUAL) or 0.0
    _rf_daily = daily_risk_free_rate(_rf)

    nav_idx = portfolio_df["nav"].dropna().index
    last_date_label = nav_idx[-1].strftime("%d/%m/%Y") if len(nav_idx) else "—"

    def _safe_pct(v, sign: bool = True):
        txt = fmt_pct(v) if v is not None and not (isinstance(v, float) and np.isnan(v)) else "N/A"
        return txt if sign else txt.replace("+", "")

    def _safe_n(v, d=2):
        return fmt_n(v, d) if v is not None and not (isinstance(v, float) and np.isnan(v)) else "N/A"

    def _signed_cls(v: float | None) -> str:
        if v is None or (isinstance(v, float) and np.isnan(v)):
            return "muted-text"
        return "positive-text" if v >= 0 else "negative-text"

    rf_label = _safe_pct(_rf, sign=False)

    def _comparison_card(active_benchmark, bm_nav_series: pd.Series) -> str:
        nav_s = portfolio_df["nav"].dropna()
        bm_nav = bm_nav_series.reindex(nav_s.index).ffill()
        bm_ret = float(bm_nav.iloc[-1]) / float(bm_nav.iloc[0]) - 1.0
        bm_pnl = float(bm_nav.iloc[-1]) - CAPITAL_INICIAL
        bm_dd_s = compute_drawdown_series(bm_nav)
        bm_maxdd = float(bm_dd_s.min())
        n_days = len(bm_nav)
        bm_cagr = (float(bm_nav.iloc[-1]) / CAPITAL_INICIAL) ** (TRADING_DAYS_PER_YEAR / n_days) - 1.0 if n_days >= 2 else np.nan

        strat_pct = nav_s.pct_change().dropna()
        bm_pct = bm_nav.pct_change().reindex(strat_pct.index).dropna()
        bm_vol = float(bm_pct.std() * np.sqrt(TRADING_DAYS_PER_YEAR)) if len(bm_pct) >= 2 else np.nan
        bm_excess = bm_pct - _rf_daily
        bm_sharpe = float(bm_excess.mean() / bm_pct.std() * np.sqrt(TRADING_DAYS_PER_YEAR)) if bm_pct.std() > 0 else np.nan
        # Target semideviation over the full series (consistent with compute_metrics).
        bm_downside = bm_excess.clip(upper=0.0)
        bm_downside_vol = (
            float(np.sqrt((bm_downside ** 2).mean()) * np.sqrt(TRADING_DAYS_PER_YEAR))
            if len(bm_excess) >= 1
            else np.nan
        )
        bm_sortino = (
            float(bm_excess.mean() * TRADING_DAYS_PER_YEAR / bm_downside_vol)
            if bm_downside_vol and not np.isnan(bm_downside_vol) and bm_downside_vol > 0
            else np.nan
        )

        rows = [
            ("Rentabilidad total", _safe_pct(_ret), _safe_pct(bm_ret), _ret, bm_ret),
            ("Resultado", fmt_eur(_pnl, compact=True), fmt_eur(bm_pnl, compact=True), _pnl, bm_pnl),
            ("Volatilidad anual", _safe_pct(_vol, sign=False), _safe_pct(bm_vol, sign=False), None, None),
            ("Volatilidad bajista", _safe_pct(_downside_vol, sign=False), _safe_pct(bm_downside_vol, sign=False), None, None),
            ("Sharpe", _safe_n(_sharpe), _safe_n(bm_sharpe), None, None),
            ("Sortino", _safe_n(_sortino), _safe_n(bm_sortino), None, None),
            ("Caída máxima", _safe_pct(_mdd), _safe_pct(bm_maxdd), _mdd, bm_maxdd),
        ]
        body = "".join(
            f"<tr><td>{label}</td>"
            f"<td class='{_signed_cls(s_raw) if s_raw is not None else ''}'>{s_val}</td>"
            f"<td class='{_signed_cls(b_raw) if b_raw is not None else ''}'>{b_val}</td></tr>"
            for label, s_val, b_val, s_raw, b_raw in rows
        )
        return f"""
        <div class="compare-card">
          <div class="compare-title">Comparativa vs {active_benchmark.short_label}</div>
          <div class="compare-sub">La cartera se compara contra una inversión pasiva de referencia con el mismo capital inicial.</div>
          <table class="mini-table">
            <thead><tr><th>Indicador</th><th>Cartera</th><th>Referencia</th></tr></thead>
            <tbody>{body}</tbody>
          </table>
          <div class="compare-sub" style="margin-top:0.7rem;margin-bottom:0;">Datos en euros. Tasa libre de riesgo: {rf_label} anual.</div>
        </div>
        """

    def _phase_card() -> str:
        p1_net = metrics.get("phase1_pnl", 0.0) or 0.0
        p1_ret = metrics.get("phase1_ret", 0.0) or 0.0
        p2_net = metrics.get("phase2_pnl", 0.0) or 0.0
        p2_ret = metrics.get("phase2_ret", 0.0) or 0.0
        rows = [
            ("Etapa inicial", fmt_eur(p1_net, compact=True), _safe_pct(p1_ret), p1_net),
            ("Etapa multi-activo", fmt_eur(p2_net, compact=True), _safe_pct(p2_ret), p2_net),
            ("Resultado total", fmt_eur(_pnl, compact=True), _safe_pct(_ret), _pnl),
        ]
        body = "".join(
            f"<tr><td>{label}</td><td class='{_signed_cls(raw)}'>{val}</td><td class='{_signed_cls(raw)}'>{ret}</td></tr>"
            for label, val, ret, raw in rows
        )
        return f"""
        <div class="compare-card" style="min-height:18.2rem;">
          <div class="compare-title">Resultado por etapa</div>
          <div class="compare-sub">Separación simple entre la cartera inicial y la cartera multi-activo actual.</div>
          <table class="mini-table">
            <thead><tr><th>Etapa</th><th>Euros</th><th>%</th></tr></thead>
            <tbody>{body}</tbody>
          </table>
          <div class="compare-sub" style="margin-top:0.7rem;margin-bottom:0;">Costes acumulados: {fmt_eur(_costs, compact=True)} · impacto {_safe_pct(_drag, sign=False)}.</div>
        </div>
        """

    # Selector superior: un único índice de referencia activo
    top_intro, top_bm, top_date = st.columns([1.3, 1.25, 1.05], vertical_alignment="bottom")
    with top_intro:
        st.markdown(
            """
            <div class="dashboard-context">
              <div class="dashboard-context__title">Resumen de cartera</div>
              <div class="dashboard-context__sub">Indicadores principales, comparación contra el índice elegido y lectura rápida del riesgo.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with top_bm:
        bm_options = benchmark_label_options(include_none=False)
        default_bm_label = next(
            (label for label, key in bm_options.items() if key == "msci_world"),
            list(bm_options.keys())[0],
        )
        st.markdown('<div class="benchmark-panel__label">Índice de referencia</div>', unsafe_allow_html=True)
        bm_choice = st.segmented_control(
            "Índice de referencia",
            options=list(bm_options.keys()),
            default=default_bm_label,
            label_visibility="collapsed",
            key="benchmark_selector_msci_default",
            help="Seleccioná un solo índice para comparar la cartera.",
            width="stretch",
        )
    with top_date:
        st.markdown('<div class="benchmark-panel__label">Periodo de datos</div>', unsafe_allow_html=True)
        st.segmented_control(
            "Periodo de datos",
            options=[LIVE_END_TODAY_LABEL, LIVE_END_LABEL],
            default=LIVE_END_TODAY_LABEL,
            key="live_end_choice",
            label_visibility="collapsed",
            help="«Hasta hoy»: última sesión de mercado disponible. «Hasta 14-may-26»: "
                 "fin del periodo en vivo (14 may 2026), la misma ventana que usa el "
                 "contrafactual del Backtest — útil para comparar la cartera real con él.",
            width="stretch",
        )
        st.caption(f"Última sesión: {last_date_label}")

    selected_bm_key = bm_options.get(bm_choice or list(bm_options.keys())[0])
    active_benchmark = get_benchmark_config(selected_bm_key)

    bm_nav_series: "pd.Series | None" = None
    if len(nav_idx):
        # Colchón previo para que el ancla día-0 del benchmark sea siempre un
        # precio real anterior (ffill), nunca uno futuro (sin bfill).
        start_str = (nav_idx[0] - pd.Timedelta(days=7)).strftime("%Y-%m-%d")
        end_str = (nav_idx[-1] + pd.Timedelta(days=2)).strftime("%Y-%m-%d")
        with st.spinner(f"Descargando {active_benchmark.short_label}..."):
            bm_nav_series = load_benchmark_nav(
                active_benchmark.ticker, start_str, end_str, tuple(nav_idx)
            )
        if bm_nav_series is None or bm_nav_series.empty:
            st.warning(f"No se pudieron obtener datos de {active_benchmark.short_label}. Se muestra la cartera sin comparación.")
            bm_nav_series = None

    k1, k2, k3, k4, k5 = st.columns(5)
    with k1:
        st.markdown(_kpi("Valor de cartera", fmt_eur(_nav, compact=False), sub=f"Dato al {last_date_label}", accent="#2563EB", icon="◌"), unsafe_allow_html=True)
    with k2:
        st.markdown(_kpi("Resultado acumulado", fmt_eur(_pnl, compact=True), sub="Desde el inicio de la cartera", cls=_color_cls(_pnl), accent="#15803D" if _pnl >= 0 else "#B91C1C", icon="↗"), unsafe_allow_html=True)
    with k3:
        st.markdown(_kpi("Rentabilidad total", _safe_pct(_ret), sub=f"Equivale a {fmt_eur(_pnl, compact=True)}", cls=_color_cls(_ret), accent="#15803D" if _ret >= 0 else "#B91C1C", icon="%"), unsafe_allow_html=True)
    with k4:
        st.markdown(_kpi("Sharpe", _safe_n(_sharpe), sub=f"Exceso sobre tasa libre de riesgo: {rf_label} anual", cls="muted", accent="#0891B2", icon="Σ"), unsafe_allow_html=True)
    with k5:
        st.markdown(_kpi("Caída máxima", _safe_pct(_mdd), sub="Mayor caída desde un máximo", cls="negative" if _mdd < -0.005 else "muted", accent="#B91C1C", icon="↘"), unsafe_allow_html=True)

    main_chart, compare = st.columns([2.35, 1.0])
    with main_chart:
        fig_equity = plot_equity_curve(
            portfolio_df,
            bm_nav=bm_nav_series,
            bm_label=active_benchmark.label,
            bm_short_label=active_benchmark.short_label,
        )
        st.plotly_chart(fig_equity, width="stretch", config=_export_cfg("panel_curva_valor"))
    with compare:
        if bm_nav_series is not None and not bm_nav_series.empty:
            st.markdown(_comparison_card(active_benchmark, bm_nav_series), unsafe_allow_html=True)
        else:
            st.markdown(
                '<div class="compare-card"><div class="compare-title">Comparativa no disponible</div>'
                '<div class="compare-sub">No se pudieron obtener precios del índice de referencia.</div></div>',
                unsafe_allow_html=True,
            )

    # Segunda fila: riesgo, asignación, resultado diario y etapas
    dd_series = compute_drawdown_series(portfolio_df["nav"].dropna())
    bm_dd_for_plot = (
        compute_drawdown_series(bm_nav_series.reindex(portfolio_df["nav"].dropna().index).ffill())
        if bm_nav_series is not None and not bm_nav_series.empty
        else None
    )

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        fig_dd = plot_drawdown(dd_series, bm_dd=bm_dd_for_plot, bm_label=active_benchmark.short_label)
        st.plotly_chart(fig_dd, width="stretch", config=_export_cfg("panel_drawdown"))
    with c2:
        fig_alloc = plot_allocation_donut(weights_long)
        st.plotly_chart(fig_alloc, width="stretch", config=_export_cfg("panel_asignacion_donut"))
    with c3:
        fig_pnl = plot_daily_pnl(portfolio_df)
        st.plotly_chart(fig_pnl, width="stretch", config=_export_cfg("panel_resultado_diario"))
    with c4:
        st.markdown(_phase_card(), unsafe_allow_html=True)

    if not weights_long.empty:
        attr_title, attr_controls = st.columns([1.7, 1.0], vertical_alignment="bottom")
        with attr_title:
            st.markdown('<div class="section-header">Atribución de resultado por ETF</div>', unsafe_allow_html=True)
        _p2_dates = sorted(weights_long[weights_long["date"] >= OPERATIVA_DATE]["date"].unique())
        _attr_selected_date = None
        with attr_controls:
            mode_col, date_col = st.columns([1, 1])
            with mode_col:
                _attr_mode = st.radio(
                    "Periodo",
                    options=["Acumulado", "Día concreto"],
                    horizontal=True,
                    key="attr_mode",
                    help="Acumulado: resultado desde el inicio de la cartera multi-activo. Día concreto: resultado de una sesión.",
                )
            with date_col:
                if _attr_mode == "Día concreto" and len(_p2_dates) > 1:
                    _date_opts = {d.strftime("%d/%m/%Y"): d for d in _p2_dates[1:]}
                    _sel_label = st.selectbox(
                        "Elegí fecha",
                        options=list(_date_opts.keys()),
                        index=len(_date_opts) - 1,
                        key="attr_date",
                        label_visibility="collapsed",
                    )
                    _attr_selected_date = _date_opts[_sel_label]
                elif _attr_mode == "Día concreto":
                    st.caption("Sin datos suficientes para vista diaria.")

        fig_attr = plot_etf_pnl_attribution(weights_long, selected_date=_attr_selected_date)
        st.plotly_chart(fig_attr, width="stretch", config=_export_cfg("panel_atribucion_etf"))

        st.markdown('<div class="section-header">Composición de la cartera</div>', unsafe_allow_html=True)
        comp_left, comp_right = st.columns([1.05, 1.0], vertical_alignment="top")
        with comp_left:
            fig_tree, _ = plot_composition_treemap(weights_long)
            st.plotly_chart(fig_tree, width="stretch", config=_export_cfg("panel_composicion_treemap"))
        with comp_right:
            comp_category_values = sorted(
                weights_long["ticker"]
                .map(ticker_category)
                .dropna()
                .unique()
                .tolist()
            )
            comp_categories = ["Todas"] + comp_category_values
            selected_comp_category = st.selectbox(
                "Detalle por categoría",
                options=comp_categories,
                index=0,
                key="composition_category_filter",
                help="Elegí una categoría para ver sus posiciones. En “Todas” se muestran las posiciones principales para que el gráfico no quede saturado.",
            )
            fig_holdings = plot_holdings_bar(
                weights_long,
                category=selected_comp_category,
                top_n=12 if selected_comp_category == "Todas" else None,
            )
            st.plotly_chart(fig_holdings, width="stretch", config=_export_cfg("panel_posiciones"))

        fig_weights = plot_weights_area(weights_long)
        st.plotly_chart(fig_weights, width="stretch", config=_export_cfg("panel_evolucion_pesos"))
        st.caption(
            "ℹ️ La cartera tiene tres fases: táctica inicial (IUSE+XEON) → despliegue "
            "multi-activo (10-abr) → rebalanceo (13-may). El **monetario XEON** del tramo "
            "final (~15%) es el colchón de liquidez del rebalanceo: la orden de XEON del "
            "13-may venía mal dimensionada (error 1 USD = 1 EUR), así que su importe se "
            "reconstruye desde la continuidad de NAV al precio EUR real."
        )

    if "cum_cost" in portfolio_df.columns and portfolio_df["cum_cost"].fillna(0).abs().sum() > 0:
        st.markdown('<div class="section-header">Costes de transacción</div>', unsafe_allow_html=True)
        st.plotly_chart(plot_cumulative_costs(portfolio_df), width="stretch",
                        config=_export_cfg("panel_costes_acumulados"))

def _ops_badge_decision(decision: str) -> str:
    """Píldora de color para COMPRAR / VENDER."""
    if decision == "COMPRAR":
        return (
            '<span style="display:inline-block;padding:2px 10px;border-radius:12px;'
            'background:#F0FDF4;color:#15803D;font-weight:700;'
            'font-size:0.78rem;letter-spacing:0.04em;">▲ COMPRAR</span>'
        )
    elif decision == "VENDER":
        return (
            '<span style="display:inline-block;padding:2px 10px;border-radius:12px;'
            'background:#FEF2F2;color:#B91C1C;font-weight:700;'
            'font-size:0.78rem;letter-spacing:0.04em;">▼ VENDER</span>'
        )
    return decision


def _ops_badge_regime(regime: str) -> str:
    """Etiqueta de color para el régimen de la estrategia."""
    if "Multi" in regime:
        return (
            '<span style="display:inline-block;padding:1px 7px;border-radius:8px;'
            'background:#EFF6FF;color:#1D4ED8;font-size:0.72rem;font-weight:650;">'
            'Multi-activo</span>'
        )
    return (
        '<span style="display:inline-block;padding:1px 7px;border-radius:8px;'
        'background:#F1F5F9;color:#475569;font-size:0.72rem;font-weight:650;">'
        'IUSE + XEON</span>'
    )


def _render_ops_html_table(df: pd.DataFrame) -> str:
    """Tabla HTML con estilo para las operaciones."""
    header_style = (
        "background:#F8FAFC;color:#64748B;font-size:0.72rem;text-transform:uppercase;"
        "letter-spacing:0.08em;padding:10px 14px;border-bottom:1px solid #E2E8F0;"
        "white-space:nowrap;"
    )
    cols = {
        "fecha":            "Fecha",
        "ticker":           "Ticker",
        "decision":         "Tipo",
        "cantidad":         "Cantidad",
        "precio":           "Precio Ref.",
        "precio_ejecutado": "Precio Ejec.",
        "importe":          "Importe",
        "coste":            "Coste",
        "regimen":          "Régimen",
    }
    present = {k: v for k, v in cols.items() if k in df.columns}

    th_row = "".join(
        f'<th style="{header_style}">{label}</th>'
        for label in present.values()
    )

    rows_html = []
    for _, row in df.iterrows():
        decision = row.get("decision", "")
        is_buy = decision == "COMPRAR"
        is_sell = decision == "VENDER"

        row_bg = (
            "#F7FEF9" if is_buy
            else "#FFF7F7" if is_sell
            else "transparent"
        )
        row_border = (
            "border-left:3px solid #15803D" if is_buy
            else "border-left:3px solid #B91C1C" if is_sell
            else "border-left:3px solid transparent"
        )

        row_style = (
            f"background:{row_bg};{row_border};"
            "border-bottom:1px solid #E2E8F0;"
        )
        cell_style = "padding:9px 14px;font-size:0.83rem;color:#0F172A;white-space:nowrap;"
        muted = "color:#64748B;"

        def _cell(key):
            val = row.get(key)
            if key == "fecha":
                v = pd.to_datetime(val).strftime("%d/%m/%Y") if pd.notna(val) else "—"
                return f'<td style="{cell_style}{muted}">{v}</td>'
            if key == "ticker":
                return f'<td style="{cell_style}font-weight:600;">{val}</td>'
            if key == "decision":
                return f'<td style="{cell_style}">{_ops_badge_decision(str(val))}</td>'
            if key == "cantidad":
                if pd.notna(val):
                    color = "#15803D" if val > 0 else "#B91C1C"
                    sign = "+" if val > 0 else ""
                    return f'<td style="{cell_style}color:{color};font-variant-numeric:tabular-nums;">{sign}{val:,.2f}</td>'
                return f'<td style="{cell_style}{muted}">—</td>'
            if key in ("precio", "precio_ejecutado"):
                v = f"€{val:,.4f}" if pd.notna(val) else "—"
                return f'<td style="{cell_style}{muted}font-variant-numeric:tabular-nums;">{v}</td>'
            if key in ("importe", "coste"):
                if pd.notna(val) and val != 0:
                    return f'<td style="{cell_style}font-variant-numeric:tabular-nums;">€{val:,.2f}</td>'
                return f'<td style="{cell_style}{muted}">—</td>'
            if key == "regimen":
                return f'<td style="{cell_style}">{_ops_badge_regime(str(val))}</td>'
            return f'<td style="{cell_style}{muted}">{val}</td>'

        cells = "".join(_cell(k) for k in present)
        rows_html.append(f'<tr style="{row_style}">{cells}</tr>')

    table = f"""
    <div style="overflow-x:auto;border-radius:16px;border:1px solid #E2E8F0;background:#FFFFFF;box-shadow:0 12px 35px rgba(15,23,42,0.055);">
      <table style="width:100%;border-collapse:collapse;">
        <thead><tr>{th_row}</tr></thead>
        <tbody>{"".join(rows_html)}</tbody>
      </table>
    </div>
    """
    return table


def _render_operations(ops_table: pd.DataFrame):
    from src.utils import TICKER_LABELS

    st.markdown('<div class="section-header">Historial de Operaciones</div>', unsafe_allow_html=True)

    if ops_table.empty:
        st.info("No hay operaciones registradas.")
        return

    n_total = len(ops_table)
    n_buy = int((ops_table["decision"] == "COMPRAR").sum())
    n_sell = int((ops_table["decision"] == "VENDER").sum())
    total_costs = float(ops_table["coste"].sum()) if "coste" in ops_table.columns else 0.0
    total_notional = float(ops_table["importe"].sum()) if "importe" in ops_table.columns else 0.0

    sc1, sc2, sc3, sc4, sc5 = st.columns(5)
    with sc1:
        st.markdown(
            _kpi("Total Operaciones", str(n_total), sub="en todo el período"),
            unsafe_allow_html=True,
        )
    with sc2:
        st.markdown(
            _kpi("Compras", str(n_buy), sub="órdenes de entrada", cls="positive"),
            unsafe_allow_html=True,
        )
    with sc3:
        st.markdown(
            _kpi("Ventas", str(n_sell), sub="órdenes de salida", cls="negative"),
            unsafe_allow_html=True,
        )
    with sc4:
        st.markdown(
            _kpi("Costes Totales", fmt_eur(total_costs, compact=False), sub="comisiones pagadas", cls="muted"),
            unsafe_allow_html=True,
        )
    with sc5:
        st.markdown(
            _kpi("Volumen Total", fmt_eur(total_notional, compact=True), sub="importe negociado"),
            unsafe_allow_html=True,
        )

    st.markdown("<br>", unsafe_allow_html=True)

    fcol1, fcol2, fcol3 = st.columns([2, 1, 1])
    with fcol1:
        tickers_available = sorted(ops_table["ticker"].unique().tolist())
        sel_tickers = st.multiselect(
            "Filtrar por ticker",
            options=tickers_available,
            default=[],
            placeholder="Todos los tickers",
        )
    with fcol2:
        types_available = sorted(ops_table["decision"].unique().tolist())
        sel_types = st.multiselect(
            "Tipo",
            options=types_available,
            default=[],
            placeholder="Todos",
        )
    with fcol3:
        regimes_available = sorted(ops_table["regimen"].unique().tolist())
        sel_regime = st.multiselect(
            "Régimen",
            options=regimes_available,
            default=[],
            placeholder="Todos",
        )

    filtered = ops_table.copy()
    if sel_tickers:
        filtered = filtered[filtered["ticker"].isin(sel_tickers)]
    if sel_types:
        filtered = filtered[filtered["decision"].isin(sel_types)]
    if sel_regime:
        filtered = filtered[filtered["regimen"].isin(sel_regime)]

    st.markdown(
        f"<p style='color:#64748B;font-size:0.82rem;margin-bottom:0.5rem;'>"
        f"Mostrando <b style='color:#0F172A;'>{len(filtered)}</b> de {n_total} operaciones</p>",
        unsafe_allow_html=True,
    )

    st.markdown(_render_ops_html_table(filtered), unsafe_allow_html=True)


def _render_prices(portfolio_df: pd.DataFrame, weights_long: pd.DataFrame):
    from src.plots import plot_historical_prices
    from src.utils import TICKER_LABELS, TICKER_YF_MAP

    st.markdown('<div class="section-header">Precios Históricos de Cierre</div>', unsafe_allow_html=True)

    # 'Caja' es una pseudo-línea (caja residual), no un instrumento con precio: se excluye
    # del selector de precios (yfinance no la encuentra).
    all_tickers = (
        sorted(t for t in weights_long["ticker"].unique() if t != "Caja")
        if not weights_long.empty else list(TICKER_YF_MAP.values())
    )

    # Ticker pre-seleccionado desde el donut del Panel
    pre_selected = st.session_state.get("selected_ticker_prices", None)
    if pre_selected and pre_selected in all_tickers:
        st.markdown(
            f'<div class="selected-ticker-banner">🔍 Mostrando: <b>{pre_selected}</b> '
            f'(seleccionado desde el gráfico de composición)</div>',
            unsafe_allow_html=True,
        )
        default_selection = [pre_selected]
    else:
        default_selection = all_tickers[:4]

    col_sel, col_norm, col_btn = st.columns([3, 1, 1])
    with col_sel:
        selected = st.multiselect(
            "Seleccioná los ETFs a visualizar:",
            options=all_tickers,
            default=default_selection,
            format_func=lambda t: TICKER_LABELS.get(t, t),
        )
    with col_norm:
        normalize = st.checkbox("Base 100 (retornos comparables)", value=False)
    with col_btn:
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("🔄 Limpiar selección", width="stretch"):
            st.session_state["selected_ticker_prices"] = None
            st.rerun()

    if not selected:
        st.info("Seleccioná al menos un ETF para ver su precio histórico.")
        return

    with st.spinner(f"Descargando precios de {', '.join(selected)}..."):
        prices, failed = load_prices_for_tab(
            tickers=tuple(sorted(selected)),
            start="2026-03-10",
            end=(pd.Timestamp.today() + pd.Timedelta(days=1)).strftime("%Y-%m-%d"),
        )

    if failed:
        st.markdown(
            f'<div class="alert-warning">⚠ Sin datos de yfinance para: '
            f'<b>{", ".join(failed)}</b></div>',
            unsafe_allow_html=True,
        )

    available = [t for t in selected if t in prices.columns]
    if not available:
        st.error("No se pudieron obtener precios para los tickers seleccionados.")
        return

    fig = plot_historical_prices(prices, available, normalize=normalize)
    st.plotly_chart(fig, width="stretch", config=_export_cfg("precios_historicos"))

    with st.expander("Ver tabla de precios recientes", expanded=False):
        recent = prices[available].tail(10).copy()
        recent.index = recent.index.strftime("%d/%m/%Y")
        recent.columns = [TICKER_LABELS.get(c, c) for c in recent.columns]
        st.dataframe(recent.style.format("€{:.4f}"), width="stretch")


# Estrategia: Backtest + Validación (leen outputs/, todo en EUR)


def _render_backtest():
    import src.strategy_data as sd
    import src.strategy_plots as sp

    st.markdown('<div class="section-header">Backtest de la estrategia (EUR)</div>',
                unsafe_allow_html=True)

    if not sd.available_windows():
        st.info("No hay backtests en `outputs/backtest/`. Corre `python scripts/run_backtest.py`.")
        return

    labels = {w: sd.window_label(w) for w in sd.BACKTEST_WINDOWS}
    win = st.radio(
        "Ventana", list(sd.BACKTEST_WINDOWS),
        format_func=lambda w: labels[w] + ("" if sd.window_available(w) else "  (pendiente)"),
        horizontal=True,
    )
    short, desc = sd.BACKTEST_WINDOWS[win]
    st.caption(desc)

    if not sd.window_available(win):
        st.info(f"La ventana «{short}» aún no se ha generado. Corre `python scripts/run_backtest.py` "
                f"(produce canonical, subperiod y counterfactual); el dashboard la recogerá sola.")
        return

    metrics = sd.load_metrics(win)
    if metrics is not None and "Strategy" in metrics.columns:
        strat = metrics["Strategy"]
        cols = st.columns(4)
        cols[0].metric("CAGR", str(strat.get("CAGR", "—")))
        cols[1].metric("Sharpe", str(strat.get("Sharpe", "—")))
        cols[2].metric("Max Drawdown", str(strat.get("Max Drawdown", "—")))
        cols[3].metric("Retorno total", str(strat.get("Total Return", "—")))
        st.dataframe(metrics, width="stretch")
        st.download_button("⬇ Métricas (CSV)", metrics.to_csv().encode("utf-8"),
                           file_name=f"metrics_{win}.csv", mime="text/csv")

    resumen = sd.load_resumen(win)
    wealth = sd.load_wealth(win)
    if wealth is not None:
        st.plotly_chart(sp.equity_curve(wealth), width="stretch",
                        config=_export_cfg(f"backtest_{win}_curva_valor"))
        # Franja de régimen bajo la curva (mismo eje temporal): cuándo manda cada régimen
        if resumen is not None and "Regime" in resumen.columns:
            st.plotly_chart(sp.regime_timeline(resumen), width="stretch",
                            config=_export_cfg(f"backtest_{win}_regimen_timeline"))
        # Drawdown con selector de benchmark para comparar
        bm_cols = [c for c in wealth.columns if c != "Strategy"]
        dd_label_to_col = {"Solo estrategia": None}
        for c in bm_cols:
            dd_label_to_col[c.replace("_", " ").replace(" (URTH)", "")] = c
        dd_choice = st.radio(
            "Comparar drawdown con:", list(dd_label_to_col), horizontal=True,
            key=f"dd_bm_{win}",
        )
        st.plotly_chart(sp.drawdown(wealth, benchmark_col=dd_label_to_col[dd_choice]),
                        width="stretch", config=_export_cfg(f"backtest_{win}_drawdown"))
        st.download_button("⬇ Curva de valor (CSV)", wealth.to_csv().encode("utf-8"),
                           file_name=f"wealth_{win}.csv", mime="text/csv")

    if resumen is not None and "Trade Cost EUR" in resumen.columns:
        st.plotly_chart(sp.cumulative_costs(resumen), width="stretch",
                        config=_export_cfg(f"backtest_{win}_costes"))

    weights = sd.load_weights(win)
    if weights is not None:
        st.plotly_chart(sp.weights_area(weights), width="stretch",
                        config=_export_cfg(f"backtest_{win}_pesos"))

    attr = sd.load_attribution_cumulative(win)
    if attr is not None:
        st.plotly_chart(sp.attribution_bar(attr), width="stretch",
                        config=_export_cfg(f"backtest_{win}_atribucion"))


def _render_validacion():
    import src.strategy_data as sd
    import src.strategy_plots as sp

    st.markdown('<div class="section-header">Validación</div>',
                unsafe_allow_html=True)
    st.caption("Cada bloque responde a una pregunta de robustez: ¿qué parámetros mueven el "
               "resultado?, ¿son especiales nuestros valores o un mono lo haría igual?, "
               "¿generaliza el ajuste fino?, ¿importa la frecuencia?, ¿aporta el ML?")

    st.markdown("#### Importancia de parámetros — `study_param_analysis`")
    st.caption("Se muestrean todos los parámetros a la vez en cientos de backtests y un "
               "random forest reparte la varianza del Sharpe entre ellos: la barra es la "
               "fracción que explica cada uno. Capta interacciones y señala cuál **domina** "
               "cuando todo varía junto — la sección siguiente (OAT) muestra el efecto de "
               "cada parámetro por separado.")
    pi = sd.load_param_importance()
    if pi is not None:
        st.plotly_chart(sp.param_importance_bar(pi), width="stretch",
                        config=_export_cfg("validacion_importancia_params"))
    else:
        st.info("Sin datos. Corre `python scripts/study_param_analysis.py`.")

    st.markdown("#### Sensibilidad por parámetro (OAT) — `study_param_analysis`")
    st.caption("Barrido uno a uno: cada parámetro se mueve solo, con el resto fijo en la "
               "config adoptada. El **span** es cuánto cambia el Sharpe de punta a punta — "
               "responde \"¿cuánto importa este parámetro aislado?\", que la importancia RF "
               "no contesta porque reparte la varianza conjunta entre todos.")
    ps = sd.load_param_sensitivity()
    if ps is not None:
        st.plotly_chart(sp.param_sensitivity_bar(ps), width="stretch",
                        config=_export_cfg("validacion_sensibilidad_params"))
    else:
        st.info("Sin datos. Corre `python scripts/study_param_analysis.py`.")

    st.markdown("#### Distribución nula de parámetros (monos) — `study_params_vs_random`")
    st.caption("Muchas carteras con los parámetros del núcleo sorteados al azar (todas a "
               "**nuestra misma frecuencia mensual**). Si la nuestra no sobresale de esa "
               "nube, sus parámetros no están elegidos a dedo. La línea roja marca dónde "
               "cae nuestra estrategia: el **percentil** es el % de monos que batimos.")
    mk = sd.load_monkeys()
    if mk:
        chosen = sd.load_monkeys_chosen() or {}
        per = st.radio("Periodo", list(mk), horizontal=True, key="monkeys_per")
        chosen_sharpe = None
        if per in chosen:
            try:
                chosen_sharpe = float(chosen[per]["chosen_Sharpe"])
            except (KeyError, ValueError, TypeError):
                chosen_sharpe = None
        st.plotly_chart(sp.monkeys_hist(mk[per], chosen=chosen_sharpe),
                        width="stretch", config=_export_cfg(f"validacion_monos_{per}"))
        mw = sd.load_monkeys_wealth(per)
        if mw is not None:
            st.caption("Las mismas carteras aleatorias, ahora como **curvas de valor**; "
                       "nuestra estrategia en azul dentro de la nube.")
            st.plotly_chart(sp.monkeys_wealth_curves(mw), width="stretch",
                            config=_export_cfg(f"validacion_monos_curvas_{per}"))
    else:
        st.info("Sin datos. Corre `python scripts/study_params_vs_random.py`.")

    st.markdown("#### Distribución nula de selección (dardos) — `study_random_portfolios`")
    st.caption("El test del **mono con dardos** (Malkiel): carteras sorteadas al azar dentro "
               "del **mismo universo de ETFs** (con la amplitud media de la estrategia), por el "
               "mismo motor. Si una cesta aleatoria nos bate, elegir *qué* ETFs y *cuánto* no "
               "aporta. **Buy-and-hold**: sortea una vez y mantiene (pura selección). "
               "**Rebalanceo mensual**: re-sortea cada mes y paga turnover.")
    _RP_MODES = {"Buy-and-hold (Malkiel)": "buyhold", "Rebalanceo mensual": "monthly"}
    mode_label = st.radio("Variante de dardo", list(_RP_MODES), horizontal=True, key="rp_mode")
    mode = _RP_MODES[mode_label]
    rp = sd.load_random_portfolios(mode)
    if rp:
        rp_chosen = sd.load_random_chosen(mode) or {}
        rper = st.radio("Periodo", list(rp), horizontal=True, key="rp_per")
        rp_sharpe = None
        if rper in rp_chosen:
            try:
                rp_sharpe = float(rp_chosen[rper]["chosen_Sharpe"])
            except (KeyError, ValueError, TypeError):
                rp_sharpe = None
        st.plotly_chart(
            sp.monkeys_hist(rp[rper], chosen=rp_sharpe, trace_name="Carteras aleatorias",
                            unit="dardos", title="Distribución nula de selección (dardos)"),
            width="stretch", config=_export_cfg(f"validacion_dardos_{mode}_{rper}"))
        rpw = sd.load_random_wealth(mode, rper)
        if rpw is not None:
            st.caption("Las mismas carteras aleatorias como **curvas de valor**; nuestra "
                       "estrategia en azul dentro de la nube.")
            st.plotly_chart(
                sp.monkeys_wealth_curves(
                    rpw, cloud_name="carteras aleatorias", median_name="Mediana de los dardos",
                    chosen_name="Estrategia",
                    title="Curvas de valor: estrategia vs carteras aleatorias"),
                width="stretch", config=_export_cfg(f"validacion_dardos_curvas_{mode}_{rper}"))
    else:
        st.info("Sin datos. Corre `python scripts/study_random_portfolios.py`.")

    st.markdown("#### Walk-forward de parámetros (OOS) — `study_params_oos`")
    st.caption("Se **optimizan** los parámetros en cada tramo de entrenamiento (pasado) y se "
               "evalúan **congelados** en el tramo siguiente, que la optimización nunca vio "
               "(*out-of-sample*), en tramos encadenados. La pregunta: ¿afinar los parámetros "
               "bate a dejarlos fijos cuando se prueba en datos no vistos?")
    oos_sum = sd.load_params_oos_summary()
    if oos_sum is not None:
        st.markdown("**Sharpe sobre el periodo OOS completo:**")
        st.plotly_chart(sp.params_oos_summary_bar(oos_sum), width="stretch",
                        config=_export_cfg("validacion_oos_resumen"))
        st.caption("La comparación fijos-vs-re-optimizados **no es robusta**: su signo depende del "
                   "presupuesto de búsqueda y los parámetros elegidos saltan entre tramos. No hay "
                   "evidencia robusta de que afinar el núcleo ayude fuera de muestra, así que los "
                   "parámetros adoptados se mantienen fijos.")
    else:
        st.info("Sin datos. Corre `python scripts/study_params_oos.py`.")

    oos = sd.load_params_oos()
    if oos is not None:
        with st.expander("Detalle por tramo (train vs OOS y parámetros elegidos)", expanded=False):
            st.plotly_chart(sp.params_oos_folds_bar(oos), width="stretch",
                            config=_export_cfg("validacion_oos_folds"))
            st.caption("El Sharpe de train (in-sample) suele superar al de test (OOS): "
                       "ese hueco es el sobreajuste. Los parámetros elegidos saltan entre tramos "
                       "(sin óptimo estable).")
            st.dataframe(oos, width="stretch", hide_index=True)

    st.markdown("#### Frecuencia de rebalanceo — `study_rebalance_frequency`")
    st.caption("Sharpe según cada cuánto se rebalancea, por periodo, con el SP500 de "
               "referencia. La frecuencia importa mucho, pero su óptimo cambia entre "
               "periodos: por eso la **fijamos en mensual** en vez de optimizarla.")
    fq = sd.load_freq_compare()
    if fq is not None:
        st.plotly_chart(sp.freq_compare_bar(fq), width="stretch",
                        config=_export_cfg("validacion_frecuencia_rebalanceo"))
        st.dataframe(fq, width="stretch", hide_index=True)
    else:
        st.info("Sin datos. Corre `python scripts/study_rebalance_frequency.py`.")

    st.markdown("#### Ablación del ML — `study_ml_value`")
    st.caption("Se construye la estrategia por capas (sin ML → +clustering → +XGBoost) "
               "y se mide cuánto suma cada una al Sharpe. Muestra la aportación de cada "
               "componente del refinamiento de la señal.")
    abl = sd.load_ablation()
    if abl is not None:
        st.plotly_chart(sp.ablation_bar(abl), width="stretch",
                        config=_export_cfg("validacion_ablacion_ml"))
        st.dataframe(abl, width="stretch", hide_index=True)
    else:
        st.info("Sin datos. Corre `python scripts/study_ml_value.py`.")

    st.markdown("#### Walk-forward del filtro ML — `run_walkforward`")
    st.caption("Validación out-of-sample del clasificador ML: en cada tramo se entrena con el "
               "pasado y se evalúa sobre el tramo siguiente no visto. El AUC mide su capacidad de "
               "discriminación fuera de muestra (0.5 = sin discriminación).")
    wf = sd.load_ml_walkforward()
    if wf is not None and "ML OOS AUC" in wf.columns:
        _auc = pd.to_numeric(wf["ML OOS AUC"], errors="coerce").dropna()
        c1, c2 = st.columns(2)
        c1.metric("AUC OOS medio", f"{_auc.mean():.3f}")
        c2.metric("Folds con AUC ≥ 0.5", f"{int((_auc > 0.5).sum())} / {len(_auc)}")
        st.plotly_chart(sp.ml_walkforward_auc(wf), width="stretch",
                        config=_export_cfg("validacion_ml_walkforward"))
        st.caption("AUC OOS del filtro por fold. El filtro XGBoost forma parte del pipeline como "
                   "capa de refinamiento de la señal previa a Black-Litterman.")
        with st.expander("Tabla completa por fold (AUC, accuracy, métricas de estrategia)"):
            st.dataframe(wf, width="stretch", hide_index=True)
    else:
        st.info("Sin datos en `outputs/walkforward/`. Corre `python scripts/run_walkforward.py`.")


def _render_parametros():
    from src.strategy_params import UNIVERSE, PARAM_BLOCKS

    _wide = st.column_config.TextColumn(width="large")

    st.markdown('<div class="section-header">Parámetros de la estrategia</div>',
                unsafe_allow_html=True)
    st.caption("Parámetros del núcleo de la estrategia con su valor actual y una breve "
               "descripción.")

    st.markdown("#### Universo, capital y ventanas")
    df_u = pd.DataFrame(UNIVERSE, columns=["Concepto", "Valor", "Detalle"])
    st.dataframe(df_u, width="stretch", hide_index=True,
                 column_config={"Detalle": _wide})

    st.markdown("#### Parámetros por bloque del pipeline")
    st.caption("Orden del pipeline: señal compuesta → ML → Black-Litterman → Merton → "
               "Davis-Norman → registrador.")
    for block, rows in PARAM_BLOCKS.items():
        st.markdown(f"**{block}**")
        df_b = pd.DataFrame(rows, columns=["Parámetro", "Valor", "Descripción"])
        st.dataframe(df_b, width="stretch", hide_index=True,
                     column_config={"Descripción": _wide})


if __name__ == "__main__":
    main()
