"""Formatting helpers and the ETF taxonomy (by portfolio role) for the dashboard.

Ambiguous tickers (e.g. COPX/LIT, equity + commodity) are classified by their
dominant exposure in this book; kept central to avoid duplicating the logic.
"""

import pandas as pd
import numpy as np


def fmt_currency(val, decimals: int = 2, prefix: str = "€") -> str:
    """Format a number as compact currency string."""
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return "N/A"
    sign = "-" if val < 0 else ""
    abs_val = abs(val)
    if abs_val >= 1_000_000:
        return f"{sign}{prefix}{abs_val / 1_000_000:.{decimals}f}M"
    if abs_val >= 1_000:
        return f"{sign}{prefix}{abs_val / 1_000:.{decimals}f}K"
    return f"{sign}{prefix}{abs_val:.{decimals}f}"


def fmt_pct(val, decimals: int = 2, show_sign: bool = True) -> str:
    """Format a decimal ratio as percentage string."""
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return "N/A"
    fmt = f"+.{decimals}f" if show_sign else f".{decimals}f"
    return f"{val * 100:{fmt}}%"


def fmt_num(val, decimals: int = 2) -> str:
    """Format a number with thousands separator."""
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return "N/A"
    return f"{val:,.{decimals}f}"


def color_pnl(val) -> str:
    """Return CSS color string for a PnL value."""
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return "#9E9E9E"
    return "#15803D" if val >= 0 else "#B91C1C"


# Business-day date ranges
PHASE1_START = pd.Timestamp("2026-03-12")
PHASE1_END = pd.Timestamp("2026-04-02")
PHASE2_START = pd.Timestamp("2026-04-10")

# Operativa date (all phase-2 Operativa trades happen on this date)
OPERATIVA_DATE = pd.Timestamp("2026-04-10")
# Fase 3: segundo rebalanceo multi-activo
PHASE3_DATE = pd.Timestamp("2026-05-13")

# Initial capital
CAPITAL_INICIAL = 10_000_000.0

# Risk-free rate used for risk-adjusted metrics.
# Source convention: ECB deposit facility rate for EUR cash, verified on 2026-05-12.
TRADING_DAYS_PER_YEAR = 252
RISK_FREE_RATE_ANNUAL = 0.02


def daily_risk_free_rate(annual_rate: float = RISK_FREE_RATE_ANNUAL) -> float:
    """Convert an annual risk-free rate into an equivalent daily rate."""
    return (1.0 + annual_rate) ** (1.0 / TRADING_DAYS_PER_YEAR) - 1.0

# Tickers used in Phase 1
IUSE_TICKER_HIST = "IUSE.L"   # Historial uses IUSE.L
IUSE_TICKER_OP = "IUSE"       # Operativa uses IUSE (same ETF, London listing)
XEON_TICKER = "XEON.DE"

# Mapping from Operativa ticker IDs to yfinance tickers
TICKER_YF_MAP: dict[str, str] = {
    "IUSE": "IUSE.L",
    "XEON.DE": "XEON.DE",
    "BOTZ": "BOTZ",
    "COPX": "COPX",
    "EWJ": "EWJ",
    "EWT": "EWT",
    "EWY": "EWY",
    "EWZ": "EWZ",
    "GLD": "GLD",
    "IEMA.L": "IEMA.L",
    "IUSN.DE": "IUSN.DE",
    "LIT": "LIT",
    "SLV": "SLV",
    "SOXX": "SOXX",
    "VNQ": "VNQ",
    "XBI": "XBI",
    "XLE": "XLE",
    "XLI": "XLI",
    "XLP": "XLP",
    "XLU": "XLU",
    "XLV": "XLV",
    "XOP": "XOP",
    # Fase 3 (13-may): nuevas posiciones
    "DBA": "DBA",
    "IEF": "IEF",
    "KBE": "KBE",
    "XLB": "XLB",
    "XLF": "XLF",
}

# Human-readable labels for tickers
TICKER_LABELS: dict[str, str] = {
    "Caja": "Liquidez (caja sin invertir)",
    "IUSE.L": "iShares S&P 500 con cobertura en euros (IUSE.L)",
    "XEON.DE": "Xtrackers tasa overnight del euro (XEON.DE)",
    "BOTZ": "Global X robótica e inteligencia artificial (BOTZ)",
    "COPX": "Global X mineras de cobre (COPX)",
    "EWJ": "iShares MSCI Japón (EWJ)",
    "EWT": "iShares MSCI Taiwán (EWT)",
    "EWY": "iShares MSCI Corea del Sur (EWY)",
    "EWZ": "iShares MSCI Brasil (EWZ)",
    "GLD": "SPDR oro físico (GLD)",
    "IEMA.L": "iShares MSCI Asia emergente (IEMA.L)",
    "IUSN.DE": "iShares MSCI pequeñas compañías globales (IUSN.DE)",
    "LIT": "Global X litio y baterías (LIT)",
    "SLV": "iShares plata física (SLV)",
    "SOXX": "iShares semiconductores (SOXX)",
    "VNQ": "Vanguard inmobiliario estadounidense (VNQ)",
    "XBI": "SPDR S&P biotecnología (XBI)",
    "XLE": "Sector energético del S&P 500 (XLE)",
    "XLI": "Sector industrial del S&P 500 (XLI)",
    "XLP": "Sector consumo básico del S&P 500 (XLP)",
    "XLU": "Sector servicios públicos del S&P 500 (XLU)",
    "XLV": "Sector salud del S&P 500 (XLV)",
    "XOP": "SPDR petróleo y gas: exploración y producción (XOP)",
    "DBA": "Invesco materias primas agrícolas (DBA)",
    "IEF": "iShares deuda EE.UU. 7-10 años (IEF)",
    "KBE": "SPDR bancos de EE.UU. (KBE)",
    "XLB": "Sector materiales del S&P 500 (XLB)",
    "XLF": "Sector financiero del S&P 500 (XLF)",
}

# ETF taxonomy used in the composition visuals. The categories prioritise the
# portfolio role over strict index vendor taxonomy.
TICKER_CATEGORIES: dict[str, str] = {
    "Caja": "Liquidez",                 # caja residual sin invertir
    "XEON.DE": "Monetario (XEON)",      # ETF monetario: SÍ está invertido (gana el tipo BCE)
    "IUSE.L": "Renta variable core",
    "IUSN.DE": "Renta variable core",
    "EWJ": "Renta variable regional",
    "EWT": "Renta variable regional",
    "EWY": "Renta variable regional",
    "EWZ": "Renta variable regional",
    "IEMA.L": "Renta variable regional",
    "BOTZ": "Temáticas y sectores",
    "COPX": "Temáticas y sectores",
    "LIT": "Temáticas y sectores",
    "SOXX": "Temáticas y sectores",
    "XBI": "Temáticas y sectores",
    "XLE": "Temáticas y sectores",
    "XLI": "Temáticas y sectores",
    "XLP": "Temáticas y sectores",
    "XLU": "Temáticas y sectores",
    "XLV": "Temáticas y sectores",
    "XOP": "Temáticas y sectores",
    "GLD": "Materias primas",
    "SLV": "Materias primas",
    "DBA": "Materias primas",
    "VNQ": "Inmobiliario",
    "IEF": "Renta fija",
    "KBE": "Temáticas y sectores",
    "XLB": "Temáticas y sectores",
    "XLF": "Temáticas y sectores",
}

CATEGORY_COLORS: dict[str, str] = {
    "Liquidez": "#CBD5E1",
    "Monetario (XEON)": "#64748B",
    "Renta variable core": "#2563EB",
    "Renta variable regional": "#0F766E",
    "Temáticas y sectores": "#7C3AED",
    "Materias primas": "#D97706",
    "Inmobiliario": "#16A34A",
    "Renta fija": "#0D9488",
    "Otros": "#64748B",
}


def ticker_category(ticker: str) -> str:
    """Return the portfolio category assigned to a ticker."""
    return TICKER_CATEGORIES.get(ticker, "Otros")


def ticker_label(ticker: str, include_ticker: bool = False) -> str:
    """Return a human-readable ETF label; drops the trailing "(TICKER)" unless include_ticker."""
    label = TICKER_LABELS.get(ticker, ticker)
    if include_ticker or "(" not in label:
        return label
    return label.split("(")[0].strip()

# Plotly color palette (light fintech professional)
CHART_COLORS = [
    "#2563EB", "#0F766E", "#D97706", "#DC2626", "#7C3AED",
    "#0891B2", "#16A34A", "#EA580C", "#BE123C", "#4F46E5",
    "#14B8A6", "#84CC16", "#F97316", "#0284C7", "#65A30D",
    "#EAB308", "#DB2777", "#3B82F6", "#92400E", "#64748B",
    "#6366F1", "#06B6D4",
]
