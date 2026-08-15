"""Load and parse the Historial/Operativa Excel files into clean DataFrames."""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from src.utils import (
    CAPITAL_INICIAL,
    IUSE_TICKER_HIST,
    IUSE_TICKER_OP,
    OPERATIVA_DATE,
    PHASE3_DATE,
    TICKER_YF_MAP,
    XEON_TICKER,
)

# Registro live real (versionado), junto al dashboard que lo consume.
# faseN_<tipo>_Grupo4.xlsx: 1 = historial diario táctico, 2/3 = hojas de operativa.
_DASH_DIR = Path(__file__).resolve().parents[1]
HISTORIAL_PATH = _DASH_DIR / "fase1_historial_Grupo4.xlsx"
OPERATIVA_PATH = _DASH_DIR / "fase2_operativa_Grupo4.xlsx"
OPERATIVA_FASE3_PATH = _DASH_DIR / "fase3_operativa_Grupo4.xlsx"


def load_historial() -> pd.DataFrame:
    """Load Historial_Grupo4.xlsx into a clean DataFrame."""
    raw = pd.read_excel(HISTORIAL_PATH, sheet_name="Historial", header=None)

    # Row 0 = title, Row 1 = column names, Row 2+ = data
    col_names = raw.iloc[1].tolist()
    data = raw.iloc[2:].copy()
    data.columns = col_names
    data = data.reset_index(drop=True)

    rename = {
        "Fecha": "fecha",
        "Precio ETF": "precio_etf",
        "Decision": "decision",
        "Alpha anterior": "alpha_anterior",
        "Alpha nuevo": "alpha",
        "Banda inf": "banda_inf",
        "Banda sup": "banda_sup",
        "Alpha* Merton": "alpha_merton",
        "mu": "mu",
        "sigma": "sigma",
        "Importe (EUR)": "importe",
        "Coste (EUR)": "coste",
        "Valor cartera": "valor_cartera",
        "Retorno acum.": "retorno_acum",
        "N operaciones": "n_operaciones",
        "Costes acum.": "costes_acum_original",
        "Razon": "razon",
    }
    data = data.rename(columns={k: v for k, v in rename.items() if k in data.columns})

    data["fecha"] = pd.to_datetime(data["fecha"])

    numeric_cols = [
        "precio_etf", "alpha_anterior", "alpha", "banda_inf", "banda_sup",
        "alpha_merton", "mu", "sigma", "importe", "coste", "valor_cartera",
        "retorno_acum", "n_operaciones", "costes_acum_original",
    ]
    for col in numeric_cols:
        if col in data.columns:
            data[col] = pd.to_numeric(data[col], errors="coerce")

    # Recalculate cumulative costs: source has 21000 instead of 2100 in first rows
    data["coste"] = data["coste"].fillna(0.0)
    data["costes_acum_recalc"] = data["coste"].cumsum()

    data["fuente"] = "Historial_Grupo4.xlsx"
    data["ticker"] = IUSE_TICKER_HIST

    return data.sort_values("fecha").reset_index(drop=True)


def load_operativa(path=OPERATIVA_PATH, fecha=OPERATIVA_DATE) -> pd.DataFrame:
    """Load an Operativa trade sheet (fase 2 o 3) into a clean DataFrame."""
    raw = pd.read_excel(path, sheet_name="Operativa", header=0)

    rename = {
        "ID": "ticker",
        "Cantidad": "cantidad",
        "Precio": "precio",
        "CT": "ct",
        "Precio Ejecutado": "precio_ejecutado",
    }
    data = raw.rename(columns={k: v for k, v in rename.items() if k in raw.columns})

    for col in ["cantidad", "precio", "ct", "precio_ejecutado"]:
        if col in data.columns:
            data[col] = pd.to_numeric(data[col], errors="coerce")

    # La hoja no trae fecha; se asigna la del rebalanceo (fase 2 = 10-abr, fase 3 = 13-may)
    data["fecha"] = fecha

    data["decision"] = data["cantidad"].apply(
        lambda x: "VENDER" if x < 0 else "COMPRAR"
    )

    # CT already embedded in precio_ejecutado; coste recomputed for reporting
    data["coste"] = (data["cantidad"].abs() * data["precio"] * data["ct"]).fillna(0.0)
    data["importe"] = data["cantidad"].abs() * data["precio_ejecutado"]
    data["yf_ticker"] = data["ticker"].map(lambda t: TICKER_YF_MAP.get(t, t))
    data["fuente"] = Path(path).name

    return data.reset_index(drop=True)


def load_operativa_fase3() -> pd.DataFrame:
    """Fase 3 (segundo rebalanceo, 13-may)."""
    return load_operativa(path=OPERATIVA_FASE3_PATH, fecha=PHASE3_DATE)


def _convert_ops_to_eur(combined: pd.DataFrame) -> pd.DataFrame:
    """Pasa precios e importes de la tabla de operaciones a EUR.

    Las hojas de operativa traen `precio`, `precio_ejecutado`, `importe` y `coste`
    en divisa NATIVA del listado (USD para los ETF de EE.UU.) por el error 1 USD = 1 EUR
    con que se cerraron. Aquí se dividen por el EURUSD=X del día de la operación, salvo
    los cotizados en EUR (EUR_QUOTED_TICKERS) que ya están en euros. Las cantidades
    (títulos) no se tocan. Así la tabla y sus totales quedan en EUR, coherentes con el
    Panel (que ya marca las posiciones en EUR).
    """
    from src.portfolio_engine import EUR_QUOTED_TICKERS, _eurusd_on, download_prices

    if combined.empty or "yf_ticker" not in combined.columns:
        return combined

    fechas = pd.to_datetime(combined["fecha"]).dropna()
    if fechas.empty:
        return combined
    prices, _ = download_prices(
        ["EURUSD=X"],
        start=(fechas.min() - pd.Timedelta(days=7)).strftime("%Y-%m-%d"),
        end=(fechas.max() + pd.Timedelta(days=2)).strftime("%Y-%m-%d"),
    )

    money_cols = [c for c in ("precio", "precio_ejecutado", "importe", "coste")
                  if c in combined.columns]

    def _fx(row) -> float:
        if row["yf_ticker"] in EUR_QUOTED_TICKERS:
            return 1.0
        return _eurusd_on(prices, pd.Timestamp(row["fecha"]))

    fx = combined.apply(_fx, axis=1)
    out = combined.copy()
    for col in money_cols:
        out[col] = pd.to_numeric(out[col], errors="coerce") / fx
    return out


def build_operations_table(
    historial_df: pd.DataFrame,
    operativa_df: pd.DataFrame,
    operativa_fase3_df: "pd.DataFrame | None" = None,
) -> pd.DataFrame:
    """Merge Historial trades (non-MANTENER) with Operativa (fases 2 y 3) chronologically.

    Devuelve precios e importes en EUR (ver `_convert_ops_to_eur`): las hojas de operativa
    venían en divisa nativa por el error 1 USD = 1 EUR.
    """
    hist_trades = historial_df[historial_df["decision"] != "MANTENER"].copy()

    # Approximate quantity from notional / price
    hist_trades["cantidad"] = hist_trades.apply(
        lambda r: (
            r["importe"] / r["precio_etf"]
            if r["precio_etf"] and r["precio_etf"] > 0
            else np.nan
        ),
        axis=1,
    )
    # Sells have negative quantity
    hist_trades["cantidad"] = hist_trades.apply(
        lambda r: -r["cantidad"] if r["decision"] == "VENDER" else r["cantidad"],
        axis=1,
    )
    hist_trades["precio_ejecutado"] = hist_trades["precio_etf"]
    hist_trades["ct"] = hist_trades.apply(
        lambda r: (
            r["coste"] / (r["cantidad"] * r["precio_etf"])
            if r["cantidad"] and r["precio_etf"] and r["precio_etf"] > 0
            else 0.0
        ),
        axis=1,
    )
    hist_trades["yf_ticker"] = IUSE_TICKER_HIST
    hist_trades.rename(columns={"precio_etf": "precio"}, inplace=True)
    hist_trades["regimen"] = "Estrategia inicial (IUSE + XEON)"

    op_trades = operativa_df.copy()
    op_trades["regimen"] = "Multi-activo (despliegue fase 2)"

    cols = [
        "fecha", "ticker", "yf_ticker", "decision", "cantidad",
        "precio", "precio_ejecutado", "importe", "coste", "ct",
        "fuente", "regimen",
    ]
    frames = [hist_trades, op_trades]
    if operativa_fase3_df is not None and not operativa_fase3_df.empty:
        op3 = operativa_fase3_df.copy()
        op3["regimen"] = "Multi-activo (rebalanceo fase 3)"
        frames.append(op3)

    cleaned = [df[[c for c in cols if c in df.columns]].copy() for df in frames]
    combined = (
        pd.concat(cleaned, ignore_index=True)
        .sort_values("fecha")
        .reset_index(drop=True)
    )

    return _convert_ops_to_eur(combined)
