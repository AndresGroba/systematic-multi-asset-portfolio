"""Reconstruct the daily portfolio over three regimes.

Phase 1 (Historial, Mar 12 – Apr 2): NAV from "Valor cartera"; IUSE.L = alpha·NAV,
XEON.DE = (1-alpha)·NAV, cash ~ 0.
Gap (Apr 3 – Apr 9): Apr-2 share counts held; NAV marked to daily prices.
Phase 2 (Apr 10 onward): Operativa positions marked to market in EUR.

Divisa: cada ETF en USD se valora en EUR con el EURUSD=X diario
(value_eur_t = titulos * cierre_nativo_t / EURUSD_t); los cotizados en EUR
(IUSN.DE, XEON.DE) no se convierten. Cantidades intactas.
"""

from __future__ import annotations

import warnings
from typing import Optional

import numpy as np
import pandas as pd
import yfinance as yf

from src.utils import (
    CAPITAL_INICIAL,
    IUSE_TICKER_HIST,
    PHASE1_END,
    OPERATIVA_DATE,
    PHASE2_START,
    PHASE3_DATE,
    TICKER_YF_MAP,
    XEON_TICKER,
)

# Fallback XEON.DE price if yfinance fails (money market ~flat at 148-149 EUR)
_XEON_FALLBACK_PRICE = 148.87

# Cotizados en EUR (no se convierten); el resto se valora en EUR con EURUSD=X.
# IUSE.L es el S&P 500 con cobertura EUR (LSE), efectivamente denominado en EUR:
# solo aparece como venta en fase 2, pero su coste no debe dividirse por EURUSD.
EUR_QUOTED_TICKERS = {"IUSN.DE", "XEON.DE", "IUSE.L"}
# Solo último recurso si yfinance no devuelve EURUSD=X (fallo puntual). ~nivel 2026
# para minimizar el error si se usa; lo normal es bajar el dato real diario.
EURUSD_FALLBACK = 1.16


def download_prices(
    tickers: list[str],
    start: str | pd.Timestamp,
    end: str | pd.Timestamp,
) -> tuple[pd.DataFrame, list[str]]:
    """Download daily Close prices from yfinance; return (prices_df, failed_tickers)."""
    if not tickers:
        return pd.DataFrame(), []

    failed: list[str] = []
    frames: dict[str, pd.Series] = {}

    for ticker in tickers:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                raw = yf.download(
                    ticker,
                    start=str(start)[:10],
                    end=str(end)[:10],
                    progress=False,
                    auto_adjust=True,
                )
            if raw.empty:
                failed.append(ticker)
                continue

            # yfinance may return MultiIndex columns when downloading a single ticker
            if isinstance(raw.columns, pd.MultiIndex):
                close = raw["Close"].squeeze()
            else:
                close = raw["Close"]

            close = close.dropna()
            if close.empty:
                failed.append(ticker)
            else:
                frames[ticker] = close
        except Exception:
            failed.append(ticker)

    if not frames:
        return pd.DataFrame(), failed

    prices = pd.DataFrame(frames)
    prices.index = pd.to_datetime(prices.index).normalize()
    prices = prices.sort_index()
    return prices, failed


def _build_phase1_rows(historial_df: pd.DataFrame) -> list[dict]:
    """Convert Historial rows into portfolio-snapshot dicts."""
    rows = []
    for _, r in historial_df.iterrows():
        nav = r["valor_cartera"]
        alpha = r["alpha"] if pd.notna(r["alpha"]) else 0.0
        rows.append(
            {
                "date": r["fecha"],
                "nav": nav,
                "cash": 0.0,
                "regime": "phase1",
                "daily_cost": r["coste"],
                f"{IUSE_TICKER_HIST}_value": alpha * nav,
                f"{XEON_TICKER}_value": (1.0 - alpha) * nav,
                f"{IUSE_TICKER_HIST}_weight": alpha,
                f"{XEON_TICKER}_weight": 1.0 - alpha,
            }
        )
    return rows


def _build_gap_rows(
    historial_df: pd.DataFrame,
    prices: pd.DataFrame,
) -> list[dict]:
    """Extrapolate the Apr 3–9 portfolio from Phase 1 end positions."""
    phase1_end = PHASE1_END
    phase2_start = PHASE2_START

    row_apr2 = historial_df[historial_df["fecha"] == phase1_end]
    if row_apr2.empty:
        row_apr2 = historial_df.iloc[-1:]  # fallback: last Historial row

    row_apr2 = row_apr2.iloc[0]
    nav_apr2 = row_apr2["valor_cartera"]
    alpha_apr2 = float(row_apr2["alpha"]) if pd.notna(row_apr2["alpha"]) else 0.48
    iuse_price_apr2 = float(row_apr2["precio_etf"])

    # Historial price used to avoid yfinance discrepancy
    iuse_shares = (alpha_apr2 * nav_apr2) / iuse_price_apr2

    xeon_value_apr2 = (1.0 - alpha_apr2) * nav_apr2
    if XEON_TICKER in prices.columns:
        xeon_idx = prices.index[prices.index <= phase1_end]
        if len(xeon_idx) > 0:
            xeon_price_apr2 = prices.loc[xeon_idx[-1], XEON_TICKER]
        else:
            xeon_price_apr2 = _XEON_FALLBACK_PRICE
    else:
        xeon_price_apr2 = _XEON_FALLBACK_PRICE

    xeon_shares = xeon_value_apr2 / xeon_price_apr2

    # Business days in the gap (excluding the Phase 2 start date)
    gap_dates = pd.bdate_range(
        start=phase1_end + pd.offsets.BDay(1),
        end=phase2_start - pd.offsets.BDay(1),
    )

    rows = []
    for date in gap_dates:
        if IUSE_TICKER_HIST in prices.columns and date in prices.index:
            iuse_price = prices.loc[date, IUSE_TICKER_HIST]
        else:
            # Forward-fill from closest prior date
            idx = prices.index[prices.index <= date] if IUSE_TICKER_HIST in prices.columns else pd.DatetimeIndex([])
            iuse_price = (
                prices.loc[idx[-1], IUSE_TICKER_HIST] if len(idx) > 0 else iuse_price_apr2
            )

        if XEON_TICKER in prices.columns and date in prices.index:
            xeon_price = prices.loc[date, XEON_TICKER]
        else:
            idx = prices.index[prices.index <= date] if XEON_TICKER in prices.columns else pd.DatetimeIndex([])
            xeon_price = (
                prices.loc[idx[-1], XEON_TICKER] if len(idx) > 0 else xeon_price_apr2
            )

        iuse_value = iuse_shares * iuse_price
        xeon_value = xeon_shares * xeon_price
        nav = iuse_value + xeon_value

        rows.append(
            {
                "date": date,
                "nav": nav,
                "cash": 0.0,
                "regime": "phase1",
                "daily_cost": 0.0,
                f"{IUSE_TICKER_HIST}_value": iuse_value,
                f"{XEON_TICKER}_value": xeon_value,
                f"{IUSE_TICKER_HIST}_weight": iuse_value / nav if nav > 0 else 0,
                f"{XEON_TICKER}_weight": xeon_value / nav if nav > 0 else 0,
            }
        )
    return rows


def _eurusd_on(prices: pd.DataFrame, date: pd.Timestamp) -> float:
    """EURUSD (USD por 1 EUR) vigente en 'date' (ultimo dato <= date)."""
    if "EURUSD=X" in prices.columns:
        s = prices["EURUSD=X"].loc[:date].dropna()
        if len(s):
            return float(s.iloc[-1])
    return EURUSD_FALLBACK


def _positions_from_buys(operativa_df: pd.DataFrame) -> dict[str, dict]:
    """Posiciones (titulos) a partir de las compras de una operativa."""
    buys = operativa_df[operativa_df["cantidad"] > 0]
    return {
        r["yf_ticker"]: {"shares": float(r["cantidad"]), "is_eur": r["yf_ticker"] in EUR_QUOTED_TICKERS}
        for _, r in buys.iterrows()
    }


def _apply_rebalance(positions: dict[str, dict], operativa_df: pd.DataFrame,
                     skip_tickers: set[str] = frozenset()) -> dict[str, dict]:
    """Aplica los deltas (compras/ventas) de una operativa a un conjunto de posiciones.

    skip_tickers: tickers cuyo delta se ignora (p. ej. XEON.DE en fase 3, cuya cantidad
    venia con el error 1 USD = 1 EUR; su liquidez se absorbe como residual).
    """
    new = {t: {"shares": v["shares"], "is_eur": v["is_eur"]} for t, v in positions.items()}
    for _, r in operativa_df.iterrows():
        t = r["yf_ticker"]
        if t in skip_tickers:
            continue
        d = float(r["cantidad"])
        if t in new:
            new[t]["shares"] += d
        else:
            new[t] = {"shares": d, "is_eur": t in EUR_QUOTED_TICKERS}
    return {t: v for t, v in new.items() if abs(v["shares"]) > 1e-6}


def _rebalance_cost_eur(operativa_df: pd.DataFrame, fx: float) -> float:
    """Coste total de un rebalanceo en EUR (USD ÷ fx; EUR intactos). One-time."""
    return float(sum(
        (float(r["coste"]) if r["yf_ticker"] in EUR_QUOTED_TICKERS else float(r["coste"]) / fx)
        for _, r in operativa_df.iterrows()
    ))


def _build_segment_rows(
    positions: dict[str, dict],
    prices: pd.DataFrame,
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
    handoff_nav: float,
    regime: str,
    one_time_cost_eur: float = 0.0,
    residual_to_ticker: str | None = None,
) -> list[dict]:
    """Marca un conjunto de posiciones a mercado en EUR entre start y end.

    Cada posicion USD se valora con el EURUSD diario (titulos * cierre / EURUSD); las EUR
    (EUR_QUOTED_TICKERS) no se dividen. La liquidez residual = handoff_nav - EUR desplegado
    el primer dia (constante) -> NAV continuo con el segmento anterior. Sirve para fase 2
    (10-abr→13-may) y fase 3 (13-may→hoy) con la misma logica.

    residual_to_ticker: si se indica, el sobrante del primer dia no se deja como caja sino
    que se aparca en ese ticker (titulos = residual / precio EUR del dia) y se marca a mercado
    como una posicion mas. Se usa para XEON.DE en fase 3: la orden venia mal dimensionada por
    el error 1 USD = 1 EUR, pero el dinero estaba destinado al colchon monetario, no a caja
    ociosa (coherente con la tesis "siempre 100% invertido").
    """
    pos = {t: {"shares": v["shares"], "is_eur": v["is_eur"]} for t, v in positions.items()}

    def deployed_eur(date: pd.Timestamp) -> dict[str, float]:
        fx = _eurusd_on(prices, date)
        vals: dict[str, float] = {}
        for yf_t, p in pos.items():
            close = (float(prices.loc[date, yf_t])
                     if (yf_t in prices.columns and date in prices.index) else float("nan"))
            if pd.notna(close) and close > 0:
                native = p["shares"] * close
                vals[yf_t] = native if p["is_eur"] else native / fx
            else:
                vals[yf_t] = p.get("_last", 0.0)
            p["_last"] = vals[yf_t]
        return vals

    seg_dates = pd.bdate_range(start=start_date, end=end_date)
    seg_tickers = [t for t in pos if t in prices.columns]

    def _coverage(d: pd.Timestamp) -> float:
        if d not in prices.index or not seg_tickers:
            return 0.0
        return sum(1 for t in seg_tickers if pd.notna(prices.loc[d, t])) / len(pos)

    available_dates = [d for d in seg_dates if _coverage(d) >= 0.5] or [start_date]
    residual_cash = float(handoff_nav) - sum(deployed_eur(available_dates[0]).values())

    # Aparcar el sobrante en un ticker monetario (XEON.DE en fase 3) en vez de dejarlo
    # como caja: titulos = residual / precio EUR del primer dia. Por construccion el nuevo
    # desplegado iguala el handoff, asi que la caja residual pasa a ser 0.
    first_date = available_dates[0]
    if (residual_to_ticker is not None and residual_cash > 1e-6
            and residual_to_ticker in prices.columns and first_date in prices.index
            and pd.notna(prices.loc[first_date, residual_to_ticker])):
        is_eur = residual_to_ticker in EUR_QUOTED_TICKERS
        close = float(prices.loc[first_date, residual_to_ticker])
        price_eur = close if is_eur else close / _eurusd_on(prices, first_date)
        if price_eur > 0:
            pos[residual_to_ticker] = {"shares": residual_cash / price_eur, "is_eur": is_eur}
            residual_cash = 0.0

    rows = []
    for i, date in enumerate(available_dates):
        day_values = deployed_eur(date)
        nav = sum(day_values.values()) + residual_cash
        row: dict = {
            "date": date,
            "nav": nav,
            "cash": residual_cash,
            "regime": regime,
            "daily_cost": one_time_cost_eur if i == 0 else 0.0,
        }
        for yf_t, val in day_values.items():
            row[f"{yf_t}_value"] = val
            row[f"{yf_t}_weight"] = val / nav if nav > 0 else 0.0
        rows.append(row)
    return rows


def build_portfolio(
    historial_df: pd.DataFrame,
    operativa_df: pd.DataFrame,
    operativa_fase3_df: "pd.DataFrame | None" = None,
    end_date: "pd.Timestamp | str | None" = None,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    """Reconstruct the full daily portfolio from Mar 12 to end_date (default: today).

    Fases: 1 (Historial, táctica IUSE+XEON) → 2 (10-abr, despliegue 43 ETF) → 3 (13-may,
    segundo rebalanceo, si se pasa operativa_fase3_df). end_date permite cortar la
    reconstrucción (p. ej. 2026-05-14, fin del periodo en vivo del contrafactual).

    Returns (portfolio_df indexed by date, weights_long for charts, failed_tickers).
    """
    phase2_yf_tickers = operativa_df["yf_ticker"].dropna().unique().tolist()
    fase3_yf_tickers = (operativa_fase3_df["yf_ticker"].dropna().unique().tolist()
                        if operativa_fase3_df is not None else [])
    all_tickers = list(set(
        [IUSE_TICKER_HIST, XEON_TICKER, "EURUSD=X"] + phase2_yf_tickers + fase3_yf_tickers
    ))

    today = (pd.Timestamp(end_date).normalize() if end_date is not None
             else pd.Timestamp.today().normalize())
    # Start before Phase 1 to ensure April gap coverage
    prices, failed_tickers = download_prices(
        tickers=all_tickers,
        start="2026-03-10",
        end=(today + pd.Timedelta(days=1)).strftime("%Y-%m-%d"),
    )

    phase1_rows = _build_phase1_rows(historial_df)
    gap_rows = _build_gap_rows(historial_df, prices)
    # NAV en el handoff (fin fase1/gap, EUR real) -> ancla la fase 2 para que sea continua
    handoff_apr10 = (gap_rows[-1]["nav"] if gap_rows
                     else phase1_rows[-1]["nav"] if phase1_rows else CAPITAL_INICIAL)

    pos2 = _positions_from_buys(operativa_df)
    cost2 = _rebalance_cost_eur(operativa_df, _eurusd_on(prices, OPERATIVA_DATE))

    has_fase3 = (operativa_fase3_df is not None and not operativa_fase3_df.empty
                 and today >= PHASE3_DATE)
    if has_fase3:
        # Fase 2 cubre 10-abr→13-may; fase 3 desde 13-may. El handoff a fase 3 = NAV de
        # las posiciones de fase 2 marcadas al 13-may (continuo). El delta de XEON.DE de la
        # hoja de fase 3 venia mal dimensionado por el error 1 USD = 1 EUR (cantidad ~6.8x la
        # correcta, valoraba el monetario contra casi todo el notional) -> se ignora su
        # cantidad y el sobrante se aparca en XEON al precio EUR real (residual_to_ticker),
        # que es lo que la orden pretendia: colchon monetario, no caja ociosa.
        phase2_rows = _build_segment_rows(pos2, prices, OPERATIVA_DATE, PHASE3_DATE,
                                          handoff_apr10, "phase2", cost2)
        handoff_may13 = phase2_rows[-1]["nav"] if phase2_rows else handoff_apr10
        pos3 = _apply_rebalance(pos2, operativa_fase3_df, skip_tickers={XEON_TICKER})
        cost3 = _rebalance_cost_eur(operativa_fase3_df, _eurusd_on(prices, PHASE3_DATE))
        phase3_rows = _build_segment_rows(pos3, prices, PHASE3_DATE, today,
                                          handoff_may13, "phase3", cost3,
                                          residual_to_ticker=XEON_TICKER)
        all_rows = phase1_rows + gap_rows + phase2_rows + phase3_rows
    else:
        phase2_rows = _build_segment_rows(pos2, prices, OPERATIVA_DATE, today,
                                          handoff_apr10, "phase2", cost2)
        all_rows = phase1_rows + gap_rows + phase2_rows

    portfolio_df = pd.DataFrame(all_rows)

    if portfolio_df.empty:
        return portfolio_df, pd.DataFrame(), failed_tickers

    portfolio_df["date"] = pd.to_datetime(portfolio_df["date"])
    portfolio_df = (
        portfolio_df
        .sort_values("date")
        .drop_duplicates("date", keep="last")
        .reset_index(drop=True)
    )
    portfolio_df = portfolio_df.set_index("date")

    portfolio_df["daily_cost"] = portfolio_df["daily_cost"].fillna(0.0)

    # Forward-fill NAV over non-trading days. Position values are ffilled only while
    # their weight column is active, so liquidated positions don't keep stale values.
    portfolio_df["nav"] = portfolio_df["nav"].ffill()
    for vcol in [c for c in portfolio_df.columns if c.endswith("_value")]:
        wcol = vcol.replace("_value", "_weight")
        if wcol not in portfolio_df.columns:
            continue
        active = portfolio_df[wcol].notna()
        portfolio_df.loc[active, vcol] = portfolio_df.loc[active, vcol].ffill()
        portfolio_df.loc[~active, vcol] = np.nan

    portfolio_df = portfolio_df.dropna(subset=["nav"])

    portfolio_df["cum_cost"] = portfolio_df["daily_cost"].cumsum()

    # PnL relative to initial capital
    portfolio_df["daily_pnl"] = portfolio_df["nav"].diff()
    portfolio_df.loc[portfolio_df.index[0], "daily_pnl"] = (
        portfolio_df["nav"].iloc[0] - CAPITAL_INICIAL
    )
    portfolio_df["cum_pnl"] = portfolio_df["nav"] - CAPITAL_INICIAL
    portfolio_df["daily_return"] = portfolio_df["nav"].pct_change()
    portfolio_df.loc[portfolio_df.index[0], "daily_return"] = (
        portfolio_df["nav"].iloc[0] / CAPITAL_INICIAL - 1.0
    )
    portfolio_df["cum_return"] = portfolio_df["nav"] / CAPITAL_INICIAL - 1.0

    # Long-format weights for area/donut charts
    weight_cols = [c for c in portfolio_df.columns if c.endswith("_weight")]
    value_cols = [c.replace("_weight", "_value") for c in weight_cols]

    weight_frames = []
    for wcol, vcol in zip(weight_cols, value_cols):
        ticker = wcol.replace("_weight", "")
        df_t = pd.DataFrame(
            {
                "date": portfolio_df.index,
                "ticker": ticker,
                "weight": portfolio_df[wcol].values,
                "value": portfolio_df[vcol].values if vcol in portfolio_df.columns else np.nan,
            }
        )
        weight_frames.append(df_t)

    # Caja residual como una línea más (categoría Liquidez): sin ella, los pesos suman
    # <100% y el donut/área ocultan la liquidez (la cartera no está 100% invertida).
    if "cash" in portfolio_df.columns:
        cash_w = (portfolio_df["cash"] / portfolio_df["nav"]).clip(lower=0.0)
        weight_frames.append(pd.DataFrame({
            "date": portfolio_df.index,
            "ticker": "Caja",
            "weight": cash_w.values,
            "value": portfolio_df["cash"].values,
        }))

    weights_long = (
        pd.concat(weight_frames, ignore_index=True)
        .dropna(subset=["weight"])
        .query("weight > 0")
        .reset_index(drop=True)
    )
    weights_long["date"] = pd.to_datetime(weights_long["date"])

    return portfolio_df, weights_long, failed_tickers
