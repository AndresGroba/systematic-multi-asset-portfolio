"""
Smoke de la ruta LIVE de ejecucion: get_execution_data devuelve el precio de la sesion
SIGUIENTE convertido a EUR para un ETF USD (SPY), y deja intacto un ETF EUR (XEON.DE).

Antes del fix, future_prices se mezclaba en divisa nativa (USD) sobre los EUR de
clean_prices -> ordenes con titulos/nominal erroneos para ETFs USD. Este smoke certifica
el boundary real (una descarga yfinance) en lugar de mocks.

El valor esperado en EUR se deriva de forma INDEPENDIENTE (nativo_USD / EURUSD de la misma
sesion), no del propio output de get_execution_data.

Uso: python scripts/check_fx_execution_live.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
import yfinance as yf

from src.data.data_loader import download_market_data, get_execution_data, _download_eurusd


# Ventana historica: deja sesiones POSTERIORES disponibles para forzar used_next_day=True.
START, END = "2023-01-01", "2023-06-15"
USD_TICKER = "SPY"
EUR_TICKER = "XEON.DE"


def _native_close(ticker: str, start: str, end: str) -> pd.Series:
    raw = yf.download(ticker, start=start, end=end, auto_adjust=True, progress=False)
    close = raw["Close"]
    if hasattr(close, "columns"):
        close = close.iloc[:, 0]
    return close.dropna()


def _value_on(series: pd.Series, date: pd.Timestamp) -> float:
    idx = pd.Index(pd.to_datetime(series.index)).normalize()
    target = pd.Timestamp(date).normalize()
    matches = series[idx == target]
    if matches.empty:
        raise AssertionError(f"No hay precio nativo de referencia en {target.date()}")
    return float(matches.iloc[0])


def check_execution_eur() -> None:
    md = download_market_data(start_date=START, end_date=END, tickers=[USD_TICKER, EUR_TICKER])
    prices_eur = md["prices"]

    result = get_execution_data([USD_TICKER, EUR_TICKER], prices_eur)

    # El smoke solo prueba la ruta del bug si se usa la sesion siguiente.
    assert result["used_next_day"], (
        f"used_next_day=False (source={result['execution_source']}): no se ejercita la ruta "
        "future_session; ajusta la ventana para dejar sesiones posteriores."
    )
    exec_date = pd.Timestamp(result["date"])
    exec_prices = result["prices"]
    print(f"[INFO] sesion de ejecucion = {exec_date.date()} (used_next_day=True)")

    # --- ETF USD: el precio de ejecucion debe estar en EUR, no en USD ---
    # Ventana amplia para cubrir la sesion de ejecucion (END es exclusivo en yfinance).
    nat_start, nat_end = START, "2023-06-30"
    spy_native_usd = _native_close(USD_TICKER, nat_start, nat_end)
    eurusd = _download_eurusd(nat_start, nat_end)

    usd_on_exec = _value_on(spy_native_usd, exec_date)
    fx_on_exec = _value_on(eurusd, exec_date)
    expected_eur = usd_on_exec / fx_on_exec  # esperado derivado de forma independiente

    actual = float(exec_prices[USD_TICKER])
    rel_to_eur = abs(actual - expected_eur) / expected_eur
    rel_to_usd = abs(actual - usd_on_exec) / usd_on_exec

    print(
        f"[INFO] {USD_TICKER}: nativo={usd_on_exec:.2f} USD, EURUSD={fx_on_exec:.4f}, "
        f"esperado={expected_eur:.2f} EUR, get_execution_data={actual:.2f}"
    )
    assert rel_to_eur < 0.01, (
        f"{USD_TICKER} no esta en EUR: actual {actual:.2f} vs esperado_EUR {expected_eur:.2f} "
        f"(rel {rel_to_eur:.4f})"
    )
    # Sanidad: USD y EUR difieren lo suficiente (EURUSD lejos de 1) para que el test discrimine.
    assert rel_to_usd > 0.02, (
        f"USD y EUR demasiado proximos (rel {rel_to_usd:.4f}); el test no discrimina con esta FX"
    )
    print(f"[OK] {USD_TICKER} convertido a EUR (rel vs EUR {rel_to_eur:.5f}, lejos del USD nativo)")

    # --- ETF EUR: debe quedar intacto (= nativo de la sesion de ejecucion) ---
    xeon_native = _native_close(EUR_TICKER, nat_start, nat_end)
    xeon_on_exec = _value_on(xeon_native, exec_date)
    actual_x = float(exec_prices[EUR_TICKER])
    rel_x = abs(actual_x - xeon_on_exec) / xeon_on_exec
    print(f"[INFO] {EUR_TICKER}: nativo={xeon_on_exec:.4f} EUR, get_execution_data={actual_x:.4f}")
    assert rel_x < 1e-6, (
        f"{EUR_TICKER} (EUR) fue alterado: actual {actual_x} vs nativo {xeon_on_exec} (rel {rel_x})"
    )
    print(f"[OK] {EUR_TICKER} (EUR) intacto (rel {rel_x:.2e})")


if __name__ == "__main__":
    try:
        check_execution_eur()
    except AssertionError as exc:
        print(f"[FALLO] {exc}")
        sys.exit(1)
    print("\nSMOKE FX EXECUTION OK")
