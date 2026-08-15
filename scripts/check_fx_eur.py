"""
Smoke de la conversion a EUR: un ticker USD (SPY) queda = nativo/EURUSD, un ticker
EUR (XEON.DE) queda intacto, y un backtest corto no produce 'nan' en las metricas.

Uso: python scripts/check_fx_eur.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import yfinance as yf

from src.data.data_loader import download_market_data, _download_eurusd


def check_fx_conversion() -> None:
    start, end = "2023-01-01", "2023-03-31"

    md = download_market_data(start_date=start, end_date=end, tickers=["SPY", "XEON.DE"])
    prices_eur = md["prices"]

    eurusd = _download_eurusd(start, end).reindex(prices_eur.index).ffill().bfill()

    # SPY nativo (USD) directo de yfinance para comparar
    spy_native = yf.download("SPY", start=start, end=end, auto_adjust=True, progress=False)
    spy_native = spy_native["Close"]
    if hasattr(spy_native, "columns"):
        spy_native = spy_native.iloc[:, 0]
    spy_native = spy_native.reindex(prices_eur.index).ffill().bfill()

    spy_expected_eur = spy_native / eurusd
    diff = (prices_eur["SPY"] - spy_expected_eur).abs() / spy_expected_eur
    max_rel = float(diff.dropna().max())
    assert max_rel < 0.01, f"SPY EUR no coincide con nativo/EURUSD (max rel diff {max_rel:.4f})"
    print(f"[OK] SPY convertido a EUR = nativo/EURUSD (max diff relativo {max_rel:.5f})")

    # XEON.DE ya cotiza en EUR: debe quedar practicamente intacto vs su nativo
    xeon_native = yf.download("XEON.DE", start=start, end=end, auto_adjust=True, progress=False)
    xeon_native = xeon_native["Close"]
    if hasattr(xeon_native, "columns"):
        xeon_native = xeon_native.iloc[:, 0]
    xeon_native = xeon_native.reindex(prices_eur.index).ffill().bfill()
    diff_x = (prices_eur["XEON.DE"] - xeon_native).abs() / xeon_native
    max_rel_x = float(diff_x.dropna().max())
    assert max_rel_x < 1e-6, f"XEON.DE (EUR) fue alterado (max rel diff {max_rel_x})"
    print(f"[OK] XEON.DE (EUR) intacto (max diff relativo {max_rel_x:.2e})")


def check_backtest_no_nan() -> None:
    from src.backtest.engine import run_backtest

    r = run_backtest(
        start_date="2023-01-01", end_date="2023-06-30",
        output_dir="/tmp/smoke_fx_out", verbose=False,
    )
    metrics = r["metrics_comparison"]
    as_text = metrics.to_string().lower()
    assert "nan" not in as_text, f"metrics_comparison contiene nan:\n{metrics}"
    print("[OK] backtest corto en EUR sin nan en metricas:")
    print(metrics.to_string(index=False))


if __name__ == "__main__":
    try:
        check_fx_conversion()
        check_backtest_no_nan()
    except AssertionError as exc:
        print(f"[FALLO] {exc}")
        sys.exit(1)
    print("\nSMOKE FX OK")
