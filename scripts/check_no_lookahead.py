"""
Certificacion empirica de ausencia de look-ahead por invariancia de truncamiento:
extender la fecha final del backtest no debe alterar ninguna decision ni el NAV de
fechas anteriores. Corre dos backtests sobre datos identicos (una sola descarga, para
aislar el drift de yfinance) y compara lo que cae en [START, E_SHORT].

Uso: python scripts/check_no_lookahead.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

import src.config as cfg
import src.data.data_loader as dl
import src.backtest.engine as engine
from src.backtest.engine import run_backtest

START = "2020-01-01"
E_SHORT = "2024-05-31"   # fin de mes: alinea el calendario de revisiones con la corrida larga
E_FULL = "2026-04-28"


def _build_fixed_data() -> dict:
    """Descarga una vez (universo + SPY + URTH) hasta E_FULL y la cachea."""
    lookback = (pd.Timestamp(START) - pd.DateOffset(months=18)).strftime("%Y-%m-%d")
    real = dl.download_market_data
    spy = cfg.BENCHMARK_TICKER
    urth = getattr(cfg, "BENCHMARK_MSCI_WORLD_TICKER", "URTH")
    return {
        "lookback": lookback,
        "uni": real(start_date=lookback, end_date=E_FULL),
        "spy": real(start_date=lookback, end_date=E_FULL, tickers=[spy]),
        "urth": real(start_date=lookback, end_date=E_FULL, tickers=[urth]),
        "spy_t": spy,
        "urth_t": urth,
    }


def _make_fake_download(fixed: dict):
    def fake(start_date=None, end_date=None, tickers=None, **kwargs):
        if tickers is None:
            base = fixed["uni"]
        elif list(tickers) == [fixed["spy_t"]]:
            base = fixed["spy"]
        else:
            base = fixed["urth"]
        end = pd.Timestamp(end_date)
        prices = base["prices"].loc[:end]
        returns = base["returns"].loc[:end]
        return {
            "tickers": [c for c in prices.columns],
            "prices": prices,
            "returns": returns,
            "metadata": base["metadata"],
            "transaction_costs": base["transaction_costs"],
        }
    return fake


def main() -> None:
    print("Descargando datos fijos (una vez)...")
    fixed = _build_fixed_data()
    fake = _make_fake_download(fixed)
    # El engine importo download_market_data en su namespace: parchear ambos.
    dl.download_market_data = fake
    engine.download_market_data = fake

    print(f"Backtest A (end={E_SHORT})...")
    run_backtest(START, E_SHORT, output_dir="/tmp/la_short", verbose=False)
    print(f"Backtest B (end={E_FULL})...")
    run_backtest(START, E_FULL, output_dir="/tmp/la_full", verbose=False)

    cut = pd.Timestamp(E_SHORT)
    ok = True

    # Decisiones: solo en fechas de revision comunes (la larga puede tener una
    # revision de borde extra que la corta no alcanzo — calendario, no look-ahead).
    da = pd.read_csv("/tmp/la_short/backtest_resumen.csv", parse_dates=["Date"]).set_index("Date")
    db = pd.read_csv("/tmp/la_full/backtest_resumen.csv", parse_dates=["Date"]).set_index("Date")
    da = da[da.index <= cut]
    db = db[db.index <= cut]
    cols = ["Rebalance", "Selected ETFs", "Weight XEON", "Regime", "Portfolio Value"]
    common = da.index.intersection(db.index)
    only_full = db.index.difference(da.index)  # revisiones de borde solo en la larga
    if da.loc[common, cols].equals(db.loc[common, cols]):
        print(f"[OK] Decisiones identicas en {len(common)} fechas de revision comunes")
    else:
        ok = False
        d = (da.loc[common, cols] != db.loc[common, cols]).any(axis=1)
        print(f"[FALLO] Decisiones DIFIEREN en {int(d.sum())} fechas:")
        print(da.loc[common][d][cols].head().to_string())
        print("--- vs ---")
        print(db.loc[common][d][cols].head().to_string())

    # NAV: identico hasta el dia anterior a la primera revision exclusiva de la
    # larga (a partir de ahi divergen por esa revision, no por fuga).
    horizon = (only_full.min() if len(only_full) else cut)
    wa = pd.read_csv("/tmp/la_short/wealth_history.csv", parse_dates=["Date"]).set_index("Date")["Strategy"]
    wb = pd.read_csv("/tmp/la_full/wealth_history.csv", parse_dates=["Date"]).set_index("Date")["Strategy"]
    cw = wa.index.intersection(wb.index)
    cw = cw[cw < horizon]
    max_diff = float((wa.loc[cw] - wb.loc[cw]).abs().max()) if len(cw) else 0.0
    if max_diff < 1e-6:
        print(f"[OK] NAV diario identico en {len(cw)} dias < {horizon.date()} (max dif={max_diff:.2e} EUR)")
    else:
        ok = False
        print(f"[FALLO] NAV diario difiere antes del borde (max dif={max_diff:.2f} EUR)")
    if len(only_full):
        print(f"  (nota: la corrida larga tiene {len(only_full)} revision(es) de borde extra: "
              f"{[str(d.date()) for d in only_full]} — calendario, no look-ahead)")

    print("\n" + ("=" * 60))
    if ok:
        print("VEREDICTO: SIN LOOK-AHEAD. Extender el futuro no altera el pasado.")
    else:
        print("VEREDICTO: HAY FUGA. El pasado cambia al extender el futuro -> look-ahead.")
    print("=" * 60)


if __name__ == "__main__":
    main()
