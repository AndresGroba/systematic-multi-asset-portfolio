"""
¿Trimestral bate a mensual de forma robusta? Compara frecuencias directamente sobre
dos periodos (2020-2026 y el largo 2013-2026) y año a año; ganar en ambos descarta el
artefacto de la ventana in-sample. Resto de params actuales, ML off.

Uso: python scripts/study_rebalance_frequency.py
Salida: outputs/studies/rebalance_frequency/freq_compare.csv
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

import src.config as cfg
import src.data.data_loader as dl
import src.backtest.engine as engine
import src.models.black_litterman as bl
from src.backtest.engine import run_backtest

FREQS = {"W": "semanal", "2W": "quincenal", "ME": "mensual", "QE": "trimestral"}
PERIODS = [("2020-2026", cfg.SUBPERIOD_START, cfg.BACKTEST_DATA_END), ("2013-2026", cfg.CANONICAL_START, cfg.BACKTEST_DATA_END)]
LOOKBACK_START = "2010-07-01"

cfg.USE_ML_FILTER = False
cfg.USE_CLUSTERING = False
bl.log_decisions = lambda *a, **k: None
SPY = cfg.BENCHMARK_TICKER
URTH = getattr(cfg, "BENCHMARK_MSCI_WORLD_TICKER", "URTH")


def _fixed():
    real = dl.download_market_data
    return {"uni": real(start_date=LOOKBACK_START, end_date=cfg.BACKTEST_DATA_END),
            "spy": real(start_date=LOOKBACK_START, end_date=cfg.BACKTEST_DATA_END, tickers=[SPY]),
            "urth": real(start_date=LOOKBACK_START, end_date=cfg.BACKTEST_DATA_END, tickers=[URTH])}


def _make_fake(fx):
    def fake(start_date=None, end_date=None, tickers=None, **k):
        base = fx["uni"] if tickers is None else (fx["spy"] if list(tickers) == [SPY] else fx["urth"])
        end = pd.Timestamp(end_date)
        return {"tickers": [c for c in base["prices"].columns], "prices": base["prices"].loc[:end],
                "returns": base["returns"].loc[:end], "metadata": base["metadata"],
                "transaction_costs": base["transaction_costs"]}
    return fake


def _sharpe(s):
    s = s.dropna(); v = s.std() * np.sqrt(252)
    return (s.mean() * 252) / v if v > 0 else 0.0


def main():
    fake = _make_fake(_fixed())
    dl.download_market_data = fake
    engine.download_market_data = fake

    rows = []
    wealth = {}  # (freq,period) -> Strategy series ; ('SP500',period)
    for name, start, end in PERIODS:
        for fq, label in FREQS.items():
            cfg.REBALANCE_FREQ = fq
            r = run_backtest(start, end, output_dir=f"/tmp/fc/{name}_{fq}", verbose=False)
            m = r["strategy_metrics"]
            w = pd.read_csv(f"/tmp/fc/{name}_{fq}/wealth_history.csv", parse_dates=["Date"]).set_index("Date")
            wealth[(fq, name)] = w["Strategy"]
            if "SP500" in w:
                wealth[("SP500", name)] = w["SP500"]
            rows.append({
                "periodo": name, "frecuencia": label, "Sharpe": round(m["Sharpe"], 3),
                "CAGR": round(m["CAGR"] * 100, 2), "MaxDD": round(m["Max Drawdown"] * 100, 2),
                "costes_%": round(r["total_transaction_costs"] / float(cfg.INITIAL_CAPITAL) * 100, 2),
                "rebal": r["n_rebalances"], "SP500_Sharpe": round(r["sp500_metrics"]["Sharpe"], 3),
            })
    cfg.REBALANCE_FREQ = "ME"

    df = pd.DataFrame(rows)
    print("=== Comparación de frecuencias (resto de params = actuales, ML off) ===")
    for name, _, _ in PERIODS:
        print(f"\n-- {name} --")
        print(df[df.periodo == name].drop(columns=["periodo"]).to_string(index=False))

    # Estabilidad año a año: Sharpe mensual vs trimestral (periodo largo)
    print("\n=== Estabilidad año a año (2013-2026): Sharpe por año ===")
    me = wealth[("ME", "2013-2026")].pct_change().dropna()
    qe = wealth[("QE", "2013-2026")].pct_change().dropna()
    spy = wealth[("SP500", "2013-2026")].pct_change().dropna()
    print(f"{'año':>6}{'mensual':>10}{'trimestral':>12}{'SP500':>10}{'  gana trim?':>12}")
    wins = 0; tot = 0
    for y in sorted(set(me.index.year)):
        sm, sq, ss = _sharpe(me[me.index.year == y]), _sharpe(qe[qe.index.year == y]), _sharpe(spy[spy.index.year == y])
        tot += 1; wins += int(sq > sm)
        print(f"{y:>6}{sm:>10.2f}{sq:>12.2f}{ss:>10.2f}{('  sí' if sq>sm else '  no'):>12}")
    print(f"\nTrimestral bate a mensual en {wins}/{tot} años")

    Path("outputs/studies/rebalance_frequency").mkdir(parents=True, exist_ok=True)
    df.to_csv("outputs/studies/rebalance_frequency/freq_compare.csv", index=False)
    print("Guardado: outputs/studies/rebalance_frequency/freq_compare.csv")


if __name__ == "__main__":
    main()
