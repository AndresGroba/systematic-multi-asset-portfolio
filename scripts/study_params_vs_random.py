"""
"Monos": distribucion nula de parametros aleatorios. Muestrea N juegos del nucleo al
azar y compara los params elegidos (fijos actuales) contra esa nube: percentil del
Sharpe, % de monos que lo baten, % que baten al SP500. Dos periodos (2020-2026, 2013-2026).

Uso: python scripts/study_params_vs_random.py [N]   # default 200
Salida: outputs/studies/params_vs_random/monkeys_<periodo>.csv
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

import src.config as cfg
import src.data.data_loader as dl
import src.backtest.engine as engine
import src.models.black_litterman as bl
from src.backtest.engine import run_backtest
from param_nucleus import names, sample

SEED = 123
N = int(sys.argv[1]) if len(sys.argv) > 1 else 500
LOOKBACK_START = "2010-07-01"   # cubre el lookback de 18m del periodo mas largo
PERIODS = [
    ("2020-2026", cfg.SUBPERIOD_START, cfg.BACKTEST_DATA_END),
    ("2013-2026", cfg.CANONICAL_START, cfg.BACKTEST_DATA_END),
]

cfg.USE_ML_FILTER = False
cfg.USE_CLUSTERING = False
bl.log_decisions = lambda *a, **k: None

SPY = cfg.BENCHMARK_TICKER
URTH = getattr(cfg, "BENCHMARK_MSCI_WORLD_TICKER", "URTH")


def _build_fixed_data():
    real = dl.download_market_data
    return {
        "uni": real(start_date=LOOKBACK_START, end_date=cfg.BACKTEST_DATA_END),
        "spy": real(start_date=LOOKBACK_START, end_date=cfg.BACKTEST_DATA_END, tickers=[SPY]),
        "urth": real(start_date=LOOKBACK_START, end_date=cfg.BACKTEST_DATA_END, tickers=[URTH]),
    }


def _make_fake(fixed):
    def fake(start_date=None, end_date=None, tickers=None, **kwargs):
        base = fixed["uni"] if tickers is None else (fixed["spy"] if list(tickers) == [SPY] else fixed["urth"])
        end = pd.Timestamp(end_date)
        return {
            "tickers": [c for c in base["prices"].columns],
            "prices": base["prices"].loc[:end],
            "returns": base["returns"].loc[:end],
            "metadata": base["metadata"],
            "transaction_costs": base["transaction_costs"],
        }
    return fake


def _run(start, end, tag):
    r = run_backtest(start, end, output_dir=f"/tmp/monkeys/{tag}", verbose=False)
    # Curva de valor a fin de mes (para el gráfico de la nube de monos en el dashboard)
    nav = r["wealth"]["Strategy"].resample("ME").last()
    return r["strategy_metrics"]["Sharpe"], r["strategy_metrics"]["CAGR"], r["sp500_metrics"]["Sharpe"], nav


def main():
    print(f"Monos | N={N} params aleatorios/periodo | ML off")
    t0 = time.time()
    fixed = _build_fixed_data()
    fake = _make_fake(fixed)
    dl.download_market_data = fake
    engine.download_market_data = fake

    orig = {k: getattr(cfg, k) for k in names()}
    Path("outputs/studies/params_vs_random").mkdir(parents=True, exist_ok=True)

    chosen_rows = []
    for name, start, end in PERIODS:
        # params elegidos (fijos actuales)
        for k, v in orig.items():
            setattr(cfg, k, v)
        sh_chosen, cagr_chosen, sh_spy, nav_chosen = _run(start, end, f"{name}_chosen")

        # monos
        rng = np.random.default_rng(SEED)
        rows = []
        curves = {}  # i -> curva NAV mensual del mono i
        for i in range(N):
            p = sample(rng)
            for k, v in p.items():
                setattr(cfg, k, v)
            sh, cagr, _, nav = _run(start, end, f"{name}_monkey")
            rows.append({"Sharpe": sh, "CAGR": cagr, **p})
            curves[f"m{i}"] = nav

        df = pd.DataFrame(rows)
        # Curvas de valor: una columna por mono + 'chosen' (nuestra estrategia), fin de mes
        wealth = pd.DataFrame(curves)
        wealth["chosen"] = nav_chosen
        wealth.index.name = "Date"
        wealth.to_csv(f"outputs/studies/params_vs_random/monkeys_wealth_{name}.csv")
        sh = df["Sharpe"].to_numpy()
        pct_chosen = float((sh < sh_chosen).mean() * 100)   # percentil del elegido
        pct_beat_chosen = float((sh >= sh_chosen).mean() * 100)
        pct_beat_spy = float((sh >= sh_spy).mean() * 100)

        print(f"\n=== {name} ({start} -> {end}) ===")
        print(f"  ELEGIDOS:  Sharpe {sh_chosen:.3f} | CAGR {cagr_chosen*100:.2f}%")
        print(f"  SP500:     Sharpe {sh_spy:.3f}")
        print(f"  MONOS (N={N}): Sharpe  p10={np.percentile(sh,10):.3f}  "
              f"mediana={np.median(sh):.3f}  p90={np.percentile(sh,90):.3f}  max={sh.max():.3f}")
        print(f"  -> percentil de los ELEGIDOS en la nube: {pct_chosen:.0f}%")
        print(f"  -> % de monos que BATEN a los elegidos: {pct_beat_chosen:.0f}%")
        print(f"  -> % de monos que BATEN al SP500:       {pct_beat_spy:.0f}%")

        df.to_csv(f"outputs/studies/params_vs_random/monkeys_{name}.csv", index=False)
        chosen_rows.append({
            "periodo": name, "chosen_Sharpe": sh_chosen, "chosen_CAGR": cagr_chosen,
            "SP500_Sharpe": sh_spy, "percentil": pct_chosen, "pct_baten_chosen": pct_beat_chosen,
        })

    # Sharpe de los params elegidos por periodo: lo consume el dashboard para marcar
    # dónde cae la estrategia en la nube de monos (percentil). Persistido aquí para no
    # depender de re-derivarlo desde la ablación (que solo cubre el periodo largo).
    pd.DataFrame(chosen_rows).to_csv(
        "outputs/studies/params_vs_random/chosen.csv", index=False)

    for k, v in orig.items():
        setattr(cfg, k, v)
    print(f"\nGuardado en outputs/studies/params_vs_random/ | tiempo {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
