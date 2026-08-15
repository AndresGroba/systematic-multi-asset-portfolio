"""
Optimizacion del nucleo por walk-forward (fuera de muestra). Selecciona params solo en
train (random search, objetivo Sharpe - LAMBDA*turnover) y los evalua en una ventana
posterior nunca vista; el OOS concatenado es el numero honesto. Compara params
walk-forward vs fijos actuales vs benchmarks.

Uso: python scripts/study_params_oos.py [N_SAMPLES]   # default 80
Salida: outputs/studies/params_oos/walkforward.csv
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

SEED = 42
# 12 dims: el random search necesita mas muestras que con los 8 de antes.
N_SAMPLES = int(sys.argv[1]) if len(sys.argv) > 1 else 150
LAMBDA_TURNOVER = 0.1          # penalizacion por turnover anual (%/año)
START = "2014-01-01"
END_FINAL = cfg.BACKTEST_DATA_END
# (train_end, test_start, test_end) — ventana anclada/expansiva
FOLDS = [
    ("2017-12-31", "2018-01-01", "2019-12-31"),
    ("2019-12-31", "2020-01-01", "2021-12-31"),
    ("2021-12-31", "2022-01-01", "2023-12-31"),
    ("2023-12-31", "2024-01-01", END_FINAL),
]

# ML off para acelerar (XGBoost es la parte lenta). Aqui se optimiza el nucleo.
cfg.USE_ML_FILTER = False
cfg.USE_CLUSTERING = False
bl.log_decisions = lambda *a, **k: None  # no escribir outputs/decisions en cientos de corridas

SPY = cfg.BENCHMARK_TICKER
URTH = getattr(cfg, "BENCHMARK_MSCI_WORLD_TICKER", "URTH")
COLS = ["Strategy", "SP500", "MSCI_World (URTH)"]


def _build_fixed_data() -> dict:
    lb = (pd.Timestamp(START) - pd.DateOffset(months=18)).strftime("%Y-%m-%d")
    real = dl.download_market_data
    return {
        "uni": real(start_date=lb, end_date=END_FINAL),
        "spy": real(start_date=lb, end_date=END_FINAL, tickers=[SPY]),
        "urth": real(start_date=lb, end_date=END_FINAL, tickers=[URTH]),
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
    r = run_backtest(start, end, output_dir=f"/tmp/wfo/{tag}", verbose=False)
    years = max((pd.Timestamp(end) - pd.Timestamp(start)).days / 365.25, 1e-6)
    cost_pct_yr = r["total_transaction_costs"] / float(cfg.INITIAL_CAPITAL) / years * 100.0
    return r["strategy_metrics"], cost_pct_yr


def _returns(tag) -> pd.DataFrame:
    w = pd.read_csv(f"/tmp/wfo/{tag}/wealth_history.csv", parse_dates=["Date"]).set_index("Date")
    return w[[c for c in COLS if c in w.columns]].pct_change().dropna()


def _agg(series):
    s = series.dropna()
    vol = s.std() * np.sqrt(252)
    sharpe = (s.mean() * 252) / vol if vol > 0 else 0.0
    total_return = (1 + s).prod() - 1
    return sharpe, total_return


def main():
    print(f"Walk-forward optim | N_SAMPLES={N_SAMPLES}/fold | {len(FOLDS)} folds | ML off")
    t0 = time.time()
    fixed = _build_fixed_data()
    fake = _make_fake(fixed)
    dl.download_market_data = fake
    engine.download_market_data = fake

    rng = np.random.default_rng(SEED)
    orig = {k: getattr(cfg, k) for k in names()}
    sel_ret, fix_ret, rows = [], [], []

    for i, (tr_end, te_start, te_end) in enumerate(FOLDS):
        best = None
        for _ in range(N_SAMPLES):
            p = sample(rng)
            for k, v in p.items():
                setattr(cfg, k, v)
            m, cost = _run(START, tr_end, f"f{i}_tr")
            obj = m["Sharpe"] - LAMBDA_TURNOVER * cost
            if best is None or obj > best[0]:
                best = (obj, p, m["Sharpe"])

        for k, v in best[1].items():       # test con los params elegidos
            setattr(cfg, k, v)
        mt, _ = _run(te_start, te_end, f"f{i}_te")
        sel_ret.append(_returns(f"f{i}_te"))

        for k, v in orig.items():          # baseline: params fijos actuales
            setattr(cfg, k, v)
        _run(te_start, te_end, f"f{i}_fx")
        fix_ret.append(_returns(f"f{i}_fx")["Strategy"])

        rows.append({
            "fold": i + 1, "test": f"{te_start[:7]}..{te_end[:7]}",
            "train_obj": round(best[0], 3), "train_Sharpe": round(best[2], 3),
            "test_Sharpe_sel": round(mt["Sharpe"], 3),
            **{k: round(best[1][k], 4) for k in names()},
        })
        print(f"  fold {i+1} (test {te_start[:7]}..{te_end[:7]}): "
              f"train Sharpe {best[2]:.2f} -> test Sharpe(sel) {mt['Sharpe']:.2f}")

    for k, v in orig.items():
        setattr(cfg, k, v)

    concat = pd.concat(sel_ret)
    shp_sel, tr_sel = _agg(concat["Strategy"])
    shp_fix, tr_fix = _agg(pd.concat(fix_ret))
    shp_spy = _agg(concat["SP500"])[0] if "SP500" in concat else float("nan")
    shp_urth = _agg(concat["MSCI_World (URTH)"])[0] if "MSCI_World (URTH)" in concat else float("nan")

    print(f"\n=== OOS concatenado ({FOLDS[0][1][:7]} .. {FOLDS[-1][2][:7]}) ===")
    print(f"  params walk-forward : Sharpe {shp_sel:.3f} | TR {tr_sel*100:6.1f}%")
    print(f"  params fijos (hoy)  : Sharpe {shp_fix:.3f} | TR {tr_fix*100:6.1f}%")
    print(f"  SP500               : Sharpe {shp_spy:.3f}")
    print(f"  MSCI World (URTH)   : Sharpe {shp_urth:.3f}")

    out = pd.DataFrame(rows)
    Path("outputs/studies/params_oos").mkdir(parents=True, exist_ok=True)
    out.to_csv("outputs/studies/params_oos/walkforward.csv", index=False)

    # Resultado titular: Sharpe OOS concatenado de cada serie (lo consume el dashboard).
    oos_start, oos_end = FOLDS[0][1][:7], FOLDS[-1][2][:7]
    pd.DataFrame([
        {"serie": "Params fijos (adoptados)", "Sharpe_OOS": round(shp_fix, 3), "TotalReturn": round(tr_fix, 4)},
        {"serie": "Params re-optimizados", "Sharpe_OOS": round(shp_sel, 3), "TotalReturn": round(tr_sel, 4)},
        {"serie": "SP500", "Sharpe_OOS": round(shp_spy, 3), "TotalReturn": float("nan")},
        {"serie": "MSCI World", "Sharpe_OOS": round(shp_urth, 3), "TotalReturn": float("nan")},
    ]).to_csv("outputs/studies/params_oos/oos_summary.csv", index=False)

    print("\nParams elegidos por fold (estabilidad):")
    print(out.to_string(index=False))
    print(f"\nGuardado en outputs/studies/params_oos/ (ventana OOS {oos_start}..{oos_end}) | "
          f"tiempo {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
