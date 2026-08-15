"""
¿El veredicto OOS (fijos vs re-optimizados) es robusto o ruido? El random search del
walk-forward es estocástico: con N=150 ganan los fijos, con N=300 los re-optimizados.
Aqui se repite el OOS concatenado con VARIAS SEMILLAS (mismo N) y se mira la distribucion
del gap (tuned - fixed). Si straddlea 0 -> la "ventaja" de afinar es ruido del diseño, no
señal -> se mantienen los fijos. Read-only: no toca los outputs canonicos.

Uso: python scripts/study_oos_robustness.py [N_SAMPLES]   # default 300
Salida: outputs/studies/oos_robustness/seeds.csv
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

N_SAMPLES = int(sys.argv[1]) if len(sys.argv) > 1 else 300
SEEDS = [42, 7, 13, 99, 123]
LAMBDA_TURNOVER = 0.1
START = "2014-01-01"
END_FINAL = cfg.BACKTEST_DATA_END
FOLDS = [
    ("2017-12-31", "2018-01-01", "2019-12-31"),
    ("2019-12-31", "2020-01-01", "2021-12-31"),
    ("2021-12-31", "2022-01-01", "2023-12-31"),
    ("2023-12-31", "2024-01-01", END_FINAL),
]
cfg.USE_ML_FILTER = False
cfg.USE_CLUSTERING = False
bl.log_decisions = lambda *a, **k: None

SPY = cfg.BENCHMARK_TICKER
URTH = getattr(cfg, "BENCHMARK_MSCI_WORLD_TICKER", "URTH")
COLS = ["Strategy", "SP500", "MSCI_World (URTH)"]


def _fixed_data():
    lb = (pd.Timestamp(START) - pd.DateOffset(months=18)).strftime("%Y-%m-%d")
    real = dl.download_market_data
    return {"uni": real(start_date=lb, end_date=END_FINAL),
            "spy": real(start_date=lb, end_date=END_FINAL, tickers=[SPY]),
            "urth": real(start_date=lb, end_date=END_FINAL, tickers=[URTH])}


def _make_fake(fx):
    def fake(start_date=None, end_date=None, tickers=None, **k):
        base = fx["uni"] if tickers is None else (fx["spy"] if list(tickers) == [SPY] else fx["urth"])
        end = pd.Timestamp(end_date)
        return {"tickers": [c for c in base["prices"].columns], "prices": base["prices"].loc[:end],
                "returns": base["returns"].loc[:end], "metadata": base["metadata"],
                "transaction_costs": base["transaction_costs"]}
    return fake


def _run(start, end, tag):
    r = run_backtest(start, end, output_dir=f"/tmp/oosrob/{tag}", verbose=False)
    years = max((pd.Timestamp(end) - pd.Timestamp(start)).days / 365.25, 1e-6)
    cost = r["total_transaction_costs"] / float(cfg.INITIAL_CAPITAL) / years * 100.0
    return r["strategy_metrics"], cost


def _returns(tag):
    w = pd.read_csv(f"/tmp/oosrob/{tag}/wealth_history.csv", parse_dates=["Date"]).set_index("Date")
    return w[[c for c in COLS if c in w.columns]].pct_change().dropna()


def _sharpe(s):
    s = s.dropna()
    v = s.std() * np.sqrt(252)
    return (s.mean() * 252) / v if v > 0 else 0.0


def oos_for_seed(seed):
    rng = np.random.default_rng(seed)
    orig = {k: getattr(cfg, k) for k in names()}
    sel_ret, fix_ret = [], []
    for i, (tr_end, te_start, te_end) in enumerate(FOLDS):
        best = None
        for _ in range(N_SAMPLES):
            p = sample(rng)
            for k, v in p.items():
                setattr(cfg, k, v)
            m, cost = _run(START, tr_end, f"tr")
            obj = m["Sharpe"] - LAMBDA_TURNOVER * cost
            if best is None or obj > best[0]:
                best = (obj, p)
        for k, v in best[1].items():
            setattr(cfg, k, v)
        _run(te_start, te_end, "te")
        sel_ret.append(_returns("te")["Strategy"])
        for k, v in orig.items():
            setattr(cfg, k, v)
        _run(te_start, te_end, "fx")
        fix_ret.append(_returns("fx")["Strategy"])
    for k, v in orig.items():
        setattr(cfg, k, v)
    return _sharpe(pd.concat(sel_ret)), _sharpe(pd.concat(fix_ret))


def main():
    print(f"OOS robustez | {len(SEEDS)} semillas x N={N_SAMPLES} | folds {len(FOLDS)} | ML off")
    t0 = time.time()
    fake = _make_fake(_fixed_data())
    dl.download_market_data = fake
    engine.download_market_data = fake

    rows = []
    for s in SEEDS:
        tuned, fixed = oos_for_seed(s)
        rows.append({"seed": s, "tuned_OOS": round(tuned, 4), "fixed_OOS": round(fixed, 4),
                     "gap_tuned_minus_fixed": round(tuned - fixed, 4)})
        print(f"  seed {s}: tuned {tuned:.3f} | fixed {fixed:.3f} | gap {tuned-fixed:+.3f} "
              f"({time.time()-t0:.0f}s)")

    df = pd.DataFrame(rows)
    out = Path("outputs/studies/oos_robustness")
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "seeds.csv", index=False)
    gaps = df["gap_tuned_minus_fixed"]
    print(f"\n=== Gap (tuned - fixed) sobre {len(SEEDS)} semillas, N={N_SAMPLES} ===")
    print(df.to_string(index=False))
    print(f"\n  media {gaps.mean():+.3f} | std {gaps.std():.3f} | "
          f"min {gaps.min():+.3f} | max {gaps.max():+.3f} | "
          f"semillas con tuned>fixed: {(gaps>0).sum()}/{len(SEEDS)}")
    verdict = ("RUIDO (gap straddlea 0)" if gaps.min() < 0 < gaps.max()
               else "tuned consistentemente por delante" if (gaps > 0).all()
               else "fijos consistentemente por delante")
    print(f"  veredicto: {verdict}")
    print(f"\nGuardado en {out}/seeds.csv | tiempo {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
