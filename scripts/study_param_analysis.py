"""
Analisis de parametros sobre la ventana canonica (2013-2026). Dos vistas que comparten
la misma definicion de parametros y la misma carga de datos:

  - Importancia GLOBAL: muestrea todos los parametros a la vez (N backtests), ajusta un
    random forest y reparte la varianza del Sharpe (rf_importance + spearman). Capta
    interacciones; dice cual domina cuando todo se mueve junto.
  - Sensibilidad LOCAL (one-at-a-time): mueve cada parametro solo con el resto fijo en la
    config adoptada y registra la curva del Sharpe (span). Dice cuanto importa aislado.

Comparten los 33 escalares/enteros + 2 categoricos. Los 5 pesos de factor (w_*) solo
entran en la importancia: el RF los muestrea como simplex (Dirichlet) y el OAT no puede
barrer un simplex moviendo uno solo. ML off (rapido y determinista).

Tanto la importancia RF como el span del OAT crecen con el ancho del rango de cada
parametro: no son comparables entre params con rangos de distinto ancho (el spearman si
es invariante al rango). Los rangos buscan un espacio de diseno razonable, no maximo.

Uso: python scripts/study_param_analysis.py [start] [end] [N] [n_points]
Salida: outputs/studies/param_analysis/{importance.csv, importance_samples.csv,
        sensitivity_summary.csv, sensitivity_curves.csv, sensitivity.png}
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor

import src.config as cfg
import src.data.data_loader as dl
import src.backtest.engine as engine
import src.models.black_litterman as bl
from src.backtest.engine import run_backtest
from param_nucleus import PARAM_NUCLEUS

START = sys.argv[1] if len(sys.argv) > 1 else cfg.CANONICAL_START
END = sys.argv[2] if len(sys.argv) > 2 else cfg.CANONICAL_END
N = int(sys.argv[3]) if len(sys.argv) > 3 else 400          # muestras para la importancia
N_POINTS = int(sys.argv[4]) if len(sys.argv) > 4 else 9     # puntos por curva del OAT
SEED = 11
LOOKBACK_START = "2010-07-01"

# Pesos de factor: solo importancia (simplex, el OAT no los puede barrer aislados).
FACTORS = ["momentum", "reversal", "trend", "vol_penalty", "drawdown_buy"]

# El analisis barre un superconjunto del nucleo de optimizacion: reutiliza sus rangos
# (fuente unica, param_nucleus) y anade los params que no se optimizan —inertes, de
# regimen, fijados por diseno—. MERTON_N_TOP se ensancha vs el nucleo (12,35) para ver su
# caida completa (N_TOP bajo hunde el Sharpe; ahi no se optimiza). Rangos de regimen
# crisis/caution disjuntos; ventanas de cov separadas (short<long). (lo, hi, kind).
PARAMS = {
    **PARAM_NUCLEUS,
    "MERTON_N_TOP": (5, 40, "int"),
    "MERTON_MAX_WEIGHT": (0.20, 0.60, "float"),
    "BL_TAU": (0.02, 0.10, "float"),
    "BL_RF": (0.0, 0.045, "float"),
    "EWMA_LAMBDA": (0.85, 0.98, "float"),
    "COV_LONG_WINDOW": (150, 378, "int"),
    "COMPOSITE_VOL_WINDOW": (21, 126, "int"),
    "COMPOSITE_DRAWDOWN_WINDOW": (126, 378, "int"),
    "DN_BAND": (0.02, 0.09, "float"),
    "DN_MIN_BAND": (0.0, 0.04, "float"),
    "DN_CRISIS_MULT": (1.0, 3.0, "float"),
    "DN_CAUTION_MULT": (1.0, 2.0, "float"),
    "VOL_CAUTION_THR": (0.18, 0.30, "float"),
    "VOL_CRISIS_THR": (0.32, 0.55, "float"),
    "CRISIS_VIEW_BOOST": (1.0, 1.6, "float"),
    "CAUTION_VIEW_BOOST": (1.0, 1.3, "float"),
    "REGIME_EW_MDD_CRISIS": (0.24, 0.42, "float"),
    "REGIME_EW_MDD_CAUTION": (0.10, 0.22, "float"),
    "REGIME_AVG_CORR_CRISIS": (0.46, 0.65, "float"),
    "REGIME_AVG_CORR_CAUTION": (0.28, 0.44, "float"),
    "REGIME_LOOKBACK_DD": (126, 378, "int"),
    "REGIME_CORR_LOOKBACK": (21, 126, "int"),
}
CATEGORICALS = {
    "REBALANCE_FREQ": ["2W", "ME", "QE"],
    "BL_PRIOR_WEIGHTS_MODE": ["equal", "inv_vol"],
}
FREQ_DAYS = {"2W": 10, "ME": 21, "QE": 63}
PRIOR_MODES = {"equal": 0, "inv_vol": 1}

SPY = cfg.BENCHMARK_TICKER
URTH = getattr(cfg, "BENCHMARK_MSCI_WORLD_TICKER", "URTH")
OUT = Path("outputs/studies/param_analysis")


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


def _sharpe() -> float:
    r = run_backtest(START, END, output_dir="/tmp/ppa", verbose=False)
    return r["strategy_metrics"]["Sharpe"]


def _metrics() -> dict:
    r = run_backtest(START, END, output_dir="/tmp/ppa", verbose=False)
    m = r["strategy_metrics"]
    return {"Sharpe": m["Sharpe"], "CAGR": m["CAGR"], "MaxDD": m["Max Drawdown"],
            "TotalReturn": m["Total Return"]}


def run_sensitivity() -> tuple[pd.DataFrame, pd.DataFrame, float]:
    """OAT con la config adoptada (CATEGORY_SIGNAL_WEIGHTS activos). No toca los w_*."""
    base = _metrics()
    rows, summary = [], []

    def sweep(name, grid, baseline_val, is_cat=False):
        original = getattr(cfg, name)
        sharpes = []
        for v in grid:
            setattr(cfg, name, v)
            m = _metrics()
            rows.append({"param": name, "value": v, "x": (grid.index(v) if is_cat else v),
                         "is_baseline": (v == baseline_val), **m})
            sharpes.append(m["Sharpe"])
        setattr(cfg, name, original)
        s = np.array(sharpes)
        summary.append({
            "param": name, "kind": "categorical" if is_cat else "scalar",
            "baseline_value": baseline_val, "baseline_Sharpe": round(base["Sharpe"], 4),
            "Sharpe_min": round(float(s.min()), 4), "Sharpe_max": round(float(s.max()), 4),
            "Sharpe_span": round(float(s.max() - s.min()), 4),
            "argmax_value": grid[int(s.argmax())], "n_points": len(grid),
        })

    for name, (lo, hi, kind) in PARAMS.items():
        b = getattr(cfg, name)
        pts = list(np.linspace(lo, hi, N_POINTS)) + [float(b)]
        grid = (sorted({int(round(p)) for p in pts}) if kind == "int"
                else sorted({round(float(p), 6) for p in pts}))
        # baseline redondeado igual que la rejilla, para que is_baseline lo case
        sweep(name, grid, int(round(b)) if kind == "int" else round(float(b), 6))
    for name, vals in CATEGORICALS.items():
        sweep(name, list(vals), getattr(cfg, name), is_cat=True)

    curves = pd.DataFrame(rows)
    summ = pd.DataFrame(summary).sort_values("Sharpe_span", ascending=False).reset_index(drop=True)
    return summ, curves, base["Sharpe"]


def _plot_sensitivity(curves, summ, base_sharpe, path):
    params = list(summ["param"])
    cols = 5
    rows_n = (len(params) + cols - 1) // cols
    fig, axes = plt.subplots(rows_n, cols, figsize=(cols * 3.0, rows_n * 2.3))
    axes = np.array(axes).reshape(-1)
    for ax, p in zip(axes, params):
        d = curves[curves["param"] == p].sort_values("x")
        ax.plot(d["x"], d["Sharpe"], "-o", ms=3, lw=1.2, color="#1f3b73")
        ax.axhline(base_sharpe, color="grey", ls=":", lw=0.8)
        b = d[d["is_baseline"]]
        if not b.empty:
            ax.plot(b["x"], b["Sharpe"], "o", ms=7, mfc="none", mec="crimson", mew=1.5)
        span = float(summ[summ["param"] == p]["Sharpe_span"].iloc[0])
        ax.set_title(f"{p}\nspan={span:.3f}", fontsize=7)
        ax.tick_params(labelsize=6)
    for ax in axes[len(params):]:
        ax.axis("off")
    fig.suptitle(f"Sensibilidad OAT del Sharpe (base={base_sharpe:.3f}, "
                 f"circulo rojo = valor adoptado)", fontsize=10)
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    fig.savefig(path, dpi=130)
    plt.close(fig)


def run_importance() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Muestreo conjunto + RF. Usa los pesos globales (CATEGORY_SIGNAL_WEIGHTS=None) para
    poder medir los 5 w_* como simplex. Restaura la config al salir (no depende del orden)."""
    touched = list(PARAMS) + ["COMPOSITE_WEIGHTS", "CATEGORY_SIGNAL_WEIGHTS",
                              "REBALANCE_FREQ", "BL_PRIOR_WEIGHTS_MODE"]
    snapshot = {k: getattr(cfg, k) for k in touched}
    rows = []
    try:
        cfg.CATEGORY_SIGNAL_WEIGHTS = None
        rng = np.random.default_rng(SEED)
        for i in range(N):
            feat = {}
            w = rng.dirichlet(np.ones(len(FACTORS)))
            cfg.COMPOSITE_WEIGHTS = {f: float(w[j]) for j, f in enumerate(FACTORS)}
            for j, f in enumerate(FACTORS):
                feat[f"w_{f}"] = float(w[j])
            for k, (lo, hi, kind) in PARAMS.items():
                v = int(rng.integers(lo, hi + 1)) if kind == "int" else float(rng.uniform(lo, hi))
                setattr(cfg, k, v)
                feat[k] = v
            fq = str(rng.choice(CATEGORICALS["REBALANCE_FREQ"]))
            cfg.REBALANCE_FREQ = fq
            feat["REBALANCE_FREQ"] = FREQ_DAYS[fq]
            pm = str(rng.choice(CATEGORICALS["BL_PRIOR_WEIGHTS_MODE"]))
            cfg.BL_PRIOR_WEIGHTS_MODE = pm
            feat["BL_PRIOR_WEIGHTS_MODE"] = PRIOR_MODES[pm]
            feat["Sharpe"] = _sharpe()
            rows.append(feat)
            if (i + 1) % 100 == 0:
                print(f"  importancia {i + 1}/{N}")
    finally:
        for k, v in snapshot.items():
            setattr(cfg, k, v)

    df = pd.DataFrame(rows)
    X, y = df.drop(columns=["Sharpe"]), df["Sharpe"]
    rf = RandomForestRegressor(n_estimators=400, random_state=SEED, n_jobs=-1).fit(X, y)
    imp = pd.DataFrame({
        "param": X.columns,
        "rf_importance": rf.feature_importances_.round(4),
        "spearman": [round(X[c].corr(y, method="spearman"), 3) for c in X.columns],
    }).sort_values("rf_importance", ascending=False).reset_index(drop=True)
    return imp, df


def main():
    print(f"Analisis de parametros | {START}->{END} | OAT {N_POINTS}pts + importancia N={N} | "
          f"seed={SEED} | ML off")
    t0 = time.time()
    fake = _make_fake(_fixed())
    dl.download_market_data = fake
    engine.download_market_data = fake
    bl.log_decisions = lambda *a, **k: None
    cfg.USE_ML_FILTER = False
    cfg.USE_CLUSTERING = False
    OUT.mkdir(parents=True, exist_ok=True)

    # OAT primero: necesita la config adoptada pristina (la importancia la muta).
    summ, curves, base_sharpe = run_sensitivity()
    curves.to_csv(OUT / "sensitivity_curves.csv", index=False)
    summ.to_csv(OUT / "sensitivity_summary.csv", index=False)
    _plot_sensitivity(curves, summ, base_sharpe, OUT / "sensitivity.png")
    print(f"OAT listo ({time.time()-t0:.0f}s). Base Sharpe={base_sharpe:.3f}. "
          f"Top span: {', '.join(summ['param'].head(3))}")

    imp, samples = run_importance()
    imp.to_csv(OUT / "importance.csv", index=False)
    samples.to_csv(OUT / "importance_samples.csv", index=False)

    print(f"\n=== Sensibilidad OAT (span del Sharpe, {START}->{END}) ===")
    print(summ.to_string(index=False))
    print(f"\n=== Importancia global (RF sobre el Sharpe, {START}->{END}) ===")
    print(imp.to_string(index=False))
    print(f"\nGuardado en {OUT}/ | tiempo {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
