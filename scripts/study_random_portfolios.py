"""
"Monos con dardos" (Malkiel): distribucion nula de SELECCION, no de parametros.
En vez de sortear los hiperparametros del nucleo, sortea N carteras al azar dentro
del mismo universo de ETFs y las pasa por el MISMO motor de backtest (mismas fechas
fin de mes, mismo modelo de costes, mismo Sharpe, mismos benchmarks). Si una cesta
de dardos bate a la estrategia, la maquinaria (señal/BL/Merton/DN) no aporta seleccion.

Dos variantes de dardo:
  - buyhold : sortea una cartera al inicio y la mantiene (test clasico de Malkiel,
              coste ~0; aisla pura seleccion).
  - monthly : re-sortea pesos cada fin de mes y rebalancea pagando turnover (mide si
              rebalancear sin criterio destruye valor).

Cada cartera tiene K ETFs, con K = amplitud media de la estrategia en ese periodo
(numero medio de ETFs seleccionados), pesos Dirichlet(1), 100% invertida en riesgo
(sin XEON: un dardo no hace market-timing defensivo). Dos periodos (2020-2026, 2013-2026).

Uso: python scripts/study_random_portfolios.py [N] [buyhold|monthly|both]   # default 500 both
Salida: outputs/studies/random_portfolios/random_<mode>_<periodo>.csv (+ wealth, chosen)
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
import src.pipeline as pipeline_mod
import src.models.black_litterman as bl
from src.backtest.engine import run_backtest

SEED = 123
N = int(sys.argv[1]) if len(sys.argv) > 1 else 500
_MODE_ARG = sys.argv[2].lower() if len(sys.argv) > 2 else "both"
MODES = ["buyhold", "monthly"] if _MODE_ARG == "both" else [_MODE_ARG]

LOOKBACK_START = "2010-07-01"   # cubre el lookback de 18m del periodo mas largo
PERIODS = [
    ("2020-2026", cfg.SUBPERIOD_START, cfg.BACKTEST_DATA_END),
    ("2013-2026", cfg.CANONICAL_START, cfg.BACKTEST_DATA_END),
]

# ML/clustering off: solo afectarian a los 2 runs 'chosen' (XGBoost por fecha de revision,
# lentisimo en la ventana canonica) y los dejaria inconsistentes con el panel de monos, que
# tambien corre con ML off. Los dardos no usan el pipeline, asi que les da igual.
cfg.USE_ML_FILTER = False
cfg.USE_CLUSTERING = False
bl.log_decisions = lambda *a, **k: None

SPY = cfg.BENCHMARK_TICKER
URTH = getattr(cfg, "BENCHMARK_MSCI_WORLD_TICKER", "URTH")

REAL_PIPELINE = pipeline_mod.run_pipeline
OUT_DIR = Path("outputs/studies/random_portfolios")


def _build_fixed_data():
    real = dl.download_market_data
    return {
        "uni": real(start_date=LOOKBACK_START, end_date=cfg.BACKTEST_DATA_END),
        "spy": real(start_date=LOOKBACK_START, end_date=cfg.BACKTEST_DATA_END, tickers=[SPY]),
        "urth": real(start_date=LOOKBACK_START, end_date=cfg.BACKTEST_DATA_END, tickers=[URTH]),
    }


def _make_fake_download(fixed):
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


def _make_random_pipeline(rng: np.random.Generator, k: int, mode: str):
    """Pipeline falso: devuelve una cartera aleatoria con la forma que espera el motor.

    buyhold -> sortea una vez y la mantiene (rebalance=False tras el primer despliegue).
    monthly -> re-sortea cada fecha de revision (rebalance=True siempre).
    """
    state = {"weights": None}

    def fake(returns_slice, prices_slice, current_weights=None, review_date=None,
             categoria_por_ticker=None, **kwargs):
        if mode == "buyhold" and state["weights"] is not None:
            w = state["weights"]
            return _result(w, rebalance=False)

        # Solo ETFs con precio valido en la fecha (respeta no-look-ahead: nada que aun no cotiza)
        avail = [c for c in prices_slice.columns if pd.notna(prices_slice[c].iloc[-1])]
        if not avail:
            return _result({}, rebalance=False)
        # Suelo: en fechas tempranas del periodo largo cotizan menos de K ETFs; el dardo
        # se queda con los disponibles (no se puede comprar lo que aun no existe). En buyhold,
        # que sortea una sola vez en la 1a fecha, esto puede fijar K por debajo de la amplitud
        # media de la estrategia para todo el run; en monthly se corrige al re-sortear.
        k_eff = min(k, len(avail))
        picks = rng.choice(len(avail), size=k_eff, replace=False)
        names = [avail[j] for j in picks]
        weights = rng.dirichlet(np.ones(k_eff))
        w = {names[j]: float(weights[j]) for j in range(k_eff)}
        if mode == "buyhold":
            state["weights"] = w
        return _result(w, rebalance=True)

    return fake


def _result(weights_full: dict, rebalance: bool) -> dict:
    return {
        "dn_result": {
            "final_weights_full": weights_full,
            "rebalance": rebalance,
            "weight_xeon": 0.0,
            "reason": "random portfolio",
        },
        "merton_result": {
            "selected_etfs": list(weights_full),
            "regime": "normal",
        },
    }


def _metrics(r):
    nav = r["wealth"]["Strategy"].resample("ME").last()
    return r["strategy_metrics"]["Sharpe"], r["strategy_metrics"]["CAGR"], r["sp500_metrics"]["Sharpe"], nav


def main():
    print(f"Dardos (Malkiel) | N={N} carteras/periodo | modos={MODES} | ML off")
    t0 = time.time()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    fixed = _build_fixed_data()
    fake_dl = _make_fake_download(fixed)
    dl.download_market_data = fake_dl
    engine.download_market_data = fake_dl

    # 1) Estrategia real (config adoptada, ML/clustering off) por periodo -> 'chosen' y amplitud K
    pipeline_mod.run_pipeline = REAL_PIPELINE
    chosen = {}
    for name, start, end in PERIODS:
        r = run_backtest(start, end, output_dir=f"/tmp/random_portfolios/{name}_chosen", verbose=False)
        sh, cagr, sh_spy, nav = _metrics(r)
        k = int(round(float(r["decisions"]["N ETFs"].mean())))
        chosen[name] = {"sh": sh, "cagr": cagr, "sh_spy": sh_spy, "nav": nav, "k": k}
        print(f"  [chosen] {name}: Sharpe {sh:.3f} | CAGR {cagr*100:.2f}% | K(amplitud media)={k}")

    # 2) Dardos por modo
    for mode in MODES:
        chosen_rows = []
        for p_idx, (name, start, end) in enumerate(PERIODS):
            ch = chosen[name]
            k = ch["k"]
            rows, curves = [], {}
            for i in range(N):
                # Semilla independiente por dardo: streams no acoplados (el dardo i no
                # depende de cuantos sorteos consumio i-1) y el dardo i es IDENTICO en
                # buyhold y monthly -> el contraste entre modos aisla solo el rebalanceo.
                dart_rng = np.random.default_rng([SEED, p_idx, i])
                pipeline_mod.run_pipeline = _make_random_pipeline(dart_rng, k, mode)
                r = run_backtest(start, end, output_dir=f"/tmp/random_portfolios/{mode}_{name}",
                                 verbose=False)
                sh, cagr, _, nav = _metrics(r)
                rows.append({"Sharpe": sh, "CAGR": cagr, "K": k})
                curves[f"m{i}"] = nav

            df = pd.DataFrame(rows)
            wealth = pd.DataFrame(curves)
            wealth["chosen"] = ch["nav"]
            wealth.index.name = "Date"
            wealth.to_csv(OUT_DIR / f"random_wealth_{mode}_{name}.csv")
            df.to_csv(OUT_DIR / f"random_{mode}_{name}.csv", index=False)

            sh_arr = df["Sharpe"].to_numpy()
            pct_chosen = float((sh_arr < ch["sh"]).mean() * 100)
            pct_beat_chosen = float((sh_arr >= ch["sh"]).mean() * 100)
            pct_beat_spy = float((sh_arr >= ch["sh_spy"]).mean() * 100)

            print(f"\n=== [{mode}] {name} ({start} -> {end}) | K={k} ===")
            print(f"  ESTRATEGIA: Sharpe {ch['sh']:.3f} | CAGR {ch['cagr']*100:.2f}%")
            print(f"  SP500:      Sharpe {ch['sh_spy']:.3f}")
            print(f"  DARDOS (N={N}): Sharpe  p10={np.percentile(sh_arr,10):.3f}  "
                  f"mediana={np.median(sh_arr):.3f}  p90={np.percentile(sh_arr,90):.3f}  max={sh_arr.max():.3f}")
            print(f"  -> percentil de la ESTRATEGIA en la nube: {pct_chosen:.0f}%")
            print(f"  -> % de dardos que la BATEN:        {pct_beat_chosen:.0f}%")
            print(f"  -> % de dardos que BATEN al SP500:  {pct_beat_spy:.0f}%")

            chosen_rows.append({
                "periodo": name, "chosen_Sharpe": ch["sh"], "chosen_CAGR": ch["cagr"],
                "SP500_Sharpe": ch["sh_spy"], "percentil": pct_chosen,
                "pct_baten_chosen": pct_beat_chosen, "K": k,
            })

        pd.DataFrame(chosen_rows).to_csv(OUT_DIR / f"random_chosen_{mode}.csv", index=False)

    pipeline_mod.run_pipeline = REAL_PIPELINE
    print(f"\nGuardado en {OUT_DIR}/ | tiempo {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
