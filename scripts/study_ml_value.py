"""
Ablacion del refinamiento ML: backtest con USE_ML_FILTER / USE_CLUSTERING on/off
para aislar la contribucion de cada pieza.

La variante 'ML risk overlay' no se reproduce: recorta exposicion en drawdown,
contradice la filosofia contrarian (experimento descartado).

Uso: python scripts/study_ml_value.py
Salida: outputs/studies/ml_value/ablation_eur.csv
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

import src.config as cfg
from src.backtest.engine import run_backtest

START, END = cfg.CANONICAL_START, cfg.CANONICAL_END

VARIANTS = [
    ("Original (sin ML)", False, False),
    ("ML clustering", False, True),
    ("XGB + clustering (final)", True, True),
]


def run_variant(name: str, use_ml: bool, use_cluster: bool) -> dict:
    cfg.USE_ML_FILTER = use_ml
    cfg.USE_CLUSTERING = use_cluster
    out = Path("/tmp/ablation_eur") / name.replace(" ", "_").replace("(", "").replace(")", "")
    r = run_backtest(start_date=START, end_date=END, output_dir=str(out), verbose=False)
    m = r["strategy_metrics"]
    cost_pct = r["total_transaction_costs"] / float(cfg.INITIAL_CAPITAL) * 100.0
    return {
        "Variante": name,
        "CAGR": f"{m['CAGR']:.2%}",
        "Sharpe": f"{m['Sharpe']:.2f}",
        "Max DD": f"{m['Max Drawdown']:.2%}",
        "Total Return": f"{m['Total Return']:.2%}",
        "Costes (% capital)": f"{cost_pct:.2f}%",
        "Rebalances": r["n_rebalances"],
    }


if __name__ == "__main__":
    rows = []
    for name, use_ml, use_cluster in VARIANTS:
        print(f"[Ablacion] Ejecutando: {name} (ML={use_ml}, clustering={use_cluster}) ...")
        rows.append(run_variant(name, use_ml, use_cluster))

    table = pd.DataFrame(rows)
    out_csv = Path("outputs/studies/ml_value/ablation_eur.csv")
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(out_csv, index=False)
    print("\n=== ABLACION EUR ===")
    print(table.to_string(index=False))
    print(f"\nGuardado en {out_csv}")
