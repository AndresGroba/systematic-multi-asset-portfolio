"""
Backtest completo con graficos sobre tres ventanas (canonica, sub-periodo,
contrafactual en vivo), todas hasta cfg.BACKTEST_DATA_END.

Uso: python scripts/run_backtest.py
Salida: outputs/backtest/{canonical,subperiod,counterfactual}/
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import src.config as cfg
from src.backtest.engine import run_backtest
from src.visualization.plots import generate_all_plots


WINDOWS = [
    ("CANONICA 2012-2026", cfg.CANONICAL_START, cfg.CANONICAL_END, "outputs/backtest/canonical"),
    ("SUB-PERIODO 2020-2026", cfg.SUBPERIOD_START, cfg.SUBPERIOD_END, "outputs/backtest/subperiod"),
    ("CONTRAFACTUAL (periodo en vivo)", cfg.LIVE_START, cfg.LIVE_END, "outputs/backtest/counterfactual"),
]


if __name__ == "__main__":
    for label, start_date, end_date, output_dir in WINDOWS:
        print(f"\n{'=' * 60}\n{label}\n{'=' * 60}")
        result = run_backtest(
            start_date=start_date,
            end_date=end_date,
            output_dir=output_dir,
        )
        generate_all_plots(output_dir=result["output_dir"])
        print(f"Resultados en: {result['output_dir']}")
    print("\nBacktest completado (canonica + sub-periodo + contrafactual).")
