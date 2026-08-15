"""
Informe de la cartera viva: reconstruye posiciones, P&L, costes y comparacion
con benchmarks acumulando los Excel de operaciones del registrador.

Uso: python scripts/track_live_portfolio.py
Salida: outputs/live/informe_cartera_vivo/ (Excel, CSVs, grafico PNG)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.portfolio.live_portfolio_report import run_live_portfolio_report


def main() -> None:
    parser = argparse.ArgumentParser(description="Informe cartera viva desde operaciones_rebalanceo_*.xlsx")
    parser.add_argument(
        "--results-dir",
        default=None,
        help="Carpeta con los Excel de operaciones (default: config PORTFOLIO_LIVE_RESULTS_DIR)",
    )
    parser.add_argument("--start", default=None, help="Fecha inicio YYYY-MM-DD (opcional)")
    parser.add_argument("--end", default=None, help="Fecha fin YYYY-MM-DD (default: hoy)")
    parser.add_argument(
        "--initial-positions",
        default=None,
        help="Excel con posiciones antes del primer archivo de operaciones (opcional)",
    )
    args = parser.parse_args()

    r = run_live_portfolio_report(
        results_dir=args.results_dir,
        start_date=args.start,
        end_date=args.end,
        initial_positions_path=args.initial_positions,
    )

    print("\n=== Informe cartera viva ===\n")
    print(f"Directorio informe: {r['output_dir']}")
    print(f"Excel principal:    {r['excel_path']}")
    print(f"Archivos operaciones leidos: {r['operaciones_files']}")
    print(f"Coste comisiones estimado (acum.): {r['coste_total_estimado_eur']:,.2f} EUR")
    print(f"Notional acumulado (|orden|):     {r['notional_acumulado_eur']:,.2f} EUR")
    m = r["metrics_formatted"]
    print("\nMetricas cartera (serie patrimonio EUR):")
    for k, v in m.items():
        print(f"  {k}: {v}")
    if r.get("plot_path"):
        print(f"\nGrafico: {r['plot_path']}")
    print("\nListo. Conserva la carpeta de informes y el CSV historial_ejecuciones para auditoria.\n")


if __name__ == "__main__":
    main()
