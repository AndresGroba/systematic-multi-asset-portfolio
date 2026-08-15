"""
Walkforward out-of-sample.

Uso: python scripts/run_walkforward.py
Salida: outputs/walkforward/
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.walkforward.walkforward import run_walkforward


if __name__ == "__main__":
    result = run_walkforward(
        start_date="2020-01-01",
        end_date="2025-12-31",
        output_dir="outputs/walkforward",
    )
    print("\nWalkforward completado.")
    print(f"Resultados en: {result['output_dir']}")
