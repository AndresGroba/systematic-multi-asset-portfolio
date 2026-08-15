"""
Refresca la tasa libre de riesgo (facilidad de deposito del BCE) desde la API oficial
(sin API key) a un snapshot offline commiteable. Re-ejecutar cuando el BCE mueva tipos.

Serie FM.B.U2.EUR.4F.KR.DFR.LEV: una fila por cambio (fecha efectiva), guardada en fraccion.

Uso: python scripts/fetch_ecb_rates.py
Salida: src/data/ecb_rates.csv
"""

from __future__ import annotations

import csv
import sys
import urllib.request
from pathlib import Path

ECB_URL = (
    "https://data-api.ecb.europa.eu/service/data/FM/"
    "B.U2.EUR.4F.KR.DFR.LEV?format=csvdata"
)
OUT_PATH = Path(__file__).resolve().parent.parent / "src" / "data" / "ecb_rates.csv"


def fetch_rows() -> list[tuple[str, float]]:
    req = urllib.request.Request(ECB_URL, headers={"User-Agent": "python"})
    text = urllib.request.urlopen(req, timeout=60).read().decode()
    lines = text.splitlines()
    header = lines[0].split(",")
    ti, oi = header.index("TIME_PERIOD"), header.index("OBS_VALUE")
    rows: list[tuple[str, float]] = []
    for line in lines[1:]:
        c = line.split(",")
        if len(c) <= max(ti, oi) or not c[oi].strip():
            continue
        rows.append((c[ti].strip(), float(c[oi]) / 100.0))  # % -> fraccion
    return sorted(rows)


def main() -> None:
    rows = fetch_rows()
    if not rows:
        print("ERROR: la API del BCE no devolvio observaciones.")
        sys.exit(1)
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["date", "rate"])
        w.writerows(rows)
    print(f"Guardado {len(rows)} cambios de tipo en {OUT_PATH}")
    print(f"  rango: {rows[0][0]} ({rows[0][1]:.4%}) -> {rows[-1][0]} ({rows[-1][1]:.4%})")


if __name__ == "__main__":
    main()
