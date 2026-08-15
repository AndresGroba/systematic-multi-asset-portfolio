"""
IC del 95% del Sharpe por block bootstrap. Un Sharpe puntual no dice si la ventaja es
estadisticamente distinguible del ruido; el bootstrap por bloques (preserva autocorrelacion)
da una banda. Lee el wealth del backtest canonico ya generado (run_backtest), en EUR.

Uso: python scripts/bootstrap_sharpe.py [n_boot] [block_days]   # default 5000, 21
Salida: outputs/studies/bootstrap/sharpe_ci.csv
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

N_BOOT = int(sys.argv[1]) if len(sys.argv) > 1 else 5000
BLOCK = int(sys.argv[2]) if len(sys.argv) > 2 else 21   # ~1 mes bursatil
SEED = 20240601
WEALTH = "outputs/backtest/canonical/wealth_history.csv"


def _ann_sharpe(daily: np.ndarray) -> float:
    if daily.size < 2:
        return float("nan")
    vol = daily.std(ddof=1) * np.sqrt(252)
    return float(daily.mean() * 252 / vol) if vol > 0 else float("nan")


def _block_bootstrap(daily: np.ndarray, rng) -> float:
    n = daily.size
    n_blocks = int(np.ceil(n / BLOCK))
    starts = rng.integers(0, n - BLOCK + 1, size=n_blocks)
    sample = np.concatenate([daily[s:s + BLOCK] for s in starts])[:n]
    return _ann_sharpe(sample)


def main():
    p = Path(WEALTH)
    if not p.is_file():
        print(f"No existe {WEALTH} (corre run_backtest antes). Abortando.")
        return
    w = pd.read_csv(p, parse_dates=["Date"]).set_index("Date")
    rng = np.random.default_rng(SEED)

    cols = [c for c in ("Strategy", "SP500", "MSCI_World (URTH)") if c in w.columns]
    rows = []
    for c in cols:
        daily = w[c].pct_change().dropna().to_numpy()
        point = _ann_sharpe(daily)
        boot = np.array([_block_bootstrap(daily, rng) for _ in range(N_BOOT)])
        boot = boot[np.isfinite(boot)]
        lo, hi = np.percentile(boot, [2.5, 97.5])
        rows.append({"serie": c, "Sharpe": round(point, 3),
                     "IC95_low": round(float(lo), 3), "IC95_high": round(float(hi), 3),
                     "p(Sharpe>0)": round(float((boot > 0).mean()), 3)})

    df = pd.DataFrame(rows)
    out = Path("outputs/studies/bootstrap")
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "sharpe_ci.csv", index=False)
    print(f"=== IC 95% del Sharpe (block bootstrap, n={N_BOOT}, bloque={BLOCK}d, canónica) ===")
    print(df.to_string(index=False))
    print(f"\nGuardado en {out}/sharpe_ci.csv")


if __name__ == "__main__":
    main()
