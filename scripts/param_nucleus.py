"""Nucleo de parametros optimizables, compartido por study_params_oos (OOS walk-forward)
y study_params_vs_random (monos) para que ambos no se desincronicen.

Dentro van los que mueven el Sharpe en el barrido OAT (study_param_analysis):
MERTON_N_TOP, ventanas de momentum, COV_* y VIEW_SCALE, mas gamma/BL_DELTA/MAX_SECTOR.
Fuera: REBALANCE_FREQ (fijado por diseno), BL_TAU (se cancela en el posterior BL) y
DN_BAND (las bandas no atan en backtest mensual). Cada entrada: (lo, hi, kind).
"""

from __future__ import annotations

PARAM_NUCLEUS: dict[str, tuple[float, float, str]] = {
    # Drivers fuertes
    "MERTON_N_TOP":              (12, 35, "int"),
    "COMPOSITE_MOMENTUM_WINDOW": (63, 190, "int"),
    "COMPOSITE_MOMENTUM_SKIP":   (0, 52, "int"),
    # Drivers medios
    "COMPOSITE_REVERSAL_WINDOW": (5, 42, "int"),
    "COMPOSITE_TREND_WINDOW":    (100, 252, "int"),
    "COV_SHORT_WINDOW":          (21, 126, "int"),
    "VIEW_SCALE":                (0.12, 0.55, "float"),
    "COV_BLEND_ALPHA":           (0.30, 0.90, "float"),
    "COV_SHRINKAGE":             (0.05, 0.40, "float"),
    # Interpretables (impacto debil pero centrales en la tesis)
    "MERTON_GAMMA":              (-1.5, -0.4, "float"),
    "BL_DELTA":                  (1.5, 3.5, "float"),
    "MERTON_MAX_SECTOR":         (0.25, 0.50, "float"),
}


def names() -> list[str]:
    return list(PARAM_NUCLEUS)


def sample(rng) -> dict[str, float | int]:
    """Un juego de parametros muestreado uniforme; los enteros se redondean al entero."""
    out: dict[str, float | int] = {}
    for k, (lo, hi, kind) in PARAM_NUCLEUS.items():
        if kind == "int":
            out[k] = int(rng.integers(int(lo), int(hi) + 1))
        else:
            out[k] = float(rng.uniform(lo, hi))
    return out
