"""
Capa de datos de la ESTRATEGIA para el dashboard (desacoplada).

Lee los artefactos generados por los scripts del paquete raíz desde `outputs/`
(backtest de las 3 ventanas + los 5 studies + walkforward del ML). Todos esos
CSV/Excel ya están en EUR (la conversión la hace `src/data/data_loader.py` del
paquete raíz), así que aquí no se descarga ni se convierte nada: el dashboard
solo lee resultados → EUR por construcción.

Diseño robusto: si un artefacto no existe (script no corrido / carpeta vacía),
el loader devuelve None y la UI muestra "pendiente de generar" en vez de romper.
No importa del paquete raíz (regla de desacople).
"""

from __future__ import annotations

import os
from pathlib import Path

import pandas as pd


# ── Resolución de la carpeta outputs/ del paquete raíz ───────────────────────
# dashboard/src/strategy_data.py -> parents[2] = raíz del repo
_DEFAULT_OUTPUTS = Path(__file__).resolve().parents[2] / "outputs"


def outputs_dir() -> Path:
    """Carpeta outputs/ del paquete raíz (override con env STRATEGY_OUTPUTS_DIR)."""
    return Path(os.environ.get("STRATEGY_OUTPUTS_DIR", _DEFAULT_OUTPUTS))


# ── Ventanas del backtest ────────────────────────────────────────────────────
# clave de carpeta -> (etiqueta corta, descripción)
BACKTEST_WINDOWS: dict[str, tuple[str, str]] = {
    "canonical": ("Canónico", "2013-08 → 2026-05 (~12.75 años, multi-régimen; warmup desde feb-2012)"),
    "subperiod": ("Sub-periodo", "2020-2026 (post-COVID)"),
    "counterfactual": ("Contrafactual", "Periodo en vivo (~2 meses): estrategia vs cartera real"),
}


def _read_csv(path: Path, **kw) -> pd.DataFrame | None:
    if not path.is_file():
        return None
    try:
        return pd.read_csv(path, **kw)
    except Exception:
        return None


def _read_excel(path: Path, sheet_name=0) -> pd.DataFrame | None:
    if not path.is_file():
        return None
    try:
        return pd.read_excel(path, sheet_name=sheet_name)
    except Exception:
        return None


# ── Backtest (por ventana) ───────────────────────────────────────────────────

def _bt_dir(window: str) -> Path:
    return outputs_dir() / "backtest" / window


def window_available(window: str) -> bool:
    """Una ventana está disponible si tiene wealth_history.csv."""
    return (_bt_dir(window) / "wealth_history.csv").is_file()


def available_windows() -> list[str]:
    return [w for w in BACKTEST_WINDOWS if window_available(w)]


def window_label(window: str) -> str:
    short, _ = BACKTEST_WINDOWS.get(window, (window, ""))
    rng = window_date_range(window)
    return f"{short} {rng}" if rng else short


def window_date_range(window: str) -> str:
    """Rango real de fechas leído del wealth_history (se autoajusta al recomputar)."""
    w = load_wealth(window)
    if w is None or w.empty:
        return ""
    return f"{w.index[0]:%Y-%m} → {w.index[-1]:%Y-%m}"


def load_metrics(window: str) -> pd.DataFrame | None:
    """Tabla comparativa (estrategia vs SP500 vs MSCI). Index = métrica."""
    df = _read_excel(_bt_dir(window) / "Metrics.xlsx", sheet_name="Comparison")
    if df is None:
        return None
    return df.set_index(df.columns[0])


def load_wealth(window: str) -> pd.DataFrame | None:
    df = _read_csv(_bt_dir(window) / "wealth_history.csv", parse_dates=["Date"])
    if df is None:
        return None
    return df.set_index("Date")


def load_weights(window: str) -> pd.DataFrame | None:
    df = _read_csv(_bt_dir(window) / "weights_history.csv", parse_dates=["Date"])
    if df is None:
        return None
    return df.set_index("Date")


def load_attribution_cumulative(window: str) -> pd.DataFrame | None:
    # BOM en la primera columna -> utf-8-sig
    df = _read_csv(_bt_dir(window) / "attribution_cumulative_eur.csv",
                   parse_dates=["Date"], encoding="utf-8-sig")
    if df is None:
        return None
    return df.set_index("Date")


def load_resumen(window: str) -> pd.DataFrame | None:
    return _read_csv(_bt_dir(window) / "backtest_resumen.csv", parse_dates=["Date"])


# ── Studies ──────────────────────────────────────────────────────────────────

def _studies_dir() -> Path:
    return outputs_dir() / "studies"


def load_ablation() -> pd.DataFrame | None:
    return _read_csv(_studies_dir() / "ml_value" / "ablation_eur.csv")


def load_monkeys() -> dict[str, pd.DataFrame]:
    """Distribución nula por periodo: {'2013-2026': df, '2020-2026': df}."""
    d = _studies_dir() / "params_vs_random"
    out: dict[str, pd.DataFrame] = {}
    if not d.is_dir():
        return out
    for p in sorted(d.glob("monkeys_*.csv")):
        # monkeys_wealth_*.csv (curvas de valor) también casa el glob pero no es una
        # distribución de Sharpe: lo carga load_monkeys_wealth aparte.
        if p.stem.startswith("monkeys_wealth_"):
            continue
        period = p.stem.replace("monkeys_", "")
        df = _read_csv(p)
        if df is not None:
            out[period] = df
    return out


def load_monkeys_wealth(period: str) -> pd.DataFrame | None:
    """Curvas de valor (NAV fin de mes) de los monos + 'chosen', por periodo.

    Columnas m0..mN (carteras de params aleatorios) + 'chosen' (nuestra estrategia).
    Index = Date. Devuelve None si el study no se ha corrido con curvas guardadas.
    """
    df = _read_csv(_studies_dir() / "params_vs_random" / f"monkeys_wealth_{period}.csv",
                   parse_dates=["Date"])
    if df is None or "Date" not in df.columns:
        return None
    return df.set_index("Date")


def load_monkeys_chosen() -> dict[str, dict] | None:
    """Sharpe de los params elegidos por periodo (para marcar el percentil).

    {'2013-2026': {'chosen_Sharpe': .., 'percentil': .., 'SP500_Sharpe': ..}, ...}
    """
    df = _read_csv(_studies_dir() / "params_vs_random" / "chosen.csv")
    if df is None or "periodo" not in df.columns:
        return None
    return {str(r["periodo"]): r.to_dict() for _, r in df.iterrows()}


def load_random_portfolios(mode: str) -> dict[str, pd.DataFrame]:
    """Distribución nula de SELECCIÓN (dardos) por periodo, para un modo.

    mode ∈ {'buyhold', 'monthly'}. {'2013-2026': df, '2020-2026': df}.
    """
    d = _studies_dir() / "random_portfolios"
    out: dict[str, pd.DataFrame] = {}
    if not d.is_dir():
        return out
    prefix = f"random_{mode}_"
    for p in sorted(d.glob(f"{prefix}*.csv")):
        period = p.stem.replace(prefix, "")
        df = _read_csv(p)
        if df is not None:
            out[period] = df
    return out


def load_random_wealth(mode: str, period: str) -> pd.DataFrame | None:
    """Curvas de valor (NAV fin de mes) de los dardos + 'chosen', por modo y periodo."""
    df = _read_csv(_studies_dir() / "random_portfolios" / f"random_wealth_{mode}_{period}.csv",
                   parse_dates=["Date"])
    if df is None or "Date" not in df.columns:
        return None
    return df.set_index("Date")


def load_random_chosen(mode: str) -> dict[str, dict] | None:
    """Sharpe de la estrategia por periodo (para marcar el percentil), por modo."""
    df = _read_csv(_studies_dir() / "random_portfolios" / f"random_chosen_{mode}.csv")
    if df is None or "periodo" not in df.columns:
        return None
    return {str(r["periodo"]): r.to_dict() for _, r in df.iterrows()}


def load_params_oos() -> pd.DataFrame | None:
    return _read_csv(_studies_dir() / "params_oos" / "walkforward.csv")


def load_params_oos_summary() -> pd.DataFrame | None:
    return _read_csv(_studies_dir() / "params_oos" / "oos_summary.csv")


def load_freq_compare() -> pd.DataFrame | None:
    return _read_csv(_studies_dir() / "rebalance_frequency" / "freq_compare.csv")


def load_param_importance() -> pd.DataFrame | None:
    return _read_csv(_studies_dir() / "param_analysis" / "importance.csv")


def load_param_sensitivity() -> pd.DataFrame | None:
    return _read_csv(_studies_dir() / "param_analysis" / "sensitivity_summary.csv")


def load_ml_walkforward() -> pd.DataFrame | None:
    """AUC OOS del filtro ML. outputs/walkforward/ (vacío hasta correr run_walkforward)."""
    d = outputs_dir() / "walkforward"
    if not d.is_dir():
        return None
    for name in ("walkforward.csv", "folds.csv", "auc_oos.csv"):
        df = _read_csv(d / name)
        if df is not None:
            return df
    # cualquier csv suelto
    for p in sorted(d.glob("*.csv")):
        df = _read_csv(p)
        if df is not None:
            return df
    return None
