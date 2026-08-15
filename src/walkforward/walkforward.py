"""
Walkforward con evaluacion OOS real del filtro ML. Por fold: (1) backtest de estrategia en el
tramo de test (engine.run_backtest, causal); (2) filtro ML entrenado en train y evaluado CONGELADO
en test con embargo de `horizon` dias. El paso 2 es lo que lo distingue de un backtest troceado.

Ventanas de test consecutivas y disjuntas (cobertura OOS de todo el periodo).
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.backtest.engine import run_backtest
from src.data.data_loader import download_market_data
from src.models.ml_etf_filter import (
    build_etf_ml_dataset,
    evaluate_etf_filter_oos,
    split_dataset_temporal,
)
import src.config as cfg

_ML_HORIZON = 21  # dias del target del filtro; tambien el embargo train/test


def _build_full_ml_dataset(start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    """Descarga datos (con lookback) y construye el dataset del filtro una sola vez."""
    lookback_start = (start - pd.DateOffset(months=18)).strftime("%Y-%m-%d")
    market_data = download_market_data(
        start_date=lookback_start, end_date=end.strftime("%Y-%m-%d")
    )
    metadata_by_ticker = {item["ticker"]: item for item in market_data["metadata"]}
    risk_tickers = [
        t for t in market_data["tickers"]
        if metadata_by_ticker.get(t, {}).get("role") != "defensive"
    ]
    returns_df = market_data["returns"][risk_tickers]
    prices_df = market_data["prices"][risk_tickers]
    return build_etf_ml_dataset(returns_df, prices_df, horizon=_ML_HORIZON)


def run_walkforward(
    start_date: str = "2020-01-01",
    end_date: str = "2025-12-31",
    train_months: int | None = None,
    test_months: int | None = None,
    output_dir: str = "outputs/walkforward",
) -> dict:
    """Walkforward: por ventana de test, backtest de estrategia + filtro ML OOS (train -> test, con embargo)."""
    if train_months is None:
        train_months = cfg.WF_TRAIN_MONTHS
    if test_months is None:
        test_months = cfg.WF_TEST_MONTHS

    start = pd.Timestamp(start_date)
    end = pd.Timestamp(end_date)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print("[Walkforward] Construyendo dataset ML para evaluacion OOS...")
    ml_dataset = _build_full_ml_dataset(start, end)

    rows: list[dict] = []
    train_start = start
    fold = 0

    while True:
        train_end = train_start + pd.DateOffset(months=train_months) - pd.Timedelta(days=1)
        test_start = train_end + pd.Timedelta(days=1)
        test_end = test_start + pd.DateOffset(months=test_months) - pd.Timedelta(days=1)

        if test_end > end:
            break

        fold += 1
        fold_dir = output_path / f"walkforward_fold_{fold}"

        # Validacion OOS del filtro ML (train del fold -> test del fold)
        train_df, test_df = split_dataset_temporal(
            ml_dataset, train_start, test_start, test_end, embargo_days=_ML_HORIZON
        )
        oos = evaluate_etf_filter_oos(train_df, test_df)

        print(f"\n{'=' * 60}")
        print(f"WALKFORWARD FOLD {fold}")
        print(f"  Train: {train_start.date()} -> {train_end.date()}")
        print(f"  Test:  {test_start.date()} -> {test_end.date()}")
        auc_str = f"{oos['oos_auc']:.3f}" if oos["oos_auc"] is not None else "n/a"
        acc_str = f"{oos['oos_accuracy']:.3f}" if oos["oos_accuracy"] is not None else "n/a"
        print(f"  ML OOS: AUC={auc_str} acc={acc_str} "
              f"(n_train={oos['n_train']}, n_test={oos['n_test']}, base={oos['test_base_rate']:.2f})")
        print(f"{'=' * 60}")

        row = {
            "Fold": fold,
            "Train Start": train_start.strftime("%Y-%m-%d"),
            "Train End": train_end.strftime("%Y-%m-%d"),
            "Test Start": test_start.strftime("%Y-%m-%d"),
            "Test End": test_end.strftime("%Y-%m-%d"),
            "ML OOS AUC": oos["oos_auc"],
            "ML OOS Accuracy": oos["oos_accuracy"],
            "ML Test Base Rate": oos["test_base_rate"],
            "ML n_train": oos["n_train"],
            "ML n_test": oos["n_test"],
        }

        try:
            bt_result = run_backtest(
                start_date=test_start.strftime("%Y-%m-%d"),
                end_date=test_end.strftime("%Y-%m-%d"),
                output_dir=str(fold_dir),
                verbose=False,
            )
            row["Rebalances"] = bt_result["n_rebalances"]
            for k, v in bt_result["strategy_metrics"].items():
                row[f"Strategy {k}"] = v
            for k, v in bt_result["sp500_metrics"].items():
                row[f"SP500 {k}"] = v
        except Exception as e:
            print(f"  [Fold {fold}] Error en backtest: {e}")
            row["Error"] = str(e)

        rows.append(row)
        train_start = train_start + pd.DateOffset(months=test_months)

    wf_df = pd.DataFrame(rows)
    wf_df.to_csv(output_path / "walkforward_resumen.csv", index=False)

    try:
        wf_df.to_excel(output_path / "walkforward_resumen.xlsx", index=False)
    except PermissionError:
        wf_df.to_excel(output_path / "walkforward_resumen_backup.xlsx", index=False)

    # Agregado OOS
    aggregate = {}
    if not wf_df.empty and "ML OOS AUC" in wf_df.columns:
        aucs = wf_df["ML OOS AUC"].dropna()
        accs = wf_df["ML OOS Accuracy"].dropna()
        aggregate = {
            "folds": int(len(wf_df)),
            "mean_oos_auc": float(aucs.mean()) if len(aucs) else float("nan"),
            "median_oos_auc": float(aucs.median()) if len(aucs) else float("nan"),
            "mean_oos_accuracy": float(accs.mean()) if len(accs) else float("nan"),
            "folds_auc_above_0.5": int((aucs > 0.5).sum()) if len(aucs) else 0,
        }

    print(f"\n{'=' * 60}")
    print("WALKFORWARD RESUMEN")
    print("=" * 60)
    if not wf_df.empty:
        print(wf_df.to_string(index=False))
        if aggregate:
            print(f"\n[ML OOS agregado] AUC medio={aggregate['mean_oos_auc']:.3f} | "
                  f"mediana={aggregate['median_oos_auc']:.3f} | "
                  f"accuracy media={aggregate['mean_oos_accuracy']:.3f} | "
                  f"folds con AUC>0.5: {aggregate['folds_auc_above_0.5']}/{aggregate['folds']}")
    else:
        print("No se completaron folds.")

    return {
        "walkforward": wf_df,
        "ml_oos_aggregate": aggregate,
        "output_dir": str(output_path),
    }
