# -*- coding: utf-8 -*-
"""Filtro ML (XGBoost) que estima la probabilidad de retorno positivo por ETF y ajusta los scores de la senal compuesta."""

import numpy as np
import pandas as pd

from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier


def _ml_seed() -> int:
    """Semilla ML desde config (fallback 42). Reproducibilidad — regla del proyecto."""
    try:
        import src.config as _cfg
        return int(getattr(_cfg, "ML_SEED", 42))
    except Exception:
        return 42

FEATURE_COLS = [
    "mom_1m",
    "mom_3m",
    "mom_6m",
   # "reversal_1m",
    "vol_3m",
    "trend_200",
    "drawdown_1y",
]


def build_etf_ml_dataset(returns_df, prices_df, horizon=21, min_history=252):
    """Dataset de entrenamiento del filtro ML. Target = 1 si el retorno a `horizon` dias es positivo."""

    rows = []

    for ticker in returns_df.columns:

        if ticker not in prices_df.columns:
            continue

        r = returns_df[ticker].dropna()
        p = prices_df[ticker].dropna()

        idx = r.index.intersection(p.index)
        r = r.loc[idx]
        p = p.loc[idx]

        if len(r) < min_history + horizon:
            continue

        for i in range(min_history, len(r) - horizon):

            past_r = r.iloc[: i + 1]
            past_p = p.iloc[: i + 1]

            mom_1m = (1 + past_r.tail(21)).prod() - 1
            mom_3m = (1 + past_r.tail(63)).prod() - 1
            mom_6m = (1 + past_r.tail(126)).prod() - 1
            #reversal_1m = -((1 + past_r.tail(21)).prod() - 1)

            vol_3m = past_r.tail(63).std() * np.sqrt(252)

            trend_200 = past_p.iloc[-1] / past_p.tail(200).mean() - 1

            drawdown_1y = past_p.iloc[-1] / past_p.tail(252).max() - 1

            future_returns = r.iloc[i + 1 : i + 1 + horizon]
            fwd_ret = (1 + future_returns).prod() - 1

            rows.append(
                {
                    "date": r.index[i],
                    "ticker": ticker,
                    "mom_1m": mom_1m,
                    "mom_3m": mom_3m,
                    "mom_6m": mom_6m,
                    #"reversal_1m": reversal_1m,
                    "vol_3m": vol_3m,
                    "trend_200": trend_200,
                    "drawdown_1y": drawdown_1y,
                    "fwd_ret": fwd_ret,
                    "target": int(fwd_ret > 0),
                }
            )

    dataset = pd.DataFrame(rows)
    dataset = dataset.replace([np.inf, -np.inf], np.nan).dropna()

    return dataset


def train_etf_filter(dataset):
    """Entrena un XGBoost para estimar la probabilidad de retorno positivo por ETF."""

    X = dataset[FEATURE_COLS]
    y = dataset["target"]

    model = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "clf",
                XGBClassifier(
                    n_estimators=100,
                    max_depth=3,
                    learning_rate=0.05,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    objective="binary:logistic",
                    eval_metric="logloss",
                    random_state=_ml_seed(),
                    n_jobs=-1,
                    ),
                ),  
        ]
    )

    model.fit(X, y)

    return model


def predict_etf_probabilities(model, returns_df, prices_df):
    """Probabilidad ML actual para cada ETF."""

    rows = []

    for ticker in returns_df.columns:

        if ticker not in prices_df.columns:
            continue

        r = returns_df[ticker].dropna()
        p = prices_df[ticker].dropna()

        idx = r.index.intersection(p.index)
        r = r.loc[idx]
        p = p.loc[idx]

        if len(r) < 252:
            continue

        rows.append(
            {
                "ticker": ticker,
                "mom_1m": (1 + r.tail(21)).prod() - 1,
                "mom_3m": (1 + r.tail(63)).prod() - 1,
                "mom_6m": (1 + r.tail(126)).prod() - 1,
                #"reversal_1m": -((1 + r.tail(21)).prod() - 1),
                "vol_3m": r.tail(63).std() * np.sqrt(252),
                "trend_200": p.iloc[-1] / p.tail(200).mean() - 1,
                "drawdown_1y": p.iloc[-1] / p.tail(252).max() - 1,
            }
        )

    live_df = pd.DataFrame(rows)

    if live_df.empty:
        return pd.Series(dtype=float, name="ml_probability")

    live_df = live_df.set_index("ticker")
    live_df = live_df.replace([np.inf, -np.inf], np.nan).dropna()

    probs = model.predict_proba(live_df[FEATURE_COLS])[:, 1]

    return pd.Series(probs, index=live_df.index, name="ml_probability")


def adjust_scores_with_ml(scores, ml_probabilities, strength=1.0):
    """Ajusta los scores por la probabilidad ML, reforzando la vista cuando el ML coincide con el signo del score y atenuandola cuando discrepa: multiplier = 1 + strength*sign(score)*(prob-0.5)."""

    probs = ml_probabilities.reindex(scores.index).fillna(0.5)

    # sign(score) hace el ajuste direccional: prob = P(ETF sube), asi que solo
    # potencia la vista (mas magnitud) cuando coincide con el signo del score y
    # la atenua cuando discrepa. Sin el signo, un score bajista (negativo) con
    # prob>0.5 se volveria MAS negativo (mas bajista), invirtiendo lo que dice el
    # ML. Con strength<2 el multiplier permanece >0 para ambos signos.
    multiplier = 1 + strength * np.sign(scores) * (probs - 0.5)

    adjusted_scores = scores * multiplier
    adjusted_scores.name = "composite_signal_ml_adjusted"

    return adjusted_scores.sort_values(ascending=False)


def split_dataset_temporal(dataset, train_start, test_start, test_end, embargo_days=21):
    """Particion temporal rolling train/test sobre la columna 'date'. El embargo (>= horizon) evita fuga por solapamiento de ventanas: el target de train mira `horizon` dias al futuro."""
    train_start = pd.Timestamp(train_start)
    test_start = pd.Timestamp(test_start)
    test_end = pd.Timestamp(test_end)
    embargo = pd.Timedelta(days=int(embargo_days) + 1)

    dates = pd.to_datetime(dataset["date"])
    train_df = dataset[(dates >= train_start) & (dates < (test_start - embargo))]
    test_df = dataset[(dates >= test_start) & (dates <= test_end)]
    return train_df, test_df


def evaluate_etf_filter_oos(train_df, test_df):
    """Entrena el filtro en train y lo evalua OOS en test. Devuelve AUC (None si test no tiene ambas clases), accuracy (umbral 0.5), tasa base y tamanos."""
    metrics = {
        "n_train": int(len(train_df)),
        "n_test": int(len(test_df)),
        "test_base_rate": float(test_df["target"].mean()) if len(test_df) else float("nan"),
        "oos_auc": None,
        "oos_accuracy": None,
    }
    if len(train_df) < 50 or len(test_df) < 20:
        return metrics

    model = train_etf_filter(train_df)
    proba = model.predict_proba(test_df[FEATURE_COLS])[:, 1]
    y_true = test_df["target"].to_numpy()

    metrics["oos_accuracy"] = float(accuracy_score(y_true, (proba >= 0.5).astype(int)))
    if len(np.unique(y_true)) == 2:
        metrics["oos_auc"] = float(roc_auc_score(y_true, proba))
    return metrics