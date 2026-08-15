# -*- coding: utf-8 -*-
"""Clustering KMeans de ETFs y penalizacion por solapamiento (max 1 por cluster)."""

import numpy as np
import pandas as pd

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


def _ml_seed() -> int:
    """Semilla ML desde config (fallback 42). Reproducibilidad — regla del proyecto."""
    try:
        import src.config as _cfg
        return int(getattr(_cfg, "ML_SEED", 42))
    except Exception:
        return 42


def build_clustering_features(
    returns_df: pd.DataFrame,
    lookback: int = 126,
) -> pd.DataFrame:
    """Features por ETF para el clustering: momentum 1m/3m, volatilidad, max drawdown y beta al mercado."""

    rows = []

    returns_df = returns_df.dropna(how="all").fillna(0.0)

    recent = returns_df.tail(lookback)

    for ticker in recent.columns:

        r = recent[ticker].dropna()

        if len(r) < 60:
            continue

        momentum_1m = (1 + r.tail(21)).prod() - 1
        momentum_3m = (1 + r.tail(63)).prod() - 1

        vol = r.std() * np.sqrt(252)

        wealth = (1 + r).cumprod()
        dd = wealth / wealth.cummax() - 1
        max_dd = dd.min()

        beta_market = np.corrcoef(
            r.values,
            recent.mean(axis=1).loc[r.index].values,
        )[0, 1]

        rows.append(
            {
                "ticker": ticker,
                "momentum_1m": momentum_1m,
                "momentum_3m": momentum_3m,
                "volatility": vol,
                "max_drawdown": max_dd,
                "beta_market": beta_market,
            }
        )

    df = pd.DataFrame(rows)

    return df


def run_etf_clustering(
    features_df: pd.DataFrame,
    n_clusters: int = 7,
):
    """Clustering KMeans de ETFs sobre las features estandarizadas."""

    feature_cols = [
        "momentum_1m",
        "momentum_3m",
        "volatility",
        "max_drawdown",
        "beta_market",
    ]

    X = features_df[feature_cols].copy()

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = KMeans(
        n_clusters=n_clusters,
        random_state=_ml_seed(),
        n_init=20,
    )

    clusters = model.fit_predict(X_scaled)

    result = features_df.copy()
    result["cluster"] = clusters

    return result


def apply_cluster_constraints(
    scores: pd.Series,
    cluster_df: pd.DataFrame,
    max_per_cluster: int = 1,
    base_penalty: float = 0.90,
) -> pd.Series:
    """Penalizacion progresiva por concentracion: el k-esimo ETF de un cluster por encima de max_per_cluster se multiplica por base_penalty**(k-max)."""

    adjusted_scores = scores.copy()
    ranked = scores.sort_values(ascending=False)

    ticker_to_cluster = dict(zip(cluster_df["ticker"], cluster_df["cluster"]))

    cluster_count = {}

    for ticker in ranked.index:
        cluster = ticker_to_cluster.get(ticker)

        if cluster is None:
            continue

        cluster_count.setdefault(cluster, 0)
        cluster_count[cluster] += 1

        excess = cluster_count[cluster] - max_per_cluster

        if excess > 0:
            adjusted_scores[ticker] *= base_penalty ** excess

    adjusted_scores.name = "composite_signal_ml_cluster_adjusted"

    return adjusted_scores.sort_values(ascending=False)