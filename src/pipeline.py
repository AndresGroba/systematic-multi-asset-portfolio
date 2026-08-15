"""Cadena de decision por fecha de revision: senal -> filtro ML -> clustering ->
Black-Litterman -> Merton -> Davis-Norman. La usan engine, walkforward y run_live.

Contrarian en caidas: VIEW_SCALE se amplifica, las bandas DN se ensanchan y Merton
sigue 100% invertido (sin cap de regimen).
"""

from __future__ import annotations

import numpy as np

from src.metrics.composite_signal import compute_composite_signal
from src.models.black_litterman import run_black_litterman
from src.models.merton import run_merton
from src.models.davis_norman_bands import run_davis_norman_bands
import src.config as cfg
from src.models.ml_etf_filter import (
    build_etf_ml_dataset,
    train_etf_filter,
    predict_etf_probabilities,
    adjust_scores_with_ml,
)
from src.models.ml_clustering import (
    build_clustering_features,
    run_etf_clustering,
    apply_cluster_constraints,
)


def _detect_regime_early(returns_df) -> str:
    """Regimen multi-factor (vol media, MDD de la cartera EW, correlacion media).
    Cualquier disparador fuerte -> crisis; si no, caution; si no, normal."""
    raw_vol = float(returns_df.std().mean() * np.sqrt(252))

    lb = min(int(getattr(cfg, "REGIME_LOOKBACK_DD", 252)), len(returns_df))
    ew_mdd = 0.0
    if lb >= 20:
        ew_ret = returns_df.tail(lb).mean(axis=1)
        w = (1.0 + ew_ret).cumprod()
        dd_series = w / w.cummax() - 1.0
        ew_mdd = float(-dd_series.min()) if len(dd_series) else 0.0

    avg_corr = 0.0
    cl = min(int(getattr(cfg, "REGIME_CORR_LOOKBACK", 63)), len(returns_df))
    if cl >= 20:
        cmat = returns_df.tail(cl).corr().values
        m = cmat.shape[0]
        if m > 1:
            tri = np.triu_indices(m, k=1)
            avg_corr = float(np.nanmean(cmat[tri]))

    mdd_c = float(getattr(cfg, "REGIME_EW_MDD_CRISIS", 0.28))
    mdd_a = float(getattr(cfg, "REGIME_EW_MDD_CAUTION", 0.16))
    cr_c = float(getattr(cfg, "REGIME_AVG_CORR_CRISIS", 0.52))
    cr_a = float(getattr(cfg, "REGIME_AVG_CORR_CAUTION", 0.38))

    if raw_vol > cfg.VOL_CRISIS_THR or ew_mdd > mdd_c or avg_corr > cr_c:
        return "crisis"
    if raw_vol > cfg.VOL_CAUTION_THR or ew_mdd > mdd_a or avg_corr > cr_a:
        return "caution"
    return "normal"


def run_pipeline(
    returns_df,
    prices_df,
    current_weights=None,
    review_date=None,
    risk_free_rate=None,
    categoria_por_ticker=None,
):
    if current_weights is None:
        current_weights = {}
    if categoria_por_ticker is None:
        categoria_por_ticker = cfg.ETF_UNIVERSE

    if risk_free_rate is None:
        if review_date is not None:
            risk_free_rate = cfg.get_risk_free_rate(review_date)
        else:
            risk_free_rate = cfg.BL_RF

    valid_tickers = [
        t for t in returns_df.columns
        if returns_df[t].dropna().shape[0] >= 60
    ]
    returns_clean = returns_df[valid_tickers]
    prices_clean = prices_df[[t for t in valid_tickers if t in prices_df.columns]]

    if returns_clean.empty:
        raise ValueError("No hay ETFs con datos suficientes para el pipeline.")

    # Regimen anticipado (ajustes contrarian)
    regime_early = _detect_regime_early(returns_clean)

    # Boost VIEW_SCALE en crisis (contrarian: comprar la caida)
    effective_view_scale = cfg.VIEW_SCALE
    if regime_early == "crisis":
        effective_view_scale *= cfg.CRISIS_VIEW_BOOST
    elif regime_early == "caution":
        effective_view_scale *= cfg.CAUTION_VIEW_BOOST

    # 1) Senal compuesta
    scores = compute_composite_signal(
        returns_clean, prices_clean,
        momentum_window=cfg.COMPOSITE_MOMENTUM_WINDOW,
        momentum_skip=cfg.COMPOSITE_MOMENTUM_SKIP,
        reversal_window=cfg.COMPOSITE_REVERSAL_WINDOW,
        trend_window=cfg.COMPOSITE_TREND_WINDOW,
        vol_window=cfg.COMPOSITE_VOL_WINDOW,
        drawdown_window=cfg.COMPOSITE_DRAWDOWN_WINDOW,
        weights=cfg.COMPOSITE_WEIGHTS,
        ticker_categories=cfg.ETF_UNIVERSE,
        category_weights=cfg.CATEGORY_SIGNAL_WEIGHTS,
    )
    # Filtro ML (XGBoost): ajusta scores por prob. de exito. Toggle USE_ML_FILTER.
    if getattr(cfg, "USE_ML_FILTER", True):
        dataset_ml = build_etf_ml_dataset(returns_clean, prices_clean)
        model_ml = train_etf_filter(dataset_ml)
        ml_probs = predict_etf_probabilities(model_ml, returns_clean, prices_clean)
        scores = adjust_scores_with_ml(scores, ml_probs, strength=cfg.ML_FILTER_STRENGTH)

    # Clustering K-Means: penaliza redundancia por cluster. Toggle USE_CLUSTERING.
    if getattr(cfg, "USE_CLUSTERING", True):
        cluster_features = build_clustering_features(
            returns_clean,
            lookback=cfg.CLUSTER_LOOKBACK,
        )
        cluster_result = run_etf_clustering(
            cluster_features,
            n_clusters=cfg.CLUSTER_N,
        )
        scores = apply_cluster_constraints(
            scores=scores,
            cluster_df=cluster_result,
            max_per_cluster=cfg.CLUSTER_MAX_PER_CLUSTER,
            base_penalty=cfg.CLUSTER_BASE_PENALTY,
        )

    # 2) Black-Litterman con VIEW_SCALE ajustado por regimen
    bl_result = run_black_litterman(
        returns_clean, scores,
        rf=risk_free_rate,
        delta=cfg.BL_DELTA,
        tau=cfg.BL_TAU,
        view_scale=effective_view_scale,
        short_window=cfg.COV_SHORT_WINDOW,
        long_window=cfg.COV_LONG_WINDOW,
        blend_alpha=cfg.COV_BLEND_ALPHA,
        ewma_lambda=cfg.EWMA_LAMBDA,
        shrinkage=cfg.COV_SHRINKAGE,
        review_date=review_date,
        prior_weights_mode=getattr(cfg, "BL_PRIOR_WEIGHTS_MODE", "equal"),
    )

    # 3) Merton (siempre 100% invertido)
    sigma_mercado = float(np.sqrt(np.diag(bl_result["Sigma"])).mean())
    merton_result = run_merton(
        bl_result,
        risk_free_rate=risk_free_rate,
        sigma_mercado=sigma_mercado,
        categoria_por_ticker=categoria_por_ticker,
        gamma=cfg.MERTON_GAMMA,
    )

    # 4) Davis-Norman con bandas ajustadas por regimen
    dn_band = cfg.DN_BAND
    if regime_early == "crisis":
        dn_band *= cfg.DN_CRISIS_MULT
    elif regime_early == "caution":
        dn_band *= cfg.DN_CAUTION_MULT

    dn_result = run_davis_norman_bands(
        current_weights,
        merton_result["weights"],
        band=dn_band,
        min_band=cfg.DN_MIN_BAND,
    )

    return {
        "scores": scores,
        "bl_result": bl_result,
        "merton_result": merton_result,
        "dn_result": dn_result,
        "risk_free_rate": risk_free_rate,
        "regime_early": regime_early,
        "effective_view_scale": effective_view_scale,
    }
