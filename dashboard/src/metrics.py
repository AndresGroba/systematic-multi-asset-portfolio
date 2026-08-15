"""Financial metrics. Daily frequency, 252 trading days/year, simple returns."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.utils import (
    CAPITAL_INICIAL,
    RISK_FREE_RATE_ANNUAL,
    TRADING_DAYS_PER_YEAR,
    daily_risk_free_rate,
)


def compute_metrics(
    portfolio_df: pd.DataFrame,
    risk_free_rate_annual: float = RISK_FREE_RATE_ANNUAL,
) -> dict:
    """Compute performance metrics from the daily portfolio DataFrame."""
    if portfolio_df.empty or "nav" not in portfolio_df.columns:
        return {}

    nav = portfolio_df["nav"].dropna()
    # NAV-to-NAV returns, comparable with benchmark pct_change stats. `daily_return`
    # keeps a synthetic first observation vs initial capital for PnL accounting.
    returns = nav.pct_change().dropna()

    if len(nav) < 2:
        return {}

    nav_current = float(nav.iloc[-1])
    nav_initial = CAPITAL_INICIAL
    pnl_cum = nav_current - nav_initial
    ret_cum = nav_current / nav_initial - 1.0

    daily_pnl = (
        float(portfolio_df["daily_pnl"].dropna().iloc[-1])
        if "daily_pnl" in portfolio_df
        else 0.0
    )

    n_days = len(nav)
    cagr = (nav_current / nav_initial) ** (TRADING_DAYS_PER_YEAR / n_days) - 1.0 if n_days >= 2 else np.nan

    vol_ann = float(returns.std() * np.sqrt(TRADING_DAYS_PER_YEAR)) if len(returns) >= 2 else np.nan

    rf_annual = float(risk_free_rate_annual)
    rf_daily = daily_risk_free_rate(rf_annual)
    excess_returns = returns - rf_daily
    excess_cagr = cagr - rf_annual if not np.isnan(cagr) else np.nan

    sharpe = float(excess_returns.mean() / returns.std() * np.sqrt(TRADING_DAYS_PER_YEAR)) if returns.std() > 0 else np.nan

    # Target semideviation (Sortino denominator): square the downside-only excess
    # returns but average over the FULL series (up-days contribute 0), not just the
    # negative days. Dividing by the count of negative days only would overstate
    # downside risk and understate Sortino.
    downside = excess_returns.clip(upper=0.0)
    downside_vol_ann = (
        float(np.sqrt((downside ** 2).mean()) * np.sqrt(TRADING_DAYS_PER_YEAR))
        if len(excess_returns) >= 1
        else np.nan
    )
    sortino = (
        float(excess_returns.mean() * TRADING_DAYS_PER_YEAR / downside_vol_ann)
        if downside_vol_ann and not np.isnan(downside_vol_ann) and downside_vol_ann > 0
        else np.nan
    )

    rolling_max = nav.cummax()
    drawdown = (nav - rolling_max) / rolling_max
    max_dd = float(drawdown.min())
    current_dd = float(drawdown.iloc[-1])
    dd_series = drawdown

    total_costs = float(portfolio_df["cum_cost"].iloc[-1]) if "cum_cost" in portfolio_df else 0.0
    cost_drag_pct = total_costs / CAPITAL_INICIAL

    cash_current = float(portfolio_df["cash"].iloc[-1]) if "cash" in portfolio_df else 0.0
    cash_pct = cash_current / nav_current if nav_current > 0 else 0.0

    weight_cols = [c for c in portfolio_df.columns if c.endswith("_weight")]
    latest_weights = (
        portfolio_df[weight_cols].iloc[-1].dropna()
        if weight_cols
        else pd.Series(dtype=float)
    )
    active_weights = latest_weights[latest_weights > 0.001]
    n_etfs = int(len(active_weights))
    gross_exposure = float(latest_weights.abs().sum()) if not latest_weights.empty else 0.0

    # Liquidity = XEON.DE weight (money market) + residual cash
    xeon_w = float(latest_weights.get("XEON.DE_weight", 0.0))
    liquidity_pct = xeon_w + max(cash_pct, 0.0)
    if abs(liquidity_pct) < 1e-8:
        liquidity_pct = 0.0

    calmar = excess_cagr / abs(max_dd) if (not np.isnan(excess_cagr) and max_dd != 0) else np.nan

    hwm = float(nav.max())
    hwm_date = nav.idxmax()
    recovery_needed = (hwm / nav_current) - 1.0 if nav_current < hwm else 0.0

    days_inception = (nav.index[-1] - nav.index[0]).days

    # Phase 1 covers Mar 12 → Apr 9 (incl. the Apr 3–9 gap, positions held). La etapa
    # multi-activo (fases 2 y 3) es todo desde el 10-abr -> base temporal, no por regime,
    # para que la fase 3 (rebalanceo 13-may) cuente como multi-activo, no aparte.
    _phase2_start = pd.Timestamp("2026-04-10")
    phase2_rows = portfolio_df[portfolio_df.index >= _phase2_start]["nav"].dropna()

    pre_p2_nav = portfolio_df[portfolio_df.index < _phase2_start]["nav"].dropna()
    if not pre_p2_nav.empty:
        phase1_pnl = float(pre_p2_nav.iloc[-1]) - CAPITAL_INICIAL
        phase1_ret = phase1_pnl / CAPITAL_INICIAL
    else:
        phase1_pnl = 0.0
        phase1_ret = 0.0

    if not phase2_rows.empty:
        # Baseline = last NAV before Phase 2, so Apr-10 PnL belongs to Phase 2, not Phase 1
        phase2_nav_open = float(pre_p2_nav.iloc[-1]) if not pre_p2_nav.empty else float(phase2_rows.iloc[0])
        phase2_pnl = float(phase2_rows.iloc[-1]) - phase2_nav_open
        phase2_ret = phase2_pnl / phase2_nav_open if phase2_nav_open != 0 else 0.0
    else:
        phase2_pnl = 0.0
        phase2_ret = 0.0

    phase1_costs = float(
        portfolio_df[portfolio_df.index < _phase2_start]["daily_cost"].sum()
    )
    phase2_costs = float(
        portfolio_df[portfolio_df.index >= _phase2_start]["daily_cost"].sum()
    )
    phase1_pnl_gross = phase1_pnl + phase1_costs
    phase2_pnl_gross = phase2_pnl + phase2_costs

    return {
        "nav_current": nav_current,
        "pnl_cum": pnl_cum,
        "ret_cum": ret_cum,
        "cagr": cagr,
        "excess_cagr": excess_cagr,
        "daily_pnl": daily_pnl,
        "vol_ann": vol_ann,
        "downside_vol_ann": downside_vol_ann,
        "risk_free_rate_annual": rf_annual,
        "risk_free_rate_daily": rf_daily,
        "sharpe": sharpe,
        "sortino": sortino,
        "calmar": calmar,
        "max_dd": max_dd,
        "current_dd": current_dd,
        "recovery_needed": recovery_needed,
        "hwm_date": hwm_date,
        "total_costs": total_costs,
        "cost_drag_pct": cost_drag_pct,
        "cash_current": cash_current,
        "cash_pct": cash_pct,
        "liquidity_pct": liquidity_pct,
        "gross_exposure": gross_exposure,
        "n_etfs": n_etfs,
        "days_inception": days_inception,
        "phase1_pnl": phase1_pnl,
        "phase1_ret": phase1_ret,
        "phase1_costs": phase1_costs,
        "phase1_pnl_gross": phase1_pnl_gross,
        "phase2_pnl": phase2_pnl,
        "phase2_ret": phase2_ret,
        "phase2_costs": phase2_costs,
        "phase2_pnl_gross": phase2_pnl_gross,
        "dd_series": dd_series,
        "nav_series": nav,
    }


def compute_drawdown_series(nav: pd.Series) -> pd.Series:
    """Compute drawdown series from NAV."""
    rolling_max = nav.cummax()
    return (nav - rolling_max) / rolling_max
