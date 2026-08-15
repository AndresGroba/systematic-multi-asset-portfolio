"""Politica unificada de rebalanceo: misma logica en engine, main/registrador y walkforward.

Regla (= if dn['rebalance'] or not positions): sin posiciones siempre invierte; con posiciones
solo si Davis-Norman pide rebalance. Pesos: siempre dn_result['final_weights_full'].
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def should_apply_rebalance_after_dn(dn_result: dict, current_positions: dict | None) -> bool:
    """Decide si rebalancear: dn['rebalance'] or not positions. Dict vacio = cartera sin titulos."""
    pos = current_positions if current_positions is not None else {}
    if not pos:
        return True
    return bool(dn_result.get("rebalance"))


def positions_from_weights_full(
    final_weights_full: dict,
    portfolio_value: float,
    prices_series: pd.Series,
    xeon_ticker: str,
) -> dict:
    """Pesos completos (incl. XEON) -> cantidades. Misma logica que engine._execute_rebalance."""
    positions: dict = {}
    used_value = 0.0

    for ticker, weight in final_weights_full.items():
        if ticker == xeon_ticker:
            continue
        price = prices_series.get(ticker, np.nan)
        if pd.isna(price) or price <= 0 or weight <= 0:
            continue
        qty = int(portfolio_value * weight / price)
        if qty > 0:
            positions[ticker] = qty
            used_value += qty * price

    xeon_price = prices_series.get(xeon_ticker, np.nan)
    if not pd.isna(xeon_price) and xeon_price > 0:
        remaining = max(portfolio_value - used_value, 0.0)
        positions[xeon_ticker] = remaining / xeon_price

    return positions
