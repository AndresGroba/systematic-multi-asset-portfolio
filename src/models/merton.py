"""Merton multivariante: pesos optimos para ETFs de riesgo. XEON.DE excluido; su peso = 1 - suma(pesos_riesgo). Merton (1969, 1971)."""

import numpy as np

# Fallback. El pipeline (run_merton) lee los MERTON_* de config.py en runtime; en produccion mandan los de config.
GAMMA             = -1      # Fallback. RRA = 1 - GAMMA = 2. Produccion: cfg.MERTON_GAMMA=-0.8 (RRA 1.8)
N_TOP_ASSETS      = 30      # Fallback. Produccion: cfg.MERTON_N_TOP=20
MAX_WEIGHT        = 0.40    # Peso máximo por ETF individual (40%). Produccion: cfg.MERTON_MAX_WEIGHT
MIN_WEIGHT        = 0.01    # Peso mínimo para considerar que hay posición (1%)
MAX_SECTOR_WEIGHT = 0.50    # Fallback (50%). Produccion: cfg.MERTON_MAX_SECTOR=0.35
FREEZE_EXIT_THR   = 0.02    # Si un congelado cae por debajo del 2% → liquidar

try:
    import src.config as _cfg
except ImportError:
    _cfg = None


def detect_regime(sigma_mercado):
    """Filosofia contrarian: SIEMPRE 100% invertido. El regimen es solo para logging y para que run_live ajuste VIEW_SCALE/DN_BAND; Merton no reduce riesgo en crisis."""
    caution_thr = getattr(_cfg, 'VOL_CAUTION_THR', 0.28) if _cfg else 0.28
    crisis_thr = getattr(_cfg, 'VOL_CRISIS_THR', 0.40) if _cfg else 0.40

    if sigma_mercado > crisis_thr:
        return 'crisis', 1.00
    elif sigma_mercado > caution_thr:
        return 'caution', 1.00
    else:
        return 'normal', 1.00


def merton_weights_risk_only(mu_BL, Sigma, risk_free_rate, tickers,
                              xeon_ticker='XEON.DE', gamma=GAMMA):
    """Pesos brutos de Merton solo para ETFs de riesgo (XEON.DE excluido): w* = (1/gamma_eff) * Sigma^-1 * (mu - r), gamma_eff = 1 - gamma."""
    risk_indices = [i for i, t in enumerate(tickers) if t != xeon_ticker]
    risk_tickers = [tickers[i] for i in risk_indices]
    N_risk = len(risk_indices)

    if N_risk == 0:
        return np.array([]), [], []

    mu_risk    = mu_BL[risk_indices]
    Sigma_risk = Sigma[np.ix_(risk_indices, risk_indices)]
    excess_return = mu_risk - risk_free_rate
    gamma_eff = 1.0 - gamma   # gamma=-0.8 (config) -> gamma_eff=1.8

    # Sigma ya viene regularizada de BL
    try:
        Sigma_inv = np.linalg.inv(Sigma_risk)
    except np.linalg.LinAlgError:
        print("  [AVISO P2] Usando pseudo-inversa para Sigma_risk")
        Sigma_inv = np.linalg.pinv(Sigma_risk)

    w_raw = (1.0 / gamma_eff) * Sigma_inv @ excess_return

    return w_raw, risk_tickers, risk_indices


def apply_constraints(w_raw, risk_tickers, categoria_por_ticker=None,
                      n_top=N_TOP_ASSETS, max_weight=MAX_WEIGHT,
                      min_weight=MIN_WEIGHT,
                      max_sector=MAX_SECTOR_WEIGHT,
                      max_risk_total=1.0):
    """Restricciones sobre los pesos brutos: long-only, top-N, cap individual, cap sectorial y cap total. Devuelve pesos con suma <= max_risk_total."""
    N = len(w_raw)
    w = w_raw.copy()

    # Sin shorts
    w = np.maximum(w, 0.0)

    if w.sum() < 1e-10:
        # Todos cero: reparto equitativo entre los primeros n_top
        w = np.zeros(N)
        w[:min(n_top, N)] = 1.0 / min(n_top, N)

    # Solo top N por peso
    sorted_idx  = np.argsort(w)[::-1]
    w_filtered  = np.zeros(N)
    w_filtered[sorted_idx[:n_top]] = w[sorted_idx[:n_top]]
    w = w_filtered

    # Cap individual
    for _ in range(10):
        if not (w > max_weight).any():
            break
        w = np.minimum(w, max_weight)

    # Cap sectorial (con overrides por categoria)
    if categoria_por_ticker and len(categoria_por_ticker) > 0:
        from collections import defaultdict
        categoria_idx = defaultdict(list)
        for i, ticker in enumerate(risk_tickers):
            cat = categoria_por_ticker.get(ticker, f'unknown_{ticker}')
            categoria_idx[cat].append(i)

        sector_overrides = {}
        if _cfg:
            sector_overrides = getattr(_cfg, 'MAX_SECTOR_OVERRIDE', {})

        for _ in range(10):
            changed = False
            for cat, indices in categoria_idx.items():
                cat_max = sector_overrides.get(cat, max_sector)
                sector_total = w[indices].sum()
                if sector_total > cat_max + 1e-6:
                    scale = cat_max / sector_total
                    for i in indices:
                        w[i] *= scale
                    changed = True
            if not changed:
                break

    # Cap total segun regimen
    total = w.sum()
    if total > max_risk_total + 1e-6:
        w = w * (max_risk_total / total)

    w[w < min_weight] = 0.0

    # Re-normalizar si la suma supera el cap (puede pasar tras eliminar pequenos)
    if w.sum() > max_risk_total + 1e-6:
        w = w * (max_risk_total / w.sum())

    return w


def check_frozen_exits(current_weights, tickers, frozen_tickers,
                       xeon_ticker='XEON.DE', threshold=FREEZE_EXIT_THR):
    """Congelados a liquidar: los que ya estaban en cartera pero cayeron por debajo del umbral (mantenerlos no compensa el coste de transaccion)."""
    liquidar = []
    for ticker in frozen_tickers:
        if ticker == xeon_ticker:
            continue
        if ticker in tickers:
            idx = tickers.index(ticker)
            if idx < len(current_weights) and current_weights[idx] < threshold:
                liquidar.append(ticker)
    return liquidar


def run_merton(bl_result, risk_free_rate,
                             xeon_ticker='XEON.DE',
                             sigma_mercado=0.15,
                             categoria_por_ticker=None,
                             current_weights=None,
                             frozen_tickers=None,
                             gamma=GAMMA):
    """Pipeline completo: mu_BL + Sigma -> pesos optimos con todas las restricciones. Devuelve weights_array, weights, selected_etfs, weight_xeon, regime, liquidar.

    Los "congelados" (frozen_tickers/current_weights) son opcionales: si no se aportan, no hay posiciones que liquidar por este motivo y liquidar sale [].
    """
    mu_BL   = bl_result['mu_BL']
    Sigma   = bl_result['Sigma']
    tickers = bl_result['tickers']
    N       = len(tickers)

    regime, max_risk_total = detect_regime(sigma_mercado)

    w_raw_risk, risk_tickers, risk_indices = merton_weights_risk_only(
        mu_BL, Sigma, risk_free_rate, tickers,
        xeon_ticker=xeon_ticker, gamma=gamma
    )

    # Lee config en runtime para la optimizacion
    n_top = getattr(_cfg, 'MERTON_N_TOP', N_TOP_ASSETS) if _cfg else N_TOP_ASSETS
    max_sector = getattr(_cfg, 'MERTON_MAX_SECTOR', MAX_SECTOR_WEIGHT) if _cfg else MAX_SECTOR_WEIGHT
    max_w = getattr(_cfg, 'MERTON_MAX_WEIGHT', MAX_WEIGHT) if _cfg else MAX_WEIGHT
    min_w = getattr(_cfg, 'MERTON_MIN_WEIGHT', MIN_WEIGHT) if _cfg else MIN_WEIGHT

    w_risk_final = apply_constraints(
        w_raw_risk,
        risk_tickers,
        categoria_por_ticker=categoria_por_ticker,
        max_risk_total=max_risk_total,
        n_top=n_top,
        max_sector=max_sector,
        max_weight=max_w,
        min_weight=min_w,
    )

    # Array completo (N elementos, XEON.DE = 0 aqui)
    weights_full = np.zeros(N)
    for i, orig_idx in enumerate(risk_indices):
        weights_full[orig_idx] = w_risk_final[i]

    # XEON.DE absorbe el complemento
    xeon_idx    = tickers.index(xeon_ticker) if xeon_ticker in tickers else -1
    weight_xeon = max(0.0, 1.0 - weights_full.sum())
    if xeon_idx >= 0:
        weights_full[xeon_idx] = weight_xeon

    liquidar = []
    if current_weights is not None and frozen_tickers:
        liquidar = check_frozen_exits(
            current_weights, tickers, frozen_tickers, xeon_ticker
        )

    selected     = [risk_tickers[i] for i in range(len(risk_tickers))
                    if w_risk_final[i] > min_w]
    weights_dict = {risk_tickers[i]: float(w_risk_final[i])
                    for i in range(len(risk_tickers))
                    if w_risk_final[i] > min_w}

    return {
        'weights_array':    weights_full,
        'weights':     weights_dict,
        'selected_etfs': selected,
        'weight_xeon':      weight_xeon,
        'regime':           regime,
        'liquidar':         liquidar,
    }