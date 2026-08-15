"""Parámetros de la estrategia para la pestaña de documentación del dashboard.

Valores tomados de `src/config.py`; el dashboard está desacoplado del paquete
raíz, así que se replican aquí de forma curada.
"""

# Universo, capital y ventanas de backtest
UNIVERSE = [
    ("Universo", "~43 ETFs", "42 de riesgo + XEON.DE como activo monetario/defensivo."),
    ("Capital inicial", "10.000.000 €", "Capital real de arranque de la cartera."),
    ("Frecuencia de rebalanceo", "Mensual (fin de mes)",
     "Mejor compromiso entre reactividad y costes; en la ventana canónica el barrido confirma "
     "que mensual es la frecuencia preferida."),
    ("Tasa libre de riesgo", "Facilidad de depósito BCE",
     "En vivo desde la API del BCE; usada en Sharpe/Sortino (≈2% anual reciente)."),
    ("Ventana canónica", "2013-08 → 2026-05",
     "~12.75 años, multi-régimen. Warmup desde feb-2012 (inicio de MSCI World URTH)."),
    ("Sub-periodo destacado", "2020-01 → 2026-05",
     "Post-COVID; donde la estrategia bate también al S&P 500."),
    ("Contrafactual (en vivo)", "2026-03 → 2026-05",
     "Periodo real de despliegue con capital (~2 meses)."),
]

# Parámetros por bloque del pipeline: (parámetro, valor, descripción)
PARAM_BLOCKS = {
    "Señal compuesta": [
        ("Momentum (ventana / skip)", "126 / 21 días",
         "Momentum 6-1 (Jegadeesh-Titman): 6 meses saltando el último (evita el reversal corto)."),
        ("Reversal", "21 días", "Short-term reversal a 1 mes (Lehmann)."),
        ("Tendencia", "200 días", "Media móvil de 200 sesiones, estándar de tendencia."),
        ("Volatilidad", "63 días", "Trimestre bursátil."),
        ("Drawdown", "252 días", "Premia activos lejos de su máximo reciente (sesgo contrarian de medio plazo)."),
        ("Pesos de la señal", "mom .40 · rev .10 · trend .35 · vol .15 · dd .00",
         "Pesos globales de los cinco factores; se ajustan por categoría de activo (renta fija, "
         "sectores volátiles, etc.) para reflejar su comportamiento."),
    ],
    "Covarianza robusta": [
        ("COV_SHORT / COV_LONG", "63 / 252 días", "Trimestre / año."),
        ("COV_BLEND_ALPHA", "0.78",
         "Peso de la covarianza corta (reactiva) en el blend con la larga; una covarianza reactiva "
         "escala el riesgo al entrar en las caídas."),
        ("EWMA_LAMBDA", "0.94", "Estándar RiskMetrics (JP Morgan 1996), half-life ≈ 11 días."),
        ("COV_SHRINKAGE", "0.10",
         "Shrinkage suave (Ledoit-Wolf) hacia una diagonal, para estabilizar la inversión de la covarianza."),
    ],
    "Black-Litterman": [
        ("BL_TAU", "0.05", "Incertidumbre del prior en Black-Litterman."),
        ("BL_DELTA", "2.5", "Aversión al riesgo de mercado ≈ 2–3 (He & Litterman)."),
        ("VIEW_SCALE", "0.37",
         "Intensidad con que las views contrarian se apartan del prior; mayor VIEW_SCALE = más "
         "convicción contrarian (se amplifica en caution y crisis)."),
        ("Prior", "inv_vol (1/σ)", "Prior 1/volatilidad como proxy de mercado sin caps."),
    ],
    "Régimen (filosofía contrarian)": [
        ("Umbrales de volatilidad", "caution 0.28 · crisis 0.40",
         "Vol anualizada que dispara cada régimen; 40% ≈ niveles de 2008/2020."),
        ("Drawdown EW", "caution 0.16 · crisis 0.28",
         "Caída de la cartera equiponderada que dispara cada régimen."),
        ("Correlación media", "caution 0.38 · crisis 0.52",
         "En crisis 'todo se mueve junto': correlación media alta."),
        ("Amplificación de views", "caution ×1.15 · crisis ×1.30",
         "Comprar más fuerte la caída (tesis contrarian)."),
        ("Ensanche de bandas DN", "caution ×1.5 · crisis ×2.0",
         "Bandas de no-trade más anchas en crisis: no vender en pánico."),
    ],
    "Merton (asignación)": [
        ("MERTON_GAMMA", "−0.8 (γ_RRA = 1.8)",
         "Aversión al riesgo; Mehra-Prescott estima γ ∈ [2,4], 1.8 algo agresivo (coherente contrarian)."),
        ("MERTON_N_TOP", "20", "Número máximo de activos de riesgo en cartera."),
        ("Límites por activo", "máx 40% · mín 1%", "Concentración por activo."),
        ("MERTON_MAX_SECTOR", "35%", "Límite por categoría; diversificación."),
    ],
    "Davis-Norman (no-trade)": [
        ("DN_BAND / DN_MIN_BAND", "0.05 / 0.02",
         "Banda de inacción: controla turnover vs. tracking. Solo se rebalancea si se cruza."),
    ],
    "Costes de transacción": [
        ("TX_COST_PER_SIDE", "0.08% (0.05–0.12%)",
         "Rango de bróker barato para ETFs por lado. Comisión real por ETF cuando existe en el Excel."),
    ],
    "Refinamiento ML de la señal": [
        ("USE_ML_FILTER / CLUSTERING", "On",
         "Filtro XGBoost (ajusta los scores con la probabilidad de un clasificador) + clustering "
         "K-Means (penaliza ETFs redundantes), como capa de refinamiento de la señal."),
        ("ML_FILTER_STRENGTH", "0.8",
         "Intensidad del ajuste direccional: 1 + 0.8·signo(score)·(p−0.5) — refuerza la vista cuando "
         "el ML coincide con el signo del score y la atenúa cuando discrepa."),
        ("CLUSTER_N", "7", "Número de clústers K-Means (penaliza ETFs redundantes)."),
        ("Semilla ML", "42", "Reproducibilidad de XGBoost / K-Means."),
    ],
}
