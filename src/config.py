"""Configuracion centralizada — filosofia contrarian + momentum."""

import pandas as pd

# UNIVERSO DE ETFs  (42 activos de riesgo + XEON.DE defensivo)

XEON_TICKER = "XEON.DE"

# Inversor EUR. Solo estos cotizan en EUR; el resto se convierte con EURUSD=X.
# Anadir aqui cualquier ETF en EUR o el NAV mezclaria divisas (sesgo A5).
EUR_QUOTED_TICKERS = {"IUSN.DE", "XEON.DE"}
FX_PAIR_EURUSD = "EURUSD=X"

ETF_UNIVERSE = {
    # ---- Equity: Desarrollada ----
    "IWDA.L":  "global_equity",
    "VGK":     "europe_equity",
    "EWJ":     "asia_developed",
    "IUSN.DE": "global_equity",

    # ---- Equity: Emergente ----
    "IEMA.L":  "emerging_equity",
    "MCHI":    "asia_emerging",
    "EWZ":     "latam",
    "INDA":    "asia_emerging",
    "EWT":     "asia_emerging",
    "EWY":     "asia_emerging",

    # ---- US Sectorial ----
    "XLY":     "us_discretionary",
    "XLC":     "us_communications",
    "XLF":     "us_financials",
    "KBE":     "us_financials",
    "XLE":     "us_energy",
    "XOP":     "us_energy",
    "XLI":     "us_industrials",
    "IYT":     "us_industrials",
    "XLB":     "us_materials",
    "ITB":     "us_housing",
    "XLV":     "us_healthcare",
    "XBI":     "us_healthcare",
    "XLP":     "us_staples",
    "XLU":     "us_utilities",

    # ---- Tech & Innovation ----
    "XLK":     "tech",
    "SOXX":    "tech",
    "IGV":     "tech",
    "CIBR":    "tech",
    "BOTZ":    "tech",

    # ---- Real Estate ----
    "VNQ":     "real_estate",

    # ---- Tematico ----
    "LIT":     "thematic",

    # ---- Commodities (UN solo grupo -> max 35-40%) ----
    "GLD":     "commodities",
    "SLV":     "commodities",
    "DBA":     "commodities",
    "COPX":    "commodities",

    # ---- Renta Fija (UN solo grupo -> max 35-40%) ----
    "TLT":     "fixed_income",
    "IEF":     "fixed_income",
    "TIP":     "fixed_income",
    "HYG":     "fixed_income",
    "LQD":     "fixed_income",
    "EMB":     "fixed_income",

    # ---- Crypto ----
    "BITO":    "crypto",
}

# Tasa libre de riesgo (BCE, facilidad de deposito).
# API del BCE en vivo (reproducible: los tipos historicos no se revisan), cache en
# memoria. Fallback offline: snapshot src/data/ecb_rates.csv, luego _ECB_RATE_FALLBACK.

_ECB_API_URL = (
    "https://data-api.ecb.europa.eu/service/data/FM/"
    "B.U2.EUR.4F.KR.DFR.LEV?format=csvdata"
)

# Ultimo recurso (sin red y sin CSV). No es la fuente; solo evita romper offline.
_ECB_RATE_FALLBACK = [
    ("2011-12-14", 0.0025), ("2012-07-11", 0.0000), ("2014-06-11", -0.001),
    ("2019-09-18", -0.005), ("2022-07-27", 0.000), ("2023-09-20", 0.040),
    ("2024-06-12", 0.0375), ("2025-06-11", 0.020),
]

BL_RF = 0.025

_ecb_rates_cache = None


def _parse_ecb_csvdata(text: str) -> list[tuple[str, float]]:
    lines = text.splitlines()
    header = lines[0].split(",")
    ti, oi = header.index("TIME_PERIOD"), header.index("OBS_VALUE")
    rows = []
    for line in lines[1:]:
        c = line.split(",")
        if len(c) > max(ti, oi) and c[oi].strip():
            rows.append((c[ti].strip(), float(c[oi]) / 100.0))  # % -> fraccion
    return sorted(rows)


def _load_ecb_rates() -> list[tuple[str, float]]:
    # 1) API del BCE en vivo
    try:
        import urllib.request
        req = urllib.request.Request(_ECB_API_URL, headers={"User-Agent": "python"})
        rows = _parse_ecb_csvdata(urllib.request.urlopen(req, timeout=10).read().decode())
        if rows:
            return rows
    except Exception:
        pass
    # 2) snapshot CSV (offline)
    try:
        import csv
        from pathlib import Path
        p = Path(__file__).resolve().parent / "data" / "ecb_rates.csv"
        with open(p, encoding="utf-8") as f:
            rows = [(r["date"], float(r["rate"])) for r in csv.DictReader(f)]
        if rows:
            return sorted(rows)
    except Exception:
        pass
    # 3) fallback minimo
    return sorted(_ECB_RATE_FALLBACK)


def get_risk_free_rate(date) -> float:
    global _ecb_rates_cache
    if _ecb_rates_cache is None:
        _ecb_rates_cache = _load_ecb_rates()
    ts = pd.Timestamp(date)
    rate = _ecb_rates_cache[0][1]  # antes del primer cambio: el primer nivel conocido
    for d, r in _ecb_rates_cache:
        if ts >= pd.Timestamp(d):
            rate = r
    return rate


# SENAL COMPUESTA  (cinco factores: momentum, reversal, trend, vol_penalty, drawdown_buy)

COMPOSITE_MOMENTUM_WINDOW = 126
COMPOSITE_MOMENTUM_SKIP   = 21
COMPOSITE_REVERSAL_WINDOW = 21
COMPOSITE_TREND_WINDOW    = 200
COMPOSITE_VOL_WINDOW      = 63
COMPOSITE_DRAWDOWN_WINDOW = 252

COMPOSITE_WEIGHTS = {
    "momentum":      0.40,
    "reversal":      0.10,
    "trend":         0.35,
    "vol_penalty":   0.15,
    "drawdown_buy":  0.00,
}

# Refinamiento ML de la senal (filtro XGBoost + clustering K-Means)
# Toggles para ablacion (scripts/study_ml_value.py); default = estrategia final.
USE_ML_FILTER            = True  # filtro XGBoost sobre los scores
USE_CLUSTERING           = True  # penalizacion por clustering K-Means
ML_FILTER_STRENGTH       = 0.8   # intensidad del ajuste por probabilidad: 1 + strength*(p-0.5)
CLUSTER_LOOKBACK         = 126   # ventana de features para el clustering
CLUSTER_N                = 7     # numero de clusters K-Means
CLUSTER_MAX_PER_CLUSTER  = 1     # ETFs sin penalizar por cluster
CLUSTER_BASE_PENALTY     = 0.90  # penalizacion progresiva base_penalty^exceso
ML_SEED                  = 42    # semilla de los modelos ML (XGBoost, K-Means) — reproducibilidad

CATEGORY_SIGNAL_WEIGHTS = {
    "global_equity":     {"momentum": 0.42, "reversal": 0.15, "trend": 0.33, "vol_penalty": 0.10},
    "europe_equity":     {"momentum": 0.42, "reversal": 0.20, "trend": 0.28, "vol_penalty": 0.10},
    "asia_developed":    {"momentum": 0.42, "reversal": 0.20, "trend": 0.28, "vol_penalty": 0.10},
    "emerging_equity":   {"momentum": 0.40, "reversal": 0.20, "trend": 0.30, "vol_penalty": 0.10},
    "asia_emerging":     {"momentum": 0.40, "reversal": 0.20, "trend": 0.30, "vol_penalty": 0.10},
    "latam":             {"momentum": 0.40, "reversal": 0.25, "trend": 0.25, "vol_penalty": 0.10},
    "us_discretionary":  {"momentum": 0.44, "reversal": 0.15, "trend": 0.31, "vol_penalty": 0.10},
    "us_communications": {"momentum": 0.49, "reversal": 0.10, "trend": 0.31, "vol_penalty": 0.10},
    "us_financials":     {"momentum": 0.42, "reversal": 0.20, "trend": 0.28, "vol_penalty": 0.10},
    "us_energy":         {"momentum": 0.37, "reversal": 0.25, "trend": 0.28, "vol_penalty": 0.10},
    "us_industrials":    {"momentum": 0.42, "reversal": 0.20, "trend": 0.28, "vol_penalty": 0.10},
    "us_materials":      {"momentum": 0.37, "reversal": 0.25, "trend": 0.28, "vol_penalty": 0.10},
    "us_housing":        {"momentum": 0.42, "reversal": 0.20, "trend": 0.28, "vol_penalty": 0.10},
    "us_healthcare":     {"momentum": 0.42, "reversal": 0.20, "trend": 0.28, "vol_penalty": 0.10},
    "us_staples":        {"momentum": 0.32, "reversal": 0.25, "trend": 0.28, "vol_penalty": 0.15},
    "us_utilities":      {"momentum": 0.32, "reversal": 0.25, "trend": 0.28, "vol_penalty": 0.15},
    "tech":              {"momentum": 0.49, "reversal": 0.10, "trend": 0.31, "vol_penalty": 0.10},
    "real_estate":       {"momentum": 0.37, "reversal": 0.25, "trend": 0.28, "vol_penalty": 0.10},
    "thematic":          {"momentum": 0.42, "reversal": 0.20, "trend": 0.28, "vol_penalty": 0.10},
    "commodities":       {"momentum": 0.35, "reversal": 0.25, "trend": 0.30, "vol_penalty": 0.10},
    "fixed_income":      {"momentum": 0.30, "reversal": 0.25, "trend": 0.35, "vol_penalty": 0.10},
    "crypto":            {"momentum": 0.54, "reversal": 0.05, "trend": 0.36, "vol_penalty": 0.05},
}

# COVARIANZA ROBUSTA

COV_SHORT_WINDOW = 63
COV_LONG_WINDOW  = 252
# 0.78: calibrado cross-régimen (cov reactiva al entrar en las caídas, coherente con la
# tesis contrarian). No es un valor "especial": con el núcleo corregido la config adoptada
# rankea ~mediana entre carteras aleatorias y afinarlo no da mejora OOS robusta (ver Validación).
COV_BLEND_ALPHA  = 0.78
EWMA_LAMBDA      = 0.94
COV_SHRINKAGE    = 0.10

# BLACK-LITTERMAN

# Inerte: Omega sale proporcional a tau (build_view_uncertainty) y tau se cancela en el
# posterior, asi que mu_BL no depende de su valor. Fuera del nucleo a optimizar.
BL_TAU    = 0.05
BL_DELTA  = 2.5
# 0.37: calibrado cross-régimen junto a COV_BLEND_ALPHA (views contrarian más intensas,
# coherente con la tesis). Mismo veredicto: ~mediana en la nube de monos, sin ventaja OOS
# robusta al afinarlo (el signo del OOS cambia con el presupuesto de búsqueda).
VIEW_SCALE = 0.37

# Prior BL: "equal" (1/N) o "inv_vol" (1/sigma_i / sum) como proxy de mercado
BL_PRIOR_WEIGHTS_MODE = "inv_vol"

# REGIMEN — FILOSOFIA CONTRARIAN
# Umbral legacy: vol media cross-sectional anualizada
VOL_CAUTION_THR    = 0.28
VOL_CRISIS_THR     = 0.40
# Multi-factor (ademas de vol_media): cartera EW ultimos REGIME_LOOKBACK_DD dias
REGIME_LOOKBACK_DD = 252
REGIME_EW_MDD_CRISIS = 0.28
REGIME_EW_MDD_CAUTION = 0.16
REGIME_CORR_LOOKBACK = 63
REGIME_AVG_CORR_CRISIS = 0.52
REGIME_AVG_CORR_CAUTION = 0.38
CRISIS_VIEW_BOOST  = 1.30
CAUTION_VIEW_BOOST = 1.15
DN_CRISIS_MULT     = 2.0
DN_CAUTION_MULT    = 1.5

# MERTON

MERTON_GAMMA      = -0.8
MERTON_N_TOP      = 20
MERTON_MAX_WEIGHT = 0.40
# Peso minimo por posicion; run_merton lo pasa a apply_constraints.
MERTON_MIN_WEIGHT = 0.01
MERTON_MAX_SECTOR = 0.35

MAX_SECTOR_OVERRIDE = {}  # Sin overrides, MAX_SECTOR aplica uniformemente

# DAVIS-NORMAN

DN_BAND     = 0.05
DN_MIN_BAND = 0.02

# COSTES DE TRANSACCION (por lado, fraccion del nominal)
# Rango tipico bróker barato para ETFs: ~0,05%–0,12% por compra o venta.
# La version antigua (0.5*spread$/precio) disparaba el coste en ETFs baratos.

TX_COST_PER_SIDE = 0.0008
TX_COST_PER_SIDE_MIN = 0.0005
TX_COST_PER_SIDE_MAX = 0.0012

# Excel en data/: tabla ticker + comision (fraccion nominal por lado; coma o punto).
# None o "" = solo TX_COST_* homogeneo.
ETF_COMMISSIONS_EXCEL_PATH = "src/data/comisiones_etfs.xlsx"
# "table" = dos columnas (ticker, comision), filas. "horizontal" = layout legacy 3 filas.
ETF_COMMISSIONS_FORMAT = "table"
# Hoja: indice 0 o nombre (ej. "comisiones_etfs").
ETF_COMMISSIONS_SHEET = 0
# True = error si algun ticker descargado no tiene fila en el Excel.
ETF_COMMISSION_REQUIRE_EXCEL_FOR_ALL = False

# BACKTEST

# Solo afecta al MOTOR DE BACKTEST (engine): en que fechas se *evalua* el pipeline.
# No limita a Davis-Norman (las bandas son independientes del calendario).
# En vivo (run_live.py) no hay "una vez al mes" salvo que ejecutes main solo ese dia.
REBALANCE_FREQ   = "ME"
INITIAL_CAPITAL  = 10_000_000   # 10 M EUR clavados (capital real de arranque del live)
BENCHMARK_TICKER = "SPY"
# ETF liquido que replica MSCI World (USD); alternativa europea: SWDA.L
BENCHMARK_MSCI_WORLD_TICKER = "URTH"

# Ventanas del backtest. El veredicto vs SP500 depende de la ventana, asi que
# estas fechas son load-bearing: viven aqui, no duplicadas como literal en los
# scripts. Los tres rangos terminan en BACKTEST_DATA_END.
BACKTEST_DATA_END = "2026-05-14"   # ultima fecha de datos del informe (yfinance)
# 1) Canonica (~12.75 anios). Empieza 18 meses tras el inicio de URTH (feb-2012)
# para que el warmup del motor caiga sobre datos reales; mide desde 2013-08.
CANONICAL_START   = "2013-08-01"
CANONICAL_END     = BACKTEST_DATA_END
# 2) Sub-periodo destacado: post-COVID (donde la estrategia bate tambien al SP500).
SUBPERIOD_START   = "2020-01-01"
SUBPERIOD_END     = BACKTEST_DATA_END
# 3) Periodo en vivo / contrafactual: desarrollo real de la estrategia (~2 meses).
# Ventana corta: las metricas anualizadas (Sharpe/CAGR) NO son significativas; sirve
# para superponer con la cartera real (outputs/live/ -> outputs/backtest/counterfactual/).
LIVE_START        = "2026-03-12"
LIVE_END          = BACKTEST_DATA_END

# REGISTRADOR (operativa al ejecutar main)
# Misma logica que backtest/engine y walkforward (run_backtest -> engine):
#   - Pesos: dn['final_weights_full']
#   - Operar si: dn['rebalance'] OR cartera sin posiciones (portfolio.rebalance_policy)
# REGISTRADOR_MATCH_ENGINE_REBALANCE_RULE: si True, no generar ordenes Excel cuando
# el engine tampoco operaria (hay posiciones y DN no rebalancea).
REGISTRADOR_MATCH_ENGINE_REBALANCE_RULE = True
# Salida operativa (mismas columnas que antes: ID, Cantidad, Precio, CT, Precio Ejecutado).
# Si no usas {date}, el nombre es fijo (p. ej. sobrescribe cada ejecucion).
REGISTRADOR_OUTPUT_TEMPLATE = "outputs/live/orders.xlsx"
# Primera hoja del Excel (antes "Ordenes").
REGISTRADOR_ORDENES_SHEET_NAME = "Operativa"

# Cartera real (titulos): Excel fuente de verdad para main.run_single cuando
# current_positions no se pasa explicitamente. Columnas: Ticker, Cantidad.
POSITIONS_EXCEL_PATH = "outputs/live/positions.xlsx"
CREATE_POSITIONS_TEMPLATE_IF_MISSING = True
# Tras cada ejecucion: posiciones teoricas post-rebalanceo (titulos) — archivo
# historizado por fecha; no sustituye positions.xlsx hasta confirmar en bróker.
SAVE_SUGGESTED_POSITIONS = True
POSICIONES_POST_REBALANCEO_TEMPLATE = "outputs/live/snapshots/positions_{date}.xlsx"
# Alias retrocompatible:
SUGGESTED_POSITIONS_TEMPLATE = POSICIONES_POST_REBALANCEO_TEMPLATE
# Copia siempre la ultima ejecucion (mismo contenido que la fecha de hoy).
POSICIONES_ULTIMO_SNAPSHOT_PATH = "outputs/live/snapshots/positions_latest.xlsx"

# Append CSV de cada ejecucion de main.run_single (auditoria).
APPEND_EJECUCION_LOG = True
EJECUCION_LOG_CSV = "outputs/live/execution_log.csv"

# Informe cartera viva (track_live_portfolio.py): reconstruccion desde operaciones_*.xlsx
PORTFOLIO_LIVE_RESULTS_DIR = "outputs/live"
PORTFOLIO_LIVE_INITIAL_POSITIONS = None  # Excel opcional antes del primer archivo de ordenes
PORTFOLIO_LIVE_START_DATE = None  # "YYYY-MM-DD" o None = primera fecha de operaciones
PORTFOLIO_LIVE_INFORME_SUBDIR = "informe_cartera_vivo"

# Correo opcional (operaciones_rebalanceo): definir en entorno o dejar vacio.
# SMTP_HOST, SMTP_PORT (default 587), SMTP_USER, SMTP_PASSWORD,
# MAIL_FROM, MAIL_TO (varios separados por coma), SMTP_USE_TLS (default 1).
EMAIL_OPERACIONES_AFTER_RUN = False

# WALKFORWARD

WF_TRAIN_MONTHS = 12
WF_TEST_MONTHS  = 3
