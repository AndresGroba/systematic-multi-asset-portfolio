"""
Gráficos plotly de la ESTRATEGIA (backtest + studies), en EUR.

Reutiliza el estilo del dashboard (`_apply_base`, paleta) y consume los DataFrames
de `strategy_data`. Todos los importes en EUR (los datos ya vienen en EUR).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from src.plots import _apply_base, _hex_to_rgba
from src.utils import CHART_COLORS, ticker_label

_BLUE = "#2563EB"
_GRAY = "#64748B"
_GREEN = "#15803D"
_RED = "#B91C1C"
_AMBER = "#B45309"
_REGIME_COLORS = {"normal": "#15803D", "caution": "#B45309", "crisis": "#B91C1C"}

# Macro-categoría por ticker para la composición del backtest (universo de 43).
# Fuente: src/config.py::ETF_UNIVERSE (grupos finos agregados a macro-categorías).
# El dashboard está desacoplado del paquete raíz, así que se replica aquí.
_STRATEGY_CATEGORY: dict[str, str] = {
    # Renta variable desarrollada
    "IWDA.L": "RV desarrollada", "IUSN.DE": "RV desarrollada", "VGK": "RV desarrollada",
    "EWJ": "RV desarrollada",
    # Renta variable emergente
    "IEMA.L": "RV emergente", "MCHI": "RV emergente", "INDA": "RV emergente",
    "EWT": "RV emergente", "EWY": "RV emergente", "EWZ": "RV emergente",
    # Sectorial EE.UU.
    "XLY": "Sectorial EE.UU.", "XLC": "Sectorial EE.UU.", "XLF": "Sectorial EE.UU.",
    "KBE": "Sectorial EE.UU.", "XLE": "Sectorial EE.UU.", "XOP": "Sectorial EE.UU.",
    "XLI": "Sectorial EE.UU.", "IYT": "Sectorial EE.UU.", "XLB": "Sectorial EE.UU.",
    "ITB": "Sectorial EE.UU.", "XLV": "Sectorial EE.UU.", "XBI": "Sectorial EE.UU.",
    "XLP": "Sectorial EE.UU.", "XLU": "Sectorial EE.UU.",
    # Tecnología
    "XLK": "Tecnología", "SOXX": "Tecnología", "IGV": "Tecnología",
    "CIBR": "Tecnología", "BOTZ": "Tecnología",
    # Resto de macro-grupos
    "VNQ": "Inmobiliario", "LIT": "Temático",
    "GLD": "Materias primas", "SLV": "Materias primas",
    "DBA": "Materias primas", "COPX": "Materias primas",
    "TLT": "Renta fija", "IEF": "Renta fija", "TIP": "Renta fija",
    "HYG": "Renta fija", "LQD": "Renta fija", "EMB": "Renta fija",
    "BITO": "Cripto", "XEON.DE": "Monetario (XEON)",
}
_STRATEGY_CATEGORY_COLORS: dict[str, str] = {
    "RV desarrollada": "#2563EB", "RV emergente": "#0F766E",
    "Sectorial EE.UU.": "#7C3AED", "Tecnología": "#0891B2",
    "Materias primas": "#D97706", "Renta fija": "#16A34A",
    "Inmobiliario": "#DB2777", "Temático": "#EA580C",
    "Cripto": "#92400E", "Monetario (XEON)": "#64748B",
    "Liquidez": "#CBD5E1", "Otros": "#94A3B8",
}


def _strategy_category(ticker: str) -> str:
    return _STRATEGY_CATEGORY.get(ticker, "Otros")


def _pct_to_float(s: pd.Series) -> pd.Series:
    """'12.99%' -> 12.99 (float)."""
    return pd.to_numeric(s.astype(str).str.replace("%", "", regex=False).str.replace(",", "", regex=False),
                         errors="coerce")


# ── Backtest ─────────────────────────────────────────────────────────────────

def equity_curve(wealth: pd.DataFrame) -> go.Figure:
    """NAV en EUR: estrategia vs benchmarks."""
    fig = go.Figure()
    styles = {
        "Strategy": (_BLUE, 2.6, "Estrategia"),
    }
    for col in wealth.columns:
        if col == "Strategy":
            continue
        styles[col] = (_GRAY, 1.6, col.replace("_", " "))
    # estrategia al final para que quede encima
    order = [c for c in wealth.columns if c != "Strategy"] + ["Strategy"]
    palette = iter([_GRAY, "#94A3B8", "#0EA5E9"])
    for col in order:
        if col == "Strategy":
            color, width, name = _BLUE, 2.6, "Estrategia"
        else:
            color, width, name = next(palette, _GRAY), 1.7, col.replace("_", " ").replace(" (URTH)", "")
        fig.add_trace(go.Scatter(
            x=wealth.index, y=wealth[col], name=name,
            line=dict(color=color, width=width),
            hovertemplate=f"{name}<br>%{{x|%d/%m/%Y}}<br>€%{{y:,.0f}}<extra></extra>",
        ))
    fig.update_yaxes(tickprefix="€", tickformat=",.0f")
    return _apply_base(fig, "Curva de valor (EUR)", height=440)


def drawdown(wealth: pd.DataFrame, benchmark_col: str | None = None) -> go.Figure:
    """Drawdown de la estrategia; opcionalmente superpone el de un benchmark."""
    def dd_pct(col: str) -> pd.Series:
        nav = wealth[col]
        return (nav / nav.cummax() - 1.0) * 100

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=wealth.index, y=dd_pct("Strategy"), name="Estrategia",
        fill="tozeroy", line=dict(color=_RED, width=1.4),
        fillcolor=_hex_to_rgba(_RED, 0.15),
        hovertemplate="Estrategia<br>%{x|%d/%m/%Y}<br>%{y:.1f}%<extra></extra>",
    ))
    if benchmark_col and benchmark_col in wealth.columns:
        name = benchmark_col.replace("_", " ").replace(" (URTH)", "")
        fig.add_trace(go.Scatter(
            x=wealth.index, y=dd_pct(benchmark_col), name=name,
            line=dict(color=_GRAY, width=1.5, dash="dot"),
            hovertemplate=f"{name}<br>%{{x|%d/%m/%Y}}<br>%{{y:.1f}}%<extra></extra>",
        ))
        fig.update_layout(legend=dict(orientation="h", yanchor="bottom", y=-0.28,
                                      xanchor="left", x=0))
    fig.update_yaxes(ticksuffix="%")
    title = "Drawdown: estrategia vs benchmark" if benchmark_col else "Drawdown de la estrategia"
    return _apply_base(fig, title, height=320)


def cumulative_costs(resumen: pd.DataFrame) -> go.Figure:
    """Evolución de los costes de transacción acumulados (EUR) del backtest."""
    fig = go.Figure()
    if resumen is None or "Trade Cost EUR" not in resumen.columns:
        return _apply_base(fig, "Costes de transacción acumulados (EUR)", height=300)
    df = resumen.sort_values("Date")
    cum = pd.to_numeric(df["Trade Cost EUR"], errors="coerce").fillna(0.0).cumsum()
    fig.add_trace(go.Scatter(
        x=df["Date"], y=cum, mode="lines", name="Costes acumulados",
        fill="tozeroy", fillcolor=_hex_to_rgba(_AMBER, 0.13),
        line=dict(color=_AMBER, width=2),
        hovertemplate="%{x|%d/%m/%Y}<br>Costes acum.: €%{y:,.0f}<extra></extra>",
    ))
    fig.update_yaxes(tickprefix="€", tickformat=",.0f")
    return _apply_base(fig, "Costes de transacción acumulados (EUR)", height=300)


def weights_area(weights: pd.DataFrame) -> go.Figure:
    """Área apilada de pesos en el tiempo, agregada por macro-categoría.

    El universo del backtest tiene 43 ETFs con rotación, así que un top-N por
    ticker dejaba un 'Resto' enorme. Agregar por categoría (como el Panel live)
    da bandas siempre coloreadas y legibles, sin cajón de sastre.
    """
    w = weights.copy()
    cat_of = {c: _strategy_category(c) for c in w.columns}
    area = w.T.groupby(cat_of).sum().T  # columnas = macro-categorías
    # Orden estable: por peso en la última fecha (la banda mayor abajo)
    order = area.iloc[-1].sort_values(ascending=False).index.tolist()
    fig = go.Figure()
    for cat in order:
        c = _STRATEGY_CATEGORY_COLORS.get(cat, _GRAY)
        fig.add_trace(go.Scatter(
            x=area.index, y=area[cat] * 100, name=cat, stackgroup="w",
            line=dict(width=0.5, color=c), fillcolor=_hex_to_rgba(c, 0.75),
            hovertemplate=f"{cat}<br>%{{x|%m/%Y}}<br>%{{y:.1f}}%<extra></extra>",
        ))
    fig.update_yaxes(ticksuffix="%", range=[0, 100])
    return _apply_base(fig, "Composición de la cartera en el tiempo (por categoría)", height=420)


def attribution_bar(attr_cum: pd.DataFrame, top_n: int = 12) -> go.Figure:
    """Contribución acumulada final por ETF (EUR), top ganadores y perdedores."""
    final = attr_cum.ffill().iloc[-1].dropna().sort_values()
    sel = pd.concat([final.head(top_n), final.tail(top_n)])
    sel = sel[~sel.index.duplicated()].sort_values()
    colors = [_GREEN if v >= 0 else _RED for v in sel.values]
    fig = go.Figure(go.Bar(
        x=sel.values, y=[ticker_label(t, include_ticker=True) for t in sel.index],
        orientation="h", marker_color=colors,
        hovertemplate="%{y}<br>€%{x:,.0f}<extra></extra>",
    ))
    fig.update_xaxes(tickprefix="€", tickformat=",.0f")
    return _apply_base(fig, "Atribución de resultado por ETF (EUR, acumulada)", height=520)


_REGIME_LABELS = {"normal": "Normal", "caution": "Cautela", "crisis": "Crisis"}


def regime_timeline(resumen: pd.DataFrame) -> go.Figure:
    """Franja temporal: cada tramo coloreado según su régimen (normal/cautela/crisis).

    Cada fecha de revisión (mensual) tiñe el tramo hasta la siguiente. Pensado para
    ir bajo la curva de valor (mismo eje temporal): se ve cuándo manda cada régimen.
    """
    fig = go.Figure()
    df = resumen[["Date", "Regime"]].dropna().sort_values("Date").reset_index(drop=True)
    if df.empty:
        return _apply_base(fig, "Régimen de mercado en el tiempo", height=170)

    dates = pd.to_datetime(df["Date"]).tolist()
    regimes = df["Regime"].astype(str).tolist()
    span = (dates[-1] - dates[-2]) if len(dates) > 1 else pd.Timedelta(days=30)
    seen: set[str] = set()

    def add_seg(start, end, reg):
        color = _REGIME_COLORS.get(reg, _GRAY)
        show = reg not in seen
        seen.add(reg)
        label = _REGIME_LABELS.get(reg, reg.capitalize())
        fig.add_trace(go.Scatter(
            x=[start, end, end, start, start], y=[0, 0, 1, 1, 0],
            fill="toself", fillcolor=_hex_to_rgba(color, 0.85),
            line=dict(width=0), mode="lines",
            name=label, legendgroup=reg, showlegend=show,
            hovertemplate=f"{label}<br>{start:%m/%Y} – {end:%m/%Y}<extra></extra>",
        ))

    seg_start, seg_reg = dates[0], regimes[0]
    for i in range(1, len(dates)):
        if regimes[i] != seg_reg:
            add_seg(seg_start, dates[i], seg_reg)
            seg_start, seg_reg = dates[i], regimes[i]
    add_seg(seg_start, dates[-1] + span, seg_reg)

    fig = _apply_base(fig, "Régimen de mercado en el tiempo", height=170)
    fig.update_yaxes(visible=False, range=[0, 1], fixedrange=True)
    fig.update_xaxes(showgrid=False)
    fig.update_layout(legend=dict(orientation="h", yanchor="bottom", y=-0.45,
                                  xanchor="left", x=0))
    return fig


# ── Studies ──────────────────────────────────────────────────────────────────

def ablation_bar(ablation: pd.DataFrame) -> go.Figure:
    """CAGR por variante (barra) + Sharpe anotado."""
    df = ablation.copy()
    cagr = _pct_to_float(df["CAGR"])
    sharpe = pd.to_numeric(df["Sharpe"], errors="coerce")
    fig = go.Figure(go.Bar(
        x=df["Variante"], y=cagr,
        marker_color=[_GRAY, "#60A5FA", _BLUE][: len(df)] if len(df) <= 3 else _BLUE,
        text=[f"CAGR {c:.2f}%<br>Sharpe {s:.2f}" for c, s in zip(cagr, sharpe)],
        textposition="outside",
        hovertemplate="%{x}<br>CAGR %{y:.2f}%<extra></extra>",
    ))
    fig.update_yaxes(ticksuffix="%")
    return _apply_base(fig, "Ablación ML — CAGR in-sample por variante", height=380)


def monkeys_hist(monkeys: pd.DataFrame, chosen: float | None = None,
                 trace_name: str = "Params aleatorios", unit: str = "monos",
                 title: str = "Distribución nula de parámetros (monos)") -> go.Figure:
    """Distribución nula del Sharpe + marca de la estrategia.

    Reutilizada por la nube de parámetros (monos) y por la de carteras aleatorias
    (dardos); `trace_name`/`unit`/`title` ajustan el texto según el experimento.
    """
    sh = pd.to_numeric(monkeys["Sharpe"], errors="coerce").dropna()
    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=sh, nbinsx=40, marker_color=_hex_to_rgba(_BLUE, 0.55),
        marker_line=dict(color=_BLUE, width=0.4), name=trace_name,
        hovertemplate=f"Sharpe %{{x:.2f}}<br>%{{y}} {unit}<extra></extra>",
    ))
    fig.add_vline(x=float(sh.median()), line=dict(color=_GRAY, width=1.4, dash="dot"),
                  annotation_text=f"mediana {unit} {sh.median():.2f}", annotation_position="top")
    if chosen is not None:
        pct = float((sh < chosen).mean() * 100.0)
        fig.add_vline(x=chosen, line=dict(color=_RED, width=2.2),
                      annotation_text=f"Estrategia {chosen:.2f} · percentil {pct:.0f}",
                      annotation_position="top right")
    fig.update_xaxes(title_text="Sharpe")
    return _apply_base(fig, title, height=380)


def monkeys_wealth_curves(wealth: pd.DataFrame, cloud_name: str = "carteras aleatorias",
                          median_name: str = "Mediana de los monos",
                          chosen_name: str = "Estrategia (params actuales)",
                          title: str = "Curvas de valor: estrategia vs carteras de parámetros aleatorios",
                          ) -> go.Figure:
    """Nube de curvas de valor: N carteras de la distribución nula (gris) + la nuestra.

    Reutilizada por la nube de parámetros (monos) y por la de carteras aleatorias
    (dardos); los `*_name`/`title` ajustan el texto. Las N curvas se dibujan como UNA
    traza concatenada (separadas por None) por rendimiento.
    """
    monkey_cols = [c for c in wealth.columns if c != "chosen"]
    n = len(monkey_cols)
    dates = list(wealth.index)

    # Una sola traza para las N curvas (None separa cada cartera)
    xs, ys = [], []
    for c in monkey_cols:
        xs.extend(dates + [None])
        ys.extend(wealth[c].tolist() + [None])

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=xs, y=ys, mode="lines", name=f"{n} {cloud_name}",
        line=dict(color=_GRAY, width=0.4), opacity=0.28,
        hoverinfo="skip",
    ))
    if monkey_cols:
        med = wealth[monkey_cols].median(axis=1)
        fig.add_trace(go.Scatter(
            x=dates, y=med, mode="lines", name=median_name,
            line=dict(color=_GRAY, width=1.6, dash="dot"),
            hovertemplate="Mediana<br>%{x|%m/%Y}<br>€%{y:,.0f}<extra></extra>",
        ))
    if "chosen" in wealth.columns:
        fig.add_trace(go.Scatter(
            x=dates, y=wealth["chosen"], mode="lines", name=chosen_name,
            line=dict(color=_BLUE, width=2.8),
            hovertemplate="Estrategia<br>%{x|%m/%Y}<br>€%{y:,.0f}<extra></extra>",
        ))
    fig.update_yaxes(tickprefix="€", tickformat=",.0f")
    return _apply_base(fig, title, height=440)


def param_importance_bar(pi: pd.DataFrame, top_n: int = 15) -> go.Figure:
    df = pi.copy().head(top_n).iloc[::-1]
    imp = pd.to_numeric(df["rf_importance"], errors="coerce")
    fig = go.Figure(go.Bar(
        x=imp, y=df["param"], orientation="h",
        marker_color=_BLUE,
        hovertemplate="%{y}<br>importancia %{x:.3f}<extra></extra>",
    ))
    return _apply_base(fig, "Importancia de parámetros (RF sobre el Sharpe)", height=480)


def param_sensitivity_bar(summ: pd.DataFrame, top_n: int = 18) -> go.Figure:
    """Span del Sharpe (max-min del barrido OAT) por parámetro; gris = casi inerte."""
    df = summ.copy().head(top_n).iloc[::-1]
    span = pd.to_numeric(df["Sharpe_span"], errors="coerce")
    colors = [_GRAY if s < 0.005 else _BLUE for s in span]
    cd = df[["Sharpe_min", "Sharpe_max", "argmax_value"]].astype(str).values
    fig = go.Figure(go.Bar(
        x=span, y=df["param"], orientation="h", marker_color=colors, customdata=cd,
        hovertemplate=("%{y}<br>span %{x:.3f}<br>Sharpe [%{customdata[0]}, %{customdata[1]}]"
                       "<br>óptimo @ %{customdata[2]}<extra></extra>"),
    ))
    return _apply_base(fig, "Sensibilidad OAT — span del Sharpe por parámetro", height=520)


def freq_compare_bar(freq: pd.DataFrame) -> go.Figure:
    """Sharpe por frecuencia, agrupado por periodo, con el SP500 de referencia."""
    df = freq.copy()
    fig = go.Figure()
    periods = list(df["periodo"].unique())
    colors = {periods[i]: [_BLUE, "#0EA5E9", _GRAY][i % 3] for i in range(len(periods))}
    for per in periods:
        sub = df[df["periodo"] == per]
        fig.add_trace(go.Bar(
            x=sub["frecuencia"], y=pd.to_numeric(sub["Sharpe"], errors="coerce"),
            name=str(per), marker_color=colors[per],
            hovertemplate=f"{per} %{{x}}<br>Sharpe %{{y:.2f}}<extra></extra>",
        ))
    # SP500 de referencia (media por periodo)
    if "SP500_Sharpe" in df.columns:
        ref = pd.to_numeric(df["SP500_Sharpe"], errors="coerce").mean()
        fig.add_hline(y=ref, line=dict(color=_RED, width=1.6, dash="dash"),
                      annotation_text=f"SP500 ≈ {ref:.2f}", annotation_position="top left")
    fig.update_layout(barmode="group")
    return _apply_base(fig, "Frecuencia de rebalanceo — Sharpe por periodo", height=380)


def params_oos_summary_bar(summary: pd.DataFrame) -> go.Figure:
    """Sharpe OOS concatenado por serie: fijos vs re-optimizados vs benchmarks."""
    df = summary.copy().iloc[::-1]   # primera fila arriba en la barra horizontal
    sh = pd.to_numeric(df["Sharpe_OOS"], errors="coerce")
    palette = {"Params fijos (adoptados)": _BLUE, "Params re-optimizados": "#93C5FD"}
    colors = [palette.get(s, _GRAY) for s in df["serie"]]
    fig = go.Figure(go.Bar(
        x=sh, y=df["serie"], orientation="h", marker_color=colors,
        text=[f"{v:.2f}" for v in sh], textposition="outside",
        hovertemplate="%{y}<br>Sharpe OOS %{x:.3f}<extra></extra>",
    ))
    return _apply_base(fig, "Sharpe OOS concatenado (todos los tramos)", height=300)


def params_oos_folds_bar(oos: pd.DataFrame) -> go.Figure:
    """Por fold: Sharpe en train (in-sample) vs en test (OOS). El hueco mide el sobreajuste."""
    df = oos.copy()
    folds = (df["test"] if "test" in df else df.get("fold")).astype(str)
    fig = go.Figure()
    fig.add_trace(go.Bar(x=folds, y=pd.to_numeric(df["train_Sharpe"], errors="coerce"),
                         name="train (in-sample)", marker_color=_GRAY,
                         hovertemplate="%{x}<br>train %{y:.2f}<extra></extra>"))
    fig.add_trace(go.Bar(x=folds, y=pd.to_numeric(df["test_Sharpe_sel"], errors="coerce"),
                         name="test (OOS)", marker_color=_BLUE,
                         hovertemplate="%{x}<br>OOS %{y:.2f}<extra></extra>"))
    fig.update_layout(barmode="group")
    return _apply_base(fig, "Por fold: Sharpe train vs OOS (params re-optimizados)", height=360)


def ml_walkforward_auc(wf: pd.DataFrame) -> go.Figure:
    """AUC OOS del filtro ML por fold, con la línea de azar (0.5). Azul ≥0.5, gris <0.5."""
    df = wf.copy()
    auc = pd.to_numeric(df["ML OOS AUC"], errors="coerce")
    x = (df["Fold"].astype(str) if "Fold" in df.columns
         else [str(i + 1) for i in range(len(df))])
    colors = [_BLUE if a >= 0.5 else _GRAY for a in auc]
    fig = go.Figure(go.Bar(
        x=x, y=auc, marker_color=colors,
        hovertemplate="fold %{x}<br>AUC %{y:.3f}<extra></extra>",
    ))
    fig.add_hline(y=0.5, line=dict(color=_RED, width=1.6, dash="dash"),
                  annotation_text="azar (0.5)", annotation_position="top left")
    fig.update_yaxes(title_text="AUC OOS", range=[0.35, 0.65])
    fig.update_xaxes(title_text="fold")
    return _apply_base(fig, "AUC OOS del filtro ML por fold", height=360)
