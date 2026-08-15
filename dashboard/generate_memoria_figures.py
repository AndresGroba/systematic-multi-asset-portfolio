"""Regenera de forma determinista las figuras de la memoria que dependen de datos:
 - Seccion 11 (cartera viva): 11 figuras desde la reconstruccion CORREGIDA del
   dashboard (conversion EUR + fix XEON fase 3). Las originales eran exports ad-hoc
   de un notebook no versionado, con datos antiguos inflados.
 - Backtest canonico 2013-2026: curva, drawdown y timeline de regimen (outputs/backtest/canonical).
 - Calibracion/optimizacion de parametros: distribucion nula de parametros (monos),
   distribucion nula de seleccion (dardos de Malkiel, buy-and-hold), importancia RF,
   sensibilidad OAT, frecuencia de rebalanceo y Sharpe OOS (outputs/studies/*).
Reutiliza plots.py / strategy_plots.py; deja la generacion versionada y reproducible.

Uso:
    cd dashboard && .venv/bin/python generate_memoria_figures.py
(Si los .so del venv dan "library load disallowed by system policy" -- cuarentena
de Gatekeeper, p. ej. tras sincronizar la carpeta por iCloud-- limpiar con
`xattr -cr .venv`. kaleido esta pineado en requirements.txt.)
Salida:
    ../memoria/figures/*.png  (sobrescribe las 11 figuras de la seccion 11)
No usa scipy (KDE manual + statistics.NormalDist), para no depender de su build.
"""

from __future__ import annotations

import sys
from pathlib import Path
from statistics import NormalDist

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

sys.path.insert(0, str(Path(__file__).resolve().parent))

import src.strategy_data as sd
import src.strategy_plots as sp
from src.data_loader import load_historial, load_operativa, load_operativa_fase3
from src.portfolio_engine import build_portfolio, download_prices
from src.metrics import compute_drawdown_series
from src.plots import _apply_base, plot_drawdown, plot_cumulative_costs
from src.utils import (
    CAPITAL_INICIAL,
    OPERATIVA_DATE,
    PHASE3_DATE,
    CATEGORY_COLORS,
    ticker_category,
    ticker_label,
)

LIVE_END = "2026-05-14"
BENCH_TICKER = "IWDA.AS"
BENCH_LABEL = "MSCI World (IWDA.AS)"
OUT = Path(__file__).resolve().parents[1] / "memoria" / "figures"

BLUE = "#2563EB"
GREEN = "#15803D"
RED = "#B91C1C"
GREEN_FILL = "rgba(21,128,61,0.16)"
RED_FILL = "rgba(185,28,28,0.16)"


def eur(v: float) -> str:
    return "€" + f"{v:,.0f}".replace(",", ".")


def _phase_dividers(fig: go.Figure) -> None:
    fig.add_vline(x=OPERATIVA_DATE, line=dict(color="#E8833A", width=1.3, dash="dash"))
    fig.add_vline(x=PHASE3_DATE, line=dict(color="#94A3B8", width=1.3, dash="dash"))


def _is_date(v) -> bool:
    return hasattr(v, "to_pydatetime") or isinstance(v, np.datetime64)


def _iso(v):
    return pd.Timestamp(v).strftime("%Y-%m-%d") if _is_date(v) else v


def _sanitize_dates(fig: go.Figure) -> bool:
    """orjson (kaleido) no serializa pd.Timestamp: pasar toda fecha a string ISO."""
    had = False
    for tr in fig.data:
        x = getattr(tr, "x", None)
        if x is not None and len(x) and _is_date(x[0]):
            tr.x = [_iso(v) for v in x]
            had = True
    for sh in fig.layout.shapes:
        for attr in ("x0", "x1"):
            v = getattr(sh, attr, None)
            if _is_date(v):
                setattr(sh, attr, _iso(v))
                had = True
    for an in fig.layout.annotations:
        v = getattr(an, "x", None)
        if _is_date(v):
            an.x = _iso(v)
            had = True
    return had


def _save(fig: go.Figure, name: str, w: int, h: int) -> None:
    path = OUT / f"{name}.png"
    fig.update_layout(paper_bgcolor="white", plot_bgcolor="white")
    if _sanitize_dates(fig):
        fig.update_xaxes(type="date")
    fig.write_image(str(path), width=w, height=h, scale=2)
    print(f"  ok  {name}.png")


# ---------------------------------------------------------------- data
def build():
    hist = load_historial()
    op2 = load_operativa()
    op3 = load_operativa_fase3()
    pf, wl, _ = build_portfolio(hist, op2, op3, end_date=LIVE_END)
    idx = pf["nav"].dropna().index
    s = (idx[0] - pd.Timedelta(days=7)).strftime("%Y-%m-%d")
    e = (idx[-1] + pd.Timedelta(days=2)).strftime("%Y-%m-%d")
    px, _ = download_prices([BENCH_TICKER], s, e)
    full = px[BENCH_TICKER].dropna()
    price = full.reindex(full.index.union(idx)).ffill().reindex(idx)
    bm_nav = CAPITAL_INICIAL * price / price.iloc[0]
    return pf, wl, bm_nav


# ---------------------------------------------------------------- figures
def fig_valor_por_fases(pf, bm_nav):
    nav = pf["nav"].dropna()
    bm = bm_nav.reindex(nav.index).ffill()
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=bm.index, y=bm.values / 1e6, name=BENCH_LABEL,
                             line=dict(color="rgba(71,85,105,0.85)", width=1.8, dash="dash")))
    fig.add_trace(go.Scatter(x=nav.index, y=nav.values / 1e6, name="Cartera",
                             line=dict(color=BLUE, width=2.6)))
    _phase_dividers(fig)
    fig.add_annotation(x=nav.index[-1], y=nav.iloc[-1] / 1e6, text=eur(nav.iloc[-1]),
                       showarrow=False, xanchor="left", xshift=6, font=dict(color=BLUE, size=11))
    fig.add_annotation(x=bm.index[-1], y=bm.iloc[-1] / 1e6, text=eur(bm.iloc[-1]),
                       showarrow=False, xanchor="left", xshift=6, font=dict(color="#475569", size=11))
    # bandas de fase (etiquetas inferiores)
    bands = [
        (nav.index[0], OPERATIVA_DATE, "Fase 1 · Modelo base", "Merton + bandas"),
        (OPERATIVA_DATE, PHASE3_DATE, "Fase 2 · Multi-asset", "Señales + Black-Litterman"),
        (PHASE3_DATE, nav.index[-1], "Fase 3 · IA + rebalanceo", "XGBoost + clustering"),
    ]
    for a, b, t1, t2 in bands:
        mid = a + (b - a) / 2
        fig.add_annotation(x=mid, y=-0.12, yref="paper", showarrow=False,
                           text=f"<b>{t1}</b><br><span style='font-size:9px;color:#94A3B8'>{t2}</span>",
                           font=dict(size=11, color="#334155"))
    _apply_base(fig, title="Comparación de valor liquidativo con MSCI World (IWDA.AS)", height=470)
    fig.update_yaxes(title_text="Millones de euros", tickformat=".1f")
    fig.update_layout(legend=dict(orientation="h", x=0, y=1.06, font=dict(size=11)),
                      margin=dict(b=90))
    _save(fig, "benchmark_comparacion_valor_por_fases", 1200, 580)


def fig_caidas(pf, bm_nav):
    nav = pf["nav"].dropna()
    dd = compute_drawdown_series(nav)
    bm_dd = compute_drawdown_series(bm_nav.reindex(nav.index).ffill())
    fig = plot_drawdown(dd, bm_dd, BENCH_LABEL)
    fig.update_layout(title=dict(text="Comparación de caídas desde máximos frente al MSCI World"))
    _save(fig, "benchmark_comparacion_caidas", 1200, 460)


def fig_rentabilidad_relativa(pf, bm_nav):
    nav = pf["nav"].dropna()
    bm = bm_nav.reindex(nav.index).ffill()
    rel = ((nav / nav.iloc[0]) - (bm / bm.iloc[0])) * 100.0
    pos = rel.clip(lower=0)
    neg = rel.clip(upper=0)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=rel.index, y=pos.values, fill="tozeroy", fillcolor=GREEN_FILL,
                             line=dict(width=0), hoverinfo="skip", showlegend=False))
    fig.add_trace(go.Scatter(x=rel.index, y=neg.values, fill="tozeroy", fillcolor=RED_FILL,
                             line=dict(width=0), hoverinfo="skip", showlegend=False))
    fig.add_trace(go.Scatter(x=rel.index, y=rel.values, line=dict(color=BLUE, width=2.4),
                             name="Exceso acumulado",
                             hovertemplate="<b>%{x|%d/%m/%Y}</b><br>Exceso: %{y:+.2f}%<extra></extra>"))
    fig.add_hline(y=0, line_width=1, line_color="#94A3B8")
    _phase_dividers(fig)
    fig.add_annotation(x=rel.index[-1], y=rel.iloc[-1], text=f"{rel.iloc[-1]:+.2f} %".replace(".", ","),
                       showarrow=False, xanchor="left", xshift=8, font=dict(color=BLUE, size=12))
    _apply_base(fig, title="Rentabilidad relativa acumulada frente a MSCI World (IWDA.AS)", height=470)
    fig.update_yaxes(title_text="Exceso acumulado", ticksuffix="%")
    _save(fig, "benchmark_rentabilidad_relativa", 1200, 560)


def fig_treemap(wl, date, fase, subt, name):
    df = wl[wl["date"] == pd.Timestamp(date)].copy()
    df = df[df["weight"] > 0]
    df["category"] = df["ticker"].map(ticker_category)
    df["label"] = df["ticker"].map(lambda t: ticker_label(t, include_ticker=True))
    df = df.sort_values("weight", ascending=False)
    cat_tot = df.groupby("category", as_index=False)["weight"].sum().sort_values("weight", ascending=False)

    labels, ids, parents, values, colors, text = ["Cartera"], ["root"], [""], [float(df["weight"].sum())], ["#FFFFFF"], [""]
    for r in cat_tot.itertuples(index=False):
        labels.append(f"{r.category}")
        ids.append(f"cat|{r.category}")
        parents.append("root")
        values.append(float(r.weight))
        colors.append(CATEGORY_COLORS.get(r.category, "#64748B"))
        text.append(f"<b>{r.category} ({r.weight:.1%})</b>".replace(".", ","))
    for r in df.itertuples(index=False):
        labels.append(r.label)
        ids.append(f"etf|{r.category}|{r.ticker}")
        parents.append(f"cat|{r.category}")
        values.append(float(r.weight))
        colors.append(CATEGORY_COLORS.get(r.category, "#64748B"))
        text.append(f"{r.label}<br>{r.weight:.1%}".replace(".", ","))
    fig = go.Figure(go.Treemap(
        labels=labels, ids=ids, parents=parents, values=values, branchvalues="total",
        marker=dict(colors=colors, line=dict(color="#FFFFFF", width=2)),
        text=text, textinfo="text", textfont=dict(size=14, color="#FFFFFF"),
        pathbar=dict(visible=False), tiling=dict(pad=4), sort=True,
        hovertemplate="<b>%{label}</b><br>Peso: %{value:.2%}<extra></extra>",
    ))
    d = pd.Timestamp(date).strftime("%d/%m/%Y")
    _apply_base(fig, title=f"Composición de cartera · {fase} · {subt} · {d}", height=720)
    fig.update_layout(margin=dict(l=12, r=12, t=60, b=12))
    _save(fig, name, 1320, 760)


def _phase_contrib_buyhold(pf, regime):
    """Contribucion EUR por ticker en una fase sin trades intermedios (fase 2/3):
    delta de valor entre el primer y ultimo dia de la fase."""
    sub = pf[pf["regime"] == regime]
    a, b = sub.iloc[0], sub.iloc[-1]
    out = {}
    for c in pf.columns:
        if c.endswith("_value"):
            t = c.replace("_value", "")
            x, y = a.get(c, np.nan), b.get(c, np.nan)
            if pd.notna(x) and pd.notna(y):
                out[t] = float(y - x)
    return out


def _phase1_contrib(pf):
    """Fase 1 (IUSE+XEON con alpha variable): PnL por activo = sum(value_{t-1} * ret_t)
    usando precios reales EUR (ambos cotizan en EUR)."""
    sub = pf[pf["regime"] == "phase1"]
    px, _ = download_prices(["IUSE.L", "XEON.DE"], "2026-03-01", "2026-04-15")
    out = {}
    for t in ["IUSE.L", "XEON.DE"]:
        vcol = f"{t}_value"
        if vcol not in sub.columns or t not in px.columns:
            continue
        prices = px[t].reindex(sub.index).ffill()
        ret = prices.pct_change()
        val = sub[vcol]
        contrib = float((val.shift(1) * ret).sum())
        out[t] = contrib
    return out


def fig_atribucion(contrib, title, name):
    items = sorted(contrib.items(), key=lambda kv: kv[1])
    ys = [ticker_label(t, include_ticker=True) for t, _ in items]
    xs = [v for _, v in items]
    colors = [GREEN if v >= 0 else RED for v in xs]
    fig = go.Figure(go.Bar(x=xs, y=ys, orientation="h", marker_color=colors,
                           hovertemplate="%{y}<br>%{x:+,.0f} €<extra></extra>"))
    fig.add_vline(x=0, line_width=0.8, line_color="#CBD5E1")
    _apply_base(fig, title=title, height=max(360, 28 * len(items) + 120))
    fig.update_xaxes(title_text="Contribución al resultado (EUR)", tickformat=",.0f")
    _save(fig, name, 1100, max(380, 30 * len(items) + 140))


def fig_costes(pf):
    fig = plot_cumulative_costs(pf)
    fig.update_layout(title=dict(text="Costes de transacción acumulados"))
    _save(fig, "costes_acumulados_cartera", 1200, 460)


def fig_distribucion(pf):
    r = pf["nav"].dropna().pct_change().dropna() * 100.0
    rv = r.values
    mean, std, skew = float(rv.mean()), float(rv.std(ddof=1)), float(pd.Series(rv).skew())
    fig = make_subplots(rows=1, cols=2, column_widths=[0.5, 0.5],
                        subplot_titles=("", ""))
    # ---- left: histogram + manual KDE
    nbins = 9
    counts, edges = np.histogram(rv, bins=nbins)
    centers = (edges[:-1] + edges[1:]) / 2
    bw = edges[1] - edges[0]
    bar_colors = [GREEN if c >= 0 else RED for c in centers]
    fig.add_trace(go.Bar(x=centers, y=counts, width=bw * 0.95, marker_color=bar_colors,
                         opacity=0.75, showlegend=False, hoverinfo="skip"), row=1, col=1)
    h = 1.06 * std * len(rv) ** (-1 / 5)
    xs = np.linspace(rv.min() - 1, rv.max() + 1, 240)
    dens = np.exp(-0.5 * ((xs[:, None] - rv[None, :]) / h) ** 2).sum(axis=1) / (len(rv) * h * np.sqrt(2 * np.pi))
    fig.add_trace(go.Scatter(x=xs, y=dens * len(rv) * bw, line=dict(color=BLUE, width=2.2),
                             showlegend=False, hoverinfo="skip"), row=1, col=1)
    fig.add_vline(x=mean, line=dict(color=BLUE, width=1.4), row=1, col=1)
    fig.add_annotation(xref="x domain", yref="y domain", x=0.03, y=0.97, showarrow=False,
                       align="left", font=dict(size=12, color="#334155"),
                       text=(f"Media: {mean:+.2f} %<br>Desviación típica: {std:.2f} %<br>"
                             f"Asimetría: {skew:.2f}").replace(".", ","), row=1, col=1)
    fig.update_xaxes(title_text="Retorno diario (%)", row=1, col=1)
    fig.update_yaxes(title_text="Frecuencia", row=1, col=1)
    # ---- right: QQ-plot vs normal
    n = len(rv)
    nd = NormalDist()
    theo = np.array([mean + std * nd.inv_cdf((i + 0.5) / n) for i in range(n)])
    obs = np.sort(rv)
    fig.add_trace(go.Scatter(x=theo, y=obs, mode="markers",
                             marker=dict(color=BLUE, size=7, opacity=0.75),
                             showlegend=False, hoverinfo="skip"), row=1, col=2)
    lo, hi = min(theo.min(), obs.min()), max(theo.max(), obs.max())
    fig.add_trace(go.Scatter(x=[lo, hi], y=[lo, hi], mode="lines",
                             line=dict(color=RED, width=1.6, dash="dash"),
                             showlegend=False, hoverinfo="skip"), row=1, col=2)
    fig.update_xaxes(title_text="Cuantiles teóricos", row=1, col=2)
    fig.update_yaxes(title_text="Cuantiles observados (%)", row=1, col=2)
    _apply_base(fig, title="Distribución estadística de los retornos diarios", height=560)
    _save(fig, "distribucion_retornos_diarios_cartera", 1500, 650)


def fig_resultado_diario(pf):
    df = pf[["daily_pnl", "cum_pnl"]].dropna(subset=["daily_pnl"])
    colors = [GREEN if v >= 0 else RED for v in df["daily_pnl"]]
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Bar(x=df.index, y=df["daily_pnl"], marker_color=colors, name="Resultado diario",
                         hovertemplate="<b>%{x|%d/%m/%Y}</b><br>%{y:+,.0f} €<extra></extra>"),
                  secondary_y=False)
    fig.add_trace(go.Scatter(x=df.index, y=df["cum_pnl"], line=dict(color=BLUE, width=2.4),
                             name="Resultado acumulado",
                             hovertemplate="<b>%{x|%d/%m/%Y}</b><br>Acum.: %{y:+,.0f} €<extra></extra>"),
                  secondary_y=True)
    fig.add_hline(y=0, line_width=0.8, line_color="#CBD5E1")
    _apply_base(fig, title="Resultado diario y resultado acumulado", height=440)
    fig.update_yaxes(title_text="Resultado diario (EUR)", tickformat=",.0f", secondary_y=False)
    fig.update_yaxes(title_text="Acumulado (EUR)", tickformat=",.0f", secondary_y=True)
    fig.update_layout(legend=dict(orientation="h", x=0, y=1.06, font=dict(size=11)))
    _save(fig, "resultado_diario_cartera", 1100, 520)


def fig_contrafactual(pf):
    """Superpone la cartera REAL (despliegue incremental) con el CONTRAFACTUAL
    (estrategia final corriendo sobre la misma ventana viva) y los benchmarks,
    desde outputs/backtest/counterfactual/wealth_history.csv (rebased a 10 M)."""
    nav = pf["nav"].dropna()
    cf_path = Path(__file__).resolve().parents[1] / "outputs" / "backtest" / "counterfactual" / "wealth_history.csv"
    cf = pd.read_csv(cf_path, parse_dates=["Date"]).set_index("Date")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=cf.index, y=cf["SP500"] / 1e6, name="S&P 500",
                             line=dict(color="rgba(148,163,184,0.9)", width=1.5, dash="dot")))
    fig.add_trace(go.Scatter(x=cf.index, y=cf["MSCI_World (URTH)"] / 1e6, name="MSCI World",
                             line=dict(color="rgba(71,85,105,0.85)", width=1.6, dash="dash")))
    fig.add_trace(go.Scatter(x=nav.index, y=nav.values / 1e6, name="Cartera real (incremental)",
                             line=dict(color="#94A3B8", width=2.2)))
    fig.add_trace(go.Scatter(x=cf.index, y=cf["Strategy"] / 1e6, name="Estrategia final (contrafactual)",
                             line=dict(color=BLUE, width=2.8)))
    fig.add_annotation(x=cf.index[-1], y=cf["Strategy"].iloc[-1] / 1e6, text=eur(cf["Strategy"].iloc[-1]),
                       showarrow=False, xanchor="left", xshift=6, font=dict(color=BLUE, size=11))
    fig.add_annotation(x=nav.index[-1], y=nav.iloc[-1] / 1e6, text=eur(nav.iloc[-1]),
                       showarrow=False, xanchor="left", xshift=6, font=dict(color="#64748B", size=11))
    _apply_base(fig, title="Despliegue real vs estrategia final (contrafactual) sobre el periodo operativo",
                height=470)
    fig.update_yaxes(title_text="Millones de euros", tickformat=".2f")
    fig.update_layout(legend=dict(orientation="h", x=0, y=1.08, font=dict(size=11)))
    _save(fig, "contrafactual_overlay", 1200, 560)


def fig_backtest_canonico():
    """Backtest canonico 2013-2026 (outputs/backtest/canonical via strategy_data)."""
    w = sd.load_wealth("canonical")
    _save(sp.equity_curve(w), "backtest_canonico_curva", 1200, 500)
    _save(sp.drawdown(w, "MSCI_World (URTH)"), "backtest_canonico_drawdown", 1200, 400)
    _save(sp.regime_timeline(sd.load_resumen("canonical")), "backtest_canonico_regimen", 1200, 240)
    _save(sp.weights_area(sd.load_weights("canonical")), "backtest_canonico_pesos", 1200, 460)
    _save(sp.attribution_bar(sd.load_attribution_cumulative("canonical")),
          "backtest_canonico_atribucion", 1100, 600)


def fig_parametros():
    """Estudios de calibracion/optimizacion (outputs/studies via strategy_data)."""
    mk = sd.load_monkeys()["2013-2026"]
    ch = sd.load_monkeys_chosen()["2013-2026"]["chosen_Sharpe"]
    _save(sp.monkeys_hist(mk, ch), "params_distribucion_nula", 1100, 460)
    # Dardos de Malkiel: distribucion nula de SELECCION (carteras aleatorias del
    # universo). La memoria muestra la variante buy-and-hold del periodo largo.
    rp = sd.load_random_portfolios("buyhold")["2013-2026"]
    rp_sh = float(sd.load_random_chosen("buyhold")["2013-2026"]["chosen_Sharpe"])
    _save(sp.monkeys_hist(rp, chosen=rp_sh, trace_name="Carteras aleatorias",
                          unit="dardos",
                          title="Distribución nula de selección · dardos buy-and-hold (2013-2026)"),
          "dardos_buyhold", 1100, 440)
    _save(sp.param_importance_bar(sd.load_param_importance()), "params_importancia", 1100, 560)
    _save(sp.param_sensitivity_bar(sd.load_param_sensitivity()), "params_sensibilidad", 1100, 600)
    _save(sp.freq_compare_bar(sd.load_freq_compare()), "params_frecuencia", 1100, 440)
    _save(sp.params_oos_summary_bar(sd.load_params_oos_summary()), "params_oos", 1100, 340)


def main():
    print("Construyendo reconstrucción corregida (end=%s)..." % LIVE_END)
    pf, wl, bm_nav = build()
    OUT.mkdir(parents=True, exist_ok=True)
    print("Generando figuras en", OUT)
    fig_valor_por_fases(pf, bm_nav)
    fig_caidas(pf, bm_nav)
    fig_rentabilidad_relativa(pf, bm_nav)
    f1 = pf[pf["regime"] == "phase1"].index[-1]
    f2 = pf[pf["regime"] == "phase2"].index[-1]
    f3 = pf[pf["regime"] == "phase3"].index[-1]
    fig_treemap(wl, f1, "Fase 1", "Modelo base", "fase1_composicion_cartera")
    fig_treemap(wl, f2, "Fase 2", "Multi-activo", "fase2_composicion_cartera")
    fig_treemap(wl, f3, "Fase 3", "IA + rebalanceo", "fase3_composicion_cartera")
    fig_atribucion(_phase1_contrib(pf), "Atribución por ETF · Fase 1", "atribucion_etf_fase1")
    fig_atribucion(_phase_contrib_buyhold(pf, "phase2"), "Atribución por ETF · Fase 2", "atribucion_etf_fase2")
    fig_costes(pf)
    fig_distribucion(pf)
    fig_resultado_diario(pf)
    fig_contrafactual(pf)
    print("-- backtest canonico --")
    fig_backtest_canonico()
    print("-- parametros / optimizacion --")
    fig_parametros()
    print("Hecho: figuras regeneradas (seccion 11 + backtest canonico + parametros).")


if __name__ == "__main__":
    main()
