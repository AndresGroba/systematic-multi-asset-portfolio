"""Plotly chart builders. Shared light fintech aesthetic via _LAYOUT_BASE."""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go

from src.utils import (
    CATEGORY_COLORS,
    CHART_COLORS,
    OPERATIVA_DATE,
    PHASE3_DATE,
    ticker_category,
    ticker_label,
)

def _hex_to_rgba(hex_color: str, alpha: float = 0.7) -> str:
    """Convert #RRGGBB to rgba(r,g,b,alpha)."""
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"


_LAYOUT_BASE = dict(
    paper_bgcolor="#FFFFFF",
    plot_bgcolor="#FFFFFF",
    font=dict(family="Inter, sans-serif", color="#0F172A", size=12),
    margin=dict(l=60, r=30, t=50, b=50),
    legend=dict(
        bgcolor="rgba(255,255,255,0.94)",
        bordercolor="#E2E8F0",
        borderwidth=1,
        font=dict(size=11),
    ),
    hoverlabel=dict(
        bgcolor="#FFFFFF",
        bordercolor="#2563EB",
        font=dict(color="#0F172A", size=12),
    ),
    xaxis=dict(
        gridcolor="#E2E8F0",
        linecolor="#CBD5E1",
        tickcolor="#CBD5E1",
        zerolinecolor="#CBD5E1",
    ),
    yaxis=dict(
        gridcolor="#E2E8F0",
        linecolor="#CBD5E1",
        tickcolor="#CBD5E1",
        zerolinecolor="#CBD5E1",
    ),
)


def _apply_base(fig: go.Figure, title: str = "", height: int = 420) -> go.Figure:
    layout = dict(**_LAYOUT_BASE, height=height)
    if title:
        layout["title"] = dict(text=title, font=dict(size=15, color="#0F172A"), x=0.02)
    fig.update_layout(**layout)
    return fig


def _strategy_date_text(fmt: str = "%d/%m/%Y") -> str:
    """Return the formatted strategy transition date from the canonical constant."""
    return OPERATIVA_DATE.strftime(fmt)


def _strategy_change_annotation(fig: go.Figure, end_date: pd.Timestamp | None = None) -> go.Figure:
    """Add a consistent strategy transition marker using datetime values."""
    x1 = end_date if end_date is not None else OPERATIVA_DATE + pd.Timedelta(days=180)
    fig.add_vline(
        x=OPERATIVA_DATE,
        line_width=1.5,
        line_dash="dash",
        line_color="#B45309",
    )
    fig.add_vrect(
        x0=OPERATIVA_DATE,
        x1=x1,
        fillcolor="rgba(37,99,235,0.045)",
        layer="below",
        line_width=0,
    )
    fig.add_annotation(
        x=OPERATIVA_DATE,
        y=1,
        xref="x",
        yref="paper",
        text=f"Transición multi-activo · {_strategy_date_text()}",
        showarrow=False,
        xanchor="left",
        yanchor="bottom",
        xshift=8,
        yshift=6,
        font=dict(color="#B45309", size=11),
        bgcolor="rgba(255,251,235,0.96)",
        bordercolor="#F59E0B",
        borderwidth=1,
        borderpad=4,
    )
    # Segundo marcador: rebalanceo de fase 3 (13-may), si la ventana lo cubre.
    if end_date is None or pd.Timestamp(end_date) >= PHASE3_DATE:
        fig.add_vline(x=PHASE3_DATE, line_width=1.5, line_dash="dash", line_color="#7C3AED")
        fig.add_annotation(
            x=PHASE3_DATE, y=1, xref="x", yref="paper",
            text=f"Rebalanceo fase 3 · {PHASE3_DATE:%d/%m/%Y}",
            showarrow=False, xanchor="left", yanchor="bottom", xshift=8, yshift=6,
            font=dict(color="#7C3AED", size=11),
            bgcolor="rgba(245,243,255,0.96)", bordercolor="#A78BFA", borderwidth=1, borderpad=4,
        )
    return fig


def plot_equity_curve(
    portfolio_df: pd.DataFrame,
    bm_nav: "pd.Series | None" = None,
    bm_label: str = "Índice de referencia",
    bm_short_label: str = "Índice",
) -> go.Figure:
    """Two-phase NAV line chart with a focused Y axis. bm_nav adds a Buy & Hold reference."""
    fig = go.Figure()

    nav = portfolio_df["nav"].dropna()
    phase1 = portfolio_df[portfolio_df["regime"] == "phase1"]["nav"].dropna()
    # Etapa multi-activo = fases 2 y 3 (despliegue + rebalanceo): una sola línea continua.
    phase2 = portfolio_df[portfolio_df["regime"].isin(["phase2", "phase3"])]["nav"].dropna()

    if nav.empty:
        return fig

    if not phase1.empty:
        fig.add_trace(
            go.Scatter(
                x=phase1.index,
                y=phase1.values,
                mode="lines",
                name="Etapa inicial — IUSE + XEON",
                line=dict(color="#64748B", width=2.5),
                hovertemplate=(
                    "<b>%{x|%d/%m/%Y}</b><br>"
                    "Valor: <b>€%{y:,.2f}</b><br>"
                    "<i>Etapa inicial: IUSE + XEON</i>"
                    "<extra></extra>"
                ),
            )
        )

    if not phase1.empty and not phase2.empty:
        fig.add_trace(
            go.Scatter(
                x=[phase1.index[-1], phase2.index[0]],
                y=[phase1.iloc[-1], phase2.iloc[0]],
                mode="lines",
                name="Conector transición",
                line=dict(color="#64748B", width=2.5),
                hoverinfo="skip",
                showlegend=False,
            )
        )

    if not phase2.empty:
        p2_x = list(phase2.index)
        p2_y = list(phase2.values)

        n_p2 = len(phase2)
        mode = "lines+markers" if n_p2 <= 3 else "lines"
        marker_cfg = dict(
            size=11 if n_p2 <= 3 else 0,
            color="#2563EB",
            symbol="circle",
            line=dict(color="#FFFFFF", width=1.8),
        )

        fig.add_trace(
            go.Scatter(
                x=p2_x,
                y=p2_y,
                mode=mode,
                name="Etapa multi-activo",
                line=dict(color="#2563EB", width=3),
                marker=marker_cfg,
                hovertemplate=(
                    "<b>%{x|%d/%m/%Y}</b><br>"
                    "Valor: <b>€%{y:,.2f}</b><br>"
                    "<i>Etapa multi-activo</i>"
                    "<extra></extra>"
                ),
            )
        )

    # Benchmark trace, rendered behind portfolio lines
    if bm_nav is not None and not bm_nav.empty:
        bm_aligned = bm_nav.reindex(nav.index).ffill()
        fig.add_trace(
            go.Scatter(
                x=bm_aligned.index,
                y=bm_aligned.values,
                mode="lines",
                name=f"Referencia: {bm_short_label}",
                line=dict(color="rgba(71,85,105,0.72)", width=1.9, dash="dot"),
                hovertemplate=(
                    "<b>%{x|%d/%m/%Y}</b><br>"
                    "Valor referencia: <b>€%{y:,.2f}</b><br>"
                    f"<i>{bm_label}</i>"
                    "<extra></extra>"
                ),
            )
        )

    nav_min = float(nav.min())
    nav_max = float(nav.max())
    nav_mean = float(nav.mean())
    span = nav_max - nav_min
    pad = max(span * 0.12, nav_mean * 0.003)

    # Expand Y range if benchmark goes outside portfolio range
    if bm_nav is not None and not bm_nav.empty:
        bm_aligned = bm_nav.reindex(nav.index).ffill()
        nav_min = min(nav_min, float(bm_aligned.min()))
        nav_max = max(nav_max, float(bm_aligned.max()))
        span = nav_max - nav_min
        pad = max(span * 0.12, nav_mean * 0.003)

    y_range = [nav_min - pad, nav_max + pad]

    fig.add_hline(
        y=10_000_000,
        line_width=1,
        line_dash="dot",
        line_color="rgba(150,150,150,0.4)",
        annotation_text="Inicio €10M",
        annotation_font_color="rgba(150,150,150,0.7)",
        annotation_font_size=10,
        annotation_position="bottom right",
    )

    if not phase2.empty:
        _strategy_change_annotation(fig, end_date=nav.index.max())

    _apply_base(fig, title="Curva de capital", height=370)
    fig.update_yaxes(
        tickprefix="€",
        tickformat=",.0f",
        title_text="Valor (EUR)",
        range=y_range,
    )
    fig.update_xaxes(title_text="")
    fig.update_layout(
        legend=dict(
            orientation="h",
            x=0,
            y=-0.12,
            font=dict(size=11),
        )
    )
    return fig


def plot_daily_pnl(portfolio_df: pd.DataFrame) -> go.Figure:
    """Coloured bar chart of daily result."""
    df = portfolio_df[["daily_pnl", "regime"]].dropna(subset=["daily_pnl"])
    colors = df["daily_pnl"].apply(lambda v: "#15803D" if v >= 0 else "#B91C1C")

    fig = go.Figure(
        go.Bar(
            x=df.index,
            y=df["daily_pnl"],
            marker_color=colors,
            hovertemplate="<b>%{x|%d/%m/%Y}</b><br>Resultado: €%{y:+,.0f}<extra></extra>",
            name="Resultado diario",
        )
    )
    fig.add_hline(y=0, line_width=0.8, line_color="#CBD5E1")
    _apply_base(fig, title="Resultado diario", height=270)
    fig.update_yaxes(tickprefix="€", tickformat="+,.0f")
    return fig


def plot_cumulative_costs(portfolio_df: pd.DataFrame) -> go.Figure:
    """Area chart of cumulative transaction costs."""
    df = portfolio_df[["cum_cost"]].dropna()

    fig = go.Figure(
        go.Scatter(
            x=df.index,
            y=df["cum_cost"],
            mode="lines",
            fill="tozeroy",
            fillcolor="rgba(180,83,9,0.13)",
            line=dict(color="#B45309", width=2),
            hovertemplate="<b>%{x|%d/%m/%Y}</b><br>Costes acum.: €%{y:,.2f}<extra></extra>",
            name="Costes acumulados",
        )
    )
    _apply_base(fig, title="Costes de Transacción Acumulados", height=280)
    fig.update_yaxes(tickprefix="€", tickformat=",.0f")
    return fig


def plot_drawdown(
    dd_series: pd.Series,
    bm_dd: "pd.Series | None" = None,
    bm_label: str = "Índice",
) -> go.Figure:
    """Area chart of portfolio drawdown. When bm_dd is provided, overlays benchmark."""
    fig = go.Figure()

    if bm_dd is not None and not bm_dd.empty:
        fig.add_trace(
            go.Scatter(
                x=bm_dd.index,
                y=bm_dd.values * 100,
                mode="lines",
                fill="tozeroy",
                fillcolor="rgba(100,116,139,0.10)",
                line=dict(color="rgba(100,116,139,0.75)", width=1.5, dash="dot"),
                hovertemplate=(
                    "<b>%{x|%d/%m/%Y}</b><br>"
                    f"Caída {bm_label}: <b>%{{y:.2f}}%</b>"
                    "<extra></extra>"
                ),
                name=f"Referencia: {bm_label}",
            )
        )

    fig.add_trace(
        go.Scatter(
            x=dd_series.index,
            y=dd_series.values * 100,
            mode="lines",
            fill="tozeroy",
            fillcolor="rgba(185,28,28,0.13)",
            line=dict(color="#B91C1C", width=1.8),
            hovertemplate=(
                "<b>%{x|%d/%m/%Y}</b><br>"
                    "Caída: <b>%{y:.2f}%</b>"
                "<extra></extra>"
            ),
            name="Estrategia",
        )
    )

    title = f"Caídas desde máximos vs {bm_label}" if bm_dd is not None else "Caídas desde máximos"
    fig.add_hline(y=0, line_width=0.8, line_color="#CBD5E1")
    _apply_base(fig, title=title, height=270)
    fig.update_yaxes(ticksuffix="%", tickformat=".1f", title_text="Caída")
    if bm_dd is not None:
        fig.update_layout(
            legend=dict(orientation="h", x=0, y=-0.16, font=dict(size=11))
        )
    return fig


def _weights_with_categories(weights_long: pd.DataFrame) -> pd.DataFrame:
    """Return long weights enriched with category and display label columns."""
    if weights_long.empty:
        return weights_long.copy()

    df = weights_long.copy()
    df["category"] = df["ticker"].map(ticker_category)
    df["ticker_label"] = df["ticker"].map(lambda t: ticker_label(t, include_ticker=False))
    return df


def plot_composition_treemap(
    weights_long: pd.DataFrame,
    date: pd.Timestamp | None = None,
) -> tuple[go.Figure, pd.DataFrame]:
    """Treemap with macro (category) and micro (ETF) composition at a given date."""
    if date is None:
        date = weights_long["date"].max()

    day_weights = _weights_with_categories(weights_long)
    day_weights = day_weights[day_weights["date"] == date].sort_values("weight", ascending=False)
    if day_weights.empty:
        return go.Figure(), day_weights

    category_totals = (
        day_weights.groupby("category", as_index=False)[["weight", "value"]]
        .sum()
        .sort_values("weight", ascending=False)
    )
    total_weight = float(category_totals["weight"].sum())

    labels = ["Cartera"]
    ids = ["root"]
    parents = [""]
    values = [total_weight]
    colors = ["#FFFFFF"]
    customdata = [["Cartera total", "Resumen", day_weights["value"].sum()]]

    for row in category_totals.itertuples(index=False):
        labels.append(row.category)
        ids.append(f"cat|{row.category}")
        parents.append("root")
        values.append(row.weight)
        colors.append(CATEGORY_COLORS.get(row.category, "#78909C"))
        customdata.append([row.category, "Categoría", row.value])

    for row in day_weights.itertuples(index=False):
        labels.append(row.ticker)
        ids.append(f"etf|{row.category}|{row.ticker}")
        parents.append(f"cat|{row.category}")
        values.append(row.weight)
        colors.append(CATEGORY_COLORS.get(row.category, "#78909C"))
        customdata.append([row.ticker_label, row.category, row.value])

    fig = go.Figure(
        go.Treemap(
            labels=labels,
            ids=ids,
            parents=parents,
            values=values,
            branchvalues="total",
            marker=dict(
                colors=colors,
                line=dict(color="#FFFFFF", width=1),
            ),
            texttemplate="<b>%{label}</b><br>%{value:.1%}",
            customdata=customdata,
            hovertemplate=(
                "<b>%{customdata[0]}</b><br>"
                "Categoría: %{customdata[1]}<br>"
                "Peso: %{value:.2%}<br>"
                "Valor: €%{customdata[2]:,.0f}<extra></extra>"
            ),
            pathbar=dict(visible=False),
            tiling=dict(pad=4),
        )
    )

    _apply_base(fig, title=f"Composición actual por categoría y ETF · {date.strftime('%d/%m/%Y')}", height=430)
    fig.update_layout(
        margin=dict(l=20, r=20, t=55, b=20),
    )
    return fig, day_weights


def plot_allocation_donut(
    weights_long: pd.DataFrame,
    date: pd.Timestamp | None = None,
) -> go.Figure:
    """Donut chart with current allocation by understandable asset category."""
    if weights_long.empty:
        return go.Figure()

    if date is None:
        date = weights_long["date"].max()

    day_weights = _weights_with_categories(weights_long)
    day_weights = day_weights[day_weights["date"] == date].copy()
    if day_weights.empty:
        return go.Figure()

    category_totals = (
        day_weights.groupby("category", as_index=False)[["weight", "value"]]
        .sum()
        .sort_values("weight", ascending=False)
    )
    colors = [CATEGORY_COLORS.get(cat, "#64748B") for cat in category_totals["category"]]

    fig = go.Figure(
        go.Pie(
            labels=category_totals["category"],
            values=category_totals["weight"],
            hole=0.58,
            marker=dict(colors=colors, line=dict(color="#FFFFFF", width=2)),
            textinfo="none",
            hovertemplate=(
                "<b>%{label}</b><br>"
                "Peso: %{value:.2%}<br>"
                "Valor: €%{customdata:,.0f}<extra></extra>"
            ),
            customdata=category_totals["value"],
            sort=False,
        )
    )
    # % invertido = todo menos la caja (categoría Liquidez). El monetario XEON SÍ está
    # invertido, así que no se descuenta. Dinámico, no 100% fijo.
    total_w = float(category_totals["weight"].sum())
    liq_w = float(category_totals.loc[category_totals["category"] == "Liquidez", "weight"].sum())
    invested_pct = (total_w - liq_w) / total_w * 100 if total_w > 0 else 0.0
    fig.add_annotation(
        text=f"<b>{invested_pct:.0f}%</b><br><span style='font-size:11px'>invertido</span>",
        x=0.5,
        y=0.5,
        showarrow=False,
        font=dict(color="#0F172A", size=18),
    )
    _apply_base(fig, title="Asignación de activos", height=270)
    fig.update_layout(
        margin=dict(l=10, r=10, t=52, b=10),
        legend=dict(
            orientation="v",
            x=1.02,
            y=0.5,
            xanchor="left",
            yanchor="middle",
            font=dict(size=10),
            bgcolor="rgba(255,255,255,0)",
            borderwidth=0,
        ),
    )
    return fig


def plot_holdings_bar(
    weights_long: pd.DataFrame,
    category: str = "Todas",
    date: pd.Timestamp | None = None,
    top_n: int | None = None,
) -> go.Figure:
    """Horizontal ranking of individual ETFs, optionally filtered by category."""
    if date is None:
        date = weights_long["date"].max()

    day_weights = _weights_with_categories(weights_long)
    day_weights = day_weights[day_weights["date"] == date].copy()
    if category != "Todas":
        day_weights = day_weights[day_weights["category"] == category]
    day_weights = day_weights.sort_values("weight", ascending=True)
    if top_n is not None:
        day_weights = day_weights.tail(top_n)

    if day_weights.empty:
        return go.Figure()

    item_count = len(day_weights)
    dynamic_height = min(max(360, 42 * item_count + 160), 920)
    text_position = "outside" if item_count <= 12 else "auto"

    fig = go.Figure(
        go.Bar(
            x=day_weights["weight"] * 100,
            y=day_weights["ticker"],
            orientation="h",
            marker=dict(
                color=[CATEGORY_COLORS.get(cat, "#78909C") for cat in day_weights["category"]],
                line=dict(color="#FFFFFF", width=1),
            ),
            customdata=day_weights[["ticker_label", "category", "value"]].values,
            text=[f"{w * 100:.2f}%" for w in day_weights["weight"]],
            textposition=text_position,
            cliponaxis=False,
            hovertemplate=(
                "<b>%{y}</b> · %{customdata[0]}<br>"
                "Categoría: %{customdata[1]}<br>"
                "Peso: %{x:.2f}%<br>"
                "Valor: €%{customdata[2]:,.0f}<extra></extra>"
            ),
            name="Peso actual",
        )
    )
    if category == "Todas" and top_n is not None:
        scope = f"Principales posiciones por ETF"
    else:
        scope = "Ranking completo por ETF" if category == "Todas" else f"ETF — {category}"
    _apply_base(fig, title=scope, height=dynamic_height)
    max_weight = float((day_weights["weight"] * 100).max()) if not day_weights.empty else 0.0
    fig.update_xaxes(title_text="Peso en cartera (%)", range=[0, max_weight * 1.18 if max_weight else 1])
    fig.update_yaxes(title_text="")
    fig.update_layout(
        margin=dict(l=20, r=36, t=56, b=20),
        bargap=0.22 if item_count <= 12 else 0.14,
    )
    return fig


def plot_weights_area(weights_long: pd.DataFrame) -> go.Figure:
    """Stacked area chart of portfolio weight evolution by ETF category."""
    fig = go.Figure()

    categorized = _weights_with_categories(weights_long)
    pivot = categorized.pivot_table(
        index="date", columns="category", values="weight", aggfunc="sum"
    ).fillna(0)

    categories = sorted(
        pivot.columns.tolist(),
        key=lambda cat: pivot.iloc[-1].get(cat, 0) if not pivot.empty else 0,
        reverse=True,
    )

    for i, category in enumerate(categories):
        color = CATEGORY_COLORS.get(category, CHART_COLORS[i % len(CHART_COLORS)])
        fig.add_trace(
            go.Scatter(
                x=pivot.index,
                y=pivot[category] * 100,
                name=category,
                stackgroup="one",
                fillcolor=_hex_to_rgba(color, 0.7),
                line=dict(color=color, width=0.5),
                hovertemplate=f"<b>{category}</b><br>%{{x|%d/%m/%Y}}: %{{y:.1f}}% de la cartera<extra></extra>",
            )
        )

    if not categorized.empty:
        _strategy_change_annotation(fig, end_date=categorized["date"].max())

    _apply_base(fig, title="Evolución de pesos por categoría", height=380)
    fig.update_yaxes(ticksuffix="%", range=[0, 101])
    return fig


def plot_etf_pnl_attribution(
    weights_long: pd.DataFrame,
    selected_date: "pd.Timestamp | None" = None,
) -> go.Figure:
    """Horizontal bar of each Phase 2 ETF's € P&L.

    selected_date None shows cumulative PnL since Phase 2 inception; a date shows
    that day's PnL (vs the previous day).
    """
    if weights_long.empty:
        return go.Figure()

    df = _weights_with_categories(weights_long)
    phase2 = df[df["date"] >= OPERATIVA_DATE]
    if phase2.empty:
        return go.Figure()

    sorted_dates = sorted(phase2["date"].unique())
    first_date = sorted_dates[0]
    last_date = sorted_dates[-1]

    if selected_date is not None:
        # Closest available date ≤ selected_date
        candidates = [d for d in sorted_dates if d <= selected_date]
        if not candidates:
            return go.Figure()
        ref_curr = candidates[-1]

        before = [d for d in sorted_dates if d < ref_curr]
        if not before:
            return go.Figure()
        ref_prev = before[-1]

        init_date = ref_prev
        curr_date = ref_curr
        period_label = ref_curr.strftime("%d/%m/%Y")
        title_text = f"Atribución de resultado por ETF — {period_label}"
    else:
        init_date = first_date
        curr_date = last_date
        period_label = first_date.strftime("%d/%m/%Y")
        title_text = f"Atribución de resultado por ETF — desde {period_label}"

    initial = (
        phase2[phase2["date"] == init_date][["ticker", "value", "category"]]
        .set_index("ticker")
    )
    current = (
        phase2[phase2["date"] == curr_date][["ticker", "value", "category"]]
        .set_index("ticker")
    )

    merged = initial.join(current, lsuffix="_init", rsuffix="_curr", how="inner")
    merged["pnl"] = merged["value_curr"] - merged["value_init"]
    merged["pnl_pct"] = merged["pnl"] / merged["value_init"]
    merged = merged.sort_values("pnl", ascending=True)
    merged["display_label"] = [
        f"{ticker} · {ticker_label(ticker, include_ticker=False)}"
        for ticker in merged.index
    ]

    bar_colors = [
        "#15803D" if p >= 0 else "#B91C1C"
        for p in merged["pnl"]
    ]

    fig = go.Figure(
        go.Bar(
            x=merged["pnl"],
            y=merged["display_label"],
            orientation="h",
            marker=dict(
                color=bar_colors,
                opacity=0.85,
                line=dict(color="#FFFFFF", width=1),
            ),
            text=[
                f"€{p:+,.0f}  ({r:+.2%})"
                for p, r in zip(merged["pnl"], merged["pnl_pct"])
            ],
            textposition="outside",
            cliponaxis=False,
            hovertemplate=(
                "<b>%{y}</b><br>"
                "Resultado: <b>€%{x:+,.0f}</b><br>"
                "<extra></extra>"
            ),
        )
    )

    fig.add_vline(x=0, line_width=1, line_color="#CBD5E1")

    _apply_base(
        fig,
        title=title_text,
        height=max(440, 34 * len(merged) + 120),
    )
    fig.update_xaxes(tickprefix="€", tickformat="+,.0f", title_text="Resultado (EUR)")
    fig.update_yaxes(title_text="")
    fig.update_layout(margin=dict(l=315, r=155, t=56, b=30))
    return fig


def plot_historical_prices(
    prices: pd.DataFrame,
    selected_tickers: list[str],
    normalize: bool = False,
) -> go.Figure:
    """Line chart of closing prices; normalize indexes each series to 100 at the start."""
    fig = go.Figure()

    for i, ticker in enumerate(selected_tickers):
        if ticker not in prices.columns:
            continue
        series = prices[ticker].dropna()
        if series.empty:
            continue

        y = (series / series.iloc[0] * 100) if normalize else series
        label = ticker_label(ticker, include_ticker=True)
        color = CHART_COLORS[i % len(CHART_COLORS)]

        fig.add_trace(
            go.Scatter(
                x=series.index,
                y=y,
                mode="lines",
                name=label,
                line=dict(color=color, width=1.8),
                hovertemplate=(
                    f"<b>{label}</b><br>"
                    "%{x|%d/%m/%Y}<br>"
                    + (
                        "Base 100: %{y:.2f}"
                        if normalize
                        else "Precio cierre: %{y:.2f}"
                    )
                    + "<extra></extra>"
                ),
            )
        )

    title = "Precios Históricos (Base 100)" if normalize else "Precios Históricos de Cierre"
    _apply_base(fig, title=title, height=460)
    if normalize:
        fig.update_yaxes(title_text="Base 100")
        fig.add_hline(y=100, line_width=0.8, line_color="#555", line_dash="dot")
    else:
        fig.update_yaxes(title_text="Precio de cierre (moneda de cotización)")
    return fig
