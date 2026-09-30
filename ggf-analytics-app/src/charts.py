"""Visualization layer: Plotly figure builders for the GGF Analytics App.

Pure figure builders over the computed analytics frames (from :mod:`src.metrics`,
:mod:`src.forecasting`). Every builder returns a :class:`plotly.graph_objects.Figure`
and has **no Streamlit dependency**: it never calls ``st.*`` and never calls
``fig.show()``. The same inputs produce the same figure, so builders are safe to
call from the Streamlit views (Req 10.2) and from the deck exporter, which renders
each figure to a static PNG via ``kaleido``.

Design direction (see project ``DESIGN.md``)
--------------------------------------------
This is a restrained, data-first internal dashboard for GGF top management. The
styling here honors the approved direction:

- A near-white background with a neutral gray scale and a **single deep-teal
  accent** (:data:`ACCENT`) used sparingly for the focal series/primary emphasis.
- **Semantic financial colors only**: a restrained red (:data:`NEG`) for losses /
  negative values and a restrained green (:data:`POS`) for profit / positive
  values. These carry meaning (gain vs. loss); they are not decoration.
- SKU category colors are a small, consistent, muted set (:data:`SKU_COLORWAY`)
  reused across every chart, so a SKU has **one** color everywhere.
- Every chart title asks a specific question (no generic "Overview").
- No animations/transitions beyond Plotly defaults; no glow/chart-junk; a light
  axis gridline only. Axis labels and units (IDR, %) are always present.

Text readability (important)
----------------------------
Streamlit's ``st.plotly_chart(..., theme="streamlit")`` (the default) injects its
own Plotly template. In Streamlit's *dark* mode that template colors tick labels,
axis titles, and legend text **light** — which is invisible on this dashboard's
near-white chart surface. Because template values for specific elements (tick
font, axis-title font, legend font) take precedence over the global
``layout.font.color``, setting only the global font is not enough.

:func:`_apply_theme` therefore sets a dark, high-contrast text color
**explicitly** on every text element (title, tick labels, axis titles, legend,
hover label, annotations, modebar). Explicit figure properties always win over
the injected template, so the figures stay readable in both Streamlit light and
dark mode, and in the kaleido PNG export.

A shared :func:`_apply_theme` helper sets ``template='plotly_white'``, the font,
margins, and a consistent colorway so styling stays uniform across builders.

Sections covered (Req 10.2 — at least one builder per pillar):

- **Sales**: :func:`sales_trend_figure`, :func:`sales_vs_target_figure`,
  :func:`attainment_figure`.
- **P&L**: :func:`pnl_by_sku_figure`, :func:`cost_driver_figure`.
- **Raw material / seasonality**: :func:`raw_material_price_figure`,
  :func:`harvest_success_figure`, :func:`seasonality_figure`.
- **Predictive**: :func:`forecast_figure`.
"""

from __future__ import annotations

from typing import Iterable, Optional, Sequence, Union

import pandas as pd
import plotly.graph_objects as go

from . import config

# ---------------------------------------------------------------------------
# Palette (see DESIGN.md)
# ---------------------------------------------------------------------------

#: The single deep-teal accent, used sparingly for the focal series.
ACCENT: str = "#0F766E"

#: Restrained red — losses / negative financial values only (semantic).
NEG: str = "#B91C1C"

#: Restrained green — profit / positive financial values only (semantic).
POS: str = "#15803D"

#: Neutral grays for secondary series, axes, and text.
GRAY: str = "#64748B"
GRAY_DARK: str = "#334155"
GRAY_LIGHT: str = "#CBD5E1"

#: High-contrast text color used for ALL chart text (ticks, axis titles,
#: legend, hover, annotations). Deep slate on the near-white surface gives a
#: contrast ratio well above WCAG AA (~12:1), and it is set explicitly so the
#: Streamlit dark theme cannot turn the text light.
TEXT: str = "#1E293B"

#: Near-white surface background (DESIGN.md).
BG: str = "#FAFAFA"

#: Consistent, muted per-SKU colorway. A SKU keeps ONE color across every chart.
#: The three SKUs map to the teal accent, a neutral slate, and a muted amber —
#: a small, colorblind-friendly set with sufficient contrast.
SKU_COLORWAY: dict[str, str] = {
    "200100001": ACCENT,      # Orange Juice  -> deep teal (focal accent)
    "200100003": "#B45309",   # Mango Juice   -> muted amber
    "200100004": GRAY,        # Pineapple Juice -> neutral slate
}

#: Fallback color for any SKU not in :data:`SKU_COLORWAY`.
_FALLBACK_COLOR: str = GRAY_DARK

#: Standard system-UI font stack (readable sans-serif; no monospace aesthetic).
_FONT_FAMILY: str = (
    "system-ui, -apple-system, 'Segoe UI', Roboto, 'Helvetica Neue', "
    "Arial, sans-serif"
)


def _sku_color(sku: str) -> str:
    """Return the consistent color for ``sku`` (fallback if unknown)."""
    return SKU_COLORWAY.get(str(sku), _FALLBACK_COLOR)


def _sku_label(sku: str) -> str:
    """Return a readable ``"Product (SKU)"`` label for legends/axes."""
    name = config.SKU_MAP.get(str(sku))
    return f"{name} ({sku})" if name else str(sku)


def _apply_theme(fig: go.Figure) -> go.Figure:
    """Apply the shared, restrained theme to ``fig`` in place and return it.

    Sets ``template='plotly_white'``, a readable system font, comfortable
    margins, a near-white paper/plot background, a light axis gridline (no heavy
    grid), and the consistent SKU colorway. Mutates and returns ``fig`` for
    fluent use. No animation/transition settings are added (MOTION 1).

    Every text element gets an explicit dark color (:data:`TEXT`) so that
    Streamlit's injected theme (which is light-on-dark in dark mode) can never
    render axis labels, legend entries, hover text, or annotations invisible.

    Args:
        fig: The figure to style.

    Returns:
        The same figure, styled.
    """
    fig.update_layout(
        template="plotly_white",
        font=dict(family=_FONT_FAMILY, size=13, color=TEXT),
        title=dict(font=dict(size=16, color=TEXT), x=0.0, xanchor="left"),
        paper_bgcolor=BG,
        plot_bgcolor="white",
        margin=dict(l=72, r=24, t=64, b=64),
        colorway=list(SKU_COLORWAY.values()),
        legend=dict(
            bgcolor="rgba(0,0,0,0)",
            borderwidth=0,
            font=dict(family=_FONT_FAMILY, size=12, color=TEXT),
            title=dict(text="", font=dict(color=TEXT)),
        ),
        hovermode="x unified",
        hoverlabel=dict(
            bgcolor="white",
            bordercolor=GRAY_LIGHT,
            font=dict(family=_FONT_FAMILY, size=12, color=TEXT),
        ),
        # Modebar icons (camera / zoom / pan ...) were pale gray on near-white.
        modebar=dict(
            bgcolor="rgba(0,0,0,0)",
            color=GRAY,
            activecolor=ACCENT,
        ),
    )

    # Axis text: tick labels + axis titles set explicitly (template-proof).
    axis_text = dict(
        tickfont=dict(family=_FONT_FAMILY, size=12, color=TEXT),
        title_font=dict(family=_FONT_FAMILY, size=13, color=TEXT),
    )
    fig.update_xaxes(
        showgrid=False,
        showline=True,
        linecolor=GRAY,
        ticks="outside",
        tickcolor=GRAY,
        **axis_text,
    )
    fig.update_yaxes(
        showgrid=True,
        gridcolor="#E2E8F0",
        zeroline=True,
        zerolinecolor=GRAY,
        showline=False,
        **axis_text,
    )

    # Any annotation already on the figure (e.g. the 100% reference label).
    fig.update_annotations(font=dict(family=_FONT_FAMILY, size=12, color=TEXT))
    return fig


def _ordered_skus(values: Iterable[object]) -> list[str]:
    """Return the unique SKU strings in ``values`` sorted deterministically."""
    return sorted({str(v) for v in values})


def _fmt_si(value: float) -> str:
    """Compact SI label (``94.5M``, ``1.23G``), matching the y-axis tick style.

    Used for value labels drawn above/inside bars so the numbers read the same
    way as the axis (``M`` = million, ``G`` = billion, as Plotly's ``.3s``).
    """
    v = float(value)
    for divisor, suffix in ((1e12, "T"), (1e9, "G"), (1e6, "M"), (1e3, "k")):
        if abs(v) >= divisor:
            return f"{v / divisor:.3g}{suffix}"
    return f"{v:.3g}"


# ===========================================================================
# Sales section (Req 4.2, 4.3, 10.2)
# ===========================================================================


def sales_trend_figure(
    monthly_sales_matrix_df: pd.DataFrame,
    anomalies: Optional[Sequence[object]] = None,
) -> go.Figure:
    """Monthly sales *value* per SKU across Jan-Dec 2022, with anomaly marks (Req 4.2).

    Draws one line per SKU of the monthly sales value (``value``, IDR) against a
    chronological Jan..Dec 2022 x-axis, using the consistent :data:`SKU_COLORWAY`.
    When ``anomalies`` (the list from :func:`src.metrics.sales_anomalies`) is
    provided, each flagged ``(sku, month)`` point is marked distinctly with a red
    diamond (:data:`NEG`) so it stands out on its SKU's line.

    Args:
        monthly_sales_matrix_df: A :func:`src.metrics.monthly_sales_matrix` frame
            with columns ``sku``, ``month``, and ``value``.
        anomalies: Optional sequence of objects each exposing ``.sku`` and
            ``.month`` (e.g. :class:`src.metrics.SalesAnomaly`) to highlight.

    Returns:
        A line figure with one trace per SKU (plus one anomaly-marker trace when
        anomalies are supplied and matched).
    """
    fig = go.Figure()
    df = monthly_sales_matrix_df

    for sku in _ordered_skus(df["sku"]):
        sub = df[df["sku"] == str(sku)].sort_values("month")
        fig.add_trace(
            go.Scatter(
                x=sub["month"],
                y=sub["value"],
                mode="lines+markers",
                name=_sku_label(sku),
                line=dict(color=_sku_color(sku), width=2),
                marker=dict(size=5),
            )
        )

    if anomalies:
        pairs = {(str(a.sku), str(a.month)) for a in anomalies}
        marked = df[
            df.apply(lambda r: (str(r["sku"]), str(r["month"])) in pairs, axis=1)
        ].sort_values(["month", "sku"])
        if not marked.empty:
            fig.add_trace(
                go.Scatter(
                    x=marked["month"],
                    y=marked["value"],
                    mode="markers",
                    name="Anomaly",
                    marker=dict(
                        color=NEG, size=11, symbol="diamond-open",
                        line=dict(color=NEG, width=2),
                    ),
                    hovertext=[_sku_label(s) for s in marked["sku"]],
                )
            )

    fig.update_layout(
        title="Monthly sales value per SKU, Jan-Dec 2022",
        xaxis_title="Month (2022)",
        yaxis_title="Sales value (IDR)",
    )
    fig.update_xaxes(categoryorder="category ascending")
    return _apply_theme(fig)


def sales_vs_target_figure(monthly_sales_matrix_df: pd.DataFrame) -> go.Figure:
    """Actual sales value vs SalesTarget per SKU per month (Req 4.1, 4.3).

    Aggregates the monthly matrix to the month level and draws grouped bars of
    total actual value against total target across Jan..Dec 2022, answering
    whether the business is tracking to plan month by month. Actual uses the
    teal accent; target uses a neutral gray outline so it reads as a reference.

    Args:
        monthly_sales_matrix_df: A :func:`src.metrics.monthly_sales_matrix` frame
            with columns ``month``, ``value``, and ``sales_target``.

    Returns:
        A grouped-bar figure with an actual-value trace and a target trace.
    """
    df = monthly_sales_matrix_df
    grouped = (
        df.groupby("month", as_index=False)
        .agg(value=("value", "sum"), sales_target=("sales_target", "sum"))
        .sort_values("month")
    )

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=grouped["month"],
            y=grouped["value"],
            name="Actual value",
            marker_color=ACCENT,
            text=[_fmt_si(v) for v in grouped["value"]],
            textposition="outside",
            textfont=dict(size=10, color=TEXT),
            cliponaxis=False,
        )
    )
    fig.add_trace(
        go.Bar(
            x=grouped["month"],
            y=grouped["sales_target"],
            name="Sales target",
            marker_color="white",
            marker_line=dict(color=GRAY, width=1.5),
            text=[_fmt_si(v) for v in grouped["sales_target"]],
            textposition="outside",
            textfont=dict(size=10, color=TEXT),
            cliponaxis=False,
        )
    )
    fig.update_layout(
        title="Actual sales value vs target per month, 2022",
        xaxis_title="Month (2022)",
        yaxis_title="Value (IDR)",
        barmode="group",
    )
    fig.update_xaxes(categoryorder="category ascending")
    return _apply_theme(fig)


def attainment_figure(target_attainment_matrix_df: pd.DataFrame) -> go.Figure:
    """Target attainment (%) per SKU per month, N/A skipped (Req 4.3).

    Draws one line per SKU of ``attainment_pct`` against a chronological Jan..Dec
    2022 x-axis. Months where the target is 0/missing carry ``NaN`` attainment
    and are skipped (no point) rather than plotted as 0. A dashed reference line
    at 100% marks on-target performance.

    Args:
        target_attainment_matrix_df: A :func:`src.metrics.target_attainment_matrix`
            frame with columns ``sku``, ``month``, and ``attainment_pct``.

    Returns:
        A line figure with one trace per SKU plus a 100% reference line.
    """
    df = target_attainment_matrix_df
    fig = go.Figure()

    for sku in _ordered_skus(df["sku"]):
        sub = df[df["sku"] == str(sku)].sort_values("month")
        fig.add_trace(
            go.Scatter(
                x=sub["month"],
                y=sub["attainment_pct"],
                mode="lines+markers",
                name=_sku_label(sku),
                line=dict(color=_sku_color(sku), width=2),
                marker=dict(size=5),
                connectgaps=False,
            )
        )

    fig.add_hline(
        y=100.0, line_dash="dash", line_color=GRAY,
        annotation_text="On target (100%)", annotation_position="top left",
        annotation_font=dict(size=12, color=TEXT),
    )
    fig.update_layout(
        title="Target attainment per SKU per month, 2022",
        xaxis_title="Month (2022)",
        yaxis_title="Attainment (%)",
    )
    fig.update_xaxes(categoryorder="category ascending")
    return _apply_theme(fig)


# ===========================================================================
# P&L section (Req 5.1, 5.3, 10.2)
# ===========================================================================


def pnl_by_sku_figure(per_sku_pnl_df: pd.DataFrame) -> go.Figure:
    """Per-SKU total 2022 profit, green for positive / red for negative (Req 5.1).

    Draws one bar per SKU of ``total_profit`` (IDR). Bars use the semantic
    financial colors: green (:data:`POS`) for profit, red (:data:`NEG`) for a
    loss — so the least-profitable / loss-making SKUs read at a glance.

    Args:
        per_sku_pnl_df: A :func:`src.metrics.per_sku_pnl` frame indexed by ``sku``
            with a ``total_profit`` column (and optional ``product_name``).

    Returns:
        A bar figure with one trace (one bar per SKU).
    """
    df = per_sku_pnl_df
    skus = _ordered_skus(df.index)
    profits = [float(df.loc[s, "total_profit"]) for s in skus]
    labels = [
        str(df.loc[s, "product_name"])
        if "product_name" in df.columns and pd.notna(df.loc[s, "product_name"])
        else _sku_label(s)
        for s in skus
    ]
    colors = [POS if p >= 0 else NEG for p in profits]

    fig = go.Figure(
        go.Bar(
            x=labels,
            y=profits,
            marker_color=colors,
            name="Total profit",
            text=[_fmt_si(p) for p in profits],
            textposition="outside",
            textfont=dict(size=13, color=TEXT),
            cliponaxis=False,
            customdata=skus,
            hovertemplate="%{x}<br>Profit: %{y:,.0f} IDR<extra></extra>",
        )
    )
    fig.update_layout(
        title="Total 2022 profit per SKU (green profit, red loss)",
        xaxis_title="SKU",
        yaxis_title="Total profit (IDR)",
        showlegend=False,
    )
    return _apply_theme(fig)


def cost_driver_figure(cost_driver_breakdown_df: pd.DataFrame) -> go.Figure:
    """Raw-material vs production cost per SKU, stacked in IDR (Req 5.3).

    Draws a stacked bar per SKU splitting the recomputed cost base into raw
    material and production cost (both IDR), so the dominant cost driver per SKU
    is visible. Raw material uses the teal accent; production uses a neutral
    gray — two colors, consistent across SKUs.

    Args:
        cost_driver_breakdown_df: A :func:`src.metrics.cost_driver_breakdown`
            frame with columns ``sku``, ``raw_material_cost``,
            ``production_cost`` (and optional ``product_name``).

    Returns:
        A stacked-bar figure with a raw-material trace and a production trace.
    """
    df = cost_driver_breakdown_df.sort_values("sku")
    labels = [
        str(row.product_name)
        if "product_name" in df.columns and pd.notna(row.product_name)
        else _sku_label(row.sku)
        for row in df.itertuples(index=False)
    ]

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=labels,
            y=df["raw_material_cost"],
            name="Raw-material cost",
            marker_color=ACCENT,
            text=[_fmt_si(v) for v in df["raw_material_cost"]],
            textposition="inside",
            insidetextanchor="middle",
            textfont=dict(size=12, color="white"),
            hovertemplate="%{x}<br>Raw material: %{y:,.0f} IDR<extra></extra>",
        )
    )
    fig.add_trace(
        go.Bar(
            x=labels,
            y=df["production_cost"],
            name="Production cost",
            marker_color=GRAY,
            text=[_fmt_si(v) for v in df["production_cost"]],
            textposition="inside",
            insidetextanchor="middle",
            textfont=dict(size=12, color="white"),
            hovertemplate="%{x}<br>Production: %{y:,.0f} IDR<extra></extra>",
        )
    )

    # Total cost label above each stacked bar (invisible text-only trace).
    totals = (df["raw_material_cost"] + df["production_cost"]).astype(float)
    fig.add_trace(
        go.Scatter(
            x=labels,
            y=list(totals),
            mode="text",
            text=[f"<b>{_fmt_si(t)}</b>" for t in totals],
            textposition="top center",
            textfont=dict(size=13, color=TEXT),
            cliponaxis=False,
            showlegend=False,
            hoverinfo="skip",
        )
    )

    fig.update_layout(
        title="Cost drivers per SKU: raw-material vs production, 2022",
        xaxis_title="SKU",
        yaxis_title="Cost (IDR)",
        barmode="stack",
    )
    _apply_theme(fig)
    # Headroom so the total label above the tallest bar is not cut off.
    if len(totals) and float(totals.max()) > 0:
        fig.update_yaxes(range=[0, float(totals.max()) * 1.12])
    return fig


# ===========================================================================
# Raw-material / seasonality section (Req 7.1, 7.2, 7.3, 10.2)
# ===========================================================================


def _prep_raw_mat(
    raw_mat_trend_df: pd.DataFrame, sku: Optional[str]
) -> pd.DataFrame:
    """Return raw-material rows (optionally filtered to ``sku``) sorted by date.

    Coerces ``Date`` to datetime for a chronological weekly x-axis and keeps only
    rows with a parseable date and non-null SKU.
    """
    df = raw_mat_trend_df.copy()
    df = df[df["SKU"].notna()]
    if sku is not None:
        df = df[df["SKU"].astype(str) == str(sku)]
    df["_date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df[df["_date"].notna()].sort_values("_date")
    return df


def raw_material_price_figure(
    raw_mat_trend_df: pd.DataFrame, sku: Optional[str] = None
) -> go.Figure:
    """Raw-material Close Price time-series over 2020-01..2022-12 per SKU (Req 7.1).

    Draws one line per SKU (or a single line when ``sku`` is given) of the weekly
    ``Close Price`` against a chronological date x-axis spanning the full period,
    using the consistent :data:`SKU_COLORWAY`.

    Args:
        raw_mat_trend_df: The bundle's ``raw_mat_trend`` frame with columns
            ``Date``, ``SKU``, and ``Close Price``.
        sku: Optional SKU key to restrict the chart to a single product.

    Returns:
        A line figure with one trace per plotted SKU.
    """
    df = _prep_raw_mat(raw_mat_trend_df, sku)
    fig = go.Figure()
    for s in _ordered_skus(df["SKU"]):
        sub = df[df["SKU"].astype(str) == str(s)]
        fig.add_trace(
            go.Scatter(
                x=sub["_date"],
                y=sub["Close Price"],
                mode="lines",
                name=_sku_label(s),
                line=dict(color=_sku_color(s), width=1.6),
            )
        )
    fig.update_layout(
        title="Raw-material close price over 2020-2022 per SKU",
        xaxis_title="Week",
        yaxis_title="Close price (IDR)",
    )
    return _apply_theme(fig)


def harvest_success_figure(
    raw_mat_trend_df: pd.DataFrame, sku: Optional[str] = None
) -> go.Figure:
    """Harvest Success Rate (%) time-series, y-axis fixed 0-100 (Req 7.2).

    Draws one line per SKU (or a single line when ``sku`` is given) of the weekly
    ``Harvest Success Rate (%)`` against a chronological date x-axis, with the
    y-axis fixed to the 0-100 scale so rates read as percentages.

    Args:
        raw_mat_trend_df: The bundle's ``raw_mat_trend`` frame with columns
            ``Date``, ``SKU``, and ``Harvest Success Rate (%)``.
        sku: Optional SKU key to restrict the chart to a single product.

    Returns:
        A line figure with one trace per plotted SKU and a fixed 0-100 y-range.
    """
    df = _prep_raw_mat(raw_mat_trend_df, sku)
    col = "Harvest Success Rate (%)"
    fig = go.Figure()
    for s in _ordered_skus(df["SKU"]):
        sub = df[df["SKU"].astype(str) == str(s)]
        fig.add_trace(
            go.Scatter(
                x=sub["_date"],
                y=sub[col],
                mode="lines",
                name=_sku_label(s),
                line=dict(color=_sku_color(s), width=1.6),
            )
        )
    fig.update_layout(
        title="Harvest success rate over 2020-2022 per SKU",
        xaxis_title="Week",
        yaxis_title="Harvest success rate (%)",
    )
    _apply_theme(fig)
    # Fixed 0-100 scale (Req 7.2) — applied after the theme so it is not
    # overridden by the shared y-axis settings.
    fig.update_yaxes(range=[0, 100])
    return fig


def seasonality_figure(seasonality_compare_df: pd.DataFrame) -> go.Figure:
    """Average close price: peak vs non-peak season per SKU (Req 7.3).

    Draws grouped bars per SKU of the average ``Close Price`` in peak-season
    weeks against non-peak weeks, so any seasonal price premium is visible. Peak
    uses the teal accent; non-peak uses a neutral gray.

    Args:
        seasonality_compare_df: A :func:`src.metrics.seasonality_price_compare`
            frame with columns ``sku``, ``avg_close_peak``, ``avg_close_non_peak``.

    Returns:
        A grouped-bar figure with a peak trace and a non-peak trace.
    """
    df = seasonality_compare_df.sort_values("sku")
    labels = [_sku_label(s) for s in df["sku"]]

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=labels,
            y=df["avg_close_peak"],
            name="Peak season",
            marker_color=ACCENT,
            text=[_fmt_si(v) for v in df["avg_close_peak"]],
            textposition="outside",
            textfont=dict(size=12, color=TEXT),
            cliponaxis=False,
        )
    )
    fig.add_trace(
        go.Bar(
            x=labels,
            y=df["avg_close_non_peak"],
            name="Non-peak season",
            marker_color=GRAY,
            text=[_fmt_si(v) for v in df["avg_close_non_peak"]],
            textposition="outside",
            textfont=dict(size=12, color=TEXT),
            cliponaxis=False,
        )
    )
    fig.update_layout(
        title="Average raw-material close price: peak vs non-peak per SKU",
        xaxis_title="SKU",
        yaxis_title="Average close price (IDR)",
        barmode="group",
    )
    return _apply_theme(fig)


# ===========================================================================
# Predictive section (Req 8.5, 10.2)
# ===========================================================================


def forecast_figure(
    forecast_result: Union[object, Iterable[object]],
    metric: str = "profit",
) -> go.Figure:
    """Forecast vs actual on one time-series, historical/forecast distinct (Req 8.5).

    Plots each SKU's historical actuals (solid line) and its forecast (dashed
    line) on a single chronological time-series, so the two periods are labeled
    and styled distinctly. Accepts a single :class:`src.forecasting.ForecastResult`
    or an iterable of them (e.g. from :func:`src.forecasting.forecast_all`).

    For ``metric="sales"`` the historical series is the SKU's monthly sales
    quantity (``ForecastResult.historical``) and the forecast is
    ``forecast_sales``. For ``metric="profit"`` (the default) the historical
    series is derived by scaling the historical quantity by the same per-unit
    margin implied by the forecast (``forecast_profit / forecast_sales``), and
    the forecast is ``forecast_profit`` — so historical and forecast profit are
    on a consistent basis. Excluded SKUs (no fit) contribute a historical trace
    only.

    Args:
        forecast_result: One ``ForecastResult`` or an iterable of them.
        metric: ``"profit"`` (default) or ``"sales"``.

    Returns:
        A figure containing, per fitted SKU, a solid historical trace and a
        dashed forecast trace on one shared time axis.
    """
    if metric not in ("profit", "sales"):
        raise ValueError("metric must be 'profit' or 'sales'")

    if _is_forecast_result(forecast_result):
        results = [forecast_result]
    else:
        results = list(forecast_result)  # type: ignore[arg-type]

    unit = "IDR" if metric == "profit" else "units"
    fig = go.Figure()

    for res in results:
        color = _sku_color(res.sku)
        label = _sku_label(res.sku)

        hist = res.historical.astype(float)
        if metric == "profit":
            hist = _historical_profit(res)

        if hist is not None and not hist.empty:
            fig.add_trace(
                go.Scatter(
                    x=list(hist.index),
                    y=list(hist.values),
                    mode="lines+markers",
                    name=f"{label} - historical",
                    legendgroup=res.sku,
                    line=dict(color=color, width=2),
                    marker=dict(size=4),
                )
            )

        fc = res.forecast_profit if metric == "profit" else res.forecast_sales
        if fc is not None and not fc.empty:
            # Bridge from the last historical point so the dashed line connects.
            x_vals = list(fc.index)
            y_vals = list(fc.astype(float).values)
            if hist is not None and not hist.empty:
                x_vals = [hist.index[-1]] + x_vals
                y_vals = [float(hist.iloc[-1])] + y_vals
            fig.add_trace(
                go.Scatter(
                    x=x_vals,
                    y=y_vals,
                    mode="lines+markers",
                    name=f"{label} - forecast",
                    legendgroup=res.sku,
                    line=dict(color=color, width=2, dash="dash"),
                    marker=dict(size=4, symbol="circle-open"),
                )
            )

    metric_label = "profit" if metric == "profit" else "sales quantity"
    fig.update_layout(
        title=(
            f"Forecast vs actual monthly {metric_label} per SKU "
            "(solid = historical, dashed = forecast)"
        ),
        xaxis_title="Month",
        yaxis_title=f"Monthly {metric_label} ({unit})",
    )
    fig.update_xaxes(categoryorder="category ascending")
    return _apply_theme(fig)


def _is_forecast_result(obj: object) -> bool:
    """Return True if ``obj`` looks like a single ``ForecastResult``.

    Duck-typed (rather than an ``isinstance`` import) so :mod:`src.charts` stays
    decoupled and a single result is distinguished from an iterable of results.
    """
    return all(
        hasattr(obj, attr)
        for attr in ("sku", "historical", "forecast_sales", "forecast_profit")
    )


def _historical_profit(res: object) -> Optional[pd.Series]:
    """Approximate a SKU's historical monthly profit for the forecast chart.

    Scales the historical sales quantity by the per-unit margin implied by the
    forecast (``forecast_profit / forecast_sales``), so historical and forecast
    profit share one basis. Returns the historical quantity unscaled if the
    margin cannot be recovered (e.g. an excluded SKU with an empty forecast).
    """
    hist = res.historical.astype(float)  # type: ignore[attr-defined]
    fc_sales = res.forecast_sales  # type: ignore[attr-defined]
    fc_profit = res.forecast_profit  # type: ignore[attr-defined]
    if fc_sales is None or fc_sales.empty:
        return hist
    nonzero = fc_sales[fc_sales != 0]
    if nonzero.empty:
        return hist
    per_unit_margin = float(fc_profit.loc[nonzero.index[0]] / nonzero.iloc[0])
    return hist * per_unit_margin