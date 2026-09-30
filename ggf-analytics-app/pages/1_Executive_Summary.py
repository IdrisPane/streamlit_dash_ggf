"""Executive Summary view (Req 3.1-3.5).

The first, headline view of the GGF Analytics App. It answers the expansion
question at a glance for GGF top management: the 2022 headline financials, the
overall target attainment, the expansion verdict with its rationale, and which
SKU(s) drag on profit.

Design direction (see project ``DESIGN.md``)
--------------------------------------------
Restrained, data-first internal dashboard. A single deep-teal accent, a neutral
gray scale, and semantic red/green used *only* for financial meaning (loss vs.
profit). No decorative icons, emoji, or status dots. Every number is a real
figure computed at runtime from ``FruitsTrxDatasets.xlsx`` and is formatted with
thousands separators and an ``IDR``/``%`` unit. Empty and error states state the
cause and the next action.

Contract (so ``app.py`` — Task 14 — can wire this and it is testable now)
-------------------------------------------------------------------------
- :func:`render` renders the page with Streamlit. When ``bundle`` is ``None`` it
  loads the workbook via :func:`src.data_loader.load_data`, wrapped so a
  :class:`~src.data_loader.LoadError` shows a clear ``st.error`` naming the
  missing file/sheet/column and returns without raising.
- ``filters`` is an optional ``{"skus": [...], "months": [...]}`` dict. The
  Executive Summary is a *global* 2022 summary, so by design it does **not**
  apply the shared SKU/month filters (they narrow the detailed breakdown views
  instead); this is stated in the UI so the behavior is explicit.
- All analytics live in :mod:`src.metrics` / :mod:`src.forecasting` /
  :mod:`src.charts` (pure). This module only formats and lays out their outputs,
  so importing it has no Streamlit side effects — ``render`` is called once at
  module end (via :func:`main`) the way Streamlit executes a page top-to-bottom.
"""

from __future__ import annotations

from typing import Optional

import streamlit as st

from src import charts, config, metrics
from src.data_loader import DataBundle, LoadError, load_data
from src.forecasting import forecast_all


# ---------------------------------------------------------------------------
# Formatting helpers (thousands separators + IDR / % units, per DESIGN.md)
# ---------------------------------------------------------------------------


def _fmt_idr(value: float) -> str:
    """Format a monetary amount with thousands separators and the IDR label."""
    return f"{value:,.0f} {config.CURRENCY_LABEL}"


def _fmt_pct(value: Optional[float]) -> str:
    """Format a percentage to one decimal place, or a clear N/A when absent."""
    if value is None:
        return "N/A"
    return f"{value:.1f}%"


def _profit_delta(profit: float) -> Optional[str]:
    """Return a semantic delta label so st.metric colors profit green / loss red.

    ``st.metric`` colors a positive delta green and a negative delta red, which
    matches the financial semantics in DESIGN.md (green profit, red loss). We
    only want the color cue, so the delta text mirrors the profit sign.
    """
    if profit > 0:
        return f"+{profit:,.0f} {config.CURRENCY_LABEL} profit"
    if profit < 0:
        return f"{profit:,.0f} {config.CURRENCY_LABEL} loss"
    return None


# ---------------------------------------------------------------------------
# Data access
# ---------------------------------------------------------------------------


def _load_bundle() -> Optional[DataBundle]:
    """Load the workbook, surfacing a LoadError as an actionable st.error.

    Returns the :class:`DataBundle` on success, or ``None`` after rendering an
    error message that names the missing file/sheet/column and the next action
    (Req 1.6). Never raises.
    """
    try:
        return load_data()
    except LoadError as exc:
        st.error(
            f"Could not load the source workbook: {exc}\n\n"
            "Next step: place `FruitsTrxDatasets.xlsx` (with the MstProduct, "
            "SalesTarget, SalesTransaction, and RawMatPriceTrend sheets) in the "
            "app's `data/` folder, or set the `GGF_WORKBOOK_PATH` environment "
            "variable to its location, then reload."
        )
        return None


# ---------------------------------------------------------------------------
# Page
# ---------------------------------------------------------------------------


def render(bundle: Optional[DataBundle] = None, filters: Optional[dict] = None) -> None:
    """Render the Executive Summary page (Req 3.1-3.5).

    Args:
        bundle: A pre-loaded :class:`DataBundle`. When ``None``, the workbook is
            loaded here (with a LoadError surfaced as an st.error).
        filters: Optional ``{"skus": [...], "months": [...]}``. Ignored by this
            global-summary view by design; the UI states this explicitly.
    """
    st.header("Executive Summary")
    st.caption(
        "GGF fruit-beverage business — 2022 headline results and the expansion "
        "recommendation. Figures are computed from the source workbook; profit "
        "uses the recomputed basis "
        f"(net revenue minus raw-material and production cost)."
    )

    if bundle is None:
        bundle = _load_bundle()
    if bundle is None:
        return  # error already surfaced by _load_bundle()

    joined = bundle.joined
    if joined is None or joined.empty:
        st.warning(
            "The workbook loaded but contains no joined sales records to "
            "summarize. Next step: confirm the SalesTransaction sheet has 2022 "
            "rows for SKUs 200100001, 200100003, and 200100004."
        )
        return

    if filters:
        st.caption(
            "Note: this summary reflects the full 2022 business. SKU and month "
            "filters apply to the detailed breakdown views, not this global "
            "overview."
        )

    # --- Compute all metrics (pure) ------------------------------------
    totals = metrics.totals_2022(joined)
    attainment = metrics.overall_target_attainment(joined)
    per_sku = metrics.per_sku_pnl(joined)
    forecasts = forecast_all(joined)
    verdict = metrics.expansion_verdict(totals, per_sku, forecasts)
    least_profitable = metrics.least_profitable_sku(per_sku)

    # --- Headline: the expansion verdict (the conclusion, up top) ------
    _render_verdict(verdict)

    st.divider()

    # --- KPI row: 2022 headline financials (Req 3.1, 3.2) --------------
    st.subheader("2022 headline financials")
    col_rev, col_cost, col_profit, col_margin = st.columns(4)
    col_rev.metric("Net Revenue", _fmt_idr(totals.net_revenue))
    col_cost.metric("Total Cost", _fmt_idr(totals.total_cost))
    col_profit.metric(
        "Total Profit",
        _fmt_idr(totals.total_profit),
        delta=_profit_delta(totals.total_profit),
    )
    col_margin.metric("Profit Margin", _fmt_pct(totals.margin_pct))

    col_att, _spacer = st.columns([1, 3])
    col_att.metric("Overall Target Attainment", _fmt_pct(attainment))
    if attainment is None:
        st.caption(
            "Target attainment is not applicable: the total sales target for "
            "2022 is zero."
        )

    st.divider()

    # --- Least-profitable SKU(s) (Req 3.4, 3.5) ------------------------
    _render_least_profitable(least_profitable)

    st.divider()

    # --- Per-SKU profit chart (supporting visual) ----------------------
    st.subheader("Profit by SKU")
    st.caption("Which products carry 2022 profit? (green profit, red loss)")
    st.plotly_chart(charts.pnl_by_sku_figure(per_sku), width="stretch")


def _render_verdict(verdict: metrics.Verdict) -> None:
    """Render the expansion verdict, rationale, and forecast relationship."""
    st.subheader("Expansion verdict")

    recommendation_text = {
        "Expand": "Expand",
        "Optimize First": "Optimize first",
        "Expand and Optimize": "Expand and optimize",
    }.get(verdict.recommendation, verdict.recommendation)

    # The recommendation is the single focal statement; keep it prominent but
    # calm (no icons/badges). A colored callout matches the financial reading.
    if verdict.recommendation == "Expand":
        st.success(f"Recommendation: {recommendation_text}")
    elif verdict.recommendation == "Optimize First":
        st.warning(f"Recommendation: {recommendation_text}")
    else:  # "Expand and Optimize"
        st.info(f"Recommendation: {recommendation_text}")

    st.write(verdict.rationale)

    relationship_text = {
        "supports": "The 12-month forecast supports this recommendation.",
        "neutral": "The 12-month forecast is neutral toward this recommendation.",
        "contradicts": "The 12-month forecast contradicts this recommendation.",
    }.get(
        verdict.forecast_relationship,
        f"Forecast relationship: {verdict.forecast_relationship}.",
    )
    st.caption(relationship_text)


def _render_least_profitable(least_profitable: list[metrics.SkuProfit]) -> None:
    """Render the lowest-profit SKU(s), listing all ties (Req 3.4, 3.5)."""
    if not least_profitable:
        st.subheader("Lowest-profit SKU")
        st.info(
            "No per-SKU profit is available to rank. Next step: confirm the "
            "joined data contains SKU-attributed transactions."
        )
        return

    if len(least_profitable) == 1:
        st.subheader("Lowest-profit SKU")
    else:
        st.subheader("Lowest-profit SKUs (tied)")

    cols = st.columns(len(least_profitable))
    for col, item in zip(cols, least_profitable):
        label = item.product_name or config.SKU_MAP.get(item.sku) or item.sku
        col.metric(
            f"{label} ({item.sku})",
            _fmt_idr(item.total_profit),
            delta=_profit_delta(item.total_profit),
        )


def main() -> None:
    """Streamlit entry point for the page (executed top-to-bottom by Streamlit).

    Filters, when present, are read from ``st.session_state`` (populated by
    ``app.py`` in Task 14). This view ignores them by design but forwards them
    so the contract matches the other pages.
    """
    filters = st.session_state.get("filters") if hasattr(st, "session_state") else None
    render(bundle=None, filters=filters)


# Auto-run when Streamlit executes this file directly (multipage nav /
# ``streamlit run``). When ``app.py`` drives the pages it sets
# ``_APP_DRIVES_PAGES`` and calls :func:`render` itself, so we skip the
# auto-run here to avoid double-rendering.
if not st.session_state.get("_APP_DRIVES_PAGES"):
    main()
