"""Predictive view — 12-month sales & profit forecasts (Req 8.2-8.7).

This Streamlit page renders the predictive pillar for GGF top management:

- A 12-month **sales** forecast and a 12-month **profit** forecast per SKU,
  produced by :func:`src.forecasting.forecast_all` (Holt linear exponential
  smoothing on the 12 monthly observations).
- **Forecast-vs-actual** time-series (:func:`src.charts.forecast_figure`), with
  the historical period drawn solid and the forecast period dashed so the two
  are labeled distinctly (Req 8.5). Both the profit and sales views are shown.
- A statement of **how profit is derived** from the sales forecast (Req 8.4).
- The **12-observation basis** and the listed forecast **assumptions** (Req 8.6),
  read from the fitted results' ``assumptions``.
- **Excluded-SKU** labeling with the reason, for any SKU that could not be fitted
  (Req 8.2). On the real workbook all three SKUs have 12 observations.
- The **forecast-to-verdict relationship** (Req 8.7): whether the forecast
  supports, is neutral to, or contradicts the expansion verdict, referencing the
  verdict's recommendation.

Design direction (see project ``DESIGN.md``): restrained, data-first, calm. The
numbers come from the real workbook at runtime; empty/error states name the
cause and the next action. All analytical logic lives in ``src``; this module
only orchestrates and renders.

Contract:
    ``render(bundle=None, filters=None)`` — when ``bundle`` is ``None`` the page
    loads the data itself (surfacing a :class:`~src.data_loader.LoadError` as an
    ``st.error``). ``filters`` is an optional ``{"skus": [...], "months": [...]}``
    mapping that narrows which SKUs are forecast/shown.
"""

from __future__ import annotations

from typing import Optional

import streamlit as st

from src import charts, config
from src.data_loader import DataBundle, LoadError, load_data
from src.forecasting import forecast_all
from src.metrics import (
    expansion_verdict,
    forecast_relationship,
    per_sku_pnl,
    totals_2022,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _sku_label(sku: str) -> str:
    """Return a readable ``"Product (SKU)"`` label for a SKU key."""
    name = config.SKU_MAP.get(str(sku))
    return f"{name} ({sku})" if name else str(sku)


def _apply_filters(bundle: DataBundle, filters: Optional[dict]):
    """Return the joined frame narrowed by the optional SKU/month filters.

    Filters are applied leniently: an absent or empty ``skus``/``months`` list
    means "no restriction on that axis". SKU keys are compared as strings and
    months as their ``YYYY-MM`` string labels, matching how ``joined`` stores
    them.

    Args:
        bundle: The loaded data bundle.
        filters: Optional ``{"skus": [...], "months": [...]}`` mapping.

    Returns:
        A (possibly filtered) copy-safe view of ``bundle.joined``.
    """
    joined = bundle.joined
    if not filters:
        return joined

    skus = filters.get("skus")
    months = filters.get("months")

    if skus:
        wanted = {str(s) for s in skus}
        joined = joined[joined["SKU"].astype("string").isin(wanted)]
    if months:
        wanted_m = {str(m) for m in months}
        joined = joined[joined["month"].astype("string").isin(wanted_m)]
    return joined


def _fmt_idr(value: float) -> str:
    """Format a numeric IDR amount with thousands separators."""
    return f"{value:,.0f} {config.CURRENCY_LABEL}"


# ---------------------------------------------------------------------------
# Render
# ---------------------------------------------------------------------------


def render(bundle: Optional[DataBundle] = None, filters: Optional[dict] = None) -> None:
    """Render the Predictive view (Req 8.2-8.7).

    Args:
        bundle: A preloaded :class:`~src.data_loader.DataBundle`. When ``None``,
            the data is loaded here and any :class:`~src.data_loader.LoadError`
            is surfaced as an ``st.error`` (analysis is then skipped).
        filters: Optional ``{"skus": [...], "months": [...]}`` filter selection
            narrowing which SKUs/months are forecast and shown.
    """
    st.title("Predictive analysis")
    st.caption(
        "Twelve-month sales and profit forecasts per SKU, and what they mean "
        "for the expansion decision."
    )

    # --- Load / error gate (Req: surface LoadError) ------------------------
    if bundle is None:
        try:
            bundle = load_data()
        except LoadError as exc:
            st.error(
                f"Could not load the workbook: {exc}. "
                "Check that FruitsTrxDatasets.xlsx is present under data/ "
                "(or set GGF_WORKBOOK_PATH), then reload."
            )
            return

    joined = _apply_filters(bundle, filters)

    if joined.empty:
        st.warning(
            "No sales records match the current filter, so there is nothing to "
            "forecast. Clear or widen the SKU/month filter to see forecasts."
        )
        return

    # --- Fit forecasts (Req 8.1-8.4, 8.6) ----------------------------------
    results = forecast_all(joined)
    fitted = [r for r in results if not r.excluded]
    excluded = [r for r in results if r.excluded]

    # --- Excluded-SKU labeling (Req 8.2) -----------------------------------
    if excluded:
        st.subheader("Excluded SKUs")
        for res in excluded:
            reason = res.assumptions[0] if res.assumptions else "insufficient data"
            st.error(f"{_sku_label(res.sku)} — excluded from forecasting. {reason}")

    if not fitted:
        st.warning(
            "No SKU has the twelve monthly observations required to fit a "
            "forecast under the current filter. Widen the month filter so each "
            "SKU has its full Jan-Dec 2022 history."
        )
        return

    # --- Forecast-vs-actual charts (Req 8.3, 8.5) --------------------------
    st.subheader("Forecast vs actual")
    st.caption(
        "Solid lines are the actual Jan-Dec 2022 history; dashed lines are the "
        "twelve-month forecast (Jan-Dec 2023)."
    )

    profit_tab, sales_tab = st.tabs(["Profit forecast", "Sales forecast"])
    with profit_tab:
        st.plotly_chart(
            charts.forecast_figure(fitted, metric="profit"),
            width="stretch",
        )
    with sales_tab:
        st.plotly_chart(
            charts.forecast_figure(fitted, metric="sales"),
            width="stretch",
        )

    # --- How profit is derived from the sales forecast (Req 8.4) -----------
    st.subheader("How the profit forecast is derived")
    st.markdown(
        "The profit forecast is not modeled separately. Each forecast month's "
        "profit is the **forecast sales quantity multiplied by that SKU's "
        "recomputed per-unit margin**, where the per-unit margin is "
        "`sum(recomputed_profit) / sum(Qty)` over 2022 (a volume-weighted "
        "average margin per unit). Recomputed profit uses the "
        f"`{config.PROFIT_BASIS_ID}` basis "
        f"(`{config.PROFIT_BASIS_FORMULA}`), so the profit forecast stays "
        "consistent with the P&L analysis."
    )

    # --- Basis + assumptions (Req 8.6) -------------------------------------
    st.subheader("Basis and assumptions")
    st.markdown(
        "Each SKU's forecast is based on **twelve monthly observations** "
        "(Jan-Dec 2022) — the only history available. The fitted forecasts rely "
        "on these assumptions:"
    )
    # Assumptions are identical across fitted SKUs (same method/basis); list the
    # first fitted result's assumptions once so the view stays scannable.
    for assumption in fitted[0].assumptions:
        st.markdown(f"- {assumption}")

    # --- Forecast-to-verdict relationship (Req 8.7) ------------------------
    st.subheader("Relationship to the expansion verdict")

    totals = totals_2022(joined)
    pnl = per_sku_pnl(joined)
    verdict = expansion_verdict(totals, pnl, results)
    relationship = forecast_relationship(results)

    relationship_phrase = {
        "supports": "**supports**",
        "neutral": "is **neutral** to",
        "contradicts": "**contradicts**",
    }.get(relationship, f"**{relationship}**")

    st.markdown(
        f"The Executive Summary's expansion verdict is "
        f"**{verdict.recommendation}**. On a first-to-last comparison of the "
        f"aggregate twelve-month forecast profit, the forecast "
        f"{relationship_phrase} that verdict."
    )
    st.caption(verdict.rationale)


def main() -> None:
    """Streamlit page entry point."""
    render()


# Auto-run when Streamlit executes this file directly. When ``app.py`` drives
# the pages it sets ``_APP_DRIVES_PAGES`` and calls :func:`render` itself, so
# skip the auto-run here to avoid double-rendering.
if not st.session_state.get("_APP_DRIVES_PAGES"):
    main()
