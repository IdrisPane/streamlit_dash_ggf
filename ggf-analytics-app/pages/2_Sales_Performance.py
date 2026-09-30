"""Sales Performance view (Req 4.1, 4.2, 4.3, 4.4, 4.6).

Presents 2022 sales performance against targets for the three tracked SKUs:

- Actual quantity, value, and SalesTarget per SKU across all twelve months,
  zero-filled where a month has no recorded sales (Req 4.1).
- A monthly sales-trend chart with anomalous (SKU, month) points highlighted
  (Req 4.2, 4.4).
- Actual-vs-target bars per month (Req 4.1) and target attainment per SKU/month,
  with a zero/missing target shown as "N/A" rather than a percentage (Req 4.3).
- Insight callouts naming each flagged anomaly's SKU, month, and deviation
  (Req 4.4).
- An empty-result indication that retains the filter controls when a filter
  selection matches no data (Req 4.6).

Design direction (DESIGN.md): restrained, data-first, near-white, a single teal
accent, semantic red/green only for financial meaning, no decorative icons or
emoji. Charts render via ``st.plotly_chart``; every number is computed at
runtime from the real workbook. Empty and error states state the cause and the
next action.

Contract
--------
``render(bundle=None, filters=None)`` is the entry point.

- ``bundle``: a loaded :class:`src.data_loader.DataBundle`. When ``None`` the
  view loads the workbook itself and surfaces a :class:`~src.data_loader.LoadError`
  via ``st.error`` without proceeding to analysis.
- ``filters``: an optional ``{"skus": [...], "months": [...]}`` dict supplied by
  the shared app-level controls (Task 14). When it is ``None`` the view provides
  its own sidebar SKU/month multiselect fallbacks so the page also works when
  opened directly. Selections are applied to the displayed tables and charts
  (Req 4.5, 4.6).

Analytics stays in ``src`` (pure, tested). This module only filters, arranges,
and renders.
"""

from __future__ import annotations

from typing import Optional

import pandas as pd
import streamlit as st

from src import charts, config, metrics
from src.data_loader import DataBundle, LoadError, load_data


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _sku_label(sku: str) -> str:
    """Return ``"<name> (<sku>)"`` when the product name is known, else the SKU."""
    name = config.SKU_MAP.get(str(sku))
    return f"{name} ({sku})" if name else str(sku)


def _resolve_filters(
    joined: pd.DataFrame, filters: Optional[dict]
) -> tuple[list[str], list[str]]:
    """Resolve the active (skus, months) selection.

    When ``filters`` is supplied by the caller its ``skus``/``months`` lists are
    used directly (empty or missing means "all"). When ``filters`` is ``None``
    the view renders its own sidebar multiselects as a fallback so the page is
    usable on its own. The option universe is the fixed three SKUs and the
    twelve 2022 months, independent of what survived filtering, so the controls
    are always available to change a selection (Req 4.6).

    Returns:
        A ``(skus, months)`` pair of the selected string keys. An empty list in
        either position means "no restriction on that dimension".
    """
    all_skus = sorted(config.VALID_SKUS)
    all_months = list(metrics.MONTHS_2022)

    if filters is not None:
        skus = [str(s) for s in (filters.get("skus") or [])]
        months = [str(m) for m in (filters.get("months") or [])]
        return skus, months

    # Fallback in-page controls (only when the caller supplies no filters).
    st.sidebar.header("Filters")
    skus = st.sidebar.multiselect(
        "SKU",
        options=all_skus,
        default=all_skus,
        format_func=_sku_label,
        help="Limit the analysis to the selected products.",
    )
    months = st.sidebar.multiselect(
        "Month (2022)",
        options=all_months,
        default=all_months,
        help="Limit the analysis to the selected calendar months.",
    )
    return skus, months


def _apply_filters(
    joined: pd.DataFrame, skus: list[str], months: list[str]
) -> pd.DataFrame:
    """Return ``joined`` restricted to the selected SKUs and months.

    An empty selection list for a dimension means "no restriction" on that
    dimension. SKU comparison is done on string keys to match the loader's
    normalized SKU type.
    """
    frame = joined
    if skus:
        frame = frame[frame["SKU"].astype("string").isin(skus)]
    if months:
        frame = frame[frame["month"].isin(months)]
    return frame


def _format_idr(value: float) -> str:
    """Format a numeric IDR value with thousands separators, no decimals."""
    if value is None or pd.isna(value):
        return "N/A"
    return f"{value:,.0f}"


def _format_pct(value: Optional[float]) -> str:
    """Format an attainment percentage, or ``"N/A"`` when not applicable."""
    if value is None or pd.isna(value):
        return "N/A"
    return f"{value:.1f}%"


# ---------------------------------------------------------------------------
# Section renderers
# ---------------------------------------------------------------------------


def _render_matrix_table(matrix: pd.DataFrame, attainment: pd.DataFrame) -> None:
    """Render the per-SKU x month actual qty/value/target/attainment table (Req 4.1, 4.3)."""
    st.subheader("Actual sales vs target, per SKU per month")
    st.caption(
        "Quantity, value, and target for each SKU across the twelve months of "
        "2022. Months with no recorded sales are shown as zero, not omitted. "
        "Attainment is value / target; a zero or missing target shows as N/A."
    )

    merged = matrix.merge(
        attainment[["sku", "month", "attainment_pct"]],
        on=["sku", "month"],
        how="left",
    )

    display = pd.DataFrame(
        {
            "SKU": [_sku_label(s) for s in merged["sku"]],
            "Month": merged["month"],
            "Qty": merged["qty"].round(0).astype("int64"),
            "Value (IDR)": merged["value"].map(_format_idr),
            "Target (IDR)": merged["sales_target"].map(_format_idr),
            "Attainment": merged["attainment_pct"].map(_format_pct),
        }
    )
    st.dataframe(display, use_container_width=True, hide_index=True)


def _render_anomaly_insights(anomalies: list) -> None:
    """Render anomaly insight callouts naming SKU, month, and deviation (Req 4.4)."""
    st.subheader("Anomaly insights")
    if not anomalies:
        st.info(
            "No sales anomalies for the current selection. A month is flagged "
            "when its value is more than two standard deviations from the SKU's "
            "12-month mean, or its target attainment is below 50% or above 150%."
        )
        return

    st.caption(
        f"{len(anomalies)} flagged (SKU, month) point(s). Each is marked on the "
        "trend chart above and explained below."
    )
    for anomaly in anomalies:
        st.warning(anomaly.reason)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def render(bundle: Optional[DataBundle] = None, filters: Optional[dict] = None) -> None:
    """Render the Sales Performance view (Req 4.1-4.4, 4.6).

    Args:
        bundle: A loaded :class:`~src.data_loader.DataBundle`. When ``None`` the
            workbook is loaded here and any :class:`~src.data_loader.LoadError`
            is surfaced via ``st.error`` (the view does not proceed to analysis).
        filters: Optional ``{"skus": [...], "months": [...]}`` selection from the
            shared app controls. When ``None`` the view shows its own sidebar
            fallback controls.
    """
    st.title("Sales Performance")
    st.caption(
        "How 2022 actual sales tracked against target for Orange, Mango, and "
        "Pineapple juice — by month, with anomalies flagged."
    )

    # --- Data-load error gate (Req 1.6) ----------------------------------
    if bundle is None:
        try:
            bundle = load_data()
        except LoadError as exc:
            st.error(
                "Could not load the source workbook, so sales performance "
                f"cannot be shown. Cause: {exc}. Fix the workbook path or the "
                "missing sheet/column and reload."
            )
            return

    joined = bundle.joined
    if joined is None or joined.empty:
        st.error(
            "The loaded dataset has no joined transactions to analyze. Confirm "
            "the workbook contains SalesTransaction rows for the tracked SKUs, "
            "then reload."
        )
        return

    # --- Filters (caller-supplied or in-page fallback) -------------------
    skus, months = _resolve_filters(joined, filters)
    filtered = _apply_filters(joined, skus, months)

    # --- Empty-result handling (Req 4.6) ---------------------------------
    # The filter controls stay available (either the shared app controls or the
    # sidebar fallback rendered above); we simply skip the analysis body.
    if filtered.empty:
        st.warning(
            "No sales records match the current filter selection. Widen or "
            "clear the SKU / month filters to see data."
        )
        return

    # --- Metrics (pure, from src) ----------------------------------------
    matrix = metrics.monthly_sales_matrix(filtered)
    attainment = metrics.target_attainment_matrix(filtered)
    anomalies = metrics.sales_anomalies(filtered)

    # --- Monthly sales trend with anomaly highlights (Req 4.2, 4.4) ------
    st.subheader("Monthly sales trend")
    st.caption(
        "Sales value per SKU across Jan-Dec 2022. Diamonds mark flagged "
        "anomalies (see insights below)."
    )
    st.plotly_chart(
        charts.sales_trend_figure(matrix, anomalies),
        use_container_width=True,
    )

    # --- Actual vs target per month (Req 4.1) ----------------------------
    st.subheader("Actual value vs target, by month")
    st.caption("Total actual sales value against total target for each month.")
    st.plotly_chart(
        charts.sales_vs_target_figure(matrix),
        use_container_width=True,
    )

    # --- Target attainment per SKU/month (Req 4.3) -----------------------
    st.subheader("Target attainment, per SKU per month")
    st.caption(
        "Value as a percentage of target. Months with a zero or missing target "
        "are skipped (shown as N/A in the table below)."
    )
    st.plotly_chart(
        charts.attainment_figure(attainment),
        use_container_width=True,
    )

    # --- Detail table (Req 4.1, 4.3) -------------------------------------
    _render_matrix_table(matrix, attainment)

    # --- Anomaly insight callouts (Req 4.4) ------------------------------
    _render_anomaly_insights(anomalies)


def main() -> None:
    """Run the view as a standalone Streamlit page (loads its own data)."""
    render()


# Auto-run when Streamlit executes this file directly. When ``app.py`` drives
# the pages it sets ``_APP_DRIVES_PAGES`` and calls :func:`render` itself, so
# skip the auto-run here to avoid double-rendering.
if not st.session_state.get("_APP_DRIVES_PAGES"):
    main()
