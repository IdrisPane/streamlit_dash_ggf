"""Raw Material & Seasonality view (Req 7.1, 7.2, 7.3, 7.5).

This Streamlit page analyzes the RawMatPriceTrend data for GGF's three fruit
SKUs across the weekly periods spanning 2020-01 through 2022-12:

- **Close price** time-series per selected SKU (Req 7.1).
- **Harvest success rate (%)** time-series on a fixed 0-100 scale (Req 7.2).
- **Peak vs non-peak** average close-price comparison per SKU, with a
  data-driven insight sentence naming the SKU with the largest seasonal price
  premium (Req 7.3).

The view honors the shared SKU filter (Req 7.4/7.5): when a SKU filter is
applied, every visualization is restricted to the selected SKUs. If the
selection leaves a visualization with no rows, that visualization shows an
``st.info`` empty-result indication stating the cause and the next action while
the filter state is retained (Req 7.5) — the page never crashes.

Design direction (see ``DESIGN.md``): restrained, data-first, calm. Charts come
from the pure builders in :mod:`src.charts`; every number shown is computed at
runtime from the real workbook. All analytics stay in ``render``; the page
exposes ``render(bundle=None, filters=None)`` so ``app.py`` can pass a shared
``DataBundle`` and shared filters, and ``main()`` runs it standalone.
"""

from __future__ import annotations

from typing import Optional

import pandas as pd
import streamlit as st

from src import charts, config
from src.data_loader import DataBundle, LoadError, load_data
from src.metrics import seasonality_price_compare


def _selected_skus(filters: Optional[dict]) -> Optional[list[str]]:
    """Return the SKU filter as a list of string keys, or ``None`` if unset.

    A ``None`` (or missing/empty) SKU filter means "no restriction" — all SKUs
    are shown. Any provided values are coerced to strings to match the string
    SKU keys used throughout the data layer and :data:`src.config.SKU_MAP`.
    """
    if not filters:
        return None
    skus = filters.get("skus")
    if not skus:
        return None
    return [str(s) for s in skus]


def _sku_display(sku: str) -> str:
    """Return a readable ``"Product (SKU)"`` label for prose/insight text."""
    name = config.SKU_MAP.get(str(sku))
    return f"{name} ({sku})" if name else str(sku)


def _filter_rows(raw_mat_trend: pd.DataFrame, skus: Optional[list[str]]) -> pd.DataFrame:
    """Return RawMatPriceTrend rows restricted to ``skus`` (Req 7.4).

    Drops rows with a null SKU, then keeps only the selected SKUs when a filter
    is applied. With no filter, all valid-SKU rows are returned.
    """
    df = raw_mat_trend[raw_mat_trend["SKU"].notna()].copy()
    if skus is not None:
        df = df[df["SKU"].astype(str).isin(skus)]
    return df


def _empty_note(skus: Optional[list[str]], what: str) -> None:
    """Render the empty-result indication for a visualization (Req 7.5).

    States the cause (the selected SKU filter left no rows) and the next action
    (clear or change the SKU filter), so the filter state is retained and the
    user knows how to recover.
    """
    if skus:
        picked = ", ".join(_sku_display(s) for s in skus)
        cause = f"No RawMatPriceTrend rows match the selected SKU filter ({picked})."
    else:
        cause = "No RawMatPriceTrend rows are available."
    st.info(f"{what}: {cause} Clear or change the SKU filter to see data.")


def _seasonality_insight(compare_df: pd.DataFrame) -> str:
    """Return a one/two-sentence, numbers-only seasonality insight (Req 7.3).

    Uses the peak-minus-non-peak ``delta`` per SKU to name the SKU with the
    largest peak-season price premium (or, if all deltas are non-positive, the
    SKU whose peak price is least discounted). Every figure quoted is a real
    computed value; nothing is fabricated.
    """
    usable = compare_df.dropna(subset=["delta"])
    if usable.empty:
        return (
            "Not enough peak and non-peak observations to compare seasonal "
            "close prices for the selected SKU(s)."
        )

    top = usable.loc[usable["delta"].idxmax()]
    sku_label = _sku_display(str(top["sku"]))
    delta = float(top["delta"])
    peak = float(top["avg_close_peak"])
    non_peak = float(top["avg_close_non_peak"])

    if delta > 0:
        pct = (delta / non_peak * 100.0) if non_peak else float("nan")
        pct_txt = f" ({pct:,.1f}% above non-peak)" if non_peak else ""
        return (
            f"{sku_label} shows the largest peak-season premium: its average "
            f"close price is {peak:,.0f} IDR in peak weeks versus "
            f"{non_peak:,.0f} IDR in non-peak weeks, a difference of "
            f"{delta:,.0f} IDR{pct_txt}."
        )
    return (
        f"No SKU shows a peak-season premium in the current selection. "
        f"{sku_label} is closest to neutral, averaging {peak:,.0f} IDR in peak "
        f"weeks versus {non_peak:,.0f} IDR in non-peak weeks "
        f"({delta:,.0f} IDR)."
    )


def render(bundle: Optional[DataBundle] = None, filters: Optional[dict] = None) -> None:
    """Render the Raw Material & Seasonality view.

    Args:
        bundle: A pre-loaded :class:`~src.data_loader.DataBundle`. When ``None``,
            the view loads data itself via :func:`~src.data_loader.load_data`
            and surfaces a :class:`~src.data_loader.LoadError` as ``st.error``
            without proceeding to the visualizations (Req 1.6 behavior).
        filters: Optional shared filter dict. Recognized key ``"skus"`` (a list
            of SKU keys) restricts every raw-material visualization to the
            selected SKUs (Req 7.4). ``None``/empty means no restriction.
    """
    st.header("Raw material & seasonality")
    st.caption(
        "Weekly raw-material price trends, harvest success, and peak-season "
        "effects across 2020-2022, from the RawMatPriceTrend data."
    )

    # --- Data-load error gate (Req 1.6) ---------------------------------
    if bundle is None:
        try:
            bundle = load_data()
        except LoadError as exc:
            st.error(
                f"Could not load the workbook: {exc}. Fix the source file, then "
                "reload the page."
            )
            return

    skus = _selected_skus(filters)
    rows = _filter_rows(bundle.raw_mat_trend, skus)

    # --- Raw-material close price time-series (Req 7.1) -----------------
    st.subheader("How have raw-material close prices moved, 2020-2022?")
    if rows.empty:
        _empty_note(skus, "Close-price trend")
    else:
        st.plotly_chart(
            charts.raw_material_price_figure(rows),
            width="stretch",
        )

    # --- Harvest success rate time-series, 0-100 (Req 7.2) --------------
    st.subheader("How reliable has the harvest been, 2020-2022?")
    if rows.empty:
        _empty_note(skus, "Harvest success rate")
    else:
        st.plotly_chart(
            charts.harvest_success_figure(rows),
            width="stretch",
        )

    # --- Peak vs non-peak comparison + insight (Req 7.3) ----------------
    st.subheader("Do raw-material prices carry a peak-season premium?")
    compare_df = seasonality_price_compare(rows)
    if compare_df.empty:
        _empty_note(skus, "Peak vs non-peak comparison")
    else:
        st.plotly_chart(
            charts.seasonality_figure(compare_df),
            width="stretch",
        )

        table = compare_df.copy()
        table.insert(0, "Product", [_sku_display(s) for s in table["sku"]])
        table = table.rename(
            columns={
                "avg_close_peak": "Avg close, peak (IDR)",
                "avg_close_non_peak": "Avg close, non-peak (IDR)",
                "delta": "Peak - non-peak (IDR)",
            }
        ).drop(columns=["sku"])
        st.dataframe(
            table.style.format(
                {
                    "Avg close, peak (IDR)": "{:,.0f}",
                    "Avg close, non-peak (IDR)": "{:,.0f}",
                    "Peak - non-peak (IDR)": "{:,.0f}",
                }
            ),
            width="stretch",
            hide_index=True,
        )
        st.markdown(_seasonality_insight(compare_df))


def main() -> None:
    """Run the view as a standalone Streamlit page."""
    render()


# Auto-run when Streamlit executes this file directly. When ``app.py`` drives
# the pages it sets ``_APP_DRIVES_PAGES`` and calls :func:`render` itself, so
# skip the auto-run here to avoid double-rendering.
if not st.session_state.get("_APP_DRIVES_PAGES"):
    main()
