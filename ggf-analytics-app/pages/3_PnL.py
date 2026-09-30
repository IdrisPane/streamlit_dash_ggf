"""P&L view (Task 11.3) — Profit & Loss analysis for the GGF Analytics App.

Streamlit page for the third analysis pillar (Req 5.1, 5.2, 5.3, 5.4, 5.6 and
the reconciliation reporting of Req 6.3, 6.4). It presents, on the recomputed
profit basis documented in :mod:`src.config`:

- per-SKU total profit and margin % (table + a green-profit/red-loss bar chart);
- the count of negative-profit months per SKU, highlighting the least-profitable
  SKU (Pineapple, 7 of 12 on the real workbook);
- the raw-material vs production cost-driver breakdown (table + stacked bar,
  IDR and % of the SKU's cost base);
- data-driven improvement proposals, each naming a specific SKU and cost driver;
- a P&L reconciliation panel surfacing the discrepancy count, the affected rows
  with signed diffs (2 dp), loss rows per SKU, unprocessable rows, and — front
  and centre — the recomputed profit basis used and *why* the source
  ``Profit&Lost`` / ``TotalCost`` fields are not trusted.

Design direction (project ``DESIGN.md``): a restrained, data-first dashboard.
Semantic red is used only for losses / negative values and green only for
profit; teal is the single accent. Every number is computed at runtime from the
real workbook — nothing is fabricated. Empty and error states state the cause
and the next action.

Contract (shared by every view; wired by ``app.py`` in Task 14):

- ``render(bundle=None, filters=None)`` holds all view logic. When ``bundle`` is
  ``None`` the page loads the data itself (so the page also runs standalone
  under ``streamlit run`` / the multipage nav); a :class:`~src.data_loader.LoadError`
  is shown via ``st.error`` and the view stops (Req 1.6).
- ``filters`` is an optional ``{"skus": [...], "months": [...]}`` mapping applied
  to the P&L figures over the joined frame (Req 5.5, 5.6). A selection that
  yields no rows shows an ``st.info`` empty-result message and retains the
  filter context without crashing (Req 5.6).
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

import pandas as pd
import streamlit as st

# Allow ``import src...`` when this page is run directly by Streamlit (the
# multipage runner executes the file, so the project root is not guaranteed to
# be on ``sys.path``).
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src import charts, config  # noqa: E402
from src.anomalies import detect_reconciliation  # noqa: E402
from src.data_loader import LoadError, load_data  # noqa: E402
from src.metrics import (  # noqa: E402
    cost_driver_breakdown,
    improvement_proposals,
    negative_profit_month_count,
    per_sku_pnl,
)

#: The least-profitable SKU called out explicitly by Req 5.2.
_LEAST_PROFITABLE_SKU = "200100004"


# ---------------------------------------------------------------------------
# Formatting helpers (display only; all figures are computed upstream)
# ---------------------------------------------------------------------------


def _fmt_idr(value: object) -> str:
    """Format a number as thousands-separated IDR, or ``"N/A"`` when missing."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return "N/A"
    try:
        return f"{float(value):,.0f} {config.CURRENCY_LABEL}"
    except (TypeError, ValueError):
        return "N/A"


def _fmt_pct(value: object) -> str:
    """Format a percentage to one decimal place, or ``"N/A"`` when missing."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return "N/A"
    try:
        return f"{float(value):.1f}%"
    except (TypeError, ValueError):
        return "N/A"


def _product_label(sku: str) -> str:
    """Return ``"Product (SKU)"`` for a SKU key, or just the key if unknown."""
    name = config.SKU_MAP.get(str(sku))
    return f"{name} ({sku})" if name else str(sku)


# ---------------------------------------------------------------------------
# Filtering (Req 5.5, 5.6)
# ---------------------------------------------------------------------------


def _apply_filters(
    joined: pd.DataFrame, filters: Optional[dict]
) -> pd.DataFrame:
    """Return ``joined`` restricted to the selected SKUs and months.

    ``filters`` is the shared ``{"skus": [...], "months": [...]}`` mapping. An
    absent, empty, or ``None`` entry means "no restriction on that dimension",
    so an all-empty filter returns the full frame. SKU and month keys are
    compared as strings to match the joined frame's canonical keys
    (``SKU`` is a string key; ``month`` is a ``"YYYY-MM"`` string).
    """
    if joined is None or not filters:
        return joined

    result = joined
    skus = filters.get("skus") or []
    months = filters.get("months") or []

    if skus:
        wanted = {str(s) for s in skus}
        result = result[result["SKU"].astype("string").isin(wanted)]
    if months:
        wanted_m = {str(m) for m in months}
        result = result[result["month"].astype("string").isin(wanted_m)]
    return result


def _filter_caption(filters: Optional[dict]) -> Optional[str]:
    """Return a short caption describing the active filters, or ``None``."""
    if not filters:
        return None
    skus = filters.get("skus") or []
    months = filters.get("months") or []
    parts: list[str] = []
    if skus:
        parts.append("SKUs: " + ", ".join(_product_label(s) for s in skus))
    if months:
        parts.append("Months: " + ", ".join(str(m) for m in months))
    return " | ".join(parts) if parts else None


# ---------------------------------------------------------------------------
# Section renderers
# ---------------------------------------------------------------------------


def _render_per_sku_pnl(joined: pd.DataFrame) -> None:
    """Per-SKU profit and margin % — table + green/red bar chart (Req 5.1)."""
    st.subheader("Profit and margin per SKU")
    st.caption(
        "Total 2022 profit and margin for each product, on the recomputed "
        "profit basis. Green marks a profit, red marks a loss."
    )

    pnl = per_sku_pnl(joined)
    if pnl.empty:
        st.info(
            "No P&L rows for the current selection. Adjust the SKU or month "
            "filter to see per-SKU profit and margin."
        )
        return

    table = pd.DataFrame(
        {
            "SKU": [str(s) for s in pnl.index],
            "Product": [
                pnl.at[s, "product_name"]
                if pd.notna(pnl.at[s, "product_name"])
                else _product_label(s)
                for s in pnl.index
            ],
            "Net revenue": [_fmt_idr(pnl.at[s, "net_revenue"]) for s in pnl.index],
            "Total profit": [_fmt_idr(pnl.at[s, "total_profit"]) for s in pnl.index],
            "Margin %": [_fmt_pct(pnl.at[s, "margin_pct"]) for s in pnl.index],
        }
    )
    st.dataframe(table, hide_index=True, width="stretch")
    st.plotly_chart(
        charts.pnl_by_sku_figure(pnl), width="stretch", key="pnl_by_sku_chart"
    )


def _render_negative_months(joined: pd.DataFrame) -> None:
    """Negative-profit month counts per SKU, Pineapple highlighted (Req 5.2)."""
    st.subheader("Negative-profit months per SKU")
    st.caption(
        "How many of the 12 months in 2022 each SKU made a loss on the "
        "recomputed profit basis. The least-profitable SKU carries the most "
        "loss-making months."
    )

    counts = negative_profit_month_count(joined)
    if not counts:
        st.info(
            "No monthly P&L rows for the current selection. Adjust the filter "
            "to see negative-profit month counts."
        )
        return

    table = pd.DataFrame(
        {
            "SKU": list(counts.keys()),
            "Product": [_product_label(s) for s in counts],
            "Negative-profit months (of 12)": [counts[s] for s in counts],
        }
    )
    st.dataframe(table, hide_index=True, width="stretch")

    least = counts.get(_LEAST_PROFITABLE_SKU)
    if least is not None:
        st.markdown(
            f"**Least-profitable SKU: {_product_label(_LEAST_PROFITABLE_SKU)}** — "
            f"loss-making in **{least} of 12 months**."
        )


def _render_cost_drivers(joined: pd.DataFrame) -> None:
    """Cost-driver breakdown — table (IDR + %) + stacked bar chart (Req 5.3)."""
    st.subheader("Cost-driver breakdown per SKU")
    st.caption(
        "Where each SKU's cost base goes: raw-material cost vs production cost, "
        "in IDR and as a share of that SKU's total cost."
    )

    breakdown = cost_driver_breakdown(joined)
    if breakdown.empty:
        st.info(
            "No cost rows for the current selection. Adjust the SKU or month "
            "filter to see the cost-driver breakdown."
        )
        return

    table = pd.DataFrame(
        {
            "SKU": breakdown["sku"].astype(str),
            "Product": [
                name if pd.notna(name) else _product_label(sku)
                for name, sku in zip(breakdown["product_name"], breakdown["sku"])
            ],
            "Raw-material cost": [_fmt_idr(v) for v in breakdown["raw_material_cost"]],
            "Raw-material %": [_fmt_pct(v) for v in breakdown["raw_material_pct"]],
            "Production cost": [_fmt_idr(v) for v in breakdown["production_cost"]],
            "Production %": [_fmt_pct(v) for v in breakdown["production_pct"]],
            "Total cost base": [_fmt_idr(v) for v in breakdown["total_cost_base"]],
        }
    )
    st.dataframe(table, hide_index=True, width="stretch")
    st.plotly_chart(
        charts.cost_driver_figure(breakdown),
        width="stretch",
        key="cost_driver_chart",
    )


def _render_proposals(joined: pd.DataFrame) -> None:
    """Improvement proposals, each naming a SKU and cost driver (Req 5.4)."""
    st.subheader("Cost-reduction and profitability proposals")
    st.caption(
        "Data-driven opportunities. Each proposal references the specific SKU "
        "and the cost driver from the breakdown that supports it."
    )

    proposals = improvement_proposals(joined)
    if not proposals:
        st.info(
            "No proposals for the current selection — there are no cost rows to "
            "analyse. Adjust the filter to generate proposals."
        )
        return

    for proposal in proposals:
        st.markdown(f"- {proposal.text}")


def _render_reconciliation(bundle) -> None:
    """P&L reconciliation panel + profit-basis statement (Req 6.3, 6.4).

    The reconciliation always runs over the *full* SalesTransaction sheet (not
    the filtered subset) because it is a data-quality statement about the source
    data, not a filtered analytical figure. The recomputed profit basis and the
    reason it is used are stated prominently so stakeholders see why the source
    ``Profit&Lost`` / ``TotalCost`` fields are not trusted.
    """
    st.subheader("P&L reconciliation and profit basis")

    # --- Profit basis, stated prominently (Req 6.4) ----------------------
    st.markdown(
        f"**Profit basis used:** `{config.PROFIT_BASIS_ID}` — "
        f"`{config.PROFIT_BASIS_FORMULA}`"
    )
    st.warning(
        f"**Why the source figures are not trusted.** {config.PROFIT_BASIS_REASON}"
    )

    # --- Reconciliation results over the full transaction sheet ----------
    recon = detect_reconciliation(bundle.sales_transaction)

    total_rows = int(len(bundle.sales_transaction))
    col_a, col_b, col_c = st.columns(3)
    col_a.metric("Reconciliation discrepancies", f"{recon.discrepancy_count:,}")
    col_b.metric("Unprocessable rows", f"{recon.unprocessable_count:,}")
    col_c.metric("Transaction rows checked", f"{total_rows:,}")

    # --- Affected rows with signed diffs to 2 dp (Req 6.3) ---------------
    if recon.discrepancy_count > 0:
        st.markdown(
            f"**Affected rows ({recon.discrepancy_count:,}).** Signed discrepancy "
            "amounts in IDR to two decimal places. "
            "`P&L diff = Profit&Lost - (TotalNetSellingPrice - TotalCost)`; "
            "`Cost diff = TotalCost - (TotalRawMaterialCost + TotalProductionCost)`."
        )
        disc = recon.discrepancies.copy()
        disc_table = pd.DataFrame(
            {
                "Row": disc["row_index"].astype(int),
                "P&L diff (IDR)": [f"{v:,.2f}" for v in disc["signed_diff_pnl"]],
                "Cost diff (IDR)": [f"{v:,.2f}" for v in disc["signed_diff_cost"]],
            }
        )
        st.dataframe(disc_table, hide_index=True, width="stretch")
    else:
        st.info(
            "No reconciliation discrepancies were found in the SalesTransaction "
            "sheet."
        )

    # --- Loss rows per SKU on the recomputed basis (Req 6.5) -------------
    st.markdown("**Loss-making rows per SKU** (recomputed profit below zero).")
    if recon.loss_rows_per_sku:
        loss_table = pd.DataFrame(
            {
                "SKU": [str(s) for s in recon.loss_rows_per_sku],
                "Product": [
                    _product_label(s) if s else "(unattributed)"
                    for s in recon.loss_rows_per_sku
                ],
                "Loss-making rows": list(recon.loss_rows_per_sku.values()),
            }
        )
        st.dataframe(loss_table, hide_index=True, width="stretch")
    else:
        st.info("No loss-making rows on the recomputed profit basis.")

    # --- Unprocessable rows and offending field (Req 6.6) ----------------
    if recon.unprocessable_count > 0:
        st.markdown(
            f"**Unprocessable rows ({recon.unprocessable_count:,}).** These rows "
            "are excluded from the reconciliation because a required field is "
            "missing or non-numeric; the offending field is named per row."
        )
        unproc_table = pd.DataFrame(
            {
                "Row": list(recon.unprocessable_fields.keys()),
                "Offending field": list(recon.unprocessable_fields.values()),
            }
        )
        st.dataframe(unproc_table, hide_index=True, width="stretch")
    else:
        st.caption(
            "All transaction rows had complete, numeric fields for the "
            "reconciliation checks."
        )


# ---------------------------------------------------------------------------
# Page entry point
# ---------------------------------------------------------------------------


def render(bundle=None, filters: Optional[dict] = None) -> None:
    """Render the P&L view (Req 5.1, 5.2, 5.3, 5.4, 5.6, 6.3, 6.4).

    Args:
        bundle: A loaded :class:`~src.data_loader.DataBundle`. When ``None`` the
            view loads the data itself and shows a :class:`LoadError` via
            ``st.error`` without proceeding (Req 1.6).
        filters: Optional ``{"skus": [...], "months": [...]}`` applied to the
            P&L figures. A selection matching no rows shows an empty-result
            message and retains the filter context (Req 5.6).
    """
    st.title("Profit & Loss")
    st.caption(
        "Per-SKU profitability, cost drivers, and data-quality reconciliation "
        "for January–December 2022. All figures use the recomputed profit "
        "basis stated below."
    )

    # --- Load / error gate (Req 1.6) -------------------------------------
    if bundle is None:
        try:
            bundle = load_data()
        except LoadError as exc:
            st.error(
                f"Could not load the source workbook: {exc}. "
                "Check that FruitsTrxDatasets.xlsx is present under data/ (or "
                "set GGF_WORKBOOK_PATH), then reload the page."
            )
            return

    joined = bundle.joined
    if joined is None:
        st.error(
            "The data bundle has no joined transactions to analyse. Reload the "
            "app to rebuild the data model."
        )
        return

    # --- Apply shared filters (Req 5.5) ----------------------------------
    caption = _filter_caption(filters)
    if caption:
        st.caption(f"Active filters — {caption}")

    filtered = _apply_filters(joined, filters)

    # --- Empty-result handling (Req 5.6) ---------------------------------
    if filtered is None or filtered.empty:
        st.info(
            "No P&L data is available for the selected filters. The filter "
            "controls are unchanged — widen the SKU or month selection to see "
            "results."
        )
        # The reconciliation panel is a source-data statement, so still show
        # the profit basis and reconciliation below even on an empty subset.
        st.divider()
        _render_reconciliation(bundle)
        return

    # --- Analysis sections ----------------------------------------------
    _render_per_sku_pnl(filtered)
    st.divider()
    _render_negative_months(filtered)
    st.divider()
    _render_cost_drivers(filtered)
    st.divider()
    _render_proposals(filtered)
    st.divider()
    _render_reconciliation(bundle)


def main() -> None:
    """Standalone entry point (multipage nav / ``streamlit run``).

    Streamlit executes a page script top-to-bottom on load, so ``main`` is
    invoked once at import time below. When ``app.py`` (Task 14) drives the page
    it calls :func:`render` directly with a shared bundle and filters instead.
    """
    render()


# Auto-run when Streamlit executes this file directly. When ``app.py`` drives
# the pages it sets ``_APP_DRIVES_PAGES`` and calls :func:`render` itself, so
# skip the auto-run here to avoid double-rendering.
if not st.session_state.get("_APP_DRIVES_PAGES"):
    main()
