"""Assumptions view (Task 11.7) — stated assumptions & data caveats (Req 9.1-9.5).

This Streamlit page documents, in one place, everything a GGF top-management
reader needs to judge how far to trust the analysis:

- The assumptions underlying each analysis, grouped and **labeled by analysis**
  under Sales, P&L, and Predictive headings (Req 9.1). The predictive
  assumptions are pulled from the fitted :class:`~src.forecasting.ForecastResult`
  so they stay in lock-step with the forecasting code; the sales and P&L
  assumptions are stated here, grounded in the real data (single outlet, monthly
  grain, 2022 only, recomputed profit basis).
- The **profit-basis choice** used to resolve the reconciliation discrepancy:
  which basis was selected and the **direction and magnitude** of the
  discrepancy it resolves, computed from
  :func:`~src.anomalies.detect_reconciliation` over the real SalesTransaction
  sheet so the statement is data-grounded (Req 9.2).
- The forecast's **assumptions and known limitations**, stating explicitly that
  it is derived from a sample of :data:`config.MIN_OBSERVATIONS` (12) monthly
  observations (Req 9.3).
- The **single-outlet caveat**: the 2022 data covers only ``Outlet 1`` and
  findings are not generalizable beyond it, verified from the transaction
  sheet's ``Outlet`` column (Req 9.4).
- A **missing-analysis indicator**: if any of the three analyses has no
  documented assumptions available, the view flags which analysis is missing
  rather than silently omitting it (Req 9.5).

Design direction (project ``DESIGN.md``): restrained, data-first, calm; no
emoji or glow. Every figure comes from the real workbook at runtime.

Contract (shared by every view; wired by ``app.py`` in Task 14):

- ``render(bundle=None, filters=None)`` holds all view logic. When ``bundle`` is
  ``None`` the page loads the data itself (so it runs standalone under
  ``streamlit run`` / the multipage nav); a :class:`~src.data_loader.LoadError`
  is surfaced via ``st.error`` and the view stops (Req 1.6).
- ``filters`` is accepted for interface parity with the other views but is not
  applied here: assumptions and data-quality caveats are statements about the
  whole 2022 dataset, not a filtered slice.
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

from src import config  # noqa: E402
from src.anomalies import detect_reconciliation  # noqa: E402
from src.data_loader import LoadError, load_data  # noqa: E402
from src.forecasting import forecast_all  # noqa: E402


# ---------------------------------------------------------------------------
# Assumption content
#
# Sales and P&L assumptions are stated here (there is no single source function
# for them the way forecasting owns its own). They are grounded in the verified
# dataset facts: a single outlet, monthly grain, 2022 only, and the recomputed
# profit basis. The predictive assumptions are read from the fitted forecast
# results so they never drift from the forecasting implementation.
# ---------------------------------------------------------------------------


def _sales_assumptions() -> list[str]:
    """Return the sales-analysis assumptions (Req 9.1)."""
    return [
        "Sales are measured at monthly grain (Jan-Dec 2022); within-month "
        "timing and daily variation are not analysed.",
        "The data covers a single outlet (Outlet 1) and one calendar year "
        "(2022), so sales patterns are specific to that outlet and year.",
        "Each transaction's SKU is one of the three catalogued products "
        "(Orange, Mango, Pineapple Juice); rows are joined to targets by SKU "
        "and calendar month.",
        "A month with no recorded sales for an SKU is treated as zero sales "
        "(not missing), so trends and target attainment cover all 12 months.",
        "Target attainment uses sales value against the SalesTarget for that "
        "SKU and month; a zero target is reported as not-applicable rather "
        "than a percentage.",
    ]


def _pnl_assumptions() -> list[str]:
    """Return the P&L-analysis assumptions (Req 9.1)."""
    return [
        "Profit is recomputed from primitive components on the "
        f"`{config.PROFIT_BASIS_ID}` basis "
        f"(`{config.PROFIT_BASIS_FORMULA}`); the source Profit&Lost and "
        "TotalCost fields are not trusted because they are internally "
        "inconsistent on every row.",
        "Total cost is taken as raw-material cost plus production cost; the "
        "cost-driver breakdown and margins are computed on that recomputed "
        "cost base.",
        "A loss-making month is one whose recomputed profit is below zero; "
        "negative-profit month counts use this basis.",
        "P&L figures cover the single outlet (Outlet 1) over Jan-Dec 2022 and "
        "exclude any tax, overhead, or financing costs not present in the "
        "source columns.",
    ]


def _predictive_assumptions(bundle) -> list[str]:
    """Return the predictive assumptions, read from the fitted forecast (Req 9.1).

    Pulls the assumptions list off the first fitted
    :class:`~src.forecasting.ForecastResult` so the documented assumptions match
    the forecasting code exactly. Falls back to the first result (even if
    excluded) so an all-excluded edge case still surfaces its recorded reason
    rather than appearing to have no assumptions.
    """
    results = forecast_all(bundle.joined)
    if not results:
        return []
    fitted = [r for r in results if not r.excluded]
    source = fitted[0] if fitted else results[0]
    return list(source.assumptions)


# ---------------------------------------------------------------------------
# Section renderers
# ---------------------------------------------------------------------------


def _render_labeled_assumptions(groups: dict[str, list[str]]) -> None:
    """Render each analysis's assumptions under its own heading (Req 9.1, 9.5).

    Each group is labeled by the analysis it applies to. If a group has no
    documented assumptions, it is flagged with an explicit missing-analysis
    indicator rather than being omitted silently (Req 9.5).
    """
    st.header("Assumptions by analysis")
    st.caption(
        "The assumptions underlying each analysis, labeled by the analysis "
        "they apply to."
    )

    missing: list[str] = []
    for analysis, assumptions in groups.items():
        st.subheader(analysis)
        if not assumptions:
            missing.append(analysis)
            st.warning(
                f"No documented assumptions are available for the {analysis} "
                "analysis. This is flagged here rather than omitted so the gap "
                "is visible."
            )
            continue
        for assumption in assumptions:
            st.markdown(f"- {assumption}")

    # A single consolidated indicator naming every analysis that is missing
    # assumptions (Req 9.5), in addition to the per-group warnings above.
    if missing:
        st.error(
            "Missing documented assumptions for: "
            + ", ".join(missing)
            + ". These analyses need their assumptions documented before the "
            "findings can be relied upon."
        )


def _render_profit_basis(bundle) -> None:
    """Document the profit-basis choice and the discrepancy it resolves (Req 9.2).

    States which basis was selected and computes the **direction and magnitude**
    of the reconciliation discrepancy it resolves from
    :func:`~src.anomalies.detect_reconciliation` over the full SalesTransaction
    sheet, so the statement is grounded in the real data rather than asserted.
    """
    st.header("Profit-basis choice and the discrepancy it resolves")

    st.markdown(
        f"**Basis selected:** `{config.PROFIT_BASIS_ID}` — "
        f"`{config.PROFIT_BASIS_FORMULA}`."
    )
    st.markdown(config.PROFIT_BASIS_REASON)

    recon = detect_reconciliation(bundle.sales_transaction)
    total_rows = int(len(bundle.sales_transaction))

    if recon.discrepancy_count == 0:
        st.info(
            "No reconciliation discrepancies were found in the SalesTransaction "
            "sheet, so the recomputed basis and the source fields agree."
        )
        return

    disc = recon.discrepancies
    # signed_diff_pnl = profit_source - (TotalNetSellingPrice - TotalCost)
    # signed_diff_cost = TotalCost - (TotalRawMaterialCost + TotalProductionCost)
    pnl_diff = disc["signed_diff_pnl"].astype(float)
    cost_diff = disc["signed_diff_cost"].astype(float)

    pnl_total = float(pnl_diff.sum())
    pnl_mean_abs = float(pnl_diff.abs().mean())
    cost_total = float(cost_diff.sum())
    cost_mean_abs = float(cost_diff.abs().mean())

    def _direction(total: float) -> str:
        if total > 0:
            return "overstates"
        if total < 0:
            return "understates"
        return "nets to zero against"

    pnl_dir = _direction(pnl_total)
    cost_dir = _direction(cost_total)

    st.markdown(
        f"The reconciliation flags **{recon.discrepancy_count} of {total_rows}** "
        "transaction rows. The recomputed basis resolves two source "
        "inconsistencies, whose direction and magnitude are:"
    )
    st.markdown(
        f"- **Profit&Lost vs. (net revenue − TotalCost):** the source "
        f"Profit&Lost figure {pnl_dir} the net-minus-cost identity by a total "
        f"of **{pnl_total:,.2f} {config.CURRENCY_LABEL}** across flagged rows "
        f"(mean absolute discrepancy **{pnl_mean_abs:,.2f} "
        f"{config.CURRENCY_LABEL}** per row)."
    )
    st.markdown(
        f"- **TotalCost vs. (raw-material + production cost):** the source "
        f"TotalCost {cost_dir} the sum of its primitive cost components by a "
        f"total of **{cost_total:,.2f} {config.CURRENCY_LABEL}** across flagged "
        f"rows (mean absolute discrepancy **{cost_mean_abs:,.2f} "
        f"{config.CURRENCY_LABEL}** per row)."
    )
    st.caption(
        "Direction is the sign of the summed signed discrepancy: 'overstates' "
        "means the source aggregate is larger than the recomputed identity, "
        "'understates' means it is smaller. Because both aggregate fields are "
        "unreliable, the analysis recomputes profit from primitive net revenue "
        "and primitive cost components instead."
    )


def _render_forecast_limitations(bundle) -> None:
    """Document forecast assumptions/limitations and the 12-obs basis (Req 9.3)."""
    st.header("Forecast assumptions and limitations")
    st.markdown(
        f"The forecast is derived from a sample of **{config.MIN_OBSERVATIONS} "
        "monthly observations** per SKU (Jan-Dec 2022) — the only history "
        "available. With so few points the forecast is indicative, not "
        "precise, and the following limitations apply:"
    )

    predictive = _predictive_assumptions(bundle)
    if not predictive:
        st.warning(
            "No documented forecast assumptions are available — this is flagged "
            "rather than omitted."
        )
    else:
        for assumption in predictive:
            st.markdown(f"- {assumption}")

    st.markdown("**Known limitations:**")
    for limitation in (
        f"Only {config.MIN_OBSERVATIONS} monthly points per SKU: a full "
        "seasonal cycle cannot be estimated, so the model captures level and "
        "trend only.",
        "Confidence in the forecast is low relative to a multi-year history; "
        "it should inform direction, not exact figures.",
        "The forecast assumes 2022 conditions (pricing, costs, demand) persist "
        "and does not model shocks, new products, or new outlets.",
    ):
        st.markdown(f"- {limitation}")


def _render_single_outlet(bundle) -> None:
    """Document the single-outlet caveat, verified from the data (Req 9.4)."""
    st.header("Single-outlet caveat")

    outlets: list[str] = []
    txn = bundle.sales_transaction
    if txn is not None and "Outlet" in txn.columns:
        outlets = sorted(
            {str(o) for o in txn["Outlet"].dropna().unique()}
        )

    if outlets:
        outlet_list = ", ".join(outlets)
        st.markdown(
            f"The 2022 SalesTransaction data covers a **single outlet** "
            f"({outlet_list}). Every sales, P&L, and predictive finding in this "
            "app is specific to that outlet."
        )
        if len(outlets) > 1:
            st.warning(
                "More than one outlet was found in the data. The single-outlet "
                "assumption below no longer holds — review the findings before "
                "generalizing."
            )
    else:
        st.markdown(
            "The 2022 SalesTransaction data covers a **single outlet "
            '("Outlet 1")**. Every sales, P&L, and predictive finding in this '
            "app is specific to that outlet."
        )

    st.warning(
        "Findings are **not generalizable beyond this outlet**. Expansion "
        "decisions that assume other locations, customer bases, or years will "
        "behave the same are not supported by this data."
    )


# ---------------------------------------------------------------------------
# Page entry point
# ---------------------------------------------------------------------------


def render(bundle=None, filters: Optional[dict] = None) -> None:
    """Render the Assumptions view (Req 9.1-9.5).

    Args:
        bundle: A loaded :class:`~src.data_loader.DataBundle`. When ``None`` the
            view loads the data itself and surfaces a
            :class:`~src.data_loader.LoadError` via ``st.error`` without
            proceeding (Req 1.6).
        filters: Accepted for interface parity with the other views but not
            applied — assumptions and data caveats describe the whole 2022
            dataset, not a filtered slice.
    """
    st.title("Assumptions and data caveats")
    st.caption(
        "What the sales, P&L, and predictive analyses assume, why profit is "
        "recomputed, and how far the findings can be trusted."
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

    if bundle.joined is None:
        st.error(
            "The data bundle has no joined transactions to document. Reload the "
            "app to rebuild the data model."
        )
        return

    # --- Assumptions by analysis, labeled + missing-analysis guard (9.1/9.5)
    groups = {
        "Sales": _sales_assumptions(),
        "P&L": _pnl_assumptions(),
        "Predictive": _predictive_assumptions(bundle),
    }
    _render_labeled_assumptions(groups)
    st.divider()

    # --- Profit-basis choice + discrepancy direction/magnitude (9.2) ------
    _render_profit_basis(bundle)
    st.divider()

    # --- Forecast assumptions, limitations, 12-observation basis (9.3) ----
    _render_forecast_limitations(bundle)
    st.divider()

    # --- Single-outlet caveat (9.4) ---------------------------------------
    _render_single_outlet(bundle)


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
