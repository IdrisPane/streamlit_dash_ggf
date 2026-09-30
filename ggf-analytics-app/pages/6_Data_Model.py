"""Data Model view (Task 11.6) — data documentation for the GGF Analytics App.

Streamlit page for the first analysis pillar (Req 2.1, 2.2, 2.3, 2.4, 2.5). It
documents the data model so stakeholders can understand how the four source
tables relate and trust the analysis built on them:

- the four source tables (MstProduct, SalesTarget, SalesTransaction,
  RawMatPriceTrend), each with its name, row count, column names, and a short
  1-500 character description (Req 2.1);
- the inter-table relationships — the SKU join linking all four tables and the
  month/period alignment between SalesTransaction and SalesTarget — naming the
  two joined tables and the joining column(s) for each (Req 2.2);
- the analysis process as an ordered sequence of stages, from raw data through
  cleaning/normalization and joining to the four analysis pillars, each with a
  short description (Req 2.3);
- the source column-name normalizations from ``bundle.normalization_map`` as
  original -> normalized pairs, including ``TranscationDate`` and
  ``Profit&Lost`` (Req 2.4);
- a small data-quality note surfacing the real parse-error and unmatched-key
  counts (0 / 0 on the real workbook).

Each documentation section is wrapped independently so that if one fails to
build, an ``st.error`` names that section and the remaining sections still
render (Req 2.5).

Design direction (project ``DESIGN.md``): a restrained, data-first dashboard.
No emoji, no decorative icons, no glow — descriptions and counts are computed
from the real bundle at runtime. Neutral tables carry the documentation; the
single teal accent is left to focal metrics elsewhere.

Contract (shared by every view; wired by ``app.py`` in Task 14):

- ``render(bundle=None, filters=None)`` holds all view logic. When ``bundle`` is
  ``None`` the page loads the data itself (so the page also runs standalone
  under ``streamlit run`` / the multipage nav); a :class:`~src.data_loader.LoadError`
  is shown via ``st.error`` and the view stops (Req 1.6).
- ``filters`` is accepted for interface parity with the other views but is not
  applied here: the data model documents the full source tables regardless of
  any SKU/month selection.
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
from src.data_loader import LoadError, load_data  # noqa: E402


# ---------------------------------------------------------------------------
# Static documentation content (descriptions are 1-500 chars, per Req 2.1/2.3)
#
# Row counts and column names are read from the live bundle at render time; only
# the human-authored prose descriptions live here.
# ---------------------------------------------------------------------------

#: Table name -> short description (1-500 chars). Keyed by source sheet name.
_TABLE_DESCRIPTIONS: dict[str, str] = {
    "MstProduct": (
        "Product master. One row per SKU catalogued by GGF, carrying the "
        "product name, category, unit of measure, currency, and list price. "
        "The analysis restricts this to the three fruit-juice SKUs and uses it "
        "to attach a readable product name to every transaction."
    ),
    "SalesTarget": (
        "Monthly sales targets. One row per SKU per calendar month of 2022, "
        "giving the SalesTarget the transaction totals are measured against. "
        "It supplies the denominator for target-attainment across the sales "
        "and executive-summary analyses."
    ),
    "SalesTransaction": (
        "The 2022 transaction ledger for a single outlet (Outlet 1). Each row "
        "records quantity, selling price, discounts, net selling price, the "
        "raw-material and production cost components, the aggregate cost, and "
        "the source Profit&Lost value. It is the primary fact table for the "
        "sales, P&L, and predictive analyses."
    ),
    "RawMatPriceTrend": (
        "Weekly raw-material price history per SKU spanning 2020 through 2022, "
        "with open/close price, moving average, harvest success rate, and a "
        "peak-season flag. It underpins the raw-material and seasonality "
        "analysis of margin volatility and cost drivers."
    ),
}

#: The inter-table relationships (Req 2.2): (name, table A, table B, columns).
_RELATIONSHIPS: list[dict[str, str]] = [
    {
        "name": "SKU join (product)",
        "left": "SalesTransaction",
        "right": "MstProduct",
        "columns": "SKU = SKU",
        "detail": (
            "Every transaction is matched to its product master by SKU, "
            "restricted to the three fruit-juice SKUs."
        ),
    },
    {
        "name": "SKU join (target)",
        "left": "SalesTransaction",
        "right": "SalesTarget",
        "columns": "SKU = SKU",
        "detail": (
            "Transactions and their monthly targets share the SKU key so "
            "attainment is computed per product."
        ),
    },
    {
        "name": "SKU join (raw material)",
        "left": "SalesTransaction",
        "right": "RawMatPriceTrend",
        "columns": "SKU = SKU",
        "detail": (
            "The raw-material price history is linked to products by SKU, "
            "tying cost trends to the same three SKUs analysed elsewhere."
        ),
    },
    {
        "name": "Month / period alignment",
        "left": "SalesTransaction",
        "right": "SalesTarget",
        "columns": "month(transaction_date) = month(Period)",
        "detail": (
            "Each transaction is aligned to its target by calendar month "
            "across the 12 monthly periods of Jan-Dec 2022."
        ),
    },
]

#: The ordered analysis stages (Req 2.3): (stage name, 1-500 char description).
_ANALYSIS_STAGES: list[tuple[str, str]] = [
    (
        "1. Raw data",
        "Read exactly the four source sheets (MstProduct, SalesTarget, "
        "SalesTransaction, RawMatPriceTrend) from FruitsTrxDatasets.xlsx as "
        "they arrive, typos and all.",
    ),
    (
        "2. Cleaning and normalization",
        "Rename the two source-typo columns to canonical names, coerce date "
        "and numeric columns to their real types (recording any unparseable "
        "cell as a parse error and nulling it), and key SKU as a string "
        "everywhere.",
    ),
    (
        "3. Joining",
        "Left-join transactions to the product master by SKU and to targets by "
        "SKU and calendar month, restricted to the three valid SKUs. Unmatched "
        "rows are retained with null joined fields and recorded so nothing is "
        "silently dropped.",
    ),
    (
        "4. Sales performance analysis",
        "Compare actual quantity and value against monthly targets per SKU, "
        "chart chronological trends, and flag anomalies (>2 std or attainment "
        "below 50% / above 150%).",
    ),
    (
        "5. Profit & Loss analysis",
        "Recompute profit from primitive net revenue and cost components, "
        "break down raw-material vs production cost drivers, reconcile the "
        "inconsistent source fields, and derive improvement proposals.",
    ),
    (
        "6. Raw material and seasonality analysis",
        "Trace weekly close price and harvest success rate per SKU and compare "
        "average price between peak and non-peak seasons to explain margin "
        "volatility.",
    ),
    (
        "7. Predictive analysis",
        "Fit a forecast to each SKU's 12 monthly observations, project sales "
        "and profit 12 months ahead, and state whether the forecast supports "
        "the expansion verdict.",
    ),
]


# ---------------------------------------------------------------------------
# Section renderers (each called under a try/except by ``render``, Req 2.5)
# ---------------------------------------------------------------------------


def _table_frames(bundle) -> list[tuple[str, pd.DataFrame]]:
    """Return the four source tables as ``(name, frame)`` in a fixed order."""
    return [
        ("MstProduct", bundle.mst_product),
        ("SalesTarget", bundle.sales_target),
        ("SalesTransaction", bundle.sales_transaction),
        ("RawMatPriceTrend", bundle.raw_mat_trend),
    ]


def _render_source_tables(bundle) -> None:
    """Document each source table: name, row count, columns, description (2.1)."""
    st.subheader("Source tables")
    st.caption(
        "The four sheets loaded from FruitsTrxDatasets.xlsx. Row counts and "
        "columns are read from the loaded data; descriptions explain each "
        "table's role in the analysis."
    )

    for name, frame in _table_frames(bundle):
        row_count = int(len(frame))
        columns = list(frame.columns)
        description = _TABLE_DESCRIPTIONS.get(name, "")

        st.markdown(f"**{name}** — {row_count:,} rows, {len(columns)} columns")
        st.markdown(description)
        col_table = pd.DataFrame({"Column": [str(c) for c in columns]})
        st.dataframe(col_table, hide_index=True, width="stretch")


def _render_relationships(bundle) -> None:
    """Document inter-table relationships with joined tables + columns (2.2)."""
    st.subheader("Table relationships")
    st.caption(
        "How the four tables connect. The SKU key links all four tables; the "
        "month alignment matches each transaction to its monthly target."
    )

    rel_table = pd.DataFrame(
        {
            "Relationship": [r["name"] for r in _RELATIONSHIPS],
            "From table": [r["left"] for r in _RELATIONSHIPS],
            "To table": [r["right"] for r in _RELATIONSHIPS],
            "Joining column(s)": [r["columns"] for r in _RELATIONSHIPS],
            "Notes": [r["detail"] for r in _RELATIONSHIPS],
        }
    )
    st.dataframe(rel_table, hide_index=True, width="stretch")


def _render_process(bundle) -> None:
    """Document the analysis process as ordered stages (Req 2.3)."""
    st.subheader("Analysis process")
    st.caption(
        "The ordered path from raw source data through cleaning and joining to "
        "the four analysis pillars."
    )

    stage_table = pd.DataFrame(
        {
            "Stage": [name for name, _ in _ANALYSIS_STAGES],
            "Description": [desc for _, desc in _ANALYSIS_STAGES],
        }
    )
    st.dataframe(stage_table, hide_index=True, width="stretch")


def _render_normalizations(bundle) -> None:
    """Document original -> normalized column-name pairs (Req 2.4)."""
    st.subheader("Column-name normalizations")
    st.caption(
        "Source-column typos renamed once to canonical names used by every "
        "downstream table."
    )

    normalization_map = bundle.normalization_map or {}
    if not normalization_map:
        st.info("No column-name normalizations were recorded for this dataset.")
        return

    norm_table = pd.DataFrame(
        {
            "Original source column": list(normalization_map.keys()),
            "Normalized name": list(normalization_map.values()),
        }
    )
    st.dataframe(norm_table, hide_index=True, width="stretch")


def _render_data_quality(bundle) -> None:
    """Surface real parse-error and unmatched-key counts as a note (optional)."""
    st.subheader("Data-quality note")
    st.caption(
        "Counts recorded while loading and joining the source data. Parse "
        "errors are cells that could not be coerced to their type; unmatched "
        "keys are transaction rows with no matching product or target."
    )

    parse_count = len(bundle.parse_errors or [])
    unmatched_count = len(bundle.unmatched_keys or [])

    col_a, col_b = st.columns(2)
    col_a.metric("Parse errors", f"{parse_count:,}")
    col_b.metric("Unmatched join keys", f"{unmatched_count:,}")

    if parse_count == 0 and unmatched_count == 0:
        st.caption(
            "The source data loaded cleanly: every cell coerced to its type "
            "and every transaction matched a product and a monthly target."
        )


# ---------------------------------------------------------------------------
# Page entry point
# ---------------------------------------------------------------------------


def _render_section(title: str, builder, bundle) -> None:
    """Run one section builder, isolating its failure (Req 2.5).

    If ``builder`` raises, an ``st.error`` names the section that could not be
    built and the caller continues with the remaining sections, so a single
    failing section never blanks the whole page.
    """
    try:
        builder(bundle)
    except Exception as exc:  # noqa: BLE001 - deliberately broad per Req 2.5.
        st.error(
            f'The "{title}" documentation section could not be loaded: {exc}. '
            "The other sections below are unaffected."
        )


def render(bundle=None, filters: Optional[dict] = None) -> None:
    """Render the Data Model view (Req 2.1, 2.2, 2.3, 2.4, 2.5).

    Args:
        bundle: A loaded :class:`~src.data_loader.DataBundle`. When ``None`` the
            view loads the data itself and shows a :class:`LoadError` via
            ``st.error`` without proceeding (Req 1.6).
        filters: Accepted for interface parity with the other views; not
            applied here since the data model documents the full source tables.
    """
    st.title("Data Model")
    st.caption(
        "How the source data is structured, related, cleaned, and analysed. "
        "Row counts, columns, and normalization pairs are read from the loaded "
        "workbook at runtime."
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

    # --- Documentation sections, each isolated (Req 2.5) -----------------
    _render_section("Source tables", _render_source_tables, bundle)
    st.divider()
    _render_section("Table relationships", _render_relationships, bundle)
    st.divider()
    _render_section("Analysis process", _render_process, bundle)
    st.divider()
    _render_section("Column-name normalizations", _render_normalizations, bundle)
    st.divider()
    _render_section("Data-quality note", _render_data_quality, bundle)


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
