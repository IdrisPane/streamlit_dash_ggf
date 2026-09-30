"""Example / integration tests for the Streamlit views (Task 11.8).

These are example (not property-based) tests that exercise the seven page
modules under ``pages/`` through Streamlit's ``AppTest`` harness. They assert
rendered-content requirements that do not vary with input:

- Data Model view content — source-table row counts and the two normalization
  pairs (Req 2.1-2.5).
- P&L view — the recomputed-profit-basis identifier and the reason text
  (Req 6.4).
- Assumptions view — the analysis-labeled assumption groups, the single-outlet
  ("Outlet 1") caveat, the "not generalizable" statement, the twelve-observation
  basis, and the missing-analysis indicator machinery (Req 9.1-9.5).
- Empty-state messages for each filterable view: Sales (Req 4.6), P&L (Req 5.6),
  Raw Material (Req 7.5), and the shared no-data behaviour (Req 10.5). The empty
  path is reached by driving each page's ``render(bundle, filters)`` with an
  impossible filter selection through an inline AppTest script.

Design notes
------------
Every page loads the real workbook and some fit Holt forecasts on run, so the
default 3s AppTest timeout is far too short — all runs use ``timeout=120``. To
keep the suite fast we run each page **once** and make several assertions per
run rather than re-running per assertion. Pages are executed standalone via
``AppTest.from_file`` (each page exposes ``render(bundle=None, filters=None)``
and calls ``main()`` on import, so running the file top-to-bottom renders it).

The empty-state tests use ``AppTest.from_string`` with a small inline script
that loads a real :class:`~src.data_loader.DataBundle` and calls the page's
``render(bundle, filters=<impossible>)`` directly. Because the page filenames
begin with digits (not importable module names) the script loads each page
module by file path with :mod:`importlib`.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

# --- Locations -------------------------------------------------------------

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_PAGES = _PROJECT_ROOT / "pages"

_DATA_MODEL_PAGE = str(_PAGES / "6_Data_Model.py")
_PNL_PAGE = str(_PAGES / "3_PnL.py")
_ASSUMPTIONS_PAGE = str(_PAGES / "7_Assumptions.py")
_SALES_PAGE = str(_PAGES / "2_Sales_Performance.py")
_RAW_MATERIAL_PAGE = str(_PAGES / "4_Raw_Material.py")

#: A generous timeout: pages load the workbook and may fit forecasts on run.
_RUN_TIMEOUT = 120


# --- Text-collection helpers ----------------------------------------------


def _all_text(at: AppTest) -> str:
    """Concatenate the visible text from every text-bearing element.

    Pulls from titles, headers, subheaders, captions, markdown, and the info /
    warning / error message blocks so a single ``in`` check can look for any
    rendered string regardless of which element type produced it.
    """
    chunks: list[str] = []
    for collection in (
        at.title,
        at.header,
        at.subheader,
        at.markdown,
        at.caption,
        at.info,
        at.warning,
        at.error,
        at.success,
    ):
        for element in collection:
            value = getattr(element, "value", None)
            if value is not None:
                chunks.append(str(value))
    return "\n".join(chunks)


def _dataframe_text(at: AppTest) -> str:
    """Concatenate a string form of every rendered dataframe.

    Row counts and normalization pairs land in ``st.dataframe`` tables, so this
    stringifies each frame (values and column headers) for substring checks.
    """
    chunks: list[str] = []
    for element in at.dataframe:
        frame = getattr(element, "value", None)
        if frame is None:
            continue
        try:
            chunks.append(frame.to_string())
        except Exception:  # noqa: BLE001 - best-effort text extraction
            chunks.append(str(frame))
    return "\n".join(chunks)


def _run_page(page_path: str) -> AppTest:
    """Run a page standalone and assert it raised no uncaught exception."""
    at = AppTest.from_file(page_path, default_timeout=_RUN_TIMEOUT)
    at.run(timeout=_RUN_TIMEOUT)
    assert not at.exception, f"{page_path} raised: {list(at.exception)}"
    return at


# ---------------------------------------------------------------------------
# Data Model view content (Req 2.1-2.5)
# ---------------------------------------------------------------------------


def test_data_model_view_shows_row_counts_and_normalization_pairs():
    """The Data Model view renders the four source tables' row counts and the
    two column-name normalization pairs without error (Req 2.1, 2.4)."""
    at = _run_page(_DATA_MODEL_PAGE)

    text = _all_text(at)
    table_text = _dataframe_text(at)
    combined = text + "\n" + table_text

    # Title and the four documented tables are present (Req 2.1).
    assert "Data Model" in text
    for table_name in (
        "MstProduct",
        "SalesTarget",
        "SalesTransaction",
        "RawMatPriceTrend",
    ):
        assert table_name in combined, f"missing table documentation for {table_name}"

    # Row counts rendered as "<n> rows" (MstProduct 3, SalesTarget/Transaction
    # 36, RawMatPriceTrend 471) — the page formats them as "N rows" (Req 2.1).
    for row_count in ("3 rows", "36 rows", "471 rows"):
        assert row_count in text, f"missing row-count marker '{row_count}'"

    # Both normalization pairs appear as original -> normalized (Req 2.4). The
    # pairs live in a rendered dataframe.
    assert "TranscationDate" in combined
    assert "transaction_date" in combined
    assert "Profit&Lost" in combined
    assert "profit_source" in combined


# ---------------------------------------------------------------------------
# P&L view — profit-basis id + reason (Req 6.4)
# ---------------------------------------------------------------------------


def test_pnl_view_shows_profit_basis_id_and_reason():
    """The P&L view displays the recomputed-profit-basis identifier and the
    reason it was chosen (Req 6.4)."""
    at = _run_page(_PNL_PAGE)

    text = _all_text(at)

    # The profit-basis identifier from config.PROFIT_BASIS_ID.
    assert "net_minus_primitive_costs" in text

    # The reason text: config.PROFIT_BASIS_REASON explains the source columns
    # are internally inconsistent on every row.
    assert "internally inconsistent" in text

    # The basis formula is surfaced too, anchoring the "why".
    assert "TotalNetSellingPrice" in text


# ---------------------------------------------------------------------------
# Assumptions view — labeled blocks, single-outlet, 12-obs, missing indicator
# (Req 9.1-9.5)
# ---------------------------------------------------------------------------


def test_assumptions_view_shows_grouped_blocks_and_caveats():
    """The Assumptions view labels assumptions by analysis (Sales / P&L /
    Predictive), documents the single-outlet caveat and the twelve-observation
    forecast basis, and states findings are not generalizable (Req 9.1-9.4)."""
    at = _run_page(_ASSUMPTIONS_PAGE)

    text = _all_text(at)

    # Analysis-labeled group headings (Req 9.1). These are st.subheader labels.
    subheaders = [str(getattr(e, "value", "")) for e in at.subheader]
    assert "Sales" in subheaders
    assert "P&L" in subheaders
    assert "Predictive" in subheaders

    # Single-outlet caveat, verified from the data (Req 9.4).
    assert "Outlet 1" in text
    assert "not generalizable" in text

    # Twelve-observation forecast basis (Req 9.3).
    assert "12 monthly observations" in text


def test_assumptions_view_flags_missing_analysis(monkeypatch):
    """When an analysis has no documented assumptions, the view flags which
    analysis is missing rather than omitting it silently (Req 9.5).

    Driven through an inline script that stubs the predictive assumptions to be
    empty, so the missing-analysis indicator machinery is exercised end-to-end
    without depending on the real data ever being empty.
    """
    script = f"""
import sys
sys.path.insert(0, r"{_PROJECT_ROOT}")

import importlib.util
spec = importlib.util.spec_from_file_location("page_assumptions", r"{_ASSUMPTIONS_PAGE}")
page = importlib.util.module_from_spec(spec)
spec.loader.exec_module(page)

from src.data_loader import load_data
bundle = load_data()

# Force the Predictive analysis to report no assumptions so the missing-analysis
# indicator (Req 9.5) fires deterministically.
page._predictive_assumptions = lambda _bundle: []

page.render(bundle)
"""
    at = AppTest.from_string(script, default_timeout=_RUN_TIMEOUT)
    at.run(timeout=_RUN_TIMEOUT)
    assert not at.exception, f"assumptions page raised: {list(at.exception)}"

    text = _all_text(at)
    # The consolidated missing-analysis indicator names the affected analysis.
    assert "Missing documented assumptions for" in text
    assert "Predictive" in text


# ---------------------------------------------------------------------------
# Empty-state messages for each filterable view (Req 4.6, 5.6, 7.5, 10.5)
#
# Each page's render(bundle, filters) is driven with an impossible filter so
# the empty branch is exercised. Pages are loaded by file path (their names
# start with digits) via importlib inside an inline AppTest script.
# ---------------------------------------------------------------------------


def _empty_state_script(page_path: str, filters_literal: str) -> str:
    """Build an inline AppTest script that renders a page with a filter.

    Loads the real bundle, imports the page module by file path (digit-prefixed
    filenames are not importable module names), and calls
    ``render(bundle, filters=<filters_literal>)`` so the empty-result branch is
    reached with the shared filter contract.
    """
    return f"""
import sys
sys.path.insert(0, r"{_PROJECT_ROOT}")

import importlib.util
spec = importlib.util.spec_from_file_location("page_under_test", r"{page_path}")
page = importlib.util.module_from_spec(spec)
spec.loader.exec_module(page)

from src.data_loader import load_data
bundle = load_data()

page.render(bundle, filters={filters_literal})
"""


def test_sales_view_empty_state_message():
    """An impossible SKU filter drives the Sales view to its empty-result path,
    which shows a no-data message and does not crash (Req 4.6, 10.5)."""
    script = _empty_state_script(_SALES_PAGE, '{"skus": ["999999999"], "months": []}')
    at = AppTest.from_string(script, default_timeout=_RUN_TIMEOUT)
    at.run(timeout=_RUN_TIMEOUT)
    assert not at.exception, f"sales page raised: {list(at.exception)}"

    text = _all_text(at)
    assert "No sales records match" in text


def test_pnl_view_empty_state_message():
    """An impossible month filter drives the P&L view to its empty-result path,
    which shows a no-data message and retains the filter context (Req 5.6)."""
    script = _empty_state_script(_PNL_PAGE, '{"skus": [], "months": ["1999-01"]}')
    at = AppTest.from_string(script, default_timeout=_RUN_TIMEOUT)
    at.run(timeout=_RUN_TIMEOUT)
    assert not at.exception, f"pnl page raised: {list(at.exception)}"

    text = _all_text(at)
    assert "No P&L data is available for the selected filters" in text


def test_raw_material_view_empty_state_message():
    """An impossible SKU filter drives the Raw Material view to its empty-result
    path for each visualization, which shows a no-data note (Req 7.5)."""
    script = _empty_state_script(_RAW_MATERIAL_PAGE, '{"skus": ["999999999"]}')
    at = AppTest.from_string(script, default_timeout=_RUN_TIMEOUT)
    at.run(timeout=_RUN_TIMEOUT)
    assert not at.exception, f"raw material page raised: {list(at.exception)}"

    text = _all_text(at)
    assert "No RawMatPriceTrend rows match the selected SKU filter" in text
