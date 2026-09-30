"""GGF Analytics App — entry point (Task 14.1).

Single controlled entry point that wires the seven analysis views together:

- one cached data load (Req 1.1, 10.4) with a hard error gate on
  :class:`~src.data_loader.LoadError` that names the missing file/sheet/column
  and refuses to show any analysis view (Req 1.6);
- persistent sidebar navigation across all seven sections with the Executive
  Summary first (Req 3.6, 10.1, 10.3);
- shared SKU + month multiselect filters held in ``st.session_state["filters"]``
  and passed into every ``render(bundle, filters)`` (Req 10.4); an empty
  selection means "no restriction" on that dimension;
- an "Export deck" download in the sidebar that runs the pillar-coverage gate
  inside :func:`src.presentation.build_default_deck` and offers the ``.pptx``,
  surfacing a :class:`~src.presentation.PresentationError` as an ``st.error``
  without tearing down the app (Req 11.4, 11.6).

Design direction (project ``DESIGN.md``): a calm, data-first internal dashboard.
Navigation and filters live in the sidebar so nothing competes with the numbers.

Run locally with::

    streamlit run app.py

on the mandated Python + Streamlit stack (Req 10.6).
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType

import streamlit as st

from src import config, metrics
from src.data_loader import DataBundle, LoadError, load_data
from src.presentation import PresentationError, build_default_deck

#: Directory holding the seven numbered page modules.
_PAGES_DIR = Path(__file__).resolve().parent / "pages"

#: Download filename + MIME for the exported deck (Req 11.4).
_DECK_FILENAME = "GGF_Analysis.pptx"
_DECK_MIME = (
    "application/vnd.openxmlformats-officedocument.presentationml.presentation"
)


# ---------------------------------------------------------------------------
# Navigation registry
#
# Executive Summary is first (Req 3.6). Each entry is (label, render callable).
# Renders are imported lazily inside the loader so importing app.py stays cheap
# and page modules do not auto-run at import time.
# ---------------------------------------------------------------------------


# Ordered (label, page-file stem) pairs. Executive Summary is first (Req 3.6).
# The files are numbered (``1_Executive_Summary.py`` …) so their stems are not
# valid Python identifiers; they are loaded by path via importlib below.
_PAGE_SPECS: list[tuple[str, str]] = [
    ("Executive Summary", "1_Executive_Summary"),
    ("Sales Performance", "2_Sales_Performance"),
    ("Profit & Loss", "3_PnL"),
    ("Raw Material & Seasonality", "4_Raw_Material"),
    ("Predictive", "5_Predictive"),
    ("Data Model", "6_Data_Model"),
    ("Assumptions", "7_Assumptions"),
]


def _load_page_module(stem: str) -> ModuleType:
    """Import a numbered page module by file path and return it.

    The page files auto-run at import unless ``_APP_DRIVES_PAGES`` is set in
    ``st.session_state`` (which :func:`main` does before loading any page), so
    importing here does not render the page — it just exposes ``render``.
    """
    module_name = f"pages._{stem}"
    path = _PAGES_DIR / f"{stem}.py"
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:  # pragma: no cover - defensive
        raise ImportError(f"Cannot load page module: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _page_renderers() -> list[tuple[str, object]]:
    """Return the ordered ``(label, render)`` pairs for the seven sections.

    Executive Summary is first (Req 3.6). Called after ``_APP_DRIVES_PAGES`` is
    set in :func:`main`, so the page modules do not auto-render on import.
    """
    renderers: list[tuple[str, object]] = []
    for label, stem in _PAGE_SPECS:
        module = _load_page_module(stem)
        renderers.append((label, module.render))
    return renderers


# ---------------------------------------------------------------------------
# Cached data load (Req 1.1, 10.4)
# ---------------------------------------------------------------------------


@st.cache_data(show_spinner="Loading the source workbook…")
def _load_bundle_cached() -> DataBundle:
    """Load the workbook once, cached across reruns and filter changes.

    Wrapped in :func:`st.cache_data` so the (relatively expensive) read + join
    happens once per session rather than on every filter interaction (Req 10.4).
    A :class:`LoadError` propagates to the caller, which renders the error gate.
    """
    return load_data()


# ---------------------------------------------------------------------------
# Sidebar: filters + export
# ---------------------------------------------------------------------------


def _sku_options() -> list[str]:
    """SKU keys offered in the filter, ordered by the config catalog."""
    return list(config.SKU_MAP.keys())


def _sku_label(sku: str) -> str:
    """``"Product Name (SKU)"`` label for a SKU key."""
    name = config.SKU_MAP.get(str(sku))
    return f"{name} ({sku})" if name else str(sku)


def _render_filters() -> dict:
    """Render the shared SKU + month filters and return the selection.

    Stores the result in ``st.session_state["filters"]`` as
    ``{"skus": [...], "months": [...]}`` and returns it. An empty selection on
    either dimension means "no restriction" on that dimension (documented in the
    UI), so the detailed views fall back to the full frame (Req 10.4).
    """
    st.sidebar.subheader("Filters")
    st.sidebar.caption(
        "Applied to the detailed breakdown views. Leave a filter empty to "
        "include everything on that dimension."
    )

    skus = st.sidebar.multiselect(
        "SKU",
        options=_sku_options(),
        format_func=_sku_label,
        default=[],
        key="filter_skus",
    )
    months = st.sidebar.multiselect(
        "Month (2022)",
        options=list(metrics.MONTHS_2022),
        default=[],
        key="filter_months",
    )

    filters = {"skus": skus, "months": months}
    st.session_state["filters"] = filters
    return filters


def _render_export(bundle: DataBundle) -> None:
    """Render the sidebar "Export deck" control (Req 11.4, 11.6).

    On click, builds the default deck via
    :func:`src.presentation.build_default_deck`, which runs the pillar-coverage
    gate internally and raises :class:`PresentationError` if a pillar is missing
    or generation fails. On success the ``.pptx`` bytes are offered via
    ``st.download_button``; on failure the reason is surfaced as an ``st.error``
    and the app keeps running (Req 11.5).
    """
    st.sidebar.divider()
    st.sidebar.subheader("Presentation")
    st.sidebar.caption(
        "Compile the analysis and every visualization into a PowerPoint deck "
        "(executive summary first, then the detailed breakdown)."
    )

    if st.sidebar.button("Build export deck", key="build_deck", width="stretch"):
        try:
            with st.spinner("Building the presentation deck…"):
                deck_bytes = build_default_deck(bundle)
        except PresentationError as exc:
            st.sidebar.error(f"Could not generate the deck: {exc}")
        else:
            st.session_state["deck_bytes"] = deck_bytes
            st.sidebar.success("Deck ready — use the download button below.")

    deck_bytes = st.session_state.get("deck_bytes")
    if deck_bytes:
        st.sidebar.download_button(
            "Download GGF_Analysis.pptx",
            data=deck_bytes,
            file_name=_DECK_FILENAME,
            mime=_DECK_MIME,
            key="download_deck",
            width="stretch",
        )


# ---------------------------------------------------------------------------
# Error gate (Req 1.6)
# ---------------------------------------------------------------------------


def _render_load_error(exc: LoadError) -> None:
    """Render the data-load error gate and stop (Req 1.6).

    Names the missing file/sheet/column (the ``LoadError`` message carries the
    specifics) and does NOT render navigation or any analysis view.
    """
    st.error(
        f"Could not load the source workbook: {exc}\n\n"
        "Next step: place `FruitsTrxDatasets.xlsx` — with the MstProduct, "
        "SalesTarget, SalesTransaction, and RawMatPriceTrend sheets and their "
        "required columns — in the app's `data/` folder, or set the "
        "`GGF_WORKBOOK_PATH` environment variable to its location, then reload."
    )
    st.info(
        "Analysis views are hidden until the workbook loads cleanly, so no "
        "figures are shown against incomplete data."
    )


# ---------------------------------------------------------------------------
# App entry point
# ---------------------------------------------------------------------------


def main() -> None:
    """Configure the page, gate on load, and render the selected section."""
    st.set_page_config(
        page_title="GGF Analytics",
        page_icon=None,
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # Tell the page modules that app.py drives them, so their bottom-of-file
    # auto-run is suppressed and only our explicit render() call fires.
    st.session_state["_APP_DRIVES_PAGES"] = True

    st.sidebar.title("GGF Analytics")

    # --- Data load + error gate (Req 1.6) --------------------------------
    try:
        bundle = _load_bundle_cached()
    except LoadError as exc:
        _render_load_error(exc)
        return

    # --- Persistent navigation, Executive Summary first (Req 3.6, 10.1) --
    pages = _page_renderers()
    labels = [label for label, _ in pages]

    st.sidebar.subheader("Sections")
    selected = st.sidebar.radio(
        "Navigate",
        options=labels,
        index=0,  # Executive Summary first
        key="nav_section",
        label_visibility="collapsed",
    )

    # --- Shared filters (Req 10.4) ---------------------------------------
    filters = _render_filters()

    # --- Export deck (Req 11.4, 11.6) ------------------------------------
    _render_export(bundle)

    # --- Render the selected section (Req 10.3) --------------------------
    renderer = dict(pages)[selected]
    renderer(bundle=bundle, filters=filters)


if __name__ == "__main__":
    main()
else:
    # Streamlit executes app.py top-to-bottom, so render on import too.
    main()
