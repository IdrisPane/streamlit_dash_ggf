"""Configuration constants for the GGF Analytics App.

This is a pure module: no I/O, no side effects, and no Streamlit/pandas
imports. It centralizes the SKU catalog, the source-column normalization map,
the required sheets and columns, the recomputed-profit-basis documentation,
analysis thresholds, forecasting parameters, and the workbook path (which is
resolved lazily and can be overridden via the ``GGF_WORKBOOK_PATH`` env var).

Downstream modules (``data_loader``, ``metrics``, ``anomalies``,
``forecasting``, ``presentation``) import these constants so that the data
contract lives in exactly one place.
"""

from __future__ import annotations

import os
from pathlib import Path

# ---------------------------------------------------------------------------
# Products / SKUs
#
# SKU is normalized to ``str`` everywhere downstream (the source stores it as an
# integer-looking value, but joins and lookups treat it as a string key).
# ---------------------------------------------------------------------------

SKU_MAP: dict[str, str] = {
    "200100001": "Orange Juice",
    "200100003": "Mango Juice",
    "200100004": "Pineapple Juice",
}

#: The three SKU keys the analysis is restricted to (join filter, Req 1.7).
VALID_SKUS: frozenset[str] = frozenset(SKU_MAP)

# ---------------------------------------------------------------------------
# Column normalization (Req 1.2, 1.3, 2.4)
#
# The two source typos are renamed once, to canonical names used everywhere.
# ---------------------------------------------------------------------------

COLUMN_NORMALIZATION_MAP: dict[str, str] = {
    "TranscationDate": "transaction_date",
    "Profit&Lost": "profit_source",
}

# ---------------------------------------------------------------------------
# Required sheets and columns (Req 1.1, 1.6)
# ---------------------------------------------------------------------------

#: The four sheets the loader reads (and requires) from the workbook.
REQUIRED_SHEETS: list[str] = [
    "MstProduct",
    "SalesTarget",
    "SalesTransaction",
    "RawMatPriceTrend",
]

#: Minimum required columns per sheet, keyed by sheet name. Column names for
#: SalesTransaction are stated post-normalization (``transaction_date`` and
#: ``profit_source`` rather than the source typos). A missing column here is a
#: hard LoadError in the data layer (Req 1.6).
REQUIRED_COLUMNS: dict[str, list[str]] = {
    "MstProduct": ["SKU", "ProductName"],
    "SalesTarget": ["Period", "SKU", "SalesTarget"],
    "SalesTransaction": [
        "transaction_date",
        "SKU",
        "TotalNetSellingPrice",
        "TotalRawMaterialCost",
        "TotalProductionCost",
        "TotalCost",
        "profit_source",
    ],
    "RawMatPriceTrend": ["Date", "SKU", "Close Price", "Is Peak Season"],
}

# ---------------------------------------------------------------------------
# Recomputed profit basis (Req 6.4, 9.2)
#
# The source ``Profit&Lost`` and ``TotalCost`` columns are internally
# inconsistent on every row, so profit is recomputed from primitive net revenue
# and primitive cost components. The identifier, formula, and reason below are
# surfaced in the P&L and Assumptions views.
# ---------------------------------------------------------------------------

PROFIT_BASIS_ID: str = "net_minus_primitive_costs"

PROFIT_BASIS_FORMULA: str = (
    "TotalNetSellingPrice - (TotalRawMaterialCost + TotalProductionCost)"
)

PROFIT_BASIS_REASON: str = (
    "The source Profit&Lost and TotalCost columns are internally inconsistent "
    "on every row (Profit&Lost != TotalNetSellingPrice - TotalCost, and "
    "TotalCost != TotalRawMaterialCost + TotalProductionCost). Profit is "
    "therefore recomputed from primitive net revenue and primitive cost "
    "components so the analysis is reproducible and independent of the two "
    "inconsistent aggregate fields."
)

# ---------------------------------------------------------------------------
# Anomaly / reconciliation thresholds (Req 4.4, 6.1, 6.2)
# ---------------------------------------------------------------------------

#: A monthly sales value beyond this many standard deviations is anomalous.
SALES_ANOMALY_STD_THRESHOLD: float = 2.0

#: Target attainment (%) at or below this is flagged as an anomaly.
ATTAINMENT_LOW_PCT: float = 50.0

#: Target attainment (%) at or above this is flagged as an anomaly.
ATTAINMENT_HIGH_PCT: float = 150.0

#: Absolute tolerance (currency units) for P&L reconciliation checks.
RECONCILIATION_TOLERANCE: float = 0.01

# ---------------------------------------------------------------------------
# Forecasting parameters (Req 8.1, 8.2, 8.3)
# ---------------------------------------------------------------------------

#: Number of future months to forecast beyond December 2022.
FORECAST_HORIZON: int = 12

#: Default forecasting method (Holt linear exponential smoothing).
DEFAULT_FORECAST_METHOD: str = "holt"

#: Minimum monthly observations required to fit a forecast for an SKU.
MIN_OBSERVATIONS: int = 12

# ---------------------------------------------------------------------------
# Presentation pillars (Req 11.3, 11.6)
# ---------------------------------------------------------------------------

#: Analysis pillars the exported deck must cover (used by the exporter's
#: pillar-coverage gate).
PILLARS: list[str] = ["data_model", "sales", "pnl", "predictive"]

# ---------------------------------------------------------------------------
# Currency
# ---------------------------------------------------------------------------

#: Currency label displayed alongside monetary figures.
CURRENCY_LABEL: str = "IDR"

# ---------------------------------------------------------------------------
# Workbook path (Req 1.1)
#
# Resolved relative to the project root (the parent of this ``src`` package) so
# it works when launched via ``streamlit run app.py``. Override with the
# ``GGF_WORKBOOK_PATH`` environment variable.
# ---------------------------------------------------------------------------

#: Project root: the parent directory of the ``src`` package.
PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent

#: Default workbook location under ``data/``.
DEFAULT_WORKBOOK_PATH: Path = PROJECT_ROOT / "data" / "FruitsTrxDatasets.xlsx"

#: Environment variable used to override the workbook path.
WORKBOOK_PATH_ENV_VAR: str = "GGF_WORKBOOK_PATH"


def get_workbook_path() -> Path:
    """Return the workbook path, honoring the ``GGF_WORKBOOK_PATH`` override.

    Reads the environment at call time (rather than import time) so tests and
    deployments can set the variable after the module is imported.
    """
    override = os.environ.get(WORKBOOK_PATH_ENV_VAR)
    if override:
        return Path(override)
    return DEFAULT_WORKBOOK_PATH


#: Convenience constant resolved at import time. Prefer ``get_workbook_path()``
#: when the environment may change after import.
WORKBOOK_PATH: Path = get_workbook_path()
