"""Example-based tests for ``src.data_loader`` load-error paths (Req 1.6).

These cover the concrete error scenarios where behavior does not vary
meaningfully with input:
  - missing workbook file -> LoadError
  - a required sheet missing -> LoadError naming the sheet
  - a required column missing -> LoadError naming the column

Synthetic workbooks are written with pandas ``to_excel`` (openpyxl engine) into
the pytest ``tmp_path`` fixture using the SOURCE column names (``TranscationDate``,
``Profit&Lost``) so that column normalization is exercised on the way in.
"""

from __future__ import annotations

import pandas as pd
import pytest

from src import config
from src.data_loader import LoadError, load_data


# ---------------------------------------------------------------------------
# Minimal valid sheet builders (source column names, three valid SKUs)
# ---------------------------------------------------------------------------

VALID_SKUS = sorted(config.VALID_SKUS)  # ["200100001", "200100003", "200100004"]


def _mst_product() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "SKU": [int(s) for s in VALID_SKUS],
            # ProductName is derived from ShortDescription by the loader.
            "ShortDescription": [config.SKU_MAP[s] for s in VALID_SKUS],
            "Price": [18500, 18500, 22500],
        }
    )


def _sales_target() -> pd.DataFrame:
    rows = []
    for sku in VALID_SKUS:
        rows.append((int(sku), pd.Timestamp("2022-01-01"), 5000))
    return pd.DataFrame(
        {
            "SKU": [r[0] for r in rows],
            "Period": [r[1] for r in rows],
            "SalesTarget": [r[2] for r in rows],
        }
    )


def _sales_transaction() -> pd.DataFrame:
    """SalesTransaction with SOURCE typo column names."""
    n = len(VALID_SKUS)
    return pd.DataFrame(
        {
            "TranscationDate": [pd.Timestamp("2022-01-15")] * n,
            "SKU": [int(s) for s in VALID_SKUS],
            "TotalNetSellingPrice": [1000.0] * n,
            "TotalRawMaterialCost": [300.0] * n,
            "TotalProductionCost": [200.0] * n,
            "TotalCost": [500.0] * n,
            "Profit&Lost": [500.0] * n,
        }
    )


def _raw_mat_trend() -> pd.DataFrame:
    n = len(VALID_SKUS)
    return pd.DataFrame(
        {
            "Date": [pd.Timestamp("2022-01-03")] * n,
            "SKU": [int(s) for s in VALID_SKUS],
            "Close Price": [10.0] * n,
            "Is Peak Season": [0] * n,
        }
    )


def _sheet_frames() -> dict[str, pd.DataFrame]:
    return {
        "MstProduct": _mst_product(),
        "SalesTarget": _sales_target(),
        "SalesTransaction": _sales_transaction(),
        "RawMatPriceTrend": _raw_mat_trend(),
    }


def _write_workbook(path, frames: dict[str, pd.DataFrame]) -> None:
    """Write the given sheet frames to an ``.xlsx`` file at ``path``."""
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        for sheet_name, frame in frames.items():
            frame.to_excel(writer, sheet_name=sheet_name, index=False)


def test_load_data_valid_workbook_loads(tmp_path) -> None:
    """A well-formed workbook loads without error and normalizes typo columns."""
    path = tmp_path / "valid.xlsx"
    _write_workbook(path, _sheet_frames())

    bundle = load_data(str(path))

    # Typo columns normalized on the transaction table.
    assert "transaction_date" in bundle.sales_transaction.columns
    assert "profit_source" in bundle.sales_transaction.columns
    assert "TranscationDate" not in bundle.sales_transaction.columns
    assert bundle.joined is not None


def test_missing_file_raises_load_error(tmp_path) -> None:
    """A path with no file present raises LoadError naming the file (Req 1.6)."""
    missing = tmp_path / "does_not_exist.xlsx"

    with pytest.raises(LoadError) as excinfo:
        load_data(str(missing))

    assert "does_not_exist.xlsx" in str(excinfo.value)


@pytest.mark.parametrize("missing_sheet", list(config.REQUIRED_SHEETS))
def test_missing_required_sheet_raises_load_error(tmp_path, missing_sheet) -> None:
    """Dropping any one of the four required sheets raises a LoadError that
    names the missing sheet (Req 1.6)."""
    frames = _sheet_frames()
    del frames[missing_sheet]
    path = tmp_path / f"missing_{missing_sheet}.xlsx"
    _write_workbook(path, frames)

    with pytest.raises(LoadError) as excinfo:
        load_data(str(path))

    assert missing_sheet in str(excinfo.value)


def test_missing_transaction_date_column_raises_load_error(tmp_path) -> None:
    """Missing the transaction-date column raises a LoadError naming it (Req 1.6).

    The source ``TranscationDate`` is dropped so that after normalization the
    required ``transaction_date`` column is absent.
    """
    frames = _sheet_frames()
    frames["SalesTransaction"] = frames["SalesTransaction"].drop(columns=["TranscationDate"])
    path = tmp_path / "missing_txn_date.xlsx"
    _write_workbook(path, frames)

    with pytest.raises(LoadError) as excinfo:
        load_data(str(path))

    assert "transaction_date" in str(excinfo.value)


def test_missing_profit_source_column_raises_load_error(tmp_path) -> None:
    """Missing the profit column raises a LoadError naming ``profit_source``.

    The source ``Profit&Lost`` is dropped so that after normalization the
    required ``profit_source`` column is absent (Req 1.6).
    """
    frames = _sheet_frames()
    frames["SalesTransaction"] = frames["SalesTransaction"].drop(columns=["Profit&Lost"])
    path = tmp_path / "missing_profit.xlsx"
    _write_workbook(path, frames)

    with pytest.raises(LoadError) as excinfo:
        load_data(str(path))

    assert "profit_source" in str(excinfo.value)


def test_missing_sku_column_raises_load_error(tmp_path) -> None:
    """Missing the SKU column on a required sheet raises a LoadError naming
    ``SKU`` (Req 1.6)."""
    frames = _sheet_frames()
    frames["SalesTransaction"] = frames["SalesTransaction"].drop(columns=["SKU"])
    path = tmp_path / "missing_sku.xlsx"
    _write_workbook(path, frames)

    with pytest.raises(LoadError) as excinfo:
        load_data(str(path))

    assert "SKU" in str(excinfo.value)
