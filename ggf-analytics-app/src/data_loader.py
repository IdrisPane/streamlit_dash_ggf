"""Data layer for the GGF Analytics App.

Pure pandas data-loading primitives: column normalization and per-cell type
coercion. This module has **no Streamlit dependency** (the ``st.cache_data``
wrapper and the workbook read/join logic live with ``load_data`` and are added
in a later task). It also defines the immutable result containers
(:class:`ParseError`, :class:`UnmatchedKey`, :class:`DataBundle`) and the
:class:`LoadError` exception used across the data layer.

The functions here are pure and deterministic: same inputs produce the same
outputs, they do no I/O, and they never mutate their caller's frame.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import pandas as pd

from . import config


# ---------------------------------------------------------------------------
# Result containers
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ParseError:
    """A single cell that could not be coerced to its target type (Req 1.5).

    Attributes:
        sheet: The sheet the cell came from (e.g. ``"SalesTransaction"``).
        column: The column name of the offending cell.
        row_index: The row position (the frame's index label) of the cell.
        raw_value: The original, unparseable value rendered as a string.
    """

    sheet: str
    column: str
    row_index: int
    raw_value: str


@dataclass(frozen=True)
class UnmatchedKey:
    """A join key with no counterpart, retained and recorded (Req 1.9).

    Attributes:
        kind: The kind of key, either ``"sku"`` or ``"month"``.
        value: The unmatched key value rendered as a string.
        row_index: The row position of the transaction that failed to match.
    """

    kind: str  # "sku" | "month"
    value: str
    row_index: int


@dataclass(frozen=True)
class DataBundle:
    """Immutable bundle of cleaned tables plus load diagnostics.

    The join-related fields (``joined``, ``unmatched_keys``) are populated by
    ``load_data`` in a later task; sensible defaults are provided here so the
    bundle is constructible from the primitives available now.

    Attributes:
        mst_product: The MstProduct table.
        sales_target: The SalesTarget table.
        sales_transaction: The SalesTransaction table (normalized columns).
        raw_mat_trend: The RawMatPriceTrend table.
        joined: Transaction joined to target (by month) and product (by SKU).
        parse_errors: Every cell that failed coercion during loading.
        unmatched_keys: Every SKU/month that failed to join.
        normalization_map: Original -> normalized column-name pairs applied.
    """

    mst_product: pd.DataFrame
    sales_target: pd.DataFrame
    sales_transaction: pd.DataFrame
    raw_mat_trend: pd.DataFrame
    joined: Optional[pd.DataFrame] = None
    parse_errors: list[ParseError] = field(default_factory=list)
    unmatched_keys: list[UnmatchedKey] = field(default_factory=list)
    normalization_map: dict[str, str] = field(default_factory=dict)


class LoadError(Exception):
    """Raised when the workbook, a required sheet, or a required column is
    missing (Req 1.6). Raised by ``load_data`` in a later task; defined here so
    the whole data-layer contract lives in one module."""


# ---------------------------------------------------------------------------
# Column normalization (Req 1.2, 1.3)
# ---------------------------------------------------------------------------


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of ``df`` with source-typo columns renamed to canonical
    names (Req 1.2, 1.3).

    Columns listed in :data:`config.COLUMN_NORMALIZATION_MAP` are renamed
    (``TranscationDate`` -> ``transaction_date``, ``Profit&Lost`` ->
    ``profit_source``); all other columns are left untouched. Cell values are
    never modified. The caller's frame is not mutated (a renamed copy is
    returned).

    Args:
        df: The frame whose columns may contain source typos.

    Returns:
        A new frame with normalized column names and identical values.
    """
    renamed = df.rename(columns=config.COLUMN_NORMALIZATION_MAP)
    # ``rename`` already returns a new frame, but its column-name change shares
    # underlying value blocks with the original. Returning ``renamed`` is safe
    # because we never mutate values; the original frame's columns are intact.
    return renamed


# ---------------------------------------------------------------------------
# Type coercion (Req 1.4, 1.5)
# ---------------------------------------------------------------------------


def coerce_types(
    df: pd.DataFrame,
    schema: dict[str, str],
    sheet: str,
) -> tuple[pd.DataFrame, list[ParseError]]:
    """Coerce the columns named in ``schema`` to their target types (Req 1.4,
    1.5).

    Each entry of ``schema`` maps a column name to ``"date"`` or ``"numeric"``.
    Parseable cells keep their value with the correct dtype, preserving
    magnitude and sign for numerics (including negatives). Any cell that cannot
    be parsed is set to null (``NaT`` for dates, ``NaN`` for numerics) and a
    :class:`ParseError` is appended recording the sheet, column, row position,
    and the original raw value as a string. Parsing never aborts: the remaining
    cells and columns are always processed.

    The caller's frame is not mutated (a copy is returned). Columns absent from
    ``df`` are skipped. Columns not named in ``schema`` are left untouched.

    Args:
        df: The frame to coerce.
        schema: Column-name -> ``"date"`` | ``"numeric"``.
        sheet: The sheet name recorded on any resulting :class:`ParseError`.

    Returns:
        A ``(coerced_frame, parse_errors)`` tuple.
    """
    result = df.copy()
    parse_errors: list[ParseError] = []

    for column, kind in schema.items():
        if column not in result.columns:
            continue

        original = result[column]

        if kind == "numeric":
            coerced = pd.to_numeric(original, errors="coerce")
        elif kind == "date":
            coerced = pd.to_datetime(original, errors="coerce")
        else:  # pragma: no cover - defensive: unknown schema kind is a no-op.
            continue

        # A cell failed to parse when the coerced value is null but the source
        # value was not already null/blank. Original nulls are not parse
        # errors (there was nothing to parse).
        original_null = original.isna()
        coerced_null = coerced.isna()
        failed_mask = coerced_null & ~original_null

        for row_index in result.index[failed_mask]:
            parse_errors.append(
                ParseError(
                    sheet=sheet,
                    column=column,
                    row_index=int(row_index),
                    raw_value=str(original.loc[row_index]),
                )
            )

        result[column] = coerced

    return result, parse_errors


# ---------------------------------------------------------------------------
# Per-sheet coercion schemas (Req 1.4)
#
# Column names for SalesTransaction are stated post-normalization
# (``transaction_date``, ``profit_source``). Columns absent from a sheet are
# skipped by ``coerce_types``, so listing a superset is safe.
# ---------------------------------------------------------------------------

_SCHEMAS: dict[str, dict[str, str]] = {
    "MstProduct": {
        "CreationDate": "date",
        "Price": "numeric",
        "CategoryCode": "numeric",
    },
    "SalesTarget": {
        "Period": "date",
        "SalesTarget": "numeric",
    },
    "SalesTransaction": {
        "transaction_date": "date",
        "Qty": "numeric",
        "SellingPrice": "numeric",
        "TotalSellingPrice": "numeric",
        "DiscPercentage": "numeric",
        "Discount": "numeric",
        "TotalDiscount": "numeric",
        "TotalNetSellingPrice": "numeric",
        "RawMaterialCost": "numeric",
        "TotalRawMaterialCost": "numeric",
        "ProductionCost": "numeric",
        "TotalProductionCost": "numeric",
        "TotalCost": "numeric",
        "profit_source": "numeric",
    },
    "RawMatPriceTrend": {
        "Date": "date",
        "Open Price": "numeric",
        "Close Price": "numeric",
        "Open-Close Change (%)": "numeric",
        "Moving Avg (3)": "numeric",
        "Harvest Success Rate (%)": "numeric",
        "Month": "numeric",
        "Is Peak Season": "numeric",
    },
}


def _normalize_sku(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of ``df`` with the ``SKU`` column coerced to a clean
    string key (Req 1.7).

    The source stores SKU as an integer-looking value; downstream joins and
    lookups treat it as a string. A source value of ``200100001`` (int or
    float ``200100001.0``) must render as ``"200100001"``, not ``"200100001.0"``.
    Genuine nulls are left as ``pd.NA`` so they never spuriously match.
    """
    result = df.copy()
    if "SKU" not in result.columns:
        return result

    def _to_key(value: object) -> object:
        if pd.isna(value):
            return pd.NA
        # Integer-valued floats/ints -> plain integer string (drop ".0").
        if isinstance(value, (int, float)) and float(value).is_integer():
            return str(int(value))
        return str(value).strip()

    result["SKU"] = result["SKU"].map(_to_key)
    return result


def _month_key(series: pd.Series) -> pd.Series:
    """Return a ``year-month`` string key (e.g. ``"2022-01"``) for a datetime
    series, used to align transactions to targets by calendar month (Req 1.8).

    Non-datetime / null entries yield ``pd.NA`` so they never match a target.
    """
    dt = pd.to_datetime(series, errors="coerce")
    return dt.dt.strftime("%Y-%m").where(dt.notna(), other=pd.NA)


def load_data(workbook_path: Optional[str] = None) -> DataBundle:
    """Read the four workbook sheets, clean them, and build the joined frame.

    Reads exactly the four required sheets from ``workbook_path`` (or
    :func:`config.get_workbook_path` when ``None``), normalizes the two typo
    columns, coerces date/numeric types (collecting :class:`ParseError`), keys
    SKU as a string everywhere, and left-joins SalesTransaction to MstProduct
    (by SKU) and SalesTarget (by SKU + calendar month), restricted to the three
    valid SKUs. Unmatched transaction rows are retained with null joined fields
    and recorded as :class:`UnmatchedKey` (Req 1.1, 1.7, 1.8, 1.9).

    Args:
        workbook_path: Path to the ``.xlsx`` workbook. When ``None``, the
            configured default (honoring the ``GGF_WORKBOOK_PATH`` override) is
            used.

    Returns:
        A fully populated :class:`DataBundle`.

    Raises:
        LoadError: If the file is absent, any required sheet is missing, or any
            required column (evaluated after normalization) is absent from its
            sheet. The message names the missing file, sheet, or column
            (Req 1.6).
    """
    from pathlib import Path

    path = Path(workbook_path) if workbook_path is not None else config.get_workbook_path()

    # --- Error gate: file present (Req 1.6) ------------------------------
    if not path.is_file():
        raise LoadError(f"Workbook not found: {path}")

    try:
        excel = pd.ExcelFile(path)
    except Exception as exc:  # pragma: no cover - unreadable/corrupt file.
        raise LoadError(f"Workbook could not be read: {path} ({exc})") from exc

    available_sheets = set(excel.sheet_names)

    # --- Error gate: every required sheet present (Req 1.6) --------------
    for sheet in config.REQUIRED_SHEETS:
        if sheet not in available_sheets:
            raise LoadError(f"Required sheet is missing from workbook: {sheet!r}")

    # --- Read, normalize columns, key SKU, coerce types ------------------
    frames: dict[str, pd.DataFrame] = {}
    parse_errors: list[ParseError] = []

    for sheet in config.REQUIRED_SHEETS:
        raw = pd.read_excel(excel, sheet_name=sheet)
        normalized = normalize_columns(raw)

        # Derive MstProduct.ProductName from ShortDescription so the required-
        # column contract holds and downstream views have a product name.
        if sheet == "MstProduct" and "ProductName" not in normalized.columns:
            if "ShortDescription" in normalized.columns:
                normalized = normalized.copy()
                normalized["ProductName"] = normalized["ShortDescription"].astype("string")

        keyed = _normalize_sku(normalized)
        coerced, sheet_errors = coerce_types(keyed, _SCHEMAS.get(sheet, {}), sheet)
        parse_errors.extend(sheet_errors)
        frames[sheet] = coerced

    # --- Error gate: every required column present (post-normalization) --
    for sheet, required in config.REQUIRED_COLUMNS.items():
        columns = set(frames[sheet].columns)
        for column in required:
            if column not in columns:
                raise LoadError(
                    f"Required column {column!r} is missing from sheet {sheet!r}"
                )

    mst_product = frames["MstProduct"]
    sales_target = frames["SalesTarget"]
    sales_transaction = frames["SalesTransaction"]
    raw_mat_trend = frames["RawMatPriceTrend"]

    # --- Build the joined frame (Req 1.7, 1.8, 1.9) ----------------------
    joined, unmatched_keys = _build_joined(
        sales_transaction, mst_product, sales_target
    )

    return DataBundle(
        mst_product=mst_product,
        sales_target=sales_target,
        sales_transaction=sales_transaction,
        raw_mat_trend=raw_mat_trend,
        joined=joined,
        parse_errors=parse_errors,
        unmatched_keys=unmatched_keys,
        normalization_map=dict(config.COLUMN_NORMALIZATION_MAP),
    )


def _build_joined(
    sales_transaction: pd.DataFrame,
    mst_product: pd.DataFrame,
    sales_target: pd.DataFrame,
) -> tuple[pd.DataFrame, list[UnmatchedKey]]:
    """Left-join transactions to product (by SKU) and target (by SKU + month),
    add ``recomputed_profit`` and ``month`` keys, and record unmatched keys
    (Req 1.7, 1.8, 1.9).

    Product and target lookups are restricted to :data:`config.VALID_SKUS`. Any
    transaction whose SKU is not a valid product, or whose month has no target,
    keeps its row with null joined fields and yields an :class:`UnmatchedKey`.
    """
    joined = sales_transaction.copy()

    # Month period key for the transaction and for the target.
    joined["month"] = _month_key(joined["transaction_date"])

    # --- Product lookup (SKU), restricted to the valid SKUs --------------
    product_cols = [c for c in ("SKU", "ProductName") if c in mst_product.columns]
    product = mst_product[product_cols].copy()
    product = product[product["SKU"].isin(config.VALID_SKUS)]
    product = product.drop_duplicates(subset="SKU", keep="first")

    joined = joined.merge(product, on="SKU", how="left")

    # --- Target lookup (SKU + calendar month), restricted to valid SKUs --
    target = sales_target.copy()
    target = target[target["SKU"].isin(config.VALID_SKUS)]
    target["month"] = _month_key(target["Period"])
    target = target[["SKU", "month", "SalesTarget"]].drop_duplicates(
        subset=["SKU", "month"], keep="first"
    )

    joined = joined.merge(target, on=["SKU", "month"], how="left")

    # --- Recomputed profit basis (net - (rawmat + production)) -----------
    joined["recomputed_profit"] = joined["TotalNetSellingPrice"] - (
        joined["TotalRawMaterialCost"] + joined["TotalProductionCost"]
    )

    # --- Record unmatched keys (Req 1.9) ---------------------------------
    valid_skus = config.VALID_SKUS
    unmatched_keys: list[UnmatchedKey] = []
    for row_index in joined.index:
        sku = joined.at[row_index, "SKU"]
        # SKU unmatched: not a valid product SKU (also covers null SKU).
        if pd.isna(sku) or sku not in valid_skus:
            unmatched_keys.append(
                UnmatchedKey(
                    kind="sku",
                    value="" if pd.isna(sku) else str(sku),
                    row_index=int(row_index),
                )
            )
            # A non-valid SKU can't have a valid month target either; still
            # record the month gap so both dimensions are captured.
        if pd.isna(joined.at[row_index, "SalesTarget"]):
            month = joined.at[row_index, "month"]
            unmatched_keys.append(
                UnmatchedKey(
                    kind="month",
                    value="" if pd.isna(month) else str(month),
                    row_index=int(row_index),
                )
            )

    return joined, unmatched_keys
