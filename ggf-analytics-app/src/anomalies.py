"""Anomaly detection and P&L reconciliation for the GGF Analytics App.

Pure, deterministic analytics over the (normalized) ``SalesTransaction`` frame.
This module has **no Streamlit and no plotting dependency**: it consumes a
pandas frame and returns an immutable :class:`ReconResult`.

The source ``Profit&Lost`` (normalized to ``profit_source``) and ``TotalCost``
columns are internally inconsistent on every row of the real dataset, so this
module surfaces the reconciliation discrepancy rather than trusting those
aggregate fields (Req 6.1, 6.2, 6.3). It also counts loss-making rows on the
recomputed profit basis per SKU (Req 6.5) and isolates rows that cannot be
reconciled because a required field is missing or non-numeric (Req 6.6).

All functions are pure: same input frame produces the same result, no I/O, and
the caller's frame is never mutated.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from . import config

#: The five fields every reconciliation check requires. A row missing or
#: non-numeric in any of these is unprocessable (Req 6.6). Order matters: the
#: attributed offending field is the *first* of these that is bad.
REQUIRED_RECON_FIELDS: list[str] = [
    "profit_source",
    "TotalNetSellingPrice",
    "TotalCost",
    "TotalRawMaterialCost",
    "TotalProductionCost",
]


@dataclass(frozen=True)
class ReconResult:
    """Immutable result of a P&L reconciliation pass (Req 6.1, 6.2, 6.3, 6.5, 6.6).

    Attributes:
        discrepancies: One row per flagged :term:`Reconciliation_Discrepancy`
            with columns ``row_index`` (the source frame's index label),
            ``signed_diff_pnl`` (``profit_source - (TotalNetSellingPrice -
            TotalCost)``) and ``signed_diff_cost`` (``TotalCost -
            (TotalRawMaterialCost + TotalProductionCost)``). Both diffs carry
            their true sign. Empty (with those columns) when nothing is flagged.
        discrepancy_count: Number of flagged rows (equals ``len(discrepancies)``).
        loss_rows_per_sku: Count of loss-making rows per SKU, where a loss row
            has ``recomputed_profit < 0`` on the basis ``TotalNetSellingPrice -
            (TotalRawMaterialCost + TotalProductionCost)``. SKU keys are ``str``.
            SKUs with zero loss rows are omitted.
        unprocessable_count: Number of rows excluded from all checks because a
            required reconciliation field was missing or non-numeric.
        unprocessable_fields: Maps each unprocessable row's index label to the
            name of the first offending required field.
    """

    discrepancies: pd.DataFrame
    discrepancy_count: int
    loss_rows_per_sku: dict[str, int]
    unprocessable_count: int
    unprocessable_fields: dict[int, str]


def _to_number(value: object) -> float | None:
    """Return ``value`` as a float, or ``None`` if it is missing/non-numeric.

    A cell is unprocessable when it is null (``NaN``/``None``/``NaT``) or cannot
    be interpreted as a finite number. Numeric strings (e.g. ``"12.5"``) are
    accepted so the check does not spuriously reject clean data that arrived as
    text; genuinely non-numeric text (e.g. ``"n/a"``) is rejected.
    """
    if value is None:
        return None
    # ``pd.isna`` handles NaN / NaT / None; guard against array-likes.
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):  # pragma: no cover - non-scalar input.
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    # Reject inf / -inf: not usable in a currency reconciliation.
    if number != number or number in (float("inf"), float("-inf")):
        return None
    return number


def detect_reconciliation(sales_transaction: pd.DataFrame) -> ReconResult:
    """Reconcile each SalesTransaction row and report anomalies (Req 6.1–6.6).

    For every row of ``sales_transaction`` (the normalized SalesTransaction
    frame) the detector:

    * Excludes the row from all checks if any of
      :data:`REQUIRED_RECON_FIELDS` (``profit_source``,
      ``TotalNetSellingPrice``, ``TotalCost``, ``TotalRawMaterialCost``,
      ``TotalProductionCost``) is missing or non-numeric, flagging it
      unprocessable and attributing the first offending field (Req 6.6). Such a
      row is never a discrepancy and never a loss row.
    * Otherwise computes the two signed diffs and flags the row as a
      :term:`Reconciliation_Discrepancy` when
      ``|signed_diff_pnl| > RECONCILIATION_TOLERANCE`` **or**
      ``|signed_diff_cost| > RECONCILIATION_TOLERANCE`` (Req 6.1, 6.2, 6.3),
      where ``signed_diff_pnl = profit_source - (TotalNetSellingPrice -
      TotalCost)`` and ``signed_diff_cost = TotalCost - (TotalRawMaterialCost +
      TotalProductionCost)``.
    * Counts the row as a loss row for its SKU when the recomputed profit
      ``TotalNetSellingPrice - (TotalRawMaterialCost + TotalProductionCost)`` is
      strictly less than ``0`` (Req 6.5).

    The reconciliation tolerance is :data:`config.RECONCILIATION_TOLERANCE`.

    Args:
        sales_transaction: The normalized SalesTransaction frame. Required
            columns: ``profit_source``, ``TotalNetSellingPrice``, ``TotalCost``,
            ``TotalRawMaterialCost``, ``TotalProductionCost``, ``SKU``.

    Returns:
        An immutable :class:`ReconResult`.
    """
    tolerance = config.RECONCILIATION_TOLERANCE

    discrepancy_rows: list[dict[str, object]] = []
    loss_rows_per_sku: dict[str, int] = {}
    unprocessable_fields: dict[int, str] = {}

    for row_index in sales_transaction.index:
        row = sales_transaction.loc[row_index]

        # --- Unprocessable gate (Req 6.6) --------------------------------
        # Attribute the first missing-or-non-numeric required field, if any.
        offending_field: str | None = None
        values: dict[str, float] = {}
        for field_name in REQUIRED_RECON_FIELDS:
            raw = row[field_name] if field_name in row.index else None
            number = _to_number(raw)
            if number is None:
                offending_field = field_name
                break
            values[field_name] = number

        if offending_field is not None:
            unprocessable_fields[int(row_index)] = offending_field
            continue

        # --- Reconciliation checks (Req 6.1, 6.2, 6.3) -------------------
        signed_diff_pnl = values["profit_source"] - (
            values["TotalNetSellingPrice"] - values["TotalCost"]
        )
        signed_diff_cost = values["TotalCost"] - (
            values["TotalRawMaterialCost"] + values["TotalProductionCost"]
        )

        if abs(signed_diff_pnl) > tolerance or abs(signed_diff_cost) > tolerance:
            discrepancy_rows.append(
                {
                    "row_index": int(row_index),
                    "signed_diff_pnl": signed_diff_pnl,
                    "signed_diff_cost": signed_diff_cost,
                }
            )

        # --- Loss rows on the recomputed basis (Req 6.5) -----------------
        recomputed_profit = values["TotalNetSellingPrice"] - (
            values["TotalRawMaterialCost"] + values["TotalProductionCost"]
        )
        if recomputed_profit < 0:
            sku = _sku_key(row.get("SKU") if hasattr(row, "get") else None)
            loss_rows_per_sku[sku] = loss_rows_per_sku.get(sku, 0) + 1

    discrepancies = pd.DataFrame(
        discrepancy_rows,
        columns=["row_index", "signed_diff_pnl", "signed_diff_cost"],
    )

    return ReconResult(
        discrepancies=discrepancies,
        discrepancy_count=len(discrepancy_rows),
        loss_rows_per_sku=loss_rows_per_sku,
        unprocessable_count=len(unprocessable_fields),
        unprocessable_fields=unprocessable_fields,
    )


def _sku_key(value: object) -> str:
    """Return a stable string SKU key for grouping loss rows (Req 6.5).

    Integer-valued SKUs (e.g. ``200100004`` or ``200100004.0``) render without a
    trailing ``".0"`` so keys match :data:`config.VALID_SKUS`. Null SKUs render
    as an empty string.
    """
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except (TypeError, ValueError):  # pragma: no cover - non-scalar input.
        return str(value)
    if isinstance(value, (int, float)) and float(value).is_integer():
        return str(int(value))
    return str(value).strip()
