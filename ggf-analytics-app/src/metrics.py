"""Analytics layer: executive-summary and P&L metrics for the GGF Analytics App.

Pure, deterministic functions over the loaded :class:`~src.data_loader.DataBundle`
(or, more precisely, over its ``joined`` frame). This module has **no Streamlit
dependency** and does no plotting or I/O: the same inputs always produce the
same outputs and the caller's frames are never mutated.

Profit basis
------------
All profit figures use the *recomputed profit basis* documented in
:mod:`src.config` (:data:`config.PROFIT_BASIS_ID`)::

    recomputed_profit = TotalNetSellingPrice
                        - (TotalRawMaterialCost + TotalProductionCost)

The loader (``data_loader._build_joined``) already materializes this as the
``recomputed_profit`` column on ``joined``; :func:`recomputed_profit` here
exposes the same computation for reuse and testing.

For consistency with that basis, "total cost" in :class:`Totals` and elsewhere
is the *recomputed cost base* ``TotalRawMaterialCost + TotalProductionCost``
(not the source ``TotalCost`` aggregate, which is internally inconsistent on
every row). Hence ``total_profit == net_revenue - total_cost`` holds exactly.

Not-applicable handling
------------------------
Percentages with a zero denominator (a margin over zero net revenue, an
attainment over a zero target) are reported as ``None`` rather than a number or
an infinity, so callers can render them as "N/A" (Req 3.1, 3.2, 4.3, 5.1).

This module is grown incrementally by later tasks (sales/seasonality metrics in
6.1, cost drivers in 5.5, the expansion verdict in 9.5). Task 5.1 implements
only the executive-summary and per-SKU P&L functions below; sections are kept
separate so additions slot in without disturbing existing code.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Union

import pandas as pd

from . import config


# ---------------------------------------------------------------------------
# Result containers
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Totals:
    """2022 headline totals on the recomputed profit basis (Req 3.1).

    Attributes:
        net_revenue: Sum of ``TotalNetSellingPrice`` across all rows (IDR).
        total_cost: Recomputed cost base, i.e. the sum of
            ``TotalRawMaterialCost + TotalProductionCost`` (IDR). This is the
            cost consistent with :attr:`total_profit`, *not* the source
            ``TotalCost`` aggregate.
        total_profit: ``net_revenue - total_cost`` (IDR). Equivalently the sum
            of the per-row ``recomputed_profit``.
        margin_pct: ``round(total_profit / net_revenue * 100, 1)`` when
            ``net_revenue`` is greater than 0; ``None`` when ``net_revenue`` is
            0 (margin not applicable).
    """

    net_revenue: float
    total_cost: float
    total_profit: float
    margin_pct: Optional[float]


@dataclass(frozen=True)
class SkuProfit:
    """A single SKU's total 2022 profit, used to report the least profitable.

    Attributes:
        sku: The SKU identifier (string key, e.g. ``"200100004"``).
        product_name: The product name if known (e.g. ``"Pineapple Juice"``),
            otherwise ``None``.
        total_profit: The SKU's total recomputed profit for the period (IDR).
    """

    sku: str
    product_name: Optional[str]
    total_profit: float


# ---------------------------------------------------------------------------
# Recomputed profit (Req 6.4 basis; reused by every profit metric)
# ---------------------------------------------------------------------------


def recomputed_profit(
    data: Union[pd.Series, pd.DataFrame]
) -> Union[float, pd.Series]:
    """Recomputed profit on the primitive-cost basis (Req 6.4).

    Computes ``TotalNetSellingPrice - (TotalRawMaterialCost +
    TotalProductionCost)``. Accepts either a single row (a :class:`pandas.Series`
    indexed by column name), returning a ``float``, or a whole frame, returning
    a :class:`pandas.Series` of per-row profits aligned to the frame's index.

    The loader already adds this as the ``recomputed_profit`` column; this
    function exposes the same formula for reuse and independent testing. It
    reads the primitive components directly (not the ``recomputed_profit``
    column) so it stays correct even on a frame that predates that column.

    Args:
        data: A transaction row (Series) or a frame of transactions.

    Returns:
        A ``float`` for a Series input, or a per-row ``Series`` for a frame.
    """
    net = data["TotalNetSellingPrice"]
    cost = data["TotalRawMaterialCost"] + data["TotalProductionCost"]
    if isinstance(data, pd.Series):
        return float(net - cost)
    return net - cost


def _margin_pct(profit: float, net_revenue: float) -> Optional[float]:
    """Return ``round(profit / net_revenue * 100, 1)`` or ``None``.

    The margin is not applicable (``None``) when ``net_revenue`` is 0 or not a
    finite positive number, so a zero-denominator never yields an infinity or a
    ``NaN`` percentage (Req 3.1, 5.1).
    """
    if net_revenue is None or not net_revenue > 0:
        return None
    return round(profit / net_revenue * 100.0, 1)


# ---------------------------------------------------------------------------
# Executive-summary totals (Req 3.1)
# ---------------------------------------------------------------------------


def totals_2022(joined: pd.DataFrame) -> Totals:
    """Compute the 2022 headline totals on the recomputed basis (Req 3.1).

    Sums net revenue and the recomputed cost base over every row of ``joined``,
    derives total profit as their difference, and computes the overall margin
    percentage (rounded to one decimal place, or ``None`` when net revenue is
    0).

    Args:
        joined: The bundle's joined transactions frame. Must contain
            ``TotalNetSellingPrice``, ``TotalRawMaterialCost``, and
            ``TotalProductionCost``.

    Returns:
        A :class:`Totals` record. On an empty frame all monetary totals are 0.0
        and ``margin_pct`` is ``None``.
    """
    net_revenue = float(joined["TotalNetSellingPrice"].sum())
    total_cost = float(
        (joined["TotalRawMaterialCost"] + joined["TotalProductionCost"]).sum()
    )
    total_profit = net_revenue - total_cost
    return Totals(
        net_revenue=net_revenue,
        total_cost=total_cost,
        total_profit=total_profit,
        margin_pct=_margin_pct(total_profit, net_revenue),
    )


# ---------------------------------------------------------------------------
# Overall target attainment (Req 3.2)
# ---------------------------------------------------------------------------


def overall_target_attainment(joined: pd.DataFrame) -> Optional[float]:
    """Overall 2022 target attainment as a percentage (Req 3.2).

    Attainment compares actual net revenue against the sales target:
    ``round(actual_net_revenue / target_net_revenue * 100, 1)``, where actual
    is the sum of ``TotalNetSellingPrice`` and the target is the sum of
    ``SalesTarget`` (the monthly IDR target per SKU) across ``joined`` rows.

    Null targets (from transactions with no matching SalesTarget row) are
    treated as 0 in the target total. When the target total is 0 attainment is
    not applicable and ``None`` is returned rather than a number.

    Args:
        joined: The bundle's joined frame. Must contain ``TotalNetSellingPrice``
            and ``SalesTarget``.

    Returns:
        The attainment percentage rounded to one decimal place, or ``None`` when
        the target total is 0.
    """
    actual = float(joined["TotalNetSellingPrice"].sum())
    target = float(joined["SalesTarget"].fillna(0).sum())
    if not target > 0:
        return None
    return round(actual / target * 100.0, 1)


# ---------------------------------------------------------------------------
# Per-SKU P&L (Req 5.1)
# ---------------------------------------------------------------------------


def per_sku_pnl(joined: pd.DataFrame) -> pd.DataFrame:
    """Per-SKU total profit, net revenue, and margin for 2022 (Req 5.1).

    Groups ``joined`` by ``SKU`` and, for each SKU, sums net revenue and the
    recomputed cost base, derives total profit as their difference, and computes
    the margin percentage (one decimal place, or ``None`` when that SKU's net
    revenue is 0). The product name is included when a ``ProductName`` column is
    present on ``joined`` (falling back to :data:`config.SKU_MAP`).

    Rows with a null SKU are excluded (they cannot be attributed to a product).

    Args:
        joined: The bundle's joined frame. Must contain ``SKU``,
            ``TotalNetSellingPrice``, ``TotalRawMaterialCost``, and
            ``TotalProductionCost``; ``ProductName`` is optional.

    Returns:
        A frame indexed by ``sku`` (string) with columns ``product_name``,
        ``net_revenue``, ``total_profit``, and ``margin_pct``, sorted by SKU for
        determinism. Empty input yields an empty frame with those columns.
    """
    columns = ["product_name", "net_revenue", "total_profit", "margin_pct"]

    frame = joined[joined["SKU"].notna()]
    if frame.empty:
        empty = pd.DataFrame(columns=columns)
        empty.index.name = "sku"
        return empty

    has_product_name = "ProductName" in frame.columns

    records: list[dict[str, object]] = []
    index: list[str] = []
    for sku, group in frame.groupby("SKU", sort=True):
        sku_key = str(sku)
        net_revenue = float(group["TotalNetSellingPrice"].sum())
        cost = float(
            (group["TotalRawMaterialCost"] + group["TotalProductionCost"]).sum()
        )
        total_profit = net_revenue - cost

        product_name: Optional[str] = None
        if has_product_name:
            names = group["ProductName"].dropna()
            if not names.empty:
                product_name = str(names.iloc[0])
        if product_name is None:
            product_name = config.SKU_MAP.get(sku_key)

        index.append(sku_key)
        records.append(
            {
                "product_name": product_name,
                "net_revenue": net_revenue,
                "total_profit": total_profit,
                "margin_pct": _margin_pct(total_profit, net_revenue),
            }
        )

    result = pd.DataFrame(records, index=pd.Index(index, name="sku"))
    return result[columns]


# ---------------------------------------------------------------------------
# Least-profitable SKU selection (Req 3.4, 3.5, 5.2)
# ---------------------------------------------------------------------------


def least_profitable_sku(per_sku_pnl_df: pd.DataFrame) -> list[SkuProfit]:
    """Return every SKU tied for the lowest total profit (Req 3.4, 3.5).

    Selects the rows of a :func:`per_sku_pnl` frame whose ``total_profit``
    equals the minimum. Normally this is a single SKU (Pineapple on the real
    workbook), but all ties are returned so the executive summary can list each
    one (Req 3.5).

    Args:
        per_sku_pnl_df: A frame produced by :func:`per_sku_pnl`, indexed by SKU
            with a ``total_profit`` column (and optionally ``product_name``).

    Returns:
        A list of :class:`SkuProfit`, one per tied SKU, ordered by SKU for
        determinism. An empty input frame yields an empty list.
    """
    if per_sku_pnl_df.empty:
        return []

    min_profit = per_sku_pnl_df["total_profit"].min()
    tied = per_sku_pnl_df[per_sku_pnl_df["total_profit"] == min_profit]

    has_product_name = "product_name" in tied.columns

    result: list[SkuProfit] = []
    for sku in sorted(tied.index, key=str):
        row = tied.loc[sku]
        product_name = (
            None
            if not has_product_name or pd.isna(row["product_name"])
            else str(row["product_name"])
        )
        result.append(
            SkuProfit(
                sku=str(sku),
                product_name=product_name,
                total_profit=float(row["total_profit"]),
            )
        )
    return result

# ===========================================================================
# Task 6.1 — Sales-performance and seasonality metrics (Req 4.1, 4.3, 4.4, 7.3)
#
# Pure additions: no Streamlit / plotting. These build on the same ``joined``
# frame and the ``raw_mat_trend`` frame from the loaded bundle. Existing
# functions above are untouched.
# ===========================================================================


#: The twelve calendar-month keys of the 2022 analysis period, in order. The
#: sales matrix is built against this fixed set (Req 4.1) so gaps in the data
#: are zero-filled rather than omitted, regardless of which months appear.
MONTHS_2022: tuple[str, ...] = tuple(f"2022-{m:02d}" for m in range(1, 13))


@dataclass(frozen=True)
class SalesAnomaly:
    """A flagged (SKU, month) sales anomaly (Req 4.4).

    A (SKU, month) is anomalous when its actual monthly value deviates from that
    SKU's 12-month mean by more than :data:`config.SALES_ANOMALY_STD_THRESHOLD`
    standard deviations, or its target attainment is below
    :data:`config.ATTAINMENT_LOW_PCT` or above :data:`config.ATTAINMENT_HIGH_PCT`.

    Attributes:
        sku: The SKU identifier (string key).
        month: The calendar-month key (``"YYYY-MM"``).
        actual_value: The SKU/month actual sales value (``TotalNetSellingPrice``,
            IDR), with unrecorded months counted as 0.
        mean: The SKU's mean monthly value across the 12 months of 2022 (IDR).
        std: The SKU's population standard deviation of the 12 monthly values.
        attainment_pct: Target attainment for the SKU/month
            (``actual / target * 100``), or ``None`` when the target is 0 or
            missing (attainment not applicable).
        reason: A human-readable explanation naming the SKU, the month, and the
            deviation(s) observed (std-based and/or attainment-based).
    """

    sku: str
    month: str
    actual_value: float
    mean: float
    std: float
    attainment_pct: Optional[float]
    reason: str


# ---------------------------------------------------------------------------
# Monthly sales matrix (Req 4.1)
# ---------------------------------------------------------------------------


def monthly_sales_matrix(joined: pd.DataFrame) -> pd.DataFrame:
    """Actual quantity, value, and target per SKU per month, zero-filled (Req 4.1).

    Builds a long-form frame with exactly one row per (SKU, month) over the
    three valid SKUs (:data:`config.VALID_SKUS`) and the twelve months of 2022
    (:data:`MONTHS_2022`) — always 3 x 12 = 36 rows. For each (SKU, month) it
    sums the actual quantity (``Qty``) and actual value (``TotalNetSellingPrice``)
    from ``joined``; any (SKU, month) with no recorded sales is filled with 0
    rather than omitted. The ``SalesTarget`` for that SKU/month is attached when
    known (``NaN`` when there is no matching target).

    Only rows whose ``SKU`` is one of the three valid SKUs contribute to the
    actuals; other rows are ignored (they cannot be attributed to a tracked
    product). The full 2022 month set is used regardless of which months are
    present in the data.

    Args:
        joined: The bundle's joined frame. Must contain ``SKU``, ``month``,
            ``Qty``, and ``TotalNetSellingPrice``; ``SalesTarget`` is optional.

    Returns:
        A frame with columns ``sku``, ``month``, ``qty``, ``value``, and
        ``sales_target``, sorted by (``sku``, ``month``) for determinism.
    """
    columns = ["sku", "month", "qty", "value", "sales_target"]
    skus = sorted(config.VALID_SKUS)

    # Full (sku, month) grid: 3 x 12 = 36 combinations, always present.
    grid = pd.MultiIndex.from_product(
        [skus, MONTHS_2022], names=["sku", "month"]
    )

    frame = joined[joined["SKU"].isin(config.VALID_SKUS)].copy()
    frame = frame[frame["month"].isin(MONTHS_2022)]

    if frame.empty:
        actuals = pd.DataFrame(
            {"qty": 0.0, "value": 0.0}, index=grid
        )
    else:
        actuals = (
            frame.groupby(["SKU", "month"])
            .agg(qty=("Qty", "sum"), value=("TotalNetSellingPrice", "sum"))
            .reindex(grid, fill_value=0.0)
        )

    # SalesTarget per (SKU, month): take the aligned target (constant per
    # SKU/month after the loader's join); missing stays NaN.
    if "SalesTarget" in frame.columns and not frame.empty:
        targets = (
            frame.groupby(["SKU", "month"])["SalesTarget"].first().reindex(grid)
        )
    else:
        targets = pd.Series(index=grid, dtype="float64")

    result = actuals.reset_index()
    result["qty"] = result["qty"].astype("float64")
    result["value"] = result["value"].astype("float64")
    result["sales_target"] = targets.to_numpy()

    result = result.sort_values(["sku", "month"]).reset_index(drop=True)
    return result[columns]


# ---------------------------------------------------------------------------
# Per-SKU/month target attainment (Req 4.3)
# ---------------------------------------------------------------------------


def target_attainment_matrix(joined: pd.DataFrame) -> pd.DataFrame:
    """Per-SKU/month target attainment percentage, zero-target aware (Req 4.3).

    For each (SKU, month) of the zero-filled :func:`monthly_sales_matrix`,
    computes ``attainment_pct = actual_value / SalesTarget * 100``. Where the
    ``SalesTarget`` is 0 or missing, attainment is not applicable and reported
    as ``NaN`` (rather than a percentage or an infinity), so callers can render
    it as "N/A".

    Args:
        joined: The bundle's joined frame (see :func:`monthly_sales_matrix`).

    Returns:
        A frame with columns ``sku``, ``month``, ``value``, ``sales_target``,
        and ``attainment_pct`` (``NaN`` where target is 0/missing), one row per
        (SKU, month), sorted by (``sku``, ``month``).
    """
    matrix = monthly_sales_matrix(joined)

    target = matrix["sales_target"]
    # Not applicable when target is missing or not strictly positive.
    valid = target.notna() & (target > 0)

    attainment = pd.Series(float("nan"), index=matrix.index, dtype="float64")
    attainment.loc[valid] = (
        matrix.loc[valid, "value"] / target[valid] * 100.0
    )

    result = matrix[["sku", "month", "value", "sales_target"]].copy()
    result["attainment_pct"] = attainment
    return result


# ---------------------------------------------------------------------------
# Sales anomalies (Req 4.4)
# ---------------------------------------------------------------------------


def sales_anomalies(joined: pd.DataFrame) -> list[SalesAnomaly]:
    """Flag anomalous (SKU, month) sales values (Req 4.4).

    Using the zero-filled monthly matrix (so gaps count as 0), computes each
    SKU's mean and population standard deviation across its 12 monthly values.
    A (SKU, month) is flagged when either condition holds:

    - The actual value deviates from the SKU's 12-month mean by more than
      :data:`config.SALES_ANOMALY_STD_THRESHOLD` standard deviations
      (``abs(value - mean) > threshold * std``), evaluated only when ``std > 0``.
    - The target attainment is below :data:`config.ATTAINMENT_LOW_PCT` or above
      :data:`config.ATTAINMENT_HIGH_PCT` (evaluated only where attainment is
      applicable, i.e. the target is non-zero).

    Each flagged entry carries a human-readable ``reason`` identifying the SKU,
    the month, and the deviation(s) observed.

    Args:
        joined: The bundle's joined frame (see :func:`monthly_sales_matrix`).

    Returns:
        A list of :class:`SalesAnomaly`, ordered by (``sku``, ``month``).
    """
    attainment = target_attainment_matrix(joined)

    std_threshold = config.SALES_ANOMALY_STD_THRESHOLD
    low = config.ATTAINMENT_LOW_PCT
    high = config.ATTAINMENT_HIGH_PCT

    # Per-SKU mean/std of the 12 monthly values (population std, ddof=0).
    stats = attainment.groupby("sku")["value"].agg(
        mean="mean", std=lambda s: float(s.std(ddof=0))
    )

    anomalies: list[SalesAnomaly] = []
    for row in attainment.itertuples(index=False):
        sku = row.sku
        month = row.month
        value = float(row.value)
        mean = float(stats.at[sku, "mean"])
        std = float(stats.at[sku, "std"])
        att = row.attainment_pct
        att_pct = None if pd.isna(att) else float(att)

        reasons: list[str] = []

        # Std-deviation condition (only meaningful with non-zero spread).
        deviation = value - mean
        if std > 0 and abs(deviation) > std_threshold * std:
            n_std = abs(deviation) / std
            direction = "above" if deviation > 0 else "below"
            reasons.append(
                f"actual value {value:,.0f} is {n_std:.1f} std {direction} the "
                f"12-month mean {mean:,.0f}"
            )

        # Attainment condition (only where attainment is applicable).
        if att_pct is not None and (att_pct < low or att_pct > high):
            band = "below 50%" if att_pct < low else "above 150%"
            reasons.append(f"target attainment {att_pct:.1f}% is {band}")

        if reasons:
            reason = (
                f"SKU {sku} in {month}: " + "; ".join(reasons) + "."
            )
            anomalies.append(
                SalesAnomaly(
                    sku=str(sku),
                    month=str(month),
                    actual_value=value,
                    mean=mean,
                    std=std,
                    attainment_pct=att_pct,
                    reason=reason,
                )
            )

    anomalies.sort(key=lambda a: (a.sku, a.month))
    return anomalies


# ---------------------------------------------------------------------------
# Seasonality price comparison (Req 7.3)
# ---------------------------------------------------------------------------


def seasonality_price_compare(raw_mat_trend: pd.DataFrame) -> pd.DataFrame:
    """Average raw-material Close Price by peak vs non-peak season per SKU (Req 7.3).

    For each SKU, computes the arithmetic mean of ``Close Price`` over the rows
    where ``Is Peak Season == 1`` (peak) and where ``Is Peak Season == 0``
    (non-peak), and the delta (peak minus non-peak). A SKU with no rows in one
    partition reports ``NaN`` for that average (and for the delta).

    Args:
        raw_mat_trend: The bundle's ``raw_mat_trend`` frame. Must contain
            ``SKU``, ``Close Price``, and ``Is Peak Season``.

    Returns:
        A frame with columns ``sku``, ``avg_close_peak``, ``avg_close_non_peak``,
        and ``delta`` (``avg_close_peak - avg_close_non_peak``), one row per SKU,
        sorted by ``sku``. Empty input yields an empty frame with those columns.
    """
    columns = ["sku", "avg_close_peak", "avg_close_non_peak", "delta"]

    frame = raw_mat_trend[raw_mat_trend["SKU"].notna()]
    if frame.empty:
        return pd.DataFrame(columns=columns)

    peak_flag = pd.to_numeric(frame["Is Peak Season"], errors="coerce")
    close = pd.to_numeric(frame["Close Price"], errors="coerce")
    work = pd.DataFrame(
        {"sku": frame["SKU"].astype(str), "close": close, "peak": peak_flag}
    )

    records: list[dict[str, object]] = []
    for sku, group in work.groupby("sku", sort=True):
        peak_vals = group.loc[group["peak"] == 1, "close"]
        non_peak_vals = group.loc[group["peak"] == 0, "close"]
        avg_peak = float(peak_vals.mean()) if not peak_vals.dropna().empty else float("nan")
        avg_non_peak = (
            float(non_peak_vals.mean()) if not non_peak_vals.dropna().empty else float("nan")
        )
        records.append(
            {
                "sku": str(sku),
                "avg_close_peak": avg_peak,
                "avg_close_non_peak": avg_non_peak,
                "delta": avg_peak - avg_non_peak,
            }
        )

    return pd.DataFrame(records, columns=columns)


# ===========================================================================
# Task 5.5 — Cost-driver breakdown, negative-profit counts, improvement
# proposals (Req 5.2, 5.3, 5.4)
#
# Pure additions: no Streamlit / plotting. These build on the same ``joined``
# frame from the loaded bundle and reuse the recomputed cost base
# (``TotalRawMaterialCost + TotalProductionCost``) used throughout this module.
# Existing functions above are untouched.
# ===========================================================================


#: Canonical cost-driver identifiers surfaced by :func:`cost_driver_breakdown`
#: and referenced by every :class:`ImprovementProposal` (Req 5.3, 5.4).
COST_DRIVERS: tuple[str, str] = ("raw_material", "production")


@dataclass(frozen=True)
class ImprovementProposal:
    """A cost-reduction / profitability-improvement proposal (Req 5.4).

    Every proposal is anchored to a concrete SKU and one of the two cost
    drivers drawn from :func:`cost_driver_breakdown`, so it can be traced back
    to the breakdown row that motivates it.

    Attributes:
        sku: The SKU identifier the proposal targets (string key, e.g.
            ``"200100004"``). Always a SKU present in the cost-driver breakdown.
        product_name: The product name if known (e.g. ``"Pineapple Juice"``),
            otherwise ``None``.
        cost_driver: The cost driver the proposal addresses, one of
            :data:`COST_DRIVERS` (``"raw_material"`` or ``"production"``).
        cost_driver_pct: The driver's share of that SKU's recomputed cost base,
            as a percentage rounded to one decimal place (the figure from the
            breakdown that supports the proposal).
        text: A short, human-readable recommendation naming the SKU and driver.
    """

    sku: str
    product_name: Optional[str]
    cost_driver: str
    cost_driver_pct: float
    text: str


# ---------------------------------------------------------------------------
# Cost-driver breakdown (Req 5.3)
# ---------------------------------------------------------------------------


def cost_driver_breakdown(joined: pd.DataFrame) -> pd.DataFrame:
    """Per-SKU raw-material vs production cost, in IDR and as a share (Req 5.3).

    Groups ``joined`` by ``SKU`` and, for each SKU, sums the raw-material cost
    (``TotalRawMaterialCost``) and production cost (``TotalProductionCost``).
    The recomputed cost base is their sum, and each component is also expressed
    as a percentage of that SKU's cost base. By construction the two IDR
    components reconstruct the cost base and the two percentages sum to 100
    (within rounding), matching the recomputed profit basis used elsewhere in
    this module (Property 14).

    Where a SKU's cost base is 0, both percentages are reported as ``NaN``
    (a share of a zero base is not applicable) rather than a number or an
    infinity.

    Rows with a null SKU are excluded (they cannot be attributed to a product).

    Args:
        joined: The bundle's joined frame. Must contain ``SKU``,
            ``TotalRawMaterialCost``, and ``TotalProductionCost``;
            ``ProductName`` is optional.

    Returns:
        A frame with columns ``sku``, ``product_name``, ``raw_material_cost``,
        ``production_cost``, ``total_cost_base``, ``raw_material_pct``, and
        ``production_pct``, one row per SKU, sorted by ``sku`` for determinism.
        Percentages are rounded to one decimal place. Empty input yields an
        empty frame with those columns.
    """
    columns = [
        "sku",
        "product_name",
        "raw_material_cost",
        "production_cost",
        "total_cost_base",
        "raw_material_pct",
        "production_pct",
    ]

    frame = joined[joined["SKU"].notna()]
    if frame.empty:
        return pd.DataFrame(columns=columns)

    has_product_name = "ProductName" in frame.columns

    records: list[dict[str, object]] = []
    for sku, group in frame.groupby("SKU", sort=True):
        sku_key = str(sku)
        raw_material_cost = float(group["TotalRawMaterialCost"].sum())
        production_cost = float(group["TotalProductionCost"].sum())
        total_cost_base = raw_material_cost + production_cost

        if total_cost_base > 0:
            raw_material_pct = round(raw_material_cost / total_cost_base * 100.0, 1)
            production_pct = round(production_cost / total_cost_base * 100.0, 1)
        else:
            raw_material_pct = float("nan")
            production_pct = float("nan")

        product_name: Optional[str] = None
        if has_product_name:
            names = group["ProductName"].dropna()
            if not names.empty:
                product_name = str(names.iloc[0])
        if product_name is None:
            product_name = config.SKU_MAP.get(sku_key)

        records.append(
            {
                "sku": sku_key,
                "product_name": product_name,
                "raw_material_cost": raw_material_cost,
                "production_cost": production_cost,
                "total_cost_base": total_cost_base,
                "raw_material_pct": raw_material_pct,
                "production_pct": production_pct,
            }
        )

    return pd.DataFrame(records, columns=columns)


# ---------------------------------------------------------------------------
# Negative-profit month count (Req 5.2)
# ---------------------------------------------------------------------------


def negative_profit_month_count(joined: pd.DataFrame) -> dict[str, int]:
    """Count of loss-making months per SKU on the recomputed basis (Req 5.2).

    For each SKU, sums ``recomputed_profit`` within each calendar ``month`` and
    counts the months whose monthly total is strictly less than 0. On the real
    workbook this is ``{"200100001": 0, "200100003": 0, "200100004": 7}`` —
    Pineapple has 7 negative-profit months of 12.

    Every valid SKU (:data:`config.VALID_SKUS`) appears in the result, even
    those with zero negative months, so callers get an explicit 0 rather than a
    missing key. Any additional SKU actually present in the data is included
    too. Rows with a null SKU or null month are ignored.

    Recomputed profit is derived from the primitive cost components via
    :func:`recomputed_profit` (not the ``recomputed_profit`` column), so the
    count is correct even on a frame that predates that column.

    Args:
        joined: The bundle's joined frame. Must contain ``SKU``, ``month``,
            ``TotalNetSellingPrice``, ``TotalRawMaterialCost``, and
            ``TotalProductionCost``.

    Returns:
        A dict mapping each SKU (string key) to its count of negative-profit
        months, always including the three valid SKUs with a 0 default.
    """
    counts: dict[str, int] = {sku: 0 for sku in sorted(config.VALID_SKUS)}

    frame = joined[joined["SKU"].notna() & joined["month"].notna()].copy()
    if frame.empty:
        return counts

    frame["_profit"] = recomputed_profit(frame)

    monthly = frame.groupby(["SKU", "month"])["_profit"].sum()
    negative_by_sku = (monthly < 0).groupby(level="SKU").sum()

    for sku, n_negative in negative_by_sku.items():
        counts[str(sku)] = int(n_negative)

    return counts


# ---------------------------------------------------------------------------
# Improvement proposals (Req 5.4)
# ---------------------------------------------------------------------------


def improvement_proposals(joined: pd.DataFrame) -> list[ImprovementProposal]:
    """Generate cost-reduction / profitability-improvement proposals (Req 5.4).

    Derives at least one proposal from :func:`cost_driver_breakdown`, each
    anchored to a specific SKU and a specific cost driver in
    :data:`COST_DRIVERS`. The proposals are data-driven:

    - **Least-profitable SKU.** The SKU with the lowest total recomputed profit
      (Pineapple on the real workbook) gets a proposal targeting its dominant
      cost driver (the larger share of its cost base), since that driver offers
      the greatest leverage to lift its profit.
    - **Raw-material-dominated SKUs.** Any SKU whose raw-material cost is the
      dominant share of its cost base gets a proposal to address raw-material
      cost (e.g. sourcing / peak-season purchasing), reflecting that
      raw-material cost dominates production cost across this dataset.

    Duplicate (SKU, cost_driver) pairs are collapsed so each proposal is
    distinct. When a breakdown is available the result always contains at least
    one proposal (Req 5.4, Property 15).

    Args:
        joined: The bundle's joined frame (see :func:`cost_driver_breakdown` and
            :func:`per_sku_pnl`).

    Returns:
        A list of :class:`ImprovementProposal`, ordered by (``sku``,
        ``cost_driver``) for determinism. Empty only when there are no SKUs to
        analyze (an empty breakdown).
    """
    breakdown = cost_driver_breakdown(joined)
    if breakdown.empty:
        return []

    # Fast lookup of each SKU's breakdown row.
    by_sku = {row["sku"]: row for _, row in breakdown.iterrows()}

    def _dominant_driver(row: pd.Series) -> str:
        """The cost driver with the larger IDR share for this SKU."""
        if float(row["raw_material_cost"]) >= float(row["production_cost"]):
            return "raw_material"
        return "production"

    def _driver_pct(row: pd.Series, driver: str) -> float:
        pct = (
            row["raw_material_pct"]
            if driver == "raw_material"
            else row["production_pct"]
        )
        return 0.0 if pd.isna(pct) else float(pct)

    def _driver_label(driver: str) -> str:
        return "raw-material cost" if driver == "raw_material" else "production cost"

    # (sku, driver) -> proposal, de-duplicated.
    proposals: dict[tuple[str, str], ImprovementProposal] = {}

    def _add(sku: str, driver: str, text: str) -> None:
        key = (sku, driver)
        if key in proposals:
            return
        row = by_sku[sku]
        proposals[key] = ImprovementProposal(
            sku=sku,
            product_name=(
                None if pd.isna(row["product_name"]) else str(row["product_name"])
            ),
            cost_driver=driver,
            cost_driver_pct=_driver_pct(row, driver),
            text=text,
        )

    # 1) Least-profitable SKU -> its dominant cost driver.
    pnl = per_sku_pnl(joined)
    if not pnl.empty:
        for sku_profit in least_profitable_sku(pnl):
            sku = sku_profit.sku
            if sku not in by_sku:
                continue
            row = by_sku[sku]
            driver = _dominant_driver(row)
            pct = _driver_pct(row, driver)
            name = row["product_name"] if not pd.isna(row["product_name"]) else sku
            _add(
                sku,
                driver,
                (
                    f"{name} is the least-profitable SKU (total profit "
                    f"{sku_profit.total_profit:,.0f} IDR); {_driver_label(driver)} "
                    f"is its dominant cost driver at {pct:.1f}% of its cost base. "
                    f"Prioritize reducing {_driver_label(driver)} for this SKU to "
                    f"restore profitability."
                ),
            )

    # 2) Any SKU where raw-material cost dominates its cost base.
    for _, row in breakdown.iterrows():
        sku = str(row["sku"])
        if float(row["raw_material_cost"]) <= float(row["production_cost"]):
            continue
        pct = _driver_pct(row, "raw_material")
        name = row["product_name"] if not pd.isna(row["product_name"]) else sku
        _add(
            sku,
            "raw_material",
            (
                f"{name}'s raw-material cost dominates its cost base at "
                f"{pct:.1f}%. Target raw-material cost through sourcing "
                f"negotiation and peak-season purchasing to widen its margin."
            ),
        )

    # Fallback: guarantee at least one proposal (Req 5.4) even in degenerate
    # cost structures where neither rule above fired.
    if not proposals:
        row = breakdown.iloc[0]
        sku = str(row["sku"])
        driver = _dominant_driver(row)
        pct = _driver_pct(row, driver)
        name = row["product_name"] if not pd.isna(row["product_name"]) else sku
        _add(
            sku,
            driver,
            (
                f"{name}'s largest cost driver is {_driver_label(driver)} at "
                f"{pct:.1f}% of its cost base; review it for cost-reduction "
                f"opportunities."
            ),
        )

    return [proposals[key] for key in sorted(proposals)]

# ===========================================================================
# Task 9.5 — Expansion verdict and forecast-to-verdict relationship
# (Req 3.3, 8.7; design: expansion_verdict; Properties 9 and 23)
#
# Pure additions: no Streamlit / plotting. These build on the executive-summary
# and P&L functions above (:class:`Totals`, :func:`per_sku_pnl`) and on the
# forecasting layer's :class:`~src.forecasting.ForecastResult`. To avoid a hard
# import cycle (``forecasting`` imports ``config`` only, and ``metrics`` must
# not depend on ``forecasting``), forecasts are accepted structurally: any
# sequence of objects exposing ``.forecast_profit`` (a pandas Series indexed
# chronologically) and ``.excluded`` (a bool) works. Existing functions above
# are untouched.
# ===========================================================================


from typing import Sequence  # noqa: E402  (kept with this task's additions)


#: The fixed domain of expansion recommendations (Req 3.3, Property 9). The
#: verdict is always exactly one of these three strings.
VERDICT_RECOMMENDATIONS: tuple[str, str, str] = (
    "Expand",
    "Optimize First",
    "Expand and Optimize",
)

#: The fixed domain of forecast-to-verdict relationship labels (Req 8.7,
#: Property 23). The relationship is always exactly one of these three strings.
FORECAST_RELATIONSHIPS: tuple[str, str, str] = (
    "supports",
    "neutral",
    "contradicts",
)

#: Minimum overall margin (%) considered "healthy" for an outright expand. Below
#: this the business is treated as too thin to expand without optimizing first.
HEALTHY_MARGIN_PCT: float = 10.0

#: A per-SKU margin (%) at or below this marks the SKU as a weak/laggard product
#: even when the business is profitable overall — the trigger to pair expansion
#: with optimization ("Expand and Optimize").
WEAK_SKU_MARGIN_PCT: float = 5.0

#: Relative change of the aggregate forecast profit trajectory (last vs first
#: forecast month) beyond +/- this fraction classifies the trend. A rise of at
#: least this fraction is "supports"; a fall of at least this fraction is
#: "contradicts"; anything in between is "neutral" (flat).
FORECAST_TREND_TOLERANCE: float = 0.05


def _aggregate_forecast_profit(
    forecasts: Sequence[object],
) -> pd.Series:
    """Sum the per-month forecast profit across all fitted SKUs.

    Adds the ``forecast_profit`` series of every non-excluded forecast, aligning
    on the shared chronological month index. Excluded SKUs (and any object whose
    ``forecast_profit`` is empty) contribute nothing.

    Args:
        forecasts: A sequence of objects exposing ``.forecast_profit`` (a
            :class:`pandas.Series` indexed by month) and ``.excluded`` (bool),
            e.g. :class:`~src.forecasting.ForecastResult` values from
            :func:`src.forecasting.forecast_all`.

    Returns:
        A :class:`pandas.Series` of summed forecast profit per month (index
        preserved and sorted). Empty when there are no fitted forecasts.
    """
    total: Optional[pd.Series] = None
    for result in forecasts:
        if getattr(result, "excluded", False):
            continue
        profit = getattr(result, "forecast_profit", None)
        if profit is None or len(profit) == 0:
            continue
        series = pd.Series(profit, dtype="float64")
        total = series if total is None else total.add(series, fill_value=0.0)

    if total is None:
        return pd.Series(dtype="float64")
    return total.sort_index()


def forecast_relationship(forecasts: Sequence[object]) -> str:
    """Classify the aggregate forecast profit trend (Req 8.7, Property 23).

    Compares the aggregate 12-month forecast profit trajectory (summed across
    all fitted SKUs) at its last month versus its first month and returns one of
    :data:`FORECAST_RELATIONSHIPS`:

    - ``"supports"`` when the aggregate forecast profit is flat-to-up, i.e. it
      rises by at least :data:`FORECAST_TREND_TOLERANCE` of the first month, or
      moves within the flat band while non-negative overall.
    - ``"contradicts"`` when it clearly declines, i.e. falls by more than
      :data:`FORECAST_TREND_TOLERANCE` of the (positive) first month, or the
      aggregate forecast profit is negative overall.
    - ``"neutral"`` otherwise (essentially flat, or no basis to judge — e.g.
      every SKU excluded, or a single forecast month).

    The comparison is first-vs-last of the aggregate trajectory (a simple,
    deterministic proxy for slope over the horizon). Thresholds are explicit
    constants so the classification is reproducible.

    Args:
        forecasts: A sequence of forecast objects (see
            :func:`_aggregate_forecast_profit`).

    Returns:
        Exactly one of :data:`FORECAST_RELATIONSHIPS`.
    """
    aggregate = _aggregate_forecast_profit(forecasts)

    # No fitted forecast, or a single point: no trend to judge -> neutral.
    if aggregate.empty or aggregate.shape[0] < 2:
        return "neutral"

    first = float(aggregate.iloc[0])
    last = float(aggregate.iloc[-1])

    # A negative aggregate forecast profit is a clear headwind regardless of
    # slope: forecast profit that is under water contradicts an expand story.
    if last < 0:
        return "contradicts"

    # Relative change vs the first month. When the first month is ~0 fall back
    # to the absolute direction of the change.
    if first > 0:
        rel_change = (last - first) / first
    else:
        rel_change = 0.0 if last == first else (1.0 if last > first else -1.0)

    if rel_change > FORECAST_TREND_TOLERANCE:
        return "supports"
    if rel_change < -FORECAST_TREND_TOLERANCE:
        return "contradicts"
    # Within the flat band: a flat-but-profitable trajectory supports staying
    # the course; a flat trajectory is otherwise neutral.
    return "supports" if last >= 0 and first >= 0 else "neutral"


@dataclass(frozen=True)
class Verdict:
    """The expansion recommendation and its forecast relationship (Req 3.3, 8.7).

    Attributes:
        recommendation: Exactly one of :data:`VERDICT_RECOMMENDATIONS`
            (``"Expand"``, ``"Optimize First"``, or ``"Expand and Optimize"``).
        rationale: A supporting explanation of between 1 and 3 sentences
            (Req 3.3, Property 9), referencing the concrete situation (overall
            profitability, the weakest SKU, and the forecast direction).
        forecast_relationship: Exactly one of :data:`FORECAST_RELATIONSHIPS`
            (``"supports"``, ``"neutral"``, or ``"contradicts"``), stating how
            the forecast relates to the recommendation (Req 8.7, Property 23).
    """

    recommendation: str
    rationale: str
    forecast_relationship: str


def _weak_skus(per_sku_pnl_df: pd.DataFrame) -> list[str]:
    """Return SKUs that are loss-prone or thin-margined (verdict laggards).

    A SKU is "weak" when its total profit is not strictly positive (a loss or
    break-even) or its margin percentage is at or below
    :data:`WEAK_SKU_MARGIN_PCT`. A ``None``/``NaN`` margin (zero net revenue) is
    also treated as weak. The result is ordered by ascending total profit so the
    weakest SKU comes first (useful for the rationale).

    Args:
        per_sku_pnl_df: A :func:`per_sku_pnl` frame indexed by SKU with
            ``total_profit`` and ``margin_pct`` columns.

    Returns:
        The list of weak SKU keys (strings), weakest first. Empty when every SKU
        is healthy.
    """
    if per_sku_pnl_df.empty:
        return []

    ordered = per_sku_pnl_df.sort_values("total_profit")
    weak: list[str] = []
    for sku, row in ordered.iterrows():
        profit = float(row["total_profit"])
        margin = row["margin_pct"]
        margin_weak = pd.isna(margin) or float(margin) <= WEAK_SKU_MARGIN_PCT
        if not profit > 0 or margin_weak:
            weak.append(str(sku))
    return weak


def _sku_label(per_sku_pnl_df: pd.DataFrame, sku: str) -> str:
    """Return a display label for ``sku`` (product name if known, else SKU)."""
    if sku in per_sku_pnl_df.index and "product_name" in per_sku_pnl_df.columns:
        name = per_sku_pnl_df.at[sku, "product_name"]
        if not pd.isna(name):
            return str(name)
    return sku


def expansion_verdict(
    totals: Totals,
    per_sku_pnl_df: pd.DataFrame,
    forecasts: Sequence[object],
) -> Verdict:
    """Derive the expansion verdict from totals, per-SKU P&L, and forecasts.

    Implements Req 3.3 (the verdict domain and 1-3 sentence rationale) and
    Req 8.7 (the forecast-to-verdict relationship). The decision is deterministic
    and documented:

    - **"Expand"** when the business is profitable overall (``total_profit > 0``
      and ``margin_pct`` at least :data:`HEALTHY_MARGIN_PCT`), *every* SKU is
      healthy (no weak/loss-prone SKU per :func:`_weak_skus`), and the aggregate
      forecast profit trend does not contradict (relationship is not
      ``"contradicts"``). Expand the whole line.
    - **"Expand and Optimize"** when the business is profitable overall but at
      least one SKU is weak (a loss-prone or thin-margin laggard) while the
      forecast does not contradict — expand the winners and fix the laggard.
    - **"Optimize First"** when the business is not profitable overall (or its
      overall margin is below the healthy threshold), or the forecast clearly
      contradicts (declining/negative aggregate forecast profit) — optimize
      before committing to expansion.

    The ``forecast_relationship`` is computed independently by
    :func:`forecast_relationship` and always lies in
    :data:`FORECAST_RELATIONSHIPS`. The rationale is always 1-3 sentences and
    references the concrete situation (overall profit/margin, the weakest SKU
    and its negative-profit months where relevant, and the forecast direction).

    Args:
        totals: The 2022 headline :class:`Totals` (has ``total_profit`` and
            ``margin_pct``).
        per_sku_pnl_df: A :func:`per_sku_pnl` frame (indexed by SKU with
            ``total_profit``, ``margin_pct``, and optionally ``product_name``).
        forecasts: The list of :class:`~src.forecasting.ForecastResult` from
            :func:`src.forecasting.forecast_all` (accepted structurally: any
            sequence of objects exposing ``.forecast_profit`` and ``.excluded``).

    Returns:
        A :class:`Verdict` whose ``recommendation`` is one of
        :data:`VERDICT_RECOMMENDATIONS` and whose ``forecast_relationship`` is
        one of :data:`FORECAST_RELATIONSHIPS`.
    """
    relationship = forecast_relationship(forecasts)

    margin_pct = totals.margin_pct
    overall_profitable = (
        totals.total_profit > 0
        and margin_pct is not None
        and margin_pct >= HEALTHY_MARGIN_PCT
    )
    forecast_contradicts = relationship == "contradicts"

    weak = _weak_skus(per_sku_pnl_df)

    margin_text = "n/a" if margin_pct is None else f"{margin_pct:.1f}%"
    forecast_phrase = {
        "supports": "the 12-month forecast supports this",
        "neutral": "the 12-month forecast is neutral",
        "contradicts": "the 12-month forecast contradicts this",
    }[relationship]

    # --- Optimize First: not profitable enough, or forecast contradicts ----
    if not overall_profitable or forecast_contradicts:
        if not overall_profitable:
            situation = (
                f"2022 profit of {totals.total_profit:,.0f} IDR at a "
                f"{margin_text} margin is not a healthy base for expansion"
            )
        else:
            situation = (
                f"2022 is profitable ({totals.total_profit:,.0f} IDR at "
                f"{margin_text} margin) but the forward outlook is declining"
            )
        rationale = (
            f"{situation}. Optimize existing operations before committing to "
            f"expansion; {forecast_phrase}."
        )
        return Verdict(
            recommendation="Optimize First",
            rationale=rationale,
            forecast_relationship=relationship,
        )

    # --- Expand and Optimize: profitable overall but a laggard SKU ---------
    if weak:
        laggard = _sku_label(per_sku_pnl_df, weak[0])
        rationale = (
            f"2022 is profitable overall ({totals.total_profit:,.0f} IDR at a "
            f"{margin_text} margin), so the winners are worth expanding, but "
            f"{laggard} is a weak, loss-prone laggard that should be optimized "
            f"in parallel. Expand the strong SKUs while fixing {laggard}; "
            f"{forecast_phrase}."
        )
        return Verdict(
            recommendation="Expand and Optimize",
            rationale=rationale,
            forecast_relationship=relationship,
        )

    # --- Expand: profitable overall, every SKU healthy, forecast not down --
    rationale = (
        f"2022 is profitable overall ({totals.total_profit:,.0f} IDR at a "
        f"{margin_text} margin) with every SKU healthy, so the business is a "
        f"sound base for expansion; {forecast_phrase}."
    )
    return Verdict(
        recommendation="Expand",
        rationale=rationale,
        forecast_relationship=relationship,
    )
