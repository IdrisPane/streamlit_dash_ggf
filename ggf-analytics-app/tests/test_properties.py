"""Property-based tests (hypothesis) for the data layer (Properties 1-6).

Each test carries a traceability tag of the form
    # Feature: ggf-analytics-app, Property N: <property text>
and runs a minimum of 100 iterations (``@settings(max_examples=100)``).

Properties 1-6 target ``src.data_loader``:
  1. Column normalization preserves values under a fixed name map.
  2. Type coercion preserves magnitude and sign.
  3. Unparseable cells become null and are recorded; valid cells survive.
  4. Join is restricted to the three SKUs and matches products correctly.
  5. Transactions align to targets by SKU and calendar month.
  6. Unmatched keys are retained, nulled, and recorded.
"""

from __future__ import annotations

import math
import re

import numpy as np
import pandas as pd
from hypothesis import assume, given, settings
from hypothesis import strategies as st

from src import config
from src.data_loader import (
    ParseError,
    UnmatchedKey,
    _build_joined,
    coerce_types,
    normalize_columns,
)

MAX_EXAMPLES = 100

VALID_SKU_LIST = sorted(config.VALID_SKUS)  # ["200100001", "200100003", "200100004"]


# ---------------------------------------------------------------------------
# Shared strategies
# ---------------------------------------------------------------------------

#: Cell values that survive an object column round-trip unchanged (no NaN/None,
#: which pandas would coerce, and no float NaN which breaks equality).
_scalar_values = st.one_of(
    st.integers(min_value=-1_000_000, max_value=1_000_000),
    st.floats(allow_nan=False, allow_infinity=False, min_value=-1e9, max_value=1e9),
    st.text(min_size=0, max_size=12),
    st.booleans(),
)

#: Column names that are never the source-typo names (so "other columns
#: untouched" is meaningfully testable).
_other_column_names = st.text(
    alphabet="abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ_0123456789",
    min_size=1,
    max_size=8,
).filter(lambda s: s not in config.COLUMN_NORMALIZATION_MAP)


# ===========================================================================
# Property 1
# ===========================================================================
# Feature: ggf-analytics-app, Property 1: Column normalization preserves values under a fixed name map
@settings(max_examples=MAX_EXAMPLES)
@given(data=st.data())
def test_property_1_normalization_preserves_values(data) -> None:
    """For any frame containing the source-typo columns (and arbitrary other
    columns with arbitrary values), after ``normalize_columns`` the canonical
    names are present, the typo names are absent, other columns are untouched,
    and every cell value is unchanged. Validates Requirements 1.2, 1.3, 2.4.
    """
    n_rows = data.draw(st.integers(min_value=1, max_value=6))

    # At least one typo column must be present for the property to be meaningful.
    include_txn = data.draw(st.booleans())
    include_pnl = data.draw(st.booleans())
    assume(include_txn or include_pnl)

    source_cols: list[str] = []
    if include_txn:
        source_cols.append("TranscationDate")
    if include_pnl:
        source_cols.append("Profit&Lost")

    n_other = data.draw(st.integers(min_value=0, max_value=3))
    other_cols = data.draw(
        st.lists(_other_column_names, min_size=n_other, max_size=n_other, unique=True)
    )

    columns = source_cols + other_cols
    frame_data: dict[str, list] = {}
    for col in columns:
        frame_data[col] = data.draw(
            st.lists(_scalar_values, min_size=n_rows, max_size=n_rows)
        )

    df = pd.DataFrame(frame_data, dtype=object)
    original = df.copy(deep=True)

    result = normalize_columns(df)

    # Canonical names present, typo names absent.
    if include_txn:
        assert "transaction_date" in result.columns
        assert "TranscationDate" not in result.columns
    if include_pnl:
        assert "profit_source" in result.columns
        assert "Profit&Lost" not in result.columns

    # Other columns untouched (name preserved).
    for col in other_cols:
        assert col in result.columns

    # Caller's frame is not mutated.
    pd.testing.assert_frame_equal(df, original)

    # Every cell value unchanged, tracked through the rename map.
    for col in columns:
        new_name = config.COLUMN_NORMALIZATION_MAP.get(col, col)
        assert list(result[new_name]) == list(original[col])


# ===========================================================================
# Property 2
# ===========================================================================
# Feature: ggf-analytics-app, Property 2: Type coercion preserves magnitude and sign
@settings(max_examples=MAX_EXAMPLES)
@given(
    numbers=st.lists(
        st.one_of(
            st.integers(min_value=-1_000_000, max_value=1_000_000),
            st.floats(
                allow_nan=False, allow_infinity=False, min_value=-1e9, max_value=1e9
            ),
        ),
        min_size=1,
        max_size=8,
    ),
    dates=st.lists(
        st.dates(min_value=pd.Timestamp("2000-01-01").date(),
                 max_value=pd.Timestamp("2030-12-31").date()),
        min_size=1,
        max_size=8,
    ),
    as_string=st.booleans(),
)
def test_property_2_coercion_preserves_magnitude_sign(numbers, dates, as_string) -> None:
    """For any numerically parseable cell (ints/floats incl. negatives), the
    coerced numeric value equals the source magnitude and sign; date-parseable
    cells coerce to the corresponding date. Validates Requirement 1.4.
    """
    n = min(len(numbers), len(dates))
    numbers = numbers[:n]
    dates = dates[:n]

    # Represent values as native or as strings to exercise both paths.
    if as_string:
        num_col = [repr(x) for x in numbers]
        date_col = [d.isoformat() for d in dates]
    else:
        num_col = list(numbers)
        date_col = [pd.Timestamp(d) for d in dates]

    df = pd.DataFrame({"num": num_col, "dt": date_col}, dtype=object)
    schema = {"num": "numeric", "dt": "date"}

    coerced, errors = coerce_types(df, schema, sheet="S")

    assert errors == [], f"unexpected parse errors on valid values: {errors}"

    for i, source in enumerate(numbers):
        got = coerced["num"].iloc[i]
        assert not pd.isna(got)
        # Magnitude and sign preserved.
        assert math.isclose(float(got), float(source), rel_tol=1e-9, abs_tol=1e-6)
        assert (float(got) < 0) == (float(source) < 0)

    for i, source in enumerate(dates):
        got = coerced["dt"].iloc[i]
        assert not pd.isna(got)
        got_ts = pd.Timestamp(got)
        assert got_ts.year == source.year
        assert got_ts.month == source.month
        assert got_ts.day == source.day


# ===========================================================================
# Property 3
# ===========================================================================

#: A token that pandas cannot parse as a number.
_unparseable_numeric = st.sampled_from(
    ["abc", "N/A", "--", "1.2.3", "12x", "$", "one", "  ", "#REF!", "??"]
)
#: A token that pandas cannot parse as a date.
_unparseable_date = st.sampled_from(
    ["notadate", "2022-13-45", "abc", "13/13/2022", "??", "month", "-", "#N/A"]
)


# Feature: ggf-analytics-app, Property 3: Unparseable cells become null and are recorded, valid cells survive
@settings(max_examples=MAX_EXAMPLES)
@given(data=st.data())
def test_property_3_unparseable_cells_nulled_and_recorded(data) -> None:
    """For a frame mixing parseable and unparseable cells across a numeric and a
    date column, every unparseable cell becomes null with exactly one recorded
    ParseError (sheet, column, row) and every parseable cell retains its value.
    Validates Requirement 1.5.
    """
    n_rows = data.draw(st.integers(min_value=1, max_value=8))

    good_numbers = data.draw(
        st.lists(
            st.integers(min_value=-10_000, max_value=10_000),
            min_size=n_rows,
            max_size=n_rows,
        )
    )
    good_dates = data.draw(
        st.lists(
            st.dates(min_value=pd.Timestamp("2010-01-01").date(),
                     max_value=pd.Timestamp("2025-12-31").date()),
            min_size=n_rows,
            max_size=n_rows,
        )
    )
    # Which rows are corrupted in each column (independent per column).
    num_bad = data.draw(st.lists(st.booleans(), min_size=n_rows, max_size=n_rows))
    date_bad = data.draw(st.lists(st.booleans(), min_size=n_rows, max_size=n_rows))

    num_col: list[object] = []
    for i in range(n_rows):
        if num_bad[i]:
            num_col.append(data.draw(_unparseable_numeric))
        else:
            num_col.append(good_numbers[i])

    date_col: list[object] = []
    for i in range(n_rows):
        if date_bad[i]:
            date_col.append(data.draw(_unparseable_date))
        else:
            date_col.append(good_dates[i].isoformat())

    df = pd.DataFrame({"num": num_col, "dt": date_col}, dtype=object)
    schema = {"num": "numeric", "dt": "date"}

    coerced, errors = coerce_types(df, schema, sheet="Sheet")

    # Every unparseable cell is null; every parseable cell retains its value.
    for i in range(n_rows):
        if num_bad[i]:
            assert pd.isna(coerced["num"].iloc[i])
        else:
            assert float(coerced["num"].iloc[i]) == float(good_numbers[i])

        if date_bad[i]:
            assert pd.isna(coerced["dt"].iloc[i])
        else:
            got = pd.Timestamp(coerced["dt"].iloc[i])
            assert got.date() == good_dates[i]

    # Exactly one ParseError per corrupted cell, with correct coordinates.
    expected = set()
    for i in range(n_rows):
        if num_bad[i]:
            expected.add(("Sheet", "num", i))
        if date_bad[i]:
            expected.add(("Sheet", "dt", i))

    got = [(e.sheet, e.column, e.row_index) for e in errors]
    assert len(got) == len(expected), f"errors={got} expected={expected}"
    assert set(got) == expected
    for e in errors:
        assert isinstance(e, ParseError)


# ---------------------------------------------------------------------------
# Synthetic-frame builders for join properties (4-6)
# ---------------------------------------------------------------------------

#: Foreign SKUs guaranteed not in VALID_SKUS.
_foreign_skus = st.sampled_from(["999", "200100002", "100000000", "abc", "0"])

_months_2022 = [f"2022-{m:02d}" for m in range(1, 13)]


def _make_mst_product(skus: list[str]) -> pd.DataFrame:
    """Build a MstProduct frame with SKU + ProductName for the given SKUs."""
    return pd.DataFrame(
        {
            "SKU": skus,
            "ProductName": [config.SKU_MAP.get(s, f"Product {s}") for s in skus],
        }
    )


def _make_sales_target(rows: list[tuple[str, str, float]]) -> pd.DataFrame:
    """Build a SalesTarget frame from (sku, 'YYYY-MM', target) tuples."""
    return pd.DataFrame(
        {
            "SKU": [r[0] for r in rows],
            "Period": [pd.Timestamp(f"{r[1]}-01") for r in rows],
            "SalesTarget": [r[2] for r in rows],
        }
    )


def _make_sales_transaction(rows: list[tuple[str, str]]) -> pd.DataFrame:
    """Build a normalized SalesTransaction frame from (sku, 'YYYY-MM') tuples.

    Numeric cost/revenue columns are filled deterministically so
    ``recomputed_profit`` is defined.
    """
    n = len(rows)
    return pd.DataFrame(
        {
            "SKU": [r[0] for r in rows],
            "transaction_date": [pd.Timestamp(f"{r[1]}-15") for r in rows],
            "TotalNetSellingPrice": [1000.0] * n,
            "TotalRawMaterialCost": [300.0] * n,
            "TotalProductionCost": [200.0] * n,
        }
    )


# ===========================================================================
# Property 4
# ===========================================================================
# Feature: ggf-analytics-app, Property 4: Join is restricted to the three SKUs and matches products correctly
@settings(max_examples=MAX_EXAMPLES)
@given(
    sku_choices=st.lists(
        st.one_of(st.sampled_from(VALID_SKU_LIST), _foreign_skus),
        min_size=1,
        max_size=10,
    )
)
def test_property_4_product_join_restricted_to_valid_skus(sku_choices) -> None:
    """Joined product fields are populated only for SKUs in the valid set and
    equal that product's MstProduct values; foreign SKUs keep null product
    fields. Validates Requirement 1.7.
    """
    # Every transaction shares a single month for which a target exists, so the
    # product-field outcome is isolated from the target join.
    month = "2022-06"
    txn_rows = [(sku, month) for sku in sku_choices]
    sales_transaction = _make_sales_transaction(txn_rows)

    mst_product = _make_mst_product(VALID_SKU_LIST)
    target_rows = [(sku, month, 5000.0) for sku in VALID_SKU_LIST]
    sales_target = _make_sales_target(target_rows)

    joined, _ = _build_joined(sales_transaction, mst_product, sales_target)

    assert len(joined) == len(sku_choices)  # every transaction retained

    for i, sku in enumerate(sku_choices):
        product_name = joined["ProductName"].iloc[i]
        if sku in config.VALID_SKUS:
            assert not pd.isna(product_name)
            assert product_name == config.SKU_MAP[sku]
        else:
            assert pd.isna(product_name)


# ===========================================================================
# Property 5
# ===========================================================================
# Feature: ggf-analytics-app, Property 5: Transactions align to targets by SKU and calendar month
@settings(max_examples=MAX_EXAMPLES)
@given(
    sku=st.sampled_from(VALID_SKU_LIST),
    month=st.sampled_from(_months_2022),
    target_value=st.floats(
        allow_nan=False, allow_infinity=False, min_value=0.0, max_value=1e9
    ),
    day=st.integers(min_value=1, max_value=28),
)
def test_property_5_target_alignment_by_sku_and_month(sku, month, target_value, day) -> None:
    """A transaction and target sharing SKU + calendar month yield a joined row
    whose SalesTarget equals that target value, regardless of day-of-month.
    Validates Requirement 1.8.
    """
    sales_transaction = pd.DataFrame(
        {
            "SKU": [sku],
            "transaction_date": [pd.Timestamp(f"{month}-{day:02d}")],
            "TotalNetSellingPrice": [1000.0],
            "TotalRawMaterialCost": [300.0],
            "TotalProductionCost": [200.0],
        }
    )
    mst_product = _make_mst_product(VALID_SKU_LIST)
    # Target period uses month-start; alignment is by calendar month only.
    sales_target = _make_sales_target([(sku, month, target_value)])

    joined, _ = _build_joined(sales_transaction, mst_product, sales_target)

    assert len(joined) == 1
    got = joined["SalesTarget"].iloc[0]
    assert not pd.isna(got)
    assert math.isclose(float(got), float(target_value), rel_tol=1e-9, abs_tol=1e-6)


# ===========================================================================
# Property 6
# ===========================================================================
# Feature: ggf-analytics-app, Property 6: Unmatched keys are retained, nulled, and recorded
@settings(max_examples=MAX_EXAMPLES)
@given(
    scenario=st.sampled_from(["bad_sku", "bad_month"]),
    valid_sku=st.sampled_from(VALID_SKU_LIST),
    foreign_sku=_foreign_skus,
    present_month=st.sampled_from(_months_2022),
)
def test_property_6_unmatched_keys_retained_and_recorded(
    scenario, valid_sku, foreign_sku, present_month
) -> None:
    """A transaction whose SKU has no product match, or whose month has no
    target, is retained with null joined fields and an UnmatchedKey recorded.
    Validates Requirement 1.9.
    """
    mst_product = _make_mst_product(VALID_SKU_LIST)

    if scenario == "bad_sku":
        # Foreign SKU: no product match. Give a target only for valid SKUs so the
        # month itself exists in the target table.
        sales_transaction = _make_sales_transaction([(foreign_sku, present_month)])
        sales_target = _make_sales_target([(valid_sku, present_month, 5000.0)])

        joined, unmatched = _build_joined(
            sales_transaction, mst_product, sales_target
        )

        assert len(joined) == 1  # retained
        assert pd.isna(joined["ProductName"].iloc[0])  # product fields null

        sku_unmatched = [u for u in unmatched if u.kind == "sku"]
        assert any(u.value == str(foreign_sku) and u.row_index == 0 for u in sku_unmatched)
        for u in unmatched:
            assert isinstance(u, UnmatchedKey)
    else:  # bad_month
        # Valid SKU + product present, but no target for the transaction's month.
        # Pick a distinct month that has no target row.
        other_months = [m for m in _months_2022 if m != present_month]
        target_month = other_months[0]
        sales_transaction = _make_sales_transaction([(valid_sku, present_month)])
        sales_target = _make_sales_target([(valid_sku, target_month, 5000.0)])

        joined, unmatched = _build_joined(
            sales_transaction, mst_product, sales_target
        )

        assert len(joined) == 1  # retained
        # Product matched (valid SKU) but target is null.
        assert not pd.isna(joined["ProductName"].iloc[0])
        assert pd.isna(joined["SalesTarget"].iloc[0])

        month_unmatched = [u for u in unmatched if u.kind == "month"]
        assert any(u.row_index == 0 for u in month_unmatched)
        assert any(u.value == present_month for u in month_unmatched)


# ===========================================================================
# Properties 7-19 (metrics.py, anomalies.py) — appended by tasks 5.2-5.4,
# 5.6-5.7, 6.2-6.5, 7.2-7.4.
#
# These target the pure analytics layer. Generators build small synthetic
# joined / transaction / raw-material frames with only the columns each
# function reads, and the expected condition is recomputed independently so
# the assertion is not a restatement of the implementation.
# ===========================================================================

from hypothesis import HealthCheck  # noqa: E402

from src import metrics  # noqa: E402
from src.anomalies import detect_reconciliation  # noqa: E402

# Suppress the data-generation health checks: several generators below build
# small frames per example and, under full-suite load, Hypothesis has
# intermittently flagged slow data generation (not a correctness issue).
_HC = [HealthCheck.too_slow, HealthCheck.data_too_large]

_finite_money = st.floats(
    allow_nan=False, allow_infinity=False, min_value=-1e7, max_value=1e7
)
_non_neg_money = st.floats(
    allow_nan=False, allow_infinity=False, min_value=0.0, max_value=1e7
)


def _joined_from_rows(rows: list[dict]) -> pd.DataFrame:
    """Build a minimal ``joined``-shaped frame from row dicts.

    Guarantees the columns the metrics functions read are present even when a
    row omits one, with sensible defaults.
    """
    defaults = {
        "SKU": None,
        "month": "2022-01",
        "Qty": 0.0,
        "TotalNetSellingPrice": 0.0,
        "TotalRawMaterialCost": 0.0,
        "TotalProductionCost": 0.0,
        "SalesTarget": float("nan"),
        "ProductName": None,
    }
    filled = [{**defaults, **r} for r in rows]
    return pd.DataFrame(filled, columns=list(defaults))


# ===========================================================================
# Property 7  (task 5.2)
# ===========================================================================
# Feature: ggf-analytics-app, Property 7: Margin percentage formula holds for totals and per SKU
@settings(max_examples=MAX_EXAMPLES, suppress_health_check=_HC)
@given(
    rows=st.lists(
        st.tuples(
            st.sampled_from(VALID_SKU_LIST),
            _non_neg_money,   # net revenue
            _non_neg_money,   # raw material cost
            _non_neg_money,   # production cost
        ),
        min_size=1,
        max_size=12,
    )
)
def test_property_7_margin_formula_totals_and_per_sku(rows) -> None:
    """The reported margin equals ``round(profit / net * 100, 1)`` when net > 0
    and is None (not applicable) when net == 0, for both the 2022 totals and the
    per-SKU P&L. Validates Requirements 3.1, 5.1.
    """
    joined = _joined_from_rows(
        [
            {
                "SKU": sku,
                "TotalNetSellingPrice": net,
                "TotalRawMaterialCost": raw,
                "TotalProductionCost": prod,
            }
            for (sku, net, raw, prod) in rows
        ]
    )

    # --- Totals ---------------------------------------------------------
    totals = metrics.totals_2022(joined)
    net_sum = sum(r[1] for r in rows)
    profit_sum = net_sum - sum(r[2] + r[3] for r in rows)
    if net_sum > 0:
        assert totals.margin_pct == round(profit_sum / net_sum * 100.0, 1)
    else:
        assert totals.margin_pct is None

    # --- Per SKU --------------------------------------------------------
    pnl = metrics.per_sku_pnl(joined)
    for sku in pnl.index:
        sku_rows = [r for r in rows if r[0] == sku]
        net = sum(r[1] for r in sku_rows)
        profit = net - sum(r[2] + r[3] for r in sku_rows)
        got = pnl.loc[sku, "margin_pct"]
        if net > 0:
            assert got == round(profit / net * 100.0, 1)
        else:
            assert got is None or (isinstance(got, float) and math.isnan(got))


# ===========================================================================
# Property 8  (task 5.3)
# ===========================================================================
# Feature: ggf-analytics-app, Property 8: Overall target-attainment formula holds
@settings(max_examples=MAX_EXAMPLES, suppress_health_check=_HC)
@given(
    rows=st.lists(
        st.tuples(
            st.sampled_from(VALID_SKU_LIST),
            _non_neg_money,                                   # net revenue
            # Target: either missing, exactly 0, or a sanely-bounded positive
            # value. Positive targets are bounded away from subnormals (>= 1)
            # so the denominator can't blow the ratio up to a value where the
            # last-ULP of round() diverges from an independent recomputation.
            st.one_of(
                st.none(),
                st.just(0.0),
                st.floats(allow_nan=False, allow_infinity=False,
                          min_value=1.0, max_value=1e7),
            ),
        ),
        min_size=1,
        max_size=12,
    )
)
def test_property_8_overall_target_attainment_formula(rows) -> None:
    """Overall attainment equals ``round(actual / target * 100, 1)`` when the
    target total > 0, and is None (not applicable) when the target total is 0.
    Null targets count as 0 in the denominator. Validates Requirement 3.2.
    """
    joined = _joined_from_rows(
        [
            {
                "SKU": sku,
                "TotalNetSellingPrice": net,
                "SalesTarget": (float("nan") if tgt is None else tgt),
            }
            for (sku, net, tgt) in rows
        ]
    )

    actual = sum(r[1] for r in rows)
    target = sum((0.0 if r[2] is None else r[2]) for r in rows)

    got = metrics.overall_target_attainment(joined)
    if target > 0:
        expected = round(actual / target * 100.0, 1)
        assert got is not None
        assert math.isclose(got, expected, rel_tol=1e-9, abs_tol=1e-6)
    else:
        assert got is None


# ===========================================================================
# Property 10  (task 5.4)
# ===========================================================================
# Feature: ggf-analytics-app, Property 10: Least-profitable selection returns all minimum-profit SKUs
@settings(max_examples=MAX_EXAMPLES, suppress_health_check=_HC)
@given(
    profits=st.lists(
        st.tuples(
            st.sampled_from(VALID_SKU_LIST),
            st.integers(min_value=-1000, max_value=1000),  # integer profit -> exact ties
        ),
        min_size=1,
        max_size=3,
        unique_by=lambda t: t[0],
    )
)
def test_property_10_least_profitable_returns_all_minima(profits) -> None:
    """The returned set is exactly the SKUs whose total profit equals the
    minimum (a single SKU normally, all tied SKUs when equal minima). Validates
    Requirements 3.4, 3.5.
    """
    pnl = pd.DataFrame(
        {
            "product_name": [config.SKU_MAP[sku] for sku, _ in profits],
            "total_profit": [float(p) for _, p in profits],
        },
        index=pd.Index([sku for sku, _ in profits], name="sku"),
    )

    min_profit = min(float(p) for _, p in profits)
    expected = {sku for sku, p in profits if float(p) == min_profit}

    result = metrics.least_profitable_sku(pnl)
    returned = {s.sku for s in result}

    assert returned == expected
    # Every returned SKU truly has the minimum profit.
    for s in result:
        assert s.total_profit == min_profit


# ===========================================================================
# Property 11  (task 6.2)
# ===========================================================================
# Feature: ggf-analytics-app, Property 11: Monthly sales matrix is fully populated and zero-fills gaps
@settings(max_examples=MAX_EXAMPLES, suppress_health_check=_HC, deadline=None)
@given(
    rows=st.lists(
        st.tuples(
            st.sampled_from(VALID_SKU_LIST),
            st.sampled_from(_months_2022),
            _non_neg_money,   # qty
            _non_neg_money,   # value
        ),
        min_size=0,
        max_size=15,
    )
)
def test_property_11_monthly_matrix_full_and_zero_filled(rows) -> None:
    """The matrix contains exactly one entry per (SKU, month) over 3 SKUs x 12
    months (36 rows), and any (SKU, month) with no recorded sales is 0 rather
    than omitted. Validates Requirement 4.1.
    """
    joined = _joined_from_rows(
        [
            {"SKU": sku, "month": month, "Qty": qty, "TotalNetSellingPrice": val}
            for (sku, month, qty, val) in rows
        ]
    )

    matrix = metrics.monthly_sales_matrix(joined)

    # Exactly 3 x 12 = 36 unique (sku, month) cells.
    assert len(matrix) == 36
    pairs = set(zip(matrix["sku"], matrix["month"]))
    assert len(pairs) == 36
    assert pairs == {(s, m) for s in VALID_SKU_LIST for m in _months_2022}

    # Cells with no recorded sales are exactly 0 (not NaN / omitted).
    recorded = {(sku, month) for (sku, month, _q, _v) in rows}
    for cell in matrix.itertuples(index=False):
        if (cell.sku, cell.month) not in recorded:
            assert cell.qty == 0.0
            assert cell.value == 0.0


# ===========================================================================
# Property 12  (task 6.3)
# ===========================================================================
# Feature: ggf-analytics-app, Property 12: Per-SKU/month attainment formula holds with zero-target handling
@settings(max_examples=MAX_EXAMPLES, suppress_health_check=_HC)
@given(
    rows=st.lists(
        st.tuples(
            st.sampled_from(VALID_SKU_LIST),
            st.sampled_from(_months_2022),
            _non_neg_money,                          # value
            st.one_of(st.just(0.0), _non_neg_money), # target (zero exercised)
        ),
        min_size=1,
        max_size=12,
        unique_by=lambda t: (t[0], t[1]),
    )
)
def test_property_12_per_sku_month_attainment_formula(rows) -> None:
    """Per-(SKU, month) attainment equals ``value / target * 100`` when the
    target is non-zero, and is not applicable (NaN) when the target is zero or
    missing. Validates Requirement 4.3.
    """
    joined = _joined_from_rows(
        [
            {
                "SKU": sku,
                "month": month,
                "TotalNetSellingPrice": val,
                "SalesTarget": tgt,
            }
            for (sku, month, val, tgt) in rows
        ]
    )

    result = metrics.target_attainment_matrix(joined)
    by_cell = {(r.sku, r.month): r for r in result.itertuples(index=False)}

    for (sku, month, val, tgt) in rows:
        cell = by_cell[(sku, month)]
        if tgt > 0:
            expected = val / tgt * 100.0
            assert not math.isnan(cell.attainment_pct)
            assert math.isclose(cell.attainment_pct, expected, rel_tol=1e-9, abs_tol=1e-6)
        else:
            assert math.isnan(cell.attainment_pct)


# ===========================================================================
# Property 13  (task 6.4)
# ===========================================================================
# Feature: ggf-analytics-app, Property 13: Sales anomaly detection matches its defining condition
@settings(max_examples=MAX_EXAMPLES, suppress_health_check=_HC, deadline=None)
@given(
    # One value + target per (SKU, month). Bounded, small integers keep the
    # generator light and produce meaningful spread / attainment bands.
    data=st.data()
)
def test_property_13_sales_anomaly_matches_condition(data) -> None:
    """A (SKU, month) is flagged iff its value deviates from that SKU's 12-month
    mean by more than 2 std OR its attainment is < 50% or > 150%. The expected
    set is recomputed independently from the same matrix. Validates
    Requirement 4.4.
    """
    # Build a full 3x12 grid of (value, target) so means/std are well defined.
    rows = []
    for sku in VALID_SKU_LIST:
        for month in _months_2022:
            val = data.draw(st.floats(
                allow_nan=False, allow_infinity=False, min_value=0.0, max_value=1000.0
            ))
            tgt = data.draw(st.one_of(
                st.just(0.0),
                st.floats(allow_nan=False, allow_infinity=False,
                          min_value=1.0, max_value=1000.0),
            ))
            rows.append({
                "SKU": sku, "month": month,
                "TotalNetSellingPrice": val, "SalesTarget": tgt,
            })
    joined = _joined_from_rows(rows)

    std_thr = config.SALES_ANOMALY_STD_THRESHOLD
    low = config.ATTAINMENT_LOW_PCT
    high = config.ATTAINMENT_HIGH_PCT

    matrix = metrics.target_attainment_matrix(joined)

    # Independent expectation of the defining condition.
    expected = set()
    for sku in VALID_SKU_LIST:
        sub = matrix[matrix["sku"] == sku]
        values = sub["value"].to_numpy()
        mean = float(values.mean())
        std = float(values.std(ddof=0))
        for r in sub.itertuples(index=False):
            flagged = False
            if std > 0 and abs(r.value - mean) > std_thr * std:
                flagged = True
            att = r.attainment_pct
            if not (isinstance(att, float) and math.isnan(att)):
                if att < low or att > high:
                    flagged = True
            if flagged:
                expected.add((sku, r.month))

    got = {(a.sku, a.month) for a in metrics.sales_anomalies(joined)}
    assert got == expected


# ===========================================================================
# Property 14  (task 5.6) — depends on metrics.cost_driver_breakdown (task 5.5)
# ===========================================================================
# Feature: ggf-analytics-app, Property 14: Cost-driver breakdown parts reconstruct the cost base
@settings(max_examples=MAX_EXAMPLES, suppress_health_check=_HC)
@given(
    rows=st.lists(
        st.tuples(
            st.sampled_from(VALID_SKU_LIST),
            st.floats(allow_nan=False, allow_infinity=False, min_value=1.0, max_value=1e6),  # raw
            st.floats(allow_nan=False, allow_infinity=False, min_value=1.0, max_value=1e6),  # prod
        ),
        min_size=1,
        max_size=12,
    )
)
def test_property_14_cost_driver_reconstructs_base(rows) -> None:
    """Raw-material + production components in IDR sum to the recomputed total
    cost base per SKU, and their percentages sum to 100 within rounding.
    Validates Requirement 5.3.

    Guarded: skips if ``metrics.cost_driver_breakdown`` (task 5.5) is absent.
    """
    if not hasattr(metrics, "cost_driver_breakdown"):
        import pytest
        pytest.skip("TODO: metrics.cost_driver_breakdown not implemented yet (task 5.5)")

    joined = _joined_from_rows(
        [
            {
                "SKU": sku,
                "TotalNetSellingPrice": raw + prod + 100.0,
                "TotalRawMaterialCost": raw,
                "TotalProductionCost": prod,
            }
            for (sku, raw, prod) in rows
        ]
    )

    breakdown = metrics.cost_driver_breakdown(joined).set_index("sku")

    # Expected per-SKU raw / production totals computed independently.
    expected_raw: dict[str, float] = {}
    expected_prod: dict[str, float] = {}
    for (sku, raw, prod) in rows:
        expected_raw[sku] = expected_raw.get(sku, 0.0) + raw
        expected_prod[sku] = expected_prod.get(sku, 0.0) + prod

    for sku in expected_raw:
        row = breakdown.loc[sku]
        raw_val = float(row["raw_material_cost"])
        prod_val = float(row["production_cost"])
        base = expected_raw[sku] + expected_prod[sku]

        # IDR parts reconstruct the recomputed cost base per SKU.
        assert math.isclose(raw_val + prod_val, float(row["total_cost_base"]),
                            rel_tol=1e-6, abs_tol=1e-3)
        assert math.isclose(raw_val + prod_val, base, rel_tol=1e-6, abs_tol=1e-3)
        assert math.isclose(raw_val, expected_raw[sku], rel_tol=1e-6, abs_tol=1e-3)
        assert math.isclose(prod_val, expected_prod[sku], rel_tol=1e-6, abs_tol=1e-3)

        # Percentages sum to 100 within rounding tolerance (base > 0 here).
        assert math.isclose(
            float(row["raw_material_pct"]) + float(row["production_pct"]),
            100.0, abs_tol=0.2,
        )


# ===========================================================================
# Property 15  (task 5.7) — depends on an improvement-proposal generator (5.5)
# ===========================================================================
# Feature: ggf-analytics-app, Property 15: At least one improvement proposal references a valid SKU and driver
@settings(max_examples=MAX_EXAMPLES, suppress_health_check=_HC)
@given(
    rows=st.lists(
        st.tuples(
            st.sampled_from(VALID_SKU_LIST),
            st.floats(allow_nan=False, allow_infinity=False, min_value=1.0, max_value=1e6),  # raw
            st.floats(allow_nan=False, allow_infinity=False, min_value=1.0, max_value=1e6),  # prod
        ),
        min_size=1,
        max_size=12,
    )
)
def test_property_15_improvement_proposal_references_sku_and_driver(rows) -> None:
    """The analysis produces >=1 improvement proposal, and each proposal
    references an SKU present in the breakdown and a driver in
    {raw-material, production}. Validates Requirement 5.4.

    Guarded: skips until the proposal generator (task 5.5) exists.
    """
    proposal_fn = getattr(metrics, "improvement_proposals", None)
    if proposal_fn is None or not hasattr(metrics, "cost_driver_breakdown"):
        import pytest
        pytest.skip("TODO: metrics.improvement_proposals not implemented yet (task 5.5)")

    joined = _joined_from_rows(
        [
            {
                "SKU": sku,
                "TotalNetSellingPrice": raw + prod + 100.0,
                "TotalRawMaterialCost": raw,
                "TotalProductionCost": prod,
            }
            for (sku, raw, prod) in rows
        ]
    )

    # SKUs actually present in the breakdown (what a proposal may reference).
    breakdown_skus = set(metrics.cost_driver_breakdown(joined)["sku"].tolist())
    valid_drivers = set(getattr(metrics, "COST_DRIVERS", ("raw_material", "production")))

    proposals = proposal_fn(joined)
    assert len(proposals) >= 1

    for p in proposals:
        # References a SKU present in the breakdown.
        assert p.sku in breakdown_skus, f"proposal references unknown SKU: {p!r}"
        # References one of the two cost drivers.
        assert p.cost_driver in valid_drivers, f"proposal has invalid driver: {p!r}"


# ===========================================================================
# Property 19  (task 6.5)
# ===========================================================================
# Feature: ggf-analytics-app, Property 19: Seasonality comparison equals the grouped means
@settings(max_examples=MAX_EXAMPLES, suppress_health_check=_HC)
@given(
    rows=st.lists(
        st.tuples(
            st.sampled_from(VALID_SKU_LIST),
            st.sampled_from([0, 1]),  # Is Peak Season
            st.floats(allow_nan=False, allow_infinity=False, min_value=0.0, max_value=1e5),  # Close Price
        ),
        min_size=1,
        max_size=20,
    )
)
def test_property_19_seasonality_equals_grouped_means(rows) -> None:
    """Per SKU, the peak / non-peak average Close Prices equal the arithmetic
    means of the rows partitioned by Is Peak Season (1 vs 0). Validates
    Requirement 7.3.
    """
    raw_mat_trend = pd.DataFrame(
        {
            "SKU": [sku for (sku, _p, _c) in rows],
            "Is Peak Season": [peak for (_s, peak, _c) in rows],
            "Close Price": [close for (_s, _p, close) in rows],
        }
    )

    result = metrics.seasonality_price_compare(raw_mat_trend)
    by_sku = {r.sku: r for r in result.itertuples(index=False)}

    skus = set(sku for (sku, _p, _c) in rows)
    for sku in skus:
        peak_vals = [c for (s, p, c) in rows if s == sku and p == 1]
        non_peak_vals = [c for (s, p, c) in rows if s == sku and p == 0]
        r = by_sku[sku]

        if peak_vals:
            assert math.isclose(r.avg_close_peak, sum(peak_vals) / len(peak_vals),
                                rel_tol=1e-9, abs_tol=1e-6)
        else:
            assert math.isnan(r.avg_close_peak)

        if non_peak_vals:
            assert math.isclose(r.avg_close_non_peak, sum(non_peak_vals) / len(non_peak_vals),
                                rel_tol=1e-9, abs_tol=1e-6)
        else:
            assert math.isnan(r.avg_close_non_peak)


# ===========================================================================
# Reconciliation generators (Properties 16-18)
# ===========================================================================

def _recon_frame_from_rows(rows: list[dict]) -> pd.DataFrame:
    """Build a normalized SalesTransaction-shaped frame for reconciliation.

    Each row dict may set any of the required fields; missing ones default to a
    clean, reconcilable value so a test can corrupt exactly one field.
    """
    defaults = {
        "SKU": "200100001",
        "profit_source": 0.0,
        "TotalNetSellingPrice": 0.0,
        "TotalCost": 0.0,
        "TotalRawMaterialCost": 0.0,
        "TotalProductionCost": 0.0,
    }
    filled = [{**defaults, **r} for r in rows]
    return pd.DataFrame(filled, columns=list(defaults))


# A value guaranteed non-numeric so the row is unprocessable.
_non_numeric_token = st.sampled_from(["n/a", "abc", "--", "?", ""])


# ===========================================================================
# Property 16  (task 7.2)
# ===========================================================================
# Feature: ggf-analytics-app, Property 16: Reconciliation detector flags rows per its predicate and reports them consistently
@settings(max_examples=MAX_EXAMPLES, suppress_health_check=_HC)
@given(
    rows=st.lists(
        st.tuples(
            st.floats(allow_nan=False, allow_infinity=False, min_value=0.0, max_value=1e6),   # net
            st.floats(allow_nan=False, allow_infinity=False, min_value=0.0, max_value=1e6),   # rawmat
            st.floats(allow_nan=False, allow_infinity=False, min_value=0.0, max_value=1e6),   # prod
            st.floats(allow_nan=False, allow_infinity=False, min_value=-1e6, max_value=1e6),  # totalcost offset
            st.floats(allow_nan=False, allow_infinity=False, min_value=-1e6, max_value=1e6),  # pnl offset
        ),
        min_size=1,
        max_size=12,
    )
)
def test_property_16_reconciliation_predicate_and_reporting(rows) -> None:
    """With all required numeric fields present, a row is flagged iff
    ``|profit_source - (net - TotalCost)| > 0.01`` or
    ``|TotalCost - (rawmat + prod)| > 0.01``; the count equals the number of
    flagged rows and each signed diff has the correct sign. Validates
    Requirements 6.1, 6.2, 6.3.
    """
    tol = config.RECONCILIATION_TOLERANCE

    frame_rows = []
    expected_flags = []
    expected_pnl_diffs = []
    expected_cost_diffs = []
    for (net, raw, prod, cost_off, pnl_off) in rows:
        total_cost = raw + prod + cost_off
        profit_source = (net - total_cost) + pnl_off
        frame_rows.append({
            "profit_source": profit_source,
            "TotalNetSellingPrice": net,
            "TotalCost": total_cost,
            "TotalRawMaterialCost": raw,
            "TotalProductionCost": prod,
        })
        diff_pnl = profit_source - (net - total_cost)
        diff_cost = total_cost - (raw + prod)
        flagged = abs(diff_pnl) > tol or abs(diff_cost) > tol
        expected_flags.append(flagged)
        expected_pnl_diffs.append(diff_pnl)
        expected_cost_diffs.append(diff_cost)

    frame = _recon_frame_from_rows(frame_rows)
    result = detect_reconciliation(frame)

    # Count matches number flagged.
    assert result.discrepancy_count == sum(expected_flags)
    assert len(result.discrepancies) == sum(expected_flags)

    flagged_idx = set(result.discrepancies["row_index"].tolist())
    expected_idx = {i for i, f in enumerate(expected_flags) if f}
    assert flagged_idx == expected_idx

    # Signed diffs carry the correct sign/value for each flagged row.
    for rec in result.discrepancies.itertuples(index=False):
        i = int(rec.row_index)
        assert math.isclose(rec.signed_diff_pnl, expected_pnl_diffs[i], rel_tol=1e-6, abs_tol=1e-3)
        assert math.isclose(rec.signed_diff_cost, expected_cost_diffs[i], rel_tol=1e-6, abs_tol=1e-3)


# ===========================================================================
# Property 17  (task 7.3)
# ===========================================================================
# Feature: ggf-analytics-app, Property 17: Loss rows are counted per SKU on the recomputed basis
@settings(max_examples=MAX_EXAMPLES, suppress_health_check=_HC)
@given(
    rows=st.lists(
        st.tuples(
            st.sampled_from(VALID_SKU_LIST),
            st.floats(allow_nan=False, allow_infinity=False, min_value=0.0, max_value=1e6),  # net
            st.floats(allow_nan=False, allow_infinity=False, min_value=0.0, max_value=1e6),  # rawmat
            st.floats(allow_nan=False, allow_infinity=False, min_value=0.0, max_value=1e6),  # prod
        ),
        min_size=1,
        max_size=15,
    )
)
def test_property_17_loss_rows_counted_per_sku(rows) -> None:
    """The reported loss-row count per SKU equals the number of that SKU's rows
    whose recomputed profit (net - (rawmat + prod)) is strictly < 0. Validates
    Requirement 6.5.
    """
    frame_rows = []
    expected: dict[str, int] = {}
    for (sku, net, raw, prod) in rows:
        # Keep TotalCost consistent so these rows are otherwise clean.
        frame_rows.append({
            "SKU": sku,
            "profit_source": net - (raw + prod),
            "TotalNetSellingPrice": net,
            "TotalCost": raw + prod,
            "TotalRawMaterialCost": raw,
            "TotalProductionCost": prod,
        })
        if net - (raw + prod) < 0:
            expected[sku] = expected.get(sku, 0) + 1

    frame = _recon_frame_from_rows(frame_rows)
    result = detect_reconciliation(frame)

    # Only SKUs with >=1 loss row are reported; counts match exactly.
    assert {k: v for k, v in result.loss_rows_per_sku.items() if v > 0} == expected


# ===========================================================================
# Property 18  (task 7.4)
# ===========================================================================
# Feature: ggf-analytics-app, Property 18: Rows missing required fields are excluded, flagged, and attributed
@settings(max_examples=MAX_EXAMPLES, suppress_health_check=_HC)
@given(data=st.data())
def test_property_18_unprocessable_rows_excluded_and_attributed(data) -> None:
    """Each row with a missing/non-numeric value in a required reconciliation
    field is excluded from the checks, counted as unprocessable, and attributed
    to the first offending field. Validates Requirement 6.6.
    """
    required = ["profit_source", "TotalNetSellingPrice", "TotalCost",
                "TotalRawMaterialCost", "TotalProductionCost"]

    n_rows = data.draw(st.integers(min_value=1, max_value=10))
    frame_rows = []
    # Track which rows are corrupted and the expected offending field
    # (the FIRST required field, in order, that is bad).
    expected_offender: dict[int, str] = {}
    for i in range(n_rows):
        corrupt = data.draw(st.booleans())
        row = {
            "SKU": "200100001",
            "profit_source": 0.0,
            "TotalNetSellingPrice": 0.0,
            "TotalCost": 0.0,
            "TotalRawMaterialCost": 0.0,
            "TotalProductionCost": 0.0,
        }
        if corrupt:
            # Corrupt a nonempty subset of required fields.
            bad_fields = data.draw(
                st.lists(st.sampled_from(required), min_size=1, max_size=len(required),
                         unique=True)
            )
            for f in bad_fields:
                # Either NaN (missing) or a non-numeric token.
                if data.draw(st.booleans()):
                    row[f] = float("nan")
                else:
                    row[f] = data.draw(_non_numeric_token)
            first_bad = next(f for f in required if f in bad_fields)
            expected_offender[i] = first_bad
        frame_rows.append(row)

    frame = _recon_frame_from_rows(frame_rows)
    result = detect_reconciliation(frame)

    # Unprocessable count and attributed field match the expectation.
    assert result.unprocessable_count == len(expected_offender)
    assert result.unprocessable_fields == expected_offender

    # Unprocessable rows never appear as discrepancies.
    flagged_idx = set(result.discrepancies["row_index"].tolist())
    assert flagged_idx.isdisjoint(set(expected_offender))


# ===========================================================================
# Forecasting properties (Properties 20-22)  — tasks 9.2, 9.3, 9.4
#
# Target: src.forecasting.forecast_sku / ForecastResult.
#
# Holt's linear exponential smoothing is fit on the 12-point series, which is a
# little slow, so these tests use deadline=None and suppress the "too slow" and
# "data too large" health checks to avoid hypothesis flakiness. Example counts
# are kept at 100 with lightweight generators (bounded value ranges).
# ===========================================================================

from hypothesis import HealthCheck  # noqa: E402

from src.forecasting import ForecastResult, forecast_sku  # noqa: E402

# Settings for the Holt-fitting forecast properties: fitting on 12 points is a
# bit slow, so deadline=None + suppressed health checks prevent flakes while
# still running 100 examples.
_FORECAST_SETTINGS = settings(
    max_examples=100,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow, HealthCheck.data_too_large],
)

#: Twelve chronological month labels Jan..Dec 2022 (the only history per SKU).
_MONTHS_2022 = [f"2022-{m:02d}" for m in range(1, 13)]

#: Lightweight monthly sales values: bounded positive quantities so the Holt
#: fit is well-behaved and fast.
_sales_value = st.floats(
    min_value=100.0, max_value=10_000.0, allow_nan=False, allow_infinity=False
)


def _make_monthly_series(values: list[float]) -> pd.Series:
    """Build a monthly sales Series indexed by 2022-01..(n) from ``values``.

    Uses the first ``len(values)`` labels of Jan..Dec 2022 so a full-length
    (12) list yields a fittable SKU and a shorter list yields an excluded one.
    """
    n = len(values)
    idx = pd.Index(_MONTHS_2022[:n], name="month")
    return pd.Series([float(v) for v in values], index=idx, dtype=float)


# ===========================================================================
# Property 20  (task 9.2)
# ===========================================================================
# Feature: ggf-analytics-app, Property 20: SKUs with fewer than twelve observations are excluded
@_FORECAST_SETTINGS
@given(data=st.data())
def test_property_20_insufficient_observations_excluded(data) -> None:
    """For any monthly sales series with fewer than 12 observations,
    ``forecast_sku`` returns ``excluded=True`` with empty forecast series and a
    non-empty assumption recording the insufficient-data reason; a series with
    exactly 12 observations is fitted (``excluded=False``) with 12-length
    forecasts. Validates Requirements 8.1, 8.2.
    """
    per_unit_margin = data.draw(
        st.floats(min_value=1.0, max_value=5_000.0,
                  allow_nan=False, allow_infinity=False)
    )

    # --- Excluded path: 0..11 observations -------------------------------
    n_short = data.draw(st.integers(min_value=0, max_value=11))
    short_values = data.draw(
        st.lists(_sales_value, min_size=n_short, max_size=n_short)
    )
    short_series = _make_monthly_series(short_values)

    excluded_result = forecast_sku(
        short_series, per_unit_margin=per_unit_margin, sku="short"
    )

    assert excluded_result.excluded is True
    assert len(excluded_result.forecast_sales) == 0
    assert len(excluded_result.forecast_profit) == 0
    # A non-empty assumption stating insufficient data is recorded.
    assert len(excluded_result.assumptions) >= 1
    assert any(
        "insufficient" in a.lower() or "excluded" in a.lower()
        for a in excluded_result.assumptions
    )

    # --- Fitted path: exactly 12 observations ----------------------------
    full_values = data.draw(st.lists(_sales_value, min_size=12, max_size=12))
    full_series = _make_monthly_series(full_values)

    fitted_result = forecast_sku(
        full_series, per_unit_margin=per_unit_margin, sku="full"
    )

    assert fitted_result.excluded is False
    assert len(fitted_result.forecast_sales) == config.FORECAST_HORIZON
    assert len(fitted_result.forecast_profit) == config.FORECAST_HORIZON


# ===========================================================================
# Property 21  (task 9.3)
# ===========================================================================
# Feature: ggf-analytics-app, Property 21: Forecast horizon is twelve months and profit is derived from sales
@_FORECAST_SETTINGS
@given(data=st.data())
def test_property_21_horizon_and_profit_derivation(data) -> None:
    """For any fitted SKU (12 observations, arbitrary positive per-unit
    margin), the sales and profit forecasts each have length equal to the
    horizon (default 12, or a custom value when passed), and each profit point
    equals its sales point times the per-unit margin (within float tolerance).
    Validates Requirements 8.3, 8.4.
    """
    per_unit_margin = data.draw(
        st.floats(min_value=0.01, max_value=10_000.0,
                  allow_nan=False, allow_infinity=False)
    )
    full_values = data.draw(st.lists(_sales_value, min_size=12, max_size=12))
    series = _make_monthly_series(full_values)

    # --- Default horizon (12) --------------------------------------------
    result = forecast_sku(series, per_unit_margin=per_unit_margin, sku="s")
    assert result.excluded is False
    assert len(result.forecast_sales) == config.FORECAST_HORIZON
    assert len(result.forecast_profit) == config.FORECAST_HORIZON

    # Profit is derived from sales: profit[i] == sales[i] * per_unit_margin.
    sales_vals = result.forecast_sales.to_numpy(dtype=float)
    profit_vals = result.forecast_profit.to_numpy(dtype=float)
    expected_profit = sales_vals * per_unit_margin
    assert np.allclose(profit_vals, expected_profit, rtol=1e-9, atol=1e-6)
    # Profit forecast shares the sales forecast index (same months).
    assert list(result.forecast_profit.index) == list(result.forecast_sales.index)

    # --- Custom horizon (6) ----------------------------------------------
    custom_horizon = data.draw(st.integers(min_value=1, max_value=18))
    custom = forecast_sku(
        series, per_unit_margin=per_unit_margin, horizon=custom_horizon, sku="s"
    )
    assert len(custom.forecast_sales) == custom_horizon
    assert len(custom.forecast_profit) == custom_horizon
    c_sales = custom.forecast_sales.to_numpy(dtype=float)
    c_profit = custom.forecast_profit.to_numpy(dtype=float)
    assert np.allclose(c_profit, c_sales * per_unit_margin, rtol=1e-9, atol=1e-6)


# ===========================================================================
# Property 22  (task 9.4)
# ===========================================================================
# Feature: ggf-analytics-app, Property 22: Fitted forecasts always carry stated assumptions
@_FORECAST_SETTINGS
@given(data=st.data())
def test_property_22_fitted_forecast_carries_assumptions(data) -> None:
    """For any fitted SKU, the result carries a non-empty list of assumptions
    and at least one assumption states the twelve-observation basis (mentions
    "12" or "twelve"). Validates Requirement 8.6.
    """
    per_unit_margin = data.draw(
        st.floats(min_value=0.01, max_value=10_000.0,
                  allow_nan=False, allow_infinity=False)
    )
    full_values = data.draw(st.lists(_sales_value, min_size=12, max_size=12))
    series = _make_monthly_series(full_values)

    result = forecast_sku(series, per_unit_margin=per_unit_margin, sku="s")

    assert result.excluded is False
    assert isinstance(result.assumptions, list)
    assert len(result.assumptions) >= 1
    # At least one assumption states the 12-observation basis.
    assert any(
        ("12" in a) or ("twelve" in a.lower())
        for a in result.assumptions
    )

# ===========================================================================
# Properties 9 and 23  (tasks 9.6, 9.7)
#
# These target the expansion verdict / forecast-to-verdict relationship in
# ``src.metrics`` (task 9.5): ``expansion_verdict`` and ``forecast_relationship``.
#
# Both functions accept forecasts *structurally* (any object exposing
# ``.forecast_profit`` -- a pandas Series -- and ``.excluded`` -- a bool). We
# therefore use a lightweight local stub instead of importing
# ``src.forecasting.ForecastResult`` (no statsmodels dependency, fast to build).
# ===========================================================================

from dataclasses import dataclass as _dataclass  # noqa: E402
from typing import Optional as _Optional  # noqa: E402

from src.metrics import (  # noqa: E402
    Totals,
    Verdict,
    VERDICT_RECOMMENDATIONS,
    FORECAST_RELATIONSHIPS,
    expansion_verdict,
    forecast_relationship,
)


@_dataclass
class _FakeForecast:
    """A lightweight stand-in for ``src.forecasting.ForecastResult``.

    Exposes only the two attributes ``expansion_verdict`` / ``forecast_relationship``
    consume structurally: ``forecast_profit`` (a pandas Series of monthly profit)
    and ``excluded`` (a bool). No statsmodels, no fitting.
    """

    forecast_profit: pd.Series
    excluded: bool


def _count_sentences(text: str) -> int:
    """Count sentences robustly: split on '.', '!', '?' and count non-empty
    trimmed segments. Mirrors the "1 to 3 sentences" bound in Req 3.3.
    """
    segments = re.split(r"[.!?]", text)
    return sum(1 for seg in segments if seg.strip())


# --- Strategies -------------------------------------------------------------

# Month labels for a forecast profit series (a few YYYY-MM labels, chronological).
_FORECAST_MONTHS = tuple(f"2023-{m:02d}" for m in range(1, 13))

# Profit values that can be rising / falling / negative / flat / large.
_forecast_profit_value = st.floats(
    min_value=-5_000_000.0, max_value=5_000_000.0,
    allow_nan=False, allow_infinity=False,
)


@st.composite
def _forecast_profit_series(draw) -> pd.Series:
    """A forecast_profit Series over a few chronological YYYY-MM labels.

    Length 0 (empty), 1 (single point), or up to 12; values arbitrary
    (rising/falling/negative/flat all reachable).
    """
    n = draw(st.integers(min_value=0, max_value=12))
    values = draw(
        st.lists(_forecast_profit_value, min_size=n, max_size=n)
    )
    index = list(_FORECAST_MONTHS[:n])
    return pd.Series(values, index=index, dtype="float64")


@st.composite
def _fake_forecast(draw) -> _FakeForecast:
    """A single fake forecast: arbitrary profit series + excluded flag."""
    return _FakeForecast(
        forecast_profit=draw(_forecast_profit_series()),
        excluded=draw(st.booleans()),
    )


# A forecasts input: list of stubs (including empty list). All-excluded and
# single-point cases are reachable because each stub is drawn independently.
_forecasts_strategy = st.lists(_fake_forecast(), min_size=0, max_size=5)


# An arbitrary Totals: net_revenue / total_cost / total_profit each arbitrary
# (incl. negatives), margin_pct arbitrary including None.
_money = st.floats(
    min_value=-1e11, max_value=1e11, allow_nan=False, allow_infinity=False
)


@st.composite
def _arbitrary_totals(draw) -> Totals:
    """A Totals with independent, arbitrary fields incl. margin_pct None/neg.

    The verdict must hold for *any* Totals, so we deliberately do NOT enforce
    the ``total_profit == net_revenue - total_cost`` invariant here -- the
    domain/rationale-bound guarantee must survive inconsistent inputs too.
    """
    margin = draw(
        st.one_of(
            st.none(),
            st.floats(min_value=-100.0, max_value=100.0,
                      allow_nan=False, allow_infinity=False),
        )
    )
    return Totals(
        net_revenue=draw(_money),
        total_cost=draw(_money),
        total_profit=draw(_money),
        margin_pct=margin,
    )


@st.composite
def _per_sku_pnl_frame(draw) -> pd.DataFrame:
    """A per_sku_pnl-shaped frame: index sku; columns product_name,
    net_revenue, total_profit, margin_pct.

    Varied profits (incl. negative and small), small margins, and margin
    None/NaN are all reachable. May be empty.
    """
    n = draw(st.integers(min_value=0, max_value=3))
    skus = draw(
        st.lists(st.sampled_from(VALID_SKU_LIST), min_size=n, max_size=n, unique=True)
    )
    records = []
    for _ in skus:
        total_profit = draw(
            st.floats(min_value=-2e7, max_value=2e7,
                      allow_nan=False, allow_infinity=False)
        )
        margin = draw(
            st.one_of(
                st.none(),
                st.just(float("nan")),
                st.floats(min_value=-20.0, max_value=60.0,
                          allow_nan=False, allow_infinity=False),
            )
        )
        net_revenue = draw(
            st.floats(min_value=0.0, max_value=2e8,
                      allow_nan=False, allow_infinity=False)
        )
        records.append(
            {
                "product_name": draw(st.one_of(st.none(), st.text(max_size=10))),
                "net_revenue": net_revenue,
                "total_profit": total_profit,
                "margin_pct": margin,
            }
        )
    columns = ["product_name", "net_revenue", "total_profit", "margin_pct"]
    frame = pd.DataFrame(records, columns=columns)
    frame.index = pd.Index(skus, name="sku")
    return frame


_VERDICT_HC = [HealthCheck.too_slow, HealthCheck.data_too_large]


# ===========================================================================
# Property 9  (task 9.6)
# ===========================================================================
# Feature: ggf-analytics-app, Property 9: Expansion verdict lies in a fixed domain with a bounded rationale
@settings(max_examples=200, deadline=None, suppress_health_check=_VERDICT_HC)
@given(
    totals=_arbitrary_totals(),
    per_sku_pnl_df=_per_sku_pnl_frame(),
    forecasts=_forecasts_strategy,
)
def test_property_9_verdict_domain_and_rationale_bound(
    totals, per_sku_pnl_df, forecasts
) -> None:
    """For ANY combination of totals, per-SKU P&L, and forecast inputs,
    ``expansion_verdict(...).recommendation`` is exactly one of the three
    VERDICT_RECOMMENDATIONS, and its rationale contains between 1 and 3
    sentences. Validates Requirement 3.3.
    """
    verdict = expansion_verdict(totals, per_sku_pnl_df, forecasts)

    assert isinstance(verdict, Verdict)
    # Domain: exactly one of the three fixed recommendations.
    assert verdict.recommendation in VERDICT_RECOMMENDATIONS

    # Rationale: a non-empty string of 1 to 3 sentences.
    assert isinstance(verdict.rationale, str)
    n_sentences = _count_sentences(verdict.rationale)
    assert 1 <= n_sentences <= 3, (
        f"rationale had {n_sentences} sentence(s), expected 1-3: "
        f"{verdict.rationale!r}"
    )

    # The verdict's own forecast_relationship must also lie in its fixed domain.
    assert verdict.forecast_relationship in FORECAST_RELATIONSHIPS


# ===========================================================================
# Property 23  (task 9.7)
# ===========================================================================
# Feature: ggf-analytics-app, Property 23: Forecast-to-verdict relationship lies in a fixed domain
@settings(max_examples=200, deadline=None, suppress_health_check=_VERDICT_HC)
@given(
    forecasts=_forecasts_strategy,
    totals=_arbitrary_totals(),
    per_sku_pnl_df=_per_sku_pnl_frame(),
)
def test_property_23_forecast_relationship_domain(
    forecasts, totals, per_sku_pnl_df
) -> None:
    """For ANY forecasts input (list of stubs incl. all-excluded, empty list,
    single-point, rising, falling, negative), ``forecast_relationship(...)`` is
    exactly one of the three FORECAST_RELATIONSHIPS; and
    ``expansion_verdict(...).forecast_relationship`` is also in that set.
    Validates Requirement 8.7.
    """
    relationship = forecast_relationship(forecasts)
    assert relationship in FORECAST_RELATIONSHIPS

    verdict = expansion_verdict(totals, per_sku_pnl_df, forecasts)
    assert verdict.forecast_relationship in FORECAST_RELATIONSHIPS
    # The verdict reports the same relationship the classifier produced.
    assert verdict.forecast_relationship == relationship
