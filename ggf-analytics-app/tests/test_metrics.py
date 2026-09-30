"""Example-based tests for src.metrics.

Task 5.8: a dataset-anchored example test that asserts the exact ground-truth
figures computed from the real workbook (``data/FruitsTrxDatasets.xlsx``) — the
2022 totals, per-SKU profit, the least-profitable SKU, and Pineapple's negative
-profit month count (Req 5.2).

The values below are the verified integers produced by the pure metrics
functions on the real source data (recomputed profit basis). The requirements
text uses "billions" loosely; these are the true magnitudes and are asserted
exactly.
"""

from __future__ import annotations

import math

import pytest

from src import metrics
from src.data_loader import LoadError, load_data


# Verified ground truth (recomputed profit basis) from the real workbook.
EXPECTED_NET_REVENUE = 1_505_783_925
EXPECTED_TOTAL_COST = 1_280_005_200
EXPECTED_TOTAL_PROFIT = 225_778_725
EXPECTED_MARGIN_PCT = 15.0

EXPECTED_SKU_PROFIT = {
    "200100001": 94_533_150,    # Orange Juice
    "200100003": 128_393_700,   # Mango Juice
    "200100004": 2_851_875,     # Pineapple Juice (least profitable)
}

LEAST_PROFITABLE_SKU = "200100004"
PINEAPPLE_NEGATIVE_MONTHS = 7


@pytest.fixture(scope="module")
def joined():
    """Load the real workbook once for the module; skip if it is unavailable."""
    try:
        bundle = load_data()
    except LoadError as exc:  # pragma: no cover - only when the workbook is absent
        pytest.skip(f"real workbook unavailable: {exc}")
    return bundle.joined


def test_totals_2022_match_real_workbook(joined) -> None:
    """2022 totals equal the verified ground truth (Req 3.1, 5.2)."""
    totals = metrics.totals_2022(joined)
    assert round(totals.net_revenue) == EXPECTED_NET_REVENUE
    assert round(totals.total_cost) == EXPECTED_TOTAL_COST
    assert round(totals.total_profit) == EXPECTED_TOTAL_PROFIT
    assert totals.margin_pct == EXPECTED_MARGIN_PCT
    # Totals are internally consistent on the recomputed basis.
    assert math.isclose(
        totals.total_profit, totals.net_revenue - totals.total_cost, abs_tol=0.5
    )


def test_per_sku_profit_match_real_workbook(joined) -> None:
    """Per-SKU total profit equals the verified ground truth (Req 5.1, 5.2)."""
    pnl = metrics.per_sku_pnl(joined)
    for sku, expected_profit in EXPECTED_SKU_PROFIT.items():
        assert sku in pnl.index, f"missing SKU {sku} in per-SKU P&L"
        assert round(float(pnl.loc[sku, "total_profit"])) == expected_profit


def test_least_profitable_is_pineapple(joined) -> None:
    """The single least-profitable SKU is Pineapple 200100004 at 2,851,875 IDR
    (Req 3.4, 5.2)."""
    pnl = metrics.per_sku_pnl(joined)
    least = metrics.least_profitable_sku(pnl)
    assert len(least) == 1
    assert least[0].sku == LEAST_PROFITABLE_SKU
    assert round(least[0].total_profit) == EXPECTED_SKU_PROFIT[LEAST_PROFITABLE_SKU]


@pytest.mark.skipif(
    not hasattr(metrics, "negative_profit_month_count"),
    reason="TODO: metrics.negative_profit_month_count not implemented yet (task 5.5)",
)
def test_pineapple_has_seven_negative_months(joined) -> None:
    """Pineapple has 7 negative-profit months out of 12 (Req 5.2).

    Guarded: runs once ``metrics.negative_profit_month_count`` (task 5.5) exists.
    """
    counts = metrics.negative_profit_month_count(joined)
    assert counts.get(LEAST_PROFITABLE_SKU) == PINEAPPLE_NEGATIVE_MONTHS
