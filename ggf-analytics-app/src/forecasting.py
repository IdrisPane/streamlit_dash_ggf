"""Analytics layer: sales/profit forecasting for the GGF Analytics App.

Pure, deterministic forecasting over the loaded joined transactions frame. This
module has **no Streamlit dependency** and does no plotting or I/O: the same
inputs always produce the same outputs and the caller's frames are never
mutated. It depends only on ``pandas``, ``numpy``, ``statsmodels``, and
:mod:`src.config`.

Method
------
Each SKU has exactly 12 monthly sales observations (Jan-Dec 2022), so heavy
machine learning is unjustified. The default method is **Holt's linear
exponential smoothing** (``statsmodels`` ``Holt`` /
``ExponentialSmoothing(trend="add", seasonal=None)``): it captures level and
trend from a short series without needing a full seasonal cycle (which would
require at least two years of monthly data). See Req 8.1.

Profit derivation (Req 8.4)
---------------------------
The profit forecast is derived from the sales (quantity) forecast by
multiplying each forecast month's quantity by a single **per-unit recomputed
margin** for the SKU::

    per_unit_margin = sum(recomputed_profit) / sum(Qty)      # over 2022

This volume-weighted average margin per unit is used (rather than the mean of
per-row ``recomputed_profit / Qty`` ratios) because it is robust to low-volume
rows and reconstructs the SKU's actual 2022 profit when applied to its actual
2022 quantity. The derivation is stated in every :class:`ForecastResult`'s
``assumptions`` (Req 8.6). ``recomputed_profit`` is the primitive-cost profit
basis documented in :mod:`src.config`.

Exclusion (Req 8.2)
-------------------
An SKU with fewer than :data:`config.MIN_OBSERVATIONS` (12) monthly
observations is **not fitted**: :func:`forecast_sku` returns a
:class:`ForecastResult` with ``excluded=True``, empty forecast series, and an
assumption recording the insufficient-data reason.

Forecast index (Req 8.3)
------------------------
The forecast covers exactly ``horizon`` (default 12, from
:data:`config.FORECAST_HORIZON`) months following the last historical month,
labeled as ``YYYY-MM`` period strings (e.g. ``"2023-01"`` .. ``"2023-12"``).
Forecast quantities are clipped at 0 so a downward trend never yields negative
sales (documented in ``assumptions``).
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd

from . import config

# ExponentialSmoothing/Holt emit convergence and index-frequency warnings on
# tiny 12-point series; they are expected here and do not affect the point
# forecast, so they are silenced locally at fit time.
try:  # pragma: no cover - import guard
    from statsmodels.tsa.holtwinters import ExponentialSmoothing
except Exception as exc:  # pragma: no cover - surfaced only if statsmodels absent
    raise ImportError(
        "forecasting requires statsmodels; install it into the environment"
    ) from exc


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ForecastResult:
    """A single SKU's forecast, or an exclusion record (Req 8.1-8.6).

    Attributes:
        sku: The SKU identifier (string key, e.g. ``"200100004"``).
        method: The forecasting method used (e.g. ``"holt"``). For an excluded
            SKU this is still the requested method label.
        historical: The SKU's monthly actual sales (quantity), indexed by
            ``YYYY-MM`` month label. Has up to 12 entries; may have fewer for an
            excluded SKU.
        forecast_sales: The forecast monthly sales quantity for the horizon,
            indexed by ``YYYY-MM`` month label. Empty when ``excluded`` is True.
        forecast_profit: The forecast monthly profit (``forecast_sales *
            per_unit_margin``) for the horizon, indexed identically to
            ``forecast_sales``. Empty when ``excluded`` is True.
        assumptions: A non-empty list of human-readable assumption strings
            stating the observation basis, the method, the profit derivation,
            and (when excluded) the insufficient-data reason (Req 8.6).
        excluded: ``True`` when the SKU had fewer than
            :data:`config.MIN_OBSERVATIONS` observations and was not fitted
            (Req 8.2); otherwise ``False``.
    """

    sku: str
    method: str
    historical: pd.Series
    forecast_sales: pd.Series
    forecast_profit: pd.Series
    assumptions: list[str]
    excluded: bool


# ---------------------------------------------------------------------------
# Forecast month index
# ---------------------------------------------------------------------------


def _future_month_labels(last_month: str, horizon: int) -> list[str]:
    """Return the ``horizon`` ``YYYY-MM`` labels following ``last_month``.

    Args:
        last_month: The last historical month label, e.g. ``"2022-12"``.
        horizon: Number of future months to generate.

    Returns:
        A list of ``YYYY-MM`` strings, e.g. ``["2023-01", ..., "2023-12"]``.
    """
    start = pd.Period(last_month, freq="M") + 1
    periods = pd.period_range(start=start, periods=horizon, freq="M")
    return [str(p) for p in periods]


# ---------------------------------------------------------------------------
# Single-SKU forecast (Req 8.1, 8.2, 8.3, 8.4, 8.6)
# ---------------------------------------------------------------------------


def forecast_sku(
    monthly_sales: pd.Series,
    per_unit_margin: float,
    horizon: int = config.FORECAST_HORIZON,
    method: str = config.DEFAULT_FORECAST_METHOD,
    sku: str = "",
) -> ForecastResult:
    """Forecast one SKU's monthly sales and profit (Req 8.1-8.4, 8.6).

    Fits Holt's linear exponential smoothing (additive trend, no seasonal) to
    the monthly sales quantity series and projects ``horizon`` months beyond the
    last historical month. The profit forecast is each forecast quantity times
    ``per_unit_margin`` (Req 8.4). Forecast quantities are clipped at 0.

    If the series has fewer than :data:`config.MIN_OBSERVATIONS` observations,
    the SKU is **not fitted**: an excluded :class:`ForecastResult` with empty
    forecasts and an insufficient-data assumption is returned (Req 8.2).

    Args:
        monthly_sales: Monthly actual sales quantity, indexed by ``YYYY-MM``
            month label in chronological order. Expected length is 12 for a
            fittable SKU.
        per_unit_margin: The SKU's recomputed per-unit margin (IDR per unit)
            used to derive the profit forecast from the sales forecast.
        horizon: Number of future months to forecast (default
            :data:`config.FORECAST_HORIZON` = 12).
        method: Method label recorded on the result (default
            :data:`config.DEFAULT_FORECAST_METHOD` = ``"holt"``).
        sku: The SKU identifier recorded on the result (optional).

    Returns:
        A :class:`ForecastResult`. When fitted, ``forecast_sales`` and
        ``forecast_profit`` each have length ``horizon`` and ``excluded`` is
        False; when excluded, both are empty and ``excluded`` is True.
    """
    historical = monthly_sales.astype(float)
    n_obs = int(historical.shape[0])

    # --- Exclusion path (Req 8.2): fewer than 12 observations, no fit ------
    if n_obs < config.MIN_OBSERVATIONS:
        empty = pd.Series(dtype=float)
        assumptions = [
            (
                f"SKU excluded from forecasting: only {n_obs} monthly "
                f"observation(s) available, fewer than the required "
                f"{config.MIN_OBSERVATIONS} (Req 8.2). No model was fitted."
            ),
        ]
        return ForecastResult(
            sku=str(sku),
            method=method,
            historical=historical,
            forecast_sales=empty,
            forecast_profit=empty,
            assumptions=assumptions,
            excluded=True,
        )

    # --- Fit Holt linear exponential smoothing -----------------------------
    # Only 12 points: additive trend, no seasonal component (a monthly seasonal
    # cycle would need >= 24 observations). Reset to a clean integer-positioned
    # series so statsmodels does not try to infer a date frequency.
    values = historical.to_numpy(dtype=float)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = ExponentialSmoothing(
            values,
            trend="add",
            seasonal=None,
            initialization_method="estimated",
        )
        fit = model.fit()
        raw_forecast = np.asarray(fit.forecast(horizon), dtype=float)

    # Guard against a negative trend producing negative sales (clip at 0).
    forecast_values = np.clip(raw_forecast, 0.0, None)

    last_month = str(historical.index[-1])
    future_labels = _future_month_labels(last_month, horizon)

    forecast_sales = pd.Series(
        forecast_values, index=pd.Index(future_labels, name="month")
    )
    forecast_profit = forecast_sales * float(per_unit_margin)

    assumptions = [
        (
            f"Forecast is based on {n_obs} monthly observations "
            f"(Jan-Dec 2022), the only history available per SKU (Req 8.6)."
        ),
        (
            f"Method: Holt's linear exponential smoothing (additive trend, no "
            f"seasonal component). With only {config.MIN_OBSERVATIONS} monthly "
            f"points a seasonal cycle cannot be estimated, so a level+trend "
            f"model is used ('{method}')."
        ),
        (
            "Profit forecast = forecast sales quantity x per-unit recomputed "
            f"margin ({per_unit_margin:,.2f} IDR/unit), where the per-unit "
            "margin is sum(recomputed_profit) / sum(Qty) over 2022 "
            "(volume-weighted average margin per unit)."
        ),
        "Forecast sales are clipped at 0 so a downward trend cannot yield "
        "negative quantities.",
        "The forecast assumes 2022 patterns persist and covers a single outlet "
        "(Outlet 1); it is not generalizable beyond that outlet.",
    ]

    return ForecastResult(
        sku=str(sku),
        method=method,
        historical=historical,
        forecast_sales=forecast_sales,
        forecast_profit=forecast_profit,
        assumptions=assumptions,
        excluded=False,
    )


# ---------------------------------------------------------------------------
# Per-SKU helpers
# ---------------------------------------------------------------------------


def _monthly_sales_series(sku_frame: pd.DataFrame) -> pd.Series:
    """Return an SKU's monthly sales-quantity series indexed by ``YYYY-MM``.

    Sums ``Qty`` per ``month`` and orders the index chronologically. Rows with a
    null month are dropped (they cannot be placed on the timeline).

    Args:
        sku_frame: Joined transaction rows for a single SKU. Must contain
            ``month`` and ``Qty``.

    Returns:
        A ``float`` Series indexed by ``YYYY-MM`` month label, sorted
        chronologically.
    """
    frame = sku_frame[sku_frame["month"].notna()]
    monthly = frame.groupby("month")["Qty"].sum().astype(float)
    monthly = monthly.sort_index()
    monthly.index = pd.Index([str(m) for m in monthly.index], name="month")
    return monthly


def _per_unit_margin(sku_frame: pd.DataFrame) -> float:
    """Return an SKU's volume-weighted per-unit recomputed margin (Req 8.4).

    Computes ``sum(recomputed_profit) / sum(Qty)`` over the SKU's rows. When the
    SKU sold no units (zero total quantity) the margin is 0.0 to avoid a
    division by zero.

    Args:
        sku_frame: Joined transaction rows for a single SKU. Must contain
            ``recomputed_profit`` and ``Qty``.

    Returns:
        The per-unit margin in IDR per unit.
    """
    total_qty = float(sku_frame["Qty"].sum())
    if not total_qty > 0:
        return 0.0
    total_profit = float(sku_frame["recomputed_profit"].sum())
    return total_profit / total_qty


# ---------------------------------------------------------------------------
# All-SKU forecast (Req 8.1-8.4, 8.6)
# ---------------------------------------------------------------------------


def forecast_all(
    joined: pd.DataFrame,
    horizon: int = config.FORECAST_HORIZON,
    method: str = config.DEFAULT_FORECAST_METHOD,
) -> list[ForecastResult]:
    """Forecast every SKU's sales and profit from the joined frame (Req 8.1-8.6).

    For each SKU (rows with a non-null ``SKU``), builds the monthly sales-
    quantity series, computes the SKU's per-unit recomputed margin, and calls
    :func:`forecast_sku`. SKUs are processed in sorted order for determinism.
    SKUs with fewer than :data:`config.MIN_OBSERVATIONS` observations come back
    excluded (Req 8.2); on the real workbook all three SKUs have 12 observations
    and are fitted.

    Args:
        joined: The bundle's joined transactions frame. Must contain ``SKU``,
            ``month``, ``Qty``, and ``recomputed_profit``.
        horizon: Number of future months to forecast (default 12).
        method: Method label recorded on each result (default ``"holt"``).

    Returns:
        A list of :class:`ForecastResult`, one per SKU, ordered by SKU.
    """
    frame = joined[joined["SKU"].notna()]
    results: list[ForecastResult] = []
    for sku, group in frame.groupby("SKU", sort=True):
        monthly_sales = _monthly_sales_series(group)
        per_unit_margin = _per_unit_margin(group)
        results.append(
            forecast_sku(
                monthly_sales=monthly_sales,
                per_unit_margin=per_unit_margin,
                horizon=horizon,
                method=method,
                sku=str(sku),
            )
        )
    return results
