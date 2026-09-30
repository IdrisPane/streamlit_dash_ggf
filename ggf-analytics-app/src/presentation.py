"""Presentation layer: compile the GGF analysis into a slide deck (PPTX).

Pure, deterministic assembly of a PowerPoint deck from Plotly figures and metrics
text. This module has **no Streamlit dependency**: it consumes figures (from
:mod:`src.charts`) and computed metrics, renders each figure to a static PNG via
``kaleido`` (``fig.to_image(format="png")``), embeds it with ``python-pptx``, and
returns the deck as ``bytes`` for the caller to download (Req 11.1-11.6).

Deck structure (per the GGF brief and project ``DESIGN.md``)
------------------------------------------------------------
The deck is for GGF top management, so it leads with the decision and the numbers
behind it, then supports them:

1. **Executive summary first** (Req 11.2): a title slide followed by the expansion
   verdict (the recommendation) and the headline 2022 KPIs as bullets. Every
   executive-summary slide precedes every detailed-breakdown slide (Property 26).
2. **Detailed breakdown** across the four analysis pillars
   (:data:`config.PILLARS` — ``data_model``, ``sales``, ``pnl``, ``predictive``),
   at least one slide per pillar (Req 11.3), each embedding that pillar's figures
   and bullets. Every supplied figure is embedded, so the deck's image count is at
   least the number of figures supplied (Property 25).

Coverage gate (Req 11.6)
------------------------
:func:`validate_pillar_coverage` returns exactly the pillars from
:data:`config.PILLARS` that have no content (no figures and no bullets).
:func:`build_deck` calls it and raises :class:`PresentationError` naming the
missing pillars when that set is non-empty, so a deck missing a pillar is never
produced. Any other generation failure also raises :class:`PresentationError`
with a clear reason and no partial bytes are returned (Req 11.5).

The convenience assembler :func:`build_default_deck` builds the exec summary and
all four pillar sections from a loaded :class:`~src.data_loader.DataBundle` and the
analytics/chart layers, so the app (task 14) can produce the deck in one call.
"""

from __future__ import annotations

import io
from dataclasses import dataclass, field
from typing import Optional

import plotly.graph_objects as go
from pptx import Presentation
from pptx.util import Emu, Inches

from . import charts, config

#: Analysis pillars the deck must cover, in detailed-breakdown order (Req 11.3).
PILLARS: list[str] = config.PILLARS

#: Slide dimensions (16:9 widescreen) used for placing images and text.
_SLIDE_WIDTH: int = Inches(13.333)
_SLIDE_HEIGHT: int = Inches(7.5)

#: Human-readable titles for each pillar's detailed-breakdown section.
_PILLAR_TITLES: dict[str, str] = {
    "data_model": "Data model & analysis process",
    "sales": "Sales performance vs target",
    "pnl": "Profit & loss and cost drivers",
    "predictive": "12-month sales & profit forecast",
}


class PresentationError(Exception):
    """Raised when the deck cannot be generated (Req 11.5, 11.6).

    Carries a human-readable reason: either a missing-pillar message (from the
    coverage gate, Req 11.6) or the underlying cause of a generation failure
    (Req 11.5). When raised, no partial deck bytes are produced.
    """


@dataclass(frozen=True)
class ExecutiveSummary:
    """The executive-summary section that leads the deck (Req 11.2).

    Attributes:
        title: The deck/title-slide headline (e.g. the recommendation as a
            statement).
        verdict: The expansion recommendation shown first (e.g. ``"Expand and
            Optimize"``); rendered as the primary point.
        kpi_bullets: Headline 2022 KPIs as short bullet strings (net revenue,
            cost, profit, margin, attainment, least-profitable SKU).
    """

    title: str
    verdict: str
    kpi_bullets: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class PillarContent:
    """One pillar's detailed-breakdown content (Req 11.1, 11.3).

    A pillar "has content" (for the coverage gate) when it has at least one
    figure or at least one bullet.

    Attributes:
        pillar: The pillar key, one of :data:`config.PILLARS`.
        title: The slide title for this pillar's section (states the point).
        figures: The Plotly figures to embed for this pillar (each becomes an
            embedded image, Req 11.1).
        bullets: Supporting text bullets for this pillar (rendered on the
            pillar's lead slide).
    """

    pillar: str
    title: str
    figures: list[go.Figure] = field(default_factory=list)
    bullets: list[str] = field(default_factory=list)

    def has_content(self) -> bool:
        """Return True when this pillar has at least one figure or bullet."""
        return bool(self.figures) or bool(self.bullets)


def validate_pillar_coverage(sections: dict[str, PillarContent]) -> list[str]:
    """Return exactly the pillars in :data:`config.PILLARS` with no content (Req 11.6).

    A pillar counts as covered when ``sections`` maps its key to a
    :class:`PillarContent` that has at least one figure or bullet. A pillar is
    missing when it is absent from ``sections`` or its content is empty. Extra
    keys in ``sections`` that are not in :data:`config.PILLARS` are ignored.

    Args:
        sections: Mapping of pillar key to its :class:`PillarContent`.

    Returns:
        The missing pillars, in :data:`config.PILLARS` order. Empty when all four
        pillars are covered (export may proceed).
    """
    missing: list[str] = []
    for pillar in PILLARS:
        content = sections.get(pillar)
        if content is None or not content.has_content():
            missing.append(pillar)
    return missing


def _render_png(fig: go.Figure) -> bytes:
    """Render a Plotly figure to PNG bytes via kaleido (Req 11.1).

    Args:
        fig: The figure to render.

    Returns:
        The PNG image bytes.

    Raises:
        PresentationError: If the figure cannot be rendered to an image.
    """
    try:
        return fig.to_image(format="png", width=1400, height=800, scale=1)
    except Exception as exc:  # noqa: BLE001 - surface any render failure uniformly.
        raise PresentationError(
            f"Failed to render a chart image for the deck: {exc}"
        ) from exc


def _add_title_slide(prs: Presentation, title: str, subtitle: str) -> None:
    """Add the deck's opening title slide (executive-summary lead)."""
    layout = prs.slide_layouts[0]  # Title slide.
    slide = prs.slides.add_slide(layout)
    slide.shapes.title.text = title
    if len(slide.placeholders) > 1:
        slide.placeholders[1].text = subtitle


def _add_bullets_slide(prs: Presentation, title: str, bullets: list[str]) -> None:
    """Add a title-and-content slide rendering ``bullets`` as a bullet list."""
    layout = prs.slide_layouts[1]  # Title and content.
    slide = prs.slides.add_slide(layout)
    slide.shapes.title.text = title

    body = slide.placeholders[1].text_frame
    body.clear()
    for i, bullet in enumerate(bullets):
        para = body.paragraphs[0] if i == 0 else body.add_paragraph()
        para.text = str(bullet)
        para.level = 0


def _add_figure_slide(prs: Presentation, title: str, fig: go.Figure) -> None:
    """Add a slide with ``title`` and one embedded figure image (Req 11.1).

    The image is rendered to PNG (kaleido) and centered in the slide's body area
    at a size that fits the 16:9 canvas.
    """
    png = _render_png(fig)

    layout = prs.slide_layouts[5]  # Title only.
    slide = prs.slides.add_slide(layout)
    slide.shapes.title.text = title

    # Fit the image within the body area, centered, preserving its 1400x800
    # aspect ratio (7:4).
    pic_width = Inches(11.0)
    pic_height = Emu(int(pic_width * 800 / 1400))
    left = Emu(int((prs.slide_width - pic_width) / 2))
    top = Inches(1.6)
    slide.shapes.add_picture(
        io.BytesIO(png), left, top, width=pic_width, height=pic_height
    )


def build_deck(
    exec_summary: ExecutiveSummary,
    sections: dict[str, PillarContent],
) -> bytes:
    """Compile the executive summary and pillar sections into a PPTX (Req 11.1-11.6).

    The deck is ordered executive-summary-first (Req 11.2, Property 26): a title
    slide, then an executive-summary content slide (the verdict as the primary
    point, headline KPIs as bullets). It is followed by the detailed breakdown,
    one section per pillar in :data:`config.PILLARS` order, each with a lead slide
    (title + any bullets) and one slide per supplied figure (Req 11.1, 11.3). Every
    supplied figure is embedded as an image, so the embedded-image count is at
    least the number of figures supplied (Property 25).

    Before building, :func:`validate_pillar_coverage` gates the export: if any of
    the four pillars has no content the deck is not produced and a
    :class:`PresentationError` naming the missing pillars is raised (Req 11.6). Any
    other failure during assembly also raises :class:`PresentationError` with the
    reason and yields no partial bytes (Req 11.5).

    Args:
        exec_summary: The executive-summary section (leads the deck).
        sections: Mapping of pillar key to its :class:`PillarContent`.

    Returns:
        The compiled ``.pptx`` file as ``bytes``.

    Raises:
        PresentationError: If a pillar is missing content (Req 11.6) or the deck
            cannot be generated (Req 11.5).
    """
    missing = validate_pillar_coverage(sections)
    if missing:
        raise PresentationError(
            "Cannot export: missing content for pillar(s): "
            + ", ".join(missing)
            + ". Every pillar (data_model, sales, pnl, predictive) needs at "
            "least one chart or bullet."
        )

    try:
        prs = Presentation()
        prs.slide_width = _SLIDE_WIDTH
        prs.slide_height = _SLIDE_HEIGHT

        # --- Executive summary first (Req 11.2) --------------------------
        _add_title_slide(prs, exec_summary.title, f"Recommendation: {exec_summary.verdict}")
        exec_bullets = [f"Verdict: {exec_summary.verdict}", *exec_summary.kpi_bullets]
        _add_bullets_slide(prs, "Executive summary", exec_bullets)

        # --- Detailed breakdown per pillar (Req 11.1, 11.3) --------------
        for pillar in PILLARS:
            content = sections[pillar]
            section_title = content.title or _PILLAR_TITLES.get(pillar, pillar)

            # Lead slide with the pillar's bullets (guarantees >=1 slide per
            # pillar even when it has figures only).
            lead_bullets = content.bullets or [section_title]
            _add_bullets_slide(prs, section_title, lead_bullets)

            for i, fig in enumerate(content.figures, start=1):
                fig_title = (
                    section_title if len(content.figures) == 1
                    else f"{section_title} ({i}/{len(content.figures)})"
                )
                _add_figure_slide(prs, fig_title, fig)

        buffer = io.BytesIO()
        prs.save(buffer)
        return buffer.getvalue()
    except PresentationError:
        raise
    except Exception as exc:  # noqa: BLE001 - surface any assembly failure.
        raise PresentationError(
            f"Failed to generate the presentation: {exc}"
        ) from exc


# ---------------------------------------------------------------------------
# Convenience assembler (for app.py, task 14)
# ---------------------------------------------------------------------------


def _fmt_idr(value: float) -> str:
    """Format an IDR amount with thousands separators (no decimals)."""
    return f"{value:,.0f} IDR"


def build_default_deck(bundle) -> bytes:
    """Assemble and build the default deck from a loaded bundle (Req 11.1-11.6).

    Computes the executive summary and all four pillar sections from the loaded
    :class:`~src.data_loader.DataBundle` using the analytics layers
    (:mod:`src.metrics`, :mod:`src.anomalies`, :mod:`src.forecasting`) and the
    Plotly builders in :mod:`src.charts`, then delegates to :func:`build_deck`. All
    four pillars are populated, so the default deck always passes the coverage gate
    and exports.

    Pillar mapping:

    - ``data_model``: a text overview slide (source-table row counts and the
      column-normalization pairs from ``bundle``) — no chart needed.
    - ``sales``: monthly sales trend (with anomaly marks), actual-vs-target, and
      per-SKU/month attainment charts, plus anomaly-count bullets.
    - ``pnl``: per-SKU profit and cost-driver charts, plus least-profitable-SKU,
      negative-profit-month, reconciliation, and improvement-proposal bullets.
    - ``predictive``: forecast-vs-actual charts (profit and sales) plus the
      forecast-to-verdict relationship and assumptions.

    Args:
        bundle: A loaded :class:`~src.data_loader.DataBundle` (has ``joined``,
            ``raw_mat_trend``, ``sales_transaction``, table frames, and
            ``normalization_map``).

    Returns:
        The compiled ``.pptx`` file as ``bytes``.

    Raises:
        PresentationError: If the deck cannot be generated (Req 11.5). By
            construction the coverage gate passes.
    """
    # Imported here (not at module top) so :mod:`src.presentation` stays a light
    # pure module and importing it does not pull in statsmodels via forecasting.
    from . import forecasting, metrics
    from .anomalies import detect_reconciliation

    joined = bundle.joined
    raw_mat = bundle.raw_mat_trend

    # --- Shared analytics -------------------------------------------------
    totals = metrics.totals_2022(joined)
    attainment = metrics.overall_target_attainment(joined)
    pnl_df = metrics.per_sku_pnl(joined)
    least = metrics.least_profitable_sku(pnl_df)
    neg_months = metrics.negative_profit_month_count(joined)
    breakdown = metrics.cost_driver_breakdown(joined)
    proposals = metrics.improvement_proposals(joined)
    matrix = metrics.monthly_sales_matrix(joined)
    attain_matrix = metrics.target_attainment_matrix(joined)
    anomalies = metrics.sales_anomalies(joined)
    seasonality = metrics.seasonality_price_compare(raw_mat)
    forecasts = forecasting.forecast_all(joined)
    verdict = metrics.expansion_verdict(totals, pnl_df, forecasts)
    recon = detect_reconciliation(bundle.sales_transaction)

    def _sku_label(sku: str) -> str:
        name = config.SKU_MAP.get(str(sku))
        return f"{name} ({sku})" if name else str(sku)

    # --- Executive summary (Req 11.2) ------------------------------------
    margin_text = "N/A" if totals.margin_pct is None else f"{totals.margin_pct:.1f}%"
    attain_text = "N/A" if attainment is None else f"{attainment:.1f}%"
    least_text = (
        ", ".join(
            f"{_sku_label(s.sku)} at {_fmt_idr(s.total_profit)}" for s in least
        )
        if least
        else "N/A"
    )
    exec_summary = ExecutiveSummary(
        title="GGF 2022 business review & expansion decision",
        verdict=verdict.recommendation,
        kpi_bullets=[
            f"Net revenue: {_fmt_idr(totals.net_revenue)}",
            f"Total cost: {_fmt_idr(totals.total_cost)}",
            f"Total profit: {_fmt_idr(totals.total_profit)} (margin {margin_text})",
            f"Overall target attainment: {attain_text}",
            f"Least-profitable SKU: {least_text}",
            f"Rationale: {verdict.rationale}",
        ],
    )

    # --- data_model pillar (text overview) -------------------------------
    tables = [
        ("MstProduct", bundle.mst_product),
        ("SalesTarget", bundle.sales_target),
        ("SalesTransaction", bundle.sales_transaction),
        ("RawMatPriceTrend", bundle.raw_mat_trend),
    ]
    data_model_bullets = [f"{name}: {len(df):,} rows" for name, df in tables]
    data_model_bullets.append(
        "SKU join links all four tables; SalesTransaction aligns to SalesTarget "
        "by calendar month (Jan-Dec 2022)."
    )
    data_model_bullets += [
        f"Normalized column: {src} -> {dst}"
        for src, dst in bundle.normalization_map.items()
    ]
    data_model_bullets.append(
        f"Profit basis: {config.PROFIT_BASIS_ID} = {config.PROFIT_BASIS_FORMULA}."
    )

    # --- sales pillar -----------------------------------------------------
    sales_bullets = [
        f"Overall 2022 target attainment: {attain_text}.",
        f"Flagged sales anomalies (>2 std or attainment <50%/>150%): {len(anomalies)}.",
    ]
    sales_bullets += [a.reason for a in anomalies[:5]]
    sales_figs = [
        charts.sales_trend_figure(matrix, anomalies),
        charts.sales_vs_target_figure(matrix),
        charts.attainment_figure(attain_matrix),
    ]

    # --- pnl pillar -------------------------------------------------------
    pnl_bullets = [
        f"Least-profitable SKU: {least_text}.",
    ]
    pnl_bullets += [
        f"{_sku_label(sku)}: {count} negative-profit month(s) of 12."
        for sku, count in sorted(neg_months.items())
    ]
    pnl_bullets.append(
        f"Reconciliation discrepancies: {recon.discrepancy_count} row(s); "
        f"unprocessable: {recon.unprocessable_count}."
    )
    pnl_bullets.append(
        f"Profit basis: {config.PROFIT_BASIS_ID} ({config.PROFIT_BASIS_FORMULA})."
    )
    pnl_bullets += [p.text for p in proposals[:3]]
    pnl_figs = [
        charts.pnl_by_sku_figure(pnl_df),
        charts.cost_driver_figure(breakdown),
        charts.seasonality_figure(seasonality),
    ]

    # --- predictive pillar ------------------------------------------------
    fitted = [f for f in forecasts if not f.excluded]
    excluded = [f for f in forecasts if f.excluded]
    predictive_bullets = [
        f"Fitted {len(fitted)} SKU(s) with 12 monthly observations; "
        f"excluded {len(excluded)} SKU(s) for insufficient data.",
        f"Forecast horizon: {config.FORECAST_HORIZON} months beyond Dec 2022.",
        "Profit forecast = forecast sales quantity x recomputed per-unit margin.",
        f"Forecast relationship to the verdict: {verdict.forecast_relationship}.",
    ]
    if fitted:
        predictive_bullets += [f"Assumption: {a}" for a in fitted[0].assumptions[:3]]
    predictive_figs = [
        charts.forecast_figure(forecasts, metric="profit"),
        charts.forecast_figure(forecasts, metric="sales"),
    ]

    sections: dict[str, PillarContent] = {
        "data_model": PillarContent(
            pillar="data_model",
            title=_PILLAR_TITLES["data_model"],
            figures=[],
            bullets=data_model_bullets,
        ),
        "sales": PillarContent(
            pillar="sales",
            title=_PILLAR_TITLES["sales"],
            figures=sales_figs,
            bullets=sales_bullets,
        ),
        "pnl": PillarContent(
            pillar="pnl",
            title=_PILLAR_TITLES["pnl"],
            figures=pnl_figs,
            bullets=pnl_bullets,
        ),
        "predictive": PillarContent(
            pillar="predictive",
            title=_PILLAR_TITLES["predictive"],
            figures=predictive_figs,
            bullets=predictive_bullets,
        ),
    }

    return build_deck(exec_summary, sections)
