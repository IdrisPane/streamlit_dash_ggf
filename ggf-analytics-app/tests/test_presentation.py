"""Tests for src.presentation (Task 13): deck compilation & pillar coverage.

Covers the presentation exporter's correctness properties and failure handling:

  * Property 25 (task 13.2): the deck embeds every analysis visualization
    (Req 11.1).
  * Property 26 (task 13.3): the executive summary precedes all
    detailed-breakdown slides (Req 11.2).
  * Property 27 (task 13.4): pillar coverage returns exactly the absent pillars
    and gates export (Req 11.3, 11.6).
  * Example test (task 13.5): a deck-generation failure raises PresentationError
    with a reason and yields no partial bytes (Req 11.5).

Building a real PPTX renders every Plotly figure to PNG via kaleido, which is
slow (each figure -> image). Tests therefore keep the figure count per example
small (1-3) and use tiny bar figures, and the hypothesis tests use a modest
``max_examples`` with ``deadline=None`` — building real decks is expensive, so a
lower example count is appropriate here.
"""

from __future__ import annotations

import io

import plotly.graph_objects as go
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st
from pptx import Presentation
from pptx.enum.shapes import MSO_SHAPE_TYPE

from src import config
from src.presentation import (
    ExecutiveSummary,
    PillarContent,
    PresentationError,
    build_deck,
    validate_pillar_coverage,
)

PILLARS = config.PILLARS  # ["data_model", "sales", "pnl", "predictive"]

# kaleido image rendering is slow: a single 4-pillar deck (4 tiny figures) takes
# ~20s (Chrome/engine startup dominates). Real deck-building hypothesis tests are
# therefore bounded to very few examples — a handful adequately validates the
# property while keeping runtime reasonable.
DECK_MAX_EXAMPLES = 3

_HC = [HealthCheck.too_slow, HealthCheck.data_too_large]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _tiny_figure() -> go.Figure:
    """A minimal Plotly figure that is cheap for kaleido to render to PNG."""
    return go.Figure(go.Bar(x=[1, 2], y=[1, 2]))


def _exec_summary() -> ExecutiveSummary:
    """A minimal executive summary for the deck's lead section."""
    return ExecutiveSummary(
        title="GGF 2022 review",
        verdict="Expand and Optimize",
        kpi_bullets=["Net revenue: 1 IDR", "Total profit: 1 IDR"],
    )


def _pillar_with_figs(pillar: str, n_figs: int) -> PillarContent:
    """A pillar with ``n_figs`` tiny figures (and a bullet for a lead slide)."""
    return PillarContent(
        pillar=pillar,
        title=f"{pillar} section",
        figures=[_tiny_figure() for _ in range(n_figs)],
        bullets=[f"{pillar} point"],
    )


def _pillar_bullets_only(pillar: str) -> PillarContent:
    """A pillar with a bullet and no figures (no kaleido cost)."""
    return PillarContent(pillar=pillar, title=f"{pillar} section", bullets=[f"{pillar} point"])


def _reopen(deck_bytes: bytes) -> Presentation:
    """Reopen compiled deck bytes as a python-pptx Presentation."""
    return Presentation(io.BytesIO(deck_bytes))


def _count_pictures(prs: Presentation) -> int:
    """Count embedded PICTURE shapes across all slides."""
    count = 0
    for slide in prs.slides:
        for shape in slide.shapes:
            if shape.shape_type == MSO_SHAPE_TYPE.PICTURE:
                count += 1
    return count


def _slide_texts(prs: Presentation) -> list[str]:
    """Return the concatenated text of every shape, per slide (index-aligned)."""
    texts: list[str] = []
    for slide in prs.slides:
        parts: list[str] = []
        for shape in slide.shapes:
            if shape.has_text_frame:
                parts.append(shape.text_frame.text)
        texts.append("\n".join(parts))
    return texts


# ---------------------------------------------------------------------------
# Property 25 (task 13.2): embedded-figure count (Req 11.1)
# ---------------------------------------------------------------------------


# Feature: ggf-analytics-app, Property 25: The deck embeds every analysis visualization
@settings(max_examples=DECK_MAX_EXAMPLES, deadline=None, suppress_health_check=_HC)
@given(figs_per_pillar=st.lists(st.integers(min_value=1, max_value=1), min_size=4, max_size=4))
def test_deck_embeds_every_figure(figs_per_pillar: list[int]) -> None:
    """The compiled deck embeds an image for each supplied figure (Req 11.1).

    Sections cover all four pillars, each with a known small number of tiny
    figures. After building and reopening the deck, the count of PICTURE shapes
    across all slides is at least the total number of figures supplied.
    """
    sections = {
        pillar: _pillar_with_figs(pillar, n)
        for pillar, n in zip(PILLARS, figs_per_pillar)
    }
    total_figures = sum(figs_per_pillar)

    deck = build_deck(_exec_summary(), sections)
    prs = _reopen(deck)

    assert _count_pictures(prs) >= total_figures


# ---------------------------------------------------------------------------
# Property 26 (task 13.3): exec summary precedes breakdown slides (Req 11.2)
# ---------------------------------------------------------------------------


# Feature: ggf-analytics-app, Property 26: Executive summary precedes all detailed-breakdown slides
@settings(max_examples=DECK_MAX_EXAMPLES, deadline=None, suppress_health_check=_HC)
@given(figs_per_pillar=st.lists(st.integers(min_value=0, max_value=1), min_size=4, max_size=4))
def test_exec_summary_precedes_breakdown(figs_per_pillar: list[int]) -> None:
    """Every executive-summary slide index is < every pillar slide index (Req 11.2).

    The slide carrying the "Executive summary" text must come before the first
    slide carrying any pillar title (a detailed-breakdown slide).
    """
    sections = {
        pillar: _pillar_with_figs(pillar, n)
        for pillar, n in zip(PILLARS, figs_per_pillar)
    }

    deck = build_deck(_exec_summary(), sections)
    texts = _slide_texts(_reopen(deck))

    exec_indices = [i for i, t in enumerate(texts) if "Executive summary" in t]
    assert exec_indices, "no executive-summary slide found"

    # Detailed-breakdown slides carry a pillar section title.
    pillar_titles = [f"{p} section" for p in PILLARS]
    pillar_indices = [
        i for i, t in enumerate(texts) if any(title in t for title in pillar_titles)
    ]
    assert pillar_indices, "no pillar slide found"

    assert max(exec_indices) < min(pillar_indices)


# ---------------------------------------------------------------------------
# Property 27 (task 13.4): pillar-coverage gating (Req 11.3, 11.6)
# ---------------------------------------------------------------------------


# Feature: ggf-analytics-app, Property 27: Pillar coverage — missing pillars are exactly the absent ones and gate export
@settings(max_examples=60, deadline=None, suppress_health_check=_HC)
@given(present=st.lists(st.sampled_from(PILLARS), min_size=0, max_size=4, unique=True))
def test_pillar_coverage_returns_exactly_absent(present: list[str]) -> None:
    """validate_pillar_coverage returns exactly the pillars with no content.

    The returned missing set equals PILLARS minus the present set, in PILLARS
    order (pure coverage check, so bullets-only content avoids kaleido cost).
    """
    sections = {p: _pillar_bullets_only(p) for p in present}
    expected_missing = [p for p in PILLARS if p not in present]

    assert validate_pillar_coverage(sections) == expected_missing


# Feature: ggf-analytics-app, Property 27: Pillar coverage — missing pillars are exactly the absent ones and gate export
@settings(max_examples=60, deadline=None, suppress_health_check=_HC)
@given(present=st.lists(st.sampled_from(PILLARS), min_size=0, max_size=3, unique=True))
def test_incomplete_coverage_blocks_export(present: list[str]) -> None:
    """build_deck raises PresentationError iff a pillar is missing content.

    ``present`` is a strict subset of the four pillars (max_size=3), so coverage
    is always incomplete and export must be blocked; the error names the missing
    pillars. Bullets-only content keeps this pure/fast (no deck is produced).
    """
    sections = {p: _pillar_bullets_only(p) for p in present}
    missing = validate_pillar_coverage(sections)
    assert missing, "expected an incomplete section set"

    with pytest.raises(PresentationError) as excinfo:
        build_deck(_exec_summary(), sections)
    for pillar in missing:
        assert pillar in str(excinfo.value)


def test_full_coverage_each_pillar_contributes_a_slide() -> None:
    """When all four pillars are present each contributes >=1 slide (Req 11.3).

    A representative build (bullets-only, no kaleido) confirms the coverage gate
    passes and a slide carrying each pillar's title is present.
    """
    sections = {p: _pillar_bullets_only(p) for p in PILLARS}
    assert validate_pillar_coverage(sections) == []

    deck = build_deck(_exec_summary(), sections)
    texts = _slide_texts(_reopen(deck))

    for pillar in PILLARS:
        assert any(f"{pillar} section" in t for t in texts), f"missing slide for {pillar}"


def test_full_coverage_with_figures_builds_and_embeds() -> None:
    """A representative full deck with one tiny figure per pillar builds & embeds.

    Exercises the real image-rendering path (kaleido) on a single small case to
    keep runtime reasonable while confirming figures are embedded.
    """
    sections = {p: _pillar_with_figs(p, 1) for p in PILLARS}
    deck = build_deck(_exec_summary(), sections)

    assert isinstance(deck, bytes) and deck
    prs = _reopen(deck)
    assert _count_pictures(prs) >= len(PILLARS)


# ---------------------------------------------------------------------------
# Example test (task 13.5): deck-failure handling (Req 11.5)
# ---------------------------------------------------------------------------


class _ExplodingFigure:
    """A stand-in figure whose image rendering fails (simulates a kaleido error)."""

    def to_image(self, *args, **kwargs):  # noqa: ANN002, ANN003, ANN201
        raise RuntimeError("boom: cannot render image")


def test_render_failure_raises_and_returns_no_partial_bytes() -> None:
    """A render failure raises PresentationError with a reason and no bytes (Req 11.5).

    A pillar is given a figure whose ``to_image`` raises. build_deck must raise
    PresentationError (surfacing the reason) rather than returning a partial deck.
    """
    sections = {p: _pillar_bullets_only(p) for p in PILLARS}
    # Replace the "sales" pillar's content with one that has an exploding figure.
    sections["sales"] = PillarContent(
        pillar="sales",
        title="sales section",
        figures=[_ExplodingFigure()],
        bullets=["sales point"],
    )

    with pytest.raises(PresentationError) as excinfo:
        build_deck(_exec_summary(), sections)

    # The error carries a human-readable reason (the underlying cause).
    assert "boom" in str(excinfo.value) or "render" in str(excinfo.value).lower()
