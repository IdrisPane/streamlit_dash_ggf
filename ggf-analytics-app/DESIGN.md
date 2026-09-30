# Design Direction — GGF Analytics Dashboard

This is the approved visual and design direction for the app. It is a data-first
internal analytics dashboard, not a marketing surface. Builders should treat this
as the source of truth for look and feel and defer to it when making UI choices.

## Identity

- GGF internal analytics dashboard for a fruit-beverage business.
- Audience: GGF top management making an expansion decision (expand now vs. optimize
  existing operations first).
- The job of the interface is to present defensible analysis clearly. Visualizations
  are supporting tools, not decoration.

## Visual language

- Restrained, data-first, clean business/analytics look.
- The interface should feel calm and credible. Nothing competes with the numbers.

## Palette

- Near-white background: `#FAFAFA`, with white surfaces for cards/panels.
- Neutral gray scale for text and borders.
- A single deep-teal accent (e.g. `#0F766E` or a similar teal) used sparingly — only
  for the key metric / focal point and primary actions.
- Cap the palette at 2–3 core colors plus the one accent (antislop R-29).
- Semantic colors, used only for financial meaning (not decoration):
  - A restrained red only for losses / negative values.
  - A restrained green only for positive / profit values.
  - Reason: this is financial data, so red/green carry semantic meaning about
    gain vs. loss; they are not applied as general styling.

## Typography

- Readable sans-serif (system UI / Source Sans style).
- No monospace-as-aesthetic.
- No wide-tracked uppercase.

## Motion dials

- ENERGY 1, RHYTHM 2, MOTION 1.
- Calm and mostly static. Use hover and native interactions only. No endless or
  looping animations.

## Notes for builders

- Every number comes from the real `FruitsTrxDatasets.xlsx`. No fabricated stats
  (antislop R-17 / R-38 satisfied by design — numbers are computed at runtime).
- Every chart must answer a specific question stated in its title.
- Empty, loading, and error states must state the cause and the next action.
- Keep glass / glow / shadow effects at their dose caps.
- Avoid decorative icons and status dots.
