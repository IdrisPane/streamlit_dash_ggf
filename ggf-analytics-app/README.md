# GGF Analytics App

An internal, locally-run analytics dashboard for GGF's fruit-beverage business.
It loads `FruitsTrxDatasets.xlsx`, documents the data model, and presents analysis
across four pillars — data documentation/modeling, sales performance, Profit & Loss,
and predictive analysis — plus supporting views for raw-material seasonality and
assumptions. The analysis compiles into a downloadable slide deck (PPTX) that leads
with an executive summary and follows with the detailed breakdown.

The audience is GGF top management deciding whether to expand the business or optimize
existing operations first. Every number is computed from the real workbook at runtime.

## Requirements

- Python 3.14 (Windows)
- Dependencies pinned in `requirements.txt`

## Setup

```powershell
python -m pip install -r requirements.txt
```

The dataset ships in `data/FruitsTrxDatasets.xlsx`. To point at a different copy,
set the workbook path via the environment variable read in `src/config.py`.

## Run

```powershell
streamlit run app.py
```

This launches the app locally on the Python + Streamlit stack (Requirement 10.6).
Use the sidebar to switch views and apply SKU/month filters; the "Export deck" button
produces the PPTX once every pillar has content.

## Test

```powershell
python -m pytest
```

Unit and example tests live alongside the analytics modules in `tests/`;
property-based tests (hypothesis) live in `tests/test_properties.py`.

## Project structure

```
ggf-analytics-app/
  app.py                      # entry: navigation, filters, load + error gate, export button
  requirements.txt            # pinned dependencies
  DESIGN.md                   # approved visual/design direction
  pages/                      # seven Streamlit views (added in Task 11)
  src/                        # pure analytics + data layer
    __init__.py
    config.py                 # constants: SKU map, column normalization, thresholds
    data_loader.py            # load_data, normalize_columns, coerce_types, joins
    metrics.py                # KPIs, attainment, per-SKU P&L, cost drivers, verdict
    anomalies.py              # reconciliation + loss + unprocessable
    forecasting.py            # Holt forecaster
    charts.py                 # Plotly figure builders
    presentation.py           # build_deck, validate_pillar_coverage
  tests/                      # unit / example / integration / property tests
  data/
    FruitsTrxDatasets.xlsx    # source workbook
```

Analytics logic is kept pure and decoupled from Streamlit and Plotly so it can be
unit- and property-tested in isolation.
