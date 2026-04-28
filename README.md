# Redispatch Visualization Dashboard — Schleswig-Holstein

> **v1.0** — historical visualization of redispatch activity across the SHN
> distribution grid. **No model predictions in v1.** Predictions ship in v2;
> the underlying ML pipeline lives in [`src/v2/`](src/v2/README.md).

## What it shows

For any day in the data window (Jan 2024 → present), see **which substations
needed redispatch and for how many hours**. Bigger and redder bubble on the
map = more hours of grid-bottleneck activity that day.

A redispatch event is a grid operator instruction to a wind farm or other
plant to **reduce or shift output** because the local transmission lines
cannot move all the generated power away. In Schleswig-Holstein this
clusters along the windy North Sea coast.

## Quick start

```bash
pip install -r requirements.txt
streamlit run app.py
```

Then open <http://localhost:8501>.

If `data/processed/ts_15min_wide.parquet` is missing, build it once from raw:

```bash
python src/build_timeseries.py
```

## Dashboard layout

**Tab 1 — Daily map.** Date picker, severity threshold slider, daily summary
banner ("It was a busy day · 12 substations had redispatch"), KPI strip,
interactive map with all 175 substations, top-15 leaderboard, CSV export.

**Tab 2 — Town deep dive.** Pick a town and an end date, get:
- 90-day daily-active-hours bar chart with a 7-day rolling average overlay
- 7-day × 24-hour heatmap of which hours of the day were congested
- KPIs: total active hours, days with redispatch, busiest day, % of time congested

## Severity metric

**Total active hours per (town, day)**, between 0 and 24. A town active 14
hours had at least one redispatch event overlapping each of those 14 hours.

Secondary metrics shown alongside (in tooltips and tables, never on the map):
- Number of distinct redispatch operations that day
- Peak concurrent ops at the busiest 15-min slot
- Dominant reason (Netzengpass / Netzengpass I) -  Grid Congestion

## Repo layout

```
.
├── app.py                          Streamlit dashboard (v1)
├── README.md
├── requirements.txt
├── .gitignore
├── .streamlit/config.toml          dark theme base
│
├── src/                            v1 visualization code
│   ├── build_timeseries.py         raw parquets -> 15-min activity matrix
│   ├── smard_client.py             SMARD API client (used by v2)
│   ├── weather_client.py           Open-Meteo client (used by v2)
│   ├── fetcher.py                  cache helper for SMARD + weather
│   ├── theme.py                    Framer-inspired colours, CSS, Plotly template
│   └── v2/                         day-ahead prediction work-in-progress
│       └── README.md               details + v2 roadmap
│
├── notebooks/
│   ├── 01_sanity_check.ipynb       data quality + grid-analyst exploration
│   └── v2/predict_demo.ipynb       prediction prototype walkthrough (v2)
│
├── data/                           (gitignored, regenerable)
│   ├── raw/shn_operations_last_2y/
│   ├── external/towns_geo.parquet
│   └── processed/ts_15min_*.parquet
│
└── models/                         (gitignored, v2 artifacts only)
```

## v2 roadmap (deferred from v1)

A full LightGBM-based day-ahead probability forecast prototype lives under
`src/v2/`. It runs end-to-end and gets 4-5× lift over baseline on a held-out
test set. Before it can shipped, six items remain:

1. **Live forecast pipeline.** `src/v2/score_today.py` works against cached
   features today. Needs an upstream live-data fetcher.
2. **Forecast-vintage features.** Replace SMARD/weather actuals with
   day-ahead forecast vintages to remove train/serve skew.
3. **Per-town isotonic calibration.** Closes the 15-25pp under-prediction
   gap at chronically-busy coast towns.
4. **Naive baselines.** Persistence, last-7-days-same-hour, base-rate.
   Required so we can prove the model beats simple heuristics.
5. **Smoke tests.** Lock in PR-AUC floors and CLI contracts.
6. **Schema validation** on raw parquets so upstream renames fail clearly.

## Reproducibility

Everything is deterministic given the raw parquet chunks under
`data/raw/shn_operations_last_2y/` and the Open-Meteo / SMARD caches under
`data/external/`. A full rebuild from scratch (timeseries → features →
optionally train) takes about 15 minutes on a 16 GB laptop.

## Limitations

- **SHN only.** No southern German DSOs in this dataset. Coverage ends at
  the Schleswig-Holstein state border.
- **Geocoding is town-centroid.** Some entries are private 110 kV
  substations whose precise location we approximate from the parent town.
- **No live data feed.** Activity ends at the latest cached parquet. To
  refresh, re-pull the SHN source and rerun `src/build_timeseries.py`.
