# `src/v2/` — Day-ahead prediction work-in-progress

These modules implement a **day-ahead probability forecast** for redispatch
events. They are **not part of the v1 visualization dashboard**. They are
preserved here because the prototype works end-to-end and v2 will pick up
from this baseline.

## Files

| File | Role |
|---|---|
| `build_features.py` | hourly feature table from 15-min activity + SMARD + weather |
| `train.py` | three LightGBM models, one per look-ahead window (1h, 6h, 24h) |
| `predict.py` | town + date -> hour-by-hour calibrated probabilities |
| `score_today.py` | scheduled-scoring CLI; writes `data/predictions/predictions_<date>.parquet` |
| `review_model.py` | end-to-end diagnostic (calibration, geographic, per-town PR-AUC) |
| `labels.py` | plain-English mapping for ML feature names |

## Why v2 not v1

v1 ships a visualization-only dashboard so operators can adopt it without
the trust burden of model predictions. Once v1 is in daily use, v2 layers
day-ahead forecast probabilities onto the same map.

## v2 deployment plan (deferred from v1)

1. **Live forecast pipeline** — `score_today.py` already runs against
   cached features. Item 2 below upgrades it to fetch fresh inputs.
2. **Forecast-vintage features** — replace SMARD/weather actuals with
   day-ahead forecast vintages to remove train/serve skew. Touches
   `smard_client.py` (add forecast filter IDs) and `weather_client.py`
   (switch to Open-Meteo previous-forecast archive).
3. **Per-town isotonic calibration** — closes the 15-25pp under-prediction
   gap at chronically-busy coast towns (Lensahn, Cismar West, etc.).
4. **Naive baselines** — persistence, last-7-days-same-hour, base-rate.
   Required for procurement sign-off ("model beats what I'd guess").
5. **Smoke tests** — `pytest` suite locking in PR-AUC floors and CLI
   contracts so refactors don't silently regress.
6. **Schema validation** on raw parquets (`pandera` or hand-rolled).

## Running anything in here from the project root

```bash
python src/v2/build_features.py
python src/v2/train.py
python src/v2/score_today.py --date 2026-02-15
python src/v2/review_model.py
```

Path roots in these scripts are computed as `Path(__file__).parents[2]`
so they resolve to the project root regardless of where they're invoked from.
