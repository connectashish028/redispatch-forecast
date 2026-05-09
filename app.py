"""
Redispatch Visualization Dashboard (v1) - SHN Schleswig-Holstein.

Pure historical visualization: no model predictions. For any day in the data
window, see which substations had redispatch and for how many hours.

Run from project root:
    streamlit run app.py
"""
from __future__ import annotations

import io
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'src'))

from theme import (inject_css, register_plotly_template,        # noqa: E402
                   COLOR_BG, COLOR_SURFACE, COLOR_TEXT,
                   COLOR_TEXT_MUTED, COLOR_ACCENT,
                   COLOR_RING, HEAT_SCALE)

st.set_page_config(
    page_title='Redispatch Visualization - SHN',
    page_icon='⚡',
    layout='wide',
    initial_sidebar_state='collapsed',
)
inject_css(st)
register_plotly_template()

WIDE_PATH = ROOT / 'data' / 'processed' / 'ts_15min_wide.parquet'
LONG_PATH = ROOT / 'data' / 'processed' / 'ts_15min_long.parquet'
GEO_PATH  = ROOT / 'data' / 'external'  / 'towns_geo.parquet'


# ---------------------- caches ----------------------

def _read_parquet(path: Path) -> pd.DataFrame:
    try:
        return pd.read_parquet(path)
    except FileNotFoundError:
        raise FileNotFoundError(f"Missing {path.relative_to(ROOT)}. Run `python src/build_timeseries.py`.")
    except Exception as exc:
        raise RuntimeError(f"Unable to load {path.relative_to(ROOT)}: {exc}")


@st.cache_resource
def load_data() -> dict:
    """Load wide (15-min × town) + long (with reasons) + geo. Returns a dict."""
    wide = _read_parquet(WIDE_PATH)
    long = _read_parquet(LONG_PATH) if LONG_PATH.exists() else None
    geo  = _read_parquet(GEO_PATH).dropna(subset=['lat', 'lon'])
    return {'wide': wide, 'long': long, 'geo': geo}


@st.cache_data
def daily_hours(date_str: str) -> pd.DataFrame:
    """For one date, return per-town daily metrics:
        active_hours        - hours with at least one redispatch op (0-24)
        active_15min_slots  - same in 15-min slots (0-96)
        n_events            - number of distinct redispatch operations starting that day
        peak_concurrency    - max simultaneous active ops at any 15-min slot
        dominant_reason     - most common reason that day (Netzengpass / Netzengpass I)
    """
    d   = pd.Timestamp(date_str).normalize()
    nxt = d + pd.Timedelta(days=1)
    data = load_data()
    wide = data['wide']
    day  = wide.loc[(wide.index >= d) & (wide.index < nxt)]
    if day.empty:
        return pd.DataFrame()

    # active 15-min slots, peak concurrency
    is_active = (day > 0).astype('int8')
    active_slots    = is_active.sum(axis=0).rename('active_15min_slots')
    peak_concurrent = day.max(axis=0).rename('peak_concurrency')

    # active hours — count distinct hours (96 slots / 4 = 24 hours)
    hourly = is_active.copy()
    hourly.index = hourly.index.floor('h')
    hours = hourly.groupby(level=0).max().sum(axis=0).rename('active_hours')

    # event count = total positive jumps in the concurrency signal during the day
    # (rising edges in `day` count distinct new ops; ops already active at the
    # start of the day count as 'overlapping' starts via the fill_value=0 prior).
    deltas = day.astype('int16').diff().fillna(day.astype('int16').iloc[0])
    n_events = deltas.clip(lower=0).sum(axis=0).astype('int32').rename('n_events')

    out = pd.concat([active_slots, hours, peak_concurrent, n_events], axis=1)
    out.index.name = 'town'
    out = out.reset_index()

    # dominant reason from the long form (if available)
    long = data['long']
    if long is not None:
        sub = long[(long['ts'] >= d) & (long['ts'] < nxt)]
        if not sub.empty:
            n_i = sub.groupby('town')['n_netzengpass_i'].sum()
            n_n = sub.groupby('town')['n_netzengpass'].sum()
            reason = pd.Series(np.where(n_i >= n_n, 'Netzengpass I', 'Netzengpass'),
                               index=n_i.index, name='dominant_reason')
            out = out.merge(reason.reset_index(), on='town', how='left')
    if 'dominant_reason' not in out.columns:
        out['dominant_reason'] = pd.NA

    out = out.merge(data['geo'][['town', 'lat', 'lon']], on='town', how='left')
    return out


@st.cache_resource
def load_topology() -> dict:
    """Read data/external/shn_grid.geojson once and pre-process it for
    plotting. Returns lat/lon arrays for substations and (line, separator)
    sequences ready to drop into a single go.Scattermap trace.

    Returns an empty structure if the GeoJSON file is missing — the dashboard
    should fall back to no-topology mode rather than fail.
    """
    import json
    path = ROOT / 'data' / 'external' / 'shn_grid.geojson'
    out = {'lines_lat': [], 'lines_lon': [],
           'subs_lat':  [], 'subs_lon':  [], 'subs_text': [],
           'n_lines': 0, 'n_subs': 0}
    if not path.exists():
        return out

    gj = json.loads(path.read_text(encoding='utf-8'))
    n_lines = 0
    for f in gj.get('features', []):
        kind = f.get('properties', {}).get('kind')
        if kind == 'line':
            coords = f.get('geometry', {}).get('coordinates') or []
            if len(coords) < 2:
                continue
            for lon, lat in coords:
                out['lines_lat'].append(lat)
                out['lines_lon'].append(lon)
            # `None` value breaks the line, so all 2,898 disjoint segments
            # render in a single trace without becoming one polyline.
            out['lines_lat'].append(None)
            out['lines_lon'].append(None)
            n_lines += 1
        elif kind == 'substation':
            coords = f.get('geometry', {}).get('coordinates') or []
            if len(coords) != 2:
                continue
            lon, lat = coords
            props = f.get('properties', {})
            name = props.get('name') or f"OSM #{props.get('osm_id', '')}"
            voltage = props.get('voltage', '') or '—'
            operator = props.get('operator', '') or '(operator unknown)'
            out['subs_lat'].append(lat)
            out['subs_lon'].append(lon)
            out['subs_text'].append(
                f"<b>{name}</b><br>{operator}<br>voltage: {voltage} V"
            )
    out['n_lines'] = n_lines
    out['n_subs'] = len(out['subs_lat'])
    return out


@st.cache_resource(show_spinner='Indexing line endpoints to nearest towns…')
def line_to_nearest_towns(threshold_km: float = 10.0) -> dict:
    """For every 110 kV line in shn_grid.geojson, find the towns whose centroid
    sits within `threshold_km` of either endpoint.

    Returns:
        {
            'line_coords':   list of full line-coord arrays (lon-lat order),
            'town_names':    list of town names in column order,
            'M':             bool ndarray of shape (n_lines, n_towns),
                             True where the town is near at least one endpoint
                             of the line. Used by `line_intensity_panel` for
                             vectorised per-frame intensity computation.
        }
    """
    import json
    topo = load_topology()
    if topo['n_lines'] == 0:
        return {'line_coords': [], 'town_names': [],
                'M': np.zeros((0, 0), dtype=bool)}

    geo = load_data()['geo']
    geo_rad = np.radians(geo[['lat', 'lon']].values)               # (n_towns, 2)
    town_names = geo['town'].astype(str).tolist()

    path = ROOT / 'data' / 'external' / 'shn_grid.geojson'
    gj = json.loads(path.read_text(encoding='utf-8'))

    line_coords: list[list] = []
    endpoints: list[list[float]] = []          # 2 rows per line (a then b)
    for f in gj['features']:
        if f['properties'].get('kind') != 'line':
            continue
        coords = f['geometry']['coordinates']
        if len(coords) < 2:
            continue
        line_coords.append(coords)
        endpoints.append([coords[0][1],  coords[0][0]])   # endpoint a (lat, lon)
        endpoints.append([coords[-1][1], coords[-1][0]])  # endpoint b

    flat_rad = np.radians(np.array(endpoints))                     # (2*n_lines, 2)
    R_KM = 6371.0
    threshold_rad = threshold_km / R_KM

    # Pairwise haversine: (2*n_lines, n_towns)
    dlat = geo_rad[None, :, 0] - flat_rad[:, None, 0]
    dlon = geo_rad[None, :, 1] - flat_rad[:, None, 1]
    a = (np.sin(dlat / 2) ** 2 +
         np.cos(flat_rad[:, None, 0]) * np.cos(geo_rad[None, :, 0]) *
         np.sin(dlon / 2) ** 2)
    d = 2 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))                   # radians

    near = d < threshold_rad                                       # (2*n_lines, n_towns)
    n_lines = len(line_coords)
    # Combine endpoint a and endpoint b: line is "near town j" if either is.
    M = (near[0::2] | near[1::2])                                  # (n_lines, n_towns)

    return {'line_coords': line_coords, 'town_names': town_names, 'M': M}


# Bucket boundaries: max-intensity threshold (concurrent ops at any nearby town).
# Bucket 0 catches everything below the next threshold.
_BUCKET_EDGES  = [1, 4, 10, 20]                # idle | low | mid | high | critical
_BUCKET_COLORS = [
    'rgba(134, 239, 172, 0.45)',               # idle — light green = "calm grid"
    'rgba( 56, 189, 248, 0.85)',               # low — cyan (clearly distinct from mid)
    'rgba( 59, 130, 246, 0.95)',               # mid — actual blue
    'rgba(255, 127,  14, 0.95)',               # high — warm transition
    'rgba(255,  48,  48, 0.95)',               # critical — red
]
_BUCKET_NAMES  = ['no activity', 'low (1-3)', 'mid (4-9)', 'high (10-19)', 'critical (20+)']

# ---- Substation marker buckets ------------------------------------------
# Static map encodes per-substation **active hours that day** (0-24) — a
# duration metric that complements the lines' instantaneous-peak metric.
# In animation frames the dots fall back to the line bucketing (peak
# concurrent ops at the frame), since "active hours" doesn't make sense
# inside a single hour-frame.
_SUB_BUCKET_EDGES = [1, 5, 10, 16]              # 0 | 1-4 | 5-9 | 10-15 | 16-24
_SUB_BUCKET_COLORS = [
    'rgba(160, 165, 175, 0.55)',                # idle — dim grey
    'rgba( 56, 189, 248, 0.95)',                # 1-4h  — cyan
    'rgba( 59, 130, 246, 0.95)',                # 5-9h  — blue
    'rgba(255, 127,  14, 0.95)',                # 10-15h — orange
    'rgba(255,  48,  48, 0.95)',                # 16-24h — red
]
_SUB_BUCKET_NAMES = ['idle (0h)', 'low (1-4h)', 'mid (5-9h)',
                     'high (10-15h)', 'all-day (16-24h)']
_SUB_DOT_SIZE = 6


def _bucket_of_hours(active_hours: float) -> int:
    """Map a float in [0, 24] to one of the 5 substation-marker buckets."""
    for b, edge in enumerate(_SUB_BUCKET_EDGES):
        if active_hours < edge:
            return b
    return len(_SUB_BUCKET_EDGES)


def _legend_html(*, swatches: list[tuple[str, str]],
                  caption: str,
                  swatch_kind: str = 'line') -> str:
    """Inline HTML legend strip — used above the map for line buckets and
    below the map for substation buckets. `swatch_kind`:
      'line' renders a 18×3 px coloured rectangle (a stylised line segment),
      'dot'  renders an 8 px circle (a marker).
    """
    if swatch_kind == 'dot':
        swatch_style = (
            'display:inline-block; width:9px; height:9px; '
            'border-radius:50%; vertical-align:middle;'
        )
    else:
        swatch_style = (
            'display:inline-block; width:18px; height:3px; '
            'background:{c}; border-radius:1px;'
        )
    items = []
    for c, n in swatches:
        if swatch_kind == 'dot':
            chip = (f"<span style='{swatch_style} background:{c};'></span>")
        else:
            chip = (f"<span style='{swatch_style.format(c=c)}'></span>")
        items.append(
            f"<span style='display:inline-flex; align-items:center; "
            f"margin-right:18px; gap:6px; font-family:JetBrains Mono,monospace; "
            f"font-size:0.78rem; color:{COLOR_TEXT_MUTED};'>"
            f"{chip}&nbsp;{n}</span>"
        )
    return (
        f"<div style='padding:8px 12px; margin: 0 0 8px 0; "
        f"background:{COLOR_SURFACE}; border:1px solid {COLOR_RING}; "
        f"border-radius:8px; line-height:1.6;'>"
        f"<span style='font-family:JetBrains Mono,monospace; "
        f"font-size:0.72rem; letter-spacing:0.08em; "
        f"color:{COLOR_TEXT_MUTED}; text-transform:uppercase; "
        f"margin-right:14px;'>{caption}</span>"
        f"{''.join(items)}"
        f"</div>"
    )


def _line_legend_html() -> str:
    """Top-of-map legend: line color = peak concurrency on nearby substations."""
    return _legend_html(
        swatches=list(zip(_BUCKET_COLORS, _BUCKET_NAMES)),
        caption='Lines · concurrent operations on nearby substations',
        swatch_kind='line',
    )


def _sub_legend_html_static() -> str:
    """Bottom-of-map legend used in static (single-day) view: dot color =
    active hours that day at this substation."""
    return _legend_html(
        swatches=list(zip(_SUB_BUCKET_COLORS, _SUB_BUCKET_NAMES)),
        caption='Substations · active hours today',
        swatch_kind='dot',
    )


def _sub_legend_html_animation() -> str:
    """Bottom-of-map legend used in animation: dot color = peak concurrent
    ops at the substation in this frame. Mirrors the line scheme so dot
    and line colours align by meaning."""
    return _legend_html(
        swatches=list(zip(_BUCKET_COLORS, _BUCKET_NAMES)),
        caption='Substations · peak concurrent ops at this frame',
        swatch_kind='dot',
    )

# Single line width used by every bucket — the green idle lines and the
# colored active lines share the same path with identical thickness, so
# active segments fully replace the green underneath rather than leaving
# a green outline peeking through.
_LINE_WIDTH = 2.4


def _bucket_of(intensity: float) -> int:
    """Map a numeric intensity to one of 5 bucket indices."""
    for b, edge in enumerate(_BUCKET_EDGES):
        if intensity < edge:
            return b
    return len(_BUCKET_EDGES)


@st.cache_data
def _line_intensity_panel_impl(d_lo: pd.Timestamp, d_hi: pd.Timestamp,
                                resolution: str) -> dict:
    """Internal — build per-frame line-bucket coordinate arrays for any
    half-open window [d_lo, d_hi) at the given resolution ('hourly' or
    'daily'). Cached by the public wrappers below."""
    topo = load_topology()
    if topo['n_lines'] == 0:
        return {'frame_labels': [], 'resolution': resolution,
                'static_lats': [], 'static_lons': [], 'frames': {}}

    idx = line_to_nearest_towns(threshold_km=10.0)
    line_coords = idx['line_coords']
    M           = idx['M']                                         # (n_lines, n_towns) bool
    town_names  = idx['town_names']
    n_lines     = M.shape[0]

    wide = load_data()['wide']
    sub = wide.loc[(wide.index >= d_lo) & (wide.index < d_hi)]
    if sub.empty:
        return {'frame_labels': [], 'resolution': resolution,
                'static_lats': [], 'static_lons': [], 'frames': {}}

    if resolution == 'hourly':
        peak = sub.resample('1h').max()
        fmt = '%a %d %b · %H:00'
    else:
        peak = sub.resample('1D').max()
        fmt = '%a %d %b'
    peak.index = peak.index.strftime(fmt)
    peak = peak.reindex(columns=town_names).fillna(0)              # align to M's column order

    # (n_frames, n_towns) matrix of peak concurrency
    P = peak.values.astype(np.float32)                             # (n_frames, n_towns)

    # For each (frame, line) pair, max peak across towns where M is True.
    # Vectorised: broadcast M against each frame's row vector and take max.
    # Done one frame at a time to keep memory bounded; each frame is fast.
    bucket_edges = np.array(_BUCKET_EDGES, dtype=np.float32)

    static_lats = topo['lines_lat']
    static_lons = topo['lines_lon']

    # Substation marker geometry: lat/lon for each town column in the wide
    # matrix, in the same order as `town_names`. Towns missing from the geo
    # parquet are skipped — animation marker arrays will index a subset.
    geo = load_data()['geo'].drop_duplicates(subset='town').set_index('town')
    sub_lats: list[float] = []
    sub_lons: list[float] = []
    sub_indices: list[int] = []                                    # column index into P
    for ti, name in enumerate(town_names):
        if name in geo.index:
            sub_lats.append(float(geo.at[name, 'lat']))
            sub_lons.append(float(geo.at[name, 'lon']))
            sub_indices.append(ti)
    sub_idx_arr = np.asarray(sub_indices, dtype=np.int32)

    frames: dict[str, list[tuple[list, list]]] = {}
    frame_marker_colors: dict[str, list[str]] = {}
    for ts_label, row in zip(peak.index, P):
        # ----- line intensities -----
        intensity = (M.astype(np.float32) * row).max(axis=1)       # (n_lines,)
        buckets = np.searchsorted(bucket_edges, intensity, side='right')

        bucket_lats = [[] for _ in range(5)]
        bucket_lons = [[] for _ in range(5)]
        for li in np.where(buckets > 0)[0]:                        # skip idle
            b = int(buckets[li])
            for lon, lat in line_coords[li]:
                bucket_lats[b].append(lat)
                bucket_lons[b].append(lon)
            bucket_lats[b].append(None)
            bucket_lons[b].append(None)
        frames[ts_label] = list(zip(bucket_lats, bucket_lons))

        # ----- per-substation marker colors -----
        # Same metric + bucketing as the lines (peak concurrent ops at the
        # frame's window). Active hours doesn't translate cleanly to a
        # single hour-frame, so we reuse the line scheme for animation.
        sub_intensity = row[sub_idx_arr]                           # (n_subs_with_geo,)
        sub_buckets   = np.searchsorted(bucket_edges, sub_intensity, side='right')
        frame_marker_colors[ts_label] = [
            _BUCKET_COLORS[int(b)] for b in sub_buckets
        ]

    return {
        'frame_labels':         list(peak.index),
        'resolution':           resolution,
        'static_lats':          static_lats,
        'static_lons':          static_lons,
        'frames':               frames,
        'sub_lats':             sub_lats,
        'sub_lons':             sub_lons,
        'frame_marker_colors':  frame_marker_colors,
    }


def line_intensity_panel(end_date_str: str, days_back: int) -> dict:
    """Public wrapper — 'Last N days' window with auto resolution.
    Hourly for ≤14 days, daily otherwise."""
    d_hi = pd.Timestamp(end_date_str).normalize() + pd.Timedelta(days=1)
    d_lo = d_hi - pd.Timedelta(days=days_back)
    resolution = 'hourly' if days_back <= 14 else 'daily'
    return _line_intensity_panel_impl(d_lo, d_hi, resolution)


def line_intensity_panel_window(start_date_str: str, end_date_str: str,
                                 resolution: str) -> dict:
    """Public wrapper — explicit [start_date, end_date] window with the
    user's chosen 'hourly' or 'daily' granularity. Used by the
    'Custom window animation' mode."""
    d_lo = pd.Timestamp(start_date_str).normalize()
    d_hi = pd.Timestamp(end_date_str).normalize() + pd.Timedelta(days=1)
    return _line_intensity_panel_impl(d_lo, d_hi, resolution)


@st.cache_data
def line_buckets_for_day(date_str: str) -> list[tuple[list, list]]:
    """Single-day version of line_intensity_panel. Returns 5 (lats, lons)
    tuples — one per intensity bucket — using the day's peak concurrent-op
    count per town as the line-intensity proxy.

    Bucket 0 (idle) entries are not included in the returned coord arrays;
    the static green-line backdrop already covers all idle lines.
    """
    idx = line_to_nearest_towns(threshold_km=10.0)
    M           = idx['M']                                         # bool (n_lines, n_towns)
    line_coords = idx['line_coords']
    town_names  = idx['town_names']

    wide = load_data()['wide']
    d    = pd.Timestamp(date_str).normalize()
    nxt  = d + pd.Timedelta(days=1)
    day  = wide.loc[(wide.index >= d) & (wide.index < nxt)]

    bucket_lats = [[] for _ in range(5)]
    bucket_lons = [[] for _ in range(5)]
    if day.empty or M.size == 0:
        return list(zip(bucket_lats, bucket_lons))

    peak = (day.max(axis=0)
              .reindex(town_names).fillna(0)
              .values.astype(np.float32))                          # (n_towns,)
    intensity = (M.astype(np.float32) * peak).max(axis=1)          # (n_lines,)
    edges = np.array(_BUCKET_EDGES, dtype=np.float32)
    buckets = np.searchsorted(edges, intensity, side='right')

    for li in np.where(buckets > 0)[0]:
        b = int(buckets[li])
        for lon, lat in line_coords[li]:
            bucket_lats[b].append(lat)
            bucket_lons[b].append(lon)
        bucket_lats[b].append(None)
        bucket_lons[b].append(None)
    return list(zip(bucket_lats, bucket_lons))


# ----------------------------------------------------------------------
# Driver attribution (TreeSHAP on the production LightGBM, grouped into
# six operator-friendly families). See src/driver_attribution.py for math.
# ----------------------------------------------------------------------
from driver_attribution import (                                  # noqa: E402
    GROUP_ORDER, GROUP_DESCRIPTIONS,
    assign_groups, decompose_day, top_features,
)

PRED_DIR = ROOT / 'data' / 'predictions'
MODELS_DIR = ROOT / 'models'


@st.cache_data(show_spinner=False)
def typical_day_volume() -> int:
    """Median town-hours-per-day across history. Used as the headline anchor
    for 'today's events vs a typical day'."""
    wide = load_data()['wide']
    daily = (wide > 0).sum(axis=1).resample('D').sum() / 4   # 15-min slots → hours
    return int(round(float(daily.median())))


@st.cache_data(show_spinner=False)
def _daily_volume_distribution() -> pd.Series:
    """All historical daily totals (town-hours), used to compute today's
    percentile rank for the headline. One number per day."""
    wide = load_data()['wide']
    return ((wide > 0).sum(axis=1).resample('D').sum() / 4).astype('float32')


def day_percentile(grid_hours: int) -> int:
    """Today's volume as a percentile of the historical daily distribution.
    Returns 0..100. Used for the headline classifier so users see 'top 15%'
    instead of an arbitrary 100/300 cutoff."""
    dist = _daily_volume_distribution()
    if dist.empty:
        return 0
    return int(round(100.0 * float((dist <= grid_hours).mean())))


@st.cache_resource(show_spinner=False)
def _feature_groups() -> dict[str, str]:
    """Cached group assignment for the model's 198 features."""
    import json
    fcols = json.loads((MODELS_DIR / 'feature_cols.json').read_text())
    return assign_groups(fcols)


@st.cache_resource(show_spinner=False)
def _calibrator_y24h():
    """Lazily load the 24h isotonic calibrator. Returns None on any failure."""
    try:
        import joblib
        return joblib.load(MODELS_DIR / 'calibrator_y_24h.joblib')
    except Exception:
        return None


@st.cache_data(show_spinner=False)
def _attribution_window() -> tuple[str, str] | None:
    """Min/max date for which the dashboard has driver attribution.

    Resolution order — picks whichever is more authoritative for what the
    UI can actually render:
      1. Daily summary parquet (always shipped in git).
      2. features.parquet (local dev / cron runner only).
    """
    summary_path = PRED_DIR / 'contributions_daily_summary.parquet'
    if summary_path.exists():
        try:
            df = pd.read_parquet(summary_path, columns=['date'])
            if not df.empty:
                return (str(df['date'].min()), str(df['date'].max()))
        except Exception:
            pass
    fpath = ROOT / 'data' / 'processed' / 'features.parquet'
    if fpath.exists():
        try:
            ts = pd.read_parquet(fpath, columns=['ts'])['ts']
            return (str(ts.min().date()), str(ts.max().date()))
        except Exception:
            pass
    return None


@st.cache_data(show_spinner=False)
def _load_daily_summary() -> pd.DataFrame | None:
    """Cached read of the rolled-up daily summary parquet (one row per day,
    six group means + bias). This is the primary data source for the
    dashboard's per-group bars; small enough to ship in git."""
    path = PRED_DIR / 'contributions_daily_summary.parquet'
    if not path.exists():
        return None
    df = pd.read_parquet(path)
    return df


def _attribution_from_summary_row(row: pd.Series) -> dict:
    """Build the dashboard's `attribution` record from a single summary row.
    Loses the per-feature top-25 table (only available when the full per-day
    parquet is on disk), but keeps the bar-chart-level data."""
    groups_loaded = _feature_groups()      # only used for membership lookups
    mean_bias = float(row['mean_bias'])
    raw_score = mean_bias + sum(float(row[g]) for g in GROUP_ORDER)

    def _sigmoid(x: float) -> float:
        import math
        return 1.0 / (1.0 + math.exp(-x))

    cal = _calibrator_y24h()
    def _calibrate(logodds: float) -> float:
        p = _sigmoid(logodds)
        if cal is not None:
            try:
                return float(np.atleast_1d(cal.predict([p]))[0])
            except Exception:
                pass
        return p

    baseline_p_cal = _calibrate(mean_bias)
    final_p_cal    = _calibrate(raw_score)

    effects = []
    for g in GROUP_ORDER:
        dl = float(row[g])
        # Raw-sigmoid marginal pp (signal-preserving, like the live path)
        p_with = _sigmoid(mean_bias + dl)
        p_base = _sigmoid(mean_bias)
        effects.append({
            'group':         g,
            'delta_p_pp':    (p_with - p_base) * 100.0,
            'delta_logodds': dl,
            'share':         0.0,
        })
    effects.sort(key=lambda r: abs(r['delta_logodds']), reverse=True)

    return {
        'date':            str(row['date']),
        'n_rows':          int(row['n_rows']),
        'baseline_p_cal':  baseline_p_cal,
        'final_p_cal':     final_p_cal,
        'baseline_p_raw':  _sigmoid(mean_bias),
        'final_p_raw':     _sigmoid(raw_score),
        'group_effects':   effects,
        'top_features':    None,         # not available from summary
    }


@st.cache_data(show_spinner=False)
def load_today_attribution(date_str: str) -> dict | None:
    """Load attribution for `date_str`.

    Resolution order:
      1. Full per-day contributions parquet (data/predictions/contributions_<date>.parquet)
         — gives per-feature top-25 table for the technical expander. Local
         only; not in git.
      2. Rolled-up daily summary (contributions_daily_summary.parquet) — six
         group means per day. Always present in the repo.
      3. None — neither source has this date; UI shows fallback message.
    """
    full_path = PRED_DIR / f'contributions_{date_str}.parquet'
    if full_path.exists():
        contribs = pd.read_parquet(full_path)
        if not contribs.empty:
            feat_and_bias = [c for c in contribs.columns if c not in ('ts', 'town')]
            groups = _feature_groups()
            cal    = _calibrator_y24h()
            cal_result = decompose_day(contribs[feat_and_bias], groups, cal)
            raw_result = decompose_day(contribs[feat_and_bias], groups, None)
            feat_only  = [c for c in feat_and_bias if c != 'bias']
            top = top_features(contribs[['bias'] + feat_only], n=5)
            return {
                'date':            date_str,
                'n_rows':          cal_result['n_rows'],
                'baseline_p_cal':  cal_result['baseline_p'],
                'final_p_cal':     cal_result['final_p'],
                'baseline_p_raw':  raw_result['baseline_p'],
                'final_p_raw':     raw_result['final_p'],
                'group_effects':   raw_result['group_effects'],
                'top_features':    top,
            }

    # Fallback: daily summary
    summary = _load_daily_summary()
    if summary is None:
        return None
    matches = summary[summary['date'] == date_str]
    if matches.empty:
        return None
    return _attribution_from_summary_row(matches.iloc[0])


@st.cache_resource(show_spinner='Indexing substations from raw ops…')
def substation_index() -> pd.DataFrame:
    """One-time read of all raw chunks to build a (town, locationBottleneck)
    summary. Cached for the lifetime of the Streamlit process — recomputes
    only when the app restarts (which is what the daily refresh triggers).

    Columns:
        town
        locationBottleneck   transformer ID (e.g. '1000346078-T122')
        n_ops                total operations on this bottleneck
        n_neng_ops           operations with reason in {Netzengpass, Netzengpass I}
        first_seen           earliest start
        last_seen            latest start
    """
    raw_dir = ROOT / 'data' / 'raw' / 'shn_operations_last_2y'
    files = sorted(raw_dir.glob('chunk_*.parquet'))
    if not files:
        return pd.DataFrame(columns=['town', 'locationBottleneck',
                                      'n_ops', 'n_neng_ops',
                                      'first_seen', 'last_seen'])

    frames = []
    for f in files:
        df = pd.read_parquet(f, columns=['location', 'locationBottleneck',
                                          'reason', 'start'])
        frames.append(df)
    raw = pd.concat(frames, ignore_index=True)
    raw['start'] = pd.to_datetime(raw['start'], format='ISO8601')

    # Match build_timeseries.py town normalisation: strip 'UW ' prefix and split on commas.
    raw = raw[raw['location'].notna() & (raw['location'] != 'UW')]
    raw['town_raw'] = raw['location'].str.replace(r'^UW\s+', '', regex=True).str.strip()
    raw['town_list'] = raw['town_raw'].str.split(',').apply(
        lambda xs: [t.replace('UW ', '').strip() for t in xs] if isinstance(xs, list) else []
    )
    raw = raw.explode('town_list').rename(columns={'town_list': 'town'})
    raw = raw[raw['town'].astype(str).str.len() > 0]

    raw['is_neng'] = raw['reason'].isin(['Netzengpass', 'Netzengpass I']).astype('int32')
    raw['locationBottleneck'] = raw['locationBottleneck'].fillna('(unspecified)')

    grouped = (raw.groupby(['town', 'locationBottleneck'], observed=True)
                  .agg(n_ops=('reason', 'size'),
                       n_neng_ops=('is_neng', 'sum'),
                       first_seen=('start', 'min'),
                       last_seen=('start', 'max'))
                  .reset_index())
    return grouped


@st.cache_data
def town_substation_breakdown(town: str, top_n: int = 10) -> pd.DataFrame:
    """Top-N transformers within a town, ranked by Netzengpass operation count."""
    idx = substation_index()
    sub = idx[idx['town'] == town].copy()
    if sub.empty:
        return sub
    sub = sub.sort_values(['n_neng_ops', 'n_ops'], ascending=False).head(top_n)
    return sub.reset_index(drop=True)


@st.cache_data
def town_history(town: str, end_date_str: str, days_back: int = 90) -> pd.DataFrame:
    """Daily active hours for `town`, ending on `end_date_str`, going back N days."""
    d_hi = pd.Timestamp(end_date_str).normalize() + pd.Timedelta(days=1)
    d_lo = d_hi - pd.Timedelta(days=days_back)
    wide = load_data()['wide']
    if town not in wide.columns:
        return pd.DataFrame()
    s = (wide[town] > 0).astype('int8')
    s = s.loc[(s.index >= d_lo) & (s.index < d_hi)]
    if s.empty:
        return pd.DataFrame()
    hourly = s.copy(); hourly.index = hourly.index.floor('h')
    by_hour = hourly.groupby(level=0).max()
    by_day = (by_hour.groupby(by_hour.index.date).sum()
                     .rename('active_hours')
                     .reset_index().rename(columns={'index': 'date'}))
    by_day['date']      = pd.to_datetime(by_day['date'])
    by_day['rolling_7'] = by_day['active_hours'].rolling(7, min_periods=1).mean()
    return by_day


@st.cache_data
def town_hour_heatmap(town: str, end_date_str: str, days_back: int = 7) -> pd.DataFrame:
    """7-day x 24-hour activity heatmap for one town. Cell = # of active 15-min slots
    in that hour (0-4)."""
    d_hi = pd.Timestamp(end_date_str).normalize() + pd.Timedelta(days=1)
    d_lo = d_hi - pd.Timedelta(days=days_back)
    wide = load_data()['wide']
    if town not in wide.columns:
        return pd.DataFrame()
    s = (wide[town] > 0).astype('int8')
    s = s.loc[(s.index >= d_lo) & (s.index < d_hi)]
    if s.empty:
        return pd.DataFrame()
    df = s.reset_index(); df.columns = ['ts', 'active']
    df['date'] = df['ts'].dt.date
    df['hour'] = df['ts'].dt.hour
    pivot = df.groupby(['date', 'hour'])['active'].sum().unstack('hour')
    pivot = pivot.reindex(columns=range(24), fill_value=0)
    return pivot


# ---------------------- preflight ----------------------
if not WIDE_PATH.exists():
    st.error(
        f'Missing {WIDE_PATH.relative_to(ROOT)}. '
        f'Run `python src/build_timeseries.py` first to generate the activity matrix.'
    )
    st.stop()

try:
    data = load_data()
except Exception as exc:
    st.error(str(exc))
    st.stop()
all_towns = sorted(data['wide'].columns.astype(str))
date_lo = data['wide'].index.min().date()
date_hi = data['wide'].index.max().date()


# ---------------------- header ----------------------
st.markdown(
    "<h1 style='margin-bottom:8px'>Redispatch in Schleswig-Holstein</h1>"
    f"<p style='color:{COLOR_TEXT_MUTED}; font-size:1.05rem; margin-top:0'>"
    "Where and when the SHN grid has had to step in. "
    f"175 substations · {date_lo} → {date_hi}."
    "</p>",
    unsafe_allow_html=True,
)

with st.expander('What is a redispatch event?', expanded=True):
    st.markdown(
        'When wind or solar farms generate more power than the local '
        'transmission lines can move out of the region, the grid operator '
        'orders selected plants to **reduce or shift their output** so the '
        'lines do not overload. Each such instruction is one *redispatch '
        'event*. In Schleswig-Holstein this happens often along the windy '
        'North Sea coast.\n\n'
        '**This dashboard shows:** for any day, which 110 kV substations '
        'were congested and for how many hours, what conditions drove that '
        'day, and how individual towns are trending over the past 90 days.'
    )

# ---------------------- tabs ----------------------
tab_map, tab_town = st.tabs(['Today', 'By town'])

# =============================================================
# TAB 1 — Daily map
# =============================================================
with tab_map:
    # Three-way segmented control:
    #   1. Single day                — pick a historical date, see one map.
    #   2. Recent days animation     — quick presets (last 3/7/14/30/90 days).
    #   3. Custom window animation   — pick start, end, and granularity.
    _MODE_LABELS = ['Single day', 'Recent days animation', 'Custom window animation']
    if hasattr(st, 'segmented_control'):
        _mode = st.segmented_control(
            'View',
            options=_MODE_LABELS,
            default=_MODE_LABELS[0],
            label_visibility='collapsed',
            help='Single day: pick any historical date. '
                 'Recent days animation: quick last-N-days presets. '
                 'Custom window animation: pick exact start/end + granularity.',
        )
    else:
        _mode = st.radio(
            'View',
            options=_MODE_LABELS,
            horizontal=True,
            label_visibility='collapsed',
        )
    animation_mode = (_mode in (_MODE_LABELS[1], _MODE_LABELS[2]))
    custom_window_mode = (_mode == _MODE_LABELS[2])

    if animation_mode:
        if custom_window_mode:
            # User-controlled window + granularity. Defaults: last 14 days,
            # hourly — large enough to be interesting, small enough to load fast.
            cw1, cw2, cw3 = st.columns([1, 1, 1])
            with cw1:
                anim_start = st.date_input(
                    'Start date',
                    value=pd.Timestamp(date_hi).date() - pd.Timedelta(days=14),
                    min_value=date_lo, max_value=date_hi, key='anim_start',
                    help='First date in the animation window.',
                )
            with cw2:
                anim_end = st.date_input(
                    'End date',
                    value=pd.Timestamp(date_hi).date(),
                    min_value=date_lo, max_value=date_hi, key='anim_end',
                    help='Last date in the animation window (inclusive).',
                )
            with cw3:
                anim_resolution = st.selectbox(
                    'Granularity',
                    options=['hourly', 'daily'],
                    index=0,
                    key='anim_resolution',
                    help='Hourly: 24 frames per day. Daily: 1 frame per day. '
                         'Pick daily for long windows so the animation stays smooth.',
                )

            if anim_start > anim_end:
                st.error('Start date must be on or before end date.')
                st.stop()

            # Soft warning when the user picks a heavy combo. 30 days × 24h = 720
            # frames is borderline; 90 days × 24h = 2160 frames will be slow.
            n_days_picked = (anim_end - anim_start).days + 1
            if anim_resolution == 'hourly' and n_days_picked > 30:
                st.warning(
                    f'Hourly × {n_days_picked} days = {n_days_picked * 24:,} frames. '
                    f'This may take a while to render — switch to daily for a smoother experience.'
                )

            with st.spinner('Pre-computing line intensities for every frame…'):
                panel = line_intensity_panel_window(
                    str(anim_start), str(anim_end), anim_resolution,
                )
            if not panel['frame_labels']:
                st.error(f'No data between {anim_start} and {anim_end}.')
                st.stop()
            window_label = f'{anim_start.strftime("%d %b %Y")} → {anim_end.strftime("%d %b %Y")}'
        else:
            anim_days = st.selectbox(
                'Window',
                options=[3, 7, 14, 30, 90],
                index=1,
                format_func=lambda d: f'Last {d} days',
                help='Resolution auto-selects: hourly for ≤14 days, daily beyond. '
                     'Longer windows mean more frames; the 90-day view runs at '
                     'daily granularity to stay smooth.',
            )

            with st.spinner('Pre-computing line intensities for every frame…'):
                panel = line_intensity_panel(str(date_hi), days_back=anim_days)
            if not panel['frame_labels']:
                st.error(f'No data in the last {anim_days} days.')
                st.stop()
            window_label = f'Last {anim_days} days'

        n_frames = len(panel['frame_labels'])
        st.markdown(
            f"<div style='padding:18px 22px; border-radius:14px; "
            f"background-color:{COLOR_SURFACE}; border:1px solid {COLOR_RING}; "
            f"margin: 8px 0 18px 0;'>"
            f"<div style='font-size:0.78rem; color:{COLOR_TEXT_MUTED}; "
            f"text-transform:uppercase; letter-spacing:0.08em; margin-bottom:4px'>"
            f"ANIMATION · {window_label} · {panel['resolution']}</div>"
            f"<div style='font-size:1.4rem; font-family:'Inter',sans-serif; "
            f"font-weight:400; letter-spacing:-0.025em; line-height:1.2'>"
            f"Watch the 110 kV grid heat up and cool down</div>"
            f"<div style='color:{COLOR_TEXT_MUTED}; margin-top:6px; font-size:0.92rem'>"
            f"<b style='color:{COLOR_TEXT}'>{n_frames}</b> {panel['resolution']} frames · "
            f"line color encodes the highest concurrent-op count among "
            f"substations the line connects to. "
            f"Press play below the map; drag the slider to jump."
            f"</div></div>",
            unsafe_allow_html=True,
        )
    else:
        date = st.date_input(
            'Date',
            value=pd.Timestamp(date_hi).date(),
            min_value=date_lo, max_value=date_hi,
            key='sel_date',
        )
        # Threshold slider used to highlight bubbles; with line-only viz the
        # KPI strip just reports counts at fixed thresholds (4h alert, 0h any).
        threshold = 4

        df = daily_hours(str(date))
        if df.empty:
            st.error(f'No data on {date}. Pick a date between {date_lo} and {date_hi}.')
            st.stop()

        n_alerts     = int((df['active_hours'] >= threshold).sum())
        n_any        = int((df['active_hours'] > 0).sum())
        grid_hours   = int(df['active_hours'].sum())
        busiest      = df.loc[df['active_hours'].idxmax()]

        # Classifier driven by percentile rank, not arbitrary cutoffs. Reads
        # naturally as 'busier than X% of historical days'.
        pct_rank = day_percentile(grid_hours)
        if grid_hours == 0:
            weather_word, headline_color = 'quiet', '#2ca02c'
            rank_phrase = 'no redispatch anywhere'
        elif pct_rank < 25:
            weather_word, headline_color = 'calm', '#2ca02c'
            rank_phrase = f'calmer than {100 - pct_rank}% of days in 2024–25'
        elif pct_rank < 75:
            weather_word, headline_color = 'normal', COLOR_TEXT
            rank_phrase = f'middle of the pack — busier than {pct_rank}% of days'
        elif pct_rank < 90:
            weather_word, headline_color = 'busy', '#ff7f0e'
            rank_phrase = f'top {100 - pct_rank}% of days in 2024–25'
        else:
            weather_word, headline_color = 'very busy', '#ff3030'
            rank_phrase = f'top {max(1, 100 - pct_rank)}% of days in 2024–25'

        st.markdown(
            f"<div style='padding:18px 22px; border-radius:14px; "
            f"background-color:{COLOR_SURFACE}; border:1px solid {COLOR_RING}; "
            f"margin: 8px 0 18px 0;'>"
            f"<div style='font-size:0.78rem; color:{COLOR_TEXT_MUTED}; "
            f"text-transform:uppercase; letter-spacing:0.08em; margin-bottom:4px'>"
            f"DAY · {date.strftime('%a %d %b %Y')}</div>"
            f"<div style='font-size:1.4rem; font-family:Space Grotesk,Inter,sans-serif; "
            f"font-weight:600; letter-spacing:-0.025em; line-height:1.2'>"
            f"<span style='color:{headline_color}'>It was a {weather_word} day</span> "
            f"<span style='color:{COLOR_TEXT_MUTED}'>·</span> "
            f"<span style='color:{COLOR_TEXT}'>{n_any} of 175 substations had redispatch, "
            f"{grid_hours} hours of redispatch grid-wide</span></div>"
            f"<div style='color:{COLOR_TEXT_MUTED}; margin-top:6px; font-size:0.92rem'>"
            f"<span style='color:{COLOR_TEXT_MUTED}'>{rank_phrase}.</span> "
            f"Busiest substation: <b style='color:{COLOR_TEXT}'>{busiest['town']}</b> "
            f"with <b style='color:{COLOR_TEXT}'>{int(busiest['active_hours'])} active hours</b>. "
            f"<b style='color:{COLOR_TEXT}'>{n_alerts}</b> substations were congested for "
            f"≥{threshold} hours."
            f"</div></div>",
            unsafe_allow_html=True,
        )

    if not animation_mode:
        k1, k2, k3, k4 = st.columns(4)
        k1.metric(f'Above {threshold}h', f"{n_alerts}",
                  help='Substations active for at least 4 hours today.')
        k2.metric('Any redispatch today', f"{n_any}",
                  help='Substations with at least one redispatch event today.')
        k3.metric('Most active town', f"{int(busiest['active_hours'])}h", busiest['town'])
        k4.metric('Hours of redispatch (grid-wide)', f"{grid_hours}",
                  help='Sum of active hours across all 175 substations. '
                       '1 = one substation in redispatch for one hour.')

    # In animation mode the data shape is (ts × town × peak_concurrency); the
    # static map uses (town × active_hours × peak_concurrency) from daily_hours.
    # Each branch builds its own DataFrame for the figure to keep things clean.
    if animation_mode:
        # Animation is now line-only: every 110 kV line is rendered, coloured
        # by the peak concurrent-op count among nearby active towns in that
        # frame. No bubbles. Five colour buckets (idle / low / mid / high /
        # critical) mean we use 5 line traces; per-frame we update the
        # coordinate arrays of the four non-idle traces, while the dim
        # backdrop (idle bucket) stays static.
        labels = panel['frame_labels']

        # --- Build base traces ---
        # Trace 0 — static idle backdrop covering all 110 kV lines.
        traces = [go.Scattermap(
            lat=panel['static_lats'], lon=panel['static_lons'],
            mode='lines',
            line=dict(width=_LINE_WIDTH, color=_BUCKET_COLORS[0]),
            hoverinfo='skip', showlegend=True, name=_BUCKET_NAMES[0],
        )]
        # Traces 1..4 — initial frame's bucket coords.
        first_buckets = panel['frames'][labels[0]]
        for i in range(1, 5):
            lats_i, lons_i = first_buckets[i]
            traces.append(go.Scattermap(
                lat=lats_i, lon=lons_i,
                mode='lines',
                line=dict(width=_LINE_WIDTH, color=_BUCKET_COLORS[i]),
                hoverinfo='skip', showlegend=True, name=_BUCKET_NAMES[i],
            ))
        # Trace 5 — substation markers. Coords are constant; only marker
        # colors update per frame.
        first_marker_colors = panel['frame_marker_colors'][labels[0]]
        traces.append(go.Scattermap(
            lat=panel['sub_lats'], lon=panel['sub_lons'],
            mode='markers',
            marker=dict(size=_SUB_DOT_SIZE, color=first_marker_colors,
                        allowoverlap=True),
            hoverinfo='skip', showlegend=False, name='substations',
        ))

        # --- Build per-frame data ---
        plotly_frames = []
        for label in labels:
            buckets = panel['frames'][label]
            frame_traces = []
            for i in range(1, 5):
                lats_i, lons_i = buckets[i]
                frame_traces.append(go.Scattermap(
                    lat=lats_i, lon=lons_i,
                    mode='lines',
                    line=dict(width=_LINE_WIDTH, color=_BUCKET_COLORS[i]),
                    hoverinfo='skip', showlegend=False,
                ))
            # Substation markers — only colors change per frame; coords
            # stay the same so we re-supply them (Plotly expects the trace
            # data to be self-contained).
            frame_traces.append(go.Scattermap(
                lat=panel['sub_lats'], lon=panel['sub_lons'],
                mode='markers',
                marker=dict(size=_SUB_DOT_SIZE,
                            color=panel['frame_marker_colors'][label],
                            allowoverlap=True),
                hoverinfo='skip', showlegend=False,
            ))
            plotly_frames.append(go.Frame(
                data=frame_traces,
                name=label,
                # Update line buckets 1..4 + substation markers (trace 5);
                # the green idle line backdrop (trace 0) stays put.
                traces=[1, 2, 3, 4, 5],
            ))

        fig_map = go.Figure(data=traces, frames=plotly_frames)
        fig_map.update_layout(
            map=dict(style='carto-darkmatter',
                     center=dict(lat=54.3, lon=9.7), zoom=7.0),
            height=620,
            margin=dict(l=0, r=0, t=10, b=0),
            # In-map Plotly legend is suppressed — we render an explicit
            # legend strip above the map via _legend_html() instead.
            showlegend=False,
            # Two separate updatemenus (rather than one with two buttons) —
            # otherwise Plotly stacks Play and Pause at the same x and they
            # render on top of each other in some viewports.
            updatemenus=[
                {
                    'type': 'buttons', 'showactive': False,
                    'x': 0.01, 'y': -0.05, 'xanchor': 'left', 'yanchor': 'top',
                    'pad': {'t': 0, 'r': 6},
                    'buttons': [
                        {'label': '▶  Play', 'method': 'animate',
                         'args': [None, {'frame': {'duration': 350, 'redraw': True},
                                         'fromcurrent': True,
                                         'transition': {'duration': 0}}]},
                    ],
                },
                {
                    'type': 'buttons', 'showactive': False,
                    'x': 0.07, 'y': -0.05, 'xanchor': 'left', 'yanchor': 'top',
                    'pad': {'t': 0, 'r': 6},
                    'buttons': [
                        {'label': '❚❚  Pause', 'method': 'animate',
                         'args': [[None], {'frame': {'duration': 0, 'redraw': False},
                                           'mode': 'immediate',
                                           'transition': {'duration': 0}}]},
                    ],
                },
            ],
            sliders=[{
                'active': 0,
                'currentvalue': {
                    'prefix': '',
                    'visible': True,
                    'xanchor': 'left',
                    # Bigger + brighter so the active frame's date is
                    # legible against the map. JetBrains Mono is the
                    # numeric font of the theme.
                    'font': dict(family='JetBrains Mono, monospace',
                                 color=COLOR_TEXT, size=15),
                    'offset': 8,
                },
                # Slider track styling — explicit colors so the rail and
                # progress fill don't blend into the map's dark backdrop.
                'bgcolor':           'rgba(255,255,255,0.12)',
                'bordercolor':       'rgba(255,255,255,0.30)',
                'borderwidth':       1,
                'tickcolor':         'rgba(255,255,255,0.45)',
                'ticklen':           5,
                'tickwidth':         1,
                'font':              dict(family='JetBrains Mono, monospace',
                                          color=COLOR_TEXT_MUTED, size=10),
                'minorticklen':      2,
                # More headroom above the slider track so the currentvalue
                # text doesn't collide with the map; a touch more below for
                # the tick labels.
                'pad': {'t': 60, 'b': 10},
                'x': 0.12, 'len': 0.85,
                'transition': {'duration': 0},
                # Ticks: when there are many frames, only label every Nth so
                # the x-axis doesn't smush into unreadable mush. Other frames
                # still exist (slider passes through them); their dates show
                # in the bigger currentvalue field above the track.
                'steps': [
                    {'method': 'animate',
                     'label': (lbl if len(labels) <= 14
                               or i % max(1, len(labels) // 10) == 0
                               else ''),
                     'args': [[lbl], {'frame': {'duration': 0, 'redraw': True},
                                       'mode': 'immediate',
                                       'transition': {'duration': 0}}]}
                    for i, lbl in enumerate(labels)
                ],
            }],
        )
    else:
        # Static daily map: line-only intensity heat for one day. Same visual
        # vocabulary as the animation, frozen on a single date.
        topo = load_topology()
        day_buckets = line_buckets_for_day(str(date))

        # Build 5 line traces: idle backdrop (all lines, light green) + 4
        # colored buckets for the day's activity.
        traces = [go.Scattermap(
            lat=topo['lines_lat'], lon=topo['lines_lon'],
            mode='lines',
            line=dict(width=_LINE_WIDTH, color=_BUCKET_COLORS[0]),
            hoverinfo='skip', showlegend=True, name=_BUCKET_NAMES[0],
        )]
        for i in range(1, 5):
            lats_i, lons_i = day_buckets[i]
            traces.append(go.Scattermap(
                lat=lats_i, lon=lons_i,
                mode='lines',
                line=dict(width=_LINE_WIDTH, color=_BUCKET_COLORS[i]),
                hoverinfo='skip', showlegend=True, name=_BUCKET_NAMES[i],
            ))
        # Substation activity dots — one per SHN substation that has a
        # geocode in towns_geo.parquet, colored by **active hours that day**
        # (0–24) rather than line-style concurrent ops. Idle substations
        # render as dim grey for spatial reference; busy ones light up.
        sub_df = df.dropna(subset=['lat', 'lon']).copy()
        if not sub_df.empty:
            sub_df['_bucket'] = sub_df['active_hours'].apply(_bucket_of_hours)
            sub_df['_color']  = sub_df['_bucket'].apply(
                lambda b: _SUB_BUCKET_COLORS[int(b)]
            )
            sub_df['_hover']  = sub_df.apply(
                lambda r: f"{r['town']}<br>{int(r['active_hours'])}h active",
                axis=1,
            )
            traces.append(go.Scattermap(
                lat=sub_df['lat'].tolist(),
                lon=sub_df['lon'].tolist(),
                mode='markers',
                marker=dict(size=_SUB_DOT_SIZE,
                            color=sub_df['_color'].tolist(),
                            allowoverlap=True),
                text=sub_df['_hover'].tolist(),
                hoverinfo='text',
                showlegend=False, name='substations',
            ))

        fig_map = go.Figure(data=traces)
        fig_map.update_layout(
            map=dict(style='carto-darkmatter',
                     center=dict(lat=54.3, lon=9.7), zoom=7.0),
            height=620,
            margin=dict(l=0, r=0, t=10, b=0),
            # In-map Plotly legend is suppressed — we render an explicit
            # legend strip above the map via _legend_html() instead.
            showlegend=False,
        )

        # If the day was completely quiet, drop a centred annotation so it
        # doesn't look like the page failed to load.
        if grid_hours == 0:
            fig_map.add_annotation(
                text='NO REDISPATCH ON THIS DAY ACROSS THE SHN GRID',
                xref='paper', yref='paper', x=0.5, y=0.5, showarrow=False,
                font=dict(family='JetBrains Mono, monospace',
                          size=12, color='rgba(255,255,255,0.70)'),
                bgcolor='rgba(42, 45, 53, 0.85)',
                bordercolor='rgba(255,255,255,0.20)',
                borderwidth=1, borderpad=10,
            )

    # Legend goes above the map so first-time users can decode the colors
    # without hovering anything.
    st.markdown(_line_legend_html(), unsafe_allow_html=True)
    st.plotly_chart(fig_map, use_container_width=True)
    # Substation-marker legend goes below — a separate scale is needed
    # because dots encode different metrics depending on mode.
    if animation_mode:
        st.markdown(_sub_legend_html_animation(), unsafe_allow_html=True)
    else:
        st.markdown(_sub_legend_html_static(), unsafe_allow_html=True)

    # ============================================================
    # Below-the-map: "Why did today look this way?"
    # ============================================================
    # Retrospective attribution via TreeSHAP on the production LightGBM.
    # Static-mode only — in animation the per-frame story IS the explanation.
    if not animation_mode:
      attribution = load_today_attribution(str(date))

      # Surface card framing — visually anchors the Why? section so users
      # don't mistake the rich attribution content for a footnote. Using
      # st.container(border=True) is Streamlit-native: it produces one
      # bordered DOM element that wraps everything inside the `with`
      # block. Raw HTML wrappers don't work here because each st.X call
      # is sandboxed in its own element-container.
      with st.container(border=True):
        st.markdown('### Why did today look this way?')

        if attribution is None:
            window = _attribution_window()
            if window is None:
                st.info(f"Not available for **{date}**.", icon='ℹ️')
            else:
                w_hi = window[1]
                st.info(
                    f"Not available for **{date}** — try **{w_hi}**, the "
                    f"latest date with driver attribution.",
                    icon='ℹ️',
                )
        else:
            typical = max(typical_day_volume(), 1)
            ratio   = grid_hours / typical
            if grid_hours == 0:
                headline_text = (
                    f"**No redispatch was needed today** — the SHN grid "
                    f"stayed within its limits. Below, the model attributes "
                    f"the calm to today's grid conditions."
                )
            elif ratio >= 1.4:
                headline_text = (
                    f"**{grid_hours} town-hours of redispatch today** — "
                    f"about **{ratio:.1f}× a typical day** ({typical}). "
                    f"Below, the model attributes the elevated stress to "
                    f"the day's grid conditions."
                )
            elif ratio <= 0.6:
                headline_text = (
                    f"**{grid_hours} town-hours of redispatch today** — "
                    f"about **{ratio:.1f}× a typical day** ({typical}). "
                    f"Below, the model attributes the calmer-than-usual "
                    f"day to today's grid conditions."
                )
            else:
                headline_text = (
                    f"**{grid_hours} town-hours of redispatch today** "
                    f"(typical day: {typical}). Below, the model attributes "
                    f"how today's grid conditions tilted the system."
                )
            st.markdown(headline_text)

            # ---- per-group bar chart -----------------------------------
            # Display marginal effects in **odds-ratio % change** because:
            #   (a) it's additive in log-odds space (the natural model space),
            #   (b) it's signal-preserving at low base rates, where pp on the
            #       calibrated-probability axis collapses to ~0,
            #   (c) it reads naturally: "wind reduced today's stress odds by 11%".
            effects = attribution['group_effects']                # already sorted

            def _odds_pct(logodds: float) -> float:
                """exp(logodds)-1 in percent."""
                return (float(np.exp(logodds)) - 1.0) * 100.0

            bar_rows = [(e['group'], _odds_pct(e['delta_logodds']),
                         e['delta_logodds']) for e in effects]
            bar_y    = [r[0] for r in bar_rows][::-1]
            bar_x    = [r[1] for r in bar_rows][::-1]
            bar_lo   = [r[2] for r in bar_rows][::-1]

            def _bar_color(pct: float) -> str:
                if pct >  10:   return 'rgba(255,  48,  48, 0.90)'   # strong push up
                if pct >   2:   return 'rgba(255, 127,  14, 0.90)'   # mild push up
                if pct < -10:   return 'rgba(134, 239, 172, 0.85)'   # strong push down
                if pct <  -2:   return 'rgba( 59, 130, 246, 0.80)'   # mild push down
                return 'rgba(160, 165, 175, 0.55)'                   # neutral

            bar_col = [_bar_color(x) for x in bar_x]
            bar_txt = [f"{x:+.1f}%" for x in bar_x]

            fig_drv = go.Figure(go.Bar(
                x=bar_x, y=bar_y, text=bar_txt, textposition='outside',
                textfont=dict(family='JetBrains Mono, monospace',
                              size=11, color=COLOR_TEXT_MUTED),
                orientation='h',
                marker=dict(color=bar_col),
                customdata=bar_lo,
                hovertemplate=(
                    '<b>%{y}</b><br>'
                    'effect on today\'s busyness: %{x:+.1f}%<extra></extra>'
                ),
            ))
            fig_drv.add_vline(
                x=0, line=dict(color='rgba(255,255,255,0.45)',
                               width=1, dash='dash'),
            )
            fig_drv.update_layout(
                height=70 + 50 * len(bar_y),
                margin=dict(l=10, r=80, t=20, b=40),
                xaxis=dict(
                    title='← made today calmer    ·    made today busier →',
                    ticksuffix='%', zeroline=False,
                ),
                yaxis=dict(automargin=True, title=None),
                showlegend=False,
            )
            st.plotly_chart(fig_drv, use_container_width=True)

            # ---- per-group plain-English narrative ---------------------
            # Translates the math into a sentence the operator can read.
            # Story beats per group:
            #   • What today's value of this driver was relative to typical
            #   • Why it matters (one-line causal mechanism)
            #   • Direction of the push, in qualitative words
            def _strength(pct: float) -> str:
                a = abs(pct)
                if a > 20: return 'a lot'
                if a > 10: return 'noticeably'
                if a >  3: return 'a little'
                return 'barely'

            def _verb(pct: float) -> str:
                if pct >  3:  return 'made redispatch **more likely**'
                if pct < -3:  return 'made redispatch **less likely**'
                return 'had **almost no effect** on today'

            STORY = {
                'Wind': (
                    'Wind is the #1 driver of redispatch in Schleswig-Holstein. '
                    'When wind farms generate more than the local 110 kV lines '
                    'can carry south, the operator has to curtail them.'
                ),
                'Recent activity': (
                    'Stress tends to persist — if the grid has been busy in the '
                    'last 24 hours and last week, today is more likely to be '
                    'busy too.'
                ),
                'Load & price': (
                    'High demand and high prices generally absorb extra '
                    'generation. Low demand or negative prices mean oversupply, '
                    'which forces curtailment.'
                ),
                'Solar & temperature': (
                    'Solar adds to renewable output. On sunny, mild days it '
                    'piles on top of wind and pushes more generation through '
                    'the same congested lines.'
                ),
                'Calendar': (
                    'Time-of-day, weekday vs weekend, and season patterns the '
                    'model has learned from history.'
                ),
                'Location': (
                    'Some substations are inherently busier than others — '
                    'their geography and identity priors.'
                ),
            }

            with st.expander('What does each driver mean?', expanded=False):
                for e in effects:
                    pct  = _odds_pct(e['delta_logodds'])
                    verb = _verb(pct)
                    strength = _strength(pct)
                    story = STORY.get(e['group'], '')
                    if abs(pct) <= 3:
                        line = (f"**{e['group']}** — {verb}. {story}")
                    else:
                        line = (f"**{e['group']}** — {verb} "
                                f"({strength}). {story}")
                    st.markdown(line)

            # ---- technical expander: per-feature contributions ---------
            # Only available when the full per-day parquet is on disk; the
            # daily-summary fallback (used in CI / fresh clones) doesn't carry
            # per-feature data.
            tf = attribution.get('top_features')
            if tf is not None and not tf.empty:
                with st.expander('See the numbers (for the technical reader)',
                                 expanded=False):
                    st.caption(
                        f"Top 5 individual signals ranked by how much they shifted "
                        f"the model's prediction for {date}. Positive numbers "
                        f"pushed the day toward stress; negative pushed it away. "
                        f"Averaged across all 175 substations and 24 hours of "
                        f"the day."
                    )
                    tf = tf.copy()
                    tf['odds_change_pct'] = (np.exp(tf['mean_logodds']) - 1) * 100
                    tf['direction']       = tf['mean_logodds'].apply(
                        lambda x: '↑ toward stress' if x > 0 else '↓ away from stress'
                    )
                    tf = tf[['feature', 'direction', 'odds_change_pct']]
                    tf.columns = ['Signal', 'Direction', 'Effect on busy odds']
                    st.dataframe(
                        tf.style.format({'Effect on busy odds': '{:+.1f}%'}),
                        use_container_width=True, hide_index=True,
                    )

            st.caption(
                "How this is computed: the forecast model looks at every "
                "substation, every hour of the day, and assigns each grid "
                "condition (wind, demand, recent activity, etc.) a portion "
                "of the day's predicted busyness. We group the signals into "
                "six families and show how much each tilted the day. "
                "Families overlap a bit (a windy day is usually a low-price "
                "day too), so the percentages don't add up exactly."
            )

    if not animation_mode:
        st.markdown('### Top 15 most-congested substations today')
        top = df.nlargest(15, 'active_hours').copy()
        if (top['active_hours'] == 0).all():
            st.info('No redispatch events anywhere on this date.')
        else:
            top.insert(0, '#', range(1, len(top) + 1))
            leaderboard = top[['#', 'town', 'active_hours',
                               'peak_concurrency', 'dominant_reason']].copy()
            leaderboard = leaderboard.rename(columns={
                'town':             'Town',
                'active_hours':     'Active hours',
                'peak_concurrency': 'Peak concurrent ops',
                'dominant_reason':  'Dominant reason',
            })
            st.dataframe(
                leaderboard.style.background_gradient(
                    cmap='YlOrRd', subset=['Active hours'], vmin=0, vmax=24,
                ),
                use_container_width=True, hide_index=True, height=560,
            )

            csv_buf = io.StringIO()
            export = df.sort_values('active_hours', ascending=False)[
                ['town', 'active_hours', 'n_events', 'peak_concurrency',
                 'dominant_reason', 'lat', 'lon']
            ].copy()
            export.insert(0, 'date', date)
            export.to_csv(csv_buf, index=False)

            col_dl1, col_dl2 = st.columns([1, 5])
            with col_dl1:
                st.download_button(
                    'Download CSV (all towns)',
                    data=csv_buf.getvalue(),
                    file_name=f'redispatch_{date}.csv',
                    mime='text/csv',
                )
            with col_dl2:
                st.caption('Full grid for this date · sorted by active hours · '
                           'lat/lon included for downstream maps.')

# =============================================================
# TAB 2 — Town deep dive
# =============================================================
with tab_town:
    c1, c2 = st.columns([1.5, 1])
    with c1:
        default_town = 'Husum' if 'Husum' in all_towns else all_towns[0]
        town = st.selectbox('Town', all_towns, index=all_towns.index(default_town))
    with c2:
        # Mirror the date the user picked on Tab 1. Streamlit treats the
        # session_state key as the source of truth once a widget is
        # instantiated, so we sync explicitly here.
        if 'deep_date' not in st.session_state:
            st.session_state['deep_date'] = st.session_state.get(
                'sel_date', pd.Timestamp(date_hi).date()
            )
        deep_date = st.date_input(
            'End date',
            min_value=date_lo, max_value=date_hi, key='deep_date',
            help='The history view ends on this date and looks back 90 days. '
                 'Defaults to the date selected on the Today tab.',
        )

    hist = town_history(town, str(deep_date), days_back=90)

    if hist.empty:
        st.warning(f'No activity recorded for {town} in the chosen 90-day window.')
    else:
        # ----- KPI strip -----
        total_h    = int(hist['active_hours'].sum())
        n_active   = int((hist['active_hours'] > 0).sum())
        busiest    = hist.loc[hist['active_hours'].idxmax()]
        pct_active = 100 * total_h / (90 * 24)

        k1, k2, k3, k4 = st.columns(4)
        k1.metric('Total active hours', f"{total_h}h")
        k2.metric('Days with redispatch', f"{n_active} / 90")
        k3.metric('Busiest day',
                  f"{int(busiest['active_hours'])}h",
                  busiest['date'].strftime('%d %b %Y'))
        k4.metric('% of time congested', f"{pct_active:.1f}%")

        # ----- daily history line chart -----
        st.markdown(f'### {town} - 90-day history')
        st.caption(
            'Bar = active hours that day. Line = 7-day rolling average. '
            'A flat-low line means a quiet town; rising lines mean ongoing '
            'congestion that needs sustained attention.'
        )

        fig_hist = go.Figure()
        fig_hist.add_trace(go.Bar(
            x=hist['date'], y=hist['active_hours'],
            name='Active hours',
            marker=dict(color=COLOR_ACCENT,
                        line=dict(color='rgba(0,0,0,0.3)', width=0.4)),
            hovertemplate='<b>%{x|%a %d %b}</b><br>%{y}h active<extra></extra>',
        ))
        fig_hist.add_trace(go.Scatter(
            x=hist['date'], y=hist['rolling_7'],
            mode='lines', name='7-day avg',
            line=dict(color='#ff7f0e', width=2.5, dash='dot'),
            hovertemplate='<b>%{x|%a %d %b}</b><br>7-day avg: %{y:.1f}h<extra></extra>',
        ))
        fig_hist.update_layout(
            height=340,
            margin=dict(l=10, r=10, t=10, b=40),
            xaxis=dict(title=None),
            yaxis=dict(title='Active hours per day', range=[0, 24]),
            legend=dict(orientation='h', y=1.05, x=1, xanchor='right'),
            hovermode='x unified',
        )
        st.plotly_chart(fig_hist, use_container_width=True)

        # ----- 7-day hour-of-day heatmap -----
        heat = town_hour_heatmap(town, str(deep_date), days_back=7)
        st.markdown(f'### {town} — last 7 days, hour-by-hour')
        st.caption(
            'Each cell = how many of the four 15-min slots in that hour were '
            'active. 4 = the entire hour was congested; 0 = nothing happened.'
        )

        if heat.empty or heat.values.sum() == 0:
            st.info(
                f'**{town}** had no redispatch in the 7 days ending '
                f'{pd.Timestamp(deep_date).strftime("%a %d %b %Y")}. '
                f'Pick an earlier end date to see this town\'s busy weeks — '
                f'the 90-day chart above shows when activity peaked.'
            )
        else:
            date_labels = [pd.Timestamp(d).strftime('%a %d %b') for d in heat.index]
            fig_heat = go.Figure(go.Heatmap(
                z=heat.values,
                x=[f'{h:02d}:00' for h in heat.columns],
                y=date_labels,
                colorscale=HEAT_SCALE,
                zmin=0, zmax=4,
                hovertemplate=('<b>%{y}</b><br>%{x}<br>'
                               'Active 15-min slots: %{z}/4<extra></extra>'),
                colorbar=dict(
                    title=dict(text='Slots / 4', font=dict(color=COLOR_TEXT_MUTED)),
                    tickfont=dict(color=COLOR_TEXT_MUTED),
                ),
            ))
            fig_heat.update_layout(
                height=320,
                margin=dict(l=10, r=10, t=10, b=40),
                xaxis=dict(side='top', title=None),
                yaxis=dict(autorange='reversed', title=None),
            )
            st.plotly_chart(fig_heat, use_container_width=True)

        # ----- Substation-level breakdown -----
        # Towns aggregate multiple physical transformers (`locationBottleneck`).
        # Showing them helps the audience see *which* substation in the town is
        # actually being curtailed, instead of just "the town."
        breakdown = town_substation_breakdown(town, top_n=10)
        st.markdown(f'### Substations in {town}')
        st.caption(
            'Top transformers within this town, ranked by lifetime Netzengpass '
            'operations. The map currently aggregates to the town centroid; '
            'this table is the substation-level detail Anton asked about.'
        )
        if breakdown.empty:
            st.info(f'No raw operations data for {town}. '
                    f'Either the town has only Netzengpass-filtered events or '
                    f'it sits outside the raw-chunk window.')
        else:
            disp = breakdown.copy()
            disp.insert(0, '#', range(1, len(disp) + 1))
            disp = disp.rename(columns={
                'locationBottleneck': 'Transformer',
                'n_ops':              'All operations',
                'n_neng_ops':         'Netzengpass ops',
                'first_seen':         'First seen',
                'last_seen':          'Last seen',
            })
            disp['First seen'] = pd.to_datetime(disp['First seen']).dt.strftime('%a %d %b %Y')
            disp['Last seen']  = pd.to_datetime(disp['Last seen']).dt.strftime('%a %d %b %Y')
            disp = disp[['#', 'Transformer', 'Netzengpass ops',
                         'All operations', 'First seen', 'Last seen']]
            st.dataframe(
                disp.style.background_gradient(
                    cmap='YlOrRd', subset=['Netzengpass ops'],
                ),
                use_container_width=True, hide_index=True, height=380,
            )
            n_total = len(substation_index().query('town == @town'))
            if n_total > len(breakdown):
                st.caption(
                    f'Showing top 10 of **{n_total}** transformers in this town. '
                    f'Long tail of low-activity feeders not shown.'
                )

# ---------------- footer ----------------
with st.expander('About this dashboard'):
    st.markdown(
        '* **Data source.** Operational redispatch records from '
        'Schleswig-Holstein Netz (SHN), filtered to grid-bottleneck reasons '
        '(*Netzengpass* / *Netzengpass I*). Refreshed daily.\n'
        '* **Severity metric.** Total active hours per (town, day) — between '
        '0 and 24. A town active 14 hours had at least one redispatch event '
        'overlapping each of those 14 hours.\n'
        '* **Window.** 1 January 2024 to the latest available data.\n'
        '* **Topology.** 110 kV substations and lines from OpenStreetMap '
        '(Overpass), refreshed when the grid changes. Lines are coloured by '
        'the highest concurrent-op count among nearby substations.\n'
        '* **"Why?" attribution.** A LightGBM 24-hour-horizon forecast '
        '(198 features, calibrated, ROC-AUC ≈ 0.83) decomposes each day\'s '
        'predicted busyness into six driver families using TreeSHAP. The '
        'narrative beneath the map summarises which families pushed the '
        'system toward or away from stress.\n'
        '* **Geocoding.** Town centroids from OpenStreetMap (Nominatim) with a '
        'small set of manual overrides for unnamed substations.'
    )
