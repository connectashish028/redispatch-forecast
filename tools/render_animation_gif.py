"""
render_animation_gif.py — bake the dashboard's 30-day animation into a
GIF for the README.

Renders one frame per day for the last `--days` days of data, writes a
PNG per frame via Plotly + kaleido, and stitches them into an animated
GIF via imageio.

Usage
-----
    python tools/render_animation_gif.py
    python tools/render_animation_gif.py --days 30 --out docs/animation_30d.gif --fps 6

Dependencies (already in the cvae env): plotly, kaleido<1, imageio.
"""
from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'src'))

# Pull the dashboard's bucket palette + topology helpers so the rendered
# GIF visually matches what users see in the live app.
from theme import COLOR_BG  # noqa: E402

# Mirrored from app.py — keeping the renderer self-contained avoids
# importing streamlit (which app.py top-levels).
_BUCKET_EDGES  = [1, 4, 10, 20]
_BUCKET_COLORS = [
    'rgba(134, 239, 172, 0.45)',
    'rgba( 56, 189, 248, 0.85)',
    'rgba( 59, 130, 246, 0.95)',
    'rgba(255, 127,  14, 0.95)',
    'rgba(255,  48,  48, 0.95)',
]
_LINE_WIDTH = 2.4


def load_topology() -> dict:
    """Read shn_grid.geojson and return line coords + endpoints."""
    import json
    geo = json.loads((ROOT / 'data' / 'external' / 'shn_grid.geojson').read_text(encoding='utf-8'))
    lats, lons, line_coords = [], [], []
    endpoints_lat, endpoints_lon = [], []
    for f in geo['features']:
        if f['properties'].get('kind') != 'line':
            continue
        coords = f['geometry']['coordinates']
        if len(coords) < 2:
            continue
        path = [(lon, lat) for lon, lat in coords]
        line_coords.append(path)
        for lon, lat in path:
            lats.append(lat); lons.append(lon)
        lats.append(None); lons.append(None)
        endpoints_lat.append((path[0][1], path[-1][1]))
        endpoints_lon.append((path[0][0], path[-1][0]))
    return {
        'lines_lat':     lats,
        'lines_lon':     lons,
        'line_coords':   line_coords,
        'endpoints_lat': np.array(endpoints_lat, dtype=np.float64),
        'endpoints_lon': np.array(endpoints_lon, dtype=np.float64),
    }


def line_to_nearest_towns(threshold_km: float = 10.0) -> dict:
    """Bool matrix M[i, j] = True iff line i passes within threshold_km of town j."""
    topo = load_topology()
    if not topo['line_coords']:
        return {'M': np.zeros((0, 0), dtype=bool), 'town_names': [],
                'line_coords': []}
    geo_towns = pd.read_parquet(ROOT / 'data' / 'external' / 'towns_geo.parquet')
    geo_towns = geo_towns.dropna(subset=['lat', 'lon'])
    town_names = geo_towns['town'].astype(str).tolist()
    town_lat   = geo_towns['lat'].values
    town_lon   = geo_towns['lon'].values

    # Approximate point-line distance in km via haversine on each line endpoint
    # midpoint — same heuristic as app.py.
    midpoints_lat = topo['endpoints_lat'].mean(axis=1)
    midpoints_lon = topo['endpoints_lon'].mean(axis=1)

    # Vectorised haversine
    R_KM = 6371.0
    lat1 = np.radians(midpoints_lat)[:, None]
    lon1 = np.radians(midpoints_lon)[:, None]
    lat2 = np.radians(town_lat)[None, :]
    lon2 = np.radians(town_lon)[None, :]
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2)**2
    dist_km = 2 * R_KM * np.arcsin(np.sqrt(a))
    M = (dist_km <= threshold_km)
    return {'M': M, 'town_names': town_names, 'line_coords': topo['line_coords']}


def build_frame(date_str: str, line_coords, M, town_names, peak_today, topo) -> go.Figure:
    """Build a Plotly figure for a single day's intensity snapshot."""
    intensity = (M.astype(np.float32) * peak_today).max(axis=1)
    bucket_edges = np.array(_BUCKET_EDGES, dtype=np.float32)
    buckets = np.searchsorted(bucket_edges, intensity, side='right')

    # Build per-bucket coord arrays
    bucket_lats = [[] for _ in range(5)]
    bucket_lons = [[] for _ in range(5)]
    for li in np.where(buckets > 0)[0]:
        b = int(buckets[li])
        for lon, lat in line_coords[li]:
            bucket_lats[b].append(lat); bucket_lons[b].append(lon)
        bucket_lats[b].append(None); bucket_lons[b].append(None)

    fig = go.Figure()
    # Idle backdrop
    fig.add_trace(go.Scattermap(
        lat=topo['lines_lat'], lon=topo['lines_lon'],
        mode='lines',
        line=dict(width=_LINE_WIDTH, color=_BUCKET_COLORS[0]),
        hoverinfo='skip', showlegend=False,
    ))
    for i in range(1, 5):
        fig.add_trace(go.Scattermap(
            lat=bucket_lats[i], lon=bucket_lons[i],
            mode='lines',
            line=dict(width=_LINE_WIDTH, color=_BUCKET_COLORS[i]),
            hoverinfo='skip', showlegend=False,
        ))
    fig.update_layout(
        map=dict(style='carto-darkmatter',
                 center=dict(lat=54.3, lon=9.7), zoom=6.6),
        height=600, width=900,
        paper_bgcolor=COLOR_BG, plot_bgcolor=COLOR_BG,
        margin=dict(l=0, r=0, t=40, b=0),
        showlegend=False,
        annotations=[dict(
            x=0.02, y=0.97, xref='paper', yref='paper',
            text=f"<b style='color:white; font-size:18px;'>{date_str}</b>",
            showarrow=False, align='left',
            bgcolor='rgba(0,0,0,0.55)', borderpad=8,
            font=dict(family='JetBrains Mono, monospace',
                      color='white', size=16),
        )],
    )
    return fig


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--days', type=int, default=30)
    ap.add_argument('--out',  type=Path,
                    default=ROOT / 'docs' / 'animation_30d.gif')
    ap.add_argument('--fps',  type=int, default=6,
                    help='Playback speed in frames per second')
    args = ap.parse_args()

    print(f'[render] loading topology + 15-min activity matrix...')
    idx = line_to_nearest_towns(threshold_km=10.0)
    topo = load_topology()
    wide = pd.read_parquet(ROOT / 'data' / 'processed' / 'ts_15min_wide.parquet')

    d_hi = wide.index.max().normalize() + pd.Timedelta(days=1)
    d_lo = d_hi - pd.Timedelta(days=args.days)
    sub  = wide.loc[(wide.index >= d_lo) & (wide.index < d_hi)]
    daily_peak = sub.resample('1D').max()
    daily_peak = daily_peak.reindex(columns=idx['town_names']).fillna(0)
    P = daily_peak.values.astype(np.float32)

    print(f'[render] generating {len(daily_peak)} daily frames '
          f'({d_lo.date()} -> {d_hi.date() - pd.Timedelta(days=1)})...')

    args.out.parent.mkdir(parents=True, exist_ok=True)
    import imageio.v2 as imageio                                  # type: ignore

    with tempfile.TemporaryDirectory() as tmp:
        png_paths = []
        for i, (ts, row) in enumerate(zip(daily_peak.index, P)):
            label = ts.strftime('%a %d %b %Y')
            fig = build_frame(label, idx['line_coords'], idx['M'],
                              idx['town_names'], row, topo)
            png = Path(tmp) / f'frame_{i:03d}.png'
            fig.write_image(str(png), format='png', scale=1)
            png_paths.append(png)
            print(f'  frame {i+1}/{len(daily_peak)}: {label}')

        print(f'[render] stitching {len(png_paths)} frames into GIF...')
        with imageio.get_writer(str(args.out), mode='I', fps=args.fps,
                                loop=0) as writer:
            for p in png_paths:
                writer.append_data(imageio.imread(str(p)))

    sz = args.out.stat().st_size / 1024
    print(f'[render] wrote {args.out.relative_to(ROOT)}  ({sz:,.1f} KB)')


if __name__ == '__main__':
    main()
