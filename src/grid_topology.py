"""
grid_topology.py — fetch the Schleswig-Holstein 110 kV grid topology from OpenStreetMap.

Queries Overpass for substations and transmission lines in the SH bounding box
at 110 kV (SHN's primary distribution voltage), and saves them as a GeoJSON
FeatureCollection at data/external/shn_grid.geojson.

Why OSM (and not SciGRID)?
    SciGRID's last open release was 2018; SciGRID-OSM is just an OSM snapshot.
    Querying OSM directly via Overpass gives the same data, fresher, no
    download dance.

Caveats baked in to the data:
    - ~30 % of 110 kV substations in OSM are unnamed (only lat/lon + voltage).
    - Operator tagging is incomplete: ~30 % of 110 kV substations are explicitly
      tagged "Schleswig-Holstein Netz". The rest are unattributed or owned by
      adjacent DSOs / TenneT TSO.
    - Voltage tags can be multi-valued ("110000;20000") for step-down stations.

Idempotent. Re-run any time. Suggest running monthly to refresh.

Usage:
    python src/grid_topology.py
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[1]
OUT  = ROOT / 'data' / 'external' / 'shn_grid.geojson'

# Schleswig-Holstein bounding box: south, west, north, east.
SH_BBOX = (53.36, 7.86, 55.06, 11.31)

# Overpass mirrors in priority order. The .de main one is overloaded most days;
# the openstreetmap.fr mirror has been most reliable in our scouting.
MIRRORS = [
    'https://overpass.openstreetmap.fr/api/interpreter',
    'https://overpass.kumi.systems/api/interpreter',
    'https://overpass-api.de/api/interpreter',
]
HEADERS = {'User-Agent': 'redispatch-forecast/1.0 (SH grid topology research)'}
HTTP_TIMEOUT_S = 180

# OSM voltages are in volts (so "110000" = 110 kV). The regex catches mixed-
# voltage tags like "110000", "110000;20000", "20000;110000;380000".
SUBS_QQL = f"""
[out:json][timeout:120];
(
  node["power"="substation"]["voltage"~"110000"]({SH_BBOX[0]},{SH_BBOX[1]},{SH_BBOX[2]},{SH_BBOX[3]});
  way["power"="substation"]["voltage"~"110000"]({SH_BBOX[0]},{SH_BBOX[1]},{SH_BBOX[2]},{SH_BBOX[3]});
);
out tags center;
"""

LINES_QQL = f"""
[out:json][timeout:120];
way["power"="line"]["voltage"~"110000"]({SH_BBOX[0]},{SH_BBOX[1]},{SH_BBOX[2]},{SH_BBOX[3]});
out tags geom;
"""


def query_overpass(qql: str) -> dict:
    """Try mirrors in order; return the first successful JSON response."""
    last_err: Exception | None = None
    for url in MIRRORS:
        print(f'    -> {url}')
        try:
            r = requests.post(url, data={'data': qql}, headers=HEADERS,
                              timeout=HTTP_TIMEOUT_S)
            r.raise_for_status()
            return r.json()
        except Exception as exc:
            print(f'       {type(exc).__name__}: {str(exc)[:90]}')
            last_err = exc
    raise RuntimeError(f'all Overpass mirrors failed; last error: {last_err}')


def make_substation_feature(el: dict) -> dict | None:
    """Convert an OSM substation element into a GeoJSON Point feature."""
    if el['type'] == 'node':
        lat, lon = el.get('lat'), el.get('lon')
    else:                                   # way / relation
        c = el.get('center') or {}
        lat, lon = c.get('lat'), c.get('lon')
    if lat is None or lon is None:
        return None
    tags = el.get('tags', {})
    return {
        'type': 'Feature',
        'geometry': {'type': 'Point', 'coordinates': [lon, lat]},
        'properties': {
            'kind':            'substation',
            'osm_type':        el['type'],
            'osm_id':          el.get('id'),
            'name':            tags.get('name', '') or '',
            'ref':             tags.get('ref', '') or '',
            'operator':        tags.get('operator', '') or '',
            'voltage':         tags.get('voltage', '') or '',
            'substation_type': tags.get('substation', '') or '',
        },
    }


def make_line_feature(el: dict) -> dict | None:
    """Convert an OSM line way into a GeoJSON LineString feature."""
    geom = el.get('geometry')               # list of {'lat':..., 'lon':...}
    if not geom or len(geom) < 2:
        return None
    coords = [[p['lon'], p['lat']] for p in geom]
    tags = el.get('tags', {})
    return {
        'type': 'Feature',
        'geometry': {'type': 'LineString', 'coordinates': coords},
        'properties': {
            'kind':     'line',
            'osm_id':   el.get('id'),
            'voltage':  tags.get('voltage', '') or '',
            'operator': tags.get('operator', '') or '',
            'cables':   tags.get('cables', '') or '',
            'circuits': tags.get('circuits', '') or '',
        },
    }


def main() -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)

    print('1. substations (~30 s)')
    subs_resp = query_overpass(SUBS_QQL)
    subs_features = [f for el in subs_resp['elements']
                     if (f := make_substation_feature(el))]

    time.sleep(1.0)                          # polite gap between mirrors

    print('2. lines (~30 s)')
    lines_resp = query_overpass(LINES_QQL)
    lines_features = [f for el in lines_resp['elements']
                      if (f := make_line_feature(el))]

    fc = {
        'type': 'FeatureCollection',
        'features': subs_features + lines_features,
    }
    OUT.write_text(json.dumps(fc), encoding='utf-8')
    size_kb = OUT.stat().st_size / 1024

    n_shn   = sum(1 for f in subs_features
                  if 'Schleswig-Holstein' in f['properties']['operator'])
    n_named = sum(1 for f in subs_features if f['properties']['name'])

    print()
    print('=' * 60)
    print(f'wrote {OUT.relative_to(ROOT)}  ({size_kb:,.0f} KB)')
    print('=' * 60)
    print(f'  substations:                     {len(subs_features)}')
    print(f'    of which SHN-tagged operator:  {n_shn} '
          f'({100*n_shn/max(len(subs_features),1):.0f}%)')
    print(f'    of which named:                {n_named} '
          f'({100*n_named/max(len(subs_features),1):.0f}%)')
    print(f'  lines:                           {len(lines_features)}')


if __name__ == '__main__':
    sys.exit(main())
