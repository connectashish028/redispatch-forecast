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


@st.cache_data
def multi_day_hours(days_back: int = 90) -> pd.DataFrame:
    """Generate DataFrame for animation: last N days, per-town daily metrics."""
    end_date = pd.Timestamp.now().normalize()
    start_date = end_date - pd.Timedelta(days=days_back)
    all_dates = pd.date_range(start=start_date, end=end_date, freq='D')

    dfs = []
    for d in all_dates:
        df_day = daily_hours(str(d.date()))
        if not df_day.empty:
            df_day['date'] = d.date()
            dfs.append(df_day)

    if not dfs:
        return pd.DataFrame()
    return pd.concat(dfs, ignore_index=True)


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

with st.expander('What is a redispatch event?'):
    st.markdown(
        'When wind or solar farms generate more power than the local '
        'transmission lines can move out of the region, the grid operator '
        'orders selected plants to **reduce or shift their output** so the '
        'lines do not overload. Each such instruction is one *redispatch '
        'event*. In Schleswig-Holstein this happens often along the windy '
        'North Sea coast.\n\n'
        '**This dashboard shows:** for any day, which substations were '
        'congested and for how many hours. Bigger / redder bubble on the map '
        '= more hours with redispatch activity that day.'
    )

# ---------------------- tabs ----------------------
tab_map, tab_town = st.tabs(['Daily map', 'Town deep dive'])

# =============================================================
# TAB 1 — Daily map
# =============================================================
with tab_map:
    animation_mode = st.checkbox(
        'Enable 90-Day Animation', value=False,
        help='Animate markers over the last 90 days instead of a single day.',
    )

    if animation_mode:
        threshold = st.slider('Highlight towns above (hours active)', 0, 24, 4, step=1)
        df = multi_day_hours(90)
        if df.empty:
            st.error('No data in the last 90 days.')
            st.stop()

        st.markdown(
            f"<div style='padding:18px 22px; border-radius:14px; "
            f"background-color:{COLOR_SURFACE}; border:1px solid {COLOR_RING}; "
            f"margin: 8px 0 18px 0;'>"
            f"<div style='font-size:0.78rem; color:{COLOR_TEXT_MUTED}; "
            f"text-transform:uppercase; letter-spacing:0.08em; margin-bottom:4px'>"
            f"ANIMATION · Last 90 Days</div>"
            f"<div style='font-size:1.4rem; font-family:Space Grotesk,Inter,sans-serif; "
            f"font-weight:600; letter-spacing:-0.025em; line-height:1.2'>"
            f"Watch congestion evolve over time</div>"
            f"<div style='color:{COLOR_TEXT_MUTED}; margin-top:6px; font-size:0.92rem'>"
            f"Use the slider below the map to scrub through days. Towns above {threshold}h are highlighted."
            f"</div></div>",
            unsafe_allow_html=True,
        )
    else:
        if 'sel_date' not in st.session_state:
            st.session_state.sel_date = pd.Timestamp(date_hi).date()

        c1, c2 = st.columns([1.2, 3])
        with c1:
            date = st.date_input('Date', value=st.session_state.sel_date,
                                 min_value=date_lo, max_value=date_hi, key='sel_date')
        with c2:
            threshold = st.slider('Highlight towns above (hours active)', 0, 24, 4, step=1)

        df = daily_hours(str(date))
        if df.empty:
            st.error(f'No data on {date}. Pick a date between {date_lo} and {date_hi}.')
            st.stop()

        n_alerts     = int((df['active_hours'] >= threshold).sum())
        n_any        = int((df['active_hours'] > 0).sum())
        grid_hours   = int(df['active_hours'].sum())
        busiest      = df.loc[df['active_hours'].idxmax()]
        if grid_hours == 0:
            weather_word, headline_color = 'quiet', '#2ca02c'
        elif grid_hours < 100:
            weather_word, headline_color = 'normal', COLOR_TEXT
        elif grid_hours < 300:
            weather_word, headline_color = 'busy', '#ff7f0e'
        else:
            weather_word, headline_color = 'very busy', '#ff3030'

        st.markdown(
            f"<div style='padding:18px 22px; border-radius:14px; "
            f"background-color:{COLOR_SURFACE}; border:1px solid {COLOR_RING}; "
            f"margin: 8px 0 18px 0;'>"
            f"<div style='font-size:0.78rem; color:{COLOR_TEXT_MUTED}; "
            f"text-transform:uppercase; letter-spacing:0.08em; margin-bottom:4px'>"
            f"DAY · {date.strftime('%A %d %B %Y')}</div>"
            f"<div style='font-size:1.4rem; font-family:Space Grotesk,Inter,sans-serif; "
            f"font-weight:600; letter-spacing:-0.025em; line-height:1.2'>"
            f"<span style='color:{headline_color}'>It was a {weather_word} day</span> "
            f"<span style='color:{COLOR_TEXT_MUTED}'>·</span> "
            f"<span style='color:{COLOR_TEXT}'>{n_any} of 175 substations had redispatch, "
            f"{grid_hours} town-hours grid-wide</span></div>"
            f"<div style='color:{COLOR_TEXT_MUTED}; margin-top:6px; font-size:0.92rem'>"
            f"Busiest substation: <b style='color:{COLOR_TEXT}'>{busiest['town']}</b> "
            f"with <b style='color:{COLOR_TEXT}'>{int(busiest['active_hours'])} active hours</b>. "
            f"<b style='color:{COLOR_TEXT}'>{n_alerts}</b> substations were congested for "
            f"≥{threshold} hours."
            f"</div></div>",
            unsafe_allow_html=True,
        )

    if not animation_mode:
        k1, k2, k3, k4 = st.columns(4)
        k1.metric(f'Above {threshold}h', f"{n_alerts}")
        k2.metric('Any redispatch today', f"{n_any}")
        k3.metric('Most active town', f"{int(busiest['active_hours'])}h", busiest['town'])
        k4.metric('Total town-hours', f"{grid_hours}")

    df['size_metric']   = df['active_hours'].clip(lower=0)
    df['display_size']  = np.where(
        df['active_hours'] >= threshold, df['size_metric'].clip(lower=2) * 1.6,
        df['size_metric'].clip(lower=2) * 0.7,
    )
    df['display_opacity'] = np.where(df['active_hours'] >= threshold, 0.95, 0.35)

    if animation_mode:
        fig_map = px.scatter_map(
            df.sort_values(['date', 'active_hours']),
            lat='lat', lon='lon',
            size='display_size',
            color='active_hours',
            color_continuous_scale=HEAT_SCALE,
            range_color=[0, 24],
            size_max=36,
            animation_frame='date',
            animation_group='town',
            hover_name='town',
            hover_data={
                'active_hours': True, 'n_events': True, 'peak_concurrency': True,
                'dominant_reason': True, 'date': True,
                'lat': False, 'lon': False, 'display_size': False, 'size_metric': False,
                'display_opacity': False, 'active_15min_slots': False,
            },
            map_style='carto-positron',
            zoom=7.0, center={'lat': 54.3, 'lon': 9.7},
            height=620,
        )
    else:
        fig_map = px.scatter_map(
            df.sort_values('active_hours'),
            lat='lat', lon='lon',
            size='display_size',
            color='active_hours',
            color_continuous_scale=HEAT_SCALE,
            range_color=[0, 24],
            size_max=36,
            hover_name='town',
            hover_data={
                'active_hours': True, 'n_events': True, 'peak_concurrency': True,
                'dominant_reason': True, 'lat': False, 'lon': False,
                'display_size': False, 'size_metric': False, 'display_opacity': False,
                'active_15min_slots': False,
            },
            map_style='carto-positron',
            zoom=7.0, center={'lat': 54.3, 'lon': 9.7},
            height=620,
        )

    fig_map.update_traces(
        marker=dict(opacity=df.sort_values('active_hours')['display_opacity']),
        hovertemplate=(
            '<b>%{hovertext}</b><br>'
            'Active hours: <b>%{customdata[0]}h</b><br>'
            'Distinct events: %{customdata[1]}<br>'
            'Peak concurrent ops: %{customdata[2]}<br>'
            'Dominant reason: %{customdata[3]}<extra></extra>'
        ),
    )
    fig_map.update_layout(
        margin=dict(l=0, r=0, t=10, b=0),
        coloraxis_colorbar=dict(
            title=dict(text='Active hours', font=dict(color=COLOR_TEXT_MUTED)),
            tickfont=dict(color=COLOR_TEXT_MUTED),
            ticksuffix='h',
            tickvals=[0, 6, 12, 18, 24],
        ),
        transition=dict(duration=200, easing='cubic-in-out'),
    )
    st.plotly_chart(fig_map, use_container_width=True)

    if not animation_mode:
        st.markdown('### Top 15 most-congested substations today')
        top = df.nlargest(15, 'active_hours').copy()
        if (top['active_hours'] == 0).all():
            st.info('No redispatch events anywhere on this date.')
        else:
            top.insert(0, '#', range(1, len(top) + 1))
            leaderboard = top[['#', 'town', 'active_hours', 'n_events',
                               'peak_concurrency', 'dominant_reason']].copy()
            leaderboard = leaderboard.rename(columns={
                'town':             'Town',
                'active_hours':     'Active hours',
                'n_events':         'Distinct events',
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
        deep_date = st.date_input(
            'End date',
            value=st.session_state.get('sel_date', pd.Timestamp(date_hi).date()),
            min_value=date_lo, max_value=date_hi, key='deep_date',
            help='The history view ends on this date and looks back 90 days.',
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
        st.markdown(f'### {town} - last 7 days, hour-by-hour')
        st.caption(
            'Each cell = how many of the four 15-min slots in that hour were '
            'active. 4 = the entire hour was congested; 0 = nothing happened.'
        )

        if heat.empty:
            st.info('No activity in the last 7 days.')
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

# ---------------- footer ----------------
with st.expander('About this dashboard'):
    st.markdown(
        '* **Data source.** Operational redispatch records from '
        'Schleswig-Holstein Netz (SHN), filtered to grid-bottleneck reasons '
        '(*Netzengpass* / *Netzengpass I*).\n'
        '* **Severity metric.** Total active hours per (town, day) - between '
        '0 and 24. A town active 14 hours had at least one redispatch event '
        'overlapping each of those 14 hours.\n'
        '* **Window.** 1 January 2024 to the latest available data.\n'
        '* **Geocoding.** Town centroids from OpenStreetMap (Nominatim) with a '
        'small set of manual overrides for unnamed substations.\n'
        '* **What is not in v1.** Day-ahead probability forecasts. The '
        'underlying ML pipeline is WIP and will be layered onto '
        'this dashboard in a future release.'
    )
