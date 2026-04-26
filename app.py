"""
Redispatch Forecast — operator-grade dashboard with Framer-inspired dark theme.

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

import predict as P                                            # noqa: E402
from labels import (HORIZON_LABEL, HORIZON_DESCRIPTION,        # noqa: E402
                    METRIC_HELP, pretty)
from theme import (inject_css, register_plotly_template,       # noqa: E402
                   COLOR_BG, COLOR_SURFACE, COLOR_TEXT,
                   COLOR_TEXT_MUTED, COLOR_ACCENT,
                   COLOR_RING, HEAT_SCALE)

st.set_page_config(
    page_title='Redispatch Forecast — SHN',
    page_icon='⚡',
    layout='wide',
    initial_sidebar_state='collapsed',
)
inject_css(st)
register_plotly_template()

UTC_OFFSET_HOURS = 1   # CET = UTC+1; CEST = UTC+2 — close enough for an operator readout
LOW_DATA_THRESHOLD = 5  # towns with <= this many positives in test treated as low-confidence

# ---------------------- caches ----------------------
@st.cache_resource
def load_pipeline():
    P._load()
    return P._FEATURES, P._MODELS, P._CALS, P._FCOLS


@st.cache_resource
def load_geo():
    return pd.read_parquet(ROOT / 'data/external/towns_geo.parquet').dropna(subset=['lat', 'lon'])


@st.cache_data(show_spinner='Forecasting all towns for this date…')
def forecast_grid(date_str: str) -> pd.DataFrame:
    feats, models, cals, fcols = load_pipeline()
    d = pd.Timestamp(date_str).normalize()
    sub = feats[(feats['ts'] >= d) & (feats['ts'] < d + pd.Timedelta(days=1))].copy()
    if sub.empty:
        return pd.DataFrame()

    X = sub[fcols].values
    out = sub[['ts', 'town']].reset_index(drop=True)
    out['hour'] = out['ts'].dt.hour
    for h in P.HORIZONS:
        raw = models[h].predict(X)
        cal = cals[h].predict(raw)
        out[f'p_{h.split("_")[1]}'] = cal
        if h in sub.columns:
            out[h] = sub[h].values

    g = load_geo()[['town', 'lat', 'lon']]
    out = out.merge(g, on='town', how='left').dropna(subset=['lat', 'lon'])
    return out


@st.cache_data
def low_data_towns() -> set[str]:
    """Towns with too few positives in the test window to trust their per-town forecast."""
    feats, _, _, _ = load_pipeline()
    test = feats[feats['ts'] >= pd.Timestamp('2026-01-01')]
    pos = test.groupby('town', observed=True)['y_24h'].sum()
    return set(pos[pos <= LOW_DATA_THRESHOLD].index.astype(str))


# ---------------------- header ----------------------
features, models, cals, fcols = load_pipeline()
test_lo = pd.Timestamp('2026-01-01').date()
test_hi = features['ts'].max().date()
LOW_DATA = low_data_towns()


def fmt_clock(ts: pd.Timestamp, local: bool) -> str:
    """Format a timestamp as HH:MM in either UTC or CET (+1h, simplified)."""
    return (ts + pd.Timedelta(hours=UTC_OFFSET_HOURS)).strftime('%H:%M') if local \
        else ts.strftime('%H:%M')


st.markdown(
    "<h1 style='margin-bottom:8px'>Redispatch Forecast</h1>"
    "<p style='color:#a6a6a6; font-size:1.05rem; margin-top:0'>"
    "Schleswig-Holstein Netz · 175 substations · day-ahead probability of grid bottlenecks"
    "</p>",
    unsafe_allow_html=True,
)

# ============================================================
# Tabs
# ============================================================
tab_map, tab_town = st.tabs(['Live forecast map', 'Town deep dive'])

# ============================================================
# TAB 1 — Live forecast map
# ============================================================
with tab_map:
    # ----------- control bar -----------
    if 'sel_date' not in st.session_state:
        st.session_state.sel_date = pd.Timestamp('2026-02-15').date()

    c1, c2, c3, c4 = st.columns([1.0, 1.6, 1.2, 1.0])
    with c1:
        date = st.date_input('Date', value=st.session_state.sel_date,
                             min_value=test_lo, max_value=test_hi, key='sel_date')
    with c2:
        hour = st.slider('Hour', min_value=0, max_value=23, value=12, key='sel_hour')
    with c3:
        horizon = st.selectbox('Forecast window', P.HORIZONS,
                               index=P.HORIZONS.index('y_24h'),
                               key='sel_horizon',
                               format_func=lambda h: HORIZON_LABEL[h])
    with c4:
        threshold = st.slider('Risk threshold', 0, 90, 30, step=5,
                              key='sel_threshold',
                              format='%d%%',
                              help='Towns at or above this probability are flagged as alerts.')

    use_cet = st.toggle('Show times in local time (CET)', value=True,
                        help='Internal clock is UTC. Toggle to display CET (+1h).')

    grid = forecast_grid(str(date))
    if grid.empty:
        st.error(f'No data for {date}. Pick a date between {test_lo} and {test_hi}.')
        st.stop()

    pcol = 'p_' + horizon.split('_')[1]
    snap = grid[grid['hour'] == hour].copy()
    snap['probability_pct'] = snap[pcol] * 100
    snap['size'] = (snap[pcol] * 100).clip(lower=4)
    snap['low_data'] = snap['town'].isin(LOW_DATA)

    # ----------- daily summary headline -----------
    n_alerts = int((snap[pcol] >= threshold / 100).sum())
    avg_today = snap[pcol].mean()
    busiest_today = snap.loc[snap[pcol].idxmax()]

    grid_24h_avg = grid['p_24h'].mean()
    if grid_24h_avg < 0.10:
        weather_word, headline_color = 'calm', '#2ca02c'
    elif grid_24h_avg < 0.20:
        weather_word, headline_color = 'normal', COLOR_TEXT
    elif grid_24h_avg < 0.30:
        weather_word, headline_color = 'busy', '#ff7f0e'
    else:
        weather_word, headline_color = 'very busy', '#ff3030'

    st.markdown(
        f"<div style='padding:18px 22px; border-radius:14px; "
        f"background-color:{COLOR_SURFACE}; border:1px solid {COLOR_RING}; "
        f"margin: 8px 0 18px 0;'>"
        f"<div style='font-size:0.78rem; color:{COLOR_TEXT_MUTED}; "
        f"text-transform:uppercase; letter-spacing:0.08em; margin-bottom:4px'>"
        f"OUTLOOK · {date}</div>"
        f"<div style='font-size:1.4rem; font-family:Space Grotesk,Inter,sans-serif; "
        f"font-weight:600; letter-spacing:-0.025em; line-height:1.2'>"
        f"<span style='color:{headline_color}'>The grid looks {weather_word} today</span> "
        f"<span style='color:{COLOR_TEXT_MUTED}'>·</span> "
        f"<span style='color:{COLOR_TEXT}'>{n_alerts} towns flagged at "
        f"≥{threshold}% probability</span></div>"
        f"<div style='color:{COLOR_TEXT_MUTED}; margin-top:6px; font-size:0.92rem'>"
        f"At {fmt_clock(pd.Timestamp(date) + pd.Timedelta(hours=hour), use_cet)} "
        f"{'CET' if use_cet else 'UTC'}, the highest predicted "
        f"<b style='color:{COLOR_TEXT}'>{HORIZON_LABEL[horizon].lower()}</b> probability "
        f"is <b style='color:{COLOR_TEXT}'>{busiest_today[pcol]:.0%}</b> at "
        f"<b style='color:{COLOR_TEXT}'>{busiest_today['town']}</b>."
        f"</div></div>",
        unsafe_allow_html=True,
    )

    # ----------- KPI strip -----------
    k1, k2, k3, k4 = st.columns(4)
    k1.metric(f'Above {threshold}% (alerts)', f"{n_alerts}",
              help='Number of substations the operator should plan to monitor.')
    k2.metric('Above 50%', f"{int((snap[pcol] >= 0.5).sum())}",
              help='High-confidence event candidates.')
    k3.metric('Highest probability', f"{snap[pcol].max():.0%}",
              busiest_today['town'])
    k4.metric('Grid average', f"{snap[pcol].mean():.0%}")

    # ----------- the map -----------
    snap['display_size'] = np.where(
        snap[pcol] >= threshold / 100, snap['size'] * 1.4, snap['size'] * 0.6
    )
    snap['marker_opacity'] = np.where(snap[pcol] >= threshold / 100, 0.95, 0.35)

    fig_map = px.scatter_mapbox(
        snap.sort_values(pcol),
        lat='lat', lon='lon',
        size='display_size',
        color='probability_pct',
        color_continuous_scale=HEAT_SCALE,
        range_color=[0, 100],
        size_max=34,
        hover_name='town',
        hover_data={
            'probability_pct': ':.1f',
            'p_1h':  ':.1%', 'p_6h':  ':.1%', 'p_24h': ':.1%',
            'lat': False, 'lon': False, 'display_size': False,
            'size': False, 'hour': False, 'marker_opacity': False,
            'low_data': False,
        },
        zoom=7.0, center={'lat': 54.3, 'lon': 9.7},
        height=620,
    )
    fig_map.update_traces(
        marker=dict(opacity=snap.sort_values(pcol)['marker_opacity']),
        hovertemplate=(
            '<b>%{hovertext}</b><br>'
            f'{HORIZON_LABEL[horizon]}: ' '%{customdata[0]:.1f}%<br>'
            'Next 1 hour: %{customdata[1]:.1%}<br>'
            'Next 6 hours: %{customdata[2]:.1%}<br>'
            'Next 24 hours: %{customdata[3]:.1%}<extra></extra>'
        ),
    )
    fig_map.update_layout(
        margin=dict(l=0, r=0, t=10, b=0),
        coloraxis_colorbar=dict(
            title=dict(text='Probability', font=dict(color=COLOR_TEXT_MUTED)),
            tickfont=dict(color=COLOR_TEXT_MUTED),
            ticksuffix='%',
        ),
    )
    st.plotly_chart(fig_map, use_container_width=True)

    # ----------- alert action list (CSV-exportable) -----------
    st.markdown('### Action list')
    st.caption(
        f'Towns at or above the **{threshold}% threshold** for **{HORIZON_LABEL[horizon].lower()}** '
        f'at this hour. Sorted by probability. ⚠ = limited history (treat with care).'
    )

    alerts = snap[snap[pcol] >= threshold / 100].copy()
    alerts = alerts.sort_values(pcol, ascending=False)
    if alerts.empty:
        st.info(
            f'No towns above {threshold}%. Drag the threshold lower or pick a busier hour.',
            icon='ℹ️',
        )
    else:
        disp = alerts[['town', 'p_1h', 'p_6h', 'p_24h', 'low_data']].copy()
        disp.insert(0, '#', range(1, len(disp) + 1))
        disp['Town'] = disp.apply(
            lambda r: f"{'⚠ ' if r['low_data'] else ''}{r['town']}", axis=1)
        disp = disp.drop(columns=['town', 'low_data'])
        disp = disp.rename(columns={
            'p_1h':  HORIZON_LABEL['y_1h'],
            'p_6h':  HORIZON_LABEL['y_6h'],
            'p_24h': HORIZON_LABEL['y_24h'],
        })
        disp = disp[['#', 'Town', HORIZON_LABEL['y_1h'],
                     HORIZON_LABEL['y_6h'], HORIZON_LABEL['y_24h']]]
        prob_cols = [HORIZON_LABEL[h] for h in P.HORIZONS]
        st.dataframe(
            disp.style
                .format({c: '{:.1%}' for c in prob_cols})
                .background_gradient(cmap='YlOrRd', subset=[HORIZON_LABEL[horizon]],
                                     vmin=0, vmax=1),
            use_container_width=True, hide_index=True, height=320,
        )

        # CSV export — full grid for this date+hour, not just the alert-list slice,
        # so the operator gets context.
        csv_buffer = io.StringIO()
        export_df = alerts[['town', 'p_1h', 'p_6h', 'p_24h']].copy()
        export_df.insert(0, 'date', date)
        export_df.insert(1, 'hour_utc', hour)
        export_df.insert(2, 'horizon_explained', HORIZON_LABEL[horizon])
        export_df['low_data_flag'] = alerts['low_data'].values
        export_df.to_csv(csv_buffer, index=False)

        col_dl1, col_dl2 = st.columns([1, 5])
        with col_dl1:
            st.download_button(
                'Download CSV',
                data=csv_buffer.getvalue(),
                file_name=f'redispatch_alerts_{date}_h{hour:02d}_{horizon}.csv',
                mime='text/csv',
            )
        with col_dl2:
            st.caption(f'{len(alerts)} alert rows · UTC time · '
                       f'thresholds and horizon embedded in the file.')

    # ----------- 7-day outlook strip -----------
    st.markdown('### 7-day outlook')
    st.caption('Daily peak Next-24-hour probability for the top 12 most-active substations.')

    busiest_towns = (grid.groupby('town', observed=True)['p_24h'].max()
                          .sort_values(ascending=False).head(12).index.tolist())

    horizon_dates = [pd.Timestamp(date) + pd.Timedelta(days=i) for i in range(7)]
    rows = []
    for d in horizon_dates:
        if d.date() < test_lo or d.date() > test_hi:
            continue
        gd = forecast_grid(str(d.date()))
        if gd.empty:
            continue
        peaks = gd[gd['town'].isin(busiest_towns)].groupby(
            'town', observed=True)['p_24h'].max()
        for tw, p in peaks.items():
            rows.append({'town': tw, 'date': d.date(), 'p': p})

    if rows:
        out = pd.DataFrame(rows)
        wide_strip = out.pivot(index='town', columns='date', values='p')
        wide_strip = wide_strip.reindex(busiest_towns)
        date_labels = [pd.Timestamp(c).strftime('%a %d %b') for c in wide_strip.columns]

        fig_strip = go.Figure(go.Heatmap(
            z=wide_strip.values,
            x=date_labels,
            y=wide_strip.index,
            colorscale=HEAT_SCALE,
            zmin=0, zmax=1,
            text=(wide_strip.values * 100).round(0).astype(int),
            texttemplate='%{text}%',
            textfont=dict(color=COLOR_TEXT, size=11),
            hovertemplate=(
                '<b>%{y}</b><br>%{x}<br>Peak P(24h): %{z:.1%}<extra></extra>'
            ),
            colorbar=dict(
                title=dict(text='Peak P(24h)', font=dict(color=COLOR_TEXT_MUTED)),
                tickformat='.0%',
                tickfont=dict(color=COLOR_TEXT_MUTED),
            ),
        ))
        fig_strip.update_layout(
            height=440,
            margin=dict(l=10, r=10, t=10, b=40),
            xaxis=dict(side='top'),
            yaxis=dict(autorange='reversed'),
        )
        st.plotly_chart(fig_strip, use_container_width=True)
    else:
        st.info('Not enough days remaining in the test window for a 7-day outlook from this date.')

# ============================================================
# TAB 2 — Town deep dive
# ============================================================
with tab_town:
    towns = sorted(features['town'].astype(str).unique())
    default_town = 'Husum' if 'Husum' in towns else towns[0]

    c1, c2 = st.columns([1.5, 1])
    with c1:
        town = st.selectbox('Town', towns, index=towns.index(default_town))
    with c2:
        deep_date = st.date_input(
            'Date',
            value=st.session_state.get('sel_date', pd.Timestamp('2026-02-15').date()),
            min_value=test_lo, max_value=test_hi, key='deep_date',
        )

    if town in LOW_DATA:
        st.warning(
            f'⚠ **{town}** has very few historical events in the test window '
            '(<5 positives). Predictions for this town are best treated as a rough guide.',
            icon='⚠️',
        )

    try:
        df = P.predict_day(town, str(deep_date))
    except ValueError as exc:
        st.error(str(exc))
        df = None

    if df is not None and not df.empty:
        # day KPIs
        peak_p24 = df['p_24h'].max()
        peak_hr  = df.loc[df['p_24h'].idxmax(), 'ts']
        actual = bool(df['y_24h'].max()) if 'y_24h' in df.columns else None

        k1, k2, k3, k4 = st.columns(4)
        k1.metric(f'Peak {HORIZON_LABEL["y_24h"]} probability',
                  f"{peak_p24:.0%}", f"at {peak_hr.strftime('%H:%M')} UTC")
        k2.metric(f'Peak {HORIZON_LABEL["y_6h"]} probability',
                  f"{df['p_6h'].max():.0%}")
        k3.metric(f'Peak {HORIZON_LABEL["y_1h"]} probability',
                  f"{df['p_1h'].max():.0%}")
        if actual is not None:
            k4.metric('Did it actually happen?',
                      'Yes' if actual else 'No')

        # ---------- hourly forecast (Plotly dark) ----------
        st.markdown(f'### {town} — hourly forecast for {deep_date}')
        st.caption(
            'One line per forecast window. Open circles mark hours where a '
            'redispatch event actually happened (back-test).'
        )

        x = df['ts'].dt.hour
        fig_ts = go.Figure()
        fig_ts.add_trace(go.Scatter(
            x=x, y=df['p_24h'], mode='lines',
            line=dict(color='rgba(0,153,255,0)'),
            fill='tozeroy', fillcolor='rgba(0,153,255,0.10)',
            showlegend=False, hoverinfo='skip',
        ))
        line_specs = [
            ('y_1h',  '#ff3030', 'circle'),
            ('y_6h',  '#ff7f0e', 'square'),
            ('y_24h', COLOR_ACCENT, 'triangle-up'),
        ]
        for h, color, sym in line_specs:
            pcol_loc = 'p_' + h.split('_')[1]
            fig_ts.add_trace(go.Scatter(
                x=x, y=df[pcol_loc],
                mode='lines+markers',
                name=HORIZON_LABEL[h],
                line=dict(color=color, width=2.4),
                marker=dict(symbol=sym, size=9),
                hovertemplate=(f'<b>{HORIZON_LABEL[h]}</b><br>'
                               'Hour: %{x:02d}:00<br>'
                               'Probability: %{y:.1%}<extra></extra>'),
            ))
        for h, color, _ in line_specs:
            if h in df.columns:
                m = df[h] == 1
                if m.any():
                    pcol_loc = 'p_' + h.split('_')[1]
                    fig_ts.add_trace(go.Scatter(
                        x=x[m], y=df.loc[m, pcol_loc],
                        mode='markers',
                        marker=dict(size=18, symbol='circle-open',
                                    line=dict(color=color, width=2.5)),
                        showlegend=False,
                        hovertemplate=(f'<b>Actual event</b> ({HORIZON_LABEL[h]})<br>'
                                       'Hour: %{x:02d}:00<extra></extra>'),
                    ))
        fig_ts.update_layout(
            height=380,
            margin=dict(l=10, r=10, t=10, b=40),
            xaxis=dict(title='Hour of day (UTC)', tickmode='array',
                       tickvals=list(range(0, 24, 2)),
                       ticktext=[f'{h:02d}:00' for h in range(0, 24, 2)]),
            yaxis=dict(title='Probability', tickformat='.0%', range=[0, 1]),
            legend=dict(orientation='h', y=1.05, x=1, xanchor='right'),
            hovermode='x unified',
        )
        st.plotly_chart(fig_ts, use_container_width=True)

        # ---------- drivers ----------
        st.markdown('### Why does the model predict that?')
        st.caption(
            'Each bar shows how strongly that input pushed the prediction up '
            '(blue) or down (red) at the peak hour.'
        )

        drv_horizon = st.radio(
            'Forecast window to explain',
            P.HORIZONS, index=P.HORIZONS.index('y_24h'),
            format_func=lambda h: HORIZON_LABEL[h],
            horizontal=True,
        )

        X_day = features[(features['town'] == town) &
                         (features['ts'].dt.normalize() == pd.Timestamp(deep_date))][fcols].copy()
        if X_day.empty:
            st.warning('No model inputs available for this town/date.')
        else:
            pcol3 = 'p_' + drv_horizon.split('_')[1]
            peak_idx = int(np.argmax(df[pcol3].values))
            peak_ts3  = df.iloc[peak_idx]['ts']
            peak_p3   = float(df.iloc[peak_idx][pcol3])

            x_peak = X_day.iloc[[peak_idx]].values
            contrib = models[drv_horizon].predict(x_peak, pred_contrib=True)[0]
            bias, contrib = float(contrib[-1]), contrib[:-1]

            ctr = pd.DataFrame({'feature': fcols, 'contribution': contrib})
            is_t = ctr['feature'].str.startswith('town_')
            town_total = float(ctr.loc[is_t, 'contribution'].sum())
            ctr = pd.concat(
                [ctr[~is_t],
                 pd.DataFrame({'feature': ['town_*'],
                               'contribution': [town_total]})],
                ignore_index=True,
            )
            ctr['abs']   = ctr['contribution'].abs()
            ctr['label'] = ctr['feature'].apply(pretty)
            top = ctr.sort_values('abs', ascending=False).head(12).iloc[::-1]

            colA, colB = st.columns([2, 1])
            with colA:
                st.metric(f'Peak {HORIZON_LABEL[drv_horizon]} probability',
                          f"{peak_p3:.0%}",
                          f"at {peak_ts3.strftime('%H:%M')} UTC")
                colors = [COLOR_ACCENT if v > 0 else '#ff3030'
                          for v in top['contribution']]
                fig_shap = go.Figure(go.Bar(
                    x=top['contribution'], y=top['label'],
                    orientation='h',
                    marker=dict(color=colors, line=dict(color='rgba(0,0,0,0.3)', width=0.5)),
                    hovertemplate=('<b>%{y}</b><br>'
                                   'Push on log-odds: %{x:+.3f}<extra></extra>'),
                ))
                fig_shap.add_vline(x=0, line_color=COLOR_TEXT_MUTED, line_width=0.6)
                fig_shap.update_layout(
                    height=460,
                    margin=dict(l=10, r=10, t=20, b=40),
                    xaxis=dict(title='Push on prediction (blue = up, red = down)'),
                    yaxis=dict(automargin=True),
                )
                st.plotly_chart(fig_shap, use_container_width=True)
            with colB:
                st.markdown('**Numbers at the peak hour**')
                row = X_day.iloc[peak_idx]
                disp_rows = []
                for raw_name in top['feature'][::-1]:
                    if raw_name == 'town_*':
                        continue
                    if raw_name in row.index:
                        disp_rows.append({
                            'Feature': pretty(raw_name),
                            'Value':   float(row[raw_name]),
                        })
                disp = pd.DataFrame(disp_rows).set_index('Feature')
                st.dataframe(disp.style.format({'Value': '{:.2f}'}),
                             use_container_width=True, height=460)

# ---------------- footer methodology ----------------
with st.expander('How this model works (and what to watch out for)'):
    st.markdown(
        '* **Three models, one per look-ahead window** (1h, 6h, 24h). LightGBM, '
        'isotonic-calibrated on the validation fold.\n'
        '* **No leakage at the 24-hour horizon.** Every input is shifted back 24 hours '
        'before any rolling window.\n'
        '* **Time-based split.** Train Jan 2024 - Jun 2025 · validate Jul - Dec 2025 · '
        'test Jan - Mar 2026.\n'
        '* **Test PR-AUC** — 0.231 / 0.325 / 0.439 for 1h / 6h / 24h '
        '(4-5× better than random).\n'
        '* **Limitation, version 1.** Weather + grid inputs are *actuals* (Berlin proxy '
        'for weather). In production you would feed in day-ahead forecasts. '
        'Real-world accuracy will drop a few percentage points.'
    )
