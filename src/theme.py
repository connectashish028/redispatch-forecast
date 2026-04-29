"""
theme.py - Framer-inspired dark theme helpers.

Provides:
  - inject_css(st) : registers global CSS overrides for Streamlit primitives.
  - register_plotly_template() : adds a 'framer_dark' Plotly template + sets it as default.

Colour reference (single source of truth):
  - Void Black       #000000   page background
  - Near Black       #0a0a0a   elevated surface
  - Pure White       #ffffff   primary text
  - Muted Silver     #a6a6a6   secondary text
  - Framer Blue      #0099ff   accent (links, focus, alerts)
  - Blue Glow        rgba(0, 153, 255, 0.15)   ring shadow
  - Frosted White    rgba(255, 255, 255, 0.08) translucent surfaces
"""
from __future__ import annotations

# Public colour tokens — import these in app.py for chart styling.
COLOR_BG          = '#000000'
COLOR_SURFACE     = '#0a0a0a'
COLOR_TEXT        = '#ffffff'
COLOR_TEXT_MUTED  = '#a6a6a6'
COLOR_ACCENT      = '#0099ff'
COLOR_RING        = 'rgba(0, 153, 255, 0.15)'
COLOR_FROSTED     = 'rgba(255, 255, 255, 0.08)'
COLOR_GRID        = 'rgba(255, 255, 255, 0.06)'

# Probability heat scale (cool -> hot) that pops on pure black.
# Used for the Tab 2 heatmap (dark Plotly background).
HEAT_SCALE = [
    [0.00, 'rgba(10, 10, 10, 0)'],   # near black for ~0%
    [0.10, '#1f3a5f'],   # deep blue
    [0.30, '#0099ff'],   # Framer Blue
    [0.55, '#ff7f0e'],   # orange
    [0.80, '#ff3030'],   # bright red
    [1.00, '#ffeb3b'],   # high alert yellow
]

# Light-tile heat scale, for the daily map (`map_style='carto-positron'`).
# ColorBrewer YlOrRd ramp — pale cream at 0h stays visible on light tiles,
# climbs through orange to deep red at 24h. Standard for severity maps.
HEAT_SCALE_MAP = [
    [0.00, '#FEF0D9'],
    [0.20, '#FDD49E'],
    [0.40, '#FDBB84'],
    [0.60, '#FC8D59'],
    [0.80, '#E34A33'],
    [1.00, '#B30000'],
]


CSS = f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Space+Grotesk:wght@500;600;700&display=swap');

/* ---------- base typography ---------- */
html, body, [class*="css"] {{
    font-family: 'Inter', -apple-system, system-ui, sans-serif !important;
    background-color: {COLOR_BG} !important;
    color: {COLOR_TEXT} !important;
    font-feature-settings: 'cv11', 'ss03';
}}

/* main app container */
.stApp {{ background-color: {COLOR_BG} !important; }}
section.main {{ background-color: {COLOR_BG} !important; }}
[data-testid="stHeader"] {{ background-color: {COLOR_BG} !important; }}

/* ---------- headings (Space Grotesk = GT Walsheim stand-in) ---------- */
h1, h2, h3, h4 {{
    font-family: 'Space Grotesk', 'Inter', sans-serif !important;
    font-weight: 600 !important;
    color: {COLOR_TEXT} !important;
}}
h1 {{ font-size: 2.6rem !important; letter-spacing: -0.04em !important; line-height: 1.05 !important; }}
h2 {{ font-size: 2.0rem !important; letter-spacing: -0.035em !important; line-height: 1.10 !important; }}
h3 {{ font-size: 1.35rem !important; letter-spacing: -0.025em !important; line-height: 1.15 !important; }}
h4 {{ font-size: 1.05rem !important; letter-spacing: -0.015em !important; }}

/* ---------- markdown body text ---------- */
.stMarkdown p, .stMarkdown li {{
    color: {COLOR_TEXT_MUTED} !important;
    font-size: 0.95rem;
    line-height: 1.55;
}}
.stMarkdown strong {{ color: {COLOR_TEXT} !important; }}
.stCaption, [data-testid="stCaptionContainer"] {{
    color: {COLOR_TEXT_MUTED} !important; font-size: 0.85rem !important;
}}

/* ---------- metric cards ---------- */
[data-testid="stMetric"] {{
    background-color: {COLOR_SURFACE};
    border: 1px solid {COLOR_RING};
    border-radius: 12px;
    padding: 16px 18px;
    box-shadow: rgba(255,255,255,0.04) 0px 0.5px 0px 0.5px,
                rgba(0,0,0,0.5) 0px 8px 24px;
}}
[data-testid="stMetricLabel"] {{
    color: {COLOR_TEXT_MUTED} !important;
    font-size: 0.75rem !important;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    font-weight: 500;
}}
[data-testid="stMetricValue"] {{
    color: {COLOR_TEXT} !important;
    font-family: 'Space Grotesk', 'Inter', sans-serif !important;
    font-size: 1.7rem !important;
    font-weight: 600 !important;
    letter-spacing: -0.02em !important;
}}
[data-testid="stMetricDelta"] {{
    color: {COLOR_ACCENT} !important;
    font-size: 0.8rem !important;
}}

/* ---------- buttons (pill shape, Framer style) ---------- */
.stButton > button, .stDownloadButton > button {{
    background-color: {COLOR_FROSTED};
    color: {COLOR_TEXT};
    border: 1px solid rgba(255,255,255,0.12);
    border-radius: 999px !important;
    padding: 8px 22px !important;
    font-weight: 500;
    font-size: 0.9rem;
    transition: all 0.15s ease;
}}
.stButton > button:hover, .stDownloadButton > button:hover {{
    background-color: rgba(255,255,255,0.16);
    border-color: {COLOR_ACCENT};
    box-shadow: 0 0 0 3px {COLOR_RING};
}}

/* ---------- inputs ---------- */
[data-baseweb="input"], [data-baseweb="select"], [data-baseweb="textarea"] {{
    background-color: {COLOR_SURFACE} !important;
    border: 1px solid rgba(255,255,255,0.1) !important;
    border-radius: 10px !important;
}}
[data-baseweb="input"]:focus-within, [data-baseweb="select"]:focus-within {{
    border-color: {COLOR_ACCENT} !important;
    box-shadow: 0 0 0 3px {COLOR_RING} !important;
}}

/* slider */
[data-baseweb="slider"] [role="slider"] {{
    background-color: {COLOR_ACCENT} !important;
    border: 2px solid {COLOR_BG} !important;
    box-shadow: 0 0 0 4px {COLOR_RING} !important;
}}

/* ---------- tabs ---------- */
.stTabs [data-baseweb="tab-list"] {{
    gap: 4px;
    background-color: transparent;
    border-bottom: 1px solid rgba(255,255,255,0.08);
}}
.stTabs [data-baseweb="tab"] {{
    background-color: transparent;
    color: {COLOR_TEXT_MUTED};
    border: none;
    border-radius: 999px !important;
    padding: 8px 18px !important;
    font-weight: 500;
    font-size: 0.92rem;
    transition: all 0.15s;
}}
.stTabs [data-baseweb="tab"]:hover {{
    background-color: {COLOR_FROSTED};
    color: {COLOR_TEXT};
}}
.stTabs [aria-selected="true"] {{
    background-color: {COLOR_FROSTED} !important;
    color: {COLOR_TEXT} !important;
    box-shadow: inset 0 0 0 1px {COLOR_RING};
}}

/* ---------- expanders ---------- */
[data-testid="stExpander"] details {{
    background-color: {COLOR_SURFACE};
    border: 1px solid {COLOR_RING};
    border-radius: 12px;
}}

/* ---------- dataframes ---------- */
[data-testid="stDataFrame"] {{
    background-color: {COLOR_SURFACE};
    border-radius: 12px;
    border: 1px solid {COLOR_RING};
}}

/* ---------- alert boxes (info / warning / error) ---------- */
[data-testid="stAlert"] {{
    background-color: {COLOR_SURFACE};
    border: 1px solid {COLOR_RING};
    border-radius: 12px;
    color: {COLOR_TEXT};
}}

/* ---------- block container & spacing ---------- */
.block-container {{
    max-width: 1280px;
    padding-top: 2.5rem !important;
    padding-bottom: 4rem;
}}
hr {{ border-color: rgba(255,255,255,0.08) !important; }}
</style>
"""


def inject_css(st):
    """Drop this once at the top of the app body."""
    st.markdown(CSS, unsafe_allow_html=True)


def register_plotly_template():
    """Register and activate a 'framer_dark' Plotly template."""
    import plotly.io as pio
    import plotly.graph_objects as go

    template = go.layout.Template(
        layout=dict(
            paper_bgcolor=COLOR_BG,
            plot_bgcolor=COLOR_BG,
            font=dict(family='Inter, system-ui, sans-serif',
                      color=COLOR_TEXT_MUTED, size=12),
            title=dict(font=dict(family='Space Grotesk, Inter, sans-serif',
                                 color=COLOR_TEXT, size=15)),
            colorway=[COLOR_ACCENT, '#ff7f0e', '#d62728', '#2ca02c',
                      '#9467bd', '#8c564b', '#e377c2'],
            xaxis=dict(gridcolor=COLOR_GRID, zerolinecolor=COLOR_GRID,
                       linecolor=COLOR_GRID, tickcolor=COLOR_GRID,
                       color=COLOR_TEXT_MUTED),
            yaxis=dict(gridcolor=COLOR_GRID, zerolinecolor=COLOR_GRID,
                       linecolor=COLOR_GRID, tickcolor=COLOR_GRID,
                       color=COLOR_TEXT_MUTED),
            legend=dict(bgcolor='rgba(0,0,0,0)', bordercolor=COLOR_GRID,
                        borderwidth=1, font=dict(color=COLOR_TEXT)),
            hoverlabel=dict(bgcolor=COLOR_SURFACE, bordercolor=COLOR_ACCENT,
                            font=dict(family='Inter', color=COLOR_TEXT, size=12)),
            map=dict(style='carto-positron'),
        )
    )
    pio.templates['framer_dark'] = template
    pio.templates.default = 'framer_dark'
