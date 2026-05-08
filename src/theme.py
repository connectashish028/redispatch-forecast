"""theme.py — Operator Dark design system.

Drop the project on `THEME.md` (root) and apply it everywhere except the
Tab 2 7-day × 24-hour heatmap, which keeps its existing palette per spec.

Provides:
    inject_css(st)             registers global CSS overrides for Streamlit primitives.
    register_plotly_template() installs an `operator_dark` Plotly template + activates it.

Single source of truth for all colours used by `app.py`. The legacy
`COLOR_*` names are kept as aliases pointing at the new tokens so existing
inline-HTML strings remain valid without touching every f-string.
"""
from __future__ import annotations

# ----------------------------------------------------------------------
# Operator Dark palette
# ----------------------------------------------------------------------
BG               = '#1f2228'                       # warm near-black, blue undertone
TEXT             = '#ffffff'
TEXT_70          = 'rgba(255, 255, 255, 0.70)'
TEXT_50          = 'rgba(255, 255, 255, 0.50)'
TEXT_30          = 'rgba(255, 255, 255, 0.30)'
BORDER           = 'rgba(255, 255, 255, 0.10)'
BORDER_STRONG    = 'rgba(255, 255, 255, 0.20)'
SURFACE          = 'rgba(255, 255, 255, 0.03)'
HOVER_BG         = '#2a2d35'

PREDICTION       = '#B8A1FF'                       # lilac — model output (v2)
PREDICTION_FILL  = 'rgba(184, 161, 255, 0.18)'
ACTUAL           = '#3B82F6'                       # blue — realised values
TSO              = 'rgba(255, 255, 255, 0.45)'     # baseline / dimmed white-dashed

# ----------------------------------------------------------------------
# Backwards-compatible aliases — older sections of app.py (inline HTML
# strings, KPI panels, etc.) reference these names. Keep them in sync.
# ----------------------------------------------------------------------
COLOR_BG         = BG
COLOR_SURFACE    = SURFACE
COLOR_TEXT       = TEXT
COLOR_TEXT_MUTED = TEXT_50
COLOR_ACCENT     = ACTUAL
COLOR_RING       = BORDER_STRONG
COLOR_FROSTED    = 'rgba(255, 255, 255, 0.04)'
COLOR_GRID       = BORDER

# ----------------------------------------------------------------------
# Heat scales — preserved palettes for severity encodings.
# ----------------------------------------------------------------------
# Tab 2 7-day × 24-hour activity heatmap. Untouched per user request — its
# existing dark-cool-to-warm-hot ramp is what they wanted to keep.
HEAT_SCALE = [
    [0.00, 'rgba(10, 10, 10, 0)'],
    [0.10, '#1f3a5f'],
    [0.30, '#0099ff'],
    [0.55, '#ff7f0e'],
    [0.80, '#ff3030'],
    [1.00, '#ffeb3b'],
]

# Severity heat scale for the daily map bubbles. ColorBrewer YlOrRd —
# functional severity encoding, reads instantly on dark mapbox tiles.
# Operator Dark's "two accents max" rule applies to UI chrome, not to
# data-encoding palettes (same exemption as the heatmap).
HEAT_SCALE_MAP = [
    [0.00, '#FEF0D9'],
    [0.20, '#FDD49E'],
    [0.40, '#FDBB84'],
    [0.60, '#FC8D59'],
    [0.80, '#E34A33'],
    [1.00, '#B30000'],
]


# ----------------------------------------------------------------------
# CSS — injected once at the top of app.py via `inject_css(st)`
# ----------------------------------------------------------------------
CSS = f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@300;400;500&family=Inter:wght@400;500;600&display=swap');

html, body, [class*="st-"], [class*="css-"] {{
    font-family: 'Inter', system-ui, -apple-system, sans-serif !important;
    background-color: {BG} !important;
    color: {TEXT} !important;
}}
.stApp, section.main {{ background-color: {BG} !important; }}
[data-testid="stHeader"] {{ background: transparent !important; height: 0 !important; }}
#MainMenu {{ visibility: hidden; }}
footer {{ visibility: hidden; }}
.stDeployButton {{ display: none; }}

.block-container {{
    padding-top: 3rem !important;
    padding-bottom: 6rem !important;
    max-width: 1200px !important;
}}

/* -------- typography -------- */
h1, h2, h3, h4 {{
    font-family: 'Inter', sans-serif !important;
    font-weight: 400 !important;
    color: {TEXT} !important;
    letter-spacing: -0.01em !important;
}}
h1 {{ font-size: 2.25rem !important; line-height: 1.1 !important; margin-bottom: 0.25rem !important; }}
h2 {{ font-size: 1.5rem !important; line-height: 1.2 !important; margin-top: 3rem !important; margin-bottom: 1rem !important; }}
h3 {{ font-size: 1rem !important; }}

p, li, .stMarkdown p, .stMarkdown li {{
    font-family: 'Inter', sans-serif !important;
    color: {TEXT_70} !important;
    line-height: 1.6 !important;
    font-size: 0.95rem;
}}
.stMarkdown strong {{ color: {TEXT} !important; }}

/* Captions & label-like helper text use mono uppercase per spec */
[data-testid="stCaptionContainer"] {{
    color: {TEXT_50} !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.72rem !important;
    letter-spacing: 0.08em !important;
    text-transform: uppercase !important;
}}

code, .mono {{
    font-family: 'JetBrains Mono', ui-monospace, monospace !important;
    font-size: 0.85rem !important;
    letter-spacing: 0.04em !important;
    color: {TEXT} !important;
}}

/* -------- inputs -------- */
input, select, textarea, .stDateInput input,
.stSelectbox div[data-baseweb="select"] > div,
.stMultiSelect div[data-baseweb="select"] > div,
[data-baseweb="input"], [data-baseweb="textarea"] {{
    background-color: transparent !important;
    border: 1px solid {BORDER_STRONG} !important;
    border-radius: 0 !important;
    color: {TEXT} !important;
    font-family: 'JetBrains Mono', monospace !important;
}}
.stDateInput label, .stSelectbox label, .stMultiSelect label,
.stCheckbox label, .stRadio label, .stSlider label,
.stToggle label {{
    color: {TEXT_70} !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.72rem !important;
    letter-spacing: 0.08em !important;
    text-transform: uppercase !important;
    font-weight: 400 !important;
}}

/* slider thumb + track */
[data-baseweb="slider"] [role="slider"] {{
    background-color: {ACTUAL} !important;
    border: 0 !important;
    border-radius: 0 !important;
    box-shadow: none !important;
}}

/* -------- buttons -------- */
.stButton > button, .stDownloadButton > button {{
    background-color: transparent !important;
    color: {TEXT} !important;
    border: 1px solid {BORDER_STRONG} !important;
    border-radius: 0 !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-weight: 400 !important;
    font-size: 0.875rem !important;
    letter-spacing: 0.1em !important;
    text-transform: uppercase !important;
    padding: 0.625rem 1.25rem !important;
    transition: all 0.15s ease !important;
    box-shadow: none !important;
}}
.stButton > button:hover, .stDownloadButton > button:hover {{
    background-color: rgba(255, 255, 255, 0.05) !important;
    border-color: {TEXT} !important;
}}

/* -------- metric tiles -------- */
[data-testid="stMetric"] {{
    background-color: {SURFACE};
    border: 1px solid {BORDER};
    border-radius: 0;
    padding: 1.25rem 1rem;
    box-shadow: none !important;
}}
[data-testid="stMetricLabel"] {{
    color: {TEXT_50} !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.7rem !important;
    text-transform: uppercase !important;
    letter-spacing: 0.12em !important;
    font-weight: 400 !important;
}}
[data-testid="stMetricValue"] {{
    color: {TEXT} !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 1.75rem !important;
    font-weight: 300 !important;
    line-height: 1 !important;
    letter-spacing: 0.02em !important;
}}
[data-testid="stMetricDelta"] {{
    color: {ACTUAL} !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.75rem !important;
    letter-spacing: 0.04em !important;
}}

/* -------- tabs -------- */
.stTabs [data-baseweb="tab-list"] {{
    gap: 0;
    background-color: transparent;
    border-bottom: 1px solid {BORDER};
}}
.stTabs [data-baseweb="tab"] {{
    background-color: transparent;
    color: {TEXT_50};
    border: none;
    border-radius: 0 !important;
    padding: 0.75rem 1.25rem !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.72rem !important;
    letter-spacing: 0.1em !important;
    text-transform: uppercase !important;
    font-weight: 400 !important;
}}
.stTabs [data-baseweb="tab"]:hover {{
    background-color: rgba(255, 255, 255, 0.03);
    color: {TEXT};
}}
.stTabs [aria-selected="true"] {{
    color: {TEXT} !important;
    border-bottom: 1px solid {TEXT} !important;
    background-color: transparent !important;
    box-shadow: none !important;
}}

/* -------- expanders -------- */
[data-testid="stExpander"] details {{
    background-color: {SURFACE};
    border: 1px solid {BORDER};
    border-radius: 0;
}}
[data-testid="stExpander"] summary {{
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.78rem !important;
    letter-spacing: 0.06em !important;
    text-transform: uppercase !important;
    color: {TEXT_70} !important;
}}
/* Streamlit's expander caret is a Material Symbols ligature ("arrow_drop_down")
   rendered by an icon font. When the font fails to load (CSP, blocked CDN,
   strict mode), the ligature text leaks through as readable letters next to
   the label — that's the 'arr' some users see. Hide every non-text child of
   summary (the icon container and any svg) and let the disclosure state read
   purely from <details open>. */
[data-testid="stExpander"] summary > div:not([data-testid="stMarkdownContainer"]),
[data-testid="stExpander"] summary span.material-icons,
[data-testid="stExpander"] summary span[class*="material" i],
[data-testid="stExpander"] summary [data-testid*="Icon" i],
[data-testid="stExpander"] summary svg {{
    display: none !important;
}}

/* -------- dataframes -------- */
[data-testid="stDataFrame"] {{
    background-color: {SURFACE};
    border-radius: 0;
    border: 1px solid {BORDER};
}}

/* -------- alerts (info/warning/error/success) -------- */
[data-testid="stAlert"] {{
    background-color: {SURFACE};
    border: 1px solid {BORDER_STRONG};
    border-radius: 0;
    color: {TEXT};
}}

hr {{
    border: none !important;
    border-top: 1px solid {BORDER} !important;
    margin: 3rem 0 2rem 0 !important;
}}

a {{
    color: {TEXT} !important;
    text-decoration: underline;
    text-decoration-color: {BORDER_STRONG};
    text-underline-offset: 4px;
    transition: text-decoration-color 0.15s ease;
}}
a:hover {{
    text-decoration-color: {TEXT};
    color: {TEXT_50} !important;
}}

.modebar {{ filter: invert(1) hue-rotate(180deg) opacity(0.4); }}
</style>
"""


def inject_css(st) -> None:
    """Drop this once at the top of the app body (after `st.set_page_config`)."""
    st.markdown(CSS, unsafe_allow_html=True)


def register_plotly_template() -> None:
    """Register and activate the `operator_dark` Plotly template."""
    import plotly.io as pio
    import plotly.graph_objects as go

    axis = dict(
        showgrid=True, gridcolor=BORDER, gridwidth=1,
        zeroline=False, color=TEXT_70,
        linecolor=BORDER, tickcolor=BORDER,
        tickfont=dict(family='JetBrains Mono, monospace', size=11, color=TEXT_70),
        title_font=dict(family='JetBrains Mono, monospace', size=11, color=TEXT_50),
    )
    template = go.layout.Template(
        layout=dict(
            paper_bgcolor=BG,
            plot_bgcolor=BG,
            font=dict(family='Inter, sans-serif', color=TEXT, size=12),
            colorway=[ACTUAL, PREDICTION, TSO],
            xaxis=axis,
            yaxis=axis,
            legend=dict(
                orientation='h',
                yanchor='bottom', y=1.02,
                xanchor='left', x=0,
                font=dict(family='JetBrains Mono, monospace', size=10, color=TEXT_70),
                bgcolor='rgba(0, 0, 0, 0)',
                bordercolor=BORDER,
                borderwidth=0,
            ),
            hoverlabel=dict(
                bgcolor=HOVER_BG,
                bordercolor=BORDER,
                font=dict(family='JetBrains Mono, monospace', color=TEXT, size=11),
            ),
            map=dict(style='carto-darkmatter'),
        )
    )
    pio.templates['operator_dark'] = template
    pio.templates.default = 'operator_dark'
