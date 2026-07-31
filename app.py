import streamlit as st
import pdfplumber
import base64
import re
import zipfile
import logging
from html import escape as _html_escape
from io import BytesIO
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS
from sklearn.metrics.pairwise import cosine_similarity

# Technical error details (raw exception text, stack traces) are logged
# here instead of ever being shown to the end user — the UI only ever
# displays short, friendly fallback messages. Check your terminal /
# server logs for the real cause if the AI features fall back.
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("ai_career_copilot")

try:
    import plotly.graph_objects as go
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False

from gemini_helper import (
    gemini_available,
    gemini_status_message,
    generate_full_analysis,
    generate_bullet_rewrite,
)
from pdf_report import generate_pdf_report, generate_simple_pdf
from email_utils import (
    email_credentials_available,
    email_status_message,
    send_email_report,
)

# ---------------- PAGE CONFIGURATION ----------------

st.set_page_config(
    page_title="AI Career Copilot",
    page_icon="🧭",
    layout="wide"
)

# ---------------- DESIGN SYSTEM (CSS) ----------------
# Palette matches the existing Plotly charts (#4F46E5 etc.) and the
# PDF report generator's colors, so native UI, charts, and exported
# PDFs all feel like one product instead of three different tools.
#
# NOTE: deliberately avoids two fragile patterns that caused real
# rendering problems in an earlier version of this file:
#   1. `[class*="css"]` — matches ANY element with "css" anywhere in
#      its class list, which collides unpredictably with Streamlit's
#      own auto-generated internal class names.
#   2. CSS `:has()` selectors for card styling — inconsistent support
#      meant cards silently rendered with no border/shadow at all.
# Every rule below targets stable `data-testid` attributes directly.

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800;900&display=swap');
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500&display=swap');

:root {
    --cc-bg: #0F1117;
    --cc-bg-elevated: #12151D;
    --cc-sidebar: #161B22;
    --cc-surface: #1E2530;
    --cc-surface-hover: #242C3A;
    --cc-border: #334155;
    --cc-border-soft: rgba(51, 65, 85, 0.55);

    --cc-primary: #4F46E5;
    --cc-primary-dark: #3730A3;
    --cc-primary-light: rgba(79, 70, 229, 0.16);
    --cc-secondary: #7C3AED;
    --cc-highlight: #06B6D4;

    --cc-success: #22C55E;
    --cc-success-light: rgba(34, 197, 94, 0.12);
    --cc-warning: #F59E0B;
    --cc-warning-light: rgba(245, 158, 11, 0.12);
    --cc-danger: #EF4444;
    --cc-danger-light: rgba(239, 68, 68, 0.12);

    --cc-text: #F8FAFC;
    --cc-text-muted: #CBD5E1;
    --cc-text-faint: #94A3B8;

    --cc-radius-lg: 18px;
    --cc-radius-md: 12px;
    --cc-radius-sm: 8px;
    --cc-shadow-card: 0 8px 28px -6px rgba(0, 0, 0, 0.45), 0 1px 0 rgba(255,255,255,0.02) inset;
    --cc-shadow-card-hover: 0 14px 38px -8px rgba(79, 70, 229, 0.35);
    --cc-transition: all 0.22s cubic-bezier(0.4, 0, 0.2, 1);
}

/* ================= BASE / RESET ================= */

/* NOTE: stAppViewContainer is deliberately NOT in this shorthand rule.
   `background: X !important` resets background-image to none as part
   of the shorthand, which was silently overriding the dot-grid pattern
   set on stAppViewContainer further down — regardless of source order,
   since !important always wins over a non-!important declaration at
   equal specificity. This was the actual reason the dot-grid was
   invisible, not the color or opacity. */
html, body, .stApp, [data-testid="stHeader"] {
    background: var(--cc-bg) !important;
}

.stApp, .stApp p, .stApp span, .stApp div, .stApp label, .stApp li,
.stMarkdown, [data-testid="stMetricValue"], [data-testid="stMetricLabel"],
[data-testid="stMetricDelta"], .stCaption, [data-testid="stCaptionContainer"] {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    color: var(--cc-text);
}

/* Streamlit renders every built-in icon (sidebar collapse chevron,
   expander arrows, file-uploader icon, etc.) as ligature text through
   the Material Symbols icon font — e.g. the literal text
   "keyboard_arrow_down" is what *becomes* the arrow glyph once that
   font loads. The broad Inter override above was being applied to
   those spans too, which killed the ligature and left the raw text
   sitting on screen, overlapping the visible label next to it. This
   rule restores the icon font for exactly those elements; higher
   specificity than the rule above so it always wins.
   Covers every icon variant Streamlit uses across versions:
   stIconMaterial (single icons), the expander/uploader material-icon
   spans, and the raw font-family fallbacks. */
.stApp [data-testid="stIconMaterial"],
.stApp [data-testid="stExpanderToggleIcon"],
.stApp [data-testid*="Icon"],
.stApp span[class*="material-symbols"],
.stApp span[class*="material-icons"],
.stApp [data-testid="stStatusWidget"] svg,
.stApp [data-testid="stStatusWidget"] span:first-child {
    font-family: 'Material Symbols Rounded', 'Material Symbols Outlined', 'Material Icons' !important;
    color: inherit;
}

/* Premium background system: a solid dark base with a very low-opacity
   dot-grid texture for subtle depth (a la Linear/Vercel/Framer), kept
   deliberately faint so it never competes with text or chart
   readability. Colored "aurora" glow is intentionally NOT applied
   page-wide — it lives only behind the hero section (see .cc-hero
   ::before below) so the rest of the app stays calm and neutral. */
[data-testid="stAppViewContainer"] {
    background-color: var(--cc-bg);
    background-image: radial-gradient(circle, rgba(129, 140, 248, 0.20) 1.6px, transparent 1.6px);
    background-size: 32px 32px;
    background-position: 0 0;
    background-attachment: fixed;
}

[data-testid="stHeader"] {
    background: transparent !important;
}

.block-container {
    padding-top: 1.75rem;
    padding-bottom: 3rem;
    max-width: 1280px;
}

h1, h2, h3, h4 {
    font-family: 'Inter', sans-serif;
    color: var(--cc-text) !important;
    font-weight: 800;
    letter-spacing: -0.02em;
}

hr { border-color: var(--cc-border-soft); opacity: 1; }

::selection { background: var(--cc-primary); color: #fff; }

.stApp a { color: var(--cc-highlight); font-weight: 600; text-decoration: none; }
.stApp a:hover { color: var(--cc-primary); text-decoration: underline; }

/* Custom scrollbar */
::-webkit-scrollbar { width: 10px; height: 10px; }
::-webkit-scrollbar-track { background: var(--cc-bg); }
::-webkit-scrollbar-thumb { background: var(--cc-border); border-radius: 8px; }
::-webkit-scrollbar-thumb:hover { background: var(--cc-primary); }

/* Fade-in entrance animation for top-level blocks */
@keyframes cc-fade-up {
    from { opacity: 0; transform: translateY(10px); }
    to { opacity: 1; transform: translateY(0); }
}
[data-testid="stVerticalBlockBorderWrapper"], .cc-hero-wrap {
    animation: cc-fade-up 0.45s ease both;
}

/* ================= HERO ================= */

/* The aurora glow lives on this wrapper, not on .cc-hero itself — that
   way it paints as a true layer BEHIND the hero card (and can bleed
   softly beyond its rounded edges into the page background), rather
   than being painted on top of the hero's own opaque gradient. This
   glow is intentionally scoped to only the hero; nothing else in the
   app gets a color glow. */
.cc-hero-wrap {
    position: relative;
    isolation: isolate;
}
.cc-hero-wrap::before {
    content: "";
    position: absolute;
    inset: -70px -50px;
    background:
        radial-gradient(circle at 18% 25%, rgba(79, 70, 229, 0.50), transparent 60%),
        radial-gradient(circle at 82% 75%, rgba(124, 58, 237, 0.42), transparent 60%);
    filter: blur(64px);
    z-index: 0;
    pointer-events: none;
}

.cc-hero {
    position: relative;
    z-index: 1;
    overflow: hidden;
    background:
        radial-gradient(circle at 85% 20%, rgba(6,182,212,0.35), transparent 55%),
        linear-gradient(135deg, var(--cc-primary) 0%, var(--cc-secondary) 100%);
    border-radius: var(--cc-radius-lg);
    padding: 3rem 3.25rem;
    margin-bottom: 1.5rem;
    color: white;
    box-shadow: 0 20px 50px -12px rgba(79, 70, 229, 0.55);
    border: 1px solid rgba(255,255,255,0.08);
}
.cc-hero-eyebrow {
    display: inline-flex;
    align-items: center;
    gap: 0.4rem;
    background: rgba(255,255,255,0.12);
    border: 1px solid rgba(255,255,255,0.28);
    padding: 0.3rem 0.9rem;
    border-radius: 999px;
    font-size: 0.76rem;
    font-weight: 700;
    letter-spacing: 0.06em;
    text-transform: uppercase;
    margin-bottom: 1.1rem;
    backdrop-filter: blur(6px);
}
.cc-hero h1 {
    color: #FFFFFF !important;
    font-size: 2.75rem;
    margin: 0 0 0.6rem 0;
    line-height: 1.08;
    font-weight: 900;
    letter-spacing: -0.03em;
}
.cc-hero p {
    color: rgba(255,255,255,0.92);
    font-size: 1.08rem;
    max-width: 680px;
    margin: 0 0 1.25rem 0;
    line-height: 1.6;
}
.cc-hero-badges { display: flex; flex-wrap: wrap; gap: 0.5rem; }
.cc-hero-badge {
    background: rgba(255,255,255,0.10);
    border: 1px solid rgba(255,255,255,0.20);
    color: #fff;
    font-size: 0.78rem;
    font-weight: 600;
    padding: 0.35rem 0.75rem;
    border-radius: 999px;
}

/* ================= METRIC CARDS ================= */

.cc-metric-card {
    background: linear-gradient(160deg, var(--cc-surface) 0%, var(--cc-bg-elevated) 100%);
    border: 1px solid var(--cc-border-soft);
    border-radius: var(--cc-radius-md);
    padding: 1.35rem 1.5rem;
    box-shadow: var(--cc-shadow-card);
    height: 100%;
    transition: var(--cc-transition);
}
.cc-metric-card:hover {
    transform: translateY(-3px);
    border-color: var(--cc-primary);
    box-shadow: var(--cc-shadow-card-hover);
}
.cc-metric-icon { font-size: 1.5rem; margin-bottom: 0.5rem; }
.cc-metric-value { font-size: 2rem; font-weight: 800; color: var(--cc-text); line-height: 1.1; }
.cc-metric-value-compact { font-size: 1.2rem; font-weight: 700; color: var(--cc-text-faint); line-height: 1.3; padding-top: 0.3rem; }
.cc-metric-label { font-size: 0.85rem; color: var(--cc-text-faint); font-weight: 500; margin-top: 0.2rem; }
.cc-metric-sublabel { font-size: 0.74rem; color: var(--cc-text-faint); opacity: 0.85; margin-top: 0.35rem; font-weight: 500; }

/* ================= STATUS PILLS ================= */

.cc-pill {
    display: inline-flex; align-items: center; gap: 0.4rem;
    padding: 0.4rem 0.9rem; border-radius: 999px;
    font-size: 0.82rem; font-weight: 600;
    border: 1px solid transparent;
}
.cc-pill-success { background: var(--cc-success-light); color: #4ADE80; border-color: rgba(34,197,94,0.35); }
.cc-pill-warning { background: var(--cc-warning-light); color: #FBBF24; border-color: rgba(245,158,11,0.35); }

/* ================= BUTTONS ================= */

.stButton > button, .stDownloadButton > button {
    border-radius: var(--cc-radius-sm);
    font-weight: 600;
    border: 1px solid var(--cc-border);
    background: var(--cc-surface);
    color: var(--cc-text);
    box-shadow: 0 1px 2px rgba(0,0,0,0.3);
    transition: var(--cc-transition);
}
.stButton > button:hover, .stDownloadButton > button:hover {
    border-color: var(--cc-primary);
    color: #fff;
    background: var(--cc-primary-light);
    transform: translateY(-1px);
}
.stButton > button:active, .stDownloadButton > button:active { transform: translateY(0); }
.stButton > button[kind="primary"] {
    background: linear-gradient(135deg, var(--cc-primary), var(--cc-secondary));
    border-color: transparent;
    color: white;
    box-shadow: 0 6px 18px -4px rgba(79, 70, 229, 0.55);
}
.stButton > button[kind="primary"]:hover {
    box-shadow: 0 10px 26px -4px rgba(79, 70, 229, 0.7);
    transform: translateY(-2px);
    color: white;
}

/* ================= TABS ================= */

.stTabs [data-baseweb="tab-list"] {
    gap: 4px;
    background: var(--cc-surface);
    padding: 6px;
    border-radius: 14px;
    border: 1px solid var(--cc-border-soft);
}
.stTabs [data-baseweb="tab"] {
    border-radius: 10px;
    font-weight: 600;
    color: var(--cc-text-faint);
    padding: 10px 18px;
    transition: var(--cc-transition);
}
.stTabs [data-baseweb="tab"]:hover { color: var(--cc-text); background: var(--cc-surface-hover); }
.stTabs [aria-selected="true"] {
    background: var(--cc-primary-light);
    color: #A5B4FC !important;
}
.stTabs [data-baseweb="tab-highlight"] { background-color: var(--cc-primary); }
.stTabs [data-baseweb="tab-border"] { display: none; }

/* ================= FILE UPLOADER ================= */

[data-testid="stFileUploaderDropzone"] {
    border: 1.5px dashed var(--cc-border);
    border-radius: var(--cc-radius-md);
    background: var(--cc-bg-elevated);
    transition: var(--cc-transition);
    padding: 0.5rem;
}
[data-testid="stFileUploaderDropzone"]:hover {
    border-color: var(--cc-primary);
    background: var(--cc-primary-light);
}
[data-testid="stFileUploaderDropzoneInstructions"] svg { fill: var(--cc-highlight); }
[data-testid="stFileUploaderDropzoneInstructions"] span,
[data-testid="stFileUploaderDropzoneInstructions"] small {
    color: var(--cc-text-muted) !important;
}
[data-testid="stFileUploader"] section button,
[data-testid="stFileUploaderDropzone"] button {
    border-radius: var(--cc-radius-sm);
    border: 1px solid var(--cc-border);
    background: var(--cc-surface);
    color: var(--cc-text) !important;
    font-weight: 600;
    transition: var(--cc-transition);
}
[data-testid="stFileUploader"] section button:hover,
[data-testid="stFileUploaderDropzone"] button:hover {
    border-color: var(--cc-primary);
    background: var(--cc-primary-light);
}
[data-testid="stFileUploaderFileName"] { color: var(--cc-text) !important; }
[data-testid="stFileUploader"] small { color: var(--cc-text-faint) !important; }

/* ================= TEXT INPUT / TEXT AREA / SELECT ================= */

.stTextInput input, .stTextArea textarea,
[data-baseweb="select"] > div, [data-baseweb="base-input"] {
    background: var(--cc-bg-elevated) !important;
    border: 1px solid var(--cc-border) !important;
    border-radius: var(--cc-radius-sm) !important;
    color: var(--cc-text) !important;
    transition: var(--cc-transition);
}
.stTextInput input:focus, .stTextArea textarea:focus,
[data-baseweb="select"] > div:focus-within {
    border-color: var(--cc-primary) !important;
    box-shadow: 0 0 0 3px var(--cc-primary-light) !important;
}
.stTextArea textarea::placeholder, .stTextInput input::placeholder { color: var(--cc-text-faint) !important; opacity: 1; }
[data-baseweb="popover"] ul, [data-baseweb="menu"] {
    background: var(--cc-surface) !important;
    border: 1px solid var(--cc-border) !important;
}
[data-baseweb="popover"] li, [role="option"] { color: var(--cc-text) !important; }
[role="option"]:hover, [aria-selected="true"][role="option"] { background: var(--cc-primary-light) !important; }
.stSelectbox svg { fill: var(--cc-text-faint) !important; }

/* ================= CARDS (native bordered containers) ================= */

[data-testid="stVerticalBlockBorderWrapper"] {
    border-radius: var(--cc-radius-lg);
    border: 1px solid var(--cc-border-soft);
    background: linear-gradient(160deg, var(--cc-surface) 0%, var(--cc-bg-elevated) 100%);
    box-shadow: var(--cc-shadow-card);
    backdrop-filter: blur(10px);
    transition: var(--cc-transition);
}
/* Columns (st.columns) default to flex with stretch alignment, but the
   card border-wrapper needs an explicit height:100% to actually fill
   that stretched space — otherwise cards in the same row hug their own
   content and end up visibly uneven heights (e.g. Upload Resume vs
   Job Description). Scoped to ONLY cards inside a horizontal block —
   applying this globally previously made every stacked card (cards
   NOT in a row, like the AI Tools tab sections) fight for height
   against their container, squeezing out the spacer divs between them
   and making adjacent cards visually touch. */
[data-testid="stHorizontalBlock"] [data-testid="stVerticalBlockBorderWrapper"] {
    height: 100%;
}
[data-testid="stHorizontalBlock"] > div[data-testid="column"] {
    display: flex;
}
[data-testid="stHorizontalBlock"] > div[data-testid="column"] > div {
    width: 100%;
}
[data-testid="stVerticalBlockBorderWrapper"]:hover {
    border-color: var(--cc-primary);
    transform: translateY(-4px);
    box-shadow: 0 15px 40px -10px rgba(79, 70, 229, 0.35);
}

/* ================= SECTION HEADER ================= */

.cc-section-header { display: flex; align-items: center; gap: 0.7rem; margin: 0 0 1rem 0; }
.cc-section-icon {
    width: 42px; height: 42px; border-radius: 12px;
    background: var(--cc-primary-light); color: var(--cc-highlight);
    display: flex; align-items: center; justify-content: center;
    font-size: 1.2rem; flex-shrink: 0;
    border: 1px solid var(--cc-border-soft);
}
.cc-section-title { font-size: 1.15rem; font-weight: 700; color: var(--cc-text); line-height: 1.25; margin: 0; }
.cc-section-subtitle { font-size: 0.85rem; color: var(--cc-text-faint); margin: 0.15rem 0 0 0; }

/* ================= SIDEBAR ================= */

[data-testid="stSidebar"] {
    background: var(--cc-sidebar) !important;
    border-right: 1px solid var(--cc-border-soft);
}
[data-testid="stSidebar"] > div { background: var(--cc-sidebar) !important; }
.cc-sidebar-brand {
    display: flex; align-items: center; gap: 0.65rem;
    margin-bottom: 0.15rem;
}
.cc-sidebar-brand-icon {
    width: 40px; height: 40px; border-radius: 11px;
    background: linear-gradient(135deg, var(--cc-primary), var(--cc-secondary));
    color: white;
    display: flex; align-items: center; justify-content: center;
    font-size: 1.25rem; flex-shrink: 0;
    box-shadow: 0 4px 14px -2px rgba(79,70,229,0.6);
}
.cc-sidebar-brand-text { font-weight: 800; font-size: 1.05rem; color: var(--cc-text); line-height: 1.2; }
.cc-sidebar-brand-sub { font-size: 0.78rem; color: var(--cc-text-faint); }
.cc-chip-group { display: flex; flex-wrap: wrap; gap: 6px; margin: 0.5rem 0 0.25rem 0; }
.cc-chip {
    background: var(--cc-bg-elevated); border: 1px solid var(--cc-border-soft);
    color: var(--cc-text-muted); font-size: 0.74rem; font-weight: 500;
    padding: 0.3rem 0.65rem; border-radius: 999px; white-space: nowrap;
    transition: var(--cc-transition);
}
.cc-chip:hover { border-color: var(--cc-primary); color: var(--cc-text); }
.cc-sidebar-group-label {
    font-size: 0.72rem; font-weight: 700; letter-spacing: 0.06em;
    text-transform: uppercase; color: var(--cc-text-faint);
    margin: 1.1rem 0 0.4rem 0;
}

/* Compact connection-status cards (replace default st.success/warning blocks) */
.cc-status-card {
    display: flex; align-items: center; gap: 0.6rem;
    padding: 0.55rem 0.75rem; margin-bottom: 22px;
    border-radius: var(--cc-radius-sm);
    border: 1px solid var(--cc-border-soft);
    background: var(--cc-bg-elevated);
    font-size: 0.8rem; font-weight: 600; color: var(--cc-text-muted);
}
.cc-status-dot { width: 8px; height: 8px; border-radius: 50%; flex-shrink: 0; }
.cc-status-dot-ok { background: var(--cc-success); box-shadow: 0 0 8px var(--cc-success); }
.cc-status-dot-warn { background: var(--cc-warning); box-shadow: 0 0 8px var(--cc-warning); }

/* Sidebar feature-group expanders: quieter than the default expander so
   they read as part of the sidebar, not a separate widget */
[data-testid="stSidebar"] [data-testid="stExpander"] {
    background: transparent; border: none; margin-bottom: 0;
}
[data-testid="stSidebar"] [data-testid="stExpander"] summary {
    font-size: 0.74rem !important; font-weight: 800 !important;
    letter-spacing: 0.07em; text-transform: uppercase;
    color: var(--cc-text-muted) !important; padding: 0.3rem 0 !important;
}
[data-testid="stSidebar"] [data-testid="stExpander"] summary:hover {
    color: var(--cc-highlight) !important;
}
[data-testid="stSidebar"] .block-container,
[data-testid="stSidebar"] [data-testid="stVerticalBlock"] {
    gap: 0.4rem;
}
/* Every st.markdown/st.caption call in the sidebar (status cards,
   chip groups, etc.) sits in its own Streamlit-managed wrapper that
   carries its own default bottom margin on top of the flex gap above
   — the two were stacking, which is what made the gap between e.g.
   the Gemini and Email status cards look larger than the rest of the
   sidebar. Zeroing it here means the flex `gap` is the single source
   of truth for spacing. */
[data-testid="stSidebar"] [data-testid="stElementContainer"],
[data-testid="stSidebar"] .element-container {
    margin-bottom: 0 !important;
}

/* ================= ALERTS (info/success/warning/error) ================= */

[data-testid="stAlert"] {
    border-radius: var(--cc-radius-sm);
    border: 1px solid var(--cc-border-soft);
    background: var(--cc-bg-elevated);
}
div[data-baseweb="notification"] { color: var(--cc-text) !important; }

/* ================= METRIC (native st.metric) ================= */

[data-testid="stMetric"] {
    background: var(--cc-bg-elevated);
    border: 1px solid var(--cc-border-soft);
    border-radius: var(--cc-radius-md);
    padding: 0.9rem 1.1rem;
}
[data-testid="stMetricValue"] { color: var(--cc-text) !important; font-weight: 800; }
[data-testid="stMetricLabel"] { color: var(--cc-text-faint) !important; }

/* ================= PROGRESS BAR ================= */

.stProgress > div > div {
    background: var(--cc-bg-elevated) !important;
    border-radius: 999px;
    overflow: hidden;
}
.stProgress > div > div > div {
    background: linear-gradient(90deg, var(--cc-primary), var(--cc-highlight)) !important;
    transition: width 0.6s cubic-bezier(0.4, 0, 0.2, 1);
}

/* ================= STATUS / EXPANDER ================= */

[data-testid="stExpander"] {
    background: var(--cc-bg-elevated);
    border: 1px solid var(--cc-border-soft);
    border-radius: var(--cc-radius-md);
}
[data-testid="stExpander"] summary { color: var(--cc-text) !important; }

div[data-testid="stStatusWidget"], [data-testid="stExpander"] > details {
    background: var(--cc-bg-elevated);
    border: 1px solid var(--cc-border-soft);
    border-radius: var(--cc-radius-md);
}

/* ================= PLOTLY CHARTS ================= */

.js-plotly-plot .plotly, .stPlotlyChart { border-radius: var(--cc-radius-md); overflow: hidden; }

/* ================= MISC SPACING ================= */

.cc-section-spacer { height: 1rem; }

/* ================= CHECKLIST GRID ================= */

.cc-check-grid {
    display: grid;
    grid-template-columns: repeat(var(--cc-check-cols, 2), 1fr);
    gap: 8px;
    margin: 0.25rem 0 0.5rem 0;
}
.cc-check-item {
    display: flex; align-items: center; gap: 0.55rem;
    padding: 0.55rem 0.75rem;
    border-radius: var(--cc-radius-sm);
    border: 1px solid var(--cc-border-soft);
    background: var(--cc-bg-elevated);
    font-size: 0.86rem; font-weight: 500;
    transition: var(--cc-transition);
}
.cc-check-item:hover { border-color: var(--cc-border); transform: translateY(-1px); }
.cc-check-item-icon {
    display: flex; align-items: center; justify-content: center;
    width: 20px; height: 20px; border-radius: 50%;
    font-size: 0.72rem; font-weight: 800; flex-shrink: 0;
}
.cc-check-item-done { color: var(--cc-text-muted); }
.cc-check-item-done .cc-check-item-icon { background: var(--cc-success-light); color: var(--cc-success); }
.cc-check-item-missing { color: var(--cc-text-faint); }
.cc-check-item-missing .cc-check-item-icon { background: var(--cc-danger-light); color: var(--cc-danger); }

@media (max-width: 768px) {
    .cc-check-grid { grid-template-columns: 1fr; }
}

/* ================= EMPTY STATES ================= */

.cc-empty-state {
    display: flex; flex-direction: column; align-items: center; text-align: center;
    gap: 0.5rem; padding: 2rem 1.25rem;
    border: 1px dashed var(--cc-border);
    border-radius: var(--cc-radius-md);
    background: var(--cc-bg-elevated);
    color: var(--cc-text-faint);
}
.cc-empty-state-icon { font-size: 1.75rem; opacity: 0.85; }
.cc-empty-state-title { font-size: 0.95rem; font-weight: 700; color: var(--cc-text-muted); }
.cc-empty-state-desc { font-size: 0.82rem; max-width: 340px; line-height: 1.5; }

/* ================= SETUP CARD (e.g. email not configured) ================= */

.cc-setup-card {
    display: flex; align-items: flex-start; gap: 0.9rem;
    padding: 1.1rem 1.25rem;
    border-radius: var(--cc-radius-md);
    border: 1px solid var(--cc-border-soft);
    background: linear-gradient(160deg, var(--cc-bg-elevated) 0%, var(--cc-surface) 100%);
}
.cc-setup-card-icon {
    width: 38px; height: 38px; border-radius: 10px; flex-shrink: 0;
    background: var(--cc-warning-light); color: var(--cc-warning);
    display: flex; align-items: center; justify-content: center; font-size: 1.05rem;
}
.cc-setup-card-title { font-weight: 700; color: var(--cc-text); font-size: 0.95rem; margin: 0 0 0.2rem 0; }
.cc-setup-card-desc { color: var(--cc-text-faint); font-size: 0.83rem; line-height: 1.55; margin: 0; }
.cc-setup-card-code {
    display: inline-block; margin-top: 0.5rem;
    background: var(--cc-bg); border: 1px solid var(--cc-border-soft); border-radius: 6px;
    padding: 0.2rem 0.5rem; font-family: 'JetBrains Mono', monospace; font-size: 0.76rem;
    color: var(--cc-highlight);
}



/* ================= FOOTER ================= */

.cc-footer {
    display: flex; flex-direction: column; align-items: center;
    text-align: center; padding: 0.5rem 0 0.25rem 0; gap: 0.35rem;
}
.cc-footer-divider {
    width: 100%; height: 1px; margin-bottom: 1.25rem;
    background: linear-gradient(90deg, transparent, var(--cc-border), transparent);
}
.cc-footer-brand { display: flex; align-items: center; gap: 0.5rem; }
.cc-footer-icon { font-size: 1.1rem; }
.cc-footer-name { font-weight: 800; color: var(--cc-text); font-size: 0.95rem; }
.cc-footer-version {
    font-size: 0.68rem; font-weight: 700; color: var(--cc-text-faint);
    border: 1px solid var(--cc-border-soft); border-radius: 999px;
    padding: 0.1rem 0.5rem;
}
.cc-footer-sub { font-size: 0.8rem; color: var(--cc-text-faint); }
.cc-footer-links { display: flex; align-items: center; gap: 0.4rem; margin-top: 0.15rem; font-size: 0.8rem; }
.cc-footer-links a { color: var(--cc-text-faint) !important; font-weight: 500 !important; }
.cc-footer-links a:hover { color: var(--cc-highlight) !important; }
.cc-footer-dot { color: var(--cc-border); }

/* ================= DIFF / BEFORE-AFTER BLOCKS ================= */

.cc-diff-block {
    border-radius: var(--cc-radius-sm);
    border: 1px solid var(--cc-border-soft);
    overflow: hidden;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.86rem;
    margin: 0.4rem 0 1rem 0;
}
.cc-diff-top { border-bottom: none; border-bottom-left-radius: 0; border-bottom-right-radius: 0; margin-bottom: 0; }
.cc-diff-bottom { border-top: none; border-top-left-radius: 0; border-top-right-radius: 0; margin-top: 0; }
.cc-diff-line {
    display: flex; gap: 0.65rem; align-items: flex-start;
    padding: 0.6rem 0.9rem; line-height: 1.5;
}
.cc-diff-line-removed { background: rgba(239, 68, 68, 0.10); color: #FCA5A5; }
.cc-diff-line-added { background: rgba(34, 197, 94, 0.10); color: #86EFAC; border-top: 1px solid var(--cc-border-soft); }
.cc-diff-marker { flex-shrink: 0; font-weight: 700; opacity: 0.8; }
.cc-diff-text { font-family: 'Inter', sans-serif; color: inherit; word-break: break-word; }
.cc-diff-empty { display: flex; align-items: center; gap: 0.5rem; padding: 0.6rem 0.9rem; background: var(--cc-bg-elevated); color: var(--cc-text-faint); font-family: 'Inter', sans-serif; font-size: 0.82rem; border-top: 1px dashed var(--cc-border-soft); }

/* ================= RESPONSIVE ================= */

@media (max-width: 768px) {
    .cc-hero { padding: 2rem 1.5rem; }
    .cc-hero h1 { font-size: 2rem; }
    .cc-metric-value { font-size: 1.5rem; }
    .block-container { padding-left: 1rem; padding-right: 1rem; }
}
</style>
""", unsafe_allow_html=True)


def render_metric_card(icon, value, label, accent="var(--cc-primary)", compact_value=False, sublabel=None):
    value_class = "cc-metric-value cc-metric-value-compact" if compact_value else "cc-metric-value"
    sublabel_html = f'<div class="cc-metric-sublabel">{sublabel}</div>' if sublabel else ""
    html = (
        f'<div class="cc-metric-card" style="border-top: 3px solid {accent};">'
        f'<div class="cc-metric-icon">{icon}</div>'
        f'<div class="{value_class}">{value}</div>'
        f'<div class="cc-metric-label">{label}</div>'
        f'{sublabel_html}'
        f'</div>'
    )
    st.markdown(html, unsafe_allow_html=True)


def render_checklist_grid(items: dict, columns: int = 2):
    """Render a dict of {label: bool} as a responsive grid of small
    checklist cards instead of a plain markdown bullet list."""
    cells = []
    for label, present in items.items():
        state_class = "cc-check-item-done" if present else "cc-check-item-missing"
        icon = "✓" if present else "✕"
        cells.append(
            f'<div class="cc-check-item {state_class}">'
            f'<span class="cc-check-item-icon">{icon}</span>'
            f'<span class="cc-check-item-label">{label}</span>'
            f'</div>'
        )
    grid_html = f'<div class="cc-check-grid" style="--cc-check-cols:{columns};">{"".join(cells)}</div>'
    st.markdown(grid_html, unsafe_allow_html=True)


def render_empty_state(icon, title, desc=""):
    """Premium empty-state card, used wherever a section has nothing to
    show yet (no JD, no bullets detected, no project section, etc.)
    instead of a plain st.info() line."""
    desc_html = f'<div class="cc-empty-state-desc">{desc}</div>' if desc else ""
    st.markdown(
        f'<div class="cc-empty-state">'
        f'<div class="cc-empty-state-icon">{icon}</div>'
        f'<div class="cc-empty-state-title">{title}</div>'
        f'{desc_html}'
        f'</div>',
        unsafe_allow_html=True,
    )


def section_header(icon, title, subtitle=None):
    """Consistent icon + title (+ optional subtitle) header used at the
    top of every card/section throughout the app, replacing plain
    st.subheader() calls for a unified, intentional look.
    Built as a single-line HTML string on purpose: a multi-line
    f-string whose content lines are indented (as they naturally are
    inside a function body) gets misread by Streamlit's Markdown
    parser as an indented code block, which silently swallows the
    HTML instead of rendering it. A single line can't trigger that."""
    subtitle_html = f'<div class="cc-section-subtitle">{subtitle}</div>' if subtitle else ""
    html = (
        f'<div class="cc-section-header">'
        f'<div class="cc-section-icon">{icon}</div>'
        f'<div><p class="cc-section-title">{title}</p>{subtitle_html}</div>'
        f'</div>'
    )
    st.markdown(html, unsafe_allow_html=True)


def card_marker():
    """No-op kept for backward compatibility with existing call sites.
    Card styling now targets the native container testid directly and
    uniformly, so no marker element is needed any more."""
    pass


# ---------------- SIDEBAR ----------------

with st.sidebar:
    st.markdown(
        '<div class="cc-sidebar-brand"><div class="cc-sidebar-brand-icon">🧭</div>'
        '<div><div class="cc-sidebar-brand-text">AI Career Copilot</div>'
        '<div class="cc-sidebar-brand-sub">Your AI career companion</div></div></div>',
        unsafe_allow_html=True,
    )

    st.markdown("---")

    feature_groups = {
        "Analysis": ["📄 PDF Parsing", "🧠 Skill Detection", "🔗 JD Match", "📊 ATS Score", "✅ ATS Checklist", "📋 Section Checker"],
        "Optimization": ["💡 Suggestions", "⚠️ Missing Skills", "🔑 JD Keywords", "🔍 Keyword Density", "🧠 Keyword Optimizer", "🎨 Formatting Check", "📖 Readability", "🛠️ Project Quality"],
        "AI Generation": ["📝 Resume Summary", "❓ Interview Prep", "✨ Bullet Rewriter", "✉️ Cover Letter", "🎯 Tailored Resume"],
        "Export": ["📑 PDF Report", "📧 Email Delivery", "⭐ Strength Meter", "📦 Career Package"],
    }
    for group_label, chips in feature_groups.items():
        with st.expander(group_label, expanded=False):
            chips_html = "".join(f'<span class="cc-chip">{c}</span>' for c in chips)
            st.markdown(f'<div class="cc-chip-group">{chips_html}</div>', unsafe_allow_html=True)

    st.markdown("---")
    st.caption("🔌 Connection Status")

    gemini_ok = gemini_available()
    gemini_dot = "cc-status-dot-ok" if gemini_ok else "cc-status-dot-warn"
    st.markdown(
        f'<div class="cc-status-card"><span class="cc-status-dot {gemini_dot}"></span>'
        f'<span>{gemini_status_message()}</span></div>',
        unsafe_allow_html=True,
    )

    email_ok = email_credentials_available()
    email_dot = "cc-status-dot-ok" if email_ok else "cc-status-dot-warn"
    st.markdown(
        f'<div class="cc-status-card"><span class="cc-status-dot {email_dot}"></span>'
        f'<span>{email_status_message()}</span></div>',
        unsafe_allow_html=True,
    )

# ---------------- MAIN HEADER / HERO ----------------

st.markdown(
    '<div class="cc-hero-wrap">'
    '<div class="cc-hero">'
    '<div class="cc-hero-eyebrow">🧭 AI Career Copilot</div>'
    '<h1>Land your next role, faster.</h1>'
    '<p>Upload your resume and let AI score it against real ATS criteria, close '
    'your keyword gaps, sharpen every bullet point, and generate a tailored '
    'resume, cover letter, and interview prep kit — all in one place.</p>'
    '<div class="cc-hero-badges">'
    '<span class="cc-hero-badge">⚡ Instant ATS Score</span>'
    '<span class="cc-hero-badge">🎯 JD Keyword Matching</span>'
    '<span class="cc-hero-badge">✨ AI Bullet Rewriter</span>'
    '<span class="cc-hero-badge">✉️ Cover Letter Generator</span>'
    '</div>'
    '</div>'
    '</div>',
    unsafe_allow_html=True,
)

upload_col, jd_col = st.columns([1, 1.3], gap="medium")

with upload_col:
    with st.container(border=True):
        card_marker()
        section_header("📂", "Upload Your Resume", "PDF only — drag &amp; drop or click to browse")
        uploaded_file = st.file_uploader(
            "Upload Your Resume (PDF)",
            type=["pdf"],
            label_visibility="collapsed",
        )
        if uploaded_file is not None:
            st.markdown(
                '<div class="cc-pill cc-pill-success">✔ Resume uploaded successfully</div>',
                unsafe_allow_html=True,
            )

        st.markdown('<div style="height: 0.9rem"></div>', unsafe_allow_html=True)
        job_role = st.selectbox(
            "🎯 Target Job Role",
            [
                "Data Analyst",
                "ML Engineer",
                "Software Developer"
            ]
        )

with jd_col:
    with st.container(border=True):
        card_marker()
        section_header("💼", "Job Description", "Optional — unlocks JD matching, tailored resume, and more")

        job_description = st.text_area(
            "Paste the Job Description here",
            height=170,
            placeholder="Paste Job Description here...\n\nPress Ctrl + Enter to Analyze",
            label_visibility="collapsed",
        )

        jd_file = st.file_uploader(
            "Or Upload Job Description (.txt or .pdf)",
            type=["txt", "pdf"]
        )

st.markdown('<div style="height: 0.75rem"></div>', unsafe_allow_html=True)

# ---------------- SKILL DATABASE ----------------
# Broader master list used for detection in both resume and JD

skills_db = [
    "python", "java", "c++", "c", "sql", "r",
    "machine learning", "deep learning", "nlp",
    "tensorflow", "pytorch", "keras", "scikit-learn",
    "pandas", "numpy", "git", "docker", "kubernetes",
    "aws", "azure", "gcp", "rest api", "flask", "django",
    "html", "css", "javascript", "react", "node.js",
    "mysql", "mongodb", "postgresql", "power bi", "tableau",
    "excel"
]

# Fallback generic recommendations, used only when no JD is provided
recommended = [
    "python",
    "sql",
    "git",
    "docker",
    "aws"
]

# ---------------- HELPER FUNCTIONS ----------------

@st.cache_data(show_spinner=False)
def detect_skills(source_text, skill_list):
    source_lower = source_text.lower()
    found = []
    for skill in skill_list:
        if skill in source_lower:
            found.append(skill)
    return found


def check_section(source_text, keywords):
    source_lower = source_text.lower()
    for kw in keywords:
        if kw in source_lower:
            return True
    return False


def get_match_quality(pct):
    if pct >= 75:
        return "Excellent", "success"
    elif pct >= 55:
        return "Good", "success"
    elif pct >= 35:
        return "Average", "warning"
    else:
        return "Poor", "error"


def stars_from_score(score):
    if score < 20:
        n = 1
    elif score < 40:
        n = 2
    elif score < 60:
        n = 3
    elif score < 80:
        n = 4
    else:
        n = 5
    return "★" * n + "☆" * (5 - n)


_JD_FILLER_WORDS = {
    "looking", "plus", "strong", "experience", "role", "description",
    "responsibilities", "join", "team", "opportunity", "required",
    "preferred", "years", "work", "ability", "including", "using",
    "candidate", "position", "job", "company", "skills", "knowledge",
    "understanding", "excellent", "good", "strongly", "requirements",
    "qualifications", "environment", "related", "field", "new",
}


@st.cache_data(show_spinner=False)
def extract_jd_keywords(resume_text, jd_text, top_n=20):
    """
    Extract the most important keywords/phrases from the job description
    using local TF-IDF (no Gemini call needed). Returns a list of
    keyword strings, ranked by relevance.
    """
    try:
        vectorizer = TfidfVectorizer(
            stop_words="english", ngram_range=(1, 2), max_features=80
        )
        matrix = vectorizer.fit_transform([resume_text, jd_text])
        feature_names = vectorizer.get_feature_names_out()
        jd_weights = matrix[1].toarray()[0]

        scored = sorted(
            zip(feature_names, jd_weights), key=lambda pair: pair[1], reverse=True
        )
        keywords = []
        for kw, weight in scored:
            if weight <= 0 or len(kw) <= 2 or kw.replace(" ", "").isdigit():
                continue
            if all(word in _JD_FILLER_WORDS for word in kw.split()):
                continue
            keywords.append(kw)
        return keywords[:top_n]
    except Exception:
        return []


SECTION_CHECKER_SUGGESTIONS = {
    "Contact Information": "Add a clear email address and phone number near the top of your resume.",
    "Professional Summary": "Add a 2-3 sentence professional summary or objective at the top summarizing your expertise.",
    "Education": "Add an Education section listing your degree(s), institution(s), and graduation year(s).",
    "Experience": "Add a Work Experience or Internship section with role, company, and achievements.",
    "Projects": "Add a Projects section showcasing relevant hands-on work.",
    "Skills": "Add a dedicated Skills section listing your technical and relevant soft skills.",
    "Certifications": "Add any relevant certifications you've completed.",
    "Achievements": "Add an Achievements/Awards section highlighting notable recognitions.",
    "GitHub": "Add a link to your GitHub profile to showcase your code.",
    "LinkedIn": "Add a link to your LinkedIn profile for professional networking.",
}


def extract_bullet_candidates(resume_text, min_words=3, max_words=40, limit=20):
    """
    Heuristically extract bullet-point-like lines from resume text
    (PDF extraction often strips bullet symbols, so this looks at line
    length/shape rather than requiring a literal bullet character).
    """
    lines = [line.strip() for line in resume_text.split("\n")]
    candidates = []
    for line in lines:
        if not line:
            continue
        stripped = line.lstrip("•-*◦▪‣○ \t")
        word_count = len(stripped.split())
        if word_count < min_words or word_count > max_words:
            continue
        if stripped.isupper():
            continue
        if stripped.endswith(":"):
            continue
        if "@" in stripped or "http" in stripped.lower():
            continue
        candidates.append(stripped)

    seen = set()
    unique = []
    for c in candidates:
        if c not in seen:
            seen.add(c)
            unique.append(c)
    return unique[:limit]


# ---------------- KEYWORD DENSITY ANALYSIS (Feature 7) ----------------

@st.cache_data(show_spinner=False)
def analyze_keyword_density(resume_text, jd_text, found_skills, skills_db_list, top_n=15, overused_threshold=6):
    """
    Local, rule-based keyword density analysis (no Gemini call):
    frequent keywords, overused words, missing important keywords, and
    a keyword relevance score vs the JD (or general skills DB if no JD).
    """
    words = re.findall(r"[a-zA-Z][a-zA-Z\+\#\.]{1,}", resume_text.lower())
    filtered = [w.strip(".") for w in words if w not in ENGLISH_STOP_WORDS and len(w) > 2]

    counts = Counter(filtered)
    frequent_keywords = counts.most_common(top_n)
    overused_words = sorted(
        [(w, c) for w, c in counts.items() if c >= overused_threshold],
        key=lambda pair: pair[1], reverse=True,
    )

    if jd_text:
        jd_keywords = extract_jd_keywords(resume_text, jd_text, top_n=20)
        missing_keywords = [kw for kw in jd_keywords if kw.lower() not in resume_text.lower()]
        relevance_score = (
            round(((len(jd_keywords) - len(missing_keywords)) / len(jd_keywords)) * 100, 1)
            if jd_keywords else 0.0
        )
    else:
        missing_keywords = [s for s in skills_db_list if s not in found_skills][:15]
        relevance_score = (
            round((len(found_skills) / len(skills_db_list)) * 100, 1)
            if skills_db_list else 0.0
        )

    return {
        "frequent_keywords": frequent_keywords,
        "overused_words": overused_words,
        "missing_keywords": missing_keywords,
        "relevance_score": relevance_score,
    }


# ---------------- RESUME FORMATTING CHECKER (Feature 8) ----------------

FORMATTING_SUGGESTIONS = {
    "Long Paragraphs": "Break long, dense paragraphs into concise bullet points for easier ATS parsing and readability.",
    "Too Many Colors": "Stick to 1-2 accent colors (plus black/dark grey text) for a clean, ATS-friendly design.",
    "Tables Detected": "Avoid tables for key content — many ATS parsers read tables incorrectly or skip them. Use plain text sections instead.",
    "Images/Icons Detected": "Remove decorative images/icons — ATS systems typically cannot read text embedded in images.",
    "Header/Footer Issues": "Avoid placing important content (name, contact info, key sections) in repeating headers/footers — some ATS systems skip these regions.",
    "Multi-Column Layout": "Avoid multi-column layouts — many ATS parsers read strictly left-to-right and can scramble multi-column content.",
    "Font Inconsistencies": "Use a single consistent font family throughout the resume for a clean, professional look.",
}


def _detect_long_paragraphs(text, avg_words_threshold=11):
    lines = [line.strip() for line in text.split("\n") if line.strip()]
    body_lines = [line for line in lines if len(line.split()) >= 6]
    if len(body_lines) < 3:
        return False
    bullet_lines = sum(1 for line in body_lines if line.startswith(("•", "-", "*", "◦", "‣")))
    non_bullet_lines = [line for line in body_lines if not line.startswith(("•", "-", "*", "◦", "‣"))]
    if len(non_bullet_lines) < 3:
        return False
    avg_words = sum(len(line.split()) for line in non_bullet_lines) / len(non_bullet_lines)
    bullet_ratio = bullet_lines / len(body_lines)
    return avg_words >= avg_words_threshold and bullet_ratio < 0.3


def _detect_multi_column(x_positions, page_width, min_points=20):
    if not x_positions or not page_width or len(x_positions) < min_points:
        return False
    mid_low = page_width * 0.35
    mid_high = page_width * 0.65
    total = len(x_positions)
    left = sum(1 for x in x_positions if x < mid_low)
    right = sum(1 for x in x_positions if x > mid_high)
    mid = sum(1 for x in x_positions if mid_low <= x <= mid_high)
    return (left > total * 0.25) and (right > total * 0.25) and (mid < total * 0.1)


def _detect_header_footer_repetition(pdf):
    if len(pdf.pages) < 2:
        return False
    first_lines, last_lines = [], []
    for page in pdf.pages:
        page_text = page.extract_text() or ""
        lines = [line.strip() for line in page_text.split("\n") if line.strip()]
        if lines:
            first_lines.append(lines[0])
            last_lines.append(lines[-1])
    first_counts = Counter(first_lines)
    last_counts = Counter(last_lines)
    return any(c >= 2 for c in first_counts.values()) or any(c >= 2 for c in last_counts.values())


@st.cache_data(show_spinner=False)
def analyze_pdf_formatting(pdf_bytes, extracted_text):
    """
    Detect common ATS-unfriendly formatting patterns from the PDF's raw
    bytes. Takes bytes (not a live pdfplumber object) specifically so
    this can be cached with st.cache_data — a pdfplumber object isn't a
    stable, hashable cache key, but bytes are. Heuristic/approximate by
    nature — PDF layout analysis can't be perfectly precise.
    """
    all_colors, all_fonts = set(), set()
    total_images, total_tables = 0, 0
    word_x_positions = []

    with pdfplumber.open(BytesIO(pdf_bytes)) as pdf:
        for page in pdf.pages:
            for ch in page.chars:
                color = ch.get("non_stroking_color")
                if color is not None:
                    all_colors.add(tuple(color) if isinstance(color, (list, tuple)) else color)
                font = ch.get("fontname")
                if font:
                    all_fonts.add(font)

            total_images += len(page.images)

            try:
                total_tables += len(page.find_tables())
            except Exception:
                pass

            try:
                words = page.extract_words()
                word_x_positions.extend([w["x0"] for w in words])
            except Exception:
                pass

        page_width = pdf.pages[0].width if pdf.pages else 612
        header_footer_issue = _detect_header_footer_repetition(pdf)

    return {
        "Long Paragraphs": _detect_long_paragraphs(extracted_text),
        "Too Many Colors": len(all_colors) > 4,
        "Tables Detected": total_tables > 0,
        "Images/Icons Detected": total_images > 0,
        "Header/Footer Issues": header_footer_issue,
        "Multi-Column Layout": _detect_multi_column(word_x_positions, page_width),
        "Font Inconsistencies": len(all_fonts) > 3,
    }


# ---------------- RESUME READABILITY ANALYSIS (Feature 9) ----------------
# Sentence length, readability score, passive voice, and repeated phrases
# are all computed locally (deterministic, reliable, no API cost).
# Grammar quality and AI suggestions come from Gemini as part of the
# combined analysis call.

def _count_syllables(word):
    word = re.sub(r"[^a-z]", "", word.lower())
    if not word:
        return 0
    vowels = "aeiouy"
    count, prev_was_vowel = 0, False
    for ch in word:
        is_vowel = ch in vowels
        if is_vowel and not prev_was_vowel:
            count += 1
        prev_was_vowel = is_vowel
    if word.endswith("e") and count > 1:
        count -= 1
    return max(count, 1)


@st.cache_data(show_spinner=False)
def compute_readability_stats(text):
    """Average sentence length + Flesch Reading Ease score, computed locally."""
    sentences = [s for s in re.split(r"(?<=[.!?])\s+", text.strip()) if len(s.split()) >= 3]
    words = re.findall(r"[A-Za-z']+", text)

    if not sentences or not words:
        return {
            "avg_sentence_length": 0, "readability_score": 0,
            "sentence_length_label": "Not enough text to analyze", "total_sentences": 0,
        }

    total_words = len(words)
    total_sentences = len(sentences)
    total_syllables = sum(_count_syllables(w) for w in words)
    avg_sentence_length = round(total_words / total_sentences, 1)

    flesch = 206.835 - 1.015 * (total_words / total_sentences) - 84.6 * (total_syllables / total_words)
    flesch = max(0, min(100, round(flesch, 1)))

    if avg_sentence_length < 10:
        label = "Short & punchy"
    elif avg_sentence_length <= 20:
        label = "Ideal length"
    else:
        label = "Too long — consider shortening"

    return {
        "avg_sentence_length": avg_sentence_length,
        "readability_score": flesch,
        "sentence_length_label": label,
        "total_sentences": total_sentences,
    }


@st.cache_data(show_spinner=False)
def detect_passive_voice(text, max_examples=5):
    """Heuristic passive-voice detection: be-verb + past participle."""
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    passive_pattern = re.compile(r"\b(am|is|are|was|were|be|been|being)\s+\w+(ed|en)\b", re.IGNORECASE)
    examples = []
    for s in sentences:
        s_clean = s.strip()
        if len(s_clean.split()) < 4:
            continue
        if passive_pattern.search(s_clean):
            examples.append(s_clean)
        if len(examples) >= max_examples:
            break
    return examples


@st.cache_data(show_spinner=False)
def detect_repeated_phrases(text, phrase_len=3, min_occurrences=3, max_results=8):
    """N-gram based repeated-phrase detection (e.g. overused 3-word phrases)."""
    words = [w for w in re.findall(r"[a-zA-Z']+", text.lower()) if w not in ENGLISH_STOP_WORDS]
    phrases = [" ".join(words[i:i + phrase_len]) for i in range(len(words) - phrase_len + 1)]
    counts = Counter(phrases)
    repeated = sorted(
        [(p, c) for p, c in counts.items() if c >= min_occurrences],
        key=lambda pair: pair[1], reverse=True,
    )
    return repeated[:max_results]


# ---------------- INTERACTIVE ATS DASHBOARD CHARTS (Feature 12) ----------------
# Built with Plotly. All charts are pure functions (no API calls).

_CHART_FONT_COLOR = "#CBD5E1"
_CHART_TITLE_COLOR = "#F8FAFC"
_CHART_GRID_COLOR = "#334155"


def _apply_dark_chart_theme(fig, title=None, height=320, margin=None):
    """Shared Plotly layout so the gauge/bar/pie/radar charts read as
    one consistent design system instead of four differently-styled
    widgets. Transparent backgrounds let the surrounding dark card
    show through instead of a jarring white chart panel."""
    layout_kwargs = dict(
        height=height,
        margin=margin or dict(l=20, r=20, t=50, b=20),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, sans-serif", color=_CHART_FONT_COLOR, size=12),
        legend=dict(font=dict(color=_CHART_FONT_COLOR)),
    )
    # Only set a layout title when one was actually passed in — Plotly.js
    # renders an explicit `title=None` as the literal text "undefined"
    # instead of simply omitting it, which is where that bug came from.
    if title:
        layout_kwargs["title"] = dict(text=title, font=dict(color=_CHART_TITLE_COLOR, size=15))
    fig.update_layout(**layout_kwargs)
    fig.update_xaxes(gridcolor=_CHART_GRID_COLOR, zerolinecolor=_CHART_GRID_COLOR, color=_CHART_FONT_COLOR)
    fig.update_yaxes(gridcolor=_CHART_GRID_COLOR, zerolinecolor=_CHART_GRID_COLOR, color=_CHART_FONT_COLOR)
    return fig


_CHART_CONFIG = {"displayModeBar": False, "displaylogo": False}


def build_gauge_chart(score):
    color = "#EF4444" if score < 50 else "#F59E0B" if score < 75 else "#22C55E"
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=score,
        number={"suffix": "%", "font": {"color": _CHART_TITLE_COLOR, "size": 36}},
        title={"text": "Overall ATS Score", "font": {"color": _CHART_TITLE_COLOR, "size": 16}},
        gauge={
            "axis": {"range": [0, 100], "tickcolor": _CHART_FONT_COLOR, "tickfont": {"color": _CHART_FONT_COLOR}},
            "bar": {"color": color, "thickness": 0.75},
            "bgcolor": "rgba(0,0,0,0)",
            "bordercolor": _CHART_GRID_COLOR,
            "steps": [
                {"range": [0, 50], "color": "rgba(239, 68, 68, 0.30)"},
                {"range": [50, 75], "color": "rgba(245, 158, 11, 0.30)"},
                {"range": [75, 100], "color": "rgba(34, 197, 94, 0.30)"},
            ],
        },
    ))
    return _apply_dark_chart_theme(fig, height=300, margin=dict(l=20, r=20, t=60, b=10))


def build_skills_pie_chart(found_count, missing_count):
    if found_count == 0 and missing_count == 0:
        found_count, missing_count = 1, 0
    total = found_count + missing_count
    present_pct = round((found_count / total) * 100, 1) if total else 0
    fig = go.Figure(go.Pie(
        labels=["Skills Present", "Skills Gap"],
        values=[found_count, missing_count],
        hole=0.55,
        marker=dict(colors=["#4F46E5", "#334155"], line=dict(color="#0F1117", width=2)),
        textfont=dict(color="#F8FAFC", size=13),
        textinfo="percent",
        showlegend=False,
    ))
    fig.update_layout(
        annotations=[dict(
            text=f"<b>{present_pct}%</b><br><span style='font-size:11px;color:#94A3B8'>Present</span>",
            x=0.5, y=0.5, showarrow=False, font=dict(color="#F8FAFC", size=20),
        )],
    )
    return _apply_dark_chart_theme(fig, title="Skills Coverage", height=320, margin=dict(l=10, r=10, t=40, b=10))


def build_section_scores_bar_chart(skills_component, jd_component, structure_component, contact_component):
    labels = ["Skills (40%)", "JD Match (30%)", "Structure (15%)", "Contact (15%)"]
    values = [skills_component, jd_component, structure_component, contact_component]
    colors = ["#4F46E5", "#7C3AED", "#06B6D4", "#22C55E"]
    fig = go.Figure(go.Bar(
        x=labels, y=values, marker_color=colors,
        text=[f"{v:.1f}%" for v in values], textposition="outside",
        textfont=dict(color=_CHART_TITLE_COLOR),
    ))
    fig.update_layout(yaxis=dict(range=[0, 115]))
    return _apply_dark_chart_theme(fig, title="Resume Section Scores", height=320, margin=dict(l=10, r=10, t=40, b=10))


def build_radar_chart(skills_score, experience_score, education_score, ats_compliance_score, jd_match_score_val):
    categories = ["Skills", "Experience", "Education", "ATS Compliance", "JD Match"]
    values = [skills_score, experience_score, education_score, ats_compliance_score, jd_match_score_val]
    fig = go.Figure(go.Scatterpolar(
        r=values + [values[0]],
        theta=categories + [categories[0]],
        fill="toself",
        line=dict(color="#818CF8", width=2.5),
        fillcolor="rgba(79,70,229,0.42)",
        marker=dict(size=6, color="#818CF8"),
    ))
    fig.update_layout(
        polar=dict(
            bgcolor="rgba(0,0,0,0)",
            radialaxis=dict(
                visible=True, range=[0, 100],
                gridcolor=_CHART_GRID_COLOR, linecolor=_CHART_GRID_COLOR,
                tickfont=dict(color=_CHART_FONT_COLOR),
            ),
            angularaxis=dict(gridcolor=_CHART_GRID_COLOR, linecolor=_CHART_GRID_COLOR, tickfont=dict(color=_CHART_FONT_COLOR, size=13)),
        ),
        showlegend=False,
    )
    return _apply_dark_chart_theme(fig, title="Overall Profile Radar", height=460, margin=dict(l=20, r=20, t=40, b=10))


# ---------------- COMPLETE CAREER PACKAGE (Feature 13) ----------------

def build_career_package_zip(
    report_pdf_bytes=None,
    report_txt=None,
    cover_letter_txt=None,
    cover_letter_pdf_bytes=None,
    tailored_resume_txt=None,
    tailored_resume_pdf_bytes=None,
):
    """
    Bundle every generated output into a single in-memory ZIP file.
    Any argument left as None is simply skipped, so the package always
    contains exactly what was successfully generated for this session.
    """
    buffer = BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as zf:
        if report_pdf_bytes:
            zf.writestr("AI_Resume_Report.pdf", report_pdf_bytes)
        if report_txt:
            zf.writestr("AI_Resume_Report.txt", report_txt)
        if cover_letter_pdf_bytes:
            zf.writestr("Cover_Letter.pdf", cover_letter_pdf_bytes)
        if cover_letter_txt:
            zf.writestr("Cover_Letter.txt", cover_letter_txt)
        if tailored_resume_pdf_bytes:
            zf.writestr("Tailored_Resume.pdf", tailored_resume_pdf_bytes)
        if tailored_resume_txt:
            zf.writestr("Tailored_Resume.txt", tailored_resume_txt)
    buffer.seek(0)
    return buffer.getvalue()


# ===========================================================
# ANALYZE ONLY AFTER PDF IS UPLOADED
# ===========================================================

if uploaded_file is not None:

    with st.status("🔍 AI is analyzing your resume...", expanded=True) as status:
        text = ""
        formatting_issues = {}
        formatting_check_error = None
        pdf_parse_error = None

        try:
            resume_pdf_bytes = uploaded_file.getvalue()
            with pdfplumber.open(BytesIO(resume_pdf_bytes)) as pdf:
                for page in pdf.pages:
                    text += page.extract_text() or ""
        except Exception as exc:
            logger.warning("PDF parsing failed: %s", exc, exc_info=True)
            pdf_parse_error = "this file couldn't be read as a PDF"

        if not pdf_parse_error:
            try:
                formatting_issues = analyze_pdf_formatting(resume_pdf_bytes, text)
            except Exception as exc:
                logger.warning("PDF formatting analysis failed: %s", exc, exc_info=True)
                formatting_check_error = "formatting analysis couldn't complete for this file"
        if pdf_parse_error:
            st.error(
                f"⚠️ Could not read this PDF — {pdf_parse_error}. It may be "
                "corrupted, password-protected, or not a valid PDF. "
                "Please try a different file."
            )
            st.stop()

        if not text.strip():
            st.warning(
                "⚠️ No readable text was found in this PDF. It may be a scanned "
                "image without a text layer, or empty. Try uploading a "
                "text-based PDF instead."
            )
            st.stop()

        text_lower = text.lower()
        status.write("✔️ Parsing Resume")

        # ---------------- JOB DESCRIPTION ----------------

        jd_text = job_description

        if jd_file is not None:
            try:
                if jd_file.type == "text/plain":
                    jd_text = jd_file.read().decode("utf-8", errors="ignore")

                elif jd_file.type == "application/pdf":
                    jd_text = ""
                    with pdfplumber.open(jd_file) as pdf:
                        for page in pdf.pages:
                            jd_text += page.extract_text() or ""
            except Exception as exc:
                logger.warning("JD file read failed: %s", exc, exc_info=True)
                st.warning(
                    "⚠️ Couldn't read the uploaded job description file — "
                    "falling back to the pasted text box, if any."
                )
                jd_text = job_description

        jd_provided = bool(jd_text and jd_text.strip())

        # ---------------- RESUME - JD MATCH SCORE ----------------

        jd_match_score = None

        if jd_provided:
            try:
                vectorizer = TfidfVectorizer(stop_words="english")
                tfidf_matrix = vectorizer.fit_transform([text, jd_text])
                similarity = cosine_similarity(
                    tfidf_matrix[0:1],
                    tfidf_matrix[1:2]
                )
                jd_match_score = round(similarity[0][0] * 100, 2)
            except Exception:
                jd_match_score = 0.0
        status.write("✔️ JD Matching" if jd_provided else "✔️ No job description provided — skipping JD Matching")

        # ---------------- SKILL DETECTION ----------------

        found_skills = detect_skills(text, skills_db)
        status.write("✔️ Detecting Skills")

        missing_source = "job description" if jd_provided else "general recommendations"

        # ---------------- ATS CHECKLIST ----------------

        email_found = bool(re.search(r"[\w\.-]+@[\w\.-]+\.\w+", text))
        phone_found = bool(re.search(r"(\+?\d[\d\-\s]{8,}\d)", text))
        name_found = bool(text.strip())
        github_found = "github.com" in text_lower or "github" in text_lower
        linkedin_found = "linkedin.com" in text_lower or "linkedin" in text_lower

        education_found = check_section(
            text, ["education", "b.tech", "bachelor", "master", "university", "college", "degree"]
        )
        projects_found = check_section(text, ["project"])
        experience_found = check_section(
            text, ["experience", "internship", "work experience", "employment"]
        )
        certifications_found = check_section(
            text, ["certification", "certified", "certificate"]
        )
        skills_section_found = len(found_skills) > 0

        professional_summary_found = check_section(
            text, ["summary", "professional summary", "objective", "profile"]
        )
        achievements_found = check_section(
            text, ["achievement", "achievements", "award", "awards", "honors", "accomplishment"]
        )
        contact_info_found = email_found and phone_found

        checklist = {
            "Name": name_found,
            "Email": email_found,
            "Phone": phone_found,
            "Skills": skills_section_found,
            "Education": education_found,
            "Projects": projects_found,
            "Experience": experience_found,
            "Certifications": certifications_found,
            "GitHub": github_found,
            "LinkedIn": linkedin_found,
        }

        section_checklist = {
            "Contact Information": contact_info_found,
            "Professional Summary": professional_summary_found,
            "Education": education_found,
            "Experience": experience_found,
            "Projects": projects_found,
            "Skills": skills_section_found,
            "Certifications": certifications_found,
            "Achievements": achievements_found,
            "GitHub": github_found,
            "LinkedIn": linkedin_found,
        }

        # ---------------- WEIGHTED ATS SCORE ----------------

        # Skills component (40%): capped ratio of detected skills vs master list
        skills_component = min((len(found_skills) / len(skills_db)) * 100 * 3, 100)

        # JD match component (30%): falls back to skills component if no JD given
        if jd_provided:
            jd_component = jd_match_score
        else:
            jd_component = skills_component

        # Structure component (15%): education, projects, experience, certifications
        structure_items = [education_found, projects_found, experience_found, certifications_found]
        structure_component = (sum(structure_items) / len(structure_items)) * 100

        # Contact component (15%): name, email, phone, github, linkedin
        contact_items = [name_found, email_found, phone_found, github_found, linkedin_found]
        contact_component = (sum(contact_items) / len(contact_items)) * 100

        ats_score = round(
            skills_component * 0.40
            + jd_component * 0.30
            + structure_component * 0.15
            + contact_component * 0.15,
            2
        )
        ats_score = min(ats_score, 100)
        status.write("✔️ ATS Analysis")

        # ---------------- COMBINED AI ANALYSIS (SINGLE GEMINI CALL) ----------------
        # Summary, ATS suggestions, resume recommendations, interview questions,
        # tailored resume, cover letter, AND missing-skill suggestions are all
        # generated in ONE Gemini request here (instead of 7 separate calls)
        # to stay well within free-tier daily quota limits.

        top_skills = ", ".join([s.title() for s in found_skills[:5]]) if found_skills else "no specific technical skills detected"
        name_preview = text.split("\n")[0] if text else "Not Found"

        # Local (no API cost) readability metrics, fed into the combined
        # Gemini call as grounding context for grammar_quality/readability_suggestions.
        readability_stats = compute_readability_stats(text)
        passive_voice_examples = detect_passive_voice(text)
        repeated_phrases = detect_repeated_phrases(text)

        status.write("🤖 Running Gemini — generating your summary, suggestions, "
                     "interview questions, cover letter, and more in a single AI request...")
        try:
            combined = generate_full_analysis(
                resume_text=text,
                jd_text=jd_text if jd_provided else "",
                found_skills=found_skills,
                ats_score=ats_score,
                checklist=checklist,
                job_role=job_role,
                candidate_name=name_preview,
                readability_stats=readability_stats,
                passive_voice_examples=passive_voice_examples,
                repeated_phrases=repeated_phrases,
            )
            suggestions = combined["ats_suggestions"]
            general_tips = combined["resume_recommendations"]
            technical_questions = combined["interview_questions"]["technical"]
            hr_questions = combined["interview_questions"]["hr"]
            project_questions = combined["interview_questions"]["project"]
            summary = combined["summary"]
            cover_letter = combined["cover_letter"]
            tailored_resume_result = combined["tailored_resume"] if jd_provided else ""
            missing = combined["missing_skills"]
            grammar_quality = combined["grammar_quality"]
            readability_suggestions = combined["readability_suggestions"]
            project_quality = combined["project_quality"]
            ats_keyword_optimizations = combined["ats_keyword_optimizer"]
            combined_error = None
        except Exception as exc:
            # Log the real, technical error for developers; the UI only
            # ever shows a short, friendly fallback notice (see
            # render_fallback_notice below) — never a raw exception or
            # JSON-parsing message.
            logger.warning("Gemini combined analysis failed, using rule-based fallback: %s", exc, exc_info=True)
            combined_error = "AI features are temporarily using rule-based results instead of Gemini."

            # ---- Fallback: original rule-based logic for every section ----

            if jd_provided:
                jd_skills = detect_skills(jd_text, skills_db)
                missing = [s for s in jd_skills if s not in found_skills]
            else:
                missing = [s for s in recommended if s not in found_skills]

            suggestions = []
            if not github_found:
                suggestions.append("Add your GitHub profile link")
            if not linkedin_found:
                suggestions.append("Add your LinkedIn profile link")
            if not projects_found:
                suggestions.append("Add a Projects section showcasing your work")
            if not certifications_found:
                suggestions.append("Add relevant Certifications")
            if not experience_found:
                suggestions.append("Add Experience or Internship details")
            if missing:
                suggestions.append(f"Add missing skills: {', '.join(missing[:5])}")

            general_tips = [
                "Use quantified achievements (e.g. 'Improved accuracy by 15%')",
                "Use strong action verbs (Built, Led, Designed, Optimized)",
                "Keep your resume within one page if possible",
                "Use consistent, clean formatting and avoid dense text blocks",
            ]

            technical_questions = []
            skill_question_bank = {
                "python": "What is the difference between a list and a tuple in Python?",
                "java": "What is the difference between an interface and an abstract class in Java?",
                "sql": "What is the difference between INNER JOIN and LEFT JOIN?",
                "machine learning": "What is overfitting and how do you prevent it?",
                "deep learning": "What is the vanishing gradient problem?",
                "nlp": "What is the difference between stemming and lemmatization?",
                "docker": "What is the difference between a Docker image and a container?",
                "aws": "What is the difference between EC2 and Lambda?",
                "git": "What is the difference between 'git merge' and 'git rebase'?",
                "react": "What is the virtual DOM and why is it used?",
                "mongodb": "What is the difference between SQL and NoSQL databases?",
            }
            for skill in found_skills:
                if skill in skill_question_bank:
                    technical_questions.append(skill_question_bank[skill])
            if not technical_questions:
                technical_questions.append("Walk me through a technical project you're proud of.")

            hr_questions = [
                "Tell me about yourself.",
                "Why do you want to work with us?",
                "What are your strengths and weaknesses?",
                "Where do you see yourself in 5 years?",
                "Describe a challenge you faced and how you handled it.",
            ]

            if projects_found:
                project_questions = [
                    "Walk me through one of your projects end to end.",
                    "What was the most challenging part of building your project, and how did you solve it?",
                    "What would you improve if you revisited this project today?",
                ]
            else:
                project_questions = [
                    "Describe a hands-on task or assignment you've completed recently.",
                ]

            summary_parts = [f"Candidate skilled in {top_skills}."]
            if experience_found:
                summary_parts.append("Has relevant work or internship experience.")
            if projects_found:
                summary_parts.append("Has hands-on project experience.")
            if certifications_found:
                summary_parts.append("Holds relevant certifications.")
            summary_parts.append(f"Overall resume strength score: {ats_score}%.")
            summary = " ".join(summary_parts)

            cover_letter = f"""Dear Hiring Manager,

    I am writing to express my interest in the {job_role} position. Based on my background in {top_skills}, I am confident in my ability to contribute effectively to your team.

    {"During my experience, I have worked on projects and practical assignments that strengthened my technical foundation." if projects_found or experience_found else "I am eager to apply my technical foundation to real-world challenges in this role."}

    {"I have also earned certifications that reflect my commitment to continuous learning." if certifications_found else ""}

    I would welcome the opportunity to discuss how my skills align with your team's needs.

    Sincerely,
    {name_preview if name_preview != "Not Found" else "Candidate"}
    """

            tailored_resume_result = ""

            grammar_quality = (
                "Automated grammar review is temporarily unavailable. As a general check, "
                "read your resume aloud to catch awkward phrasing, and keep verb tense "
                "consistent (past tense for past roles, present tense only for your current role)."
            )
            readability_suggestions = [
                f"Your average sentence length is {readability_stats.get('avg_sentence_length', 'n/a')} words — aim for 10-20 words per bullet for easy scanning.",
                "Favor active voice ('Led the team') over passive voice ('The team was led by me').",
                "Vary your sentence openings and phrasing to avoid repetition.",
            ]
            project_quality = {
                "better_titles": [],
                "better_descriptions": [],
                "missing_technologies": [],
                "missing_impact_metrics": [],
                "github_presentation_tips": [
                    "Add a clear README to each project with setup instructions and screenshots.",
                    "Pin your best projects to your GitHub profile so they're immediately visible.",
                ],
            }
            ats_keyword_optimizations = (
                [f"Consider naturally incorporating: {', '.join(missing[:5])}"] if missing else []
            )

        # These error variables are kept (all pointing at the same combined
        # error) so the existing per-section UI warnings below still work as-is.
        suggestions_error = combined_error
        general_tips_error = combined_error
        interview_questions_error = combined_error
        summary_error = combined_error
        cover_letter_error = combined_error
        missing_error = combined_error
        readability_error = combined_error
        project_quality_error = combined_error
        keyword_optimizer_error = combined_error
        status.write("✔️ AI Analysis Complete" if not combined_error else "⚠️ AI Analysis fell back to rule-based results")
        status.write("✔️ Generating Resume Summary")
        status.write("✔️ Preparing Interview Questions")
        status.write("✔️ Creating Cover Letter")
        status.write("✔️ Preparing Dashboard")
        status.write("✔️ Finalizing Report")


    status.update(label="Resume analysis pipeline finished", state="complete", expanded=False)

    st.markdown(
        '<div class="cc-pill cc-pill-success" style="font-size: 1rem; padding: 0.6rem 1.1rem; margin: 0.5rem 0 1.5rem 0;">✔ Analysis Complete — your resume has been successfully analyzed.</div>',
        unsafe_allow_html=True,
    )
    # ---------------- CAREER COPILOT TABBED WORKSPACE ----------------

    tabs = st.tabs([
        "🏠 Overview",
        "📊 ATS Dashboard",
        "🔍 Resume Insights",
        "🎯 AI Tools",
        "📥 Downloads & Export",
        "📧 Email Report",
    ])

    with tabs[0]:
        mcard1, mcard2, mcard3, mcard4 = st.columns(4)
        with mcard1:
            ats_tier = "Excellent ATS Readiness" if ats_score >= 75 else "Good — room to improve" if ats_score >= 50 else "Needs attention"
            render_metric_card("🎯", f"{ats_score}%", "ATS Score", "var(--cc-primary)", sublabel=ats_tier)
        with mcard2:
            jd_ready = jd_match_score is not None
            match_display = f"{jd_match_score}%" if jd_ready else "No JD yet"
            match_tier = None
            if jd_ready:
                match_tier = "Strong keyword alignment" if jd_match_score >= 75 else "Some gaps to close" if jd_match_score >= 50 else "Low alignment — add keywords"
            render_metric_card("🔗", match_display, "Resume Match", "var(--cc-success)", compact_value=not jd_ready, sublabel=match_tier)
        with mcard3:
            render_metric_card("🧠", str(len(found_skills)), "Skills Found", "var(--cc-success)", sublabel="Detected across your resume")
        with mcard4:
            render_metric_card("⚠️", str(len(missing)), "Missing Skills", "var(--cc-warning)", sublabel="Worth adding if relevant")

        st.markdown("<div style='height: 1.25rem'></div>", unsafe_allow_html=True)

        # ---------------- COLUMNS ----------------

        col1, col2 = st.columns(2)

        # ---------------- LEFT SIDE ----------------

        with col1:
            with st.container(border=True):
                card_marker()
                section_header("👤", "Candidate Snapshot", "Parsed resume content, contact details, and section health")

                with st.expander("📄 View Extracted Resume Text", expanded=False):
                    st.write(text)

                name = text.split("\n")[0] if text else "Not Found"
                email = "Found" if email_found else "Not Found"
                phone = "Found" if phone_found else "Not Found"
                name_display = name if name != "Not Found" else "Not detected"

                snapshot_rows = (
                    ("👤", "Name", name_display, name != "Not Found"),
                    ("📧", "Email", "Detected on resume" if email_found else "No email address detected — add one near the top", email_found),
                    ("📱", "Phone", "Detected on resume" if phone_found else "No phone number detected — add one near the top", phone_found),
                )
                for icon, label, display_value, present in snapshot_rows:
                    if present:
                        st.success(f"{icon} **{label}:** {display_value}")
                    else:
                        st.warning(f"{icon} **{label}:** {display_value}")

                st.markdown("### ✅ ATS Resume Checklist")
                render_checklist_grid(checklist, columns=2)

                st.markdown("### 📋 Resume Section Checker")
                st.caption("Checks for the sections a well-rounded resume typically includes.")

                for item, present in section_checklist.items():
                    if present:
                        st.success(f"✓ {item}")
                    else:
                        suggestion = SECTION_CHECKER_SUGGESTIONS.get(item, "")
                        st.warning(f"⚠ {item} — Missing. {suggestion}")

                st.markdown("### 💡 Improvement Suggestions")
                if suggestions_error:
                    st.caption(f"⚠️ Gemini unavailable, showing rule-based suggestions. ({suggestions_error})")

                if suggestions:
                    for s in suggestions:
                        st.warning(s)
                else:
                    st.success("Your resume covers all the key sections!")

                st.markdown("#### Resume Improvement Recommendations")
                if general_tips_error:
                    st.caption(f"⚠️ Gemini unavailable, showing default tips. ({general_tips_error})")
                for tip in general_tips:
                    st.write(f"• {tip}")

        # ---------------- RIGHT SIDE ----------------

        with col2:
            with st.container(border=True):
                card_marker()
                section_header("🧭", "Resume Health & AI Insights", "JD match, skills coverage, AI summary, and interview prep")

                st.markdown("### 🔗 Resume – JD Match")

                if jd_match_score is not None:
                    quality_label, quality_type = get_match_quality(jd_match_score)

                    st.progress(min(jd_match_score / 100, 1.0))
                    st.metric(label="JD Match Score", value=f"{jd_match_score}%")

                    if quality_type == "success":
                        st.success(f"Match Quality: {quality_label}")
                    elif quality_type == "warning":
                        st.warning(f"Match Quality: {quality_label}")
                    else:
                        st.error(f"Match Quality: {quality_label}")
                else:
                    render_empty_state("🔗", "No job description yet", "Paste or upload one on the home screen to see your match score.")

                st.caption("💡 See the full ATS Score gauge, breakdown, and skills charts in the **📊 ATS Dashboard** tab.")

                st.markdown("### ⭐ Resume Strength Meter")
                st.markdown(f"## {stars_from_score(ats_score)}")

                st.markdown("### 🧠 Skills Detected")

                if found_skills:
                    for skill in found_skills:
                        st.success(skill.title())
                else:
                    st.error("No skills detected")

                st.markdown(f"### ⚠️ Missing Skills (based on {missing_source})")
                if missing_error:
                    st.caption(f"⚠️ Gemini unavailable, showing keyword-based results. ({missing_error})")

                if missing:
                    for skill in missing:
                        st.warning(skill.title())
                else:
                    st.success("Excellent! No missing skills.")

                st.markdown("### 🔑 JD Keyword Match")
                if not jd_provided:
                    render_empty_state("🔑", "No job description yet", "Paste or upload one on the home screen to see keyword matching.")
                else:
                    jd_keywords = extract_jd_keywords(text, jd_text)
                    if not jd_keywords:
                        st.caption("Could not extract distinct keywords from this job description.")
                    else:
                        present_kw = [kw for kw in jd_keywords if kw.lower() in text_lower]
                        missing_kw = [kw for kw in jd_keywords if kw.lower() not in text_lower]

                        kw_present_col, kw_missing_col = st.columns(2)
                        with kw_present_col:
                            st.markdown("**✓ Present**")
                            if present_kw:
                                for kw in present_kw:
                                    st.success(f"✓ {kw}")
                            else:
                                st.caption("None of the extracted keywords were found.")
                        with kw_missing_col:
                            st.markdown("**✗ Missing**")
                            if missing_kw:
                                for kw in missing_kw:
                                    st.error(f"✗ {kw}")
                            else:
                                st.caption("All extracted keywords are present!")

                st.markdown("### 📝 AI Resume Summary")
                if summary_error:
                    st.caption(f"⚠️ Gemini unavailable, showing rule-based summary. ({summary_error})")
                st.success(summary)

                st.markdown("### ❓ AI Interview Questions")
                if interview_questions_error:
                    st.caption(f"⚠️ Gemini unavailable, showing default question bank. ({interview_questions_error})")

                question_groups = (
                    ("🧠", "Technical", technical_questions),
                    ("🤝", "HR", hr_questions),
                    ("🛠️", "Project-Based", project_questions),
                )
                for icon, group_title, questions in question_groups:
                    with st.expander(f"{icon} {group_title} ({len(questions)})", expanded=(group_title == "Technical")):
                        if not questions:
                            st.caption("No questions generated for this category.")
                        for q in questions:
                            st.code(q, language=None)

    with tabs[1]:
        # ---------------- INTERACTIVE ATS DASHBOARD ----------------

        section_header("📊", "Interactive ATS Dashboard", "A visual breakdown of your resume's ATS profile")

        if not PLOTLY_AVAILABLE:
            st.error("⚠️ Interactive charts require the 'plotly' package. Run: pip install plotly")
        else:
            experience_score = 100 if experience_found else (50 if projects_found else 0)
            education_score = 100 if education_found else 0
            ats_compliance_score = round((sum(checklist.values()) / len(checklist)) * 100, 1)

            gauge_col, breakdown_col = st.columns(2, gap="medium")
            with gauge_col:
                with st.container(border=True):
                    card_marker()
                    st.plotly_chart(build_gauge_chart(ats_score), use_container_width=True, key="gauge_chart_ats_score", config=_CHART_CONFIG)
            with breakdown_col:
                with st.container(border=True):
                    card_marker()
                    st.plotly_chart(
                        build_section_scores_bar_chart(skills_component, jd_component, structure_component, contact_component),
                        use_container_width=True, key="bar_chart_section_scores", config=_CHART_CONFIG,
                    )

            pie_col, radar_col = st.columns(2, gap="medium")
            with pie_col:
                with st.container(border=True):
                    card_marker()
                    st.plotly_chart(
                        build_skills_pie_chart(len(found_skills), len(missing)),
                        use_container_width=True, key="pie_chart_skills_coverage", config=_CHART_CONFIG,
                    )
            with radar_col:
                with st.container(border=True):
                    card_marker()
                    st.plotly_chart(
                        build_radar_chart(
                            skills_score=skills_component,
                            experience_score=experience_score,
                            education_score=education_score,
                            ats_compliance_score=ats_compliance_score,
                            jd_match_score_val=jd_component,
                        ),
                        use_container_width=True, key="radar_chart_overall_profile", config=_CHART_CONFIG,
                    )

    with tabs[2]:
        # ---------------- RESUME KEYWORD DENSITY ANALYSIS ----------------

        with st.container(border=True):
            card_marker()
            section_header("🔍", "Resume Keyword Density Analysis", "Local analysis (no API call) of how your resume's word usage lines up with the target role")

            density = analyze_keyword_density(text, jd_text if jd_provided else "", found_skills, skills_db)

            density_col1, density_col2 = st.columns(2, gap="medium")

            with density_col1:
                st.markdown("**📈 Frequently Used Keywords**")
                if density["frequent_keywords"]:
                    for word, count in density["frequent_keywords"]:
                        st.write(f"• {word} — {count}x")
                else:
                    st.caption("No significant keywords detected.")

                st.markdown("**⚠️ Overused Words**")
                if density["overused_words"]:
                    for word, count in density["overused_words"]:
                        st.warning(f"{word} — used {count} times, consider varying your wording")
                else:
                    st.success("No overused words detected.")

            with density_col2:
                st.markdown(f"**🎯 Keyword Relevance Score:** {density['relevance_score']}%")
                st.progress(min(density["relevance_score"] / 100, 1.0))
                st.caption(
                    "Based on JD keyword coverage." if jd_provided
                    else "Based on general technical skill coverage (no JD provided)."
                )

                st.markdown("**❌ Missing Important Keywords**")
                if density["missing_keywords"]:
                    for kw in density["missing_keywords"]:
                        st.error(f"✗ {kw}")
                else:
                    st.success("No important missing keywords detected.")

        st.markdown('<div style="height: 1rem"></div>', unsafe_allow_html=True)

        # ---------------- ATS KEYWORD OPTIMIZER ----------------

        with st.container(border=True):
            card_marker()
            section_header("🧠", "ATS Keyword Optimizer", "Where to naturally work in missing JD keywords — not keyword stuffing")

            if keyword_optimizer_error:
                st.caption(f"⚠️ Gemini unavailable, showing a basic fallback. ({keyword_optimizer_error})")

            if not jd_provided:
                render_empty_state("🧠", "No job description yet", "Paste or upload one on the home screen to get keyword optimization suggestions.")
            elif ats_keyword_optimizations:
                for suggestion in ats_keyword_optimizations:
                    st.info(f"💡 {suggestion}")
            else:
                st.success("No additional keyword suggestions — your resume already covers the JD well.")

        st.markdown('<div style="height: 1rem"></div>', unsafe_allow_html=True)

        # ---------------- RESUME FORMATTING CHECKER ----------------

        with st.container(border=True):
            card_marker()
            section_header("🎨", "Resume Formatting Checker", "Heuristic detection of formatting patterns that can trip up ATS parsers")

            if formatting_check_error:
                st.caption(f"⚠️ Formatting analysis was incomplete. ({formatting_check_error})")

            if not formatting_issues:
                st.info("Could not analyze formatting for this file.")
            else:
                any_issue_found = False
                for issue_name, detected in formatting_issues.items():
                    if detected:
                        any_issue_found = True
                        st.warning(f"⚠ {issue_name} — {FORMATTING_SUGGESTIONS.get(issue_name, '')}")
                    else:
                        st.success(f"✓ {issue_name} — not detected")
                if not any_issue_found:
                    st.success("No ATS-unfriendly formatting patterns detected. 🎉")

        st.markdown('<div style="height: 1rem"></div>', unsafe_allow_html=True)

        # ---------------- RESUME READABILITY ANALYSIS ----------------

        with st.container(border=True):
            card_marker()
            section_header("📖", "Resume Readability Analysis", "Sentence length, passive voice, and repeated phrases computed locally; grammar review from Gemini")

            read_col1, read_col2 = st.columns(2, gap="medium")

            with read_col1:
                st.metric("Average Sentence Length", f"{readability_stats['avg_sentence_length']} words")
                st.caption(readability_stats["sentence_length_label"])

                st.metric("Readability Score (Flesch)", f"{readability_stats['readability_score']}/100")
                score = readability_stats["readability_score"]
                if score >= 70:
                    st.success("Easy to read")
                elif score >= 50:
                    st.warning("Moderately readable")
                else:
                    st.error("Difficult to read — consider simplifying")

            with read_col2:
                st.markdown("**🗣️ Passive Voice**")
                if passive_voice_examples:
                    st.warning(f"{len(passive_voice_examples)} passive-voice sentence(s) detected:")
                    for ex in passive_voice_examples:
                        st.caption(f"“{ex}”")
                else:
                    st.success("No obvious passive voice detected.")

                st.markdown("**🔁 Repeated Phrases**")
                if repeated_phrases:
                    for phrase, count in repeated_phrases:
                        st.warning(f"\"{phrase}\" — repeated {count}x")
                else:
                    st.success("No significantly repeated phrases detected.")

            st.markdown("**✏️ Grammar Quality**")
            if readability_error:
                st.caption(f"⚠️ Gemini unavailable, showing general guidance. ({readability_error})")
            st.info(grammar_quality)

            st.markdown("**💡 AI Readability Suggestions**")
            for tip in readability_suggestions:
                st.write(f"• {tip}")

        st.markdown('<div style="height: 1rem"></div>', unsafe_allow_html=True)

        # ---------------- PROJECT QUALITY ANALYZER ----------------

        with st.container(border=True):
            card_marker()
            section_header("🛠️", "Project Quality Analyzer", "Gemini reviews your Projects section and suggests improvements")

            if project_quality_error:
                st.caption(f"⚠️ Gemini unavailable, showing general tips only. ({project_quality_error})")

            pq_labels = [
                ("better_titles", "📌 Better Project Titles"),
                ("better_descriptions", "📝 Better Descriptions"),
                ("missing_technologies", "🧩 Missing Technologies"),
                ("missing_impact_metrics", "📊 Missing Impact Metrics"),
                ("github_presentation_tips", "🐙 Better GitHub Presentation"),
            ]

            has_any_project_feedback = any(project_quality.get(key) for key, _ in pq_labels)

            if not has_any_project_feedback:
                render_empty_state("🛠️", "No Projects section detected", "Add a Projects section to your resume to get AI feedback on titles, descriptions, and impact metrics.")
            else:
                pq_col1, pq_col2 = st.columns(2, gap="medium")
                for i, (key, label) in enumerate(pq_labels):
                    target_col = pq_col1 if i % 2 == 0 else pq_col2
                    with target_col:
                        st.markdown(f"**{label}**")
                        items = project_quality.get(key, [])
                        if items:
                            for item in items:
                                st.write(f"• {item}")
                        else:
                            st.caption("No suggestions here.")

    with tabs[3]:
        # ---------------- AI TAILORED RESUME GENERATOR ----------------

        with st.container(border=True):
            card_marker()
            section_header(
                "🎯", "AI Tailored Resume Generator",
                "Rewrites your resume for the job description — sharper keywords and "
                "bullet points, same facts, unchanged"
            )

            if not jd_provided:
                render_empty_state(
                    "🎯", "No job description yet",
                    "Paste or upload one on the home screen (then re-run the analysis) to enable the tailored resume generator.",
                )
            else:
                if combined_error:
                    st.error(
                        f"⚠️ Could not generate a tailored resume right now. "
                        f"({combined_error})"
                    )
                elif st.button("✨ Generate Tailored Resume", type="primary"):
                    st.session_state["tailored_resume"] = tailored_resume_result

                if st.session_state.get("tailored_resume"):
                    st.markdown("#### 📄 Tailored Resume Preview")
                    st.text_area(
                        "Tailored Resume",
                        st.session_state["tailored_resume"],
                        height=400,
                        label_visibility="collapsed",
                    )

                    tailored_b64 = base64.b64encode(
                        st.session_state["tailored_resume"].encode()
                    ).decode()
                    tailored_download_link = (
                        f'<a href="data:file/txt;base64,{tailored_b64}" '
                        f'download="Tailored_Resume.txt">'
                        "📥 Click here to Download Tailored Resume"
                        "</a>"
                    )
                    st.markdown(tailored_download_link, unsafe_allow_html=True)

        st.markdown('<div style="height: 1rem"></div>', unsafe_allow_html=True)

        # ---------------- REWRITE INDIVIDUAL BULLET POINTS ----------------

        with st.container(border=True):
            card_marker()
            section_header(
                "✨", "Rewrite Individual Bullet Points",
                "Click 'Rewrite' next to any bullet point for a stronger, achievement-oriented version"
            )

            bullet_candidates = extract_bullet_candidates(text)

            if not bullet_candidates:
                render_empty_state("✨", "No clear bullet points detected", "This resume doesn't have lines that look like distinct bullet points to rewrite.")
            else:
                for idx, bullet in enumerate(bullet_candidates):
                    removed_line = (
                        f'<div class="cc-diff-block cc-diff-top">'
                        f'<div class="cc-diff-line cc-diff-line-removed">'
                        f'<span class="cc-diff-marker">−</span>'
                        f'<span class="cc-diff-text">{_html_escape(bullet)}</span></div></div>'
                    )
                    st.markdown(removed_line, unsafe_allow_html=True)

                    _, b_col2 = st.columns([5, 1])
                    with b_col2:
                        has_rewrite = bool(st.session_state.get(f"rewritten_{idx}"))
                        clicked = st.button(
                            "✨ Rewrite" if not has_rewrite else "🔁 Rewrite Again",
                            key=f"rewrite_btn_{idx}", use_container_width=True,
                        )

                    if clicked:
                        with st.spinner("Rewriting..."):
                            try:
                                rewritten = generate_bullet_rewrite(bullet, found_skills)
                                st.session_state[f"rewritten_{idx}"] = rewritten
                                st.session_state[f"rewritten_error_{idx}"] = None
                            except Exception as exc:
                                logger.warning("Bullet rewrite failed: %s", exc, exc_info=True)
                                st.session_state[f"rewritten_{idx}"] = None
                                st.session_state[f"rewritten_error_{idx}"] = True

                    if st.session_state.get(f"rewritten_error_{idx}"):
                        st.error("⚠️ Couldn't rewrite this bullet right now — please try again.")

                    rewritten_text = st.session_state.get(f"rewritten_{idx}")
                    if rewritten_text:
                        added_line = (
                            f'<div class="cc-diff-block cc-diff-bottom">'
                            f'<div class="cc-diff-line cc-diff-line-added">'
                            f'<span class="cc-diff-marker">+</span>'
                            f'<span class="cc-diff-text">{_html_escape(rewritten_text)}</span></div></div>'
                        )
                    else:
                        added_line = (
                            '<div class="cc-diff-block cc-diff-bottom">'
                            '<div class="cc-diff-empty">✨ Click "Rewrite" to generate a stronger version</div></div>'
                        )
                    st.markdown(added_line, unsafe_allow_html=True)

                    if idx < len(bullet_candidates) - 1:
                        st.markdown('<div style="height: 0.6rem"></div>', unsafe_allow_html=True)

    # ---------------- SHARED: TAILORED RESUME PDF (if generated) ----------------
    # Built once here so the standalone download button, the email
    # feature, and the Complete Career Package all reuse the same
    # bytes instead of regenerating the PDF multiple times.

    tailored_resume_pdf_bytes = None
    tailored_resume_pdf_error = None
    if st.session_state.get("tailored_resume"):
        try:
            tailored_resume_pdf_bytes = generate_simple_pdf(
                title="Tailored Resume",
                subtitle=f"Tailored for: {job_role}",
                body_text=st.session_state["tailored_resume"],
            )
        except Exception as exc:
            logger.warning("Tailored resume PDF generation failed: %s", exc, exc_info=True)
            tailored_resume_pdf_error = "the PDF couldn't be generated"


    with tabs[4]:
        # ---------------- AI COVER LETTER (already generated in the combined call above) ----------------

        with st.container(border=True):
            card_marker()
            section_header("✉️", "AI-Generated Cover Letter", f"Tailored for the {job_role} role")

            if cover_letter_error:
                st.caption(f"⚠️ Gemini unavailable, showing template-based letter. ({cover_letter_error})")
            st.text_area("Cover Letter Preview", cover_letter, height=250, label_visibility="collapsed")

            cover_b64 = base64.b64encode(cover_letter.encode()).decode()
            cover_download_link = (
                f'<a href="data:file/txt;base64,{cover_b64}" '
                f'download="Cover_Letter.txt">'
                "✉️ Click here to Download Cover Letter"
                "</a>"
            )
            st.markdown(cover_download_link, unsafe_allow_html=True)

            cover_letter_pdf_bytes = None
            try:
                cover_letter_pdf_bytes = generate_simple_pdf(
                    title="Cover Letter",
                    subtitle=f"For: {job_role}",
                    body_text=cover_letter,
                )
                st.download_button(
                    label="✉️ Download Cover Letter PDF",
                    data=cover_letter_pdf_bytes,
                    file_name="Cover_Letter.pdf",
                    mime="application/pdf",
                    key="download_cover_letter_pdf",
                )
            except Exception as exc:
                logger.warning("Cover letter PDF generation failed: %s", exc, exc_info=True)
                st.caption("⚠️ Couldn't generate the Cover Letter PDF right now.")

            if tailored_resume_pdf_bytes:
                st.download_button(
                    label="🎯 Download Tailored Resume PDF",
                    data=tailored_resume_pdf_bytes,
                    file_name="Tailored_Resume.pdf",
                    mime="application/pdf",
                    key="download_tailored_resume_pdf",
                )
            elif tailored_resume_pdf_error:
                st.caption(f"⚠️ Couldn't generate the Tailored Resume PDF — {tailored_resume_pdf_error}.")

        st.markdown('<div style="height: 1rem"></div>', unsafe_allow_html=True)

        report_text = f"""
    AI Resume Analysis Report

    Candidate Details
    -----------------
    Name: {name}
    Email: {email}
    Phone: {phone}

    Job Role:
    {job_role}

    ATS Checklist:
    {chr(10).join([f"{'[x]' if v else '[ ]'} {k}" for k, v in checklist.items()])}

    Skills Detected:
    {", ".join(found_skills) if found_skills else "No skills detected"}

    Weighted ATS Score:
    {ats_score}%
      - Skills (40%): {round(skills_component, 1)}%
      - JD Match (30%): {round(jd_component, 1)}%
      - Structure (15%): {round(structure_component, 1)}%
      - Contact Info (15%): {round(contact_component, 1)}%

    Resume - JD Match Score:
    {f"{jd_match_score}%" if jd_match_score is not None else "No job description provided"}

    Missing Skills (based on {missing_source}):
    {", ".join(missing) if missing else "None"}

    Improvement Suggestions:
    {chr(10).join(suggestions) if suggestions else "None - resume covers key sections"}

    Interview Questions - Technical:
    {chr(10).join(technical_questions)}

    Interview Questions - HR:
    {chr(10).join(hr_questions)}

    Interview Questions - Project-Based:
    {chr(10).join(project_questions)}

    Summary:
    {summary}

    Resume Strength: {stars_from_score(ats_score)}

    Generated by AI Career Copilot
    """

        b64 = base64.b64encode(
            report_text.encode()
        ).decode()

        download_link = (
            f'<a href="data:file/txt;base64,{b64}" '
            f'download="AI_Resume_Report.txt">'
            "📄 Click here to Download Report"
            "</a>"
        )

        # ---------------- PROFESSIONAL PDF REPORT ----------------

        with st.container(border=True):
            card_marker()
            section_header("📥", "Full Resume Analysis Report", "Every insight in one document — plain text or polished PDF")

            st.markdown(download_link, unsafe_allow_html=True)

            strength_stars = stars_from_score(ats_score)
            strength_count = strength_stars.count("★")
            strength_label = {
                1: "Poor", 2: "Fair", 3: "Average", 4: "Good", 5: "Excellent",
            }.get(strength_count, "Average")

            try:
                pdf_bytes = generate_pdf_report(
                    name=name,
                    email=email,
                    phone=phone,
                    job_role=job_role,
                    summary=summary,
                    jd_provided=jd_provided,
                    jd_match_score=jd_match_score,
                    ats_score=ats_score,
                    skills_component=skills_component,
                    jd_component=jd_component,
                    structure_component=structure_component,
                    contact_component=contact_component,
                    strength_count=strength_count,
                    strength_label=strength_label,
                    found_skills=found_skills,
                    missing_skills=missing,
                    missing_source=missing_source,
                    suggestions=suggestions,
                    general_tips=general_tips,
                    technical_questions=technical_questions,
                    hr_questions=hr_questions,
                    project_questions=project_questions,
                    cover_letter=cover_letter,
                    tailored_resume=st.session_state.get("tailored_resume", ""),
                )
                pdf_generation_error = None
                st.download_button(
                    label="📑 Download Professional PDF Report",
                    data=pdf_bytes,
                    file_name="AI_Resume_Report.pdf",
                    mime="application/pdf",
                )
            except Exception as exc:
                pdf_bytes = None
                pdf_generation_error = "unknown error"
                logger.warning("PDF report generation failed: %s", exc, exc_info=True)
                st.error("⚠️ Couldn't generate the PDF report right now — please try again in a moment.")

        st.markdown('<div style="height: 1rem"></div>', unsafe_allow_html=True)

        # ---------------- DOWNLOAD COMPLETE CAREER PACKAGE ----------------

        with st.container(border=True):
            card_marker()
            section_header("📦", "Export Everything", "Bundle every generated document into one ZIP file")

            try:
                career_package_bytes = build_career_package_zip(
                    report_pdf_bytes=pdf_bytes,
                    report_txt=report_text,
                    cover_letter_txt=cover_letter,
                    cover_letter_pdf_bytes=cover_letter_pdf_bytes,
                    tailored_resume_txt=st.session_state.get("tailored_resume"),
                    tailored_resume_pdf_bytes=tailored_resume_pdf_bytes,
                )
                st.download_button(
                    label="📦 Download Complete Career Package (.zip)",
                    data=career_package_bytes,
                    file_name="Career_Package.zip",
                    mime="application/zip",
                    key="download_career_package",
                )
            except Exception as exc:
                logger.warning("Career package ZIP build failed: %s", exc, exc_info=True)
                st.error("⚠️ Couldn't build the Complete Career Package right now — please try again in a moment.")


    with tabs[5]:
        # ---------------- EMAIL REPORT ----------------

        with st.container(border=True):
            card_marker()
            section_header("📧", "Email Your Reports", "Send your generated documents straight to any inbox")

            if not email_credentials_available():
                st.markdown(
                    '<div class="cc-setup-card">'
                    '<div class="cc-setup-card-icon">⚙️</div>'
                    '<div>'
                    '<p class="cc-setup-card-title">Email delivery isn\'t set up yet</p>'
                    '<p class="cc-setup-card-desc">Add these two variables to your project\'s '
                    '<code>.env</code> file to enable sending reports by email:</p>'
                    '<div class="cc-setup-card-code">EMAIL_ADDRESS=you@example.com</div><br>'
                    '<div class="cc-setup-card-code">EMAIL_APP_PASSWORD=your_app_password</div>'
                    '</div></div>',
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    f'<div class="cc-pill cc-pill-success">{email_status_message()}</div>',
                    unsafe_allow_html=True,
                )

            st.markdown('<div style="height: 0.6rem"></div>', unsafe_allow_html=True)

            recipient_email = st.text_input(
                "Recipient Email Address",
                placeholder="recipient@example.com",
                key="recipient_email_input",
            )

            tailored_available = bool(st.session_state.get("tailored_resume"))

            email_col1, email_col2, email_col3 = st.columns(3)

            with email_col1:
                if st.button("📑 Email PDF Report", disabled=pdf_bytes is None, use_container_width=True):
                    if not email_credentials_available():
                        st.error(email_status_message())
                    else:
                        with st.spinner("Sending PDF report..."):
                            success, msg = send_email_report(
                                recipient_email,
                                subject="Your AI Resume Analysis Report",
                                body_text=(
                                    f"Hi {name if name != 'Not Found' else ''},\n\n"
                                    "Please find your AI Resume Analysis Report attached.\n\n"
                                    "- AI Career Copilot"
                                ),
                                attachment_bytes=pdf_bytes,
                                attachment_filename="AI_Resume_Report.pdf",
                                attachment_mime_subtype="pdf",
                            )
                        st.success(msg) if success else st.error(msg)

            with email_col2:
                if st.button("✉️ Email Cover Letter", use_container_width=True):
                    if not email_credentials_available():
                        st.error(email_status_message())
                    else:
                        with st.spinner("Sending cover letter..."):
                            success, msg = send_email_report(
                                recipient_email,
                                subject="Your AI-Generated Cover Letter",
                                body_text=cover_letter,
                                attachment_bytes=cover_letter.encode("utf-8"),
                                attachment_filename="Cover_Letter.txt",
                                attachment_mime_subtype="plain",
                            )
                        st.success(msg) if success else st.error(msg)

            with email_col3:
                if st.button("🎯 Email Tailored Resume PDF", disabled=not tailored_available, use_container_width=True):
                    if not tailored_available:
                        st.info("Generate the tailored resume first (see the AI Tools tab).")
                    elif not email_credentials_available():
                        st.error(email_status_message())
                    elif tailored_resume_pdf_bytes is None:
                        st.error(f"⚠️ Tailored Resume PDF isn't available. ({tailored_resume_pdf_error or 'unknown error'})")
                    else:
                        with st.spinner("Sending tailored resume PDF..."):
                            try:
                                success, msg = send_email_report(
                                    recipient_email,
                                    subject="Your AI-Tailored Resume",
                                    body_text=(
                                        f"Hi {name if name != 'Not Found' else ''},\n\n"
                                        "Please find your tailored resume attached.\n\n"
                                        "- AI Career Copilot"
                                    ),
                                    attachment_bytes=tailored_resume_pdf_bytes,
                                    attachment_filename="Tailored_Resume.pdf",
                                    attachment_mime_subtype="pdf",
                                )
                            except Exception as exc:
                                logger.warning("Emailing tailored resume PDF failed: %s", exc, exc_info=True)
                                success, msg = False, "⚠️ Couldn't send the tailored resume email right now — please try again."
                        st.success(msg) if success else st.error(msg)

# ---------------- FOOTER ----------------
# Fill these in (or leave as None to hide the icon) — left as simple
# constants rather than hardcoded fake links, since the real profile
# URLs weren't provided.
FOOTER_GITHUB_URL = None
FOOTER_LINKEDIN_URL = None
APP_VERSION = "v1.0"

st.markdown('<div style="height: 0.5rem"></div>', unsafe_allow_html=True)

_footer_links = []
if FOOTER_GITHUB_URL:
    _footer_links.append(f'<a href="{FOOTER_GITHUB_URL}" target="_blank">GitHub</a>')
if FOOTER_LINKEDIN_URL:
    _footer_links.append(f'<a href="{FOOTER_LINKEDIN_URL}" target="_blank">LinkedIn</a>')
_footer_dot_separator = "<span class=\"cc-footer-dot\">•</span>"
_footer_links_joined = _footer_dot_separator.join(_footer_links)
_footer_links_html = (
    f'<div class="cc-footer-links">{_footer_links_joined}</div>'
    if _footer_links else ""
)

st.markdown(
    f"""
<div class="cc-footer">
    <div class="cc-footer-divider"></div>
    <div class="cc-footer-brand">
        <span class="cc-footer-icon">🧭</span>
        <span class="cc-footer-name">AI Career Copilot</span>
        <span class="cc-footer-version">{APP_VERSION}</span>
    </div>
    <div class="cc-footer-sub">Built with Streamlit &amp; Python by Shruti Verma</div>
    {_footer_links_html}
</div>
""",
    unsafe_allow_html=True,
)