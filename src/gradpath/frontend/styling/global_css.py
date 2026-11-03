"""One-off global CSS injected once at app startup.

Covers styling that Streamlit's own widgets cannot express through their
Python arguments alone (uppercase micro-labels, a crisper white/bordered
dropdown control), matching the approved UI mock (UI.html). Kept in one
place so no other module needs to inject raw `<style>` blocks of its own.
"""

import streamlit as st

from gradpath.frontend.styling.brand_colours import MUTED_TEXT_COLOUR

_GLOBAL_CSS = f"""
<style>
/* Uppercase, letter-spaced, muted micro-labels above the model dropdown
   and the file uploader, matching the mock's "SELECT MODEL" style. */
[data-testid="stWidgetLabel"] p {{
    text-transform: uppercase;
    letter-spacing: 0.06em;
    font-size: 0.75rem;
    font-weight: 600;
    color: {MUTED_TEXT_COLOUR};
}}

/* A crisp white, bordered dropdown control instead of the default
   theme-tinted background, so it reads as a clean input field. */
[data-testid="stSelectbox"] div[data-baseweb="select"] > div {{
    background-color: #ffffff;
    border: 1px solid #d7dee8;
}}
</style>
"""


def inject_global_css() -> None:
    """Render the shared `<style>` block once, near the top of the page."""
    st.markdown(_GLOBAL_CSS, unsafe_allow_html=True)
