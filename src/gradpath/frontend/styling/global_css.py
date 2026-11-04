"""One-off global CSS injected once at app startup.

Covers styling that Streamlit's own widgets cannot express through their
Python arguments alone (uppercase micro-labels, a crisper white/bordered
dropdown control), matching the approved UI mock (UI.html). Kept in one
place so no other module needs to inject raw `<style>` blocks of its own.
"""

import streamlit as st

from gradpath.frontend.styling.brand_colours import BRAND_TEXT_COLOUR

_GLOBAL_CSS = f"""
<style>
/* Uppercase, letter-spaced micro-labels above the model dropdown and the
   file uploader, matching the mock's "SELECT MODEL" style. Dark, not
   muted, so they stay readable on a white background. */
[data-testid="stWidgetLabel"] p {{
    text-transform: uppercase;
    letter-spacing: 0.06em;
    font-size: 0.75rem;
    font-weight: 600;
    color: {BRAND_TEXT_COLOUR};
}}

/* A crisp white, bordered dropdown control instead of the default
   theme-tinted background, so it reads as a clean input field. Targets the
   widget's accessible "group" role rather than an emotion-generated class
   name, since those class names are not stable across Streamlit builds. */
[data-testid="stSelectbox"] [role="group"] {{
    background-color: #ffffff !important;
    border: 1px solid #8aa0bd !important;
}}

/* The file uploader's dropzone ships with no visible border at all, so it
   reads as a plain grey panel rather than a distinct drop target. Give it
   the same border colour and a slightly thicker line than the dropdown
   above, so it is unmistakably a bordered upload control. */
[data-testid="stFileUploaderDropzone"] {{
    border: 2px solid #8aa0bd !important;
}}

/* Bordered section containers (Uploaded file, Metrics, Confusion matrix,
   Curves, Comparison) use Streamlit's own default border colour, which is
   too close to the page background to read clearly. A slightly darker,
   brand-toned grey keeps each section visually distinct. */
[data-testid="stVerticalBlockBorderWrapper"] {{
    border-color: #b9c4d4 !important;
}}

/* st.divider() renders as a plain <hr>; darken it to the same level of
   contrast as the container borders above. */
hr {{
    border-top: 1px solid #b9c4d4 !important;
}}
</style>
"""


def inject_global_css() -> None:
    """Render the shared `<style>` block once, near the top of the page."""
    st.markdown(_GLOBAL_CSS, unsafe_allow_html=True)
