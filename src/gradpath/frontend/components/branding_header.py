"""Centered "GradPath AI" branding header, shown once at the top of the page."""

import streamlit as st

from gradpath.frontend.styling import text_constants
from gradpath.frontend.styling.brand_colours import (
    BRAND_ACCENT_COLOUR,
    BRAND_TEXT_COLOUR,
    TAGLINE_COLOUR,
)


def render_branding_header() -> None:
    """Draw the centered, two-tone brand wordmark and its one-line tagline.

    "GradPath" is rendered in the brand accent colour and "AI" in the
    regular brand text colour, matching the two-tone wordmark in the
    approved UI mock (UI.html).

    A `position: fixed` sticky version was tried (so the header stays
    visible for the entire page while scrolling) but was dropped: it
    exposed a Streamlit rendering glitch where the uploaded-file chart
    image bleeds over the header for a narrow scroll range, which looked
    worse than a plain, non-sticky header. See UI_DESIGN_FIXES.md item 8.
    """
    brand_word, ai_word = text_constants.APP_HEADING.rsplit(" ", 1)
    st.markdown(
        f"<h1 style='text-align:center; margin-bottom:0;'>"
        f"<span style='color:{BRAND_ACCENT_COLOUR};'>{brand_word}</span> "
        f"<span style='color:{BRAND_TEXT_COLOUR};'>{ai_word}</span></h1>",
        unsafe_allow_html=True,
    )
    st.markdown(
        f"<p style='text-align:center; color:{TAGLINE_COLOUR}; "
        f"font-family:Georgia, \"Times New Roman\", serif; "
        f"font-size:1.15rem; font-weight:500; letter-spacing:0.3px; "
        f"margin-top:0.35rem;'>"
        f"{text_constants.BRAND_TAGLINE}</p>",
        unsafe_allow_html=True,
    )

