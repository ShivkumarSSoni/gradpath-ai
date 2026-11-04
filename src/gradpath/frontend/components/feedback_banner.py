"""Branded feedback banners for success, error, warning and info messages.

Streamlit's own st.success/st.error/st.warning/st.info render the same
green/red/orange/blue boxes in every Streamlit app, which makes different
projects look alike at a glance. These banners use the project's own left
accent-border card style (matching the look already used for the model-name
banner and the metric cards) instead, so a status message reads as a
GradPath AI design choice, not an out-of-the-box widget.
"""

import html

import streamlit as st

from gradpath.frontend.styling.brand_colours import (
    BRAND_TEXT_COLOUR,
    FEEDBACK_ERROR_COLOUR,
    FEEDBACK_INFO_COLOUR,
    FEEDBACK_SUCCESS_COLOUR,
    FEEDBACK_WARNING_COLOUR,
)

_BANNER_TEMPLATE = """
<div style="
    border-left: 4px solid {colour};
    background-color: {colour}14;
    border-radius: 8px;
    padding: 0.7rem 1rem;
    margin: 0.6rem 0;
    display: flex;
    align-items: center;
    gap: 0.6rem;
">
    <span style="color:{colour}; font-size:1.05rem; font-weight:700; line-height:1;">{icon}</span>
    <span style="color:{text_colour}; font-size:0.95rem;">{message}</span>
</div>
"""


def _render_banner(message: str, colour: str, icon: str) -> None:
    """Draw one banner, escaping the message so an upload's own text cannot
    inject markup into the page."""
    st.markdown(
        _BANNER_TEMPLATE.format(
            colour=colour,
            icon=icon,
            text_colour=BRAND_TEXT_COLOUR,
            message=html.escape(message),
        ),
        unsafe_allow_html=True,
    )


def render_success(message: str) -> None:
    _render_banner(message, FEEDBACK_SUCCESS_COLOUR, "\u2713")


def render_error(message: str) -> None:
    _render_banner(message, FEEDBACK_ERROR_COLOUR, "\u2715")


def render_warning(message: str) -> None:
    _render_banner(message, FEEDBACK_WARNING_COLOUR, "\u26a0")


def render_info(message: str) -> None:
    _render_banner(message, FEEDBACK_INFO_COLOUR, "\u2139")
