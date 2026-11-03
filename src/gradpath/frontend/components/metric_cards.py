"""Six metric cards, laid out three per row, each in a bordered card.

Each card's value is rendered in the selected classifier's own fixed colour
(see `styling/model_colours.py`), so a reader always associates the same
colour with the same model across the metric cards, the confusion matrix
and the curves.
"""

import streamlit as st

from gradpath.frontend.styling.brand_colours import BRAND_TEXT_COLOUR

_CARDS_PER_ROW = 3

_CARD_TEMPLATE = """
<div style="
    background-color: #ffffff;
    border: 1px solid #e2e8f0;
    border-radius: 12px;
    padding: 1.1rem 1.25rem;
    margin-bottom: 1rem;
    box-shadow: 0 1px 3px rgba(16, 24, 40, 0.08), 0 1px 2px rgba(16, 24, 40, 0.04);
    text-align: center;
">
    <div style="
        text-transform: uppercase;
        letter-spacing: 0.06em;
        font-size: 0.72rem;
        font-weight: 600;
        color: {text_colour};
        margin-bottom: 0.4rem;
    ">{label}</div>
    <div style="
        font-size: 2.1rem;
        font-weight: 700;
        color: {colour};
        line-height: 1.2;
    ">{value}</div>
</div>
"""


def render_metric_cards(metrics: list[tuple[str, str]], model_colour: str) -> None:
    """Draw the six (label, display value) pairs as cards tinted in `model_colour`."""
    for row_start in range(0, len(metrics), _CARDS_PER_ROW):
        row_metrics = metrics[row_start : row_start + _CARDS_PER_ROW]
        columns = st.columns(len(row_metrics))
        for column, (label, value) in zip(columns, row_metrics):
            with column:
                st.markdown(
                    _CARD_TEMPLATE.format(
                        label=label.upper(),
                        colour=model_colour,
                        value=value,
                        text_colour=BRAND_TEXT_COLOUR,
                    ),
                    unsafe_allow_html=True,
                )
