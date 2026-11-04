"""Vertical bar chart of how many uploaded rows fall into each outcome.

Reuses the same panel-drawing function and per-outcome colours as the
README/notebook outcome-distribution chart (`gradpath.charts`), so a reader
sees one consistent colour per outcome everywhere in the project, instead of
the app re-implementing its own bar chart with different colours.
"""

import matplotlib.pyplot as plt
import streamlit as st

from gradpath.charts.outcome_distribution_chart import draw_outcome_distribution_panel


def render_class_distribution(counts_by_label: dict[str, int]) -> None:
    """Draw one vertical bar per outcome, labelled with its row count."""
    labels = list(counts_by_label)
    values = [counts_by_label[label] for label in labels]

    # Taller than a plain bar chart would need, since the shared panel
    # function also draws a title, which needs extra headroom above the
    # tallest bar's count/percentage label to avoid overlapping it.
    figure, axis = plt.subplots(figsize=(5, 3.4))
    draw_outcome_distribution_panel(
        axis,
        labels,
        values,
        title="Outcome distribution of the uploaded file",
        ylabel="Rows uploaded",
    )
    figure.tight_layout()
    st.pyplot(figure, clear_figure=True)
