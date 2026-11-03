"""Vertical bar chart of how many uploaded rows fall into each outcome."""

import matplotlib.pyplot as plt
import streamlit as st

from gradpath.frontend.styling.brand_colours import BRAND_ACCENT_COLOUR


def render_class_distribution(counts_by_label: dict[str, int]) -> None:
    """Draw one vertical bar per outcome, labelled with its row count."""
    labels = list(counts_by_label)
    values = [counts_by_label[label] for label in labels]

    figure, axes = plt.subplots(figsize=(5, 2.6))
    bars = axes.bar(labels, values, color=BRAND_ACCENT_COLOUR)
    axes.bar_label(bars, padding=3)
    axes.set_ylabel("Rows uploaded")
    axes.spines[["top", "right"]].set_visible(False)
    figure.tight_layout()
    st.pyplot(figure, clear_figure=True)
