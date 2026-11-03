"""A small vertical bar panel per metric, matching the notebook's chart style."""

import matplotlib.pyplot as plt
import streamlit as st

from gradpath.frontend.styling.model_colours import MODEL_DISPLAY_COLOURS

_PANELS_PER_ROW = 3


def render_comparison_bar_charts(chart_data_by_metric: dict[str, list[tuple]]) -> None:
    """Draw one vertical bar panel per metric, bars sorted best to worst.

    Args:
        chart_data_by_metric: Metric name to a list of (classifier_kind,
            value) pairs, already sorted best first, with unavailable
            scores left out.
    """
    metric_names = list(chart_data_by_metric)
    for row_start in range(0, len(metric_names), _PANELS_PER_ROW):
        row_metric_names = metric_names[row_start : row_start + _PANELS_PER_ROW]
        columns = st.columns(len(row_metric_names))
        for column, metric_name in zip(columns, row_metric_names):
            with column:
                _render_single_panel(metric_name, chart_data_by_metric[metric_name])


def _render_single_panel(metric_name: str, entries: list[tuple]) -> None:
    """Draw one metric's five bars, one per model, in its model colour."""
    short_labels = [kind.display_name.split(" (")[0] for kind, _ in entries]
    values = [value for _, value in entries]
    colours = [MODEL_DISPLAY_COLOURS[kind] for kind, _ in entries]

    figure, axes = plt.subplots(figsize=(2.6, 2.6))
    bar_positions = range(len(values))
    bars = axes.bar(bar_positions, values, color=colours)
    axes.bar_label(bars, fmt="%.3f", padding=2, fontsize=7)
    axes.set_xticks(list(bar_positions))
    axes.set_xticklabels(short_labels, rotation=90, fontsize=6)
    axes.set_title(metric_name, fontsize=9)
    # Headroom above the tallest bar so its value label is never clipped.
    axes.set_ylim(0, max(values) * 1.25 if values else 1)
    axes.spines[["top", "right"]].set_visible(False)
    figure.tight_layout()
    st.pyplot(figure, clear_figure=True)
