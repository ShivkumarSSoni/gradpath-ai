"""A combined per-metric bar panel grid, drawn with the same shared chart
function used by the notebook and the README image script.
"""

import math

import matplotlib.pyplot as plt
import streamlit as st

from gradpath.charts.metric_bar_chart import draw_metric_bar_panel
from gradpath.frontend.styling.model_colours import MODEL_DISPLAY_COLOURS

_PANELS_PER_ROW = 3


def render_comparison_bar_charts(chart_data_by_metric: dict[str, list[tuple]]) -> None:
    """Draw one bar panel per metric, bars sorted best to worst, in one grid.

    Args:
        chart_data_by_metric: Metric name to a list of (classifier_kind,
            value) pairs, already sorted best first, with unavailable
            scores left out.
    """
    metric_names = list(chart_data_by_metric)
    row_count = math.ceil(len(metric_names) / _PANELS_PER_ROW)
    figure, axes = plt.subplots(row_count, _PANELS_PER_ROW, figsize=(16, 4.5 * row_count))
    flat_axes = axes.ravel() if row_count * _PANELS_PER_ROW > 1 else [axes]

    for axis, metric_name in zip(flat_axes, metric_names):
        ranked_entries = [
            (kind.display_name, value, MODEL_DISPLAY_COLOURS[kind])
            for kind, value in chart_data_by_metric[metric_name]
        ]
        draw_metric_bar_panel(axis, ranked_entries, metric_name)
    for unused_axis in flat_axes[len(metric_names) :]:
        unused_axis.axis("off")

    figure.tight_layout()
    st.pyplot(figure, clear_figure=True)

