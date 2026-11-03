"""ROC, Precision-Recall and Calibration curves for the selected model."""

import matplotlib.pyplot as plt
import streamlit as st

from gradpath.backend.domain.classifier_kind import ClassifierKind
from gradpath.charts.curve_chart import draw_curve_panel
from gradpath.frontend.styling.model_colours import MODEL_DISPLAY_COLOURS


def render_curves(curves_by_name: dict, classifier_kind: ClassifierKind) -> None:
    """Draw one full-width panel per curve kind, stacked so each stays readable."""
    curve_names = list(curves_by_name)
    if not curve_names:
        return
    colour = MODEL_DISPLAY_COLOURS[classifier_kind]
    for curve_name in curve_names:
        _render_single_curve(curve_name, curves_by_name[curve_name], colour)


def _render_single_curve(curve_name: str, curves_by_outcome: dict, colour: str) -> None:
    """Draw one full-width curve panel, with one line per outcome that has a curve."""
    figure, axes = plt.subplots(figsize=(9.0, 4.5))
    draw_curve_panel(
        axes,
        curve_name,
        curves_by_outcome,
        colour,
        title_fontsize=15,
        legend_fontsize=11,
        line_width=2.2,
    )
    figure.tight_layout()
    st.pyplot(figure, clear_figure=True)
