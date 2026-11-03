"""ROC, Precision-Recall and Calibration curves for the selected model."""

import matplotlib.pyplot as plt
import streamlit as st

from gradpath.backend.domain.classifier_kind import ClassifierKind
from gradpath.frontend.styling.model_colours import MODEL_DISPLAY_COLOURS

_LINE_STYLES = ["-", "--", ":"]


def render_curves(curves_by_name: dict, classifier_kind: ClassifierKind) -> None:
    """Draw one panel per curve kind, one line per outcome that has a curve."""
    curve_names = list(curves_by_name)
    if not curve_names:
        return
    colour = MODEL_DISPLAY_COLOURS[classifier_kind]
    columns = st.columns(len(curve_names))
    for column, curve_name in zip(columns, curve_names):
        with column:
            _render_single_curve(curve_name, curves_by_name[curve_name], colour)


def _render_single_curve(curve_name: str, curves_by_outcome: dict, colour: str) -> None:
    """Draw one curve panel, with one line per outcome that has a defined curve."""
    figure, axes = plt.subplots(figsize=(3.4, 3.0))
    if not curves_by_outcome:
        axes.text(0.5, 0.5, "Not available", ha="center", va="center", fontsize=9)
        axes.set_xticks([])
        axes.set_yticks([])
    else:
        for (outcome, points), style in zip(curves_by_outcome.items(), _LINE_STYLES):
            axes.plot(
                points.x_values,
                points.y_values,
                style,
                color=colour,
                label=f"{outcome} ({points.summary_score:.2f})",
            )
        axes.legend(fontsize=6)
    axes.set_title(curve_name, fontsize=9)
    figure.tight_layout()
    st.pyplot(figure, clear_figure=True)
