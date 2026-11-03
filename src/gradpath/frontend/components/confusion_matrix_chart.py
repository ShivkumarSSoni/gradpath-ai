"""Confusion matrix heatmap for the app, drawn with the same shared chart
function used by the notebook and the README image script.
"""

import matplotlib.pyplot as plt
import streamlit as st

from gradpath.backend.domain.classifier_kind import ClassifierKind
from gradpath.charts.confusion_matrix_chart import draw_confusion_matrix_panel
from gradpath.frontend.styling.model_colours import MODEL_DISPLAY_COLOURS
from gradpath.frontend.view_models.evaluation_view_model import ConfusionMatrixView


def render_confusion_matrix(view: ConfusionMatrixView, classifier_kind: ClassifierKind) -> None:
    """Draw the selected model's confusion matrix, shaded in its own fixed colour.

    Uses the exact same drawing function as the notebook and the README
    image script, so the app never shows a different-looking chart for the
    same data.
    """
    figure, axis = plt.subplots(figsize=(5.5, 4.6))
    draw_confusion_matrix_panel(
        axis,
        view.counts,
        view.labels,
        classifier_kind.display_name,
        MODEL_DISPLAY_COLOURS[classifier_kind],
    )
    figure.tight_layout()
    st.pyplot(figure, clear_figure=True)

