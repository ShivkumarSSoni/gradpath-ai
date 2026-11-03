"""Confusion matrix heatmap, row-normalised, drawn with matplotlib."""

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import streamlit as st

from gradpath.backend.domain.classifier_kind import ClassifierKind
from gradpath.frontend.styling.model_colours import MODEL_DISPLAY_COLOURS, build_model_colour_gradient
from gradpath.frontend.view_models.evaluation_view_model import ConfusionMatrixView


def render_confusion_matrix(view: ConfusionMatrixView, classifier_kind: ClassifierKind) -> None:
    """Draw a heatmap of row percentages, shaded in the selected model's own colour.

    Each model has a fixed colour used consistently in the bar charts and
    curves elsewhere on the page; the heatmap now reuses that same colour
    (as a white-to-colour gradient) instead of always being blue.
    """
    percentage_matrix = np.array(view.row_percentages)
    count_matrix = np.array(view.counts)
    outcome_count = len(view.labels)
    annotations = np.array(
        [
            [
                f"{percentage_matrix[row][col]:.0f}%\n({count_matrix[row][col]})"
                for col in range(outcome_count)
            ]
            for row in range(outcome_count)
        ]
    )

    colour_gradient = build_model_colour_gradient(MODEL_DISPLAY_COLOURS[classifier_kind])
    figure, axes = plt.subplots(figsize=(5, 4))
    sns.heatmap(
        percentage_matrix,
        annot=annotations,
        fmt="",
        cmap=colour_gradient,
        vmin=0,
        vmax=100,
        xticklabels=[f"Prediction: {label}" for label in view.labels],
        yticklabels=[f"Actual: {label}" for label in view.labels],
        cbar=False,
        linewidths=0.5,
        linecolor="white",
        ax=axes,
    )
    axes.set_ylabel("")
    axes.set_xlabel("")
    figure.tight_layout()
    st.pyplot(figure, clear_figure=True)
