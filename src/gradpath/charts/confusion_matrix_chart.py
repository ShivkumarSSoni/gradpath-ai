"""Confusion-matrix heatmap drawing, shared by the app, the notebook and the
README image generator, so all three show the exact same visual: a
white-to-model-colour shaded heatmap, the diagonal (correct predictions)
outlined in black, and a title coloured in the model's own fixed colour.
"""

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap


def compute_row_shares(count_matrix) -> np.ndarray:
    """Return each cell as a fraction (0 to 1) of its own row's total.

    A row with zero records (an outcome entirely missing from the data)
    would divide by zero; its share is 0 everywhere instead of raising.
    """
    counts = np.asarray(count_matrix, dtype=float)
    row_totals = counts.sum(axis=1, keepdims=True)
    return np.divide(counts, row_totals, out=np.zeros_like(counts), where=row_totals != 0)


def draw_confusion_matrix_panel(axis, count_matrix, labels, model_name, model_colour) -> None:
    """Draw one confusion-matrix heatmap onto `axis`.

    Each cell shows the raw count and, below it, its share of that row's
    actual-outcome total (so every row adds up to 100%); shares, not raw
    counts, decide the shade, so a small outcome is not pale just because it
    has fewer students.

    Args:
        axis: The matplotlib Axes to draw into.
        count_matrix: Raw counts, rows are the actual outcome.
        labels: Outcome names, in row/column order.
        model_name: Shown in the panel's coloured title.
        model_colour: This classifier's fixed display colour.
    """
    counts = np.asarray(count_matrix)
    row_shares = compute_row_shares(count_matrix)
    cell_texts = np.array(
        [
            [f"{counts[row, col]}\n{row_shares[row, col]:.0%}" for col in range(len(labels))]
            for row in range(len(labels))
        ]
    )
    white_to_model_colour = LinearSegmentedColormap.from_list(
        "white_to_model_colour", ["#ffffff", model_colour]
    )

    sns.heatmap(
        row_shares,
        annot=cell_texts,
        fmt="",
        cmap=white_to_model_colour,
        vmin=0,
        vmax=1,
        xticklabels=labels,
        yticklabels=labels,
        cbar=False,
        ax=axis,
        linewidths=1,
        linecolor="white",
        annot_kws={"fontsize": 12, "fontweight": "bold"},
    )
    for diagonal_position in range(len(labels)):
        axis.add_patch(
            plt.Rectangle(
                (diagonal_position, diagonal_position),
                1,
                1,
                fill=False,
                edgecolor="black",
                linewidth=2.5,
            )
        )
    axis.set_title(f"Confusion matrix: {model_name}", fontsize=11, fontweight="bold", color=model_colour)
    axis.set_xlabel("Predicted outcome")
    axis.set_ylabel("Actual outcome")
