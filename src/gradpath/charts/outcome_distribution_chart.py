"""Outcome (class-balance) bar chart, shared by the notebook and the README
image generator, so the dataset's target distribution is drawn identically
in both places instead of being built twice.
"""

import matplotlib.pyplot as plt

# One fixed colour per outcome, independent of the per-model colours used
# elsewhere, since this chart is about the dataset's target, not a model.
# Chosen to not repeat any of the five MODEL_DISPLAY_COLOURS hex values, so a
# reader never confuses an outcome bar with a classifier's colour.
OUTCOME_DISPLAY_COLOURS = {
    "Dropout": "#c9a227",
    "Enrolled": "#6a4c93",
    "Graduate": "#2f9d8f",
}


def draw_outcome_distribution_panel(axis, outcome_names, outcome_counts) -> None:
    """Draw the count of students in each outcome class onto `axis`.

    Args:
        axis: The matplotlib Axes to draw into.
        outcome_names: Outcome class names, in display order.
        outcome_counts: Number of students for each outcome, same order.
    """
    total_students = sum(outcome_counts)
    bar_colours = [OUTCOME_DISPLAY_COLOURS[name] for name in outcome_names]
    bars = axis.bar(outcome_names, outcome_counts, color=bar_colours)
    axis.bar_label(
        bars,
        labels=[f"{count}\n({count / total_students:.1%})" for count in outcome_counts],
        padding=3,
        fontsize=10,
        fontweight="bold",
    )
    axis.set_xlabel("Student outcome")
    axis.set_ylabel("Number of students")
    axis.set_title("Class balance of the outcome", fontsize=12)
    axis.margins(y=0.12)


def render_outcome_distribution_figure(outcome_names, outcome_counts):
    """Build a standalone one-panel figure, used by the README image generator.

    Args:
        outcome_names: Outcome class names, in display order.
        outcome_counts: Number of students for each outcome, same order.

    Returns:
        The matplotlib Figure, not yet saved or closed.
    """
    figure, axis = plt.subplots(figsize=(7, 5))
    draw_outcome_distribution_panel(axis, outcome_names, outcome_counts)
    figure.tight_layout()
    return figure
