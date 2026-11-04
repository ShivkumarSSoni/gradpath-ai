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


def draw_outcome_distribution_panel(
    axis,
    outcome_names,
    outcome_counts,
    *,
    title: str = "Class balance of the outcome",
    ylabel: str = "Number of students",
) -> None:
    """Draw the count of rows in each outcome class onto `axis`.

    Shared by the notebook/README image (full dataset counts) and the
    Streamlit app's uploaded-file chart (uploaded-row counts), so both
    contexts use identical colours and bar/label styling; only the title
    and y-axis wording differ, since one describes the whole dataset and the
    other describes a single uploaded file.

    Args:
        axis: The matplotlib Axes to draw into.
        outcome_names: Outcome class names, in display order.
        outcome_counts: Row count for each outcome, same order.
        title: Panel title, overridable so each caller's wording matches
            what the counts actually represent.
        ylabel: Y-axis label, overridable for the same reason.
    """
    total_rows = sum(outcome_counts)
    bar_colours = [OUTCOME_DISPLAY_COLOURS[name] for name in outcome_names]
    bars = axis.bar(outcome_names, outcome_counts, color=bar_colours)
    # Guard against dividing by zero when every count is zero (e.g. an
    # uploaded file with no rows for a known outcome), instead of crashing.
    bar_labels = [
        f"{count}\n({count / total_rows:.1%})" if total_rows > 0 else str(count)
        for count in outcome_counts
    ]
    axis.bar_label(bars, labels=bar_labels, padding=3, fontsize=9, fontweight="bold")
    axis.set_xlabel("Student outcome")
    axis.set_ylabel(ylabel)
    # A fixed title pad (not left to tight_layout's own estimate) and a
    # larger top margin keep the tallest bar's two-line label clear of the
    # title at any render DPI; tight_layout's own spacing estimate can run
    # too tight at the lower DPI Streamlit renders at by default.
    axis.set_title(title, fontsize=12, pad=14)
    axis.margins(y=0.25)


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
