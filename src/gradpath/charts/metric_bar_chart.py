"""Per-metric bar panel drawing, shared by the notebook and the app: bars
sorted best first, the exact score above every bar, and every model's bar,
value label and name caption coloured in that model's own fixed colour.
"""

import matplotlib.pyplot as plt


def draw_metric_bar_panel(axis, ranked_entries, metric_name) -> None:
    """Draw one metric's bars onto `axis`, best score first.

    Args:
        axis: The matplotlib Axes to draw into.
        ranked_entries: A list of (display_name, value, colour) tuples,
            already sorted best first.
        metric_name: The panel title.
    """
    if not ranked_entries:
        axis.set_title(metric_name, fontsize=12)
        axis.axis("off")
        return

    names = [name for name, _, _ in ranked_entries]
    values = [value for _, value, _ in ranked_entries]
    colours = [colour for _, _, colour in ranked_entries]

    bars = axis.bar(range(len(values)), values, color=colours)
    value_labels = axis.bar_label(
        bars, labels=[f"{value:.3f}" for value in values], padding=3, fontsize=9, fontweight="bold"
    )
    axis.set_xticks(range(len(values)))
    caption_labels = axis.set_xticklabels(
        [name.replace(" ", "\n") for name in names], fontsize=8, fontweight="bold"
    )
    for value_label, caption_label, colour in zip(value_labels, caption_labels, colours):
        value_label.set_color(colour)
        caption_label.set_color(colour)
    # Zoomed to this panel's own score range (not 0 to 1), so gaps of a few
    # hundredths stay visible; the bar labels still show the true values.
    axis.set_ylim(min(values) - 0.08, max(values) + 0.04)
    axis.set_title(metric_name, fontsize=12)


def render_metric_bar_grid(metric_names, ranked_entries_by_metric, grid_shape=(3, 2), figure_size=(11, 13)):
    """Build a grid figure with one panel per metric, shared by the
    notebook and the README image generator so the grid is assembled once.

    Args:
        metric_names: The metric names, one per panel, in panel order.
        ranked_entries_by_metric: Maps each metric name to the list of
            (display_name, value, colour) tuples `draw_metric_bar_panel`
            expects, already sorted best first.
        grid_shape: (rows, cols) of the subplot grid.
        figure_size: (width, height) in inches passed to `plt.subplots`.

    Returns:
        The matplotlib Figure, not yet saved or closed.
    """
    figure, axes = plt.subplots(*grid_shape, figsize=figure_size)
    flat_axes = axes.ravel()
    for axis, metric_name in zip(flat_axes, metric_names):
        draw_metric_bar_panel(axis, ranked_entries_by_metric[metric_name], metric_name)
    for unused_axis in flat_axes[len(metric_names):]:
        unused_axis.axis("off")

    figure.suptitle("Held-out scores of the five classifiers (best first in each panel)", fontsize=14)
    figure.tight_layout()
    return figure
