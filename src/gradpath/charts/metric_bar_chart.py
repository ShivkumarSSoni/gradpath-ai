"""Per-metric bar panel drawing, shared by the notebook and the app: bars
sorted best first, the exact score above every bar, and every model's bar,
value label and name caption coloured in that model's own fixed colour.
"""


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
