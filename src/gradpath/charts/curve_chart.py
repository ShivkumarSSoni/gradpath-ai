"""Shared drawing of one ROC, Precision-Recall or Calibration panel.

Used by the Streamlit app (one model's curves at a time) and by
`tools/generate_curve_images.py` (a grid with one panel per model), so the
two never drift apart.
"""

from gradpath.backend.ml.metrics.curves.curve_points import CurvePoints

_LINE_STYLES = ["-", "--", ":"]


def draw_curve_panel(
    axis,
    curve_name: str,
    curves_by_outcome: dict[str, CurvePoints],
    colour: str,
    panel_title: str | None = None,
    title_fontsize: int = 9,
    legend_fontsize: int = 6,
    line_width: float = 1.5,
) -> None:
    """Draw one curve panel onto `axis`, one line per outcome that has a curve.

    Args:
        axis: A matplotlib axis to draw onto.
        curve_name: The curve kind (for example "ROC"), used in the legend
            label only when `panel_title` is not given.
        curves_by_outcome: Maps outcome name to its `CurvePoints`. An outcome
            missing from this dict (too few records of that class) is simply
            not drawn; an empty dict means "not available" is shown instead.
        colour: The one fixed colour for the model these curves belong to.
        panel_title: Overrides the axis title (for example the model name in
            a grid where `curve_name` is already the figure's title).
        title_fontsize: Font size of the axis title. The default suits a
            small grid panel (five models side by side); the app passes a
            larger value for its single full-width panel.
        legend_fontsize: Font size of the outcome legend, same reasoning as
            `title_fontsize`.
        line_width: Width of each plotted line, same reasoning as above.
    """
    if not curves_by_outcome:
        axis.text(
            0.5, 0.5, "Not available", ha="center", va="center", fontsize=title_fontsize
        )
        axis.set_xticks([])
        axis.set_yticks([])
    else:
        for (outcome, points), style in zip(curves_by_outcome.items(), _LINE_STYLES):
            axis.plot(
                points.x_values,
                points.y_values,
                style,
                color=colour,
                linewidth=line_width,
                label=f"{outcome} ({points.summary_score:.2f})",
            )
        axis.legend(fontsize=legend_fontsize)
    axis.set_title(panel_title if panel_title is not None else curve_name, fontsize=title_fontsize)
    axis.tick_params(labelsize=max(legend_fontsize, 8))
