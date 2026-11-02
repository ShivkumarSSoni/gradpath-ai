"""One point set of a two-dimensional curve, plus its one-number summary."""

from dataclasses import dataclass


@dataclass(frozen=True)
class CurvePoints:
    """Everything needed to draw one curve for one outcome.

    Attributes:
        x_values: Horizontal axis values, in plotting order.
        y_values: Vertical axis values, matched index for index with `x_values`.
        summary_score: The curve's single-number summary (for example the
            area under an ROC curve), shown alongside the plot.
    """

    x_values: list[float]
    y_values: list[float]
    summary_score: float
