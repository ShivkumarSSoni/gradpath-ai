"""The five-model, six-metric comparison table of fixed reference scores."""

from gradpath.frontend.components import styled_table
from gradpath.frontend.view_models.comparison_view_model import METRIC_NAMES


def render_comparison_table(rows: list[dict]) -> None:
    """Draw the comparison rows, best model first, with colour dots."""
    styled_table.render_styled_table(["Model", *METRIC_NAMES], rows)
