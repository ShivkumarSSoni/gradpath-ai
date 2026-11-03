"""Per-model observation sentences, plus the computed winner, as a table."""

from gradpath.frontend.components import styled_table


def render_observations_table(rows: list[dict]) -> None:
    """Draw the model-to-observation rows, best model first, with colour dots."""
    styled_table.render_styled_table(["Model", "Observation"], rows)
