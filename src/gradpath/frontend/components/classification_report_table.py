"""Per-outcome precision, recall, F1 and support, as a formatted table."""

from gradpath.frontend.components.styled_table import render_styled_table

_COLUMN_NAMES = ["Outcome", "Precision", "Recall", "F1", "Support"]


def render_classification_report(rows: list[dict]) -> None:
    """Draw the classification report rows as a bordered, dark-text table.

    Reuses the same HTML table renderer as the comparison and observations
    tables (`styled_table.py`) instead of `st.dataframe`, whose canvas-drawn
    header text cannot be reliably recoloured through CSS.
    """
    render_styled_table(_COLUMN_NAMES, rows)
