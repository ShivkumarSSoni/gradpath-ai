"""Shared bordered HTML table renderer, with an optional model colour dot.

`st.dataframe` cannot wrap long sentences and auto-truncates text, and it
has no way to draw a colour swatch next to a cell. The observations and
comparison tables need both, so they render this lightweight HTML table
instead, styled to match the rest of the bordered-card design.
"""

import html

import streamlit as st

from gradpath.frontend.styling.model_colours import MODEL_DISPLAY_COLOURS

_HEADER_BACKGROUND_COLOUR = "#eef2f8"
_BORDER_COLOUR = "#d9e0ea"
_WINNER_ROW_BACKGROUND_COLOUR = "#fff7e0"


def render_styled_table(column_names: list[str], rows: list[dict]) -> None:
    """Render `rows` as a bordered table, with a colour dot in the first column.

    Each `row` dict is expected to hold `column_names` as keys, plus an
    optional `"classifier_kind"` key (a `ClassifierKind` or `None`) used to
    look up that model's fixed colour for the swatch.
    """
    st.markdown(_build_table_html(column_names, rows), unsafe_allow_html=True)


def _build_table_html(column_names: list[str], rows: list[dict]) -> str:
    """Build the table markup; kept separate so it is testable without Streamlit."""
    header_cells = "".join(f"<th style='padding:8px; text-align:left;'>{html.escape(name)}</th>" for name in column_names)
    body_rows = "".join(_build_row_html(column_names, row) for row in rows)
    return (
        "<table style='width:100%; border-collapse:collapse;'>"
        f"<thead><tr style='background-color:{_HEADER_BACKGROUND_COLOUR};'>{header_cells}</tr></thead>"
        f"<tbody>{body_rows}</tbody>"
        "</table>"
    )


def _build_row_html(column_names: list[str], row: dict) -> str:
    """Build one `<tr>`, shading the final "Winner" row and adding a colour dot."""
    classifier_kind = row.get("classifier_kind")
    is_winner_row = classifier_kind is None and row.get("Model") == "Winner"
    row_style = f"background-color:{_WINNER_ROW_BACKGROUND_COLOUR};" if is_winner_row else ""
    cells = []
    for index, column_name in enumerate(column_names):
        cell_text = html.escape(str(row.get(column_name, "")))
        if index == 0 and classifier_kind is not None:
            cell_text = _build_colour_dot_html(classifier_kind) + cell_text
        cells.append(
            f"<td style='border:1px solid {_BORDER_COLOUR}; padding:8px; "
            f"text-align:left; white-space:normal;'>{cell_text}</td>"
        )
    return f"<tr style='{row_style}'>{''.join(cells)}</tr>"


def _build_colour_dot_html(classifier_kind) -> str:
    """Build a small coloured circle matching this model's fixed colour."""
    colour = MODEL_DISPLAY_COLOURS[classifier_kind]
    return (
        f"<span style='display:inline-block; width:10px; height:10px; "
        f"border-radius:50%; background-color:{colour}; margin-right:6px;'></span>"
    )
