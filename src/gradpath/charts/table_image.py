"""Dataframe-as-image table rendering, shared by the notebook and the
README image generator, so every embedded table uses the same banded,
wrapped-text style.
"""

import textwrap

import matplotlib.pyplot as plt


def draw_table_as_image(table_frame, figure_width=15, wrap_widths=None, title=None):
    """Build a banded table image from a small DataFrame; long text wraps inside its cell.

    Each column is as wide as its longest line of text and each row as tall
    as its tallest cell, the header row is dark with white bold text, and
    every other body row is lightly shaded for readability.

    Returns:
        The created matplotlib Figure. The caller decides what to do with
        it: the notebook calls `plt.show()` to display it inline, the
        README image script calls `figure.savefig(...)` to write a PNG.
    """
    wrap_widths = wrap_widths or {}
    column_names = list(table_frame.columns)
    header_cells = [textwrap.fill(str(name), wrap_widths.get(name, 24)) for name in column_names]
    body_rows = [
        [textwrap.fill(str(value), wrap_widths.get(name, 24)) for name, value in zip(column_names, row)]
        for row in table_frame.itertuples(index=False)
    ]

    # Each column is as wide as its longest line of text, and each row as
    # tall as its tallest cell.
    longest_line_per_column = [
        max(
            len(line)
            for cell_text in [header_cells[position]] + [row[position] for row in body_rows]
            for line in cell_text.split("\n")
        )
        + 2
        for position in range(len(column_names))
    ]
    column_width_shares = [width / sum(longest_line_per_column) for width in longest_line_per_column]
    row_heights_in_inches = [
        0.30 * max(cell_text.count("\n") + 1 for cell_text in row) + 0.16
        for row in [header_cells] + body_rows
    ]
    figure_height = sum(row_heights_in_inches) + (0.6 if title else 0.2)

    figure, axis = plt.subplots(figsize=(figure_width, figure_height))
    axis.axis("off")
    image_table = axis.table(
        cellText=body_rows,
        colLabels=header_cells,
        cellLoc="left",
        colLoc="left",
        loc="upper left",
        colWidths=column_width_shares,
    )
    image_table.auto_set_font_size(False)
    image_table.set_fontsize(10)
    for (row_position, _), table_cell in image_table.get_celld().items():
        table_cell.set_height(row_heights_in_inches[row_position] / figure_height)
        table_cell.set_edgecolor("#c9d3e3")
        if row_position == 0:
            table_cell.set_facecolor("#1f3b6d")
            table_cell.get_text().set_color("white")
            table_cell.get_text().set_fontweight("bold")
        elif row_position % 2 == 0:
            table_cell.set_facecolor("#eef2f9")
    if title:
        axis.set_title(title, fontsize=12, fontweight="bold", loc="left")
    return figure
