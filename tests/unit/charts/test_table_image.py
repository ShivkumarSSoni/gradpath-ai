"""Unit tests for the shared dataframe-as-image table renderer."""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from gradpath.charts.table_image import draw_table_as_image


def test_draw_table_as_image_returns_a_figure_with_header_and_body_rows():
    table_frame = pd.DataFrame({"Model": ["Logistic Regression", "Decision Tree"], "Accuracy": [0.9, 0.8]})

    figure = draw_table_as_image(table_frame, title="Example scores")

    axis = figure.axes[0]
    image_table = axis.tables[0]
    row_positions = {row for row, _ in image_table.get_celld()}
    # One header row (0) plus one row per data record.
    assert row_positions == {0, 1, 2}
    plt.close(figure)


def test_draw_table_as_image_wraps_long_text_within_its_cell():
    table_frame = pd.DataFrame({"Note": ["a very long note that should wrap across more than one line of text"]})

    figure = draw_table_as_image(table_frame, wrap_widths={"Note": 20})

    axis = figure.axes[0]
    image_table = axis.tables[0]
    body_cell = image_table.get_celld()[(1, 0)]
    assert "\n" in body_cell.get_text().get_text()
    plt.close(figure)
