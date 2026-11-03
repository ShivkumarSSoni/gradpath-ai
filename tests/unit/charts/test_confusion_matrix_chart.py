"""Unit tests for the shared confusion-matrix chart function."""

import numpy as np

from gradpath.charts.confusion_matrix_chart import compute_row_shares, draw_confusion_matrix_panel


def test_compute_row_shares_normalises_each_row_to_one():
    shares = compute_row_shares([[8, 1, 1], [0, 5, 0], [2, 0, 10]])

    assert np.isclose(shares[0], [0.8, 0.1, 0.1]).all()
    assert np.isclose(shares.sum(axis=1), [1.0, 1.0, 1.0]).all()


def test_compute_row_shares_handles_a_row_with_zero_records():
    shares = compute_row_shares([[0, 0, 0], [1, 2, 1]])

    assert list(shares[0]) == [0.0, 0.0, 0.0]


def test_draw_confusion_matrix_panel_outlines_the_diagonal():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    figure, axis = plt.subplots()
    labels = ["Dropout", "Enrolled", "Graduate"]
    draw_confusion_matrix_panel(axis, [[8, 1, 1], [0, 5, 0], [2, 0, 10]], labels, "Logistic Regression", "#1f4e9c")

    diagonal_outlines = [patch for patch in axis.patches if isinstance(patch, Rectangle) and not patch.get_fill()]
    assert len(diagonal_outlines) == len(labels)
    assert axis.get_title() == "Confusion matrix: Logistic Regression"
    plt.close(figure)
