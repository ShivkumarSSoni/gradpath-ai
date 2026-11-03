"""Unit tests for the shared curve-panel chart function."""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from gradpath.backend.ml.metrics.curves.curve_points import CurvePoints
from gradpath.charts.curve_chart import draw_curve_panel


def test_draw_curve_panel_draws_one_line_per_outcome():
    figure, axis = plt.subplots()
    curves_by_outcome = {
        "Dropout": CurvePoints(x_values=[0.0, 1.0], y_values=[0.0, 1.0], summary_score=0.8),
        "Graduate": CurvePoints(x_values=[0.0, 1.0], y_values=[0.0, 0.9], summary_score=0.75),
    }

    draw_curve_panel(axis, "ROC", curves_by_outcome, "#1f4e9c")

    assert len(axis.lines) == len(curves_by_outcome)
    assert axis.get_title() == "ROC"
    plt.close(figure)


def test_draw_curve_panel_shows_not_available_when_no_outcomes_have_curves():
    figure, axis = plt.subplots()

    draw_curve_panel(axis, "Calibration", {}, "#b35a00")

    assert len(axis.lines) == 0
    assert any(text.get_text() == "Not available" for text in axis.texts)
    plt.close(figure)


def test_draw_curve_panel_honours_panel_title_override():
    figure, axis = plt.subplots()
    curves_by_outcome = {
        "Enrolled": CurvePoints(x_values=[0.0, 1.0], y_values=[0.0, 1.0], summary_score=0.6),
    }

    draw_curve_panel(axis, "Precision-Recall", curves_by_outcome, "#1b7a3d", panel_title="Decision Tree")

    assert axis.get_title() == "Decision Tree"
    plt.close(figure)
