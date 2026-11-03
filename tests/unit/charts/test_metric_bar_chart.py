"""Unit tests for the shared metric-bar-panel chart function."""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from gradpath.charts.metric_bar_chart import draw_metric_bar_panel


def test_draw_metric_bar_panel_draws_one_bar_per_entry():
    figure, axis = plt.subplots()
    ranked_entries = [
        ("Logistic Regression", 0.9, "#1f4e9c"),
        ("Decision Tree", 0.8, "#b35a00"),
    ]

    draw_metric_bar_panel(axis, ranked_entries, "Accuracy")

    assert len(axis.patches) == len(ranked_entries)
    assert axis.get_title() == "Accuracy"
    plt.close(figure)


def test_draw_metric_bar_panel_colours_bars_per_entry():
    figure, axis = plt.subplots()
    ranked_entries = [("Logistic Regression", 0.9, "#1f4e9c")]

    draw_metric_bar_panel(axis, ranked_entries, "Accuracy")

    bar = axis.patches[0]
    assert bar.get_facecolor()[:3] == (0.12156862745098039, 0.3058823529411765, 0.611764705882353)
    plt.close(figure)


def test_draw_metric_bar_panel_handles_no_entries_without_raising():
    figure, axis = plt.subplots()

    draw_metric_bar_panel(axis, [], "AUC")

    assert axis.get_title() == "AUC"
    plt.close(figure)
