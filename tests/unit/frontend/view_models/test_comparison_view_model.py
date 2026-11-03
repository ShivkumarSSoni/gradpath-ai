"""Unit tests for turning the comparison rows into display-ready data."""

from gradpath.backend.domain.classifier_kind import ClassifierKind
from gradpath.backend.domain.comparison_row import ComparisonRow
from gradpath.backend.domain.evaluation_scores import EvaluationScores
from gradpath.frontend.view_models.comparison_view_model import (
    _build_chart_data,
    _build_observation_rows,
    _build_table_rows,
    _count_metric_wins,
    _sort_rows_best_first,
)


def make_row(kind: ClassifierKind, **score_overrides) -> ComparisonRow:
    base_scores = {"accuracy": 0.5, "auc": 0.5, "precision": 0.5, "recall": 0.5, "f1": 0.5, "mcc": 0.5}
    base_scores.update(score_overrides)
    return ComparisonRow(classifier_kind=kind, scores=EvaluationScores(**base_scores))


def test_table_rows_show_not_available_for_a_model_with_no_scores():
    rows = [ComparisonRow(classifier_kind=ClassifierKind.DECISION_TREE, scores=None)]

    table_rows = _build_table_rows(rows)

    assert table_rows[0]["Accuracy"] == "Not available"
    assert table_rows[0]["classifier_kind"] == ClassifierKind.DECISION_TREE


def test_chart_data_sorts_each_metric_best_first():
    rows = [
        make_row(ClassifierKind.LOGISTIC_REGRESSION, accuracy=0.9),
        make_row(ClassifierKind.DECISION_TREE, accuracy=0.7),
        make_row(ClassifierKind.SUPPORT_VECTOR_MACHINE, accuracy=0.95),
    ]

    chart_data = _build_chart_data(rows)

    assert [kind for kind, _ in chart_data["Accuracy"]] == [
        ClassifierKind.SUPPORT_VECTOR_MACHINE,
        ClassifierKind.LOGISTIC_REGRESSION,
        ClassifierKind.DECISION_TREE,
    ]


def test_chart_data_leaves_out_an_undefined_metric():
    rows = [make_row(ClassifierKind.LOGISTIC_REGRESSION, auc=float("nan"))]

    chart_data = _build_chart_data(rows)

    assert chart_data["AUC"] == []


def test_the_model_winning_the_most_metrics_is_counted_correctly():
    rows = [
        make_row(
            ClassifierKind.LOGISTIC_REGRESSION,
            accuracy=0.9,
            auc=0.9,
            precision=0.9,
            recall=0.4,
            f1=0.4,
            mcc=0.4,
        ),
        make_row(
            ClassifierKind.SUPPORT_VECTOR_MACHINE,
            accuracy=0.8,
            auc=0.8,
            precision=0.8,
            recall=0.9,
            f1=0.9,
            mcc=0.9,
        ),
    ]

    win_counts = _count_metric_wins(rows)

    assert win_counts[ClassifierKind.LOGISTIC_REGRESSION] == 3
    assert win_counts[ClassifierKind.SUPPORT_VECTOR_MACHINE] == 3


def test_observation_rows_include_a_winner_row_naming_the_top_scorer():
    rows = [
        make_row(ClassifierKind.LOGISTIC_REGRESSION, accuracy=0.95, auc=0.95, precision=0.95,
                 recall=0.95, f1=0.95, mcc=0.95),
        make_row(ClassifierKind.DECISION_TREE, accuracy=0.5, auc=0.5, precision=0.5,
                  recall=0.5, f1=0.5, mcc=0.5),
    ]

    observation_rows = _build_observation_rows(rows)

    winner_row = observation_rows[-1]
    assert winner_row["Model"] == "Winner"
    assert winner_row["classifier_kind"] is None
    assert "Logistic Regression" in winner_row["Observation"]
    assert "6 of 6" in winner_row["Observation"]


def test_sort_rows_best_first_orders_by_win_count_then_accuracy():
    rows = [
        make_row(ClassifierKind.DECISION_TREE, accuracy=0.6),
        make_row(ClassifierKind.LOGISTIC_REGRESSION, accuracy=0.9),
        make_row(ClassifierKind.SUPPORT_VECTOR_MACHINE, accuracy=0.7),
    ]

    sorted_rows = _sort_rows_best_first(rows)

    assert [row.classifier_kind for row in sorted_rows] == [
        ClassifierKind.LOGISTIC_REGRESSION,
        ClassifierKind.SUPPORT_VECTOR_MACHINE,
        ClassifierKind.DECISION_TREE,
    ]


def test_sort_rows_best_first_pushes_unavailable_models_last():
    rows = [
        ComparisonRow(classifier_kind=ClassifierKind.DECISION_TREE, scores=None),
        make_row(ClassifierKind.LOGISTIC_REGRESSION, accuracy=0.5),
    ]

    sorted_rows = _sort_rows_best_first(rows)

    assert sorted_rows[-1].classifier_kind == ClassifierKind.DECISION_TREE
