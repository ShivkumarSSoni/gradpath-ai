"""Unit tests for turning an EvaluationResponse into display-ready data."""

from gradpath.backend.domain.classifier_kind import ClassifierKind
from gradpath.backend.domain.evaluation_report import EvaluationReport
from gradpath.backend.domain.evaluation_scores import EvaluationScores
from gradpath.backend.schemas.evaluation_response import EvaluationResponse
from gradpath.frontend.view_models.evaluation_view_model import (
    build_classification_report_rows,
    build_confusion_matrix_view,
    build_metric_cards,
    count_rows_per_outcome,
    format_metric_value,
)

import pandas as pd


def make_response(scores: EvaluationScores) -> EvaluationResponse:
    report = EvaluationReport(
        classifier_kind=ClassifierKind.LOGISTIC_REGRESSION,
        scores=scores,
        confusion_matrix=[[8, 1, 1], [0, 5, 0], [2, 0, 10]],
        classification_report={
            "Dropout": {"precision": 0.8, "recall": 0.8, "f1_score": 0.8, "support": 10},
            "Enrolled": {"precision": 1.0, "recall": 1.0, "f1_score": 1.0, "support": 5},
            "Graduate": {"precision": 0.9, "recall": 0.83, "f1_score": 0.86, "support": 12},
        },
        curves_by_name={},
    )
    return EvaluationResponse(classifier_kind=ClassifierKind.LOGISTIC_REGRESSION, report=report)


def test_format_metric_value_renders_a_normal_number():
    assert format_metric_value(0.8456) == "0.846"


def test_format_metric_value_renders_nan_as_not_available():
    assert format_metric_value(float("nan")) == "Not available"


def test_build_metric_cards_covers_all_six_metrics_with_nan_handled():
    scores = EvaluationScores(
        accuracy=0.9, auc=float("nan"), precision=0.8, recall=0.7, f1=0.75, mcc=0.6
    )
    cards = build_metric_cards(make_response(scores))

    assert dict(cards) == {
        "Accuracy": "0.900",
        "AUC": "Not available",
        "Precision": "0.800",
        "Recall": "0.700",
        "F1": "0.750",
        "MCC": "0.600",
    }


def test_confusion_matrix_view_exposes_labels_and_raw_counts():
    scores = EvaluationScores(accuracy=1.0, auc=1.0, precision=1.0, recall=1.0, f1=1.0, mcc=1.0)
    view = build_confusion_matrix_view(make_response(scores))

    assert view.labels == ["Dropout", "Enrolled", "Graduate"]
    assert view.counts == [[8, 1, 1], [0, 5, 0], [2, 0, 10]]


def test_classification_report_rows_follow_the_fixed_outcome_order():
    scores = EvaluationScores(accuracy=1.0, auc=1.0, precision=1.0, recall=1.0, f1=1.0, mcc=1.0)
    rows = build_classification_report_rows(make_response(scores))

    assert [row["Outcome"] for row in rows] == ["Dropout", "Enrolled", "Graduate"]
    assert rows[0]["Support"] == 10


def test_count_rows_per_outcome_reports_zero_for_a_missing_class():
    records = pd.DataFrame({"Target": ["Dropout", "Dropout", "Graduate"]})

    counts = count_rows_per_outcome(records, "Target")

    assert counts == {"Dropout": 2, "Enrolled": 0, "Graduate": 1}
