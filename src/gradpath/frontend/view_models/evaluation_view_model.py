"""Turns the backend's evaluation response into display-ready data.

Nothing here imports Streamlit, so it can be unit tested without a running
app, per the project's testing conventions.
"""

import math
from dataclasses import dataclass

import pandas as pd

from gradpath.backend.domain.classifier_kind import ClassifierKind
from gradpath.backend.domain.feature_catalog import STUDENT_FEATURE_CATALOG
from gradpath.backend.domain.student_outcome import StudentOutcome
from gradpath.backend.schemas.evaluation_response import EvaluationResponse
from gradpath.frontend import backend_gateway
from gradpath.frontend.state.session_state_manager import StoredEvaluationRun
from gradpath.frontend.styling.text_constants import NOT_AVAILABLE_TEXT

_OUTCOME_LABELS = StudentOutcome.all_labels()


def count_rows_per_outcome(records: pd.DataFrame, outcome_column: str) -> dict[str, int]:
    """Return how many uploaded rows have each known outcome, zero if absent."""
    observed_counts = records[outcome_column].value_counts()
    return {label: int(observed_counts.get(label, 0)) for label in _OUTCOME_LABELS}


def list_classifier_options() -> list[ClassifierKind]:
    """Return the five classifiers, for the model dropdown."""
    return backend_gateway.list_classifier_options()


def run_evaluation(classifier_kind: ClassifierKind, uploaded_file) -> StoredEvaluationRun | None:
    """Read, validate and score an uploaded file; return the run to store.

    Returns:
        None when there is no file to evaluate (nothing was uploaded).
        Otherwise a `StoredEvaluationRun`, whose `response` carries either
        the full report or a readable error message.
    """
    if uploaded_file is None:
        return None

    records, read_error = backend_gateway.read_uploaded_file(uploaded_file)
    if read_error is not None:
        return StoredEvaluationRun(
            file_name=uploaded_file.name,
            row_count=0,
            classifier_kind=classifier_kind,
            class_counts={},
            response=EvaluationResponse(classifier_kind=classifier_kind, error_message=read_error),
        )

    outcome_column = STUDENT_FEATURE_CATALOG.outcome_column
    class_counts = (
        count_rows_per_outcome(records, outcome_column)
        if outcome_column in records.columns
        else {}
    )
    response = backend_gateway.evaluate_uploaded_records(classifier_kind, records)
    return StoredEvaluationRun(
        file_name=uploaded_file.name,
        row_count=len(records),
        classifier_kind=classifier_kind,
        class_counts=class_counts,
        response=response,
    )


def format_metric_value(value: float) -> str:
    """Return a metric as a fixed-precision string, or "Not available" for NaN."""
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return NOT_AVAILABLE_TEXT
    return f"{value:.3f}"


def build_metric_cards(response: EvaluationResponse) -> list[tuple[str, str]]:
    """Return (label, display value) pairs for the six metric cards."""
    scores = response.report.scores.as_display_dict()
    return [(label, format_metric_value(value)) for label, value in scores.items()]


@dataclass(frozen=True)
class ConfusionMatrixView:
    """Display-ready confusion matrix: raw counts and row percentages."""

    labels: list[str]
    counts: list[list[int]]
    row_percentages: list[list[float]]


def build_confusion_matrix_view(response: EvaluationResponse) -> ConfusionMatrixView:
    """Normalise each row of the confusion matrix to a percentage of that row.

    A row with zero records (an outcome entirely missing from the upload)
    would divide by zero; its percentages are all shown as 0 instead.
    """
    counts = response.report.confusion_matrix
    row_percentages = []
    for row in counts:
        row_total = sum(row)
        if row_total == 0:
            row_percentages.append([0.0] * len(row))
        else:
            row_percentages.append([100.0 * value / row_total for value in row])
    return ConfusionMatrixView(
        labels=_OUTCOME_LABELS, counts=counts, row_percentages=row_percentages
    )


def build_classification_report_rows(response: EvaluationResponse) -> list[dict]:
    """Return one row per outcome, formatted for a display table."""
    report = response.report.classification_report
    return [
        {
            "Outcome": label,
            "Precision": f"{report[label]['precision']:.3f}",
            "Recall": f"{report[label]['recall']:.3f}",
            "F1": f"{report[label]['f1_score']:.3f}",
            "Support": int(report[label]["support"]),
        }
        for label in _OUTCOME_LABELS
    ]
