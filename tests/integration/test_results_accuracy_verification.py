"""Integration test: prove the published metrics are correct for every model.

An evaluator checking this project is expected to open the confusion matrix
shown for a model, recompute Accuracy, Precision, Recall, F1 and MCC from it
by hand, and compare the result with the numbers in the README comparison
table. This test does exactly that in code, for all five saved classifiers
against the real held-out `test_data.csv`, so a silent mismatch (a stale
results row, a wrong averaging choice, a class-order bug) would be caught
here instead of by the evaluator.
"""

import numpy as np
import pandas as pd
import pytest

from gradpath.backend import dependency_wiring
from gradpath.backend.config.settings import HELD_OUT_TEST_DATA_FILE
from gradpath.backend.domain.classifier_kind import ClassifierKind
from gradpath.backend.domain.feature_catalog import STUDENT_FEATURE_CATALOG

pytestmark = pytest.mark.integration


def recompute_macro_metrics_from_confusion_matrix(confusion_matrix: list[list[int]]) -> dict:
    """Recompute Accuracy, macro Precision/Recall/F1 and MCC from a matrix alone.

    This mirrors the arithmetic an evaluator would do by hand from the
    confusion matrix shown in the app or the README, independently of
    `ScoresCalculator`, so a bug in that calculator cannot hide from this
    cross-check.

    Args:
        confusion_matrix: Rows are true outcomes, columns are predicted
            outcomes, in the fixed outcome order used throughout the project.

    Returns:
        A dict with the hand-derived "accuracy", "precision", "recall", "f1"
        and "mcc" values. AUC is not included: it needs predicted
        probabilities, which a confusion matrix alone does not carry.
    """
    matrix = np.array(confusion_matrix, dtype=float)
    total_records = matrix.sum()
    correct_predictions = np.trace(matrix)
    true_counts_per_class = matrix.sum(axis=1)
    predicted_counts_per_class = matrix.sum(axis=0)
    diagonal = np.diag(matrix)

    # zero_division is treated as 0, matching ScoresCalculator's own choice,
    # for an outcome that is never predicted or never truly occurs.
    precision_per_class = np.divide(
        diagonal,
        predicted_counts_per_class,
        out=np.zeros_like(diagonal),
        where=predicted_counts_per_class != 0,
    )
    recall_per_class = np.divide(
        diagonal,
        true_counts_per_class,
        out=np.zeros_like(diagonal),
        where=true_counts_per_class != 0,
    )
    precision_plus_recall = precision_per_class + recall_per_class
    f1_per_class = np.divide(
        2 * precision_per_class * recall_per_class,
        precision_plus_recall,
        out=np.zeros_like(diagonal),
        where=precision_plus_recall != 0,
    )

    # Multiclass Matthews Correlation Coefficient, generalised directly from
    # the confusion matrix (Gorodkin, 2004), the same formula scikit-learn
    # uses internally for matthews_corrcoef with more than two classes.
    covariance_numerator = correct_predictions * total_records - np.dot(
        predicted_counts_per_class, true_counts_per_class
    )
    covariance_denominator = np.sqrt(
        (total_records**2 - np.dot(predicted_counts_per_class, predicted_counts_per_class))
        * (total_records**2 - np.dot(true_counts_per_class, true_counts_per_class))
    )
    mcc = 0.0 if covariance_denominator == 0 else covariance_numerator / covariance_denominator

    return {
        "accuracy": float(correct_predictions / total_records),
        "precision": float(np.mean(precision_per_class)),
        "recall": float(np.mean(recall_per_class)),
        "f1": float(np.mean(f1_per_class)),
        "mcc": float(mcc),
    }


@pytest.fixture
def held_out_records() -> pd.DataFrame:
    """The real, committed held-out test split, never used for training."""
    return pd.read_csv(HELD_OUT_TEST_DATA_FILE)


@pytest.mark.parametrize("classifier_kind", ClassifierKind.all_kinds())
def test_confusion_matrix_reproduces_the_published_scores(classifier_kind, held_out_records):
    """Scoring test_data.csv must give a confusion matrix whose hand-derived
    metrics match both the evaluation service's own scores and the official
    scores published in results/evaluation_results.csv (and the README).
    """
    model_repository = dependency_wiring.build_model_repository()
    confusion_matrix_calculator = dependency_wiring.build_confusion_matrix_calculator()
    evaluation_service = dependency_wiring.build_evaluation_service()
    official_scores = dependency_wiring.build_evaluation_results_repository().load_all_scores()

    pipeline = model_repository.load_model(classifier_kind)
    input_columns = STUDENT_FEATURE_CATALOG.input_columns(list(held_out_records.columns))
    record_inputs = held_out_records[input_columns]
    true_labels = held_out_records[STUDENT_FEATURE_CATALOG.outcome_column]
    predicted_labels = pipeline.predict(record_inputs)

    confusion_matrix = confusion_matrix_calculator.calculate(true_labels, predicted_labels)
    hand_derived_scores = recompute_macro_metrics_from_confusion_matrix(confusion_matrix)

    service_scores = evaluation_service.evaluate_classifier(classifier_kind, held_out_records)
    published_scores = official_scores[classifier_kind]

    for metric_name in ("accuracy", "precision", "recall", "f1", "mcc"):
        hand_derived_value = hand_derived_scores[metric_name]
        service_value = getattr(service_scores, metric_name)
        published_value = getattr(published_scores, metric_name)

        assert hand_derived_value == pytest.approx(service_value, abs=1e-9), (
            f"{classifier_kind.display_name}: confusion-matrix {metric_name} does not "
            "match the evaluation service's own reported value."
        )
        assert hand_derived_value == pytest.approx(published_value, abs=1e-6), (
            f"{classifier_kind.display_name}: confusion-matrix {metric_name} does not "
            "match the published value in evaluation_results.csv."
        )

    # AUC needs predicted probabilities, not just the confusion matrix, so it
    # is cross-checked separately: the service's own AUC must still match
    # what is officially published for this classifier.
    assert service_scores.auc == pytest.approx(published_scores.auc, abs=1e-6), (
        f"{classifier_kind.display_name}: AUC does not match the published value."
    )
