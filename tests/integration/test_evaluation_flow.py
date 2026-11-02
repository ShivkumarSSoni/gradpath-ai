"""Component integration test: evaluation controller against the real saved models.

Uses the actual `model/*.joblib` files and the committed `test_data.csv`, so
this is the same path the Streamlit app will take when a user uploads the
held-out test file.
"""

import pandas as pd
import pytest

from gradpath.backend import dependency_wiring
from gradpath.backend.config.settings import EVALUATION_RESULTS_FILE, HELD_OUT_TEST_DATA_FILE
from gradpath.backend.domain.classifier_kind import ClassifierKind
from gradpath.backend.schemas.evaluation_request import EvaluationRequest

pytestmark = pytest.mark.integration


def read_results_file_bytes() -> bytes:
    return EVALUATION_RESULTS_FILE.read_bytes()


@pytest.fixture
def uploaded_test_records() -> pd.DataFrame:
    return pd.read_csv(HELD_OUT_TEST_DATA_FILE)


def test_evaluating_an_upload_scores_it_without_retraining(uploaded_test_records):
    controller = dependency_wiring.build_evaluation_controller()
    request = EvaluationRequest(
        classifier_kind=ClassifierKind.DECISION_TREE, records=uploaded_test_records
    )

    response = controller.evaluate_uploaded_records(request)

    assert response.succeeded, response.error_message
    assert 0.0 <= response.report.scores.accuracy <= 1.0
    assert len(response.report.confusion_matrix) == 3
    assert "Dropout" in response.report.classification_report


def test_evaluating_an_upload_leaves_the_official_results_file_unchanged(uploaded_test_records):
    results_before = read_results_file_bytes()
    controller = dependency_wiring.build_evaluation_controller()
    request = EvaluationRequest(
        classifier_kind=ClassifierKind.LOGISTIC_REGRESSION, records=uploaded_test_records
    )

    response = controller.evaluate_uploaded_records(request)

    assert response.succeeded, response.error_message
    assert read_results_file_bytes() == results_before


def test_an_upload_missing_the_outcome_column_is_rejected_with_a_clear_message(
    uploaded_test_records,
):
    controller = dependency_wiring.build_evaluation_controller()
    request = EvaluationRequest(
        classifier_kind=ClassifierKind.DECISION_TREE,
        records=uploaded_test_records.drop(columns=["Target"]),
    )

    response = controller.evaluate_uploaded_records(request)

    assert not response.succeeded
    assert "Target" in response.error_message


def test_the_comparison_controller_reads_the_official_results():
    controller = dependency_wiring.build_comparison_controller()

    rows = controller.compare_all_classifiers()

    assert [row.classifier_kind for row in rows] == ClassifierKind.all_kinds()
    assert all(row.is_available for row in rows)


def test_wiring_builds_the_evaluation_and_comparison_controllers():
    from gradpath.backend.controllers.comparison_controller import ComparisonController
    from gradpath.backend.controllers.evaluation_controller import EvaluationController

    assert isinstance(dependency_wiring.build_evaluation_controller(), EvaluationController)
    assert isinstance(dependency_wiring.build_comparison_controller(), ComparisonController)
