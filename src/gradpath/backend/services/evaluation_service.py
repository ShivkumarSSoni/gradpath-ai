"""Scores a saved classifier on any table of records (held-out or uploaded).

This is the one place that turns "a saved model plus some records" into the
six evaluation scores. It is used by the official build workflow and, later,
by the Streamlit app when a user uploads a test file, so an upload must be
able to reuse it without ever retraining or touching the saved results.
"""

from gradpath.backend.domain.classifier_kind import ClassifierKind
from gradpath.backend.domain.evaluation_scores import EvaluationScores
from gradpath.backend.domain.feature_catalog import FeatureCatalog
from gradpath.backend.exceptions.classifier_training_error import (
    ClassifierTrainingError,
)
from gradpath.backend.ml.metrics.scores_calculator import ScoresCalculator
from gradpath.backend.repositories.interfaces.model_repository import ModelRepository
from pandas import DataFrame


class EvaluationService:
    """Loads a saved model and scores it on a table of labelled records."""

    def __init__(
        self,
        model_repository: ModelRepository,
        scores_calculator: ScoresCalculator,
        feature_catalog: FeatureCatalog,
        expected_class_labels: list[str],
    ) -> None:
        """Receive every collaborator, so each one can be replaced in tests.

        Args:
            expected_class_labels: Outcome names in the order the scores
                calculator reads the probability columns.
        """
        self._model_repository = model_repository
        self._scores_calculator = scores_calculator
        self._feature_catalog = feature_catalog
        self._expected_class_labels = list(expected_class_labels)

    def evaluate_classifier(
        self,
        classifier_kind: ClassifierKind,
        records: DataFrame,
    ) -> EvaluationScores:
        """Load the saved model for `classifier_kind` and score it on `records`.

        Args:
            records: Labelled records to score, including the outcome column.
                Only the input columns and the outcome column are read; the
                saved pipeline handles its own preprocessing.

        Raises:
            ModelNotFoundError: If no model has been saved for this kind.
            ClassifierTrainingError: If the saved pipeline orders the outcomes
                differently from what the scores calculator expects. This
                would make the AUC silently wrong, so it is refused instead.
        """
        pipeline = self._model_repository.load_model(classifier_kind)

        # The probability columns follow the classifier's own class order.
        # If that differs from the calculator's order, the AUC would be wrong.
        if list(pipeline.classes_) != self._expected_class_labels:
            raise ClassifierTrainingError(
                f"{classifier_kind.display_name} orders the outcomes as "
                f"{list(pipeline.classes_)}, expected {self._expected_class_labels}."
            )

        outcome_column = self._feature_catalog.outcome_column
        input_columns = self._feature_catalog.input_columns(list(records.columns))
        record_inputs = records[input_columns]

        return self._scores_calculator.calculate(
            true_labels=records[outcome_column],
            predicted_labels=pipeline.predict(record_inputs),
            predicted_probabilities=pipeline.predict_proba(record_inputs),
        )
