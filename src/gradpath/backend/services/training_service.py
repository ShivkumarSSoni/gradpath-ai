"""Trains one classifier, scores it on held-out records and saves the result."""

from sklearn.pipeline import Pipeline

from gradpath.backend.domain.classifier_kind import ClassifierKind
from gradpath.backend.domain.evaluation_scores import EvaluationScores
from gradpath.backend.domain.feature_catalog import FeatureCatalog
from gradpath.backend.exceptions.classifier_training_error import (
    ClassifierTrainingError,
)
from gradpath.backend.ml.classifier_builders.classifier_registry import (
    ClassifierRegistry,
)
from gradpath.backend.ml.metrics.scores_calculator import ScoresCalculator
from gradpath.backend.ml.preprocessing.feature_transformer_builder import (
    FeatureTransformerBuilder,
)
from gradpath.backend.repositories.interfaces.evaluation_results_repository import (
    EvaluationResultsRepository,
)
from gradpath.backend.repositories.interfaces.model_repository import ModelRepository
from gradpath.backend.schemas.records_split import RecordsSplit


class TrainingService:
    """Turns a classifier kind and a records split into a saved, scored model."""

    def __init__(
        self,
        classifier_registry: ClassifierRegistry,
        feature_transformer_builder: FeatureTransformerBuilder,
        scores_calculator: ScoresCalculator,
        model_repository: ModelRepository,
        evaluation_results_repository: EvaluationResultsRepository,
        feature_catalog: FeatureCatalog,
        expected_class_labels: list[str],
    ) -> None:
        """Receive every collaborator, so each one can be replaced in tests.

        Args:
            expected_class_labels: Outcome names in the order the scores
                calculator reads the probability columns.
        """
        self._classifier_registry = classifier_registry
        self._feature_transformer_builder = feature_transformer_builder
        self._scores_calculator = scores_calculator
        self._model_repository = model_repository
        self._evaluation_results_repository = evaluation_results_repository
        self._feature_catalog = feature_catalog
        self._expected_class_labels = list(expected_class_labels)

    def trainable_kinds(self) -> list[ClassifierKind]:
        """Return the classifiers this service is able to train."""
        return self._classifier_registry.registered_kinds()

    def train_and_evaluate(
        self,
        classifier_kind: ClassifierKind,
        records_split: RecordsSplit,
    ) -> EvaluationScores:
        """Train on the training records, score on the held-out records, then save.

        The preprocessing and the classifier are fitted together as one
        pipeline, and only on the training records. The held-out records are
        used for scoring alone, so the scores are an honest estimate.

        Raises:
            ClassifierTrainingError: If the trained classifier orders the
                outcomes differently from what the scores calculator expects.
        """
        outcome_column = self._feature_catalog.outcome_column
        training_records = records_split.training_records
        held_out_records = records_split.held_out_records
        input_columns = self._feature_catalog.input_columns(list(training_records.columns))

        pipeline = Pipeline(
            steps=[
                ("preprocessing", self._feature_transformer_builder.build(input_columns)),
                ("classifier", self._classifier_registry.build_untrained(classifier_kind)),
            ]
        )
        pipeline.fit(training_records[input_columns], training_records[outcome_column])

        # The probability columns follow the classifier's own class order.
        # If that differs from the calculator's order, the AUC would be wrong.
        if list(pipeline.classes_) != self._expected_class_labels:
            raise ClassifierTrainingError(
                f"{classifier_kind.display_name} orders the outcomes as "
                f"{list(pipeline.classes_)}, expected {self._expected_class_labels}."
            )

        held_out_inputs = held_out_records[input_columns]
        scores = self._scores_calculator.calculate(
            true_labels=held_out_records[outcome_column],
            predicted_labels=pipeline.predict(held_out_inputs),
            predicted_probabilities=pipeline.predict_proba(held_out_inputs),
        )

        self._model_repository.save_model(classifier_kind, pipeline)
        self._evaluation_results_repository.save_scores(classifier_kind, scores)
        return scores
