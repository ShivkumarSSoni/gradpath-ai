"""Fits one classifier pipeline on training records and saves it.

Scoring the fitted pipeline is a separate concern (see `EvaluationService`),
because the Streamlit app scores saved models on uploaded files too, and
that must never retrain anything or touch this service.
"""

from sklearn.base import BaseEstimator
from sklearn.pipeline import Pipeline

from gradpath.backend.domain.classifier_kind import ClassifierKind
from gradpath.backend.domain.feature_catalog import FeatureCatalog
from gradpath.backend.ml.classifier_builders.classifier_registry import (
    ClassifierRegistry,
)
from gradpath.backend.ml.preprocessing.feature_transformer_builder import (
    FeatureTransformerBuilder,
)
from gradpath.backend.repositories.interfaces.model_repository import ModelRepository
from gradpath.backend.schemas.records_split import RecordsSplit


class TrainingService:
    """Fits one classifier kind on the training records and saves it.

    This service has exactly one job: produce a saved, fitted pipeline. It
    never scores anything and never reads the held-out records.
    """

    def __init__(
        self,
        classifier_registry: ClassifierRegistry,
        feature_transformer_builder: FeatureTransformerBuilder,
        model_repository: ModelRepository,
        feature_catalog: FeatureCatalog,
    ) -> None:
        """Receive every collaborator, so each one can be replaced in tests."""
        self._classifier_registry = classifier_registry
        self._feature_transformer_builder = feature_transformer_builder
        self._model_repository = model_repository
        self._feature_catalog = feature_catalog

    def trainable_kinds(self) -> list[ClassifierKind]:
        """Return the classifiers this service is able to train."""
        return self._classifier_registry.registered_kinds()

    def train_classifier(
        self,
        classifier_kind: ClassifierKind,
        records_split: RecordsSplit,
    ) -> BaseEstimator:
        """Fit a pipeline on the training records only, save it, and return it.

        The preprocessing and the classifier are fitted together as one
        pipeline, and only on the training records, so nothing from the
        held-out records leaks into the fitted model.
        """
        outcome_column = self._feature_catalog.outcome_column
        training_records = records_split.training_records
        input_columns = self._feature_catalog.input_columns(list(training_records.columns))

        pipeline = Pipeline(
            steps=[
                ("preprocessing", self._feature_transformer_builder.build(input_columns)),
                ("classifier", self._classifier_registry.build_untrained(classifier_kind)),
            ]
        )
        pipeline.fit(training_records[input_columns], training_records[outcome_column])

        self._model_repository.save_model(classifier_kind, pipeline)
        return pipeline
