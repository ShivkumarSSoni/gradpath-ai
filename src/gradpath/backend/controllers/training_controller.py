"""Entry point for training: one classifier, or all five."""

import logging

from gradpath.backend.domain.classifier_kind import ClassifierKind
from gradpath.backend.schemas.classifier_training_result import (
    ClassifierTrainingResult,
)
from gradpath.backend.schemas.records_split import RecordsSplit
from gradpath.backend.services.data_preparation_service import DataPreparationService
from gradpath.backend.services.training_service import TrainingService

logger = logging.getLogger(__name__)


class TrainingController:
    """Coordinates the services needed to train classifiers."""

    def __init__(
        self,
        data_preparation_service: DataPreparationService,
        training_service: TrainingService,
    ) -> None:
        self._data_preparation_service = data_preparation_service
        self._training_service = training_service

    def train_one_classifier(self, classifier_kind: ClassifierKind) -> ClassifierTrainingResult:
        """Train, evaluate and save a single classifier.

        Handy to debug one model without retraining the other four.
        """
        records_split = self._data_preparation_service.prepare_records_split()
        return self._train_without_stopping_others(classifier_kind, records_split)

    def train_all_classifiers(self) -> list[ClassifierTrainingResult]:
        """Train every classifier on one shared split, so the comparison is fair.

        One classifier failing does not stop the others. Each result says
        whether that classifier worked.
        """
        records_split = self._data_preparation_service.prepare_records_split()
        return [
            self._train_without_stopping_others(classifier_kind, records_split)
            for classifier_kind in self._training_service.trainable_kinds()
        ]

    def _train_without_stopping_others(
        self,
        classifier_kind: ClassifierKind,
        records_split: RecordsSplit,
    ) -> ClassifierTrainingResult:
        try:
            scores = self._training_service.train_and_evaluate(classifier_kind, records_split)
        except Exception as error:  # noqa: BLE001 - any failure must be reported, not raised
            logger.exception("Training %s failed.", classifier_kind.display_name)
            return ClassifierTrainingResult(
                classifier_kind=classifier_kind,
                error_message=f"{type(error).__name__}: {error}",
            )
        return ClassifierTrainingResult(classifier_kind=classifier_kind, scores=scores)
