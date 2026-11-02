"""The official workflow for one classifier: train, evaluate, record.

This is the only service that produces the numbers the README and the
notebook call "official": it always evaluates on the same held-out split
that every other classifier is compared against, and it is the only caller
of `ResultsRecorderService`.
"""

from gradpath.backend.domain.classifier_kind import ClassifierKind
from gradpath.backend.domain.evaluation_scores import EvaluationScores
from gradpath.backend.schemas.records_split import RecordsSplit
from gradpath.backend.services.evaluation_service import EvaluationService
from gradpath.backend.services.results_recorder_service import ResultsRecorderService
from gradpath.backend.services.training_service import TrainingService


class ClassifierBuildService:
    """Builds the official result for one classifier, or for all of them."""

    def __init__(
        self,
        training_service: TrainingService,
        evaluation_service: EvaluationService,
        results_recorder_service: ResultsRecorderService,
    ) -> None:
        self._training_service = training_service
        self._evaluation_service = evaluation_service
        self._results_recorder_service = results_recorder_service

    def buildable_kinds(self) -> list[ClassifierKind]:
        """Return the classifiers this service is able to build."""
        return self._training_service.trainable_kinds()

    def build_classifier(
        self,
        classifier_kind: ClassifierKind,
        records_split: RecordsSplit,
    ) -> EvaluationScores:
        """Train on the training records, score on the held-out records, record.

        Raises:
            ClassifierTrainingError: If the trained pipeline orders the
                outcomes differently from what the scores calculator expects.
                The model stays saved (training already finished); only the
                official scores are left unrecorded.
        """
        self._training_service.train_classifier(classifier_kind, records_split)
        scores = self._evaluation_service.evaluate_classifier(
            classifier_kind, records_split.held_out_records
        )
        self._results_recorder_service.record_official_scores(classifier_kind, scores)
        return scores
