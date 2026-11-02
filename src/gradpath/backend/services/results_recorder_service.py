"""Records the official scores of a classifier, and only the official ones.

This is the single writer of the official results file. Scoring an
uploaded file (see `EvaluationService`) must never call this class, so an
upload can never overwrite the comparison numbers the README quotes.
"""

from gradpath.backend.domain.classifier_kind import ClassifierKind
from gradpath.backend.domain.evaluation_scores import EvaluationScores
from gradpath.backend.repositories.interfaces.evaluation_results_repository import (
    EvaluationResultsRepository,
)


class ResultsRecorderService:
    """Writes one classifier's official held-out scores to the results store."""

    def __init__(self, evaluation_results_repository: EvaluationResultsRepository) -> None:
        self._evaluation_results_repository = evaluation_results_repository

    def record_official_scores(
        self,
        classifier_kind: ClassifierKind,
        scores: EvaluationScores,
    ) -> None:
        """Save `scores` as the official result for `classifier_kind`.

        Replaces that classifier's earlier official scores, if any; the
        scores of the other classifiers are left untouched.
        """
        self._evaluation_results_repository.save_scores(classifier_kind, scores)
