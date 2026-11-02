"""Turns the recorded official scores into the model comparison table."""

from gradpath.backend.domain.classifier_kind import ClassifierKind
from gradpath.backend.domain.comparison_row import ComparisonRow
from gradpath.backend.repositories.interfaces.evaluation_results_repository import (
    EvaluationResultsRepository,
)


class ComparisonService:
    """Reads the official results file and shapes it into comparison rows."""

    def __init__(self, evaluation_results_repository: EvaluationResultsRepository) -> None:
        self._evaluation_results_repository = evaluation_results_repository

    def build_comparison_rows(self) -> list[ComparisonRow]:
        """Return one row per classifier, in the standard classifier order.

        A classifier with no recorded scores yet still gets a row, with
        `scores` set to None, so the comparison page can show "not available"
        instead of silently dropping it from the table.
        """
        scores_by_kind = self._evaluation_results_repository.load_all_scores()
        return [
            ComparisonRow(classifier_kind=kind, scores=scores_by_kind.get(kind))
            for kind in ClassifierKind.all_kinds()
        ]
