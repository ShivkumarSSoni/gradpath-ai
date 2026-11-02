from gradpath.backend.domain.classifier_kind import ClassifierKind
from gradpath.backend.domain.evaluation_scores import EvaluationScores
from gradpath.backend.repositories.interfaces.evaluation_results_repository import (
    EvaluationResultsRepository,
)
from gradpath.backend.services.comparison_service import ComparisonService

SOME_SCORES = EvaluationScores(accuracy=0.7, auc=0.8, precision=0.6, recall=0.6, f1=0.6, mcc=0.5)


class InMemoryResultsRepository(EvaluationResultsRepository):
    def __init__(self, scores_by_kind):
        self._scores_by_kind = dict(scores_by_kind)

    def save_scores(self, classifier_kind, scores):
        self._scores_by_kind[classifier_kind] = scores

    def load_all_scores(self):
        return dict(self._scores_by_kind)


def test_rows_follow_the_standard_classifier_order_regardless_of_insertion_order():
    repository = InMemoryResultsRepository(
        {
            ClassifierKind.SUPPORT_VECTOR_MACHINE: SOME_SCORES,
            ClassifierKind.DECISION_TREE: SOME_SCORES,
        }
    )
    service = ComparisonService(repository)

    rows = service.build_comparison_rows()

    assert [row.classifier_kind for row in rows] == ClassifierKind.all_kinds()


def test_a_classifier_without_recorded_scores_gets_an_unavailable_row():
    service = ComparisonService(InMemoryResultsRepository({}))

    rows = service.build_comparison_rows()

    assert all(not row.is_available and row.scores is None for row in rows)


def test_a_classifier_with_recorded_scores_gets_an_available_row():
    repository = InMemoryResultsRepository({ClassifierKind.DECISION_TREE: SOME_SCORES})
    service = ComparisonService(repository)

    rows = {row.classifier_kind: row for row in service.build_comparison_rows()}

    assert rows[ClassifierKind.DECISION_TREE].is_available
    assert rows[ClassifierKind.DECISION_TREE].scores == SOME_SCORES
    assert rows[ClassifierKind.DECISION_TREE].display_name == "Decision Tree"
