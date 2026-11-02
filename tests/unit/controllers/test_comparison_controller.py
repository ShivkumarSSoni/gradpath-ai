from gradpath.backend.controllers.comparison_controller import ComparisonController
from gradpath.backend.domain.classifier_kind import ClassifierKind
from gradpath.backend.domain.comparison_row import ComparisonRow

SOME_ROWS = [ComparisonRow(classifier_kind=ClassifierKind.DECISION_TREE, scores=None)]


class FakeComparisonService:
    def build_comparison_rows(self):
        return SOME_ROWS


def test_comparison_controller_returns_whatever_the_service_builds():
    controller = ComparisonController(FakeComparisonService())

    assert controller.compare_all_classifiers() == SOME_ROWS
