from gradpath.backend.domain.classifier_kind import ClassifierKind
from gradpath.backend.domain.evaluation_scores import EvaluationScores
from gradpath.backend.repositories.interfaces.evaluation_results_repository import (
    EvaluationResultsRepository,
)
from gradpath.backend.services.results_recorder_service import ResultsRecorderService

SOME_SCORES = EvaluationScores(accuracy=0.7, auc=0.8, precision=0.6, recall=0.6, f1=0.6, mcc=0.5)


class InMemoryResultsRepository(EvaluationResultsRepository):
    def __init__(self):
        self.saved_scores = {}

    def save_scores(self, classifier_kind, scores):
        self.saved_scores[classifier_kind] = scores

    def load_all_scores(self):
        return dict(self.saved_scores)


def test_recording_scores_saves_them_through_the_repository():
    repository = InMemoryResultsRepository()
    recorder = ResultsRecorderService(repository)

    recorder.record_official_scores(ClassifierKind.DECISION_TREE, SOME_SCORES)

    assert repository.saved_scores == {ClassifierKind.DECISION_TREE: SOME_SCORES}


def test_recording_one_classifier_keeps_the_others():
    repository = InMemoryResultsRepository()
    recorder = ResultsRecorderService(repository)
    recorder.record_official_scores(ClassifierKind.DECISION_TREE, SOME_SCORES)

    recorder.record_official_scores(ClassifierKind.LOGISTIC_REGRESSION, SOME_SCORES)

    assert set(repository.saved_scores) == {
        ClassifierKind.DECISION_TREE,
        ClassifierKind.LOGISTIC_REGRESSION,
    }
