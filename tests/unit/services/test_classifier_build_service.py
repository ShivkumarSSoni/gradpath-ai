import pytest

from gradpath.backend.domain.classifier_kind import ClassifierKind
from gradpath.backend.domain.evaluation_scores import EvaluationScores
from gradpath.backend.exceptions.classifier_training_error import (
    ClassifierTrainingError,
)
from gradpath.backend.services.classifier_build_service import ClassifierBuildService

SOME_SCORES = EvaluationScores(accuracy=0.7, auc=0.8, precision=0.6, recall=0.6, f1=0.6, mcc=0.5)


class FakeSplit:
    """Stand-in for RecordsSplit, carrying only the held-out records attribute."""

    held_out_records = "held-out-marker"


SPLIT_MARKER = FakeSplit()


class FakeTrainingService:
    def __init__(self):
        self.trained_with = []

    def trainable_kinds(self):
        return ClassifierKind.all_kinds()

    def train_classifier(self, classifier_kind, records_split):
        self.trained_with.append((classifier_kind, records_split))
        return object()


class FakeEvaluationService:
    def __init__(self, raises=None):
        self._raises = raises
        self.evaluated_with = []

    def evaluate_classifier(self, classifier_kind, records):
        if self._raises is not None:
            raise self._raises
        self.evaluated_with.append((classifier_kind, records))
        return SOME_SCORES


class FakeResultsRecorderService:
    def __init__(self):
        self.recorded = []

    def record_official_scores(self, classifier_kind, scores):
        self.recorded.append((classifier_kind, scores))


def test_build_classifier_trains_evaluates_and_records_in_order():
    training = FakeTrainingService()
    evaluation = FakeEvaluationService()
    recorder = FakeResultsRecorderService()
    service = ClassifierBuildService(training, evaluation, recorder)

    scores = service.build_classifier(ClassifierKind.DECISION_TREE, SPLIT_MARKER)

    assert scores == SOME_SCORES
    assert [kind for kind, _ in training.trained_with] == [ClassifierKind.DECISION_TREE]
    assert recorder.recorded == [(ClassifierKind.DECISION_TREE, SOME_SCORES)]


def test_evaluation_uses_the_held_out_records_from_the_split():
    training = FakeTrainingService()
    evaluation = FakeEvaluationService()
    recorder = FakeResultsRecorderService()
    service = ClassifierBuildService(training, evaluation, recorder)

    service.build_classifier(ClassifierKind.DECISION_TREE, FakeSplit())

    assert evaluation.evaluated_with == [(ClassifierKind.DECISION_TREE, "held-out-marker")]


def test_a_failing_evaluation_leaves_nothing_recorded():
    training = FakeTrainingService()
    evaluation = FakeEvaluationService(raises=ClassifierTrainingError("bad order"))
    recorder = FakeResultsRecorderService()
    service = ClassifierBuildService(training, evaluation, recorder)

    with pytest.raises(ClassifierTrainingError):
        service.build_classifier(ClassifierKind.DECISION_TREE, SPLIT_MARKER)

    assert recorder.recorded == []


def test_buildable_kinds_delegates_to_the_training_service():
    service = ClassifierBuildService(
        FakeTrainingService(), FakeEvaluationService(), FakeResultsRecorderService()
    )

    assert service.buildable_kinds() == ClassifierKind.all_kinds()
