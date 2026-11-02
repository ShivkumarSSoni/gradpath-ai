from gradpath.backend.controllers.training_controller import TrainingController
from gradpath.backend.domain.classifier_kind import ClassifierKind
from gradpath.backend.domain.evaluation_scores import EvaluationScores

SOME_SCORES = EvaluationScores(accuracy=0.7, auc=0.8, precision=0.6, recall=0.6, f1=0.6, mcc=0.5)
SPLIT_MARKER = object()


class FakeDataPreparationService:
    def __init__(self):
        self.prepare_count = 0

    def prepare_records_split(self):
        self.prepare_count += 1
        return SPLIT_MARKER


class FakeClassifierBuildService:
    """Pretends to build classifiers; fails for the kinds listed in `failing_kinds`."""

    def __init__(self, failing_kinds=()):
        self._failing_kinds = set(failing_kinds)
        self.built_with = []

    def buildable_kinds(self):
        return ClassifierKind.all_kinds()

    def build_classifier(self, classifier_kind, records_split):
        if classifier_kind in self._failing_kinds:
            raise RuntimeError("simulated failure")
        self.built_with.append((classifier_kind, records_split))
        return SOME_SCORES


def test_train_all_returns_one_successful_result_per_kind():
    controller = TrainingController(FakeDataPreparationService(), FakeClassifierBuildService())

    results = controller.train_all_classifiers()

    assert [result.classifier_kind for result in results] == ClassifierKind.all_kinds()
    assert all(result.succeeded and result.scores == SOME_SCORES for result in results)


def test_all_classifiers_share_one_split_prepared_once():
    preparation = FakeDataPreparationService()
    build_service = FakeClassifierBuildService()

    TrainingController(preparation, build_service).train_all_classifiers()

    assert preparation.prepare_count == 1
    assert {split for _, split in build_service.built_with} == {SPLIT_MARKER}


def test_one_failing_classifier_does_not_block_the_others():
    failing_kind = ClassifierKind.K_NEAREST_NEIGHBORS
    controller = TrainingController(
        FakeDataPreparationService(), FakeClassifierBuildService(failing_kinds=[failing_kind])
    )

    results = {result.classifier_kind: result for result in controller.train_all_classifiers()}

    assert not results[failing_kind].succeeded
    assert "simulated failure" in results[failing_kind].error_message
    assert results[failing_kind].scores is None
    assert sum(result.succeeded for result in results.values()) == 4


def test_train_one_trains_only_the_requested_classifier():
    build_service = FakeClassifierBuildService()
    controller = TrainingController(FakeDataPreparationService(), build_service)

    result = controller.train_one_classifier(ClassifierKind.DECISION_TREE)

    assert result.succeeded
    assert [kind for kind, _ in build_service.built_with] == [ClassifierKind.DECISION_TREE]
