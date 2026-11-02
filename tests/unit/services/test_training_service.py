import pytest

from gradpath.backend.domain.classifier_kind import ClassifierKind
from gradpath.backend.domain.evaluation_scores import EvaluationScores
from gradpath.backend.domain.student_outcome import StudentOutcome
from gradpath.backend.exceptions.classifier_training_error import (
    ClassifierTrainingError,
)
from gradpath.backend.ml.classifier_builders.decision_tree_builder import (
    DecisionTreeBuilder,
)
from gradpath.backend.ml.classifier_builders.classifier_registry import (
    ClassifierRegistry,
)
from gradpath.backend.ml.metrics.scores_calculator import ScoresCalculator
from gradpath.backend.ml.preprocessing.feature_transformer_builder import (
    FeatureTransformerBuilder,
)
from gradpath.backend.repositories.interfaces.evaluation_results_repository import (
    EvaluationResultsRepository,
)
from gradpath.backend.repositories.interfaces.model_repository import ModelRepository
from gradpath.backend.services.data_split_service import DataSplitService
from gradpath.backend.services.training_service import TrainingService


class InMemoryModelRepository(ModelRepository):
    def __init__(self):
        self.saved_models = {}

    def save_model(self, classifier_kind, trained_model):
        self.saved_models[classifier_kind] = trained_model

    def load_model(self, classifier_kind):
        return self.saved_models[classifier_kind]

    def model_exists(self, classifier_kind):
        return classifier_kind in self.saved_models


class InMemoryResultsRepository(EvaluationResultsRepository):
    def __init__(self):
        self.saved_scores = {}

    def save_scores(self, classifier_kind, scores):
        self.saved_scores[classifier_kind] = scores

    def load_all_scores(self):
        return dict(self.saved_scores)


@pytest.fixture
def records_split(tiny_student_records, tiny_feature_catalog):
    splitter = DataSplitService(
        outcome_column=tiny_feature_catalog.outcome_column,
        test_fraction=0.2,
        random_seed=1,
    )
    return splitter.split_records(tiny_student_records)


def build_training_service(tiny_feature_catalog, model_repository, results_repository, labels=None):
    return TrainingService(
        classifier_registry=ClassifierRegistry([DecisionTreeBuilder(random_seed=1)]),
        feature_transformer_builder=FeatureTransformerBuilder(tiny_feature_catalog),
        scores_calculator=ScoresCalculator(StudentOutcome.all_labels()),
        model_repository=model_repository,
        evaluation_results_repository=results_repository,
        feature_catalog=tiny_feature_catalog,
        expected_class_labels=labels or StudentOutcome.all_labels(),
    )


def test_training_returns_scores_and_saves_model_and_scores(tiny_feature_catalog, records_split):
    models, results = InMemoryModelRepository(), InMemoryResultsRepository()
    service = build_training_service(tiny_feature_catalog, models, results)

    scores = service.train_and_evaluate(ClassifierKind.DECISION_TREE, records_split)

    assert isinstance(scores, EvaluationScores)
    assert results.saved_scores[ClassifierKind.DECISION_TREE] == scores
    assert models.model_exists(ClassifierKind.DECISION_TREE)


def test_saved_model_accepts_raw_records_without_the_outcome_column(
    tiny_feature_catalog, records_split
):
    models, results = InMemoryModelRepository(), InMemoryResultsRepository()
    service = build_training_service(tiny_feature_catalog, models, results)
    service.train_and_evaluate(ClassifierKind.DECISION_TREE, records_split)
    raw_inputs = records_split.held_out_records.drop(columns=["Target"])

    predictions = models.load_model(ClassifierKind.DECISION_TREE).predict(raw_inputs)

    assert len(predictions) == len(raw_inputs)
    assert set(predictions) <= set(StudentOutcome.all_labels())


def test_preprocessing_is_fitted_on_training_records_only(tiny_feature_catalog, records_split):
    models, results = InMemoryModelRepository(), InMemoryResultsRepository()
    service = build_training_service(tiny_feature_catalog, models, results)
    service.train_and_evaluate(ClassifierKind.DECISION_TREE, records_split)

    scaler = models.load_model(ClassifierKind.DECISION_TREE).named_steps[
        "preprocessing"
    ].named_transformers_["numeric_measurements"]
    training_mean = records_split.training_records[["Age at enrollment", "Admission grade"]].mean()
    held_out_mean = records_split.held_out_records[["Age at enrollment", "Admission grade"]].mean()

    assert list(scaler.mean_) == pytest.approx(list(training_mean))
    assert list(scaler.mean_) != pytest.approx(list(held_out_mean))


def test_wrong_class_order_is_rejected_and_nothing_is_saved(tiny_feature_catalog, records_split):
    models, results = InMemoryModelRepository(), InMemoryResultsRepository()
    reversed_labels = list(reversed(StudentOutcome.all_labels()))
    service = build_training_service(tiny_feature_catalog, models, results, reversed_labels)

    with pytest.raises(ClassifierTrainingError):
        service.train_and_evaluate(ClassifierKind.DECISION_TREE, records_split)

    assert models.saved_models == {}
    assert results.saved_scores == {}


def test_only_registered_kinds_are_trainable(tiny_feature_catalog):
    service = build_training_service(
        tiny_feature_catalog, InMemoryModelRepository(), InMemoryResultsRepository()
    )

    assert service.trainable_kinds() == [ClassifierKind.DECISION_TREE]
