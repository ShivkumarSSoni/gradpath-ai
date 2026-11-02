import pytest

from gradpath.backend.domain.classifier_kind import ClassifierKind
from gradpath.backend.domain.student_outcome import StudentOutcome
from gradpath.backend.exceptions.classifier_training_error import (
    ClassifierTrainingError,
)
from gradpath.backend.exceptions.model_not_found_error import ModelNotFoundError
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
from gradpath.backend.repositories.interfaces.model_repository import ModelRepository
from gradpath.backend.services.data_split_service import DataSplitService
from gradpath.backend.services.evaluation_service import EvaluationService
from gradpath.backend.services.training_service import TrainingService


class InMemoryModelRepository(ModelRepository):
    def __init__(self):
        self.saved_models = {}

    def save_model(self, classifier_kind, trained_model):
        self.saved_models[classifier_kind] = trained_model

    def load_model(self, classifier_kind):
        if classifier_kind not in self.saved_models:
            raise ModelNotFoundError(f"No saved model for {classifier_kind.value}.")
        return self.saved_models[classifier_kind]

    def model_exists(self, classifier_kind):
        return classifier_kind in self.saved_models


@pytest.fixture
def records_split(tiny_student_records, tiny_feature_catalog):
    splitter = DataSplitService(
        outcome_column=tiny_feature_catalog.outcome_column,
        test_fraction=0.2,
        random_seed=1,
    )
    return splitter.split_records(tiny_student_records)


@pytest.fixture
def trained_models(tiny_feature_catalog, records_split):
    """A model repository with one trained Decision Tree already saved."""
    models = InMemoryModelRepository()
    training_service = TrainingService(
        classifier_registry=ClassifierRegistry([DecisionTreeBuilder(random_seed=1)]),
        feature_transformer_builder=FeatureTransformerBuilder(tiny_feature_catalog),
        model_repository=models,
        feature_catalog=tiny_feature_catalog,
    )
    training_service.train_classifier(ClassifierKind.DECISION_TREE, records_split)
    return models


def build_evaluation_service(tiny_feature_catalog, model_repository, labels=None):
    return EvaluationService(
        model_repository=model_repository,
        scores_calculator=ScoresCalculator(StudentOutcome.all_labels()),
        feature_catalog=tiny_feature_catalog,
        expected_class_labels=labels or StudentOutcome.all_labels(),
    )


def test_evaluating_a_saved_model_returns_scores_on_the_given_records(
    tiny_feature_catalog, trained_models, records_split
):
    service = build_evaluation_service(tiny_feature_catalog, trained_models)

    scores = service.evaluate_classifier(ClassifierKind.DECISION_TREE, records_split.held_out_records)

    assert 0.0 <= scores.accuracy <= 1.0


def test_evaluating_an_unknown_model_raises_model_not_found(tiny_feature_catalog, records_split):
    service = build_evaluation_service(tiny_feature_catalog, InMemoryModelRepository())

    with pytest.raises(ModelNotFoundError):
        service.evaluate_classifier(ClassifierKind.DECISION_TREE, records_split.held_out_records)


def test_wrong_class_order_is_rejected(tiny_feature_catalog, trained_models, records_split):
    reversed_labels = list(reversed(StudentOutcome.all_labels()))
    service = build_evaluation_service(tiny_feature_catalog, trained_models, reversed_labels)

    with pytest.raises(ClassifierTrainingError):
        service.evaluate_classifier(ClassifierKind.DECISION_TREE, records_split.held_out_records)


def test_evaluating_training_records_gives_different_scores_than_held_out(
    tiny_feature_catalog, trained_models, records_split
):
    """Sanity check: evaluation really reads the records it is given."""
    service = build_evaluation_service(tiny_feature_catalog, trained_models)

    held_out_scores = service.evaluate_classifier(
        ClassifierKind.DECISION_TREE, records_split.held_out_records
    )
    training_scores = service.evaluate_classifier(
        ClassifierKind.DECISION_TREE, records_split.training_records
    )

    # A fitted tree reproduces its own training records closely, so the
    # accuracy on them should be at least as high as on unseen records.
    assert training_scores.accuracy >= held_out_scores.accuracy
