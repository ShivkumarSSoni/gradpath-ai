import pytest

from gradpath.backend.domain.classifier_kind import ClassifierKind
from gradpath.backend.domain.student_outcome import StudentOutcome
from gradpath.backend.ml.classifier_builders.decision_tree_builder import (
    DecisionTreeBuilder,
)
from gradpath.backend.ml.classifier_builders.classifier_registry import (
    ClassifierRegistry,
)
from gradpath.backend.ml.preprocessing.feature_transformer_builder import (
    FeatureTransformerBuilder,
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


@pytest.fixture
def records_split(tiny_student_records, tiny_feature_catalog):
    splitter = DataSplitService(
        outcome_column=tiny_feature_catalog.outcome_column,
        test_fraction=0.2,
        random_seed=1,
    )
    return splitter.split_records(tiny_student_records)


def build_training_service(tiny_feature_catalog, model_repository):
    return TrainingService(
        classifier_registry=ClassifierRegistry([DecisionTreeBuilder(random_seed=1)]),
        feature_transformer_builder=FeatureTransformerBuilder(tiny_feature_catalog),
        model_repository=model_repository,
        feature_catalog=tiny_feature_catalog,
    )


def test_training_returns_the_fitted_pipeline_and_saves_it(tiny_feature_catalog, records_split):
    models = InMemoryModelRepository()
    service = build_training_service(tiny_feature_catalog, models)

    pipeline = service.train_classifier(ClassifierKind.DECISION_TREE, records_split)

    assert models.load_model(ClassifierKind.DECISION_TREE) is pipeline
    assert models.model_exists(ClassifierKind.DECISION_TREE)


def test_saved_model_accepts_raw_records_without_the_outcome_column(
    tiny_feature_catalog, records_split
):
    models = InMemoryModelRepository()
    service = build_training_service(tiny_feature_catalog, models)
    service.train_classifier(ClassifierKind.DECISION_TREE, records_split)
    raw_inputs = records_split.held_out_records.drop(columns=["Target"])

    predictions = models.load_model(ClassifierKind.DECISION_TREE).predict(raw_inputs)

    assert len(predictions) == len(raw_inputs)
    assert set(predictions) <= set(StudentOutcome.all_labels())


def test_preprocessing_is_fitted_on_training_records_only(tiny_feature_catalog, records_split):
    models = InMemoryModelRepository()
    service = build_training_service(tiny_feature_catalog, models)
    service.train_classifier(ClassifierKind.DECISION_TREE, records_split)

    scaler = models.load_model(ClassifierKind.DECISION_TREE).named_steps[
        "preprocessing"
    ].named_transformers_["numeric_measurements"]
    training_mean = records_split.training_records[["Age at enrollment", "Admission grade"]].mean()
    held_out_mean = records_split.held_out_records[["Age at enrollment", "Admission grade"]].mean()

    assert list(scaler.mean_) == pytest.approx(list(training_mean))
    assert list(scaler.mean_) != pytest.approx(list(held_out_mean))


def test_only_registered_kinds_are_trainable(tiny_feature_catalog):
    service = build_training_service(tiny_feature_catalog, InMemoryModelRepository())

    assert service.trainable_kinds() == [ClassifierKind.DECISION_TREE]
