"""Component integration test: real registry, pipeline, repositories and controller."""

import pytest

from gradpath.backend import dependency_wiring
from gradpath.backend.controllers.training_controller import TrainingController
from gradpath.backend.domain.classifier_kind import ClassifierKind
from gradpath.backend.domain.student_outcome import StudentOutcome
from gradpath.backend.ml.metrics.scores_calculator import ScoresCalculator
from gradpath.backend.ml.preprocessing.feature_transformer_builder import (
    FeatureTransformerBuilder,
)
from gradpath.backend.repositories.csv_evaluation_results_repository import (
    CsvEvaluationResultsRepository,
)
from gradpath.backend.repositories.interfaces.records_repository import RecordsRepository
from gradpath.backend.repositories.joblib_model_repository import JoblibModelRepository
from gradpath.backend.services.data_preparation_service import DataPreparationService
from gradpath.backend.services.data_split_service import DataSplitService
from gradpath.backend.services.training_service import TrainingService

pytestmark = pytest.mark.integration


class InMemoryRecordsRepository(RecordsRepository):
    def __init__(self, records):
        self._records = records

    def load_all_records(self):
        return self._records

    def save_held_out_records(self, held_out_records):
        pass


@pytest.fixture
def controller_with_temp_storage(tmp_path, tiny_student_records, tiny_feature_catalog):
    models = JoblibModelRepository(tmp_path / "model")
    results = CsvEvaluationResultsRepository(tmp_path / "model" / "evaluation_results.csv")
    training_service = TrainingService(
        classifier_registry=dependency_wiring.build_classifier_registry(),
        feature_transformer_builder=FeatureTransformerBuilder(tiny_feature_catalog),
        scores_calculator=ScoresCalculator(StudentOutcome.all_labels()),
        model_repository=models,
        evaluation_results_repository=results,
        feature_catalog=tiny_feature_catalog,
        expected_class_labels=StudentOutcome.all_labels(),
    )
    preparation = DataPreparationService(
        records_repository=InMemoryRecordsRepository(tiny_student_records),
        data_split_service=DataSplitService("Target", test_fraction=0.2, random_seed=1),
    )
    return TrainingController(preparation, training_service), models, results


def test_training_all_five_saves_five_models_and_five_score_rows(controller_with_temp_storage):
    controller, models, results = controller_with_temp_storage

    outcomes = controller.train_all_classifiers()

    assert all(outcome.succeeded for outcome in outcomes), [o.error_message for o in outcomes]
    assert all(models.model_exists(kind) for kind in ClassifierKind.all_kinds())
    assert list(results.load_all_scores()) == ClassifierKind.all_kinds()


def test_saved_scores_match_the_returned_scores(controller_with_temp_storage):
    controller, _, results = controller_with_temp_storage

    outcomes = controller.train_all_classifiers()

    saved = results.load_all_scores()
    for outcome in outcomes:
        assert saved[outcome.classifier_kind].accuracy == pytest.approx(outcome.scores.accuracy)


def test_training_one_classifier_leaves_the_others_untouched(controller_with_temp_storage):
    controller, models, results = controller_with_temp_storage

    controller.train_one_classifier(ClassifierKind.GAUSSIAN_NAIVE_BAYES)

    assert models.model_exists(ClassifierKind.GAUSSIAN_NAIVE_BAYES)
    assert not models.model_exists(ClassifierKind.DECISION_TREE)
    assert list(results.load_all_scores()) == [ClassifierKind.GAUSSIAN_NAIVE_BAYES]


def test_wiring_builds_the_training_controller():
    assert isinstance(dependency_wiring.build_training_controller(), TrainingController)
