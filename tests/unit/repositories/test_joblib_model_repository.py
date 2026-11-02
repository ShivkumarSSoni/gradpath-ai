import numpy as np
import pytest
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

from gradpath.backend.domain.classifier_kind import ClassifierKind
from gradpath.backend.exceptions.model_not_found_error import ModelNotFoundError
from gradpath.backend.repositories.joblib_model_repository import JoblibModelRepository


@pytest.fixture
def small_fitted_pipeline():
    features = np.array([[0.0], [1.0], [2.0], [3.0]])
    labels = ["a", "a", "b", "b"]
    return make_pipeline(StandardScaler(), DecisionTreeClassifier()).fit(features, labels)


def test_saved_model_loads_back_and_predicts_the_same(tmp_path, small_fitted_pipeline):
    repository = JoblibModelRepository(tmp_path)
    kind = ClassifierKind.DECISION_TREE

    repository.save_model(kind, small_fitted_pipeline)
    loaded_model = repository.load_model(kind)

    sample = np.array([[0.5], [2.5]])
    assert list(loaded_model.predict(sample)) == list(small_fitted_pipeline.predict(sample))


def test_each_kind_is_saved_in_its_own_file(tmp_path, small_fitted_pipeline):
    repository = JoblibModelRepository(tmp_path)

    repository.save_model(ClassifierKind.DECISION_TREE, small_fitted_pipeline)

    assert (tmp_path / "decision_tree.joblib").is_file()
    assert repository.model_exists(ClassifierKind.DECISION_TREE)
    assert not repository.model_exists(ClassifierKind.GAUSSIAN_NAIVE_BAYES)


def test_missing_folder_is_created_on_first_save(tmp_path, small_fitted_pipeline):
    repository = JoblibModelRepository(tmp_path / "not" / "created" / "yet")

    repository.save_model(ClassifierKind.DECISION_TREE, small_fitted_pipeline)

    assert repository.model_exists(ClassifierKind.DECISION_TREE)


def test_loading_a_model_that_was_never_saved_raises_a_clear_error(tmp_path):
    repository = JoblibModelRepository(tmp_path)

    with pytest.raises(ModelNotFoundError, match="Decision Tree"):
        repository.load_model(ClassifierKind.DECISION_TREE)


def test_saving_again_replaces_the_earlier_model(tmp_path, small_fitted_pipeline):
    repository = JoblibModelRepository(tmp_path)
    kind = ClassifierKind.DECISION_TREE
    other_pipeline = make_pipeline(StandardScaler(), DecisionTreeClassifier()).fit(
        np.array([[0.0], [1.0]]), ["x", "y"]
    )

    repository.save_model(kind, small_fitted_pipeline)
    repository.save_model(kind, other_pipeline)

    assert list(repository.load_model(kind).classes_) == ["x", "y"]
