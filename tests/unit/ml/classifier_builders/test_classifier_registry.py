import numpy as np
import pytest
from sklearn.base import is_classifier
from sklearn.exceptions import NotFittedError
from sklearn.utils.validation import check_is_fitted

from gradpath.backend.dependency_wiring import build_classifier_registry
from gradpath.backend.domain.classifier_kind import ClassifierKind
from gradpath.backend.ml.classifier_builders.classifier_registry import (
    ClassifierRegistry,
)
from gradpath.backend.ml.classifier_builders.decision_tree_builder import (
    DecisionTreeBuilder,
)


@pytest.fixture
def three_class_data():
    generator = np.random.default_rng(0)
    features = generator.normal(size=(90, 4))
    labels = np.repeat(["a", "b", "c"], 30)
    features[labels == "b"] += 2.0
    features[labels == "c"] -= 2.0
    return features, labels


def test_registry_covers_all_five_kinds_in_standard_order():
    registry = build_classifier_registry()
    assert registry.registered_kinds() == ClassifierKind.all_kinds()


@pytest.mark.parametrize("kind", ClassifierKind.all_kinds())
def test_each_builder_returns_untrained_classifier(kind):
    classifier = build_classifier_registry().build_untrained(kind)
    assert is_classifier(classifier)
    with pytest.raises(NotFittedError):
        check_is_fitted(classifier)


@pytest.mark.parametrize("kind", ClassifierKind.all_kinds())
def test_each_classifier_fits_and_gives_probabilities_for_every_class(
    kind, three_class_data
):
    features, labels = three_class_data
    classifier = build_classifier_registry().build_untrained(kind)
    classifier.fit(features, labels)
    probabilities = classifier.predict_proba(features)
    assert probabilities.shape == (90, 3)
    assert np.allclose(probabilities.sum(axis=1), 1.0)


def test_every_call_returns_a_new_object():
    registry = build_classifier_registry()
    kind = ClassifierKind.DECISION_TREE
    assert registry.build_untrained(kind) is not registry.build_untrained(kind)


@pytest.mark.parametrize("kind", ClassifierKind.all_kinds())
def test_every_classifier_chooses_its_settings_by_cross_validation(kind, three_class_data):
    features, labels = three_class_data
    classifier = build_classifier_registry().build_untrained(kind)

    classifier.fit(features, labels)

    assert set(classifier.best_params_) == set(classifier.param_grid)
    for setting_name, chosen_value in classifier.best_params_.items():
        assert chosen_value in classifier.param_grid[setting_name]


def test_unregistered_kind_raises_key_error():
    registry = ClassifierRegistry([DecisionTreeBuilder(random_seed=1)])
    assert registry.registered_kinds() == [ClassifierKind.DECISION_TREE]
    with pytest.raises(KeyError):
        registry.build_untrained(ClassifierKind.GAUSSIAN_NAIVE_BAYES)


def test_duplicate_builders_are_rejected():
    with pytest.raises(ValueError):
        ClassifierRegistry(
            [DecisionTreeBuilder(random_seed=1), DecisionTreeBuilder(random_seed=2)]
        )
