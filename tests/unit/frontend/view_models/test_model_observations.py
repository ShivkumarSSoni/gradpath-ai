"""Every classifier must have its own fixed observation sentence."""

from gradpath.backend.domain.classifier_kind import ClassifierKind
from gradpath.frontend.view_models.model_observations import MODEL_OBSERVATIONS


def test_every_classifier_kind_has_an_observation():
    assert set(MODEL_OBSERVATIONS.keys()) == set(ClassifierKind.all_kinds())


def test_every_observation_is_non_empty_text():
    for observation in MODEL_OBSERVATIONS.values():
        assert isinstance(observation, str)
        assert observation.strip() != ""
