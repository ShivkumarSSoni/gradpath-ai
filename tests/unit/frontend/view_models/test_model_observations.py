"""Every classifier must have a fixed reason and a correctly combined text."""

from gradpath.backend.domain.classifier_kind import ClassifierKind
from gradpath.frontend.view_models.model_observations import (
    MODEL_OBSERVATION_REASONS,
    build_observation_text,
)


def test_every_classifier_kind_has_a_reason():
    assert set(MODEL_OBSERVATION_REASONS.keys()) == set(ClassifierKind.all_kinds())


def test_every_reason_is_non_empty_text():
    for reason in MODEL_OBSERVATION_REASONS.values():
        assert isinstance(reason, str)
        assert reason.strip() != ""


def test_build_observation_text_appends_the_live_scores_in_order():
    scores = {
        "Accuracy": 0.767,
        "AUC": 0.881,
        "Precision": 0.712,
        "Recall": 0.685,
        "F1": 0.693,
        "MCC": 0.614,
    }

    text = build_observation_text(ClassifierKind.LOGISTIC_REGRESSION, scores)

    assert text.startswith(MODEL_OBSERVATION_REASONS[ClassifierKind.LOGISTIC_REGRESSION])
    assert "Accuracy 0.767" in text
    assert "AUC 0.881" in text
    assert "MCC 0.614" in text


def test_build_observation_text_falls_back_to_the_reason_when_scores_are_missing():
    text = build_observation_text(ClassifierKind.DECISION_TREE, {})

    assert text == MODEL_OBSERVATION_REASONS[ClassifierKind.DECISION_TREE]
