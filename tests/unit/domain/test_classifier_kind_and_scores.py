"""Unit tests for the domain value objects and the classifier list."""

import dataclasses
import math

import pytest

from gradpath.backend.domain.classifier_kind import ClassifierKind
from gradpath.backend.domain.evaluation_scores import EvaluationScores


def test_there_are_exactly_five_classifiers():
    assert len(ClassifierKind.all_kinds()) == 5


def test_every_classifier_has_a_unique_readable_display_name():
    display_names = [kind.display_name for kind in ClassifierKind.all_kinds()]

    assert len(set(display_names)) == 5
    assert "Logistic Regression" in display_names
    assert "Decision Tree" in display_names


def test_classifier_values_are_safe_file_names():
    for kind in ClassifierKind.all_kinds():
        assert kind.value == kind.value.lower()
        assert " " not in kind.value


def test_evaluation_scores_are_immutable():
    scores = EvaluationScores(0.9, 0.9, 0.9, 0.9, 0.9, 0.9)

    with pytest.raises(dataclasses.FrozenInstanceError):
        scores.accuracy = 0.1


def test_display_dict_uses_the_six_required_metric_names_in_order():
    scores = EvaluationScores(0.1, 0.2, 0.3, 0.4, 0.5, 0.6)

    assert list(scores.as_display_dict()) == [
        "Accuracy",
        "AUC",
        "Precision",
        "Recall",
        "F1",
        "MCC",
    ]
    assert scores.as_display_dict()["F1"] == 0.5


def test_display_dict_keeps_nan_auc_as_nan():
    scores = EvaluationScores(1.0, float("nan"), 1.0, 1.0, 1.0, 1.0)

    assert math.isnan(scores.as_display_dict()["AUC"])
