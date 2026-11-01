"""Unit tests for the six evaluation scores, checked against hand calculations."""

import math
import warnings

import numpy as np
import pytest

from gradpath.backend.exceptions.invalid_scoring_input_error import (
    InvalidScoringInputError,
)
from gradpath.backend.ml.metrics.scores_calculator import ScoresCalculator

LABELS = ["Dropout", "Enrolled", "Graduate"]
D, E, G = LABELS


def make_calculator() -> ScoresCalculator:
    return ScoresCalculator(class_labels=LABELS)


def one_hot_probabilities(labels):
    """Probabilities that put all the weight on the given label."""
    return np.array([[1.0 if label == name else 0.0 for name in LABELS] for label in labels])


def test_hand_calculated_scores_for_a_small_example():
    # Per outcome: Dropout precision 1/2 recall 1/2, Enrolled precision 2/3
    # recall 1, Graduate precision 1 recall 1/2. Macro values follow from it.
    true_labels = [D, D, E, E, G, G]
    predicted_labels = [D, E, E, E, G, D]

    scores = make_calculator().calculate(
        true_labels, predicted_labels, one_hot_probabilities(predicted_labels)
    )

    assert scores.accuracy == pytest.approx(4 / 6)
    assert scores.precision == pytest.approx(13 / 18)
    assert scores.recall == pytest.approx(2 / 3)
    assert scores.f1 == pytest.approx(59 / 90)
    assert scores.mcc == pytest.approx(12 / math.sqrt(22 * 24))


def test_perfect_predictions_score_one_everywhere():
    true_labels = [D, E, G, D, E, G]

    scores = make_calculator().calculate(
        true_labels, true_labels, one_hot_probabilities(true_labels)
    )

    assert scores.as_display_dict() == {
        "Accuracy": 1.0,
        "AUC": 1.0,
        "Precision": 1.0,
        "Recall": 1.0,
        "F1": 1.0,
        "MCC": 1.0,
    }


def test_completely_wrong_predictions_score_zero_and_negative_mcc():
    true_labels = [D, E, G, D, E, G]
    predicted_labels = [E, G, D, E, G, D]
    # The true outcome gets probability 0, the two others get 0.5 each.
    probabilities = np.array(
        [[0.0 if name == label else 0.5 for name in LABELS] for label in true_labels]
    )

    scores = make_calculator().calculate(true_labels, predicted_labels, probabilities)

    assert scores.accuracy == 0.0
    assert scores.auc == 0.0
    assert scores.precision == 0.0
    assert scores.recall == 0.0
    assert scores.f1 == 0.0
    assert scores.mcc < 0


def test_never_predicting_an_outcome_gives_zero_not_a_division_error():
    true_labels = [D, D, E, E, G, G]
    predicted_labels = [D, D, D, G, G, G]  # Enrolled is never predicted

    with warnings.catch_warnings():
        warnings.simplefilter("error")  # any division-by-zero warning fails the test
        scores = make_calculator().calculate(
            true_labels, predicted_labels, one_hot_probabilities(predicted_labels)
        )

    # Enrolled precision is 0/0 and counts as 0: (2/3 + 0 + 2/3) / 3
    assert scores.precision == pytest.approx((2 / 3 + 0 + 2 / 3) / 3)


def test_an_outcome_missing_from_the_true_labels_is_handled():
    true_labels = [D, D, G, G]  # no Enrolled records at all
    predicted_labels = [D, D, G, G]

    scores = make_calculator().calculate(
        true_labels, predicted_labels, one_hot_probabilities(predicted_labels)
    )

    assert scores.auc == 1.0  # Enrolled is skipped, the others are perfect
    assert scores.recall == pytest.approx(2 / 3)  # Enrolled recall 0/0 counts as 0


def test_auc_is_nan_when_only_one_outcome_is_present():
    true_labels = [G, G, G]

    scores = make_calculator().calculate(
        true_labels, true_labels, one_hot_probabilities(true_labels)
    )

    assert math.isnan(scores.auc)
    assert scores.accuracy == 1.0


def test_mcc_is_zero_when_every_prediction_is_the_same_outcome():
    true_labels = [D, E, G, D, E, G]
    predicted_labels = [G] * 6

    scores = make_calculator().calculate(
        true_labels, predicted_labels, one_hot_probabilities(predicted_labels)
    )

    assert scores.mcc == 0.0


def test_accepts_pandas_series_and_numpy_arrays(tiny_student_records):
    outcome = tiny_student_records["Target"]
    probabilities = one_hot_probabilities(outcome.tolist())

    scores = make_calculator().calculate(outcome, outcome.to_numpy(), probabilities)

    assert scores.accuracy == 1.0


def test_no_records_raises_a_clear_error():
    with pytest.raises(InvalidScoringInputError, match="no records"):
        make_calculator().calculate([], [], np.empty((0, 3)))


def test_different_lengths_raise_a_clear_error():
    with pytest.raises(InvalidScoringInputError, match="same number"):
        make_calculator().calculate([D, E], [D], np.zeros((2, 3)))


def test_wrong_number_of_probability_columns_raises_a_clear_error():
    with pytest.raises(InvalidScoringInputError, match="3 columns"):
        make_calculator().calculate([D, E], [D, E], np.zeros((2, 2)))
