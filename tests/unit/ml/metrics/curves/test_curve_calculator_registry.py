import numpy as np

from gradpath.backend.ml.metrics.curves.curve_calculator_registry import (
    CurveCalculatorRegistry,
)
from gradpath.backend.ml.metrics.curves.roc_curve_calculator import RocCurveCalculator

LABELS = ["Dropout", "Enrolled", "Graduate"]
D, E, G = LABELS


def test_each_calculator_runs_for_every_outcome_that_has_both_classes():
    registry = CurveCalculatorRegistry([RocCurveCalculator()])
    true_labels = [D, D, E, G, G]
    probabilities = np.array(
        [
            [0.8, 0.1, 0.1],
            [0.7, 0.2, 0.1],
            [0.1, 0.8, 0.1],
            [0.1, 0.1, 0.8],
            [0.2, 0.1, 0.7],
        ]
    )

    curves_by_name = registry.calculate_all(LABELS, true_labels, probabilities)

    assert set(curves_by_name) == {"ROC"}
    assert set(curves_by_name["ROC"]) == {D, E, G}


def test_an_outcome_missing_from_the_records_is_left_out_of_the_curve():
    registry = CurveCalculatorRegistry([RocCurveCalculator()])
    true_labels = [D, D, G, G]  # no Enrolled records at all
    probabilities = np.array(
        [
            [0.8, 0.1, 0.1],
            [0.7, 0.1, 0.2],
            [0.1, 0.1, 0.8],
            [0.2, 0.1, 0.7],
        ]
    )

    curves_by_name = registry.calculate_all(LABELS, true_labels, probabilities)

    assert E not in curves_by_name["ROC"]
