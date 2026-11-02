import numpy as np

from gradpath.backend.ml.metrics.curves.roc_curve_calculator import RocCurveCalculator


def test_a_perfect_separator_has_auc_one():
    calculator = RocCurveCalculator()
    is_positive = np.array([False, False, True, True])
    probabilities = np.array([0.1, 0.2, 0.8, 0.9])

    curve = calculator.calculate_for_outcome(is_positive, probabilities)

    assert curve.summary_score == 1.0
    assert curve.x_values[0] == 0.0
    assert curve.y_values[-1] == 1.0


def test_no_positive_records_is_undefined():
    calculator = RocCurveCalculator()
    is_positive = np.array([False, False, False])

    curve = calculator.calculate_for_outcome(is_positive, np.array([0.1, 0.2, 0.3]))

    assert curve is None


def test_all_positive_records_is_undefined():
    calculator = RocCurveCalculator()
    is_positive = np.array([True, True, True])

    curve = calculator.calculate_for_outcome(is_positive, np.array([0.7, 0.8, 0.9]))

    assert curve is None
