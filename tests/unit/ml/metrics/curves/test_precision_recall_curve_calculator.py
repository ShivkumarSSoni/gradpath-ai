import numpy as np

from gradpath.backend.ml.metrics.curves.precision_recall_curve_calculator import (
    PrecisionRecallCurveCalculator,
)


def test_a_perfect_separator_has_average_precision_one():
    calculator = PrecisionRecallCurveCalculator()
    is_positive = np.array([False, False, True, True])
    probabilities = np.array([0.1, 0.2, 0.8, 0.9])

    curve = calculator.calculate_for_outcome(is_positive, probabilities)

    assert curve.summary_score == 1.0


def test_no_positive_records_is_undefined():
    calculator = PrecisionRecallCurveCalculator()
    is_positive = np.array([False, False, False])

    curve = calculator.calculate_for_outcome(is_positive, np.array([0.1, 0.2, 0.3]))

    assert curve is None
