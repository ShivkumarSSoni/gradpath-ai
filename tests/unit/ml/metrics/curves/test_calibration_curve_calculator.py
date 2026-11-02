import numpy as np

from gradpath.backend.ml.metrics.curves.calibration_curve_calculator import (
    CalibrationCurveCalculator,
)


def test_perfectly_calibrated_probabilities_have_zero_error():
    calculator = CalibrationCurveCalculator(bin_count=2)
    is_positive = np.array([False, False, True, True] * 5)
    probabilities = np.array([0.0, 0.0, 1.0, 1.0] * 5)

    curve = calculator.calculate_for_outcome(is_positive, probabilities)

    assert curve.summary_score == 0.0


def test_no_positive_records_is_undefined():
    calculator = CalibrationCurveCalculator()
    is_positive = np.array([False, False, False])

    curve = calculator.calculate_for_outcome(is_positive, np.array([0.1, 0.2, 0.3]))

    assert curve is None


def test_all_positive_records_is_undefined():
    calculator = CalibrationCurveCalculator()
    is_positive = np.array([True, True, True])

    curve = calculator.calculate_for_outcome(is_positive, np.array([0.7, 0.8, 0.9]))

    assert curve is None
