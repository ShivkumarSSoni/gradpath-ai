"""One-vs-rest calibration (reliability) curve, one per outcome.

Checks whether a predicted probability of, say, 0.7 actually comes true
about 70% of the time. This matters most for the Support Vector Machine,
whose probabilities are only estimated through calibration (see
`SupportVectorMachineBuilder`), not produced directly by the classifier.
"""

import numpy as np
from sklearn.calibration import calibration_curve

from gradpath.backend.ml.metrics.curves.curve_points import CurvePoints
from gradpath.backend.ml.metrics.curves.outcome_curve_calculator import (
    OutcomeCurveCalculator,
)

DEFAULT_BIN_COUNT = 10


class CalibrationCurveCalculator(OutcomeCurveCalculator):
    """Plots the observed frequency of an outcome against its predicted probability."""

    def __init__(self, bin_count: int = DEFAULT_BIN_COUNT) -> None:
        self._bin_count = bin_count

    @property
    def curve_name(self) -> str:
        return "Calibration"

    def calculate_for_outcome(
        self,
        is_positive_outcome: np.ndarray,
        outcome_probabilities: np.ndarray,
    ) -> CurvePoints | None:
        positive_count = int(is_positive_outcome.sum())
        # With only one class present there is nothing to calibrate against.
        if positive_count == 0 or positive_count == len(is_positive_outcome):
            return None

        observed_frequency, predicted_probability = calibration_curve(
            is_positive_outcome, outcome_probabilities, n_bins=self._bin_count
        )
        if len(predicted_probability) == 0:
            return None

        # The mean absolute gap between what was predicted and what was
        # observed, in each bin: 0 is perfectly calibrated.
        calibration_error = float(np.mean(np.abs(observed_frequency - predicted_probability)))
        return CurvePoints(
            x_values=predicted_probability.tolist(),
            y_values=observed_frequency.tolist(),
            summary_score=calibration_error,
        )
