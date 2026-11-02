"""One-vs-rest Precision-Recall curve, one per outcome.

This curve matters most for the rare Enrolled outcome: ROC can look good
even when a classifier struggles with a small class, but Precision-Recall
does not hide that.
"""

import numpy as np
from sklearn.metrics import average_precision_score, precision_recall_curve

from gradpath.backend.ml.metrics.curves.curve_points import CurvePoints
from gradpath.backend.ml.metrics.curves.outcome_curve_calculator import (
    OutcomeCurveCalculator,
)


class PrecisionRecallCurveCalculator(OutcomeCurveCalculator):
    """Plots precision against recall as the decision threshold changes."""

    @property
    def curve_name(self) -> str:
        return "Precision-Recall"

    def calculate_for_outcome(
        self,
        is_positive_outcome: np.ndarray,
        outcome_probabilities: np.ndarray,
    ) -> CurvePoints | None:
        # Precision is undefined with zero positive records (0/0).
        if int(is_positive_outcome.sum()) == 0:
            return None

        precision, recall, _ = precision_recall_curve(is_positive_outcome, outcome_probabilities)
        return CurvePoints(
            x_values=recall.tolist(),
            y_values=precision.tolist(),
            summary_score=float(
                average_precision_score(is_positive_outcome, outcome_probabilities)
            ),
        )
