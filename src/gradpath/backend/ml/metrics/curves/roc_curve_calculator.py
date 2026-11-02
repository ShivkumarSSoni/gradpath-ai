"""One-vs-rest ROC curve, one per outcome."""

import numpy as np
from sklearn.metrics import auc, roc_curve

from gradpath.backend.ml.metrics.curves.curve_points import CurvePoints
from gradpath.backend.ml.metrics.curves.outcome_curve_calculator import (
    OutcomeCurveCalculator,
)


class RocCurveCalculator(OutcomeCurveCalculator):
    """Plots the true positive rate against the false positive rate."""

    @property
    def curve_name(self) -> str:
        return "ROC"

    def calculate_for_outcome(
        self,
        is_positive_outcome: np.ndarray,
        outcome_probabilities: np.ndarray,
    ) -> CurvePoints | None:
        positive_count = int(is_positive_outcome.sum())
        # An ROC curve needs at least one positive and one negative record;
        # with only one of them it is undefined, not just noisy.
        if positive_count == 0 or positive_count == len(is_positive_outcome):
            return None

        false_positive_rate, true_positive_rate, _ = roc_curve(
            is_positive_outcome, outcome_probabilities
        )
        return CurvePoints(
            x_values=false_positive_rate.tolist(),
            y_values=true_positive_rate.tolist(),
            summary_score=float(auc(false_positive_rate, true_positive_rate)),
        )
