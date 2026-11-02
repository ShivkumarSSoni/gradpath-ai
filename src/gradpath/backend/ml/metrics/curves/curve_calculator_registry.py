"""Runs every registered curve calculator for every outcome."""

import numpy as np

from gradpath.backend.ml.metrics.curves.curve_points import CurvePoints
from gradpath.backend.ml.metrics.curves.outcome_curve_calculator import (
    OutcomeCurveCalculator,
)


class CurveCalculatorRegistry:
    """Holds the curve calculators and applies each to every outcome.

    Adding a sixth curve means adding one new `OutcomeCurveCalculator` and
    listing it here; nothing else in the evaluation pipeline changes.
    """

    def __init__(self, calculators: list[OutcomeCurveCalculator]) -> None:
        self._calculators = list(calculators)

    def calculate_all(
        self,
        class_labels: list[str],
        true_labels,
        predicted_probabilities,
    ) -> dict[str, dict[str, CurvePoints]]:
        """Return every curve, for every outcome that has one.

        Returns:
            A dictionary from curve name to a dictionary from outcome label
            to that outcome's curve. An outcome is left out of a curve's
            dictionary when the curve is undefined for it, for example a
            missing outcome.
        """
        true_array = np.asarray(true_labels)
        probability_matrix = np.asarray(predicted_probabilities, dtype=float)

        curves_by_name: dict[str, dict[str, CurvePoints]] = {}
        for calculator in self._calculators:
            curves_by_outcome: dict[str, CurvePoints] = {}
            for column_index, class_label in enumerate(class_labels):
                is_positive_outcome = true_array == class_label
                curve = calculator.calculate_for_outcome(
                    is_positive_outcome, probability_matrix[:, column_index]
                )
                if curve is not None:
                    curves_by_outcome[class_label] = curve
            curves_by_name[calculator.curve_name] = curves_by_outcome
        return curves_by_name
