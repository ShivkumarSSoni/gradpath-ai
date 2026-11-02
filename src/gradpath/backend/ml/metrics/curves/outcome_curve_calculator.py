"""The contract every one-vs-rest curve calculator follows."""

from abc import ABC, abstractmethod

import numpy as np

from gradpath.backend.ml.metrics.curves.curve_points import CurvePoints


class OutcomeCurveCalculator(ABC):
    """Computes one kind of curve for one outcome treated as positive.

    A new curve (for example a lift curve) means one new subclass, never an
    edit to an existing calculator or to the registry that holds them.
    """

    @property
    @abstractmethod
    def curve_name(self) -> str:
        """Return the readable name shown as this curve's chart title."""

    @abstractmethod
    def calculate_for_outcome(
        self,
        is_positive_outcome: np.ndarray,
        outcome_probabilities: np.ndarray,
    ) -> CurvePoints | None:
        """Return this curve for one outcome against all the other outcomes.

        Args:
            is_positive_outcome: One boolean per record, True where the
                record's true outcome is the one being curved.
            outcome_probabilities: One probability per record, that the
                classifier assigned to the outcome being curved.

        Returns:
            The curve's points, or None when it is undefined, for example
            because the records contain none of this outcome, or only this
            outcome.
        """
