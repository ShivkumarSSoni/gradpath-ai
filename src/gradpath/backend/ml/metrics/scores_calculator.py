"""Computes the six evaluation scores used to compare the classifiers."""

import warnings

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)

from gradpath.backend.domain.evaluation_scores import EvaluationScores
from gradpath.backend.exceptions.invalid_scoring_input_error import (
    InvalidScoringInputError,
)


class ScoresCalculator:
    """Scores predictions with Accuracy, AUC, Precision, Recall, F1 and MCC."""

    def __init__(self, class_labels: list[str]) -> None:
        """Remember the outcome labels, in the column order of the probabilities.

        Args:
            class_labels: Outcome names in the same order as the columns of the
                predicted probabilities (the order of a fitted classifier's
                classes_ attribute).
        """
        self._class_labels = list(class_labels)

    def calculate(
        self,
        true_labels,
        predicted_labels,
        predicted_probabilities,
    ) -> EvaluationScores:
        """Return the six scores for one classifier.

        Precision, Recall and F1 are macro averages: each outcome counts
        equally, so the rare Enrolled outcome is not hidden by the large
        Graduate outcome.

        Args:
            true_labels: The real outcome of every record.
            predicted_labels: The outcome the classifier predicted for every record.
            predicted_probabilities: One row per record, one column per outcome,
                in the order of the class labels given to the constructor.

        Raises:
            InvalidScoringInputError: If there are no records, the inputs have
                different lengths, or the probability columns do not match the
                number of outcomes.
        """
        true_array = np.asarray(true_labels)
        predicted_array = np.asarray(predicted_labels)
        probability_matrix = np.asarray(predicted_probabilities, dtype=float)
        self._check_inputs(true_array, predicted_array, probability_matrix)

        # zero_division=0 covers the division by zero that happens when a
        # classifier never predicts an outcome (precision = 0 / 0) or when an
        # outcome never occurs in the true labels (recall = 0 / 0). The score
        # for that outcome is then defined as 0 instead of raising an error.
        return EvaluationScores(
            accuracy=float(accuracy_score(true_array, predicted_array)),
            auc=self._average_one_vs_rest_auc(true_array, probability_matrix),
            precision=float(
                precision_score(
                    true_array,
                    predicted_array,
                    labels=self._class_labels,
                    average="macro",
                    zero_division=0,
                )
            ),
            recall=float(
                recall_score(
                    true_array,
                    predicted_array,
                    labels=self._class_labels,
                    average="macro",
                    zero_division=0,
                )
            ),
            f1=float(
                f1_score(
                    true_array,
                    predicted_array,
                    labels=self._class_labels,
                    average="macro",
                    zero_division=0,
                )
            ),
            # When all predictions fall in one outcome the formula divides by
            # zero; scikit-learn returns 0 for this case, meaning "no better
            # than guessing".
            mcc=self._matthews_correlation(true_array, predicted_array),
        )

    @staticmethod
    def _matthews_correlation(
        true_array: np.ndarray, predicted_array: np.ndarray
    ) -> float:
        """Return MCC, ignoring scikit-learn's notice about a single outcome.

        When the records contain only one outcome the score is still defined
        (it is 0), so the notice is only noise for the user.
        """
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=UserWarning)
            return float(matthews_corrcoef(true_array, predicted_array))

    def _check_inputs(
        self,
        true_array: np.ndarray,
        predicted_array: np.ndarray,
        probability_matrix: np.ndarray,
    ) -> None:
        """Reject inputs that cannot be scored, with a clear message."""
        if len(true_array) == 0:
            # Scoring zero records would divide by zero (0 correct out of 0).
            raise InvalidScoringInputError("There are no records to score.")
        if not (len(true_array) == len(predicted_array) == len(probability_matrix)):
            raise InvalidScoringInputError(
                "True labels, predicted labels and probabilities must have "
                "the same number of records."
            )
        if probability_matrix.ndim != 2 or probability_matrix.shape[1] != len(
            self._class_labels
        ):
            raise InvalidScoringInputError(
                f"Probabilities must have {len(self._class_labels)} columns, "
                "one for each outcome."
            )

    def _average_one_vs_rest_auc(
        self, true_array: np.ndarray, probability_matrix: np.ndarray
    ) -> float:
        """Average the AUC of each outcome against all the other outcomes.

        The AUC of an outcome compares records that have it with records that
        do not, so it needs both kinds. An outcome missing from the records,
        or the only outcome present, has no AUC and is skipped. If no outcome
        qualifies the result is NaN (not a number).
        """
        auc_per_outcome = []
        for column_index, class_label in enumerate(self._class_labels):
            has_this_outcome = true_array == class_label
            records_with_outcome = int(has_this_outcome.sum())
            if records_with_outcome == 0 or records_with_outcome == len(true_array):
                continue
            auc_per_outcome.append(
                roc_auc_score(has_this_outcome, probability_matrix[:, column_index])
            )

        if not auc_per_outcome:
            return float("nan")
        return float(np.mean(auc_per_outcome))
