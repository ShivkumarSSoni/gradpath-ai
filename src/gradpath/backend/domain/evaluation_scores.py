"""The six evaluation scores of one classifier on one set of records."""

from dataclasses import dataclass


@dataclass(frozen=True)
class EvaluationScores:
    """Immutable result of scoring a classifier.

    All scores are between 0 and 1, except Matthews Correlation Coefficient
    which is between -1 and 1. The AUC is NaN when it cannot be computed,
    for example when the records contain only one outcome.

    Attributes:
        accuracy: Share of records predicted correctly.
        auc: Area under the ROC curve, one-vs-rest, averaged over outcomes.
        precision: Macro-averaged precision.
        recall: Macro-averaged recall.
        f1: Macro-averaged F1 score.
        mcc: Matthews Correlation Coefficient for multiple classes.
    """

    accuracy: float
    auc: float
    precision: float
    recall: float
    f1: float
    mcc: float

    def as_display_dict(self) -> dict[str, float]:
        """Return the scores keyed by the metric names used on screen and in tables."""
        return {
            "Accuracy": self.accuracy,
            "AUC": self.auc,
            "Precision": self.precision,
            "Recall": self.recall,
            "F1": self.f1,
            "MCC": self.mcc,
        }
