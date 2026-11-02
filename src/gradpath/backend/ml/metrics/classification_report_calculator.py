"""Per-outcome precision, recall, F1 and support, for a detailed table.

The six official scores (see `ScoresCalculator`) already give the macro
averages; this calculator gives the per-outcome breakdown that explains
them, for example "the Enrolled recall is low even though macro recall
looks fine".
"""

from sklearn.metrics import classification_report


class ClassificationReportCalculator:
    """Builds one row of precision, recall, F1 and support per outcome."""

    def __init__(self, class_labels: list[str]) -> None:
        self._class_labels = list(class_labels)

    def calculate(self, true_labels, predicted_labels) -> dict[str, dict[str, float]]:
        """Return a dictionary keyed by outcome, each with its four numbers.

        zero_division=0 covers an outcome the classifier never predicts, the
        same guard used for the six official scores.
        """
        report = classification_report(
            true_labels,
            predicted_labels,
            labels=self._class_labels,
            output_dict=True,
            zero_division=0,
        )
        return {
            label: {
                "precision": float(report[label]["precision"]),
                "recall": float(report[label]["recall"]),
                "f1_score": float(report[label]["f1-score"]),
                "support": float(report[label]["support"]),
            }
            for label in self._class_labels
        }
