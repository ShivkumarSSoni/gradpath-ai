"""Confusion matrix in a fixed, readable outcome order."""

from sklearn.metrics import confusion_matrix


class ConfusionMatrixCalculator:
    """Counts true outcome against predicted outcome, one cell per pair."""

    def __init__(self, class_labels: list[str]) -> None:
        """Remember the outcome order rows and columns must follow."""
        self._class_labels = list(class_labels)

    def calculate(self, true_labels, predicted_labels) -> list[list[int]]:
        """Return the matrix as plain nested lists, row = true, column = predicted.

        Using `labels=` instead of letting scikit-learn infer them keeps the
        row and column order identical every time, even if one outcome is
        completely absent from a small uploaded file.
        """
        matrix = confusion_matrix(true_labels, predicted_labels, labels=self._class_labels)
        return matrix.tolist()
