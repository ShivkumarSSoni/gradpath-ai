"""The five classifiers that are trained and compared."""

from enum import Enum


class ClassifierKind(str, Enum):
    """Identifies one classifier. The value is also its saved file name."""

    LOGISTIC_REGRESSION = "logistic_regression"
    DECISION_TREE = "decision_tree"
    K_NEAREST_NEIGHBORS = "k_nearest_neighbors"
    GAUSSIAN_NAIVE_BAYES = "gaussian_naive_bayes"
    SUPPORT_VECTOR_MACHINE = "support_vector_machine"

    @property
    def display_name(self) -> str:
        """Return the readable name shown to the user, for example in a dropdown."""
        return _DISPLAY_NAMES[self]

    @classmethod
    def all_kinds(cls) -> list["ClassifierKind"]:
        """Return the five classifiers in a fixed, readable order."""
        return list(cls)


_DISPLAY_NAMES = {
    ClassifierKind.LOGISTIC_REGRESSION: "Logistic Regression",
    ClassifierKind.DECISION_TREE: "Decision Tree",
    ClassifierKind.K_NEAREST_NEIGHBORS: "K-Nearest Neighbors (kNN)",
    ClassifierKind.GAUSSIAN_NAIVE_BAYES: "Gaussian Naive Bayes",
    ClassifierKind.SUPPORT_VECTOR_MACHINE: "Support Vector Machine (SVM)",
}
