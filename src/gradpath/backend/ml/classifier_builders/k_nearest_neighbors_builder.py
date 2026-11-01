"""Builder for the K-Nearest Neighbors classifier."""

from sklearn.base import BaseEstimator
from sklearn.neighbors import KNeighborsClassifier

from gradpath.backend.domain.classifier_kind import ClassifierKind
from gradpath.backend.ml.classifier_builders.classifier_builder import (
    ClassifierBuilder,
)

# An odd count of neighbours avoids many tied votes.
NEIGHBOUR_COUNT = 7


class KNearestNeighborsBuilder(ClassifierBuilder):
    @property
    def kind(self) -> ClassifierKind:
        return ClassifierKind.K_NEAREST_NEIGHBORS

    def build_untrained(self) -> BaseEstimator:
        # Closer neighbours get a larger say in the vote.
        return KNeighborsClassifier(
            n_neighbors=NEIGHBOUR_COUNT,
            weights="distance",
        )
