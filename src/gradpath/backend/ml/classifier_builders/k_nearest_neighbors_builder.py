"""Builder for the K-Nearest Neighbors classifier."""

from sklearn.base import BaseEstimator
from sklearn.neighbors import KNeighborsClassifier

from gradpath.backend.domain.classifier_kind import ClassifierKind
from gradpath.backend.ml.classifier_builders.classifier_builder import (
    ClassifierBuilder,
)
from gradpath.backend.ml.classifier_builders.cross_validated_tuning import (
    tune_by_cross_validation,
)

# Odd neighbour counts avoid many tied votes. "distance" weighting lets closer
# neighbours have a larger say than farther ones.
NEIGHBOUR_COUNT_CANDIDATES = [3, 5, 7, 11, 15]
NEIGHBOUR_VOTE_WEIGHTING_CANDIDATES = ["uniform", "distance"]


class KNearestNeighborsBuilder(ClassifierBuilder):
    def __init__(self, random_seed: int) -> None:
        self._random_seed = random_seed

    @property
    def kind(self) -> ClassifierKind:
        return ClassifierKind.K_NEAREST_NEIGHBORS

    def build_untrained(self) -> BaseEstimator:
        return tune_by_cross_validation(
            KNeighborsClassifier(),
            {
                "n_neighbors": NEIGHBOUR_COUNT_CANDIDATES,
                "weights": NEIGHBOUR_VOTE_WEIGHTING_CANDIDATES,
            },
            self._random_seed,
        )
