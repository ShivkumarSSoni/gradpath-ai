"""Builder for the Decision Tree classifier."""

from sklearn.base import BaseEstimator
from sklearn.tree import DecisionTreeClassifier

from gradpath.backend.domain.classifier_kind import ClassifierKind
from gradpath.backend.ml.classifier_builders.classifier_builder import (
    ClassifierBuilder,
)
from gradpath.backend.ml.classifier_builders.cross_validated_tuning import (
    tune_by_cross_validation,
)

# A shallow tree and large leaves stop the tree from memorising the training data.
MAXIMUM_TREE_DEPTH_CANDIDATES = [3, 5, 7, 10]
MINIMUM_RECORDS_PER_LEAF_CANDIDATES = [5, 10, 20]


class DecisionTreeBuilder(ClassifierBuilder):
    def __init__(self, random_seed: int) -> None:
        self._random_seed = random_seed

    @property
    def kind(self) -> ClassifierKind:
        return ClassifierKind.DECISION_TREE

    def build_untrained(self) -> BaseEstimator:
        return tune_by_cross_validation(
            DecisionTreeClassifier(random_state=self._random_seed),
            {
                "max_depth": MAXIMUM_TREE_DEPTH_CANDIDATES,
                "min_samples_leaf": MINIMUM_RECORDS_PER_LEAF_CANDIDATES,
            },
            self._random_seed,
        )
