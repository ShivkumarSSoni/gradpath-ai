"""Builder for the Decision Tree classifier."""

from sklearn.base import BaseEstimator
from sklearn.tree import DecisionTreeClassifier

from gradpath.backend.domain.classifier_kind import ClassifierKind
from gradpath.backend.ml.classifier_builders.classifier_builder import (
    ClassifierBuilder,
)

# Limiting depth and leaf size stops the tree from memorising the training data.
MAXIMUM_TREE_DEPTH = 6
MINIMUM_RECORDS_PER_LEAF = 10


class DecisionTreeBuilder(ClassifierBuilder):
    def __init__(self, random_seed: int) -> None:
        self._random_seed = random_seed

    @property
    def kind(self) -> ClassifierKind:
        return ClassifierKind.DECISION_TREE

    def build_untrained(self) -> BaseEstimator:
        return DecisionTreeClassifier(
            max_depth=MAXIMUM_TREE_DEPTH,
            min_samples_leaf=MINIMUM_RECORDS_PER_LEAF,
            random_state=self._random_seed,
        )
