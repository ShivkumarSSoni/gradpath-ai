"""Builder for the Gaussian Naive Bayes classifier."""

from sklearn.base import BaseEstimator
from sklearn.naive_bayes import GaussianNB as GaussianNaiveBayesClassifier

from gradpath.backend.domain.classifier_kind import ClassifierKind
from gradpath.backend.ml.classifier_builders.classifier_builder import (
    ClassifierBuilder,
)
from gradpath.backend.ml.classifier_builders.cross_validated_tuning import (
    tune_by_cross_validation,
)

# var_smoothing is the share of the largest feature variance that is added to
# every variance. The default (1e-9) is far too small for one-hot columns: a
# rare category has almost zero variance, so one matching record gives an
# extreme likelihood and the model collapses to one class.
# A plain log-spaced range: 1e-9, 1e-8, ..., 1, 10.
VARIANCE_SMOOTHING_CANDIDATES = [10.0**exponent for exponent in range(-9, 2)]


class GaussianNaiveBayesBuilder(ClassifierBuilder):
    def __init__(self, random_seed: int) -> None:
        self._random_seed = random_seed

    @property
    def kind(self) -> ClassifierKind:
        return ClassifierKind.GAUSSIAN_NAIVE_BAYES

    def build_untrained(self) -> BaseEstimator:
        return tune_by_cross_validation(
            GaussianNaiveBayesClassifier(),
            {"var_smoothing": VARIANCE_SMOOTHING_CANDIDATES},
            self._random_seed,
        )
