"""Builder for the Gaussian Naive Bayes classifier."""

from sklearn.base import BaseEstimator
from sklearn.naive_bayes import GaussianNB as GaussianNaiveBayesClassifier

from gradpath.backend.domain.classifier_kind import ClassifierKind
from gradpath.backend.ml.classifier_builders.classifier_builder import (
    ClassifierBuilder,
)


class GaussianNaiveBayesBuilder(ClassifierBuilder):
    @property
    def kind(self) -> ClassifierKind:
        return ClassifierKind.GAUSSIAN_NAIVE_BAYES

    def build_untrained(self) -> BaseEstimator:
        # Defaults are kept on purpose: this model is the simple baseline.
        return GaussianNaiveBayesClassifier()
