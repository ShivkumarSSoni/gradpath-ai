"""The contract every classifier builder follows."""

from abc import ABC, abstractmethod

from sklearn.base import BaseEstimator

from gradpath.backend.domain.classifier_kind import ClassifierKind


class ClassifierBuilder(ABC):
    """Creates one kind of classifier, configured but not yet trained.

    A builder never fits anything. Training is the job of the training
    service, which keeps "how a model is configured" separate from
    "how a model is trained".
    """

    @property
    @abstractmethod
    def kind(self) -> ClassifierKind:
        """Return which classifier this builder creates."""

    @abstractmethod
    def build_untrained(self) -> BaseEstimator:
        """Return a brand new, untrained classifier on every call."""
