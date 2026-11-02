"""Contract for saving and loading trained models."""

from abc import ABC, abstractmethod

from sklearn.base import BaseEstimator

from gradpath.backend.domain.classifier_kind import ClassifierKind


class ModelRepository(ABC):
    """Where trained models are kept between training and use.

    A saved model is a complete pipeline: preprocessing plus classifier.
    That way raw records can be given to it directly, and the preprocessing
    learned during training is always applied the same way.
    """

    @abstractmethod
    def save_model(self, classifier_kind: ClassifierKind, trained_model: BaseEstimator) -> None:
        """Store the trained model, replacing any earlier one of the same kind."""

    @abstractmethod
    def load_model(self, classifier_kind: ClassifierKind) -> BaseEstimator:
        """Return the saved model.

        Raises:
            ModelNotFoundError: If that classifier has not been saved yet.
        """

    @abstractmethod
    def model_exists(self, classifier_kind: ClassifierKind) -> bool:
        """Tell whether a saved model is available for the classifier."""
