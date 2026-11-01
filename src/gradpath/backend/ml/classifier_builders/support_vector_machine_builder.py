"""Builder for the Support Vector Machine classifier."""

from sklearn.base import BaseEstimator
from sklearn.calibration import CalibratedClassifierCV
from sklearn.svm import SVC

from gradpath.backend.domain.classifier_kind import ClassifierKind
from gradpath.backend.ml.classifier_builders.classifier_builder import (
    ClassifierBuilder,
)


class SupportVectorMachineBuilder(ClassifierBuilder):
    def __init__(self, random_seed: int) -> None:
        self._random_seed = random_seed

    @property
    def kind(self) -> ClassifierKind:
        return ClassifierKind.SUPPORT_VECTOR_MACHINE

    def build_untrained(self) -> BaseEstimator:
        # AUC needs class probabilities, which a plain SVM does not produce.
        # sklearn 1.9 deprecates SVC(probability=True), so the recommended
        # calibration wrapper is used. It learns a sigmoid that turns the SVM
        # distances into probabilities (Platt scaling).
        return CalibratedClassifierCV(
            SVC(kernel="rbf", random_state=self._random_seed),
            method="sigmoid",
            ensemble=False,
        )
