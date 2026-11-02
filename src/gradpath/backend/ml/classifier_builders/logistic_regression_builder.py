"""Builder for the Logistic Regression classifier."""

from sklearn.base import BaseEstimator
from sklearn.linear_model import LogisticRegression

from gradpath.backend.domain.classifier_kind import ClassifierKind
from gradpath.backend.ml.classifier_builders.classifier_builder import (
    ClassifierBuilder,
)
from gradpath.backend.ml.classifier_builders.cross_validated_tuning import (
    tune_by_cross_validation,
)

# lbfgs may need more rounds than the default 100 to settle on 236 columns.
MAXIMUM_SOLVER_ITERATIONS = 2000

# C is the inverse of the regularisation strength: a small C keeps the
# coefficients small and guards against overfitting.
REGULARISATION_C_CANDIDATES = [0.01, 0.1, 1.0, 10.0]


class LogisticRegressionBuilder(ClassifierBuilder):
    def __init__(self, random_seed: int) -> None:
        self._random_seed = random_seed

    @property
    def kind(self) -> ClassifierKind:
        return ClassifierKind.LOGISTIC_REGRESSION

    def build_untrained(self) -> BaseEstimator:
        # With three outcomes, lbfgs fits one multinomial (softmax) model.
        classifier = LogisticRegression(
            max_iter=MAXIMUM_SOLVER_ITERATIONS,
            random_state=self._random_seed,
        )
        return tune_by_cross_validation(
            classifier, {"C": REGULARISATION_C_CANDIDATES}, self._random_seed
        )
