"""Contract for keeping the scores of every trained classifier."""

from abc import ABC, abstractmethod

from gradpath.backend.domain.classifier_kind import ClassifierKind
from gradpath.backend.domain.evaluation_scores import EvaluationScores


class EvaluationResultsRepository(ABC):
    """Where the scores of the trained classifiers are recorded.

    This is the single source of truth for the comparison table, so the
    notebook, the README and the app can never disagree.
    """

    @abstractmethod
    def save_scores(self, classifier_kind: ClassifierKind, scores: EvaluationScores) -> None:
        """Record the scores of one classifier, replacing its earlier scores.

        The scores of the other classifiers stay untouched.
        """

    @abstractmethod
    def load_all_scores(self) -> dict[ClassifierKind, EvaluationScores]:
        """Return the recorded scores, in the standard classifier order.

        Returns an empty dictionary when nothing has been recorded yet.
        """
