"""One row of the model comparison table shown in the README and the app."""

from dataclasses import dataclass

from gradpath.backend.domain.classifier_kind import ClassifierKind
from gradpath.backend.domain.evaluation_scores import EvaluationScores


@dataclass(frozen=True)
class ComparisonRow:
    """Pairs one classifier with its official held-out scores.

    Attributes:
        classifier_kind: Which classifier this row describes.
        scores: The six official scores, or None if that classifier has not
            been built yet (so the comparison page can show "not available"
            instead of crashing).
    """

    classifier_kind: ClassifierKind
    scores: EvaluationScores | None

    @property
    def display_name(self) -> str:
        """Return the classifier's readable name, for use in a table."""
        return self.classifier_kind.display_name

    @property
    def is_available(self) -> bool:
        """Tell whether this row has scores to show."""
        return self.scores is not None
