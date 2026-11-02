"""The outcome of training one classifier."""

from dataclasses import dataclass

from gradpath.backend.domain.classifier_kind import ClassifierKind
from gradpath.backend.domain.evaluation_scores import EvaluationScores


@dataclass(frozen=True)
class ClassifierTrainingResult:
    """Tells the caller whether one classifier trained, and how well.

    Exactly one of `scores` and `error_message` is filled in.

    Attributes:
        classifier_kind: Which classifier this result belongs to.
        scores: The six scores on the held-out records, when training worked.
        error_message: A readable reason, when training failed.
    """

    classifier_kind: ClassifierKind
    scores: EvaluationScores | None = None
    error_message: str | None = None

    @property
    def succeeded(self) -> bool:
        return self.scores is not None
