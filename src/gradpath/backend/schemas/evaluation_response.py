"""What the evaluation controller hands back across the layer boundary."""

from dataclasses import dataclass, field

from gradpath.backend.domain.classifier_kind import ClassifierKind
from gradpath.backend.domain.evaluation_report import EvaluationReport


@dataclass(frozen=True)
class EvaluationResponse:
    """Tells the caller whether an evaluation succeeded, and with what detail.

    Exactly one of `report` and `error_message` is filled in, the same
    success-or-reason shape as `ClassifierTrainingResult`.

    Attributes:
        classifier_kind: Which classifier this response is about.
        report: The full evaluation detail, when scoring worked.
        error_message: A readable reason, when validation or scoring failed.
        warnings: Non-fatal notices about the upload, for example rows that
            were dropped during cleaning or an outcome missing from the
            file. Always empty when `error_message` is set.
    """

    classifier_kind: ClassifierKind
    report: EvaluationReport | None = None
    error_message: str | None = None
    warnings: list[str] = field(default_factory=list)

    @property
    def succeeded(self) -> bool:
        return self.report is not None
