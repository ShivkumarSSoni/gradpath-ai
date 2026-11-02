"""The full detail behind one classifier's scores on one set of records."""

from dataclasses import dataclass

from gradpath.backend.domain.classifier_kind import ClassifierKind
from gradpath.backend.domain.evaluation_scores import EvaluationScores
from gradpath.backend.ml.metrics.curves.curve_points import CurvePoints


@dataclass(frozen=True)
class EvaluationReport:
    """Everything the Streamlit app shows after scoring an uploaded file.

    Attributes:
        classifier_kind: Which classifier produced this report.
        scores: The six official metrics, macro-averaged where that applies.
        confusion_matrix: Row = true outcome, column = predicted outcome, in
            the fixed outcome order used throughout the project.
        classification_report: Per-outcome precision, recall, F1 and support.
        curves_by_name: Per-curve, per-outcome plot points (for example
            curves_by_name["ROC"]["Enrolled"]).
    """

    classifier_kind: ClassifierKind
    scores: EvaluationScores
    confusion_matrix: list[list[int]]
    classification_report: dict[str, dict[str, float]]
    curves_by_name: dict[str, dict[str, CurvePoints]]
