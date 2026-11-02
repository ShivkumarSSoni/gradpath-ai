"""Builds the full evaluation detail shown by the Streamlit app.

This only reads a saved model and the records it is given; it never
retrains and never writes to the official results file (see
`ResultsRecorderService`, the only writer of that file).
"""

from pandas import DataFrame

from gradpath.backend.domain.classifier_kind import ClassifierKind
from gradpath.backend.domain.evaluation_report import EvaluationReport
from gradpath.backend.domain.feature_catalog import FeatureCatalog
from gradpath.backend.exceptions.classifier_training_error import (
    ClassifierTrainingError,
)
from gradpath.backend.ml.metrics.classification_report_calculator import (
    ClassificationReportCalculator,
)
from gradpath.backend.ml.metrics.confusion_matrix_calculator import (
    ConfusionMatrixCalculator,
)
from gradpath.backend.ml.metrics.curves.curve_calculator_registry import (
    CurveCalculatorRegistry,
)
from gradpath.backend.ml.metrics.scores_calculator import ScoresCalculator
from gradpath.backend.repositories.interfaces.model_repository import ModelRepository


class EvaluationReportService:
    """Scores a saved model once and builds every detail the app can show."""

    def __init__(
        self,
        model_repository: ModelRepository,
        scores_calculator: ScoresCalculator,
        confusion_matrix_calculator: ConfusionMatrixCalculator,
        classification_report_calculator: ClassificationReportCalculator,
        curve_calculator_registry: CurveCalculatorRegistry,
        feature_catalog: FeatureCatalog,
        expected_class_labels: list[str],
    ) -> None:
        self._model_repository = model_repository
        self._scores_calculator = scores_calculator
        self._confusion_matrix_calculator = confusion_matrix_calculator
        self._classification_report_calculator = classification_report_calculator
        self._curve_calculator_registry = curve_calculator_registry
        self._feature_catalog = feature_catalog
        self._expected_class_labels = list(expected_class_labels)

    def build_report(
        self,
        classifier_kind: ClassifierKind,
        records: DataFrame,
    ) -> EvaluationReport:
        """Load the saved model for `classifier_kind` and score it on `records`.

        Raises:
            ModelNotFoundError: If no model has been saved for this kind.
            ClassifierTrainingError: If the saved pipeline orders the outcomes
                differently from what every calculator here expects.
        """
        pipeline = self._model_repository.load_model(classifier_kind)
        if list(pipeline.classes_) != self._expected_class_labels:
            raise ClassifierTrainingError(
                f"{classifier_kind.display_name} orders the outcomes as "
                f"{list(pipeline.classes_)}, expected {self._expected_class_labels}."
            )

        outcome_column = self._feature_catalog.outcome_column
        input_columns = self._feature_catalog.input_columns(list(records.columns))
        record_inputs = records[input_columns]
        true_labels = records[outcome_column]
        predicted_labels = pipeline.predict(record_inputs)
        predicted_probabilities = pipeline.predict_proba(record_inputs)

        return EvaluationReport(
            classifier_kind=classifier_kind,
            scores=self._scores_calculator.calculate(
                true_labels, predicted_labels, predicted_probabilities
            ),
            confusion_matrix=self._confusion_matrix_calculator.calculate(
                true_labels, predicted_labels
            ),
            classification_report=self._classification_report_calculator.calculate(
                true_labels, predicted_labels
            ),
            curves_by_name=self._curve_calculator_registry.calculate_all(
                self._expected_class_labels, true_labels, predicted_probabilities
            ),
        )
