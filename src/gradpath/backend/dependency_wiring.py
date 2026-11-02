"""The one place where backend objects are created and connected.

Every other module receives its dependencies through its constructor, so
swapping an implementation means editing only this file.
"""

import pandas as pd

from gradpath.backend.config.settings import (
    DATASET_CACHE_FILE,
    EVALUATION_RESULTS_FILE,
    HELD_OUT_TEST_DATA_FILE,
    RANDOM_SEED_FOR_REPRODUCIBILITY,
    SAVED_MODELS_FOLDER,
    TEST_SPLIT_FRACTION,
    UCI_DATASET_IDENTIFIER,
)
from gradpath.backend.controllers.comparison_controller import ComparisonController
from gradpath.backend.controllers.evaluation_controller import EvaluationController
from gradpath.backend.controllers.training_controller import TrainingController
from gradpath.backend.domain.feature_catalog import STUDENT_FEATURE_CATALOG
from gradpath.backend.domain.student_outcome import StudentOutcome
from gradpath.backend.ml.classifier_builders.classifier_registry import (
    ClassifierRegistry,
)
from gradpath.backend.ml.classifier_builders.decision_tree_builder import (
    DecisionTreeBuilder,
)
from gradpath.backend.ml.classifier_builders.gaussian_naive_bayes_builder import (
    GaussianNaiveBayesBuilder,
)
from gradpath.backend.ml.classifier_builders.k_nearest_neighbors_builder import (
    KNearestNeighborsBuilder,
)
from gradpath.backend.ml.classifier_builders.logistic_regression_builder import (
    LogisticRegressionBuilder,
)
from gradpath.backend.ml.classifier_builders.support_vector_machine_builder import (
    SupportVectorMachineBuilder,
)
from gradpath.backend.ml.metrics.classification_report_calculator import (
    ClassificationReportCalculator,
)
from gradpath.backend.ml.metrics.confusion_matrix_calculator import (
    ConfusionMatrixCalculator,
)
from gradpath.backend.ml.metrics.curves.calibration_curve_calculator import (
    CalibrationCurveCalculator,
)
from gradpath.backend.ml.metrics.curves.curve_calculator_registry import (
    CurveCalculatorRegistry,
)
from gradpath.backend.ml.metrics.curves.precision_recall_curve_calculator import (
    PrecisionRecallCurveCalculator,
)
from gradpath.backend.ml.metrics.curves.roc_curve_calculator import RocCurveCalculator
from gradpath.backend.ml.metrics.scores_calculator import ScoresCalculator
from gradpath.backend.ml.preprocessing.feature_transformer_builder import (
    FeatureTransformerBuilder,
)
from gradpath.backend.repositories.csv_evaluation_results_repository import (
    CsvEvaluationResultsRepository,
)
from gradpath.backend.repositories.interfaces.evaluation_results_repository import (
    EvaluationResultsRepository,
)
from gradpath.backend.repositories.interfaces.model_repository import ModelRepository
from gradpath.backend.repositories.interfaces.records_repository import (
    RecordsRepository,
)
from gradpath.backend.repositories.joblib_model_repository import JoblibModelRepository
from gradpath.backend.repositories.uci_records_repository import (
    UciRecordsRepository,
)
from gradpath.backend.services.classifier_build_service import ClassifierBuildService
from gradpath.backend.services.comparison_service import ComparisonService
from gradpath.backend.services.data_preparation_service import DataPreparationService
from gradpath.backend.services.data_split_service import DataSplitService
from gradpath.backend.services.evaluation_report_service import EvaluationReportService
from gradpath.backend.services.evaluation_service import EvaluationService
from gradpath.backend.services.results_recorder_service import ResultsRecorderService
from gradpath.backend.services.training_service import TrainingService
from gradpath.backend.services.upload_validation_service import UploadValidationService


def build_records_repository() -> RecordsRepository:
    """Create the repository that supplies student records."""
    return UciRecordsRepository(
        uci_dataset_identifier=UCI_DATASET_IDENTIFIER,
        cache_file=DATASET_CACHE_FILE,
        held_out_file=HELD_OUT_TEST_DATA_FILE,
        outcome_column=STUDENT_FEATURE_CATALOG.outcome_column,
    )


def build_data_split_service() -> DataSplitService:
    """Create the service that makes the stratified training/test split."""
    return DataSplitService(
        outcome_column=STUDENT_FEATURE_CATALOG.outcome_column,
        test_fraction=TEST_SPLIT_FRACTION,
        random_seed=RANDOM_SEED_FOR_REPRODUCIBILITY,
    )


def build_scores_calculator() -> ScoresCalculator:
    """Create the calculator of the six evaluation scores."""
    return ScoresCalculator(class_labels=StudentOutcome.all_labels())


def build_classifier_registry() -> ClassifierRegistry:
    """Create the registry holding the builders of all five classifiers."""
    seed = RANDOM_SEED_FOR_REPRODUCIBILITY
    return ClassifierRegistry(
        [
            LogisticRegressionBuilder(random_seed=seed),
            DecisionTreeBuilder(random_seed=seed),
            KNearestNeighborsBuilder(random_seed=seed),
            GaussianNaiveBayesBuilder(random_seed=seed),
            SupportVectorMachineBuilder(random_seed=seed),
        ]
    )


def build_feature_transformer_builder() -> FeatureTransformerBuilder:
    """Create the builder of the shared preprocessing step."""
    return FeatureTransformerBuilder(STUDENT_FEATURE_CATALOG)


def build_model_repository() -> ModelRepository:
    """Create the repository that saves and loads the trained models."""
    return JoblibModelRepository(models_folder=SAVED_MODELS_FOLDER)


def build_evaluation_results_repository() -> EvaluationResultsRepository:
    """Create the repository that records the scores of each classifier."""
    return CsvEvaluationResultsRepository(results_file=EVALUATION_RESULTS_FILE)


def build_data_preparation_service() -> DataPreparationService:
    """Create the service that supplies the training and held-out records."""
    return DataPreparationService(
        records_repository=build_records_repository(),
        data_split_service=build_data_split_service(),
    )


def build_training_service() -> TrainingService:
    """Create the service that fits one classifier pipeline and saves it."""
    return TrainingService(
        classifier_registry=build_classifier_registry(),
        feature_transformer_builder=build_feature_transformer_builder(),
        model_repository=build_model_repository(),
        feature_catalog=STUDENT_FEATURE_CATALOG,
    )


def build_evaluation_service() -> EvaluationService:
    """Create the service that scores a saved model with the six official metrics."""
    return EvaluationService(
        model_repository=build_model_repository(),
        scores_calculator=build_scores_calculator(),
        feature_catalog=STUDENT_FEATURE_CATALOG,
        expected_class_labels=StudentOutcome.all_labels(),
    )


def build_results_recorder_service() -> ResultsRecorderService:
    """Create the service that is the only writer of the official results file."""
    return ResultsRecorderService(
        evaluation_results_repository=build_evaluation_results_repository(),
    )


def build_classifier_build_service() -> ClassifierBuildService:
    """Create the service that trains, scores and records one classifier."""
    return ClassifierBuildService(
        training_service=build_training_service(),
        evaluation_service=build_evaluation_service(),
        results_recorder_service=build_results_recorder_service(),
    )


def build_confusion_matrix_calculator() -> ConfusionMatrixCalculator:
    """Create the calculator of the confusion matrix shown in the app."""
    return ConfusionMatrixCalculator(class_labels=StudentOutcome.all_labels())


def build_classification_report_calculator() -> ClassificationReportCalculator:
    """Create the calculator of the per-outcome precision, recall, F1 and support."""
    return ClassificationReportCalculator(class_labels=StudentOutcome.all_labels())


def build_curve_calculator_registry() -> CurveCalculatorRegistry:
    """Create the registry holding the ROC, Precision-Recall and calibration curves."""
    return CurveCalculatorRegistry(
        [
            RocCurveCalculator(),
            PrecisionRecallCurveCalculator(),
            CalibrationCurveCalculator(),
        ]
    )


def build_evaluation_report_service() -> EvaluationReportService:
    """Create the service that builds the app's full evaluation detail."""
    return EvaluationReportService(
        model_repository=build_model_repository(),
        scores_calculator=build_scores_calculator(),
        confusion_matrix_calculator=build_confusion_matrix_calculator(),
        classification_report_calculator=build_classification_report_calculator(),
        curve_calculator_registry=build_curve_calculator_registry(),
        feature_catalog=STUDENT_FEATURE_CATALOG,
        expected_class_labels=StudentOutcome.all_labels(),
    )


def build_upload_required_columns() -> list[str]:
    """Return the columns an uploaded file must have, read from the held-out file.

    The committed `test_data.csv` is the one file guaranteed to carry every
    column a saved pipeline needs, so its header is the single source of
    truth for what counts as a valid upload.
    """
    return pd.read_csv(HELD_OUT_TEST_DATA_FILE, nrows=0).columns.tolist()


def build_upload_validation_service() -> UploadValidationService:
    """Create the service that checks an uploaded file before it is scored."""
    return UploadValidationService(
        feature_catalog=STUDENT_FEATURE_CATALOG,
        required_columns=build_upload_required_columns(),
        known_outcome_labels=StudentOutcome.all_labels(),
    )


def build_comparison_service() -> ComparisonService:
    """Create the service that shapes the official results into comparison rows."""
    return ComparisonService(
        evaluation_results_repository=build_evaluation_results_repository(),
    )


def build_training_controller() -> TrainingController:
    """Create the controller that trains one classifier or all five."""
    return TrainingController(
        data_preparation_service=build_data_preparation_service(),
        classifier_build_service=build_classifier_build_service(),
    )


def build_evaluation_controller() -> EvaluationController:
    """Create the controller that validates and scores an uploaded file."""
    return EvaluationController(
        upload_validation_service=build_upload_validation_service(),
        evaluation_report_service=build_evaluation_report_service(),
    )


def build_comparison_controller() -> ComparisonController:
    """Create the controller behind the model comparison page."""
    return ComparisonController(comparison_service=build_comparison_service())
