"""The one place where backend objects are created and connected.

Every other module receives its dependencies through its constructor, so
swapping an implementation means editing only this file.
"""

from gradpath.backend.config.settings import (
    DATASET_CACHE_FILE,
    HELD_OUT_TEST_DATA_FILE,
    RANDOM_SEED_FOR_REPRODUCIBILITY,
    TEST_SPLIT_FRACTION,
    UCI_DATASET_IDENTIFIER,
)
from gradpath.backend.domain.feature_catalog import STUDENT_FEATURE_CATALOG
from gradpath.backend.domain.student_outcome import StudentOutcome
from gradpath.backend.ml.metrics.scores_calculator import ScoresCalculator
from gradpath.backend.ml.preprocessing.feature_transformer_builder import (
    FeatureTransformerBuilder,
)
from gradpath.backend.repositories.interfaces.records_repository import (
    RecordsRepository,
)
from gradpath.backend.repositories.uci_records_repository import (
    UciRecordsRepository,
)
from gradpath.backend.services.data_split_service import DataSplitService


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


def build_feature_transformer_builder() -> FeatureTransformerBuilder:
    """Create the builder of the shared preprocessing step."""
    return FeatureTransformerBuilder(STUDENT_FEATURE_CATALOG)
