"""Component integration tests: wiring, split and preprocessing working together."""

import importlib
from pathlib import Path

import pandas as pd
import pytest

from gradpath.backend import dependency_wiring
from gradpath.backend.config import settings
from gradpath.backend.domain.feature_catalog import STUDENT_FEATURE_CATALOG
from gradpath.backend.repositories.interfaces.records_repository import RecordsRepository
from gradpath.backend.services.data_split_service import DataSplitService

pytestmark = pytest.mark.integration

OUTCOME = STUDENT_FEATURE_CATALOG.outcome_column


def with_every_code_column(records):
    """Add the catalog's code columns that the small fake records lack."""
    padded = records.copy()
    for column in STUDENT_FEATURE_CATALOG.nominal_code_columns:
        if column not in padded.columns:
            padded[column] = 1
    return padded


def split_and_fit_transformer(records):
    """Run the real split service and fit the real transformer on training only."""
    records = with_every_code_column(records)
    split = dependency_wiring.build_data_split_service().split_records(records)
    input_columns = STUDENT_FEATURE_CATALOG.input_columns(list(records.columns))
    transformer = dependency_wiring.build_feature_transformer_builder().build(input_columns)
    transformer.fit(split.training_records[input_columns])
    return split, transformer, input_columns


def test_wiring_builds_objects_of_the_expected_types():
    assert isinstance(dependency_wiring.build_records_repository(), RecordsRepository)
    assert isinstance(dependency_wiring.build_data_split_service(), DataSplitService)


def test_settings_point_inside_the_repository_root():
    assert (settings.PROJECT_ROOT_FOLDER / "src" / "gradpath").is_dir()
    assert settings.HELD_OUT_TEST_DATA_FILE.parent == settings.PROJECT_ROOT_FOLDER


def test_gradpath_package_is_importable_from_the_src_folder():
    gradpath_package = importlib.import_module("gradpath")

    package_folder = Path(gradpath_package.__file__).resolve().parent
    assert package_folder == settings.PROJECT_ROOT_FOLDER / "src" / "gradpath"


def test_training_and_held_out_matrices_have_the_same_columns(tiny_student_records):
    split, transformer, input_columns = split_and_fit_transformer(tiny_student_records)

    training_matrix = transformer.transform(split.training_records[input_columns])
    held_out_matrix = transformer.transform(split.held_out_records[input_columns])

    assert training_matrix.shape[1] == held_out_matrix.shape[1]
    assert OUTCOME not in transformer.get_feature_names_out()


def test_exported_test_file_can_be_reloaded_and_transformed(tmp_path, tiny_student_records):
    split, transformer, input_columns = split_and_fit_transformer(tiny_student_records)
    repository = dependency_wiring.build_records_repository()
    repository._held_out_file = tmp_path / "test_data.csv"

    repository.save_held_out_records(split.held_out_records)
    uploaded = pd.read_csv(tmp_path / "test_data.csv")
    encoded_upload = transformer.transform(uploaded[input_columns])

    assert encoded_upload.shape[0] == len(split.held_out_records)
