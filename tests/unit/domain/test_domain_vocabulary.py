"""Unit tests for the domain vocabulary."""

from gradpath.backend.domain.feature_catalog import (
    STUDENT_FEATURE_CATALOG,
    FeatureCatalog,
)
from gradpath.backend.domain.student_outcome import StudentOutcome


def test_outcome_labels_match_dataset_spelling_and_order():
    assert StudentOutcome.all_labels() == ["Dropout", "Enrolled", "Graduate"]


def test_outcome_compares_equal_to_plain_string():
    assert StudentOutcome.DROPOUT == "Dropout"


def test_numeric_columns_exclude_outcome_and_nominal_codes():
    catalog = FeatureCatalog(outcome_column="y", nominal_code_columns=("code",))

    assert catalog.numeric_columns(["code", "age", "y", "grade"]) == ["age", "grade"]


def test_input_columns_drop_only_the_outcome_and_keep_order():
    catalog = FeatureCatalog(outcome_column="y", nominal_code_columns=("code",))

    assert catalog.input_columns(["code", "age", "y"]) == ["code", "age"]


def test_input_columns_unchanged_when_outcome_absent():
    catalog = FeatureCatalog(outcome_column="y", nominal_code_columns=())

    assert catalog.input_columns(["a", "b"]) == ["a", "b"]


def test_real_catalog_uses_target_as_outcome_and_nine_code_columns():
    assert STUDENT_FEATURE_CATALOG.outcome_column == "Target"
    assert len(STUDENT_FEATURE_CATALOG.nominal_code_columns) == 9
