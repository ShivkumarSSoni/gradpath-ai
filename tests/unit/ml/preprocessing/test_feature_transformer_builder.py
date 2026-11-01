"""Unit tests for the shared preprocessing step."""

import numpy as np
import pandas as pd

from gradpath.backend.domain.feature_catalog import FeatureCatalog
from gradpath.backend.ml.preprocessing.feature_transformer_builder import (
    FeatureTransformerBuilder,
)

CATALOG = FeatureCatalog(
    outcome_column="Target",
    nominal_code_columns=("Marital Status", "Course"),
)


def build_transformer(records: pd.DataFrame):
    return FeatureTransformerBuilder(CATALOG).build(list(records.columns))


def test_output_is_dense_and_has_no_missing_values(tiny_student_records):
    inputs = tiny_student_records.drop(columns=["Target"])

    encoded = build_transformer(inputs).fit_transform(inputs)

    assert isinstance(encoded, np.ndarray)
    assert not np.isnan(encoded).any()


def test_column_count_is_one_hot_codes_plus_numeric_columns(tiny_student_records):
    inputs = tiny_student_records.drop(columns=["Target"])
    code_levels = inputs["Marital Status"].nunique() + inputs["Course"].nunique()

    encoded = build_transformer(inputs).fit_transform(inputs)

    assert encoded.shape == (len(inputs), code_levels + 2)


def test_numeric_columns_are_standardised(tiny_student_records):
    inputs = tiny_student_records.drop(columns=["Target"])
    transformer = build_transformer(inputs)

    encoded = transformer.fit_transform(inputs)
    names = list(transformer.get_feature_names_out())
    admission_column = encoded[:, names.index("Admission grade")]

    assert abs(admission_column.mean()) < 1e-9
    assert abs(admission_column.std() - 1.0) < 1e-9


def test_outcome_column_is_ignored_when_present(tiny_student_records):
    transformer = build_transformer(tiny_student_records)

    encoded = transformer.fit_transform(tiny_student_records)

    assert "Target" not in transformer.get_feature_names_out()
    assert encoded.shape[0] == len(tiny_student_records)


def test_unseen_category_code_becomes_all_zeros_instead_of_crashing(tiny_student_records):
    inputs = tiny_student_records.drop(columns=["Target"])
    transformer = build_transformer(inputs).fit(inputs)
    unseen_row = inputs.head(1).assign(Course=999)

    encoded = transformer.transform(unseen_row)
    names = list(transformer.get_feature_names_out())
    course_positions = [i for i, name in enumerate(names) if name.startswith("Course_")]

    assert encoded[0, course_positions].sum() == 0


def test_each_call_returns_a_new_unfitted_transformer(tiny_student_records):
    builder = FeatureTransformerBuilder(CATALOG)
    columns = list(tiny_student_records.columns)

    assert builder.build(columns) is not builder.build(columns)
