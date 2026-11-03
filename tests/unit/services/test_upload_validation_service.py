import pandas as pd
import pytest

from gradpath.backend.exceptions.invalid_records_error import InvalidRecordsError
from gradpath.backend.services.upload_validation_service import UploadValidationService

REQUIRED_COLUMNS = ["Marital Status", "Age at enrollment", "Target"]
KNOWN_LABELS = ["Dropout", "Enrolled", "Graduate"]


def make_service(tiny_feature_catalog, max_row_count=10_000):
    return UploadValidationService(
        feature_catalog=tiny_feature_catalog,
        required_columns=REQUIRED_COLUMNS,
        known_outcome_labels=KNOWN_LABELS,
        max_row_count=max_row_count,
    )


def valid_upload():
    return pd.DataFrame(
        {
            "Marital Status": [1, 2, 3],
            "Age at enrollment": [18, 25, 30],
            "Target": ["Dropout", "Enrolled", "Graduate"],
        }
    )


def test_a_valid_upload_is_returned_unchanged_with_no_warnings(tiny_feature_catalog):
    cleaned, warnings = make_service(tiny_feature_catalog).validate(valid_upload())

    assert len(cleaned) == 3
    assert warnings == []


def test_an_empty_upload_is_rejected(tiny_feature_catalog):
    empty = valid_upload().iloc[0:0]

    with pytest.raises(InvalidRecordsError, match="no rows"):
        make_service(tiny_feature_catalog).validate(empty)


def test_a_missing_required_column_is_rejected(tiny_feature_catalog):
    missing_target = valid_upload().drop(columns=["Target"])

    with pytest.raises(InvalidRecordsError, match="Target"):
        make_service(tiny_feature_catalog).validate(missing_target)


def test_more_rows_than_the_cap_is_rejected(tiny_feature_catalog):
    with pytest.raises(InvalidRecordsError, match="row limit"):
        make_service(tiny_feature_catalog, max_row_count=2).validate(valid_upload())


def test_a_missing_value_in_a_required_column_drops_only_that_row(tiny_feature_catalog):
    with_null = valid_upload()
    with_null.loc[0, "Age at enrollment"] = None

    cleaned, warnings = make_service(tiny_feature_catalog).validate(with_null)

    assert len(cleaned) == 2
    assert any("missing value" in warning for warning in warnings)


def test_an_unknown_outcome_label_drops_only_that_row(tiny_feature_catalog):
    unknown_label = valid_upload()
    unknown_label.loc[0, "Target"] = "Withdrawn"

    cleaned, warnings = make_service(tiny_feature_catalog).validate(unknown_label)

    assert len(cleaned) == 2
    assert "Dropout" not in cleaned["Target"].values
    assert any("1 row(s) dropped" in warning for warning in warnings)


def test_every_row_invalid_is_rejected(tiny_feature_catalog):
    all_unknown = valid_upload()
    all_unknown["Target"] = "Withdrawn"

    with pytest.raises(InvalidRecordsError, match="nothing left to score"):
        make_service(tiny_feature_catalog).validate(all_unknown)


def test_a_missing_outcome_class_is_warned_about(tiny_feature_catalog):
    no_enrolled = valid_upload()
    no_enrolled.loc[1, "Target"] = "Graduate"

    _, warnings = make_service(tiny_feature_catalog).validate(no_enrolled)

    assert any("Enrolled" in warning for warning in warnings)


def test_an_extra_unrelated_column_is_allowed(tiny_feature_catalog):
    extra_column = valid_upload()
    extra_column["Some Extra Column"] = [1, 2, 3]

    cleaned, _ = make_service(tiny_feature_catalog).validate(extra_column)

    assert "Some Extra Column" in cleaned.columns
