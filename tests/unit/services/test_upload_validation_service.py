import pandas as pd
import pytest

from gradpath.backend.exceptions.invalid_records_error import InvalidRecordsError
from gradpath.backend.services.upload_validation_service import UploadValidationService

REQUIRED_COLUMNS = ["Marital Status", "Age at enrollment", "Target"]
KNOWN_LABELS = ["Dropout", "Enrolled", "Graduate"]


def make_service(tiny_feature_catalog):
    return UploadValidationService(
        feature_catalog=tiny_feature_catalog,
        required_columns=REQUIRED_COLUMNS,
        known_outcome_labels=KNOWN_LABELS,
    )


def valid_upload():
    return pd.DataFrame(
        {
            "Marital Status": [1, 2],
            "Age at enrollment": [18, 25],
            "Target": ["Dropout", "Graduate"],
        }
    )


def test_a_valid_upload_raises_nothing(tiny_feature_catalog):
    make_service(tiny_feature_catalog).validate(valid_upload())


def test_an_empty_upload_is_rejected(tiny_feature_catalog):
    empty = valid_upload().iloc[0:0]

    with pytest.raises(InvalidRecordsError, match="no rows"):
        make_service(tiny_feature_catalog).validate(empty)


def test_a_missing_required_column_is_rejected(tiny_feature_catalog):
    missing_target = valid_upload().drop(columns=["Target"])

    with pytest.raises(InvalidRecordsError, match="Target"):
        make_service(tiny_feature_catalog).validate(missing_target)


def test_a_missing_value_in_a_required_column_is_rejected(tiny_feature_catalog):
    with_null = valid_upload()
    with_null.loc[0, "Age at enrollment"] = None

    with pytest.raises(InvalidRecordsError, match="missing values"):
        make_service(tiny_feature_catalog).validate(with_null)


def test_an_unknown_outcome_label_is_rejected(tiny_feature_catalog):
    unknown_label = valid_upload()
    unknown_label.loc[0, "Target"] = "Withdrawn"

    with pytest.raises(InvalidRecordsError, match="Withdrawn"):
        make_service(tiny_feature_catalog).validate(unknown_label)


def test_an_extra_unrelated_column_is_allowed(tiny_feature_catalog):
    extra_column = valid_upload()
    extra_column["Some Extra Column"] = [1, 2]

    make_service(tiny_feature_catalog).validate(extra_column)
