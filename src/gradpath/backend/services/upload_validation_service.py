"""Checks an uploaded CSV before it is ever given to a saved model.

Kept as its own service so the app can show one clear, specific message
instead of letting a missing column or an unknown label crash with a raw
stack trace.
"""

import pandas as pd

from gradpath.backend.domain.feature_catalog import FeatureCatalog
from gradpath.backend.exceptions.invalid_records_error import InvalidRecordsError

# Protects the shared, free-tier app from an unreasonably large upload. The
# real test_data.csv has 885 rows, so this leaves generous headroom.
DEFAULT_MAX_UPLOAD_ROW_COUNT = 10_000


class UploadValidationService:
    """Validates an uploaded table, then cleans it row by row.

    A structural problem (a required column is missing, or there are no
    rows left after cleaning) rejects the whole file, because nothing can be
    scored without it. A row-level data problem (one bad outcome label, one
    null cell) drops only that row, so one bad row never throws away an
    otherwise usable file.
    """

    def __init__(
        self,
        feature_catalog: FeatureCatalog,
        required_columns: list[str],
        known_outcome_labels: list[str],
        max_row_count: int = DEFAULT_MAX_UPLOAD_ROW_COUNT,
    ) -> None:
        """Remember what a valid upload must contain.

        Args:
            required_columns: Every column the saved pipelines and the
                scores calculator need, including the outcome column.
            known_outcome_labels: The only outcome values that mean anything
                to a saved model.
            max_row_count: Hard cap on how many rows a single upload may have.
        """
        self._feature_catalog = feature_catalog
        self._required_columns = list(required_columns)
        self._known_outcome_labels = list(known_outcome_labels)
        self._max_row_count = max_row_count

    def validate(self, uploaded_records: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
        """Reject a structurally unusable file, otherwise clean it row by row.

        Returns:
            A tuple of the cleaned records (bad rows removed, extra columns
            kept untouched) and a list of plain-English warnings describing
            what, if anything, was dropped or looks thin.

        Raises:
            InvalidRecordsError: If the file has no rows, is missing a
                required column, has more rows than the cap allows, or has
                zero valid rows left after cleaning.
        """
        if uploaded_records.empty:
            raise InvalidRecordsError("The uploaded file has no rows to score.")

        missing_columns = [
            column for column in self._required_columns if column not in uploaded_records.columns
        ]
        if missing_columns:
            raise InvalidRecordsError(
                "The uploaded file is missing required columns: "
                f"{', '.join(missing_columns)}."
            )

        if len(uploaded_records) > self._max_row_count:
            raise InvalidRecordsError(
                f"The uploaded file has {len(uploaded_records)} rows, which is "
                f"more than the {self._max_row_count}-row limit. Upload a "
                "smaller file."
            )

        cleaned_records = uploaded_records.copy()
        warnings: list[str] = []
        cleaned_records, warnings = self._drop_rows_with_nulls(cleaned_records, warnings)
        cleaned_records, warnings = self._drop_rows_with_unknown_labels(cleaned_records, warnings)

        if cleaned_records.empty:
            raise InvalidRecordsError(
                "Every row had a missing value or an unrecognized outcome; "
                "there is nothing left to score."
            )

        warnings = self._warn_about_missing_outcomes(cleaned_records, warnings)
        return cleaned_records.reset_index(drop=True), warnings

    def _drop_rows_with_nulls(
        self, records: pd.DataFrame, warnings: list[str]
    ) -> tuple[pd.DataFrame, list[str]]:
        """Remove rows with a missing value in a required column, with a count."""
        has_null = records[self._required_columns].isna().any(axis=1)
        dropped_count = int(has_null.sum())
        if dropped_count == 0:
            return records, warnings
        return records[~has_null], [
            *warnings,
            f"{dropped_count} row(s) dropped: missing value in a required column.",
        ]

    def _drop_rows_with_unknown_labels(
        self, records: pd.DataFrame, warnings: list[str]
    ) -> tuple[pd.DataFrame, list[str]]:
        """Remove rows whose outcome is not one of the three known outcomes."""
        outcome_column = self._feature_catalog.outcome_column
        is_unknown = ~records[outcome_column].isin(self._known_outcome_labels)
        dropped_count = int(is_unknown.sum())
        if dropped_count == 0:
            return records, warnings
        return records[~is_unknown], [
            *warnings,
            f"{dropped_count} row(s) dropped: outcome value is not Dropout, "
            "Enrolled or Graduate.",
        ]

    def _warn_about_missing_outcomes(
        self, records: pd.DataFrame, warnings: list[str]
    ) -> list[str]:
        """Add a warning for every known outcome absent from the cleaned rows.

        A metric that needs every outcome, for example AUC, is undefined
        without this, so the user is told before seeing "Not available".
        """
        outcome_column = self._feature_catalog.outcome_column
        present_labels = set(records[outcome_column].unique())
        missing_labels = [
            label for label in self._known_outcome_labels if label not in present_labels
        ]
        if not missing_labels:
            return warnings
        return [
            *warnings,
            "No rows with outcome "
            f"{', '.join(missing_labels)} in the uploaded file. Metrics that "
            "need every outcome may show 'Not available'.",
        ]
