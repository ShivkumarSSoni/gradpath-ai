"""Checks an uploaded CSV before it is ever given to a saved model.

Kept as its own service so the app can show one clear, specific message
instead of letting a missing column or an unknown label crash with a raw
stack trace.
"""

import pandas as pd

from gradpath.backend.domain.feature_catalog import FeatureCatalog
from gradpath.backend.exceptions.invalid_records_error import InvalidRecordsError


class UploadValidationService:
    """Validates an uploaded table against the columns and labels it must have."""

    def __init__(
        self,
        feature_catalog: FeatureCatalog,
        required_columns: list[str],
        known_outcome_labels: list[str],
    ) -> None:
        """Remember what a valid upload must contain.

        Args:
            required_columns: Every column the saved pipelines and the
                scores calculator need, including the outcome column.
            known_outcome_labels: The only outcome values that mean anything
                to a saved model.
        """
        self._feature_catalog = feature_catalog
        self._required_columns = list(required_columns)
        self._known_outcome_labels = set(known_outcome_labels)

    def validate(self, uploaded_records: pd.DataFrame) -> None:
        """Raise a clear error for the first problem found, otherwise do nothing.

        Raises:
            InvalidRecordsError: If the file is empty, is missing a required
                column, has missing values in a required column, or has an
                outcome value that is not one of the three known outcomes.
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

        required_table = uploaded_records[self._required_columns]
        if required_table.isna().any().any():
            raise InvalidRecordsError(
                "The uploaded file has missing values in a required column."
            )

        outcome_column = self._feature_catalog.outcome_column
        unknown_labels = set(uploaded_records[outcome_column].unique()) - self._known_outcome_labels
        if unknown_labels:
            raise InvalidRecordsError(
                f"The uploaded file has unknown outcome values: {sorted(unknown_labels)}."
            )
