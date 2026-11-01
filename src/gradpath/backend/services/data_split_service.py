"""Dividing student records into a training part and a held-out test part."""

import pandas as pd
from sklearn.model_selection import train_test_split

from gradpath.backend.exceptions.invalid_records_error import InvalidRecordsError
from gradpath.backend.schemas.records_split import RecordsSplit


class DataSplitService:
    """Creates a reproducible, class-balanced training and test split."""

    def __init__(
        self,
        outcome_column: str,
        test_fraction: float,
        random_seed: int,
    ) -> None:
        """Store how the split must be made.

        Args:
            outcome_column: Column whose class proportions must be preserved.
            test_fraction: Share of records, between 0 and 1, held out for testing.
            random_seed: Fixed seed so the same split is produced every time.
        """
        self._outcome_column = outcome_column
        self._test_fraction = test_fraction
        self._random_seed = random_seed

    def split_records(self, student_records: pd.DataFrame) -> RecordsSplit:
        """Split the records, keeping each outcome's share equal in both parts.

        Stratifying matters here because the Enrolled outcome is rare. A plain
        random split could leave the test part with too few Enrolled students
        to judge a model fairly.

        Args:
            student_records: Every record, including the outcome column.

        Returns:
            The training records and the held-out test records.

        Raises:
            InvalidRecordsError: If the outcome column is missing.
        """
        if self._outcome_column not in student_records.columns:
            raise InvalidRecordsError(
                f"Outcome column '{self._outcome_column}' is missing from the records."
            )

        training_records, held_out_records = train_test_split(
            student_records,
            test_size=self._test_fraction,
            random_state=self._random_seed,
            stratify=student_records[self._outcome_column],
        )
        return RecordsSplit(
            training_records=training_records.reset_index(drop=True),
            held_out_records=held_out_records.reset_index(drop=True),
        )
