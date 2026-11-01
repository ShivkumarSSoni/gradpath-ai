"""Student records served from the UCI repository with a local cache.

The records come from the UCI Machine Learning Repository. The first load
downloads them and keeps a local copy, so later runs work offline and never
download twice.
"""

import logging
from pathlib import Path

import pandas as pd

from gradpath.backend.exceptions.invalid_records_error import InvalidRecordsError
from gradpath.backend.repositories.interfaces.records_repository import (
    RecordsRepository,
)

logger = logging.getLogger(__name__)


class UciRecordsRepository(RecordsRepository):
    """Reads records from the UCI dataset and writes the held-out test file."""

    def __init__(
        self,
        uci_dataset_identifier: int,
        cache_file: Path,
        held_out_file: Path,
        outcome_column: str,
    ) -> None:
        """Store where to download from and where to read and write files.

        Args:
            uci_dataset_identifier: Numeric id of the dataset on UCI.
            cache_file: Local CSV that keeps the downloaded records.
            held_out_file: CSV that receives the held-out test records.
            outcome_column: Name of the outcome column that must be present.
        """
        self._uci_dataset_identifier = uci_dataset_identifier
        self._cache_file = cache_file
        self._held_out_file = held_out_file
        self._outcome_column = outcome_column

    def load_all_records(self) -> pd.DataFrame:
        """Return every record, downloading from UCI only on the first call.

        Raises:
            InvalidRecordsError: If the table has no outcome column.
        """
        if self._cache_file.exists():
            logger.info("Reading cached records from %s", self._cache_file)
            student_records = pd.read_csv(self._cache_file)
        else:
            student_records = self._download_records()
            self._cache_file.parent.mkdir(parents=True, exist_ok=True)
            student_records.to_csv(self._cache_file, index=False)

        if self._outcome_column not in student_records.columns:
            raise InvalidRecordsError(
                f"Outcome column '{self._outcome_column}' is missing from the records."
            )
        return student_records

    def save_held_out_records(self, held_out_records: pd.DataFrame) -> None:
        """Write the held-out records as CSV without the row index."""
        held_out_records.to_csv(self._held_out_file, index=False)
        logger.info("Saved %d held-out records", len(held_out_records))

    def _download_records(self) -> pd.DataFrame:
        """Download features and outcome from UCI and join them side by side."""
        # Imported here so the Streamlit app, which never downloads, does not
        # need the ucimlrepo package installed.
        from ucimlrepo import fetch_ucirepo

        logger.info("Downloading dataset %s from UCI", self._uci_dataset_identifier)
        uci_dataset = fetch_ucirepo(id=self._uci_dataset_identifier)
        return pd.concat(
            [uci_dataset.data.features, uci_dataset.data.targets], axis=1
        )
