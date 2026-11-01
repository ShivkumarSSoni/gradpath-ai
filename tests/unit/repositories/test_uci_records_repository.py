"""Unit tests for the records repository, with the UCI download faked."""

import pandas as pd
import pytest

from gradpath.backend.exceptions.invalid_records_error import InvalidRecordsError
from gradpath.backend.repositories.uci_records_repository import UciRecordsRepository


class RepositoryUnderTest(UciRecordsRepository):
    """Replaces the network download with fixed records and counts the calls."""

    def __init__(self, downloaded_records: pd.DataFrame, **kwargs) -> None:
        super().__init__(**kwargs)
        self._downloaded_records = downloaded_records
        self.download_count = 0

    def _download_records(self) -> pd.DataFrame:
        self.download_count += 1
        return self._downloaded_records


def make_repository(tmp_path, records, outcome_column="Target"):
    return RepositoryUnderTest(
        downloaded_records=records,
        uci_dataset_identifier=697,
        cache_file=tmp_path / "cache" / "records.csv",
        held_out_file=tmp_path / "test_data.csv",
        outcome_column=outcome_column,
    )


def test_first_load_downloads_and_writes_the_cache(tmp_path, tiny_student_records):
    repository = make_repository(tmp_path, tiny_student_records)

    loaded = repository.load_all_records()

    assert repository.download_count == 1
    assert (tmp_path / "cache" / "records.csv").exists()
    assert len(loaded) == len(tiny_student_records)


def test_second_load_reads_the_cache_without_downloading(tmp_path, tiny_student_records):
    repository = make_repository(tmp_path, tiny_student_records)
    repository.load_all_records()

    reloaded = repository.load_all_records()

    assert repository.download_count == 1
    pd.testing.assert_frame_equal(reloaded, tiny_student_records, check_dtype=False)


def test_missing_outcome_column_raises_clear_error(tmp_path, tiny_student_records):
    repository = make_repository(tmp_path, tiny_student_records, outcome_column="Nope")

    with pytest.raises(InvalidRecordsError, match="Nope"):
        repository.load_all_records()


def test_saved_held_out_file_has_no_index_column_and_round_trips(tmp_path, tiny_student_records):
    repository = make_repository(tmp_path, tiny_student_records)

    repository.save_held_out_records(tiny_student_records)
    exported = pd.read_csv(tmp_path / "test_data.csv")

    assert "Unnamed: 0" not in exported.columns
    pd.testing.assert_frame_equal(exported, tiny_student_records, check_dtype=False)
