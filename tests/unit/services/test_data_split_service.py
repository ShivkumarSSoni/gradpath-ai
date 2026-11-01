"""Unit tests for the stratified training/test split."""

import pandas as pd
import pytest

from gradpath.backend.exceptions.invalid_records_error import InvalidRecordsError
from gradpath.backend.services.data_split_service import DataSplitService

OUTCOME = "Target"


def make_service(seed: int = 42) -> DataSplitService:
    return DataSplitService(outcome_column=OUTCOME, test_fraction=0.20, random_seed=seed)


def test_split_sizes_follow_the_test_fraction(tiny_student_records):
    split = make_service().split_records(tiny_student_records)

    assert len(split.held_out_records) == 20
    assert len(split.training_records) == 80


def test_split_keeps_class_shares_in_both_parts(tiny_student_records):
    split = make_service().split_records(tiny_student_records)

    held_out_counts = split.held_out_records[OUTCOME].value_counts().to_dict()
    assert held_out_counts == {"Graduate": 10, "Dropout": 6, "Enrolled": 4}


def test_split_parts_do_not_share_rows(tiny_student_records):
    records = tiny_student_records.assign(row_id=range(len(tiny_student_records)))

    split = make_service().split_records(records)

    shared_ids = set(split.training_records["row_id"]) & set(split.held_out_records["row_id"])
    assert shared_ids == set()
    assert len(split.training_records) + len(split.held_out_records) == len(records)


def test_same_seed_gives_identical_split(tiny_student_records):
    first = make_service(seed=1).split_records(tiny_student_records)
    second = make_service(seed=1).split_records(tiny_student_records)

    pd.testing.assert_frame_equal(first.held_out_records, second.held_out_records)


def test_different_seed_gives_different_split(tiny_student_records):
    first = make_service(seed=1).split_records(tiny_student_records)
    second = make_service(seed=2).split_records(tiny_student_records)

    assert not first.held_out_records.equals(second.held_out_records)


def test_row_index_is_reset_in_both_parts(tiny_student_records):
    split = make_service().split_records(tiny_student_records)

    assert list(split.training_records.index) == list(range(80))
    assert list(split.held_out_records.index) == list(range(20))


def test_missing_outcome_column_raises_clear_error(tiny_student_records):
    without_outcome = tiny_student_records.drop(columns=[OUTCOME])

    with pytest.raises(InvalidRecordsError, match="Target"):
        make_service().split_records(without_outcome)
