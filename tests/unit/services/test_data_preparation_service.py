from gradpath.backend.services.data_preparation_service import DataPreparationService
from gradpath.backend.services.data_split_service import DataSplitService
from gradpath.backend.repositories.interfaces.records_repository import RecordsRepository


class StubRecordsRepository(RecordsRepository):
    def __init__(self, records):
        self._records = records
        self.load_count = 0

    def load_all_records(self):
        self.load_count += 1
        return self._records

    def save_held_out_records(self, held_out_records):
        raise AssertionError("Preparing a split must not write any file.")


def test_split_covers_all_records_and_is_identical_on_every_call(tiny_student_records):
    repository = StubRecordsRepository(tiny_student_records)
    service = DataPreparationService(
        records_repository=repository,
        data_split_service=DataSplitService("Target", test_fraction=0.2, random_seed=3),
    )

    first = service.prepare_records_split()
    second = service.prepare_records_split()

    assert len(first.training_records) + len(first.held_out_records) == 100
    assert first.held_out_records.equals(second.held_out_records)
    assert repository.load_count == 2
