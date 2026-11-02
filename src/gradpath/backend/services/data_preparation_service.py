"""Supplies the training and held-out records to whoever needs them."""

from gradpath.backend.repositories.interfaces.records_repository import (
    RecordsRepository,
)
from gradpath.backend.schemas.records_split import RecordsSplit
from gradpath.backend.services.data_split_service import DataSplitService


class DataPreparationService:
    """Loads every record and divides it into training and held-out parts."""

    def __init__(
        self,
        records_repository: RecordsRepository,
        data_split_service: DataSplitService,
    ) -> None:
        self._records_repository = records_repository
        self._data_split_service = data_split_service

    def prepare_records_split(self) -> RecordsSplit:
        """Return the same split every time, because the split seed is fixed."""
        all_records = self._records_repository.load_all_records()
        return self._data_split_service.split_records(all_records)
