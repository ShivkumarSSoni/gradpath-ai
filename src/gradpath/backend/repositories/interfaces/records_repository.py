"""Contract for reading and writing student records."""

from abc import ABC, abstractmethod

import pandas as pd


class RecordsRepository(ABC):
    """Where student records come from and where the test part is stored.

    Services depend on this class, never on a concrete data source, so the
    source (UCI download, a local file, a test double) can be swapped freely.
    """

    @abstractmethod
    def load_all_records(self) -> pd.DataFrame:
        """Return every student record, including the outcome column."""

    @abstractmethod
    def save_held_out_records(self, held_out_records: pd.DataFrame) -> None:
        """Store the held-out test records so the app can use them later."""
