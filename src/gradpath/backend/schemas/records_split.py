"""The result of dividing student records into training and test parts."""

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class RecordsSplit:
    """Two disjoint tables of records, each including the outcome column.

    Attributes:
        training_records: Records used to fit preprocessing and classifiers.
        held_out_records: Records never seen during training, used only for
            testing and exported as test_data.csv.
    """

    training_records: pd.DataFrame
    held_out_records: pd.DataFrame
