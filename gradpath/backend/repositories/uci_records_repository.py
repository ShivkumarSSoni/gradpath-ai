"""Loading of the student dropout and academic success records.

The records come from the UCI Machine Learning Repository (dataset id 697).
The first call downloads them and keeps a local copy in the data/ folder, so
later runs work offline and never download twice.
"""

import logging

import pandas as pd

from gradpath.backend.config.settings import (
    OUTCOME_COLUMN_NAME,
    PROJECT_ROOT_FOLDER,
)

logger = logging.getLogger(__name__)

# Identifier of the dataset "Predict Students' Dropout and Academic Success".
UCI_DATASET_IDENTIFIER = 697

# Local copy of the downloaded records (this folder is git-ignored).
LOCAL_RECORDS_CACHE_FILE = PROJECT_ROOT_FOLDER / "data" / "student_records.csv"


def load_student_records() -> pd.DataFrame:
    """Return every student record as one table, including the outcome column.

    Returns:
        A DataFrame with the 36 input features followed by the outcome column
        named in project_config.OUTCOME_COLUMN_NAME.

    Raises:
        ValueError: If the loaded table has no outcome column.
    """
    if LOCAL_RECORDS_CACHE_FILE.exists():
        logger.info("Reading cached records from %s", LOCAL_RECORDS_CACHE_FILE)
        student_records_table = pd.read_csv(LOCAL_RECORDS_CACHE_FILE)
    else:
        student_records_table = _download_records_from_uci()
        LOCAL_RECORDS_CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
        student_records_table.to_csv(LOCAL_RECORDS_CACHE_FILE, index=False)

    if OUTCOME_COLUMN_NAME not in student_records_table.columns:
        raise ValueError(
            f"Outcome column '{OUTCOME_COLUMN_NAME}' is missing from the records."
        )
    return student_records_table


def _download_records_from_uci() -> pd.DataFrame:
    """Download the features and outcome from UCI and join them side by side."""
    # Imported here so the Streamlit app, which never downloads, does not need
    # the ucimlrepo package installed.
    from ucimlrepo import fetch_ucirepo

    logger.info("Downloading dataset %s from UCI", UCI_DATASET_IDENTIFIER)
    uci_dataset = fetch_ucirepo(id=UCI_DATASET_IDENTIFIER)
    return pd.concat(
        [uci_dataset.data.features, uci_dataset.data.targets], axis=1
    )
