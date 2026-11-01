"""Shared small, fake student records so tests never touch the network."""

import numpy as np
import pandas as pd
import pytest

from gradpath.backend.domain.feature_catalog import STUDENT_FEATURE_CATALOG
from gradpath.backend.domain.student_outcome import StudentOutcome


@pytest.fixture
def tiny_student_records() -> pd.DataFrame:
    """Return 100 fake records with the same column roles as the real data.

    The outcome shares are 50 Graduate, 30 Dropout and 20 Enrolled, so a
    stratified split has an exact, checkable answer.
    """
    random_generator = np.random.default_rng(seed=7)
    record_count = 100
    outcomes = (
        [StudentOutcome.GRADUATE.value] * 50
        + [StudentOutcome.DROPOUT.value] * 30
        + [StudentOutcome.ENROLLED.value] * 20
    )
    return pd.DataFrame(
        {
            "Marital Status": random_generator.integers(1, 4, record_count),
            "Course": random_generator.integers(1, 6, record_count),
            "Age at enrollment": random_generator.integers(17, 40, record_count),
            "Admission grade": random_generator.normal(130, 10, record_count),
            STUDENT_FEATURE_CATALOG.outcome_column: outcomes,
        }
    )
