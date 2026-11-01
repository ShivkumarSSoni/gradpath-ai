"""The three outcomes a student in the dataset can end up with."""

from enum import Enum


class StudentOutcome(str, Enum):
    """Final academic status of a student, exactly as spelled in the dataset."""

    DROPOUT = "Dropout"
    ENROLLED = "Enrolled"
    GRADUATE = "Graduate"

    @classmethod
    def all_labels(cls) -> list[str]:
        """Return the outcome labels in a fixed, readable order."""
        return [outcome.value for outcome in cls]
