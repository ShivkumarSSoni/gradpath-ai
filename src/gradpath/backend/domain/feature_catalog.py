"""Describes which columns of a student record play which role."""

from dataclasses import dataclass


@dataclass(frozen=True)
class FeatureCatalog:
    """Groups the columns of the student records by how they must be treated.

    Attributes:
        outcome_column: Column holding the outcome that models must predict.
        nominal_code_columns: Columns stored as numbers that are really
            category codes (for example a course code). They have no natural
            order, so they must be one-hot encoded, not scaled.
    """

    outcome_column: str
    nominal_code_columns: tuple[str, ...]

    def numeric_columns(self, all_columns: list[str]) -> list[str]:
        """Return every input column that is a real numeric measurement.

        Args:
            all_columns: All column names of a records table.

        Returns:
            Columns that are neither the outcome nor a nominal code, in their
            original order. These are scaled instead of one-hot encoded.
        """
        excluded_columns = {self.outcome_column, *self.nominal_code_columns}
        return [name for name in all_columns if name not in excluded_columns]

    def input_columns(self, all_columns: list[str]) -> list[str]:
        """Return every column except the outcome, in their original order."""
        return [name for name in all_columns if name != self.outcome_column]


# Column names follow the UCI dataset exactly, including its original
# spelling "Nacionality".
STUDENT_FEATURE_CATALOG = FeatureCatalog(
    outcome_column="Target",
    nominal_code_columns=(
        "Marital Status",
        "Application mode",
        "Course",
        "Previous qualification",
        "Nacionality",
        "Mother's qualification",
        "Father's qualification",
        "Mother's occupation",
        "Father's occupation",
    ),
)
