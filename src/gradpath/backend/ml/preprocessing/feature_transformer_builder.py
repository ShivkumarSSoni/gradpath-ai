"""Builds the unfitted preprocessing step shared by all five classifiers."""

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from gradpath.backend.domain.feature_catalog import FeatureCatalog


class FeatureTransformerBuilder:
    """Creates the one-hot encoding and scaling step for the input columns."""

    def __init__(self, feature_catalog: FeatureCatalog) -> None:
        """Remember which columns are category codes and which are numeric."""
        self._feature_catalog = feature_catalog

    def build(self, all_columns: list[str]) -> ColumnTransformer:
        """Return a new, unfitted transformer for the given record columns.

        The transformer is only fitted on training records, then reused on
        test records and on uploaded files. Fitting it on all data would leak
        information from the test records into training.

        Args:
            all_columns: Column names of the records, with or without the
                outcome column.

        Returns:
            A transformer that one-hot encodes the category-code columns and
            standardises the numeric columns.
        """
        numeric_columns = self._feature_catalog.numeric_columns(all_columns)
        nominal_columns = list(self._feature_catalog.nominal_code_columns)

        return ColumnTransformer(
            transformers=[
                (
                    "category_codes",
                    # A code never seen in training (for example in an uploaded
                    # file) becomes all zeros instead of crashing the app.
                    # Dense output is required because Gaussian Naive Bayes
                    # cannot read sparse matrices.
                    OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                    nominal_columns,
                ),
                ("numeric_measurements", StandardScaler(), numeric_columns),
            ],
            remainder="drop",
            verbose_feature_names_out=False,
        )
