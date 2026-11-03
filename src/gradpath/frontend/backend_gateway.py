"""The only file the frontend uses to reach the backend.

Every other frontend module reaches the backend through the functions here,
never through `dependency_wiring` or a controller directly, so there is one
place to change if the backend's entry points ever move.
"""

import csv
import io

import pandas as pd
import streamlit as st

from gradpath.backend.controllers.comparison_controller import ComparisonController
from gradpath.backend.controllers.evaluation_controller import EvaluationController
from gradpath.backend.dependency_wiring import (
    build_comparison_controller,
    build_evaluation_controller,
)
from gradpath.backend.domain.classifier_kind import ClassifierKind
from gradpath.backend.domain.comparison_row import ComparisonRow
from gradpath.backend.schemas.evaluation_request import EvaluationRequest
from gradpath.backend.schemas.evaluation_response import EvaluationResponse

# Protects the shared, free-tier app from a huge upload before it is even
# parsed; the row-count cap in UploadValidationService protects it afterward.
MAX_UPLOAD_FILE_SIZE_BYTES = 5 * 1024 * 1024


@st.cache_resource(show_spinner=False)
def _cached_evaluation_controller() -> EvaluationController:
    """Build the evaluation controller once per server process."""
    return build_evaluation_controller()


@st.cache_resource(show_spinner=False)
def _cached_comparison_controller() -> ComparisonController:
    """Build the comparison controller once per server process."""
    return build_comparison_controller()


def list_classifier_options() -> list[ClassifierKind]:
    """Return the five classifiers, for the model dropdown."""
    return ClassifierKind.all_kinds()


def read_uploaded_file(uploaded_file) -> tuple[pd.DataFrame | None, str | None]:
    """Turn a Streamlit upload into a DataFrame, rejecting an oversized file.

    Args:
        uploaded_file: The object returned by `st.file_uploader`, or None.

    Returns:
        A tuple of the parsed records (or None) and a plain-English error
        message (or None). Exactly one of the two is None.
    """
    if uploaded_file is None:
        return None, None

    if uploaded_file.size > MAX_UPLOAD_FILE_SIZE_BYTES:
        max_size_megabytes = MAX_UPLOAD_FILE_SIZE_BYTES / (1024 * 1024)
        return None, f"The uploaded file is larger than the {max_size_megabytes:.0f} MB limit."

    try:
        csv_text = uploaded_file.getvalue().decode("utf-8")
    except UnicodeDecodeError:
        return None, "This does not look like a valid CSV file."

    # pandas silently reinterprets a row with more fields than the header
    # as an extra index column instead of raising a parse error, so a
    # genuinely malformed upload would otherwise slip through undetected.
    # Checking the raw field counts first catches that case reliably,
    # regardless of pandas version.
    parsed_rows = [row for row in csv.reader(io.StringIO(csv_text)) if row]
    if parsed_rows:
        expected_field_count = len(parsed_rows[0])
        if any(len(row) != expected_field_count for row in parsed_rows[1:]):
            return None, "This does not look like a valid CSV file."

    try:
        records = pd.read_csv(io.StringIO(csv_text))
    except (pd.errors.ParserError, ValueError):
        # A friendly message instead of a raw parser traceback (OWASP:
        # information disclosure).
        return None, "This does not look like a valid CSV file."
    return records, None


def evaluate_uploaded_records(
    classifier_kind: ClassifierKind, records: pd.DataFrame
) -> EvaluationResponse:
    """Validate and score `records` with the saved `classifier_kind` model."""
    request = EvaluationRequest(classifier_kind=classifier_kind, records=records)
    return _cached_evaluation_controller().evaluate_uploaded_records(request)


def load_comparison_rows() -> list[ComparisonRow]:
    """Return the fixed, held-out-split scores of all five classifiers."""
    return _cached_comparison_controller().compare_all_classifiers()
