"""All `st.session_state` access, so no other frontend module touches it.

Run History was deliberately dropped (see the UI design decisions): only
the most recent evaluation run is kept, so the page still shows a result
after an unrelated widget interaction triggers a rerun.
"""

from dataclasses import dataclass

import streamlit as st

from gradpath.backend.domain.classifier_kind import ClassifierKind
from gradpath.backend.schemas.evaluation_response import EvaluationResponse

_LAST_RUN_STATE_KEY = "gradpath_last_evaluation_run"


@dataclass(frozen=True)
class StoredEvaluationRun:
    """One evaluation run, kept only until the next run replaces it.

    Attributes:
        file_name: Name of the uploaded file, for the recap line.
        row_count: Rows in the uploaded file before any cleaning.
        classifier_kind: Which model the user selected for this run.
        class_counts: Outcome label to row count, from the uploaded file
            before cleaning (used for the class distribution chart).
        response: The backend's evaluation response for this run.
    """

    file_name: str
    row_count: int
    classifier_kind: ClassifierKind
    class_counts: dict[str, int]
    response: EvaluationResponse


def store_last_run(run: StoredEvaluationRun) -> None:
    """Replace whatever run was stored before with this one."""
    st.session_state[_LAST_RUN_STATE_KEY] = run


def get_last_run() -> StoredEvaluationRun | None:
    """Return the most recent run, or None if nothing has been run yet."""
    return st.session_state.get(_LAST_RUN_STATE_KEY)
