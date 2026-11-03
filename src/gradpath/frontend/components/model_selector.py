"""Dropdown to choose which saved classifier to score with."""

import streamlit as st

from gradpath.backend.domain.classifier_kind import ClassifierKind
from gradpath.frontend.styling import text_constants


def render_model_selector(classifier_kinds: list[ClassifierKind]) -> ClassifierKind:
    """Render the model dropdown and return the classifier the user picked."""
    display_names = [kind.display_name for kind in classifier_kinds]
    selected_display_name = st.selectbox(text_constants.SELECT_MODEL_LABEL, display_names)
    selected_index = display_names.index(selected_display_name)
    return classifier_kinds[selected_index]
