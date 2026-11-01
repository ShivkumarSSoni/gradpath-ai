"""The temporary first screen, replaced by the real pages in later steps."""

import streamlit as st

from gradpath.frontend.styling import text_constants
from gradpath.frontend.view_models.environment_summary import (
    collect_environment_summary,
)


def render_home_page() -> None:
    """Draw the title, the student details and the environment summary."""
    # Must be the first Streamlit call of the script run.
    st.set_page_config(page_title=text_constants.APP_PAGE_TITLE, layout="centered")

    st.title(text_constants.APP_HEADING)
    st.caption(
        f"Name: {text_constants.STUDENT_NAME} | "
        f"BITS ID: {text_constants.STUDENT_BITS_ID}"
    )
    st.write(text_constants.DATASET_DESCRIPTION)
    st.info(text_constants.WORK_IN_PROGRESS_MESSAGE)

    with st.expander("Environment check"):
        for library_name, library_version in collect_environment_summary().items():
            st.write(f"{library_name}: {library_version}")
