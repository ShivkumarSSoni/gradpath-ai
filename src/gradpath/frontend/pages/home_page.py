"""The home screen: student details, dataset summary and environment check."""

import streamlit as st

from gradpath.frontend.styling import text_constants
from gradpath.frontend.view_models.environment_summary import (
    collect_environment_summary,
)


def render_home_page() -> None:
    """Draw the title, the dataset summary and the environment summary."""
    # Must be the first Streamlit call of the script run.
    st.set_page_config(page_title=text_constants.APP_PAGE_TITLE, layout="centered")

    st.title(text_constants.APP_HEADING)
    st.write(text_constants.DATASET_DESCRIPTION)
    st.info(text_constants.HOME_PAGE_INFO_MESSAGE)

    with st.expander("Environment check"):
        for library_name, library_version in collect_environment_summary().items():
            st.write(f"{library_name}: {library_version}")
