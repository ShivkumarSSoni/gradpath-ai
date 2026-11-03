"""Streamlit entry point: the file that Streamlit Cloud and `streamlit run` start."""

import sys
from pathlib import Path

# The gradpath package lives in the src folder. Adding that folder to Python's
# import path lets this file work the same on a laptop, on the BITS virtual
# lab and on Streamlit Cloud, without installing the project as a package.
SOURCE_FOLDER = str(Path(__file__).resolve().parent / "src")
if SOURCE_FOLDER not in sys.path:
    sys.path.insert(0, SOURCE_FOLDER)

import streamlit as st  # noqa: E402

from gradpath.frontend.components.branding_header import render_branding_header  # noqa: E402
from gradpath.frontend.pages import model_comparison_page, model_evaluation_page  # noqa: E402
from gradpath.frontend.styling import text_constants  # noqa: E402
from gradpath.frontend.styling.global_css import inject_global_css  # noqa: E402
from gradpath.frontend.view_models.environment_summary import (  # noqa: E402
    collect_environment_summary,
)

# Must be the first Streamlit call of the script run.
st.set_page_config(page_title=text_constants.APP_PAGE_TITLE, layout="centered")

inject_global_css()
render_branding_header()

stored_run = model_evaluation_page.render_input_and_results()
if stored_run is not None and stored_run.response.succeeded:
    model_comparison_page.render_page()
    model_evaluation_page.render_curves_section(stored_run)

with st.expander("Environment"):
    for library_name, library_version in collect_environment_summary().items():
        st.write(f"{library_name}: {library_version}")
