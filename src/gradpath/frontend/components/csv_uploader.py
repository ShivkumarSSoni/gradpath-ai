"""File uploader widget for the test CSV.

No validation happens here; every check runs server-side in the backend
(see UploadValidationService), never in this widget.
"""

import streamlit as st

from gradpath.frontend.styling import text_constants


def render_csv_uploader():
    """Render the upload widget and return the raw Streamlit upload, or None."""
    return st.file_uploader(
        text_constants.UPLOAD_LABEL,
        type=["csv"],
        help=text_constants.UPLOAD_HELP,
    )
