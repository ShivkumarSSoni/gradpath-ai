"""Streamlit entry point: the file that Streamlit Cloud and `streamlit run` start."""

import sys
from pathlib import Path

# The gradpath package lives in the src folder. Adding that folder to Python's
# import path lets this file work the same on a laptop, on the BITS virtual
# lab and on Streamlit Cloud, without installing the project as a package.
SOURCE_FOLDER = str(Path(__file__).resolve().parent / "src")
if SOURCE_FOLDER not in sys.path:
    sys.path.insert(0, SOURCE_FOLDER)

from gradpath.frontend.pages.home_page import render_home_page  # noqa: E402

render_home_page()
