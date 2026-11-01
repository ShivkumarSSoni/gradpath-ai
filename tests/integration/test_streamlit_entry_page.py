"""Tests for the Streamlit entry file, run with Streamlit's own test harness."""

import pytest
from streamlit.testing.v1 import AppTest

from gradpath.backend.config import settings
from gradpath.frontend.styling import text_constants

pytestmark = pytest.mark.integration

APP_FILE = settings.PROJECT_ROOT_FOLDER / "app.py"


def run_app() -> AppTest:
    return AppTest.from_file(str(APP_FILE), default_timeout=30).run()


def test_app_starts_without_any_exception():
    assert not run_app().exception


def test_app_shows_the_heading_and_student_details():
    app = run_app()

    assert app.title[0].value == text_constants.APP_HEADING
    assert text_constants.STUDENT_NAME in app.caption[0].value
    assert text_constants.STUDENT_BITS_ID in app.caption[0].value


def test_app_shows_the_work_in_progress_notice():
    app = run_app()

    assert app.info[0].value == text_constants.WORK_IN_PROGRESS_MESSAGE
