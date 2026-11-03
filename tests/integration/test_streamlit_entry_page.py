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


def test_app_shows_the_brand_heading():
    app = run_app()

    # The heading is rendered as two separately coloured <span> tags (the
    # two-tone "GradPath"/"AI" wordmark from the approved UI mock), so the
    # literal "GradPath AI" substring never appears as one run of text;
    # check that one markdown block carries both halves instead.
    brand_word, ai_word = text_constants.APP_HEADING.rsplit(" ", 1)
    heading_markdown = [block.value for block in app.markdown]
    assert any(brand_word in value and ai_word in value for value in heading_markdown)


def test_app_has_no_sidebar_content():
    app = run_app()

    assert len(app.sidebar.children) == 0


def test_app_shows_no_results_before_a_run():
    app = run_app()

    assert len(app.info) == 0
    assert len(app.subheader) == 0
    assert len(app.metric) == 0


def test_app_offers_all_five_models_in_the_dropdown():
    app = run_app()

    from gradpath.backend.domain.classifier_kind import ClassifierKind

    expected_names = {kind.display_name for kind in ClassifierKind.all_kinds()}
    assert set(app.selectbox[0].options) == expected_names
