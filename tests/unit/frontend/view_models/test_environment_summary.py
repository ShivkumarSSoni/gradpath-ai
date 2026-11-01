"""Unit tests for the environment summary shown on the first page."""

import platform

import sklearn

from gradpath.frontend.view_models.environment_summary import (
    collect_environment_summary,
)


def test_summary_lists_python_and_the_main_libraries():
    summary = collect_environment_summary()

    assert list(summary) == ["Python", "Streamlit", "scikit-learn", "pandas"]


def test_summary_reports_the_versions_actually_in_use():
    summary = collect_environment_summary()

    assert summary["Python"] == platform.python_version()
    assert summary["scikit-learn"] == sklearn.__version__
