"""Unit tests for the shared bordered HTML table builder."""

from gradpath.backend.domain.classifier_kind import ClassifierKind
from gradpath.frontend.components.styled_table import _build_table_html
from gradpath.frontend.styling.model_colours import MODEL_DISPLAY_COLOURS


def test_table_html_includes_every_header_and_cell_value():
    rows = [{"classifier_kind": ClassifierKind.DECISION_TREE, "Model": "Decision Tree", "Accuracy": "0.80"}]

    table_html = _build_table_html(["Model", "Accuracy"], rows)

    assert "<th" in table_html and "Model" in table_html and "Accuracy" in table_html
    assert "Decision Tree" in table_html
    assert "0.80" in table_html


def test_table_html_draws_a_colour_dot_matching_the_model():
    rows = [{"classifier_kind": ClassifierKind.SUPPORT_VECTOR_MACHINE, "Model": "SVM", "Accuracy": "0.90"}]

    table_html = _build_table_html(["Model", "Accuracy"], rows)

    assert MODEL_DISPLAY_COLOURS[ClassifierKind.SUPPORT_VECTOR_MACHINE] in table_html


def test_table_html_escapes_untrusted_text_and_skips_the_dot_for_the_winner_row():
    rows = [{"classifier_kind": None, "Model": "Winner", "Observation": "<script>alert(1)</script>"}]

    table_html = _build_table_html(["Model", "Observation"], rows)

    assert "<script>" not in table_html
    assert "&lt;script&gt;" in table_html
