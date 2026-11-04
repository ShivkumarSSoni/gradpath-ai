"""The control panel, class distribution, metrics and confusion/report sections.

This module owns everything derived from one stored evaluation run. The
comparison section (fixed reference data, unrelated to the uploaded file)
lives in `model_comparison_page.py` and is rendered between
`render_input_and_results` and `render_curves_section` by `app.py`, so the
curves (which belong to the selected model, not the comparison) still
appear last on the page, matching the confirmed section order.
"""

import html

import streamlit as st

from gradpath.frontend.components import (
    class_distribution_chart,
    classification_report_table,
    confusion_matrix_chart,
    csv_uploader,
    curve_charts,
    feedback_banner,
    metric_cards,
    model_selector,
)
from gradpath.frontend.state.session_state_manager import (
    StoredEvaluationRun,
    get_last_run,
    store_last_run,
)
from gradpath.frontend.styling import text_constants
from gradpath.frontend.styling.brand_colours import BRAND_TEXT_COLOUR
from gradpath.frontend.styling.model_colours import MODEL_DISPLAY_COLOURS
from gradpath.frontend.view_models import evaluation_view_model
from gradpath.frontend.view_models.evaluation_view_model import run_evaluation


def render_input_and_results() -> StoredEvaluationRun | None:
    """Render the control panel, then the per-run results if one exists.

    Returns:
        The stored run (successful or not), or None if nothing has been run
        yet. The caller uses this to decide whether to also render the
        comparison section and the curves section.
    """
    classifier_kind, uploaded_file, run_clicked = _render_control_panel()

    if run_clicked:
        with st.spinner(text_constants.EVALUATION_SPINNER_MESSAGE):
            new_run = run_evaluation(classifier_kind, uploaded_file)
        if new_run is None:
            feedback_banner.render_warning(text_constants.MISSING_FILE_WARNING_MESSAGE)
        else:
            store_last_run(new_run)

    stored_run = get_last_run()
    if stored_run is None:
        return None

    _render_recap_and_status(stored_run)
    if not stored_run.response.succeeded:
        return stored_run

    _render_uploaded_file_section(stored_run)
    _render_metrics_section(stored_run)
    _render_confusion_and_report_section(stored_run)
    return stored_run


def render_curves_section(stored_run: StoredEvaluationRun) -> None:
    """Render the curves section for the stored run's selected model."""
    st.divider()
    with st.container(border=True):
        st.subheader(text_constants.CURVES_SECTION_TITLE)
        st.caption(text_constants.CURVES_SECTION_SUBTEXT)
        curve_charts.render_curves(
            stored_run.response.report.curves_by_name, stored_run.classifier_kind
        )


def _render_control_panel():
    """Render the model dropdown, uploader and Run button; return the choices."""
    with st.container(border=True):
        classifier_kind = model_selector.render_model_selector(
            evaluation_view_model.list_classifier_options()
        )
        uploaded_file = csv_uploader.render_csv_uploader()
        # A narrow, centred button column keeps "Run Evaluation" at a
        # natural width instead of stretching across the whole control panel.
        _left_spacer, button_column, _right_spacer = st.columns([1, 1, 1])
        with button_column:
            run_clicked = st.button(
                text_constants.RUN_BUTTON_LABEL, use_container_width=True, type="primary"
            )
        st.caption(text_constants.RUN_BUTTON_CAPTION)
    return classifier_kind, uploaded_file, run_clicked


def _render_recap_and_status(stored_run: StoredEvaluationRun) -> None:
    """Render the recap line, then a success, error or warning banner."""
    st.markdown(
        f"<p style='color:{BRAND_TEXT_COLOUR}; font-size:0.95rem;'>"
        f"<strong>{html.escape(stored_run.file_name)}</strong> "
        f"({stored_run.row_count} row(s)) scored with "
        f"<strong>{html.escape(stored_run.classifier_kind.display_name)}</strong>.</p>",
        unsafe_allow_html=True,
    )
    response = stored_run.response
    if not response.succeeded:
        feedback_banner.render_error(text_constants.UPLOAD_ERROR_PREFIX + response.error_message)
        return

    feedback_banner.render_success(text_constants.EVALUATION_SUCCESS_MESSAGE)
    _render_model_name_banner(stored_run.classifier_kind)
    for warning in response.warnings:
        feedback_banner.render_warning(warning)


def _render_model_name_banner(classifier_kind) -> None:
    """Render a left-indented banner naming the model, in its unique colour."""
    model_colour = MODEL_DISPLAY_COLOURS[classifier_kind]
    st.markdown(
        f"<div style='border-left: 4px solid {model_colour}; "
        f"padding: 0.4rem 0.9rem; margin: 0.5rem 0; "
        f"background-color: {model_colour}1a;'>"
        f"<span style='color:{model_colour}; font-weight:700; font-size:1.05rem;'>"
        f"{html.escape(classifier_kind.display_name)}</span></div>",
        unsafe_allow_html=True,
    )


def _render_uploaded_file_section(stored_run: StoredEvaluationRun) -> None:
    st.divider()
    with st.container(border=True):
        st.subheader(text_constants.UPLOADED_FILE_SECTION_TITLE)
        st.caption(text_constants.UPLOADED_FILE_SECTION_SUBTEXT)
        class_distribution_chart.render_class_distribution(stored_run.class_counts)


def _render_metrics_section(stored_run: StoredEvaluationRun) -> None:
    st.divider()
    with st.container(border=True):
        st.subheader(text_constants.METRICS_SECTION_TITLE)
        st.caption(text_constants.METRICS_SECTION_SUBTEXT)
        metric_cards.render_metric_cards(
            evaluation_view_model.build_metric_cards(stored_run.response),
            MODEL_DISPLAY_COLOURS[stored_run.classifier_kind],
        )


def _render_confusion_and_report_section(stored_run: StoredEvaluationRun) -> None:
    st.divider()
    with st.container(border=True):
        st.subheader(text_constants.CONFUSION_REPORT_SECTION_TITLE)
        st.caption(text_constants.CONFUSION_REPORT_SECTION_SUBTEXT)

        st.markdown(f"**{text_constants.CONFUSION_MATRIX_SUBHEADING}**")
        confusion_matrix_chart.render_confusion_matrix(
            evaluation_view_model.build_confusion_matrix_view(stored_run.response),
            stored_run.classifier_kind,
        )

        st.markdown(f"**{text_constants.CLASSIFICATION_REPORT_SUBHEADING}**")
        classification_report_table.render_classification_report(
            evaluation_view_model.build_classification_report_rows(stored_run.response)
        )
