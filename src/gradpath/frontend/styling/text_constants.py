"""Text shown by the Streamlit pages, kept in one place."""

APP_PAGE_TITLE = "GradPath AI"
APP_HEADING = "GradPath AI"
BRAND_TAGLINE = (
    "Predicting whether a student will drop out, stay enrolled, or "
    "graduate, from academic and demographic records."
)
DATASET_DESCRIPTION = (
    "UCI dataset: Predict Students' Dropout and Academic Success "
    "(4,424 students, 36 features, 3 outcomes)."
)
HOME_PAGE_INFO_MESSAGE = (
    "This screen confirms the dataset summary and the installed library "
    "versions."
)

# Control panel.
SELECT_MODEL_LABEL = "Select model"
UPLOAD_LABEL = "Upload test CSV"
UPLOAD_HELP = (
    "Upload a CSV with the same feature columns as test_data.csv, plus the "
    "Target column."
)
RUN_BUTTON_LABEL = "Run Evaluation"
RUN_BUTTON_CAPTION = (
    "The selected model is loaded as-is and scored on your file. Nothing "
    "is retrained."
)
MISSING_FILE_WARNING_MESSAGE = "Upload a test CSV before selecting Run Evaluation."

# Status and recap.
RUN_RECAP_TEMPLATE = "**{file_name}** ({row_count} row(s)) scored with **{model_name}**."
EVALUATION_SUCCESS_MESSAGE = "Evaluation complete."
UPLOAD_ERROR_PREFIX = "This file could not be scored: "

# Uploaded-file class distribution section.
UPLOADED_FILE_SECTION_TITLE = "Your uploaded file"
UPLOADED_FILE_SECTION_SUBTEXT = (
    "Row count and outcome balance in the file you uploaded, before scoring."
)

# Evaluation metrics section.
METRICS_SECTION_TITLE = "Evaluation metrics"
METRICS_SECTION_SUBTEXT = "Six metrics for the selected model, computed on your uploaded file."

# Confusion matrix and classification report section.
CONFUSION_REPORT_SECTION_TITLE = "Confusion matrix & classification report"
CONFUSION_REPORT_SECTION_SUBTEXT = (
    "Computed fresh from your uploaded rows. Row-normalised confusion "
    "matrix (percent of each actual class)."
)
CONFUSION_MATRIX_SUBHEADING = "Confusion matrix"
CLASSIFICATION_REPORT_SUBHEADING = "Classification report"

# Comparison section.
COMPARISON_SECTION_TITLE = "How does this model compare to the other four?"
COMPARISON_SECTION_BADGE = "fixed reference data"
COMPARISON_SECTION_SUBTEXT = (
    "All five models, scored once on our own held-out split (not your "
    "uploaded file). Does not change between runs."
)
OBSERVATIONS_SUBHEADING = "Observations"
COMPARISON_TABLE_SUBHEADING = "Comparison table"
COMPARISON_CHARTS_SUBHEADING = "Comparison charts"
COMPARISON_CHARTS_CAPTION = (
    "Each panel shows all five models for one metric, sorted best to worst."
)

# Curves section.
CURVES_SECTION_TITLE = "Curves for the selected model"
CURVES_SECTION_SUBTEXT = (
    "ROC (one-vs-rest), Precision-Recall and Calibration curves, computed "
    "from your uploaded file."
)

NOT_AVAILABLE_TEXT = "Not available"

