"""Generates the learning, validation, ROC, Precision-Recall and calibration
curve images, plus the overfitting-check table.

A learning curve needs many refits per model and a validation curve needs
one refit per candidate setting; with a fixed random seed these are
deterministic, so this script computes them once and writes the resulting
PNG files to docs/images/, which the notebook and README then embed. The
ROC, Precision-Recall and calibration curves need no refitting (they only
score the already-saved pipelines on the held-out students), but they are
generated here too, as static images, so the same curves shown live in the
Streamlit app (for whichever file a user uploads) are also visible to
anyone reading the notebook or the README without running the app.

It never retrains or re-tunes the five saved classifiers and never changes
model/ or results/evaluation_results.csv: it only reads the training and
held-out records and the already-tuned, already-saved pipelines. A hash
check at the end of main() confirms that those files are unchanged.
"""

import hashlib
import sys
from pathlib import Path

PROJECT_ROOT_FOLDER = Path(__file__).resolve().parent.parent
SOURCE_FOLDER = str(PROJECT_ROOT_FOLDER / "src")
if SOURCE_FOLDER not in sys.path:
    sys.path.insert(0, SOURCE_FOLDER)

import matplotlib  # noqa: E402

matplotlib.use("Agg")  # no display is available when this script is run standalone

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.base import clone  # noqa: E402
from sklearn.calibration import CalibratedClassifierCV  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.model_selection import StratifiedKFold, learning_curve, validation_curve  # noqa: E402
from sklearn.naive_bayes import GaussianNB as GaussianNaiveBayesClassifier  # noqa: E402
from sklearn.neighbors import KNeighborsClassifier  # noqa: E402
from sklearn.pipeline import Pipeline  # noqa: E402
from sklearn.svm import SVC  # noqa: E402
from sklearn.tree import DecisionTreeClassifier  # noqa: E402

from gradpath.backend.config.settings import (  # noqa: E402
    EVALUATION_RESULTS_FILE,
    IMAGES_FOLDER,
    RANDOM_SEED_FOR_REPRODUCIBILITY,
    SAVED_MODELS_FOLDER,
)
from gradpath.backend.dependency_wiring import (  # noqa: E402
    build_data_preparation_service,
    build_evaluation_report_service,
    build_evaluation_results_repository,
    build_evaluation_service,
    build_feature_transformer_builder,
    build_model_repository,
)
from gradpath.backend.domain.classifier_kind import ClassifierKind  # noqa: E402
from gradpath.backend.domain.feature_catalog import STUDENT_FEATURE_CATALOG  # noqa: E402
from gradpath.backend.ml.classifier_builders.decision_tree_builder import (  # noqa: E402
    MAXIMUM_TREE_DEPTH_CANDIDATES,
)
from gradpath.backend.ml.classifier_builders.gaussian_naive_bayes_builder import (  # noqa: E402
    VARIANCE_SMOOTHING_CANDIDATES,
)
from gradpath.backend.ml.classifier_builders.k_nearest_neighbors_builder import (  # noqa: E402
    NEIGHBOUR_COUNT_CANDIDATES,
)
from gradpath.backend.ml.classifier_builders.logistic_regression_builder import (  # noqa: E402
    MAXIMUM_SOLVER_ITERATIONS,
    REGULARISATION_C_CANDIDATES,
)
from gradpath.backend.ml.classifier_builders.support_vector_machine_builder import (  # noqa: E402
    MARGIN_PENALTY_C_CANDIDATES,
)
from gradpath.charts.curve_chart import draw_curve_panel  # noqa: E402
from gradpath.charts.table_image import draw_table_as_image  # noqa: E402
from gradpath.frontend.styling.model_colours import MODEL_DISPLAY_COLOURS  # noqa: E402

# The same 5-fold stratified cross-validation used to tune every classifier,
# so these curves are directly comparable with the tuning in section 8.
CROSS_VALIDATION_FOLD_COUNT = 5
TUNING_SCORE_NAME = "f1_macro"

# Eight training-set sizes from 10% to 100% of the training students.
LEARNING_CURVE_TRAINING_SIZE_FRACTIONS = np.linspace(0.1, 1.0, 8)

LEARNING_CURVES_IMAGE_FILE = IMAGES_FOLDER / "learning_curves.png"
VALIDATION_CURVES_IMAGE_FILE = IMAGES_FOLDER / "validation_curves.png"
TRAIN_VS_HELD_OUT_TABLE_IMAGE_FILE = IMAGES_FOLDER / "train_vs_held_out_scores.png"
ROC_CURVES_IMAGE_FILE = IMAGES_FOLDER / "roc_curves.png"
PRECISION_RECALL_CURVES_IMAGE_FILE = IMAGES_FOLDER / "precision_recall_curves.png"
CALIBRATION_CURVES_IMAGE_FILE = IMAGES_FOLDER / "calibration_curves.png"


def cross_validation_folds() -> StratifiedKFold:
    """Return a fresh copy of the 5-fold stratified split used for tuning."""
    return StratifiedKFold(
        n_splits=CROSS_VALIDATION_FOLD_COUNT,
        shuffle=True,
        random_state=RANDOM_SEED_FOR_REPRODUCIBILITY,
    )


def validation_curve_plan_for(classifier_kind: ClassifierKind, best_params: dict) -> dict:
    """Describe how to sweep one classifier's main hyperparameter.

    Only the "main" hyperparameter of each model (the one CLAUDE.md's design
    calls out, for example Decision Tree's max_depth) is swept. Any other
    hyperparameter that was also tuned is fixed at the value already chosen
    by cross-validation, so the chart isolates one setting at a time.

    Args:
        best_params: The `best_params_` of that model's already-fitted
            GridSearchCV, read from the saved model, never refitted here.

    Returns:
        A dict with a fresh, untrained classifier to sweep, the pipeline
        parameter name for `validation_curve` to vary, its candidate values
        (the exact list already used to tune it), and a plot axis label.
    """
    seed = RANDOM_SEED_FOR_REPRODUCIBILITY

    if classifier_kind is ClassifierKind.LOGISTIC_REGRESSION:
        return {
            "classifier": LogisticRegression(max_iter=MAXIMUM_SOLVER_ITERATIONS, random_state=seed),
            "param_name": "classifier__C",
            "param_range": REGULARISATION_C_CANDIDATES,
            "chosen_value": best_params["C"],
            "x_axis_label": "C (regularisation strength, inverse)",
            "log_x_axis": True,
        }

    if classifier_kind is ClassifierKind.DECISION_TREE:
        return {
            "classifier": DecisionTreeClassifier(
                random_state=seed, min_samples_leaf=best_params["min_samples_leaf"]
            ),
            "param_name": "classifier__max_depth",
            "param_range": MAXIMUM_TREE_DEPTH_CANDIDATES,
            "chosen_value": best_params["max_depth"],
            "x_axis_label": "max_depth",
            "log_x_axis": False,
        }

    if classifier_kind is ClassifierKind.K_NEAREST_NEIGHBORS:
        return {
            "classifier": KNeighborsClassifier(weights=best_params["weights"]),
            "param_name": "classifier__n_neighbors",
            "param_range": NEIGHBOUR_COUNT_CANDIDATES,
            "chosen_value": best_params["n_neighbors"],
            "x_axis_label": "n_neighbors",
            "log_x_axis": False,
        }

    if classifier_kind is ClassifierKind.GAUSSIAN_NAIVE_BAYES:
        return {
            "classifier": GaussianNaiveBayesClassifier(),
            "param_name": "classifier__var_smoothing",
            "param_range": VARIANCE_SMOOTHING_CANDIDATES,
            "chosen_value": best_params["var_smoothing"],
            "x_axis_label": "var_smoothing",
            "log_x_axis": True,
        }

    if classifier_kind is ClassifierKind.SUPPORT_VECTOR_MACHINE:
        return {
            "classifier": CalibratedClassifierCV(
                SVC(kernel="rbf", random_state=seed), method="sigmoid", ensemble=False
            ),
            "param_name": "classifier__estimator__C",
            "param_range": MARGIN_PENALTY_C_CANDIDATES,
            "chosen_value": best_params["estimator__C"],
            "x_axis_label": "C (margin penalty)",
            "log_x_axis": True,
        }

    raise KeyError(f"No validation curve plan is defined for {classifier_kind.value}.")


def _new_pipeline(feature_transformer_builder, input_columns: list[str], classifier) -> Pipeline:
    """Build a fresh, unfitted pipeline around a plain (not grid-searched) classifier."""
    return Pipeline(
        steps=[
            ("preprocessing", feature_transformer_builder.build(input_columns)),
            ("classifier", classifier),
        ]
    )


def _file_hashes(*files: Path) -> dict[Path, str]:
    """Return a SHA-256 hash per existing file, used to prove nothing changed."""
    hashes = {}
    for file_path in files:
        if file_path.is_file():
            hashes[file_path] = hashlib.sha256(file_path.read_bytes()).hexdigest()
    return hashes


def _protected_files() -> list[Path]:
    """Every saved model plus the official scores file, which this script must not touch."""
    model_files = sorted(SAVED_MODELS_FOLDER.glob("*.joblib")) if SAVED_MODELS_FOLDER.is_dir() else []
    return [*model_files, EVALUATION_RESULTS_FILE]


def draw_learning_curves(axis, classifier_kind, pipeline, feature_inputs, outcome_labels) -> None:
    """Plot training and cross-validated macro F1 against the training set size."""
    model_colour = MODEL_DISPLAY_COLOURS[classifier_kind]

    train_sizes_abs, train_scores, test_scores = learning_curve(
        pipeline,
        feature_inputs,
        outcome_labels,
        train_sizes=LEARNING_CURVE_TRAINING_SIZE_FRACTIONS,
        cv=cross_validation_folds(),
        scoring=TUNING_SCORE_NAME,
    )

    axis.plot(train_sizes_abs, train_scores.mean(axis=1), "--", color=model_colour, label="Training score")
    axis.plot(train_sizes_abs, test_scores.mean(axis=1), "-", color=model_colour, label="Cross-validated score")
    axis.fill_between(
        train_sizes_abs,
        test_scores.mean(axis=1) - test_scores.std(axis=1),
        test_scores.mean(axis=1) + test_scores.std(axis=1),
        color=model_colour,
        alpha=0.15,
    )
    axis.set_title(classifier_kind.display_name, fontsize=11, fontweight="bold", color=model_colour)
    axis.set_xlabel("Number of training students")
    axis.set_ylabel("Macro F1 score")
    axis.set_ylim(0, 1)
    axis.legend(fontsize=8, loc="lower right")
    axis.grid(alpha=0.3)


def draw_validation_curve(axis, classifier_kind, plan, feature_inputs, outcome_labels, feature_transformer_builder, input_columns) -> None:
    """Plot training and cross-validated macro F1 against one hyperparameter."""
    model_colour = MODEL_DISPLAY_COLOURS[classifier_kind]
    pipeline = _new_pipeline(feature_transformer_builder, input_columns, plan["classifier"])

    train_scores, test_scores = validation_curve(
        pipeline,
        feature_inputs,
        outcome_labels,
        param_name=plan["param_name"],
        param_range=plan["param_range"],
        cv=cross_validation_folds(),
        scoring=TUNING_SCORE_NAME,
    )

    x_values = plan["param_range"]
    axis.plot(x_values, train_scores.mean(axis=1), "--", color=model_colour, label="Training score")
    axis.plot(x_values, test_scores.mean(axis=1), "-", color=model_colour, label="Cross-validated score")
    axis.axvline(plan["chosen_value"], color=model_colour, linestyle=":", alpha=0.7, label="Chosen value")
    if plan["log_x_axis"]:
        axis.set_xscale("log")
    axis.set_title(classifier_kind.display_name, fontsize=11, fontweight="bold", color=model_colour)
    axis.set_xlabel(plan["x_axis_label"])
    axis.set_ylabel("Macro F1 score")
    axis.set_ylim(0, 1)
    axis.legend(fontsize=8, loc="lower right")
    axis.grid(alpha=0.3)


def render_curve_grid(curve_name: str, curves_by_name_per_kind: dict, file_path: Path, suptitle: str) -> None:
    """Save a 2x3 grid with one panel per model, each showing `curve_name`.

    Reuses the exact same `draw_curve_panel` function the Streamlit app
    calls for one model at a time, so a reader of the notebook or README
    sees precisely the same lines the app would draw for the held-out file.
    """
    classifier_kinds = ClassifierKind.all_kinds()
    figure, axes = plt.subplots(3, 2, figsize=(11, 13))
    for axis, classifier_kind in zip(axes.flat, classifier_kinds):
        curves_by_outcome = curves_by_name_per_kind[classifier_kind].get(curve_name, {})
        draw_curve_panel(
            axis,
            curve_name,
            curves_by_outcome,
            MODEL_DISPLAY_COLOURS[classifier_kind],
            panel_title=classifier_kind.display_name,
            title_colour=MODEL_DISPLAY_COLOURS[classifier_kind],
        )
    # Five models in a 2x3 grid leaves one unused panel; hide it.
    axes.flat[-1].axis("off")
    figure.suptitle(suptitle, fontsize=13, fontweight="bold")
    figure.tight_layout(rect=(0, 0, 1, 0.96))
    figure.savefig(file_path, dpi=150, bbox_inches="tight")
    plt.close(figure)


def render_table_image(table_frame: pd.DataFrame, file_path: Path, title: str) -> None:
    """Save the shared table-image renderer's figure to `file_path` as a PNG."""
    figure = draw_table_as_image(table_frame, title=title)
    figure.savefig(file_path, dpi=150, bbox_inches="tight")
    plt.close(figure)


def build_curves_by_name_per_kind() -> dict:
    """Score every saved model once on the held-out students for their curves.

    Calls the same `EvaluationReportService` the Streamlit app uses, so the
    ROC, Precision-Recall and calibration points saved here are identical to
    what the app would show for `test_data.csv`. Read-only: it only predicts
    with the already-fitted pipelines, never refits or retunes anything.
    """
    evaluation_report_service = build_evaluation_report_service()
    held_out_records = build_data_preparation_service().prepare_records_split().held_out_records
    return {
        classifier_kind: evaluation_report_service.build_report(
            classifier_kind, held_out_records
        ).curves_by_name
        for classifier_kind in ClassifierKind.all_kinds()
    }


def build_train_vs_held_out_table() -> pd.DataFrame:
    """Compare each model's macro F1 on training students versus held-out students.

    Uses the already-fitted saved pipelines to predict on the training
    records (no new fitting) and reads the already-recorded held-out scores,
    so this is a read-only check, not a second evaluation workflow.
    """
    evaluation_service = build_evaluation_service()
    held_out_scores_by_kind = build_evaluation_results_repository().load_all_scores()
    records_split = build_data_preparation_service().prepare_records_split()

    rows = []
    for classifier_kind in ClassifierKind.all_kinds():
        training_f1 = evaluation_service.evaluate_classifier(
            classifier_kind, records_split.training_records
        ).f1
        held_out_f1 = held_out_scores_by_kind[classifier_kind].f1
        rows.append(
            {
                "Model": classifier_kind.display_name,
                "Training F1 (macro)": round(training_f1, 3),
                "Held-out F1 (macro)": round(held_out_f1, 3),
                "Gap (training minus held-out)": round(training_f1 - held_out_f1, 3),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    """Generate the learning, validation and ROC/Precision-Recall/calibration curves, and the overfitting table."""
    protected_files = _protected_files()
    hashes_before = _file_hashes(*protected_files)

    records_split = build_data_preparation_service().prepare_records_split()
    training_records = records_split.training_records
    input_columns = STUDENT_FEATURE_CATALOG.input_columns(list(training_records.columns))
    feature_inputs = training_records[input_columns]
    outcome_labels = training_records[STUDENT_FEATURE_CATALOG.outcome_column]

    feature_transformer_builder = build_feature_transformer_builder()
    model_repository = build_model_repository()

    IMAGES_FOLDER.mkdir(parents=True, exist_ok=True)

    learning_curve_figure, learning_curve_axes = plt.subplots(3, 2, figsize=(11, 13))
    validation_curve_figure, validation_curve_axes = plt.subplots(3, 2, figsize=(11, 13))

    classifier_kinds = ClassifierKind.all_kinds()
    for axis_index, classifier_kind in enumerate(classifier_kinds):
        print(f"Computing curves for {classifier_kind.display_name} ...")

        fitted_pipeline = model_repository.load_model(classifier_kind)
        fitted_search = fitted_pipeline.named_steps["classifier"]
        best_params = fitted_search.best_params_

        # A clone keeps the tuned hyperparameters but discards the fitted
        # state, so the learning curve refits cheaply at every size instead
        # of repeating the full hyperparameter search.
        learning_curve_classifier = clone(fitted_search.best_estimator_)
        learning_pipeline = _new_pipeline(feature_transformer_builder, input_columns, learning_curve_classifier)
        draw_learning_curves(
            learning_curve_axes.flat[axis_index],
            classifier_kind,
            learning_pipeline,
            feature_inputs,
            outcome_labels,
        )

        validation_plan = validation_curve_plan_for(classifier_kind, best_params)
        draw_validation_curve(
            validation_curve_axes.flat[axis_index],
            classifier_kind,
            validation_plan,
            feature_inputs,
            outcome_labels,
            feature_transformer_builder,
            input_columns,
        )

    # Five models in a 2x3 grid leaves one unused panel; hide it in both figures.
    learning_curve_axes.flat[-1].axis("off")
    validation_curve_axes.flat[-1].axis("off")

    learning_curve_figure.suptitle(
        "Learning curves: training score versus cross-validated score (training students only)",
        fontsize=13,
        fontweight="bold",
    )
    learning_curve_figure.tight_layout(rect=(0, 0, 1, 0.96))
    learning_curve_figure.savefig(LEARNING_CURVES_IMAGE_FILE, dpi=150, bbox_inches="tight")
    plt.close(learning_curve_figure)

    validation_curve_figure.suptitle(
        "Validation curves: training score versus cross-validated score against the main hyperparameter",
        fontsize=13,
        fontweight="bold",
    )
    validation_curve_figure.tight_layout(rect=(0, 0, 1, 0.96))
    validation_curve_figure.savefig(VALIDATION_CURVES_IMAGE_FILE, dpi=150, bbox_inches="tight")
    plt.close(validation_curve_figure)

    render_table_image(
        build_train_vs_held_out_table(),
        TRAIN_VS_HELD_OUT_TABLE_IMAGE_FILE,
        "Training versus held-out macro F1 (quick overfitting check, no new fitting)",
    )

    print("Computing ROC, Precision-Recall and calibration curves for the held-out students ...")
    curves_by_name_per_kind = build_curves_by_name_per_kind()
    render_curve_grid(
        "ROC",
        curves_by_name_per_kind,
        ROC_CURVES_IMAGE_FILE,
        "ROC curves, one-vs-rest per outcome (held-out students)",
    )
    render_curve_grid(
        "Precision-Recall",
        curves_by_name_per_kind,
        PRECISION_RECALL_CURVES_IMAGE_FILE,
        "Precision-Recall curves, one-vs-rest per outcome (held-out students)",
    )
    render_curve_grid(
        "Calibration",
        curves_by_name_per_kind,
        CALIBRATION_CURVES_IMAGE_FILE,
        "Calibration curves, one-vs-rest per outcome (held-out students)",
    )

    hashes_after = _file_hashes(*protected_files)
    if hashes_before != hashes_after:
        raise RuntimeError(
            "generate_curve_images.py changed a saved model or the official "
            "results file. It must only read them. Investigate before committing."
        )

    print(f"Saved {LEARNING_CURVES_IMAGE_FILE}")
    print(f"Saved {VALIDATION_CURVES_IMAGE_FILE}")
    print(f"Saved {TRAIN_VS_HELD_OUT_TABLE_IMAGE_FILE}")
    print(f"Saved {ROC_CURVES_IMAGE_FILE}")
    print(f"Saved {PRECISION_RECALL_CURVES_IMAGE_FILE}")
    print(f"Saved {CALIBRATION_CURVES_IMAGE_FILE}")
    print("model/ and results/evaluation_results.csv are unchanged (hash check passed).")


if __name__ == "__main__":
    main()
