"""Generates three images for the README and the notebook, from the same
shared grid-building functions the notebook itself calls, so there is only
one piece of code behind each chart:

- the outcome (class-balance) distribution of the full dataset,
- the five classifiers' confusion matrices on the held-out students,
- the six evaluation metrics, one bar-chart panel per metric.

It never retrains or re-tunes the five saved classifiers and never changes
model/ or results/evaluation_results.csv: it only reads the full dataset,
the held-out records and the already-tuned, already-saved pipelines. A hash
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
from sklearn.metrics import confusion_matrix  # noqa: E402

from gradpath.backend.config.settings import (  # noqa: E402
    EVALUATION_RESULTS_FILE,
    IMAGES_FOLDER,
    SAVED_MODELS_FOLDER,
)
from gradpath.backend.dependency_wiring import (  # noqa: E402
    build_data_preparation_service,
    build_evaluation_results_repository,
    build_model_repository,
    build_records_repository,
)
from gradpath.backend.domain.classifier_kind import ClassifierKind  # noqa: E402
from gradpath.backend.domain.feature_catalog import STUDENT_FEATURE_CATALOG  # noqa: E402
from gradpath.backend.domain.student_outcome import StudentOutcome  # noqa: E402
from gradpath.charts.confusion_matrix_chart import render_confusion_matrix_grid  # noqa: E402
from gradpath.charts.metric_bar_chart import render_metric_bar_grid  # noqa: E402
from gradpath.charts.outcome_distribution_chart import render_outcome_distribution_figure  # noqa: E402
from gradpath.frontend.styling.model_colours import MODEL_DISPLAY_COLOURS  # noqa: E402

OUTCOME_DISTRIBUTION_IMAGE_FILE = IMAGES_FOLDER / "outcome_distribution.png"
CONFUSION_MATRICES_IMAGE_FILE = IMAGES_FOLDER / "confusion_matrices.png"
METRIC_COMPARISON_IMAGE_FILE = IMAGES_FOLDER / "metric_comparison.png"


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


def save_outcome_distribution_image() -> None:
    """Save the full dataset's outcome (class-balance) distribution."""
    outcome_class_names = StudentOutcome.all_labels()
    student_records_table = build_records_repository().load_all_records()
    outcome_counts = (
        student_records_table[STUDENT_FEATURE_CATALOG.outcome_column]
        .value_counts()
        .reindex(outcome_class_names)
    )

    figure = render_outcome_distribution_figure(outcome_class_names, outcome_counts.values)
    figure.savefig(OUTCOME_DISTRIBUTION_IMAGE_FILE, dpi=150, bbox_inches="tight")
    plt.close(figure)


def save_confusion_matrices_image() -> None:
    """Save the five classifiers' confusion matrices on the held-out students."""
    outcome_class_names = StudentOutcome.all_labels()
    held_out_records = build_data_preparation_service().prepare_records_split().held_out_records
    input_columns = STUDENT_FEATURE_CATALOG.input_columns(list(held_out_records.columns))
    held_out_inputs = held_out_records[input_columns]
    held_out_true_outcomes = held_out_records[STUDENT_FEATURE_CATALOG.outcome_column]

    model_repository = build_model_repository()
    panels = []
    for classifier_kind in ClassifierKind.all_kinds():
        model_name = classifier_kind.display_name
        predicted_outcomes = model_repository.load_model(classifier_kind).predict(held_out_inputs)
        student_counts = confusion_matrix(held_out_true_outcomes, predicted_outcomes, labels=outcome_class_names)
        panels.append((model_name, MODEL_DISPLAY_COLOURS[classifier_kind], student_counts))

    figure = render_confusion_matrix_grid(panels, outcome_class_names)
    figure.savefig(CONFUSION_MATRICES_IMAGE_FILE, dpi=150, bbox_inches="tight")
    plt.close(figure)


def save_metric_comparison_image() -> None:
    """Save the six evaluation metrics, one ranked bar-chart panel per metric."""
    saved_scores_by_classifier = build_evaluation_results_repository().load_all_scores()
    colour_of_classifier = {kind.display_name: colour for kind, colour in MODEL_DISPLAY_COLOURS.items()}

    first_scores = next(iter(saved_scores_by_classifier.values()))
    metric_names = list(first_scores.as_display_dict().keys())

    ranked_entries_by_metric = {}
    for metric_name in metric_names:
        scores_by_name = {
            kind.display_name: scores.as_display_dict()[metric_name]
            for kind, scores in saved_scores_by_classifier.items()
        }
        ranked_names = sorted(scores_by_name, key=scores_by_name.get, reverse=True)
        ranked_entries_by_metric[metric_name] = [
            (name, scores_by_name[name], colour_of_classifier[name]) for name in ranked_names
        ]

    figure = render_metric_bar_grid(metric_names, ranked_entries_by_metric)
    figure.savefig(METRIC_COMPARISON_IMAGE_FILE, dpi=150, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    """Generate the outcome distribution, confusion matrix and metric comparison images."""
    protected_files = _protected_files()
    hashes_before = _file_hashes(*protected_files)

    IMAGES_FOLDER.mkdir(parents=True, exist_ok=True)

    print("Saving outcome distribution image ...")
    save_outcome_distribution_image()

    print("Saving confusion matrices image ...")
    save_confusion_matrices_image()

    print("Saving metric comparison image ...")
    save_metric_comparison_image()

    hashes_after = _file_hashes(*protected_files)
    if hashes_before != hashes_after:
        raise RuntimeError(
            "generate_dataset_and_results_images.py changed a saved model or the official "
            "results file. It must only read them. Investigate before committing."
        )

    print(f"Saved {OUTCOME_DISTRIBUTION_IMAGE_FILE}")
    print(f"Saved {CONFUSION_MATRICES_IMAGE_FILE}")
    print(f"Saved {METRIC_COMPARISON_IMAGE_FILE}")
    print("model/ and results/evaluation_results.csv are unchanged (hash check passed).")


if __name__ == "__main__":
    main()
