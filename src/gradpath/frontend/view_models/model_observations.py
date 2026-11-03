"""Fixed, hand-written observations shown next to the comparison table.

Only the qualitative reasoning (the "why") is hand-written here; it can
never be derived from the scores themselves (it is about scaling,
independence assumptions, margin width, and so on). The actual numbers
are never hand-typed: `build_observation_text` below appends the live
six scores read from `results/evaluation_results.csv` (the single source
of truth) through `comparison_view_model`, so the app, the notebook and
the README all show the same combined sentence for a given model.
"""

from gradpath.backend.domain.classifier_kind import ClassifierKind
from gradpath.frontend.view_models.evaluation_view_model import format_metric_value

MODEL_OBSERVATION_REASONS: dict[ClassifierKind, str] = {
    ClassifierKind.LOGISTIC_REGRESSION: (
        "A simple linear boundary already separates the three outcomes "
        "well; it is consistently among the strongest models on every "
        "metric below."
    ),
    ClassifierKind.DECISION_TREE: (
        "Splits on a few curricular-unit columns capture most of the "
        "signal, but it overfits more than the other models and trails on "
        "most metrics."
    ),
    ClassifierKind.K_NEAREST_NEIGHBORS: (
        "Performance depends heavily on the scaled distance between "
        "students; competitive overall, weaker on the rare Enrolled "
        "outcome."
    ),
    ClassifierKind.GAUSSIAN_NAIVE_BAYES: (
        "Assumes the features are independent given the outcome, which "
        "does not hold well here, so it scores clearly lower than the "
        "other four."
    ),
    ClassifierKind.SUPPORT_VECTOR_MACHINE: (
        "A wide-margin boundary with calibrated probabilities; close to "
        "the top overall and the strongest model specifically at "
        "recognising the Enrolled outcome."
    ),
}

# Order the six scores appear in within a combined observation sentence;
# matches the comparison table's column order.
_METRIC_DISPLAY_ORDER = ["Accuracy", "AUC", "Precision", "Recall", "F1", "MCC"]


def build_observation_text(classifier_kind: ClassifierKind, scores: dict[str, float]) -> str:
    """Combine the hand-written reason with the model's live six scores.

    Args:
        classifier_kind: Which of the five classifiers this row is for.
        scores: The model's six scores keyed by display name (as returned
            by `EvaluationScores.as_display_dict()`), or an empty dict if
            the model has no saved scores yet.

    Returns:
        One sentence of qualitative reasoning followed by the model's own
        scores, e.g. "... Accuracy 0.767, AUC 0.881, Precision 0.712,
        Recall 0.685, F1 0.693, MCC 0.614."
    """
    reason = MODEL_OBSERVATION_REASONS[classifier_kind]
    if not scores:
        return reason
    score_text = ", ".join(
        f"{metric_name} {format_metric_value(scores.get(metric_name))}"
        for metric_name in _METRIC_DISPLAY_ORDER
    )
    return f"{reason} {score_text}."
