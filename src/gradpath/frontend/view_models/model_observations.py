"""Fixed, hand-written observations shown next to the comparison table.

These are our own commentary on known classifier behaviour for this
dataset (numbers come from `results/evaluation_results.csv`, the single
source of truth); the sentences themselves are narrative prose, the same
kind of content planned for the README's observations column.
"""

from gradpath.backend.domain.classifier_kind import ClassifierKind

MODEL_OBSERVATIONS: dict[ClassifierKind, str] = {
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
