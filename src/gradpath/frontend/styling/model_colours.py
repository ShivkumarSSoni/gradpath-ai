"""One fixed colour per classifier, reused everywhere a model is drawn.

Keeping a single colour per model here means the notebook, the curve-image
generation script and the Streamlit app always show the same classifier in
the same colour, so a reader never has to re-learn the colour key.
"""

from gradpath.backend.domain.classifier_kind import ClassifierKind

MODEL_DISPLAY_COLOURS: dict[ClassifierKind, str] = {
    ClassifierKind.LOGISTIC_REGRESSION: "#1f4e9c",
    ClassifierKind.DECISION_TREE: "#b35a00",
    ClassifierKind.K_NEAREST_NEIGHBORS: "#1b7a3d",
    ClassifierKind.GAUSSIAN_NAIVE_BAYES: "#8e1b6b",
    ClassifierKind.SUPPORT_VECTOR_MACHINE: "#b01e1e",
}
