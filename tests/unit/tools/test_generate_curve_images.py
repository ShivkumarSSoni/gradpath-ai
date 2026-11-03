"""Fast, pure tests for the curve-generation script's hyperparameter plans.

These check the sweep plans only (which parameter path to vary, with which
candidates, holding which other settings fixed). They never fit anything and
never call main(), because generating the real images takes several minutes
(many cross-validated refits per model) and is meant to be run by hand, not
on every pytest run.
"""

import sys
from pathlib import Path

PROJECT_ROOT_FOLDER = Path(__file__).resolve().parents[3]
TOOLS_FOLDER = str(PROJECT_ROOT_FOLDER / "tools")
if TOOLS_FOLDER not in sys.path:
    sys.path.insert(0, TOOLS_FOLDER)

from generate_curve_images import validation_curve_plan_for  # noqa: E402

from gradpath.backend.domain.classifier_kind import ClassifierKind
from gradpath.backend.ml.classifier_builders.decision_tree_builder import (
    MAXIMUM_TREE_DEPTH_CANDIDATES,
)
from gradpath.backend.ml.classifier_builders.k_nearest_neighbors_builder import (
    NEIGHBOUR_COUNT_CANDIDATES,
)
from gradpath.backend.ml.classifier_builders.logistic_regression_builder import (
    REGULARISATION_C_CANDIDATES,
)
from gradpath.backend.ml.classifier_builders.support_vector_machine_builder import (
    MARGIN_PENALTY_C_CANDIDATES,
)


def test_logistic_regression_plan_sweeps_c_only():
    plan = validation_curve_plan_for(ClassifierKind.LOGISTIC_REGRESSION, best_params={"C": 1.0})

    assert plan["param_name"] == "classifier__C"
    assert plan["param_range"] == REGULARISATION_C_CANDIDATES
    assert plan["chosen_value"] == 1.0


def test_decision_tree_plan_fixes_min_samples_leaf_and_sweeps_max_depth():
    plan = validation_curve_plan_for(
        ClassifierKind.DECISION_TREE, best_params={"max_depth": 7, "min_samples_leaf": 10}
    )

    assert plan["param_name"] == "classifier__max_depth"
    assert plan["param_range"] == MAXIMUM_TREE_DEPTH_CANDIDATES
    assert plan["chosen_value"] == 7
    assert plan["classifier"].min_samples_leaf == 10


def test_k_nearest_neighbors_plan_fixes_weights_and_sweeps_neighbour_count():
    plan = validation_curve_plan_for(
        ClassifierKind.K_NEAREST_NEIGHBORS, best_params={"n_neighbors": 11, "weights": "distance"}
    )

    assert plan["param_name"] == "classifier__n_neighbors"
    assert plan["param_range"] == NEIGHBOUR_COUNT_CANDIDATES
    assert plan["chosen_value"] == 11
    assert plan["classifier"].weights == "distance"


def test_support_vector_machine_plan_reaches_c_through_the_calibration_wrapper():
    plan = validation_curve_plan_for(
        ClassifierKind.SUPPORT_VECTOR_MACHINE, best_params={"estimator__C": 10.0}
    )

    assert plan["param_name"] == "classifier__estimator__C"
    assert plan["param_range"] == MARGIN_PENALTY_C_CANDIDATES
    assert plan["chosen_value"] == 10.0


def test_every_classifier_kind_has_a_plan():
    best_params_by_kind = {
        ClassifierKind.LOGISTIC_REGRESSION: {"C": 1.0},
        ClassifierKind.DECISION_TREE: {"max_depth": 5, "min_samples_leaf": 5},
        ClassifierKind.K_NEAREST_NEIGHBORS: {"n_neighbors": 5, "weights": "uniform"},
        ClassifierKind.GAUSSIAN_NAIVE_BAYES: {"var_smoothing": 0.1},
        ClassifierKind.SUPPORT_VECTOR_MACHINE: {"estimator__C": 1.0},
    }

    for classifier_kind in ClassifierKind.all_kinds():
        plan = validation_curve_plan_for(classifier_kind, best_params_by_kind[classifier_kind])
        assert "param_name" in plan
        assert len(plan["param_range"]) > 1
