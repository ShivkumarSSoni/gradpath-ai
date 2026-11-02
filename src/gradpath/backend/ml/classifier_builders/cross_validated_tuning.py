"""Shared way of choosing a classifier's settings by cross-validation."""

from sklearn.base import BaseEstimator
from sklearn.model_selection import GridSearchCV, StratifiedKFold

CROSS_VALIDATION_FOLD_COUNT = 5

# Macro F1 gives the rare Enrolled outcome the same weight as the large ones.
TUNING_SCORE_NAME = "f1_macro"


def tune_by_cross_validation(
    classifier: BaseEstimator,
    candidate_settings: dict[str, list],
    random_seed: int,
) -> GridSearchCV:
    """Wrap a classifier so that fitting it first picks its best settings.

    Fitting the returned object tries every combination of the candidate
    settings with stratified 5-fold cross-validation on the records it is
    given, then refits the best combination on all of those records. Inside
    the training service that means the choice uses training records only,
    so the held-out test records never influence it.
    """
    return GridSearchCV(
        estimator=classifier,
        param_grid=candidate_settings,
        scoring=TUNING_SCORE_NAME,
        cv=StratifiedKFold(
            n_splits=CROSS_VALIDATION_FOLD_COUNT,
            shuffle=True,
            random_state=random_seed,
        ),
    )
