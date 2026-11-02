"""A request to score one classifier on one table of uploaded records."""

from dataclasses import dataclass

import pandas as pd

from gradpath.backend.domain.classifier_kind import ClassifierKind


@dataclass(frozen=True)
class EvaluationRequest:
    """What the frontend gateway sends in to ask for an evaluation.

    Attributes:
        classifier_kind: Which saved model to score with.
        records: The uploaded table, including the outcome column, exactly
            as read from the file (not yet validated).
    """

    classifier_kind: ClassifierKind
    records: pd.DataFrame
