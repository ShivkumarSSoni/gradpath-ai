"""Evaluation scores stored in one small CSV file."""

from pathlib import Path

import pandas as pd

from gradpath.backend.domain.classifier_kind import ClassifierKind
from gradpath.backend.domain.evaluation_scores import EvaluationScores
from gradpath.backend.repositories.interfaces.evaluation_results_repository import (
    EvaluationResultsRepository,
)

MODEL_NAME_COLUMN = "Model"


class CsvEvaluationResultsRepository(EvaluationResultsRepository):
    """Keeps one row per classifier: its readable name and the six scores."""

    def __init__(self, results_file: Path) -> None:
        self._results_file = Path(results_file)

    def save_scores(self, classifier_kind: ClassifierKind, scores: EvaluationScores) -> None:
        scores_by_kind = self.load_all_scores()
        scores_by_kind[classifier_kind] = scores

        rows = []
        for kind in ClassifierKind.all_kinds():
            if kind in scores_by_kind:
                rows.append(
                    {MODEL_NAME_COLUMN: kind.display_name, **scores_by_kind[kind].as_display_dict()}
                )

        self._results_file.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows).to_csv(self._results_file, index=False)

    def load_all_scores(self) -> dict[ClassifierKind, EvaluationScores]:
        if not self._results_file.is_file():
            return {}

        kinds_by_display_name = {kind.display_name: kind for kind in ClassifierKind.all_kinds()}
        results_table = pd.read_csv(self._results_file)

        scores_by_kind = {}
        for _, row in results_table.iterrows():
            kind = kinds_by_display_name[row[MODEL_NAME_COLUMN]]
            scores_by_kind[kind] = EvaluationScores(
                accuracy=float(row["Accuracy"]),
                auc=float(row["AUC"]),
                precision=float(row["Precision"]),
                recall=float(row["Recall"]),
                f1=float(row["F1"]),
                mcc=float(row["MCC"]),
            )
        return scores_by_kind
