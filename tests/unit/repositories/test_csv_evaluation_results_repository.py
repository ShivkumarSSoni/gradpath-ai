import math

import pandas as pd

from gradpath.backend.domain.classifier_kind import ClassifierKind
from gradpath.backend.domain.evaluation_scores import EvaluationScores
from gradpath.backend.repositories.csv_evaluation_results_repository import (
    CsvEvaluationResultsRepository,
)


def make_scores(accuracy: float, auc: float = 0.9) -> EvaluationScores:
    return EvaluationScores(
        accuracy=accuracy, auc=auc, precision=0.5, recall=0.6, f1=0.55, mcc=0.4
    )


def test_nothing_is_returned_before_any_scores_are_saved(tmp_path):
    repository = CsvEvaluationResultsRepository(tmp_path / "results.csv")

    assert repository.load_all_scores() == {}


def test_saved_scores_load_back_unchanged(tmp_path):
    repository = CsvEvaluationResultsRepository(tmp_path / "results.csv")
    scores = make_scores(0.75)

    repository.save_scores(ClassifierKind.DECISION_TREE, scores)

    assert repository.load_all_scores() == {ClassifierKind.DECISION_TREE: scores}


def test_saving_one_classifier_keeps_the_others(tmp_path):
    repository = CsvEvaluationResultsRepository(tmp_path / "results.csv")
    repository.save_scores(ClassifierKind.DECISION_TREE, make_scores(0.7))
    repository.save_scores(ClassifierKind.LOGISTIC_REGRESSION, make_scores(0.8))

    repository.save_scores(ClassifierKind.DECISION_TREE, make_scores(0.71))
    loaded = repository.load_all_scores()

    assert loaded[ClassifierKind.DECISION_TREE].accuracy == 0.71
    assert loaded[ClassifierKind.LOGISTIC_REGRESSION].accuracy == 0.8


def test_rows_follow_the_standard_classifier_order(tmp_path):
    results_file = tmp_path / "results.csv"
    repository = CsvEvaluationResultsRepository(results_file)

    repository.save_scores(ClassifierKind.SUPPORT_VECTOR_MACHINE, make_scores(0.7))
    repository.save_scores(ClassifierKind.LOGISTIC_REGRESSION, make_scores(0.8))

    saved_table = pd.read_csv(results_file)
    assert list(saved_table["Model"]) == ["Logistic Regression", "Support Vector Machine (SVM)"]
    assert list(saved_table.columns) == [
        "Model", "Accuracy", "AUC", "Precision", "Recall", "F1", "MCC"
    ]


def test_a_missing_auc_survives_the_round_trip(tmp_path):
    repository = CsvEvaluationResultsRepository(tmp_path / "results.csv")

    repository.save_scores(ClassifierKind.DECISION_TREE, make_scores(0.7, auc=float("nan")))

    assert math.isnan(repository.load_all_scores()[ClassifierKind.DECISION_TREE].auc)
