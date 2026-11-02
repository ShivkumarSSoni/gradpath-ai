from gradpath.backend.ml.metrics.classification_report_calculator import (
    ClassificationReportCalculator,
)

LABELS = ["Dropout", "Enrolled", "Graduate"]
D, E, G = LABELS


def test_each_outcome_gets_its_own_precision_recall_f1_and_support():
    calculator = ClassificationReportCalculator(class_labels=LABELS)
    true_labels = [D, D, E, E, G, G]
    predicted_labels = [D, E, E, E, G, D]

    report = calculator.calculate(true_labels, predicted_labels)

    assert set(report) == set(LABELS)
    assert report[D]["precision"] == 0.5
    assert report[D]["recall"] == 0.5
    assert report[D]["support"] == 2.0
    assert report[E]["recall"] == 1.0


def test_never_predicted_outcome_scores_zero_not_an_error():
    calculator = ClassificationReportCalculator(class_labels=LABELS)
    true_labels = [D, D, E, E, G, G]
    predicted_labels = [D, D, D, G, G, G]  # Enrolled is never predicted

    report = calculator.calculate(true_labels, predicted_labels)

    assert report[E]["precision"] == 0.0
