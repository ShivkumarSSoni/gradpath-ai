from gradpath.backend.ml.metrics.confusion_matrix_calculator import (
    ConfusionMatrixCalculator,
)

LABELS = ["Dropout", "Enrolled", "Graduate"]
D, E, G = LABELS


def test_confusion_matrix_rows_are_true_outcomes_and_columns_are_predictions():
    calculator = ConfusionMatrixCalculator(class_labels=LABELS)
    true_labels = [D, D, E, G, G]
    predicted_labels = [D, E, E, G, D]

    matrix = calculator.calculate(true_labels, predicted_labels)

    # Dropout: 1 correct, 1 predicted as Enrolled, 0 as Graduate.
    assert matrix[0] == [1, 1, 0]
    # Enrolled: 0 as Dropout, 1 correct, 0 as Graduate.
    assert matrix[1] == [0, 1, 0]
    # Graduate: 1 as Dropout, 0 as Enrolled, 1 correct.
    assert matrix[2] == [1, 0, 1]


def test_an_outcome_absent_from_the_records_still_gets_a_row_and_column():
    calculator = ConfusionMatrixCalculator(class_labels=LABELS)

    matrix = calculator.calculate([D, D], [D, D])

    assert len(matrix) == 3
    assert all(len(row) == 3 for row in matrix)
    assert matrix[0][0] == 2
