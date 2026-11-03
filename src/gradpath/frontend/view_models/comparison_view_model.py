"""Turns the backend's comparison rows into display-ready tables and charts.

Nothing here imports Streamlit, so it can be unit tested without a running
app, per the project's testing conventions.
"""

import math
from dataclasses import dataclass

from gradpath.backend.domain.classifier_kind import ClassifierKind
from gradpath.backend.domain.comparison_row import ComparisonRow
from gradpath.frontend import backend_gateway
from gradpath.frontend.view_models.evaluation_view_model import format_metric_value
from gradpath.frontend.view_models.model_observations import build_observation_text

METRIC_NAMES = ["Accuracy", "AUC", "Precision", "Recall", "F1", "MCC"]


@dataclass(frozen=True)
class ComparisonView:
    """Everything the comparison section needs to render.

    Attributes:
        observation_rows: One row per model, its fixed commentary, plus a
            final "Winner" row naming the overall best model.
        table_rows: One row per model with its six scores (or "Not
            available" where a score is undefined).
        chart_data_by_metric: Metric name to a list of (classifier_kind,
            value) pairs, sorted best first, with undefined scores left out.
    """

    observation_rows: list[dict]
    table_rows: list[dict]
    chart_data_by_metric: dict[str, list[tuple[ClassifierKind, float]]]


def load_comparison_view() -> ComparisonView:
    """Fetch the comparison rows and shape them for the comparison section."""
    rows = backend_gateway.load_comparison_rows()
    sorted_rows = _sort_rows_best_first(rows)
    return ComparisonView(
        observation_rows=_build_observation_rows(sorted_rows),
        table_rows=_build_table_rows(sorted_rows),
        chart_data_by_metric=_build_chart_data(rows),
    )


def _sort_rows_best_first(rows: list[ComparisonRow]) -> list[ComparisonRow]:
    """Order models best first: most metric wins, ties broken by Accuracy.

    Matches the ordering already used in the comparison bar charts, so the
    observations and comparison tables read top-to-bottom the same way.
    """
    win_counts = _count_metric_wins(rows)

    def _sort_key(row: ComparisonRow) -> tuple[int, float]:
        accuracy = row.scores.as_display_dict()["Accuracy"] if row.is_available else float("-inf")
        if math.isnan(accuracy):
            accuracy = float("-inf")
        return (-win_counts[row.classifier_kind], -accuracy)

    return sorted(rows, key=_sort_key)


def _build_table_rows(rows: list[ComparisonRow]) -> list[dict]:
    """Return one table row per model, metrics formatted for display."""
    table_rows = []
    for row in rows:
        scores = row.scores.as_display_dict() if row.is_available else {}
        table_rows.append(
            {
                "classifier_kind": row.classifier_kind,
                "Model": row.display_name,
                **{
                    metric_name: format_metric_value(scores.get(metric_name))
                    for metric_name in METRIC_NAMES
                },
            }
        )
    return table_rows


def _build_chart_data(
    rows: list[ComparisonRow],
) -> dict[str, list[tuple[ClassifierKind, float]]]:
    """Return each metric's (classifier, value) pairs, best value first."""
    chart_data = {}
    for metric_name in METRIC_NAMES:
        entries = [
            (row.classifier_kind, row.scores.as_display_dict()[metric_name])
            for row in rows
            if row.is_available and not math.isnan(row.scores.as_display_dict()[metric_name])
        ]
        # MCC can be negative, every other metric is 0 to 1; higher is
        # always better for all six, so a plain descending sort is correct
        # for every panel.
        entries.sort(key=lambda entry: entry[1], reverse=True)
        chart_data[metric_name] = entries
    return chart_data


def _count_metric_wins(rows: list[ComparisonRow]) -> dict[ClassifierKind, int]:
    """Count, for each model, how many of the six metrics it scores best on.

    An exact tie for the best value on a metric counts as a win for no one:
    crediting whichever tied model happens to be listed first would be an
    arbitrary, order-dependent result, not a genuine best score.
    """
    win_counts = {row.classifier_kind: 0 for row in rows}
    for metric_name in METRIC_NAMES:
        values_by_kind = {
            row.classifier_kind: row.scores.as_display_dict()[metric_name]
            for row in rows
            if row.is_available and not math.isnan(row.scores.as_display_dict()[metric_name])
        }
        if not values_by_kind:
            continue
        best_value = max(values_by_kind.values())
        top_scorers = [kind for kind, value in values_by_kind.items() if value == best_value]
        if len(top_scorers) == 1:
            win_counts[top_scorers[0]] += 1
    return win_counts


def _build_observation_rows(rows: list[ComparisonRow]) -> list[dict]:
    """Return the per-model observation rows, plus a final "Winner" row."""
    observation_rows = [
        {
            "classifier_kind": row.classifier_kind,
            "Model": row.display_name,
            "Observation": build_observation_text(
                row.classifier_kind,
                row.scores.as_display_dict() if row.is_available else {},
            ),
        }
        for row in rows
    ]

    win_counts = _count_metric_wins(rows)
    if any(win_counts.values()):
        winner_kind = max(win_counts, key=win_counts.get)
        observation_rows.append(
            {
                "classifier_kind": None,
                "Model": "Winner",
                "Observation": (
                    f"{winner_kind.display_name} - highest score on "
                    f"{win_counts[winner_kind]} of {len(METRIC_NAMES)} metrics."
                ),
            }
        )
    return observation_rows
