"""Entry point for the model comparison page."""

from gradpath.backend.domain.comparison_row import ComparisonRow
from gradpath.backend.services.comparison_service import ComparisonService


class ComparisonController:
    """Thin pass-through to the comparison service, for the frontend gateway."""

    def __init__(self, comparison_service: ComparisonService) -> None:
        self._comparison_service = comparison_service

    def compare_all_classifiers(self) -> list[ComparisonRow]:
        """Return one row per classifier, in the standard classifier order."""
        return self._comparison_service.build_comparison_rows()
