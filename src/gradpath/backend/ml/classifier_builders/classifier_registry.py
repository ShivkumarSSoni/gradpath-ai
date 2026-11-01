"""Looks up the right builder for each classifier kind."""

from sklearn.base import BaseEstimator

from gradpath.backend.domain.classifier_kind import ClassifierKind
from gradpath.backend.ml.classifier_builders.classifier_builder import (
    ClassifierBuilder,
)


class ClassifierRegistry:
    """Holds one builder per classifier kind."""

    def __init__(self, builders: list[ClassifierBuilder]) -> None:
        self._builders_by_kind: dict[ClassifierKind, ClassifierBuilder] = {}
        for builder in builders:
            if builder.kind in self._builders_by_kind:
                raise ValueError(f"Two builders were given for {builder.kind.value}.")
            self._builders_by_kind[builder.kind] = builder

    def registered_kinds(self) -> list[ClassifierKind]:
        """Return the kinds that have a builder, in the standard order."""
        return [
            kind
            for kind in ClassifierKind.all_kinds()
            if kind in self._builders_by_kind
        ]

    def build_untrained(self, kind: ClassifierKind) -> BaseEstimator:
        """Return a fresh untrained classifier of the requested kind."""
        if kind not in self._builders_by_kind:
            raise KeyError(f"No builder is registered for {kind.value}.")
        return self._builders_by_kind[kind].build_untrained()
