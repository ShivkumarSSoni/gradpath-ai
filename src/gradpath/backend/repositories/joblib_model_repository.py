"""Trained models stored as joblib files, one file per classifier."""

from pathlib import Path

import joblib
from sklearn.base import BaseEstimator

from gradpath.backend.domain.classifier_kind import ClassifierKind
from gradpath.backend.exceptions.model_not_found_error import ModelNotFoundError
from gradpath.backend.repositories.interfaces.model_repository import ModelRepository


class JoblibModelRepository(ModelRepository):
    """Keeps each model in <folder>/<classifier name>.joblib."""

    def __init__(self, models_folder: Path) -> None:
        self._models_folder = Path(models_folder)

    def save_model(self, classifier_kind: ClassifierKind, trained_model: BaseEstimator) -> None:
        self._models_folder.mkdir(parents=True, exist_ok=True)
        joblib.dump(trained_model, self._file_for(classifier_kind))

    def load_model(self, classifier_kind: ClassifierKind) -> BaseEstimator:
        if not self.model_exists(classifier_kind):
            raise ModelNotFoundError(
                f"No saved model for {classifier_kind.display_name}. Train it first."
            )
        return joblib.load(self._file_for(classifier_kind))

    def model_exists(self, classifier_kind: ClassifierKind) -> bool:
        return self._file_for(classifier_kind).is_file()

    def _file_for(self, classifier_kind: ClassifierKind) -> Path:
        return self._models_folder / f"{classifier_kind.value}.joblib"
