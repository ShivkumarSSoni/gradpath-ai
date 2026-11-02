"""Error for a classifier that cannot be trained or evaluated."""


class ClassifierTrainingError(RuntimeError):
    """Raised when training a classifier gives an unusable result."""
