"""Error for predictions that cannot be scored."""


class InvalidScoringInputError(ValueError):
    """Raised when true labels and predictions do not fit together."""
